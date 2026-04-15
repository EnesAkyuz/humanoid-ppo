#!/bin/bash
# Launch a Lambda GPU instance, setup environment, and run Humanoid training.
# Usage: bash lambda_run.sh [--resume]
#
# Required .env variables:
#   LAMBDA_API_KEY, SSH_KEY_NAME, SSH_KEY_PATH
# Optional .env variables (for a persistent filesystem — recommended so that
# the venv + checkpoints survive between spot instances):
#   LAMBDA_FILESYSTEM_NAME, LAMBDA_FILESYSTEM_MOUNT
# See .env.example.
#
# Syncs checkpoints back to local machine every 10 minutes.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_DIR/.env"

FILESYSTEM_NAME="${LAMBDA_FILESYSTEM_NAME:-}"
FS_MOUNT="${LAMBDA_FILESYSTEM_MOUNT:-/lambda/nfs/$FILESYSTEM_NAME}"
REMOTE_DIR="$FS_MOUNT/humanoid"
RESUME_FLAG="${1:-}"

# --- Preferred instance types (cheapest single-GPU first) ---
# Only us-east-1 instances can attach the filesystem
PREFERRED_TYPES=(
    "gpu_1x_a10"
    "gpu_1x_a100_sxm4"
    "gpu_1x_h100_pcie"
    "gpu_1x_h100_sxm5"
    "gpu_1x_gh200"
    "gpu_1x_b200_sxm6"
    "gpu_8x_v100_n"
)

# --- Find an available instance in us-east-1 (filesystem region) ---
find_available_instance() {
    local avail
    avail=$(curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types)

    for itype in "${PREFERRED_TYPES[@]}"; do
        local region
        region=$(echo "$avail" | python3 -c "
import sys, json
data = json.load(sys.stdin)['data']
entry = data.get('$itype', {})
regions = entry.get('regions_with_capacity_available', [])
for r in regions:
    if r['name'] == 'us-east-1':
        print(r['name'])
        break
" 2>/dev/null)

        if [ -n "$region" ]; then
            echo "$itype $region"
            return 0
        fi
    done

    # Fallback: any region (but warn about no filesystem)
    for itype in "${PREFERRED_TYPES[@]}"; do
        local region
        region=$(echo "$avail" | python3 -c "
import sys, json
data = json.load(sys.stdin)['data']
entry = data.get('$itype', {})
regions = entry.get('regions_with_capacity_available', [])
for r in regions:
    print(r['name'])
    break
" 2>/dev/null)

        if [ -n "$region" ]; then
            echo "$itype $region NOFS"
            return 0
        fi
    done
    return 1
}

# --- Launch instance ---
echo "=== Looking for available GPU instance (prefer us-east-1 for filesystem) ==="
MATCH=$(find_available_instance) || {
    echo "No GPU instances available. Run 'bash check_availability.sh' to monitor."
    exit 1
}

INSTANCE_TYPE=$(echo "$MATCH" | awk '{print $1}')
REGION=$(echo "$MATCH" | awk '{print $2}')
NO_FS=$(echo "$MATCH" | awk '{print $3}')

if [ -z "$FILESYSTEM_NAME" ] || [ "$NO_FS" = "NOFS" ]; then
    if [ -z "$FILESYSTEM_NAME" ]; then
        echo "No LAMBDA_FILESYSTEM_NAME set — launching without a persistent filesystem."
    else
        echo "WARNING: No instance available in the filesystem's region. Using $REGION without persistent filesystem."
    fi
    echo "         Checkpoints will only be saved via rsync."
    REMOTE_DIR="/home/ubuntu/humanoid"
    FS_ARGS=""
else
    FS_ARGS="\"file_system_names\": [\"$FILESYSTEM_NAME\"],"
fi

echo "Found: $INSTANCE_TYPE in $REGION"

echo "=== Launching $INSTANCE_TYPE in $REGION ==="
LAUNCH_RESP=$(curl -s -u "$LAMBDA_API_KEY:" \
    -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/launch \
    -H "Content-Type: application/json" \
    -d "{
        \"region_name\": \"$REGION\",
        \"instance_type_name\": \"$INSTANCE_TYPE\",
        \"ssh_key_names\": [\"$SSH_KEY_NAME\"],
        $FS_ARGS
        \"name\": \"humanoid-train\"
    }")

INSTANCE_ID=$(echo "$LAUNCH_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['instance_ids'][0])" 2>/dev/null) || {
    echo "Launch failed: $LAUNCH_RESP"
    exit 1
}
echo "Instance ID: $INSTANCE_ID"

# --- Wait for instance to be active ---
echo "=== Waiting for instance to boot ==="
IP=""
for i in $(seq 1 60); do
    DETAILS=$(curl -s -u "$LAMBDA_API_KEY:" \
        "https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID")

    STATUS=$(echo "$DETAILS" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['status'])" 2>/dev/null)
    IP=$(echo "$DETAILS" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'].get('ip','') or '')" 2>/dev/null)

    echo "  [$i/60] status=$STATUS ip=$IP"

    if [ "$STATUS" = "active" ] && [ -n "$IP" ]; then
        break
    fi

    if [ "$STATUS" = "terminated" ] || [ "$STATUS" = "error" ]; then
        echo "Instance failed with status: $STATUS"
        exit 1
    fi

    sleep 10
done

if [ -z "$IP" ]; then
    echo "Timed out waiting for instance."
    exit 1
fi

echo "Instance ready at $IP"

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $SSH_KEY_PATH ubuntu@$IP"
SCP="scp -o StrictHostKeyChecking=no -i $SSH_KEY_PATH"

# --- Wait for SSH ---
echo "=== Waiting for SSH ==="
for i in $(seq 1 30); do
    if $SSH "echo ok" 2>/dev/null; then
        break
    fi
    echo "  [$i/30] SSH not ready..."
    sleep 5
done

# --- Setup remote environment ---
echo "=== Setting up remote environment ==="
$SSH "mkdir -p $REMOTE_DIR/checkpoints $REMOTE_DIR/logs $REMOTE_DIR/videos"

# Upload project files (always upload latest code)
$SCP "$PROJECT_DIR/config.yaml" \
     "$PROJECT_DIR/train.py" \
     "$PROJECT_DIR/evaluate.py" \
     "$PROJECT_DIR/plot.py" \
     "$PROJECT_DIR/requirements.txt" \
     "ubuntu@$IP:$REMOTE_DIR/"

# Upload local checkpoints if resuming and no checkpoint exists on filesystem yet
if [ "$RESUME_FLAG" = "--resume" ] && [ -d "$PROJECT_DIR/checkpoints" ]; then
    HAS_REMOTE_CKPT=$($SSH "ls $REMOTE_DIR/checkpoints/humanoid_ppo_*_steps.zip 2>/dev/null | head -1" || true)
    if [ -z "$HAS_REMOTE_CKPT" ]; then
        echo "=== Uploading local checkpoints (none found on filesystem) ==="
        $SCP -r "$PROJECT_DIR/checkpoints/" "ubuntu@$IP:$REMOTE_DIR/checkpoints/"
    else
        echo "=== Checkpoints already on filesystem, skipping upload ==="
    fi
fi

# Install deps — reuse venv on filesystem if it exists
$SSH << SETUP
set -e
cd $REMOTE_DIR

if [ -d .venv ] && [ -f .venv/bin/activate ]; then
    echo "Reusing existing venv from filesystem"
    source .venv/bin/activate
    pip install -q -r requirements.txt
else
    echo "Creating new venv"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
SETUP

# --- Start training ---
echo "=== Starting training ==="
RESUME_ARG=""
if [ "$RESUME_FLAG" = "--resume" ]; then
    # Find latest checkpoint on the remote (filesystem persists them)
    LATEST=$($SSH "ls -t $REMOTE_DIR/checkpoints/humanoid_ppo_*_steps.zip 2>/dev/null | head -1" || true)
    if [ -n "$LATEST" ]; then
        RESUME_ARG="--resume $LATEST"
        echo "Resuming from: $LATEST"
    fi
fi

# Run training in background with nohup so it survives SSH disconnects
$SSH "cd $REMOTE_DIR && source .venv/bin/activate && nohup python3 train.py $RESUME_ARG > train.log 2>&1 &"
echo "Training started in background."

# --- Checkpoint sync loop ---
echo "=== Syncing checkpoints every 10 minutes ==="
echo "Instance: $INSTANCE_TYPE @ $IP (ID: $INSTANCE_ID)"
echo "Storage:  $REMOTE_DIR (persistent filesystem)"
echo "To monitor:  ssh -i $SSH_KEY_PATH ubuntu@$IP 'tail -f $REMOTE_DIR/train.log'"
echo "To terminate: curl -s -u \"\$LAMBDA_API_KEY:\" -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate -H 'Content-Type: application/json' -d '{\"instance_ids\":[\"$INSTANCE_ID\"]}'"
echo ""

mkdir -p "$PROJECT_DIR/checkpoints" "$PROJECT_DIR/logs"

while true; do
    sleep 600

    # Check if instance is still alive
    STATUS=$(curl -s -u "$LAMBDA_API_KEY:" \
        "https://cloud.lambdalabs.com/api/v1/instances/$INSTANCE_ID" | \
        python3 -c "import sys,json; print(json.load(sys.stdin)['data']['status'])" 2>/dev/null || echo "unknown")

    if [ "$STATUS" != "active" ]; then
        echo "Instance no longer active (status=$STATUS). Final sync..."
        break
    fi

    # Check if training is still running
    RUNNING=$($SSH "pgrep -f 'python3 train.py'" 2>/dev/null || true)

    echo "[$(date '+%H:%M:%S')] Syncing checkpoints..."
    rsync -az -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY_PATH" \
        "ubuntu@$IP:$REMOTE_DIR/checkpoints/" "$PROJECT_DIR/checkpoints/" 2>/dev/null || true
    rsync -az -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY_PATH" \
        "ubuntu@$IP:$REMOTE_DIR/logs/" "$PROJECT_DIR/logs/" 2>/dev/null || true

    if [ -z "$RUNNING" ]; then
        echo "Training finished! Final sync done."
        echo ""
        echo "Terminating instance to save costs..."
        curl -s -u "$LAMBDA_API_KEY:" \
            -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
            -H "Content-Type: application/json" \
            -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}"
        echo "Instance terminated."
        break
    fi
done

echo ""
echo "=== Done ==="
echo "Checkpoints: $PROJECT_DIR/checkpoints/"
echo "Logs:        $PROJECT_DIR/logs/"
echo "Evaluate:    cd local && python evaluate.py ../checkpoints/best_model.zip"
