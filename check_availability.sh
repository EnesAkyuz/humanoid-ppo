#!/bin/bash
# Check Lambda GPU availability
# Usage: bash check_availability.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$PROJECT_DIR/.env"

curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-types | python3 -c "
import sys, json
data = json.load(sys.stdin)['data']
found = False
for name, info in sorted(data.items()):
    regions = info.get('regions_with_capacity_available', [])
    price = info['instance_type'].get('price_cents_per_hour', 0) / 100
    if regions:
        found = True
        for r in regions:
            print(f'  {name:<35} \${price:.2f}/hr   {r[\"name\"]}')
if not found:
    print('  Nothing available. Try again later.')
"
