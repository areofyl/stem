#!/bin/sh
# train all 4 micro separators
# usage: ./train_all.sh /path/to/data

set -e

DATA="${1:?usage: ./train_all.sh /path/to/data}"

for target in vocals drums bass other; do
    echo ""
    echo "=== training: $target ==="
    python3 train.py --data "$DATA" --target "$target" --epochs 100 --batch_size 32
done

echo ""
echo "done. models saved to ./checkpoints/"
ls -lh checkpoints/*.pt
