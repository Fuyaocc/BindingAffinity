#!/bin/bash
shopt -s nullglob
for file in /mnt/data/xukeyu/data/clean_pdb/*; do
    filename=$(basename "$file")
    echo "$filename"
    python utils/renumber_pdb.py -i "$file"  -r  -c > "/mnt/data/xukeyu/data/pdbs/$filename"
done