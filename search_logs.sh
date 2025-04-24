#!/bin/bash

# Folder containing the .out files
FOLDER="report/10659672_LSSboost_M5"
# FOLDER="report/10659758_XGBLSSboost_M5"
OUTPUT_FILE="oom_killed_ids.txt"

# Clear output file
> "$OUTPUT_FILE"

# Loop through each .out file in the folder
for file in "$FOLDER"/*.out; do
    if grep -q "oom_kill" "$file"; then
        # Extract number between _ and .out in filename
        filename=$(basename "$file")
        id=$(echo "$filename" | sed -n 's/.*_\([0-9]\+\)\.out/\1/p')
        echo "$id" >> "$OUTPUT_FILE"
    fi
done

# Sort numerically in-place
sort -n "$OUTPUT_FILE" -o "$OUTPUT_FILE"

echo -n "OOM-killed job IDs: "
paste -sd, "$OUTPUT_FILE"

echo "Done. Saved OOM-killed job IDs to $OUTPUT_FILE (sorted)"