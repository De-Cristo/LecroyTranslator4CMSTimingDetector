#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <input_dir> <output_dir> [max_files]"
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_FILES=${3:-0}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# File to store the summary of processed files
SUMMARY_FILE="$OUTPUT_DIR/file_summary.txt"
echo "Processing Summary - $(date)" > "$SUMMARY_FILE"
echo "----------------------------------------" >> "$SUMMARY_FILE"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist."
    exit 1
fi

count=0

# Iterate over all C1 files in the input directory
while read -r c1_file; do
    # Optional early stop for quick debug
    if [ "$MAX_FILES" -gt 0 ] && [ "$count" -ge "$MAX_FILES" ]; then
        echo "Reached max_files=$MAX_FILES, stopping."
        break
    fi
    # Extract the filename
    c1_basename=$(basename "$c1_file")
    
    # Construct the corresponding C2 file path by replacing _C1_ with _C2_
    c2_basename="${c1_basename/_C1_/_C2_}"
    c2_file="$INPUT_DIR/$c2_basename"
    
    # Check if the paired C2 file exists
    if [ -f "$c2_file" ]; then
        echo "Pair found: $c1_basename and $c2_basename"
        
        # Run the python processing script
        python3 combine_mcp_clock.py \
            --mcp "$c1_file" \
            --clock "$c2_file" \
            --out-dir "$OUTPUT_DIR" \
            --clock-polarity rising \
            --clock-min-edge-spacing-ns 3 \
            --clock-drop-last-edge 2 \
            --clock-plot-first 0
            
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "Successfully processed: $c1_basename + $c2_basename" >> "$SUMMARY_FILE"
            ((count++))
        else
            echo "Failed to process: $c1_basename + $c2_basename" >> "$SUMMARY_FILE"
        fi
    else
        echo "Warning: No matching C2 file found for $c1_basename"
        echo "Missing pair for: $c1_basename" >> "$SUMMARY_FILE"
    fi
done < <(find "$INPUT_DIR" -maxdepth 1 -name "*raw_C1_*_data.csv")

echo "----------------------------------------" >> "$SUMMARY_FILE"
echo "Finished processing. A total of $count pairs were successfully processed." >> "$SUMMARY_FILE"
echo "Summary saved to $SUMMARY_FILE"
