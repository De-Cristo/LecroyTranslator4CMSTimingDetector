#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>"
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# File to store the list of processed files
FILE_LIST="$OUTPUT_DIR/file_list.txt"
> "$FILE_LIST"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist."
    exit 1
fi

# Iterate over all .trc files in the input directory
# Using find to handle potential issues with too many files or special characters
find "$INPUT_DIR" -maxdepth 1 -name "*.trc" | while read -r file; do
    echo "Processing $file..."
    
    # Run the read_lecroy command
    # Assumes process_trc.sh is in the same directory as read_lecroy or it's in the PATH
    # Based on user request, it's ./read_lecroy
    ./read_lecroy "$file" "$OUTPUT_DIR"
    
    # Save the filename to the list
    echo "$(basename "$file")" >> "$FILE_LIST"
done

echo "Finished processing. File list saved to $FILE_LIST"
