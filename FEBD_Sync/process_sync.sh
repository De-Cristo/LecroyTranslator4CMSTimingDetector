#!/bin/bash

# ============================================================================
# process_sync.sh — Batch FEBD sync + MCP attach
# Wraps fast_sync_add_mcp.py over all ROOT files in a run directory.
# ============================================================================

usage() {
    echo "Usage: $0 --root-dir <dir> --peaks-dir <dir> --out-dir <dir> [options]"
    echo ""
    echo "Required:"
    echo "  --root-dir DIR       Input directory containing ROOT files (e.g. .../reco/4441)"
    echo "  --peaks-dir DIR      Directory containing peaks CSVs"
    echo "  --out-dir DIR        Output directory for ROOT, CSV, and plots"
    echo ""
    echo "Optional:"
    echo "  --meta-dir DIR       Directory containing meta CSVs (default: ../trc_out)"
    echo "  --channel NUM        Channel to synchronize (default: 192)"
    echo "  --step-size NUM      Chunk size for fast scan (default: 300000)"
    echo "  --max-files NUM      Maximum number of ROOT files to process"
    echo "  --dedup              Enable double-trigger dedup using t1coarse"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --root-dir /eos/.../reco/4441 --peaks-dir ../trc_out_MCP_clock_reco --out-dir ./output --max-files 10"
}

# Defaults
META_DIR="../trc_out"
CHANNEL=192
STEP_SIZE=300000
MAX_FILES=""
DEDUP=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --root-dir)   INPUT_ROOT_DIR="$2"; shift 2 ;;
        --peaks-dir)  PEAKS_DIR="$2"; shift 2 ;;
        --out-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --meta-dir)   META_DIR="$2"; shift 2 ;;
        --channel)    CHANNEL="$2"; shift 2 ;;
        --step-size)  STEP_SIZE="$2"; shift 2 ;;
        --max-files)  MAX_FILES="$2"; shift 2 ;;
        --dedup)      DEDUP="--dedup"; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_ROOT_DIR" ] || [ -z "$PEAKS_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --root-dir, --peaks-dir, and --out-dir are all required."
    usage
    exit 1
fi

INPUT_ROOT_DIR=$(realpath -s "$INPUT_ROOT_DIR")

# Check if input directory exists
if [ ! -d "$INPUT_ROOT_DIR" ]; then
    echo "Error: Input root directory $INPUT_ROOT_DIR does not exist."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# File to store the summary of processed files
SUMMARY_FILE="$OUTPUT_DIR/file_summary.txt"
echo "FEBD Sync Processing Summary - $(date)" > "$SUMMARY_FILE"
echo "  root-dir:  $INPUT_ROOT_DIR" >> "$SUMMARY_FILE"
echo "  peaks-dir: $PEAKS_DIR" >> "$SUMMARY_FILE"
echo "  meta-dir:  $META_DIR" >> "$SUMMARY_FILE"
echo "  out-dir:   $OUTPUT_DIR" >> "$SUMMARY_FILE"
echo "  channel:   $CHANNEL" >> "$SUMMARY_FILE"
if [ -n "$MAX_FILES" ]; then
    echo "  max-files: $MAX_FILES" >> "$SUMMARY_FILE"
fi
echo "  dedup:     ${DEDUP:-off}" >> "$SUMMARY_FILE"
echo "----------------------------------------" >> "$SUMMARY_FILE"

# Extract RUN number from the input directory name
RUN_NUM=$(basename "$INPUT_ROOT_DIR")

# Verify Run number is strictly numeric
if ! [[ "$RUN_NUM" =~ ^[0-9]+$ ]]; then
    echo "Warning: Could not reliably extract numeric run from directory name ($RUN_NUM). Using as prefix anyway."
    echo "Warning: Directory was not purely numeric. Base pattern derived: $RUN_NUM" >> "$SUMMARY_FILE"
fi

count=0
fail_count=0
skip_count=0

# Iterate over all .root files in the input directory
while read -r root_file; do
    root_basename=$(basename "$root_file")
    
    echo "Processing $root_basename from run $RUN_NUM..."
    
    # Extract spill number assuming format <spill>_<something>.root
    spill=$(echo "$root_basename" | grep -o '^[0-9]\+')
    
    if [ -n "$spill" ] && [[ "$RUN_NUM" =~ ^[0-9]+$ ]]; then
        padded_run=$(printf "%07d" "$RUN_NUM")
        padded_spill=$(printf "%07d" "$spill")
        
        # Pre-check: verify meta and peak files exist
        shopt -s nullglob
        meta_files=("${META_DIR}"/raw_C2_${padded_run}_${padded_spill}_*_meta.csv)
        peak_files=("${PEAKS_DIR}"/peaks_raw_C1_${padded_run}_${padded_spill}_*_data_with_tave.csv)
        shopt -u nullglob
        
        if [ ${#meta_files[@]} -eq 0 ] || [ ${#peak_files[@]} -eq 0 ]; then
            echo "Skipping $root_basename: Missing meta or peak dependencies."
            echo "Skipped (missing dependencies): $root_basename" >> "$SUMMARY_FILE"
            ((skip_count++))
            continue
        fi
    fi
    
    # Define output files
    out_root="${OUTPUT_DIR}/${RUN_NUM}_${root_basename}"
    out_csv="${OUTPUT_DIR}/${RUN_NUM}_${root_basename%.root}_matched.csv"
    
    # Run the fast sync python script
    # python3 fast_sync_add_mcp_newv_debug.py \
    python3 simple_sync.py \
        --root "$root_file" \
        --peaks-dir "$PEAKS_DIR" \
        --meta-dir "$META_DIR" \
        --out-root "$out_root" \
        --out-matched-csv "$out_csv" \
        --channel "$CHANNEL" \
        --fast \
        --require-trigger \
        --step-size "$STEP_SIZE" \
        $DEDUP

    if [ $? -eq 0 ]; then
        echo "Successfully processed: $root_basename -> ${RUN_NUM}_${root_basename}" >> "$SUMMARY_FILE"
        ((count++))
    else
        echo "Failed to process: $root_basename" >> "$SUMMARY_FILE"
        ((fail_count++))
    fi
    
    # Check if we've reached the max files limit
    total_processed=$((count + fail_count + skip_count))
    if [ -n "$MAX_FILES" ] && [ "$total_processed" -ge "$MAX_FILES" ]; then
        echo "Reached --max-files limit ($MAX_FILES). Stopping."
        echo "STOPPED EARLY: Reached --max-files limit of $MAX_FILES" >> "$SUMMARY_FILE"
        break
    fi

done < <(find "$INPUT_ROOT_DIR" -maxdepth 1 -name "*.root" | sort)

echo "----------------------------------------" >> "$SUMMARY_FILE"
echo "Finished processing." >> "$SUMMARY_FILE"
echo "Successfully processed: $count files." >> "$SUMMARY_FILE"
echo "Failed processing: $fail_count files." >> "$SUMMARY_FILE"
echo "Skipped (missing dependencies): $skip_count files." >> "$SUMMARY_FILE"
echo "Summary saved to $SUMMARY_FILE"
