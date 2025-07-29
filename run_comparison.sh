#!/bin/bash

# Script to run comparison between VINN and BehaviorRetrieval methods

echo "======================================================================"
echo "🔍 COMPARING VINN AND BEHAVIOR RETRIEVAL ON ALL TARGET SAMPLES"
echo "======================================================================"

# Default values
TARGET_DIR="/mnt/storage/owen/robot-dataset/rt-cache/raw/"
VINN_MODEL_DIR="./vinn_target_models"
BR_MODEL_DIR="../BehaviorRetrieval/br_target_models"
K=16
DEVICE="cuda"
VISUALIZE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target_dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        --vinn_model_dir)
            VINN_MODEL_DIR="$2"
            shift 2
            ;;
        --br_model_dir)
            BR_MODEL_DIR="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-visualize)
            VISUALIZE=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
OUTPUT_DIR="comparison_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Run the comparison
if [ "$VISUALIZE" = true ]; then
    python compare_retrieval_methods.py \
        --target_dir "$TARGET_DIR" \
        --vinn_model_dir "$VINN_MODEL_DIR" \
        --br_model_dir "$BR_MODEL_DIR" \
        --k $K \
        --device "$DEVICE" \
        --visualize \
        --plot_dir "$OUTPUT_DIR/plots" \
        --output_file "$OUTPUT_DIR/comparison_results.json"
else
    python compare_retrieval_methods.py \
        --target_dir "$TARGET_DIR" \
        --vinn_model_dir "$VINN_MODEL_DIR" \
        --br_model_dir "$BR_MODEL_DIR" \
        --k $K \
        --device "$DEVICE" \
        --output_file "$OUTPUT_DIR/comparison_results.json"
fi

echo ""
echo "✅ Comparison complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "📊 Output files:"
echo "  - Detailed results: $OUTPUT_DIR/comparison_results.json"
echo "  - Summary: $OUTPUT_DIR/comparison_results_summary.json"
echo "  - Retrieval analysis: $OUTPUT_DIR/comparison_results_retrieval_analysis.json"
if [ "$VISUALIZE" = true ]; then
    echo "  - Visualization plots: $OUTPUT_DIR/plots/"
fi