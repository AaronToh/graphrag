#!/bin/bash
# Simple SuperHFBP Graph Pruning Script
#
# This script prunes your GraphRAG graph using SuperHFBP algorithm.
# Output: workspace/pruned_superhfbp/ containing pruned entities.parquet and relationships.parquet
#
# Usage:
#   ./prune_graph.sh                    # Use defaults (10 queries)
#   ./prune_graph.sh 20                 # Use 20 queries
#   ./prune_graph.sh --help             # Show help

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default configuration
NUM_QUERIES=10
WORKSPACE="workspace"
OUTPUT_DIR="workspace/pruned_superhfbp"
STRATEGY="superhfbp"

# Help message
show_help() {
    cat << EOF
${BLUE}SuperHFBP Graph Pruning Script${NC}

Prunes your GraphRAG graph to reduce size while maintaining quality.

${GREEN}Usage:${NC}
    $0 [NUM_QUERIES] [OPTIONS]

${GREEN}Arguments:${NC}
    NUM_QUERIES     Number of queries to use for pruning (default: 10)
                    More queries = better coverage but larger output

${GREEN}Options:${NC}
    --workspace DIR     GraphRAG workspace directory (default: workspace)
    --output DIR        Output directory for pruned graph (default: workspace/pruned_superhfbp)
    --strategy NAME     Pruning strategy: superhfbp or hfbp (default: superhfbp)
    --verbose          Enable verbose logging
    --help             Show this help message

${GREEN}Examples:${NC}
    $0                              # Prune using 10 queries
    $0 20                           # Prune using 20 queries
    $0 15 --verbose                 # Prune using 15 queries with verbose output
    $0 --workspace my_workspace     # Use custom workspace

${GREEN}Output:${NC}
    Pruned graph artifacts will be saved to: ${OUTPUT_DIR}/
    - entities.parquet          (pruned entities)
    - relationships.parquet     (pruned relationships)
    - text_units.parquet        (optional)
    - pruning_metadata.json     (pruning statistics)

${GREEN}Expected Results:${NC}
    - 60-80% reduction in entities
    - 70-85% reduction in relationships
    - Maintained retrieval quality
    - Faster query performance

EOF
    exit 0
}

# Parse arguments
VERBOSE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        [0-9]*)
            NUM_QUERIES="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Banner
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          SuperHFBP Graph Pruning Script                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Validate workspace
if [ ! -d "$WORKSPACE/output" ]; then
    echo -e "${RED}❌ Error: GraphRAG workspace not found at $WORKSPACE/output${NC}"
    echo ""
    echo "Please ensure you have:"
    echo "  1. Run data ingestion: python ingest/build_index.py"
    echo "  2. Or specify correct workspace: $0 --workspace /path/to/workspace"
    echo ""
    exit 1
fi

# Check if output already exists
if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/entities.parquet" ]; then
    echo -e "${YELLOW}⚠️  Warning: Pruned graph already exists at $OUTPUT_DIR${NC}"
    echo ""
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

# Display configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Workspace:      $WORKSPACE"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Num queries:    $NUM_QUERIES"
echo "  Strategy:       $STRATEGY"
echo ""

# Check original graph size
echo -e "${BLUE}📊 Original Graph Statistics:${NC}"
pixi run python -c "
import pandas as pd
from pathlib import Path

output_dir = Path('$WORKSPACE/output')
entities = pd.read_parquet(output_dir / 'entities.parquet')
relationships = pd.read_parquet(output_dir / 'relationships.parquet')

print(f'  Entities:       {len(entities):,}')
print(f'  Relationships:  {len(relationships):,}')
" 2>/dev/null || {
    echo -e "${RED}  Error: Could not read graph statistics${NC}"
}
echo ""

# Run pruning
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Running SuperHFBP graph pruning...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

pixi run python pruning/run_superhfbp_pruning.py \
    --workspace "$WORKSPACE" \
    --output "$OUTPUT_DIR" \
    --mode auto \
    --num-queries "$NUM_QUERIES" \
    --strategy "$STRATEGY" \
    $VERBOSE

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Pruning failed!${NC}"
    echo ""
    echo "Common issues:"
    echo "  - Missing dependencies: Run 'pixi install' in the project root"
    echo "  - Invalid workspace: Check that $WORKSPACE/output exists"
    echo "  - Memory issues: Try reducing --num-queries"
    echo ""
    exit 1
fi

# Success message
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Pruning completed successfully!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Display results
if [ -f "$OUTPUT_DIR/pruning_metadata.json" ]; then
    echo -e "${BLUE}📊 Pruning Results:${NC}"
    pixi run python -c "
import json
with open('$OUTPUT_DIR/pruning_metadata.json') as f:
    meta = json.load(f)
    orig = meta['original_stats']
    pruned = meta['pruned_stats']
    reduction = meta['reduction_rate']

    print(f'  Original:')
    print(f'    - Entities:       {orig[\"num_entities\"]:,}')
    print(f'    - Relationships:  {orig[\"num_relationships\"]:,}')
    print(f'')
    print(f'  Pruned:')
    print(f'    - Entities:       {pruned[\"num_entities\"]:,}')
    print(f'    - Relationships:  {pruned[\"num_relationships\"]:,}')
    print(f'')
    print(f'  Reduction:')
    print(f'    - Entities:       {reduction[\"entities\"]*100:.1f}%')
    print(f'    - Relationships:  {reduction[\"relationships\"]*100:.1f}%')
    print(f'')
    print(f'  Duration:          {meta[\"duration_seconds\"]:.1f}s')
    print(f'  Queries used:      {meta[\"num_queries\"]}')
" 2>/dev/null || {
    echo -e "${YELLOW}  (Could not read metadata)${NC}"
}
else
    echo -e "${YELLOW}⚠️  Warning: Pruning metadata not found${NC}"
fi

echo ""
echo -e "${GREEN}📁 Output Files:${NC}"
echo "  Pruned artifacts saved to: ${OUTPUT_DIR}/"
ls -lh "$OUTPUT_DIR"/*.parquet "$OUTPUT_DIR"/*.json 2>/dev/null | awk '{print "    - " $9 " (" $5 ")"}'

echo ""
echo -e "${GREEN}🎯 Next Steps:${NC}"
echo ""
echo "  1. Review pruning metadata:"
echo "     ${BLUE}cat $OUTPUT_DIR/pruning_metadata.json${NC}"
echo ""
echo "  2. Run evaluation on pruned graph:"
echo "     ${BLUE}python eval/generate_answers.py --config superhfbp_pruned_graph --questions 50${NC}"
echo ""
echo "  3. Compare with baseline:"
echo "     ${BLUE}python eval/run_eval.py --ablation${NC}"
echo ""
echo -e "${GREEN}✨ Pruning complete! Your graph is now ready for evaluation.${NC}"
