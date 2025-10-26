#!/bin/bash
# Complete SuperHFBP Pipeline: Prune → Eval
#
# This script runs the complete pipeline:
# 1. Generate pruned graph using SuperHFBP
# 2. Run evaluation on pruned graph
#
# Usage: ./run_superhfbp_pipeline.sh [--num-queries N] [--questions M]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
NUM_QUERIES=10
NUM_QUESTIONS=50
WORKSPACE="workspace"
PRUNED_OUTPUT="workspace/pruned_superhfbp"
EVAL_OUTPUT="superhfbp_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-queries)
      NUM_QUERIES="$2"
      shift 2
      ;;
    --questions)
      NUM_QUESTIONS="$2"
      shift 2
      ;;
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --num-queries N    Number of queries for pruning (default: 10)"
      echo "  --questions M      Number of questions for eval (default: 50)"
      echo "  --workspace PATH   Workspace directory (default: workspace)"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       SuperHFBP Complete Pipeline: Prune → Eval             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if workspace exists
if [ ! -d "$WORKSPACE/output" ]; then
  echo -e "${RED}❌ Error: GraphRAG workspace not found at $WORKSPACE/output${NC}"
  echo "   Please run data ingestion first or specify correct workspace with --workspace"
  exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Workspace:      $WORKSPACE"
echo "  Pruned output:  $PRUNED_OUTPUT"
echo "  Prune queries:  $NUM_QUERIES"
echo "  Eval questions: $NUM_QUESTIONS"
echo ""

# Step 1: Run SuperHFBP pruning
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 1/2: Running SuperHFBP graph pruning...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

python pruning/run_superhfbp_pruning.py \
  --workspace "$WORKSPACE" \
  --output "$PRUNED_OUTPUT" \
  --mode auto \
  --num-queries "$NUM_QUERIES" \
  --strategy superhfbp \
  --verbose

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Pruning failed!${NC}"
  exit 1
fi

echo ""
echo -e "${GREEN}✅ Pruning completed successfully!${NC}"
echo ""

# Check if pruned artifacts were created
if [ ! -f "$PRUNED_OUTPUT/entities.parquet" ]; then
  echo -e "${RED}❌ Error: Pruned entities.parquet not found${NC}"
  exit 1
fi

echo -e "${BLUE}📊 Pruning Statistics:${NC}"
if [ -f "$PRUNED_OUTPUT/pruning_metadata.json" ]; then
  # Display key statistics using Python
  python3 -c "
import json
with open('$PRUNED_OUTPUT/pruning_metadata.json') as f:
    meta = json.load(f)
    orig = meta['original_stats']
    pruned = meta['pruned_stats']
    reduction = meta['reduction_rate']
    print(f\"  Original:  {orig['num_entities']:,} entities, {orig['num_relationships']:,} relationships\")
    print(f\"  Pruned:    {pruned['num_entities']:,} entities, {pruned['num_relationships']:,} relationships\")
    print(f\"  Reduction: {reduction['entities']*100:.1f}% entities, {reduction['relationships']*100:.1f}% relationships\")
    print(f\"  Duration:  {meta['duration_seconds']:.2f}s\")
"
fi

echo ""

# Step 2: Run evaluation on pruned graph
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Step 2/2: Running evaluation on pruned graph...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
  echo -e "${YELLOW}⚠️  Warning: OPENAI_API_KEY not set. Evaluation may fail.${NC}"
  echo "   Set it with: export OPENAI_API_KEY=your_key"
  echo ""
fi

# Update ablation config to point to pruned artifacts (already done in JSON)
# Run eval using the pruned graph config
python eval/generate_answers.py \
  --config superhfbp_pruned_graph \
  --questions "$NUM_QUESTIONS" \
  --output-dir "$EVAL_OUTPUT"

if [ $? -ne 0 ]; then
  echo -e "${RED}❌ Evaluation failed!${NC}"
  exit 1
fi

echo ""
echo -e "${GREEN}✅ Evaluation completed successfully!${NC}"
echo ""

# Summary
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     Pipeline Complete!                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}📁 Outputs:${NC}"
echo "  Pruned graph:     $PRUNED_OUTPUT/"
echo "  Eval results:     $EVAL_OUTPUT/"
echo ""
echo -e "${GREEN}📄 Next Steps:${NC}"
echo "  1. Review pruning metadata: cat $PRUNED_OUTPUT/pruning_metadata.json"
echo "  2. Review eval results:     cat $EVAL_OUTPUT/superhfbp_pruned_graph_results.json"
echo "  3. Run evaluation metrics:  python eval/evaluate_answers.py"
echo ""
echo -e "${GREEN}🎉 All done!${NC}"
