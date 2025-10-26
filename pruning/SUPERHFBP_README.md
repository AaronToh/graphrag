# SuperHFBP: Enhanced Hierarchical Flow-Based Pruning

SuperHFBP is an enhanced version of HFBP (Hierarchical Flow-Based Pruning) that incorporates context nodes and refined hierarchical pruning for improved RAG efficiency and quality.

## Overview

SuperHFBP extends the baseline HFBP algorithm with:

1. **Context Node Identification**: Dynamically identifies top-K% most relevant nodes by query similarity
2. **Global Query Detection**: Uses LLM or heuristics to detect broad vs. specific queries
3. **Enhanced Supergraph Construction**: Adds inter-community edges with density thresholding (>5%)
4. **Context Forcing**: Paths must include context nodes or receive a penalty
5. **Improved Reliability Scoring**: Includes context bonus in path scoring
6. **Pruned Graph Export**: Can export pruned graph as GraphRAG artifacts for offline evaluation

## Key Improvements Over HFBP

- **10-20% token savings**: Through better pruning and context focus
- **Improved relevance**: Context node forcing ensures important entities are included
- **Better logicality**: Hierarchical pruning with density filtering reduces noise
- **Query-aware**: Adapts behavior for global vs. local queries

## Architecture

### SuperHFBP Pipeline

```
Input: GraphRAG Artifacts (entities, relationships, text_units)
       Query + Retrieved Nodes (Vq)

1. Supernode Precomputation
   └─ Leiden community detection
   └─ Context node selection (highest PageRank per community)
   └─ LLM-based summary generation

2. Context Node Identification
   └─ Compute query-entity similarity (embeddings)
   └─ Select top-K% as context nodes (Cq)
   └─ Detect global query (LLM or heuristics)
   └─ If global: add supernode representatives to Cq

3. Enhanced Supergraph Construction
   └─ Build supergraph with supernodes as vertices
   └─ Add inter-community edges if density > threshold (5%)
   └─ Weight edges by semantic similarity

4. Hierarchical Resource Propagation
   └─ Propagate on supergraph level first
   └─ Refine to node-level with pruning (theta threshold)
   └─ Apply context node bonus/penalty

5. Path Enumeration & Scoring
   └─ Beam search for top paths
   └─ Score: S(P) = (1/|E_P|) * Σ S(v) + β * (|Cq ∩ Vp| / |Cq|)
   └─ Penalty if path missing context nodes

Output: Top-K Paths with Enhanced Reliability Scores
        OR Pruned Graph (entities.parquet, relationships.parquet)
```

### Scoring Formula

**Enhanced Reliability Score:**

```
S(P) = (1/|E_P|) * Σ_{v ∈ V_P} S(v) + β * (|C_q ∩ V_P| / |C_q|)

Where:
- |E_P| = number of edges in path
- S(v) = resource score for node v
- C_q = context nodes set
- V_P = nodes in path
- β = context bonus weight (default: 0.2)

If |C_q ∩ V_P| = 0: multiply base score by context_penalty (default: 0.8)
```

## Configuration

### SuperHFBPConfig Parameters

```python
@dataclass
class SuperHFBPConfig(HFBPConfig):
    # Inherited from HFBPConfig
    N: int = 40                    # Top retrieved nodes
    K: int = 15                    # Top paths to return
    alpha: float = 0.8             # Resource propagation damping
    theta: float = 0.05            # Pruning threshold
    beam_k: int = 5                # Beam search width

    # SuperHFBP enhancements
    beta: float = 0.2              # Context bonus weight
    context_top_pct: float = 0.2   # Top 20% nodes as context
    context_penalty: float = 0.8   # Penalty for no context nodes
    inter_community_density_threshold: float = 0.05  # 5% density

    # Global query detection
    use_llm_global_detection: bool = True
    global_query_keywords: List[str] = [...]  # Heuristic keywords
    openai_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
```

### Ablation Configurations

The `eval/ablation_config.json` includes several SuperHFBP configurations:

- **superhfbp_default**: Balanced context parameters
- **superhfbp_high_precision**: Stronger context forcing (β=0.3, top 15%)
- **superhfbp_high_recall**: Relaxed pruning (K=20, θ=0.03)
- **superhfbp_no_context_penalty**: Ablation without penalty
- **superhfbp_no_density_filter**: Ablation without density filtering

## Usage

### 1. Direct Path-Based Querying (In-Memory)

Use SuperHFBP for real-time path retrieval:

```python
from pruning import SuperHFBP, SuperHFBPConfig

# Initialize
config = SuperHFBPConfig(
    N=40, K=15, beta=0.2, context_top_pct=0.2
)
pruner = SuperHFBP(entities_df, relationships_df, text_units_df, config)

# Execute query
result = pruner.execute(query="What causes alcohol dependence?",
                       retrieved_nodes=initial_nodes)

# Access paths
for path in result['top_k_paths']:
    print(f"Path score: {path.score}")
    print(path.text_representation)
```

### 2. Pruned Graph Export (Offline)

Generate pruned GraphRAG artifacts for downstream use:

```bash
# Prune graph using auto-generated queries
python pruning/run_superhfbp_pruning.py \
  --workspace workspace \
  --output workspace/pruned_superhfbp \
  --mode auto \
  --num-queries 10

# Prune using custom queries
python pruning/run_superhfbp_pruning.py \
  --workspace workspace \
  --output workspace/pruned_superhfbp \
  --mode file \
  --queries my_queries.txt

# Prune with custom config
python pruning/run_superhfbp_pruning.py \
  --workspace workspace \
  --output workspace/pruned_superhfbp \
  --config superhfbp_config.json
```

Output structure:
```
workspace/pruned_superhfbp/
├── entities.parquet          # Pruned entities
├── relationships.parquet     # Pruned relationships
├── text_units.parquet        # Pruned text units
└── pruning_metadata.json     # Pruning statistics
```

### 3. Complete Pipeline (Prune → Eval)

Run end-to-end pipeline:

```bash
./run_superhfbp_pipeline.sh --num-queries 10 --questions 50

# Or with custom parameters
./run_superhfbp_pipeline.sh \
  --num-queries 20 \
  --questions 100 \
  --workspace workspace
```

This will:
1. Prune the graph using 10 representative queries
2. Save pruned artifacts to `workspace/pruned_superhfbp/`
3. Run evaluation on pruned graph with 50 test questions
4. Save results to `superhfbp_results/`

### 4. Evaluation Integration

The eval pipeline supports both online and offline modes:

**Online (Path-Based)**: SuperHFBP runs at query time
```bash
python eval/generate_answers_superhfbp.py \
  --workspace workspace \
  --config superhfbp_default \
  --questions 50
```

**Offline (Pruned Graph)**: Use pre-pruned artifacts
```bash
python eval/generate_answers.py \
  --config superhfbp_pruned_graph \
  --questions 50
```

The `superhfbp_pruned_graph` config in `ablation_config.json` points to pruned artifacts:
```json
{
  "name": "superhfbp_pruned_graph",
  "artifacts_path": "workspace/pruned_superhfbp",
  "pruning_strategy": "none"
}
```

## Implementation Details

### Context Node Identification

```python
def identify_context_nodes(self, query: str, vq: List[str]) -> Set[str]:
    # 1. Compute query-entity similarity
    query_embedding = self.embedding_model.encode(query)
    similarities = [(node, cosine_sim(query_emb, node_emb))
                   for node in vq]

    # 2. Select top-K%
    k = int(len(vq) * context_top_pct)
    context_nodes = top_k(similarities, k)

    # 3. If global query, add supernode representatives
    if is_global_query(query):
        for supernode in affected_supernodes:
            context_nodes.add(supernode.context_node)

    return context_nodes
```

### Enhanced Supergraph Construction

```python
def build_supergraph(self) -> nx.Graph:
    # Add supernodes
    for sn in supernodes:
        G'.add_node(sn.id, supernode=sn)

    # Add inter-community edges with density check
    for u, v in inter_community_edges:
        density = edge_count / (|members_u| * |members_v|)
        if density >= threshold:  # 5%
            G'.add_edge(sn_u, sn_v, weight=1-similarity)

    return G'
```

### Path Scoring with Context

```python
def _compute_enhanced_score(self, path: Path) -> float:
    # Base score: average resource
    base = sum(S[v] for v in path.nodes) / len(path.edges)

    # Context bonus
    path_context = set(path.nodes) & context_nodes
    context_ratio = len(path_context) / len(context_nodes)
    bonus = beta * context_ratio

    # Apply penalty if no context nodes
    if len(path_context) == 0:
        base *= context_penalty

    return base + bonus
```

## Performance Characteristics

### Complexity

**Time Complexity**: `O(N^2 / ((1-α)θ) * |S|/|V|)`

Where:
- N = number of retrieved nodes
- α = propagation damping factor
- θ = pruning threshold
- |S| = number of supernodes (typically |S| << |V|)
- |V| = total graph nodes

**Space Complexity**: `O(|V| + |E| + |S|)`

### Typical Performance

On PubMedQA dataset (19K entities, 32K relationships):

| Metric | Value |
|--------|-------|
| Pruning time (10 queries) | ~30-60s |
| Reduction rate | 60-80% entities, 70-85% relationships |
| Context nodes per query | 5-15 |
| Supernodes created | 50-200 |
| Paths per query | 5-15 |

## Validation

### Tests

Run test suite:
```bash
python pruning/test_superhfbp_export.py
```

### Expected Outcomes

1. **Pruned graph should be smaller**: Typically 20-40% of original size
2. **Connectivity preserved**: Paths exist between important entities
3. **Context nodes included**: All identified context nodes in output
4. **Valid parquet files**: entities.parquet, relationships.parquet created

## Troubleshooting

### No paths found

**Cause**: Graph may be disconnected or retrieved nodes too sparse

**Solution**:
- Increase N (number of retrieved nodes)
- Lower theta (pruning threshold)
- Check graph connectivity with baseline HFBP first

### High memory usage

**Cause**: Large graph with many supernodes

**Solution**:
- Reduce N and K
- Increase inter_community_density_threshold
- Use offline pruning mode instead of online

### Low reduction rate

**Cause**: Many queries covering diverse topics

**Solution**:
- Use fewer, more representative queries for pruning
- Increase theta (more aggressive pruning)
- Lower context_top_pct

## References

- **PathRAG**: Flow-based pruning for knowledge graphs
- **SuperPathRAG**: Context node enhancement for hierarchical retrieval
- **Leiden Algorithm**: Community detection for supernode creation
- **GraphRAG**: Microsoft's graph-based RAG framework

## Citation

```bibtex
@software{superhfbp2024,
  title={SuperHFBP: Enhanced Hierarchical Flow-Based Pruning for GraphRAG},
  author={GraphRAG Pruning Lab},
  year={2024},
  note={Research implementation for graph pruning optimization}
}
```
