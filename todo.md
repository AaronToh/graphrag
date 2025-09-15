# GraphRAG Pruning Lab - Comprehensive Implementation Plan

## 🎯 **PHASE 1: Environment Setup & Verification** ✅ PARTIALLY COMPLETE

### **1.1 Verify Official GraphRAG Configuration**
- [x] Use `python -m graphrag init` to generate official settings.yaml
- [x] Compare generated format with our custom version
- [x] Update settings.yaml with correct structure and our research modifications
- [x] Verify settings.yaml loads correctly with `python -c "from graphrag.config import load_config; load_config('.')"`

### **1.2 Environment & Dependencies**
- [ ] Install and verify all dependencies from requirements.txt
- [ ] Test graphrag CLI commands work: `python -m graphrag --help`
- [ ] Verify API keys and environment variables are properly configured
- [ ] Test basic graphrag functionality with minimal data

### **1.3 Project Structure Validation**
- [x] Verify all directories exist: data/, workspace/, ingest/, pruning/, eval/
- [x] Confirm all __init__.py files are in place
- [ ] Test Python imports work: `python -c "from pruning.scoring_utils import GraphScorer"`

---

## 🎯 **PHASE 2: Data Preparation & Baseline**

### **2.1 Create Test Dataset**
- [ ] Research and select appropriate test dataset (Wikipedia articles, technical docs, etc.)
- [ ] Download/create sample documents for initial testing (start with 5-10 documents)
- [ ] Place documents in `data/input/` directory
- [ ] Verify file formats are compatible (.txt, .md supported)
- [ ] Create initial file: `data/input/sample_article_01.txt`

### **2.2 Build Baseline GraphRAG Index**
- [ ] Run `python ingest/build_index.py --verbose` to create baseline index
- [ ] Verify output artifacts are generated:
  - `workspace/output/entities.parquet`
  - `workspace/output/relationships.parquet`
  - `workspace/output/communities.parquet`
  - `workspace/output/community_reports.parquet`
  - `workspace/output/lancedb/` directory
- [ ] Inspect generated artifacts to understand data structure
- [ ] Document baseline statistics (node count, edge count, communities)

### **2.3 Create Gold Standard Evaluation Data**
- [ ] Design evaluation questions that test different aspects of RAG
- [ ] Create Q&A pairs in `data/gold/evaluation_data.json`
- [ ] Include questions testing:
  - Factual recall
  - Multi-hop reasoning
  - Contextual understanding
  - Entity relationships
- [ ] Start with 10-20 question-answer pairs

---

## 🎯 **PHASE 3: Scoring Framework Implementation**

### **3.1 Node Scoring Methods**
- [ ] Implement `GraphScorer.score_nodes_degree_centrality()`
  - Calculate degree centrality using NetworkX
  - Normalize scores between 0-1
  - Handle disconnected components
- [ ] Implement `GraphScorer.score_nodes_frequency()`
  - Use entity mention frequency from entities.parquet
  - Handle missing frequency data gracefully
- [ ] Implement `GraphScorer.score_nodes_semantic_relevance()`
  - Use embeddings to compute query-entity similarity
  - Implement placeholder for now, enhance later
- [ ] Implement `GraphScorer.score_nodes_custom_method()`
  - Add your novel scoring algorithm here
  - Document the scoring logic and rationale

### **3.2 Edge Scoring Methods**
- [ ] Implement `GraphScorer.score_edges_weight()`
  - Extract edge weights from relationships.parquet
  - Handle missing weight values
- [ ] Implement `GraphScorer.score_edges_plausibility()`
  - Create relationship type plausibility scoring
  - Use domain knowledge or KGE models
- [ ] Implement `GraphScorer.score_edges_custom_method()`
  - Add your novel edge scoring algorithm

### **3.3 Community Scoring Methods**
- [ ] Implement `GraphScorer.score_communities_size()`
  - Count entities per community from entities.parquet
- [ ] Implement `GraphScorer.score_communities_density()`
  - Calculate graph density for each community subgraph
- [ ] Implement `GraphScorer.score_communities_custom_method()`
  - Add your novel community scoring algorithm

### **3.4 Combined Scoring Framework**
- [ ] Implement `GraphScorer.get_combined_node_scores()`
  - Support weighted combination of multiple scoring methods
  - Add ranking functionality
  - Save scores to CSV/parquet for analysis
- [ ] Implement `GraphScorer.get_combined_edge_scores()`
  - Similar to node scoring but for edges
- [ ] Implement `GraphScorer.get_combined_community_scores()`
  - Similar to node scoring but for communities

---

## 🎯 **PHASE 4: Pruning Framework Implementation**

### **4.1 Node Pruning Strategies**
- [ ] Implement `GraphPruner.prune_nodes(strategy="top_k")`
  - Keep top-k highest scoring nodes
  - Update edges that reference pruned nodes
- [ ] Implement `GraphPruner.prune_nodes(strategy="threshold")`
  - Keep nodes above score threshold
- [ ] Implement `GraphPruner.prune_nodes(strategy="percentile")`
  - Keep top percentile of nodes
- [ ] Implement custom pruning strategies

### **4.2 Edge Pruning Strategies**
- [ ] Implement `GraphPruner.prune_edges(strategy="top_k")`
  - Keep top-k edges per node (most common strategy)
  - Handle degree distribution changes
- [ ] Implement `GraphPruner.prune_edges(strategy="threshold")`
  - Keep edges above weight/plausibility threshold
- [ ] Implement `GraphPruner.prune_edges(strategy="percentile")`
  - Keep top percentile of edges

### **4.3 Community Pruning Strategies**
- [ ] Implement `GraphPruner.prune_communities(strategy="top_k")`
  - Keep top-k communities by score
- [ ] Implement `GraphPruner.prune_communities(strategy="recluster")`
  - Re-run community detection after node/edge pruning
  - Use Leiden algorithm with different parameters

### **4.4 Pruning Pipeline Orchestration**
- [ ] Implement `GraphPruner.apply_pruning_pipeline()`
  - Execute scoring → pruning → validation sequence
  - Save pruning configuration and results
  - Generate pruning summary statistics

---

## 🎯 **PHASE 5: Evaluation Framework Implementation**

### **5.1 Answer Quality Evaluation**
- [ ] Implement `RAGEvaluator.evaluate_answer_quality(method="llm_judge")`
  - Use GPT-4o-mini to score answer quality
  - Implement rubric-based evaluation
- [ ] Implement `RAGEvaluator.evaluate_answer_quality(method="semantic_similarity")`
  - Use embeddings to compare predicted vs reference answers
- [ ] Implement `RAGEvaluator.evaluate_answer_quality(method="rouge")`
  - Calculate ROUGE scores for answer similarity

### **5.2 Retrieval Quality Evaluation**
- [ ] Implement `RAGEvaluator.evaluate_retrieval_quality()`
  - Calculate Hit@k (k=1,3,5,10)
  - Calculate Mean Reciprocal Rank (MRR)
  - Calculate Mean Average Precision (MAP)
  - Handle multiple relevant documents per query

### **5.3 Efficiency Evaluation**
- [ ] Implement `RAGEvaluator.evaluate_efficiency()`
  - Track token usage per query
  - Measure response latency
  - Monitor memory usage (if available)
  - Calculate efficiency metrics (tokens/second, etc.)

### **5.4 Graph Structure Analysis**
- [ ] Implement `RAGEvaluator.evaluate_graph_structure()`
  - Compare node/edge retention rates
  - Analyze connectivity changes
  - Measure graph density changes
  - Track community structure preservation

### **5.5 Evaluation Pipeline**
- [ ] Implement `RAGEvaluationRunner.run_evaluation_pipeline()`
  - Load test queries and gold answers
  - Query both baseline and pruned systems
  - Collect and analyze all metrics
  - Generate comparison reports
- [ ] Implement `RAGEvaluationRunner.run_ablation_study()`
  - Test multiple pruning configurations
  - Compare different pruning strategies
  - Identify optimal pruning parameters

---

## 🎯 **PHASE 6: Experimentation & Analysis**

### **6.1 Baseline vs Pruned Comparison**
- [ ] Run comprehensive evaluation with baseline system
- [ ] Apply different pruning strategies and parameters
- [ ] Compare performance across all metrics
- [ ] Identify trade-offs between efficiency and quality

### **6.2 Ablation Studies**
- [ ] Test individual scoring methods in isolation
- [ ] Compare different pruning strategies
- [ ] Analyze impact of different parameter settings
- [ ] Identify most effective combinations

### **6.3 Graph Analysis**
- [ ] Analyze pruned graph structure using GraphML snapshots
- [ ] Visualize before/after graphs (if possible)
- [ ] Study impact on different entity types
- [ ] Investigate community structure changes

### **6.4 Performance Optimization**
- [ ] Profile code performance bottlenecks
- [ ] Optimize scoring calculations for large graphs
- [ ] Implement parallel processing where beneficial
- [ ] Consider memory-efficient data structures

---

## 🎯 **PHASE 7: Documentation & Reporting**

### **7.1 Results Documentation**
- [ ] Create comprehensive experiment log
- [ ] Document all pruning configurations tested
- [ ] Record performance metrics for each experiment
- [ ] Maintain reproducible experiment setup

### **7.2 Final Report**
- [ ] Summarize findings and insights
- [ ] Create visualizations of results
- [ ] Document methodology and limitations
- [ ] Provide recommendations for production use

### **7.3 Code Documentation**
- [ ] Add comprehensive docstrings to all functions
- [ ] Create usage examples and tutorials
- [ ] Document configuration options
- [ ] Provide troubleshooting guide

---

## 🔧 **Infrastructure & DevOps**

### **Local LLM Setup** (Optional)
- [ ] Research local LLM deployment options (Ollama, LM Studio, etc.)
- [ ] Set up quantized model for cost reduction
- [ ] Test GraphRAG with local LLM
- [ ] Compare local vs API performance

### **HPC/Cluster Deployment** (If needed)
- [ ] Containerize the application (Docker)
- [ ] Create SLURM job scripts for cluster execution
- [ ] Implement distributed processing for large graphs
- [ ] Set up monitoring and logging for cluster jobs

---

## 📋 **Current Status & Next Steps**

### **Completed:**
- ✅ Project structure setup
- ✅ Official GraphRAG configuration verified and updated
- ✅ Framework skeleton implemented
- ✅ Dependencies documented

### **Immediate Next Steps (Priority Order):**
1. **Test the framework** - Run `python ingest/build_index.py --help` to verify it works
2. **Create sample data** - Add 2-3 test documents to `data/input/`
3. **Build baseline** - Generate initial GraphRAG index
4. **Implement first scoring method** - Start with degree centrality
5. **Test scoring framework** - Verify data loading and processing works

### **Long-term Goals:**
- Answer the research question: *Does pruning improve efficiency without hurting quality?*
- Develop novel pruning algorithms
- Contribute insights to the GraphRAG community

---

## ❓ **Open Questions & Research Areas**

### **GraphRAG Internals:**
- How does Microsoft handle incremental indexing?
- What built-in optimizations exist?
- How do different entity types affect pruning effectiveness?

### **Pruning Research:**
- Which scoring methods work best for different domains?
- How much can we prune before quality degrades?
- Are there domain-specific pruning strategies?

### **Evaluation Methodology:**
- What metrics best capture RAG performance?
- How to balance different evaluation aspects?
- How to ensure evaluation is reproducible?

# Task overview
1. Connect to local LLM - deploying quantized model locally? / HPC 
2. Workflow, work out a diagram etc
    a. Ingest
    b. Pruning algo 
    c. Run Evals
3. Work on Algo, verify that it can run locally 
4. potentially run on the compute cluster if local compute is not enough

# Hanging questions: 
how does microsoft's RAG handle the ingestion of data and building of the graph? 
what kinda of optimizations are already there
