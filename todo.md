# Next steps

* [ ] Define the **test set** for RAG evaluation (can start with toy text).
* [ ] Implement `ingest/build_index.py` (wraps `graphrag index --method standard`).
* [ ] Sketch `pruning/prune_graph.py` with scoring + top-k pruning.
* [ ] Build a **dummy eval harness** in `eval/run_eval.py`.
* [ ] Iterate scoring/pruning methods; record results.


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
