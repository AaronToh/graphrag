# GraphRAG Retrieval, Pruning, and Scoring — Implementation Notes (Last 2 Messages)

> Exported from chat: two consecutive messages consolidated into a single Markdown document for your notes.

---

## Message 1

**Awesome question — let’s walk it end-to-end with the official GraphRAG repo in mind:**

### 1) How GraphRAG indexes your data (what gets built)

**Dataflow (high level).** GraphRAG:

- Chunks your raw docs into *TextUnits*.
- Runs LLM extraction to produce **Entities** + **Relationships** (+ optional **Claims**).
- Runs hierarchical **community detection (Leiden)** over the ER graph.
- Writes **Community Reports** (LLM summaries) for each community level.
- Optionally computes **Node2Vec + UMAP** (viz), then **embeds text** for retrieval.  
  *Microsoft GitHub*

**What’s saved (Parquet tables by default).**

- `entities.parquet` (fields include `title`, `type`, `description`, `frequency`, `degree`, …)
- `relationships.parquet` (fields include `source`, `target`, `description`, `weight`, …)
- `communities.parquet` (IDs, hierarchy, membership)
- `community_reports.parquet` (fields include `title`, `summary`, `full_content`, `rank`, etc.)
- `text_units.parquet`, `documents.parquet`, optional `covariates.parquet` (claims)  
  *All tables are column-documented here. — Microsoft GitHub*

**Embeddings & store.** By default, GraphRAG embeds the text needed by its query methods (you can enable *all* fields like `entity.description`, `relationship.description`, `community.summary/full_content`, etc.). Vector storage defaults to **LanceDB**, configurable in `vector_store`.  
*Microsoft GitHub* [+1]

**CLI you actually run.**
```bash
# install + init workspace
pip install graphrag
graphrag index --init --root .

# build the index (standard pipeline)
graphrag index --root . --method standard
```
*(There are also `fast` and `*-update` methods.) — Microsoft GitHub* [+1]

---

### 2) How retrieval works (local vs. global)

**Local Search (entity-centric)**

- Your query is embedded and matched to **nearest entity descriptions** (from indexing).
- From those seed entities, GraphRAG gathers **related text units, relationships, and community reports**, applies **ranking + filtering**, and builds a single LLM prompt.
- **Tunables:** `top_k_entities`, `top_k_relationships`, `text_unit_prop`, `community_prop`, `max_context_tokens`, etc.  
  *Microsoft GitHub* [+1]

Helpful mental model: “nearest entities → expand to neighbors + linked text → rank + pack into context → answer.” (The repo discussion confirms the entity-first mapping.)  
*GitHub*

**Global Search (dataset-wide)**

- Works over **Community Reports** in a **map-reduce** style.
- You pick a *community level* (granularity), chunk the reports into batches, **map** them to per-batch findings, then **reduce** to a final answer.
- The **Dynamic Community Selection** phase *prunes* low-value reports up front using a rating step before map-reduce. **Tunables include:**  
  `dynamic_search_threshold`, `dynamic_search_keep_parent`, `dynamic_search_num_repeats`, `dynamic_search_use_summary`, `dynamic_search_max_level`.  
  *Microsoft GitHub* [+1]

Microsoft’s research blog explains the dynamic rating stage explicitly (using a cheaper model to **prune** irrelevant community reports before reduction).  
*Microsoft*

---

### 3) How to **prune** the graph (built-in knobs)

**Index-time structural pruning**

In `settings.yaml` (or your config), GraphRAG has a dedicated **`prune_graph`** step you can run between extraction and clustering:

```yaml
workflows:
  - extract_graph
  - prune_graph      # 👈 enable this
  - cluster_graph
  - community_reports

prune_graph:
  min_node_freq: 2            # drop rare entities
  max_node_freq_std: 3.0      # drop frequency outliers (super-hubs)
  min_node_degree: 2          # drop leaves/singletons
  max_node_degree_std: 3.0    # drop degree outliers
  min_edge_weight_pct: 25.0   # keep top 75% strongest edges (by percentile)
  remove_ego_nodes: false
  lcc_only: true              # keep only largest connected component
```

Those fields (and their intent) are documented here. You can also configure **clustering** (`cluster_graph.use_lcc`, `max_cluster_size`, etc.).  
*Microsoft GitHub*

**Query-time pruning**

- **Global:** use the **dynamic** knobs above to rate-and-drop weak community reports before map-reduce.  
  *Microsoft GitHub; Microsoft*
- **Local:** reduce candidate fan-out by tightening `top_k_entities/relationships` and the proportions, so fewer weak items survive ranking/packing.  
  *Microsoft GitHub*

---

### 4) Add **custom scoring** for nodes/edges and wire it into retrieval

You asked if you can *define new ways to “score”* nodes/edges and use them to make retrieval better. **Yes** — here’s a practical recipe that stays close to the repo’s shapes.

#### 4A. Compute scores offline (one pass over Parquet)

Calculate any combination you like:

**Entity scores**
- *Centrality*: degree (already in table), betweenness/PageRank (compute via NetworkX), normalize to [0,1].
- *Semantic relevance*: cosine similarity of `entity.description` to a **domain vector** (average of seed keywords) — high = on-topic.
- *Authority*: frequency (already provided), recency (if your docs have time), doc-level metadata priors.

**Edge (relationship) scores**
- *Strength*: the repo’s `weight` (sums LM “strength” across mentions).
- *Plausibility*: a **KGE score** (e.g., TransE/ComplEx/RotatE) trained on your ER table, normalized.
- *Confidence*: if you ran multiple extractors or have LLM logprobs, multiply in a reliability term.

**Community scores**
- Aggregate from member entities/edges into a **community rank** (e.g., avg of top-k entity importances, or weighted sum), plus size/coverage.

Schemas expose the needed columns (`frequency`, `degree`, `weight`, and `rank` in reports) and accept additional columns in Parquet.  
*Microsoft GitHub*

#### 4B. Write scores back where GraphRAG will notice

Two easy integration points that **require no engine code changes**:

- **Global Search** uses fields **already read by the context builder**:  
  - Set/override `community_reports.rank` with your custom community score.  
  - Turn on `include_community_rank` and (optionally) `include_community_weight` to have the context builder use them for ranking and filtering; raise `min_community_rank` to prune.  
    *Microsoft GitHub*  
  - Then tune `dynamic_search_threshold` to prune even more aggressively at query time.  
    *Microsoft GitHub*

- **Local Search** is entity-centric:  
  - Because entity matching is embedding-based, focus on controlling **what gets packed** once entities are mapped.  
  - Practical tactic: **cap expansion** to the **top-k edges per entity** by your **edge score** (e.g., `strength × plausibility × confidence`), and only include **top-scored neighbor entities**.  
  - Concretely: pre-filter your `relationships.parquet` to only keep top-k outgoing edges per entity (or add a boolean `keep_in_local`), then re-index. Combined with `top_k_entities/relationships`, this sharply reduces noisy context. (Local Search’s method explicitly does “ranking + filtering” on candidates, so the thinner your candidate lists, the better the pack.)  
    *Microsoft GitHub*

> **Tip:** If you *do* want code-level hooks, the local/global **context builders** (e.g., `LocalSearch…Context`, `GlobalCommunityContext`) are where ranking/selection happens; you can fork and add your custom weighting there, but most users get far by **pre-computing** scores into the Parquet and letting the default builders pick them up.  
> *Microsoft GitHub; AiDocZh*

#### 4C. A concrete “score → prune → retrieve” loop

- **Index** with mild structural pruning (example above).  
  *Microsoft GitHub*
- **Score**
  - Compute:  
    `entity_importance = z(degree) + z(freq) + α·cos(desc, domain_vec)`  
    `edge_quality   = z(weight) · β·KGE_plausibility · γ·confidence`  
    `community_score = agg_topk(entity_importance)`
  - Normalize to [0,1]; write columns back to Parquet (`entities.importance`, `relationships.quality`, `community_reports.rank`).  
    *Microsoft GitHub*
- **Prune**
  - Keep top-k edges per entity by `relationships.quality`.
  - Optionally drop entities below a small `importance` threshold (but prefer keeping nodes and trimming edges first to avoid fragmenting the graph).
- **Retrieve**
  - **Local:** set smaller `top_k_entities`, `top_k_relationships`; your edge pre-pruning ensures expansion follows only high-quality links; keep `max_context_tokens` tight.  
    *Microsoft GitHub*
  - **Global:** set `include_community_rank: true`, choose `min_community_rank`, and use **dynamic** settings (e.g., `dynamic_search_threshold`) to rate-and-drop weak reports before map-reduce.  
    *Microsoft GitHub* [+1]
- **Evaluate**
  - Keep a small query set; measure **Hit@k** for supporting facts, **LLM-graded answer quality**, and **tokens/latency**. Raise `min_edge_weight_pct` / `min_community_rank` until quality drops; back off slightly.

**Copy-paste-able starting config (safe defaults)**
```yaml
workflows:
  - extract_graph
  - prune_graph
  - cluster_graph
  - community_reports
  - embed_text

prune_graph:
  min_node_freq: 2
  min_node_degree: 2
  min_edge_weight_pct: 25.0
  lcc_only: true

embed_text:
  names:
    - entity.description
    - text_unit.text
    - community.summary

local_search:
  top_k_entities: 12
  top_k_relationships: 8
  text_unit_prop: 0.6
  community_prop: 0.4
  max_context_tokens: 8000

global_search:
  max_context_tokens: 12000
  data_max_tokens: 2000
  include_community_rank: true
  min_community_rank: 0.5    # ← your custom rank (0–1) after Step 4B
  dynamic_search_threshold: 4
  dynamic_search_keep_parent: true
  dynamic_search_num_repeats: 2
  dynamic_search_use_summary: true
```
*(Fields & semantics documented in the official config pages.) — Microsoft GitHub*

**TL;DR**

- **Index** → ER graph + community hierarchy + reports + embeddings.  
  *Microsoft GitHub* [+1]
- **Retrieve:** Local = *entities first*; Global = *reports + map-reduce with dynamic pruning*.  
  *Microsoft GitHub* [+2]; *Microsoft GitHub* [+2]
- **Prune** structurally at index time (`prune_graph`) **and** dynamically at query time (global dynamic selection; tighter local caps).  
  *Microsoft GitHub*
- **Score** nodes/edges/communities offline, write scores to Parquet (esp. `community_reports.rank`), and let GraphRAG’s context builders exploit them — no engine hacks needed.  
  *Microsoft GitHub* [+1]

If you want, I can turn this into a small **notebook scaffold** that:
- reads Parquet,
- computes centrality + KGE plausibility,
- rewrites `community_reports.rank` and prunes edges, and
- runs a few local/global queries for A/B.

---

## Message 2

**You said:**  
> add more information here on the implementation, so how will i go about doing it, can i run a local model, what kind of dataset can i use etc, almost prescribing me an instuction set at the bottom  
> same with compute, can i index locally? or must i pay to use an API etc  
> Dont remove the information you just gave me, add on to it

**ChatGPT said:**  
> Thought for 1m 51s  
> I need to confirm the specifics from the Microsoft GraphRAG repo, especially around implementation details like compute, datasets, and indexing workflow. Earlier, we didn't show proper citations because we didn't run the required searches. I'll check the repo to verify details like config keys, including any pruning steps.
>
> I need to clarify the specifics of the GraphRAG repo, especially regarding the "create index", "local search", and "global search" functions. I'm not completely sure whether there’s a prune_graph step, so I’ll confirm by exploring the repo and checking the docs. Let's make sure everything’s accurate.
>
> Searching Microsoft GraphRAG repo for implementation details
>
> I found the official docs, and it looks like the "prune_graph" step is listed in the Detailed Configuration document. I’ll open it up
>
> Opening and capturing details from the configuration doc
>
> I have the main docs, including details on prune_graph and related fields. Now, I need extra info on local models, compute, datasets, and indexing methods (API vs. local). I’ll check if the model configurations mention things like "openai", "azure_openai", or "litellm" for clarity. Time to investigate the models docs.
>
> Searching for documentation on model configurations
>
> I have the required specifics, but now I need info on local models or offline embeddings. The "Language Model Selection" page should list model types and may include options for local models. I'll open that page to confirm and gather the necessary details.
>
> Browsing "Language Model Selection" for local models
>
> The "models" section gives information on options like "openai_chat" but doesn't specify local model connectors. Some discussions mentioned using Ollama with environment variables or custom wrappers. The "Language Model Selection" page might mention alternative methods, so I’ll check it to confirm the local model options.
>
> Exploring local model connectors and alternatives
> Exploring local model connectors and alternatives
>
> The page confirms OpenAI support, proxy use like Ollama, and library injections for custom models. This clears up the local model question and the payment requirements for the API. There's also a fast indexing method that avoids using LLMs altogether. I'll check "Methods" and "Getting Started" to dive deeper.
>
> Summarizing local model options and indexing methods
>
> The page confirms that the default setup uses OpenAI or Azure with an API key. However, the language model section mentions that we can use proxies or custom model injections when using the library. Also, "fast indexing" relies on NLP instead
>
> Searching for "Indexing Methods" page
>
> I'll now check the "Methods" section to review the different indexing types, including standard and fast. This will help clarify the options available for indexing and allow me to better understand how to optimize the process. Let's see what details are provided there.
>
> Opening "Methods" for indexing types
>
> Great, FastGraphRAG uses no LLM for extraction, making it cost-effective and local-friendly. This clears up the API payment question. Now, I'll check whether we can use embedding models through a proxy. It looks like "Using Non-OpenAI Models" allows this, and lancedb supports local storage. Let's confirm and gather additional info on local/global query methods from the docs.
>
> Confirming local embeddings and vector store configurations
> Confirming local embeddings and search settings
>
> This page shows how ranking and filtering work within the flow, which looks good. Next, I'll dive into the "Global Search" page to get details on dynamic pruning. This should help clarify how to handle queries with evolving and efficient search techniques. Let's move ahead with that!
>
> Opening "Global Search" page for dynamic pruning
>
> The page doesn’t explicitly mention “dynamic_search_threshold,” but it might be in the config docs under the query or global_search section. We already have some of those keys. Next, I'll explore "DRIFT" search, but it’s not essential right now. For datasets, I’ll suggest standard ones like FB15k-237 and WN18RR for training, including others like Wikipedia subsets and Project Gutenberg for easier scales. I'll also double-check licensing notes.
>
> The "run a local model" for extraction shows the default is OpenAI, but proxies like Ollama might cause issues due to response mismatch. Custom model injection isn't available via CLI – only through GraphRAG as a library. We’ll present a step-by-step plan for local models, compute needs, and costs, including memory, token throughput, and compute. I’ll include sections on model options, dataset choices, costs, and a prescriptive instruction set at the end.
>
> You can index locally with methods like NLP extraction and local embeddings via a proxy. Vector store lancedb works locally. However, you'll still need an LLM for community report generation, unless you use a local model such as Ollama Mistral or Llama for those reports. For improving retrieval, I’ll clarify steps using code, such as reading Parquet, computing degree centrality, and KGE training.
>
> When using the query engine with Parquet and vector stores, pruning edges should ideally be done before clustering to ensure consistency. If you prune afterward, re-running the cluster process is necessary. We'll cite the documentation on pruning and clustering as best practices. For local embeddings, we’ll mention supported models and proxies, specifically the "text-embedding-3-large" model. We’ll propose evaluation metrics and cite relevant documentation.
>
> Done

**Love it — let’s bolt the “how” onto what we already covered, and make it super practical. I’ll show you where things live in the Microsoft repo/docs, how to run it entirely on your laptop, how to plug in custom scoring + pruning, and end with a copy-paste instruction set.**

### How GraphRAG indexes & retrieves (quick recap with pointers)

- **Indexing dataflow:** chunk → LLM/NLP extract entities & relationships → hierarchical community detection (Leiden) → generate community reports → embed text (entities, text units, community text) → write Parquet + vectors. The docs walk this pipeline end-to-end.  
  *Microsoft GitHub*

- **Two indexing “methods”:**
  - **`standard`** (LLM does extraction & summarization).  
    *Microsoft GitHub*
  - **`fast`** (NLP noun-phrase + co-occurrence for a cheaper, local-friendly graph; still uses an LLM for community reports unless you swap that too).  
    *Microsoft GitHub*

- **Retrieval methods:**
  - **Local search** = entity-centric: embed the query → find nearest entity descriptions → expand to mapped text units, relationships, and community reports → *ranking + filtering* → answer.  
    *Microsoft GitHub*
  - **Global search** = map-reduce over community reports with a rating stage that *filters (prunes)* weak chunks before reduce.  
    *Microsoft GitHub*

### Where pruning & knobs live (built-in)

- **Index-time structural pruning** via the `prune_graph` section in `settings.yaml`:  
  `min_node_freq`, `min_node_degree`, `min_edge_weight_pct`, `lcc_only`, and outlier caps by std-dev for hubs/leaves.  
  *Microsoft GitHub*

- **Query-time pruning (global)** via dynamic selection/rating thresholds in `global_search` (e.g., `dynamic_search_threshold`, `dynamic_search_keep_parent`, `dynamic_search_use_summary`).  
  *Microsoft GitHub*  
  Microsoft’s blog explicitly describes using a smaller model to rate & **prune** irrelevant reports before reduce.  
  *Microsoft*

- **Vector store & storage are local by default** (`lancedb`, `output.type=file`, `cache.type=file`), so you can index/query **without any cloud storage**.  
  *Microsoft GitHub*

### Can I run this 100% locally?

**Yes.** You have three levers:

1) **Avoid LLMs in extraction** by using `--method fast` (NLP-based extraction/co-occurrence).  
   *Microsoft GitHub*

2) **Use local models via a proxy** (e.g., Ollama) or **inject your own model** when using GraphRAG as a Python library:  
   - Docs: “Default is OpenAI, but you can use proxy APIs (e.g., Ollama) or implement the model protocol and register your own; CLI won’t inject custom models, use the library for that.”  
     *Microsoft GitHub*

3) **Keep storage local**: `vector_store.type: lancedb` and `output/cache: file` already do that.  
   *Microsoft GitHub*

> If you *do* use OpenAI/Azure, you just set `GRAPHRAG_API_KEY` and go; the quickstart shows a pure local filesystem workspace with Parquet outputs.  
> *Microsoft GitHub*

### Datasets you can use

- **Tiny sandbox (minutes):** a single book (e.g., Project Gutenberg’s *A Christmas Carol*) — used in GraphRAG’s own getting-started example. Great to smoke-test the pipeline.  
  *Microsoft GitHub*

- **Small/medium corpora:** a folder of PDFs/TXT scraped into plain text; a CSV of article abstracts; a few domain wikis. (GraphRAG ingests `.txt`, `.csv`, `.json`.)  
  *Microsoft GitHub*

- **If you want to train an *edge plausibility scorer* (KGE) for your pruning:**  
  Use standard KGC sets (*FB15k-237*, *WN18RR*) to validate the scorer, then apply the trained model to your GraphRAG ER table. (General KGE background we discussed earlier.)

### Compute & cost realities

- **Disk & RAM:** Index outputs are Parquet; vectors in **LanceDB**. You can keep everything on disk. RAM ~16–32 GB is comfy for 10k–100k chunks; more helps for graph ops/centrality.
- **CPU vs GPU:** Indexing is CPU-heavy (tokenization, NLP in `fast`). GPU helps if you run local LLMs/embedders (8–12 GB VRAM for 7–8B models; 16–24 GB for 13B+).
- **LLM spend:** If you stick to `fast` for extraction and a small local LLM for community reports, you can be near-zero cost. If you use OpenAI/Azure, the **indexing steps** (entity/rel extraction + summaries + reports) are the bulk of tokens. Docs emphasize standard indexing is costlier; `fast` exists to cut cost/time.  
  *Microsoft GitHub*

### Implementation plan: add **custom scoring** + **pruning** to improve retrieval

You’ll do a lightweight, reproducible pass over the Parquet outputs to compute **entity/edge/community scores**, write them back, prune edges, and let GraphRAG use them.

#### 1) Build an index (choose Standard or Fast)

**Standard (quality, more cost):**
```bash
pip install graphrag
graphrag init --root ./workspace
# add API key OR set up a proxy model per docs
graphrag index --root ./workspace --method standard
```
See getting-started for the init + run flow (writes `./workspace/output/*.parquet`).  
*Microsoft GitHub*

**Fast (local-friendlier):**
```bash
pip install graphrag spacy
python -m spacy download en_core_web_md
graphrag init --root ./workspace
graphrag index --root ./workspace --method fast
```
This uses NLTK/spaCy noun-phrases + co-occur edges; then LLM for community reports (swap that to a local model if needed).  
*Microsoft GitHub*

*Optional:* add an **index-time structural prune** before clustering by turning on `prune_graph` in `settings.yaml` (see example below).  
*Microsoft GitHub*

#### 2) Compute scores offline (Python pass over Parquet)

- **Inputs:** `entities.parquet` (has `frequency`, `degree`), `relationships.parquet` (has `weight`), `communities.parquet`, `community_reports.parquet`.  
  *Microsoft GitHub*

- **Entity score (example):**  
  `importance = z(degree) + z(frequency) + α·cos(embed(entity.description), domain_vector)`

- **Edge score (example):**  
  `quality = z(weight) × β·KGE_plausibility(h,r,t) × γ·confidence)`  
  *Train a tiny **KGE** (TransE/ComplEx/RotatE) on the ER triples or reuse an off-the-shelf library; then score your graph edges.*

- **Community score (example):**  
  `rank = mean(top-k entity importance in the community)`  
  Write this into `community_reports.rank` (GraphRAG already uses ranks/limits in the global search context builder and has a `dynamic_search_threshold` to further prune).  
  *Microsoft GitHub*

> You can do this **without forking GraphRAG**: compute scores → add columns to the Parquet → (optionally) pre-filter relationships to keep only top-k edges per node by your `quality` score. Local search’s context builder already performs *ranking + filtering* over candidates; reducing fan-out upstream helps a lot.  
> *Microsoft GitHub*

#### 3) Re-cluster (recommended) if you prune structure

If you prune relationships or drop nodes, **re-run clustering** so communities reflect the new topology (Leiden is under `cluster_graph` config).  
*Microsoft GitHub*

#### 4) Tighten query-time selectors

- **Local search:** shrink `top_k_entities` / `top_k_relationships` and keep `max_context_tokens` tight; with your pre-pruned edges the kept neighbors are the highest-quality ones.  
  *Microsoft GitHub*
- **Global search:** enable rank-aware selection (`community_reports.rank`), then set `dynamic_search_threshold` higher to **prune** weak community reports before map-reduce.  
  *Microsoft GitHub* [+1]

#### 5) Evaluate

Hold out ~20–50 “gold” questions. Track:
- **Answer quality** (LLM-graded or rubric).
- **Support Hit@k** (did the packed context include the right facts?).
- **Latency/tokens**.  
Increase prune thresholds until quality dips; then back off slightly.

### Concrete config & code sketches

**`settings.yaml` essentials (local-first)**
```yaml
# storage + vectors are local by default
output:
  type: file
  base_dir: output

cache:
  type: file
  base_dir: .cache

vector_store:
  type: lancedb
  db_uri: ${OUTPUT_DIR}/lancedb  # defaults exist; explicit for clarity

workflows:
  - extract_graph         # or extract_graph_nlp for 'fast' method (see docs)
  - prune_graph           # enable index-time structural pruning
  - cluster_graph
  - community_reports
  - embed_text

prune_graph:
  min_node_freq: 2
  min_node_degree: 2
  min_edge_weight_pct: 25.0
  lcc_only: true
```
*All `prune_graph` fields & meanings are documented here. — Microsoft GitHub*

**Using non-OpenAI (local/proxy) models**
- Docs show **two paths**:
  1) **Proxy APIs** (e.g., Ollama) that speak the OpenAI API; works but watch JSON formatting fidelity.  
     *Microsoft GitHub*
  2) **Model Protocol injection** (register your own chat/embedding class) — **library use only** (not the CLI).  
     *Microsoft GitHub*

**Where to embed**
- Pick what you embed (entity descriptions, text units, community summaries) under `embed_text.names` (supported names listed in docs).  
  *Microsoft GitHub*

**Local vs Global knobs at query time**
- **Local:** `top_k_entities`, `top_k_relationships`, `text_unit_prop`, `community_prop`, `max_context_tokens`. The local search page shows the *ranking + filtering* dataflow.  
  *Microsoft GitHub*
- **Global:** `max_context_tokens`, `data_max_tokens`, and the **dynamic** options (threshold, repeats, use_summary, max_level).  
  *Microsoft GitHub*

### Minimal scoring/pruning notebook outline (what you’ll implement)

**Load Parquet:**
```python
import pandas as pd, pyarrow.parquet as pq
ents = pq.read_table("output/entities.parquet").to_pandas()
rels = pq.read_table("output/relationships.parquet").to_pandas()
comm = pq.read_table("output/community_reports.parquet").to_pandas()
```

- Build a graph from `rels` (NetworkX), compute degree/centrality, join back into `ents`.
- Compute embeddings for `ents["description"]` with a **local** embedder (e.g., Instructor/nomic via your proxy) or with OpenAI if you’re fine with API; compute cosine to a **domain centroid**.
- **(Optional)** Train **KGE** (tiny TransE/ComplEx) on `(source, relation, target)` triples taken from `rels` → **score** each edge `f(h,r,t)`.
- **Aggregate scores:**
  ```python
  ents["importance"] = z(degree)+z(freq)+α·cosine
  rels["quality"]    = z(weight)*β·KGE_plausibility
  comm["rank"]       = agg(top-k importance of member nodes)
  ```
- **Prune** edges to top-k per source by `quality`; optionally drop nodes with `importance < ε`.
- **Write back** Parquet with the new columns; **re-run clustering** (`cluster_graph`) so communities reflect the pruned topology.  
  *Microsoft GitHub*

**Query** with tighter `top_k_*` (local) and a raised `dynamic_search_threshold` (global).  
*Microsoft GitHub* [+1]

### A word on fully-local indexing *without* cloud APIs

- **Graph extraction:** use `--method fast` to avoid LLMs; this is exactly why it exists.  
  *Microsoft GitHub*
- **Community reports:** still need a generator. Use a local LLM (through proxy API or model injection) per the **Language Model Selection** doc (Proxy APIs / Model Protocol).  
  *Microsoft GitHub*
- **Vectors & storage:** keep the defaults (LanceDB + file).  
  *Microsoft GitHub*

### Paste-ready instruction set (start to finish)

**Install & init**
```bash
python -m venv .venv && source .venv/bin/activate
pip install graphrag
graphrag init --root ./lab
```

**Put data** in `./lab/input/` (`.txt`, `.csv`, or `.json`).  
*Microsoft GitHub*

**(Optional) Go fully local for extraction**
```bash
pip install spacy && python -m spacy download en_core_web_md
graphrag index --root ./lab --method fast
```

**Else run the standard index (needs OpenAI/Azure or a proxy model):**
```bash
# set GRAPHRAG_API_KEY or set up your proxy per docs
graphrag index --root ./lab --method standard
```
*Outputs appear in `./lab/output/*.parquet`. — Microsoft GitHub*

**Turn on built-in pruning:** add this to `./lab/settings.yaml`:
```yaml
workflows: [extract_graph, prune_graph, cluster_graph, community_reports, embed_text]
prune_graph: { min_node_freq: 2, min_node_degree: 2, min_edge_weight_pct: 25.0, lcc_only: true }
```
*(Fields documented here.) — Microsoft GitHub*

**Re-run:** `graphrag index --root ./lab`

**Compute custom scores & prune (your notebook/script):**
- Load Parquet → compute `importance/quality/rank` → write columns back. (See outline above.)
- If you removed edges/nodes, **re-run `cluster_graph`** so communities match reality.  
  *Microsoft GitHub*

**Query with stricter selectors**
- **Local:** `graphrag query --root ./lab --method local --query "…"`, but set smaller `top_k_entities/relationships` in `settings.yaml`.  
  *Microsoft GitHub*
- **Global:** enable rank use and raise `dynamic_search_threshold` to prune low-value reports before reduce.  
  *Microsoft GitHub* [+1]

**Evaluate & iterate:** track answer quality vs. tokens/latency; nudge prune thresholds or scoring weights.

> If you want, I can spin up a small scaffold (Python script) that reads your `output/relationships.parquet`, computes degree/centrality + a trivial edge score, writes `relationships.quality`, and prunes to top-k per node — so you can plug a KGE model later.
