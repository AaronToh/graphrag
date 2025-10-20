#!/usr/bin/env python3
"""
GraphRAG Built-in Visualization Tools

This script demonstrates how to use GraphRAG's built-in visualization capabilities:
1. GraphML snapshots for Gephi
2. yfiles-jupyter-graphs for interactive visualization
3. Graph embeddings and UMAP for node positioning
"""

import pandas as pd
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_graphml_snapshots(workspace_dir: str = "workspace"):
    """Check if GraphML snapshots were generated during indexing."""
    workspace_path = Path(workspace_dir)
    output_path = workspace_path / "output"
    
    # Look for GraphML files
    graphml_files = list(output_path.glob("*.graphml"))
    
    if graphml_files:
        logger.info(f"Found {len(graphml_files)} GraphML files:")
        for file in graphml_files:
            logger.info(f"  - {file}")
            logger.info(f"    Size: {file.stat().st_size / 1024:.1f} KB")
        
        return graphml_files
    else:
        logger.warning("No GraphML files found. Make sure 'snapshots.graphml: true' in settings.yaml and re-run indexing.")
        return []

def check_embeddings(workspace_dir: str = "workspace"):
    """Check if graph embeddings and UMAP coordinates were generated."""
    workspace_path = Path(workspace_dir)
    output_path = workspace_path / "output"
    
    # Check entities.parquet for embedding columns
    entities_file = output_path / "entities.parquet"
    if entities_file.exists():
        entities_df = pd.read_parquet(entities_file)
        
        logger.info("Entities DataFrame columns:")
        for col in entities_df.columns:
            logger.info(f"  - {col}")
        
        # Check for embedding-related columns
        embedding_cols = [col for col in entities_df.columns if 'embed' in col.lower() or 'umap' in col.lower() or 'x' in col.lower() or 'y' in col.lower()]
        
        if embedding_cols:
            logger.info(f"Found embedding/position columns: {embedding_cols}")
            return True
        else:
            logger.warning("No embedding/position columns found. Make sure 'embed_graph.enabled: true' and 'umap.enabled: true' in settings.yaml and re-run indexing.")
            return False
    else:
        logger.error("entities.parquet not found!")
        return False

def setup_yfiles_visualization():
    """Set up yfiles-jupyter-graphs for interactive visualization."""
    try:
        # Check if yfiles-jupyter-graphs is available
        import yfiles_jupyter_graphs
        logger.info("yfiles-jupyter-graphs is available!")
        
        # Add it to pixi dependencies if not already there
        pixi_file = Path("pixi.toml")
        if pixi_file.exists():
            content = pixi_file.read_text()
            if "yfiles-jupyter-graphs" not in content:
                logger.info("Adding yfiles-jupyter-graphs to pixi.toml...")
                # Add to pypi-dependencies section
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() == 'networkx = ">=2.8"':
                        lines.insert(i + 1, 'yfiles-jupyter-graphs = ">=1.0.0"')
                        break
                
                pixi_file.write_text('\n'.join(lines))
                logger.info("Added yfiles-jupyter-graphs to pixi.toml. Run 'pixi install' to install.")
        
        return True
        
    except ImportError:
        logger.warning("yfiles-jupyter-graphs not installed. Install with: pip install yfiles-jupyter-graphs")
        return False

def create_yfiles_notebook():
    """Create a Jupyter notebook demonstrating yfiles visualization."""
    notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# GraphRAG Interactive Visualization with yfiles-jupyter-graphs\\n",
    "\\n",
    "This notebook demonstrates how to use GraphRAG's built-in visualization capabilities."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install yfiles-jupyter-graphs if not already installed\\n",
    "%pip install yfiles-jupyter-graphs --quiet\\n",
    "\\n",
    "import pandas as pd\\n",
    "from yfiles_jupyter_graphs import GraphWidget\\n",
    "from IPython.display import display\\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load GraphRAG data\\n",
    "OUTPUT_DIR = \\"workspace/output\\"\\n",
    "\\n",
    "entities_df = pd.read_parquet(f\\"{OUTPUT_DIR}/entities.parquet\\")\\n",
    "relationships_df = pd.read_parquet(f\\"{OUTPUT_DIR}/relationships.parquet\\")\\n",
    "communities_df = pd.read_parquet(f\\"{OUTPUT_DIR}/communities.parquet\\")\\n",
    "\\n",
    "print(f\\"Loaded {len(entities_df)} entities, {len(relationships_df)} relationships, {len(communities_df)} communities\\")\\n",
    "print(f\\"Entity columns: {list(entities_df.columns)}\\")\\n",
    "print(f\\"Relationship columns: {list(relationships_df.columns)}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert entities to yfiles format\\n",
    "def convert_entities_to_dicts(df, max_nodes=500):\\n",
    "    \\"\\"\\"Convert the entities dataframe to a list of dicts for yfiles-jupyter-graphs.\\"\\"\\"\\n",
    "    nodes_dict = {}\\n",
    "    count = 0\\n",
    "    \\n",
    "    for _, row in df.iterrows():\\n",
    "        if count >= max_nodes:\\n",
    "            break\\n",
    "            \\n",
    "        # Use 'id' or 'title' as node identifier\\n",
    "        node_id = row.get('id') or row.get('title') or row.get('name')\\n",
    "        \\n",
    "        if node_id and node_id not in nodes_dict:\\n",
    "            nodes_dict[node_id] = {\\n",
    "                \\"id\\": node_id,\\n",
    "                \\"properties\\": row.to_dict(),\\n",
    "            }\\n",
    "            count += 1\\n",
    "    \\n",
    "    return list(nodes_dict.values())\\n",
    "\\n",
    "# Convert relationships to yfiles format\\n",
    "def convert_relationships_to_dicts(df, node_ids, max_edges=1000):\\n",
    "    \\"\\"\\"Convert the relationships dataframe to a list of dicts for yfiles-jupyter-graphs.\\"\\"\\"\\n",
    "    relationships = []\\n",
    "    count = 0\\n",
    "    \\n",
    "    for _, row in df.iterrows():\\n",
    "        if count >= max_edges:\\n",
    "            break\\n",
    "            \\n",
    "        source = row.get('source')\\n",
    "        target = row.get('target')\\n",
    "        \\n",
    "        # Only include edges where both nodes exist\\n",
    "        if source in node_ids and target in node_ids:\\n",
    "            relationships.append({\\n",
    "                \\"start\\": source,\\n",
    "                \\"end\\": target,\\n",
    "                \\"properties\\": row.to_dict(),\\n",
    "            })\\n",
    "            count += 1\\n",
    "    \\n",
    "    return relationships"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create interactive graph visualization\\n",
    "print(\\"Creating interactive graph visualization...\\")\\n",
    "\\n",
    "# Convert data\\n",
    "nodes = convert_entities_to_dicts(entities_df, max_nodes=200)\\n",
    "node_ids = {node[\\"id\\"] for node in nodes}\\n",
    "edges = convert_relationships_to_dicts(relationships_df, node_ids, max_edges=500)\\n",
    "\\n",
    "print(f\\"Visualizing {len(nodes)} nodes and {len(edges)} edges\\")\\n",
    "\\n",
    "# Create widget\\n",
    "w = GraphWidget()\\n",
    "w.nodes = nodes\\n",
    "w.edges = edges\\n",
    "w.directed = True\\n",
    "\\n",
    "# Configure appearance\\n",
    "w.node_label_mapping = lambda node: node[\\"properties\\"].get(\\"title\\") or node[\\"properties\\"].get(\\"name\\") or node[\\"id\\"]\\n",
    "\\n",
    "# Color by community if available\\n",
    "def community_to_color(community):\\n",
    "    colors = [\\"crimson\\", \\"darkorange\\", \\"indigo\\", \\"cornflowerblue\\", \\"cyan\\", \\"teal\\", \\"green\\", \\"purple\\", \\"pink\\", \\"brown\\"]\\n",
    "    return colors[int(community) % len(colors)] if community is not None else \\"lightgray\\"\\n",
    "\\n",
    "if 'community' in entities_df.columns:\\n",
    "    w.node_color_mapping = lambda node: community_to_color(node[\\"properties\\"].get(\\"community\\"))\\n",
    "\\n",
    "# Size by degree if available\\n",
    "if 'degree' in entities_df.columns:\\n",
    "    w.node_scale_factor_mapping = lambda node: 0.5 + (node[\\"properties\\"].get(\\"degree\\", 1) * 1.5 / 20)\\n",
    "\\n",
    "# Edge thickness by weight\\n",
    "w.edge_thickness_factor_mapping = lambda edge: edge[\\"properties\\"].get(\\"weight\\", 1)\\n",
    "\\n",
    "# Apply layout\\n",
    "w.circular_layout()  # Use circular layout for smaller graphs\\n",
    "\\n",
    "# Display the widget\\n",
    "display(w)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## GraphML Export for Gephi\\n",
    "\\n",
    "If GraphML snapshots are enabled in your settings.yaml, you can find .graphml files in the output directory.\\n",
    "These can be opened directly in Gephi for advanced visualization and analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for GraphML files\\n",
    "import glob\\n",
    "\\n",
    "graphml_files = glob.glob(f\\"{OUTPUT_DIR}/*.graphml\\")\\n",
    "if graphml_files:\\n",
    "    print(f\\"Found GraphML files: {graphml_files}\\")\\n",
    "    print(\\"You can open these in Gephi for advanced visualization!\\")\\n",
    "else:\\n",
    "    print(\\"No GraphML files found. Enable 'snapshots.graphml: true' in settings.yaml and re-run indexing.\\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    notebook_path = Path("graphrag_visualization.ipynb")
    notebook_path.write_text(notebook_content)
    logger.info(f"Created Jupyter notebook: {notebook_path}")
    return notebook_path

def main():
    """Main function to demonstrate GraphRAG's built-in visualization tools."""
    print("="*60)
    print("GRAPHRAG BUILT-IN VISUALIZATION TOOLS")
    print("="*60)
    
    # Check current status
    logger.info("Checking GraphRAG visualization artifacts...")
    
    # 1. Check GraphML snapshots
    graphml_files = check_graphml_snapshots()
    
    # 2. Check embeddings
    has_embeddings = check_embeddings()
    
    # 3. Setup yfiles visualization
    has_yfiles = setup_yfiles_visualization()
    
    # 4. Create demonstration notebook
    notebook_path = create_yfiles_notebook()
    
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    if graphml_files:
        print("✅ GraphML snapshots found!")
        print("   → Open .graphml files in Gephi for professional visualization")
        for file in graphml_files:
            print(f"     - {file}")
    else:
        print("❌ No GraphML snapshots found")
        print("   → Enable 'snapshots.graphml: true' in settings.yaml")
        print("   → Re-run indexing: pixi run python -m graphrag index --root workspace")
    
    if has_embeddings:
        print("✅ Graph embeddings found!")
        print("   → Node positions available for visualization")
    else:
        print("❌ No graph embeddings found")
        print("   → Enable 'embed_graph.enabled: true' and 'umap.enabled: true' in settings.yaml")
        print("   → Re-run indexing: pixi run python -m graphrag index --root workspace")
    
    if has_yfiles:
        print("✅ yfiles-jupyter-graphs available!")
        print(f"   → Use the notebook: {notebook_path}")
        print("   → Run: jupyter notebook graphrag_visualization.ipynb")
    else:
        print("❌ yfiles-jupyter-graphs not available")
        print("   → Install: pixi add yfiles-jupyter-graphs")
        print("   → Or: pip install yfiles-jupyter-graphs")
    
    print("\n📋 NEXT STEPS:")
    print("1. Update settings.yaml (already done)")
    print("2. Re-run indexing to generate visualization artifacts:")
    print("   pixi run python -m graphrag index --root workspace")
    print("3. Use the generated Jupyter notebook for interactive visualization")
    print("4. Open .graphml files in Gephi for advanced analysis")

if __name__ == "__main__":
    main()
