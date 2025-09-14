# Task overview
1. Connect to local LLM - deploying quantized model locally? / HPC 
2. Workflow, work out a diagram etc

### initializing the env
This project uses pixi for env management:
https://pixi.sh/dev/

To initialize the pixi environment and activate the shell:

```bash
pixi shell
```

Or if you need to specify the manifest path:

```bash
pixi shell --manifest-path pixi.toml
```
This will activate the pixi environment with all dependencies configured in `pixi.toml`.
