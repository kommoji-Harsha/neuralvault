# NeuralVault Offline Operation & Air-Gapped Deployment Guide

## Overview

NeuralVault is built **offline-first**. All retrieval, chunking, keyword search, vector similarity search, and reranking operations function with no internet connection and no cloud API keys.

---

## 1. NEURALVAULT_OFFLINE Guard

When the environment variable `NEURALVAULT_OFFLINE=1` is set:
- Any attempted network request raises an explicit `OfflineError`.
- No models, packages, or remote files are downloaded implicitly or at import time.

```bash
export NEURALVAULT_OFFLINE=1
```

---

## 2. Model Caching & Pre-downloading

All FastEmbed and local model files are stored locally under `NEURALVAULT_HOME/models` (default: `~/.neuralvault/models`).

### Pre-downloading Models Online
Before deploying to an air-gapped environment, download models while connected to the internet:

```bash
neuralvault models download --model BAAI/bge-small-en-v1.5
```

---

## 3. Air-Gapped Deployment Workflow

1. **Pre-download Models**: Run `neuralvault models download` on a connected machine.
2. **Transfer Cache**: Copy the `~/.neuralvault` directory to the target air-gapped machine.
3. **Enable Offline Mode**: Set `NEURALVAULT_OFFLINE=1` on the target machine.
4. **Execute Operations**: Run `neuralvault ingest`, `neuralvault search`, `neuralvault ask`, `neuralvault eval`, or `neuralvault mcp`.
