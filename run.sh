#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB_BASE_URL=http://localhost:8080
export WANDB_API_KEY=local-cbe13d5c8234115c580513d3f98f3ad088b07fa3
export DWAVE_API_TOKEN=DEV-82d0ad84a8a39543d7dd091bc2476e207ae6b2a2

python scripts/run.py