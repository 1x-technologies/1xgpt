#!/usr/bin/bash

python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE python -m pip install flash-attn==2.5.8 --no-build-isolation

# Download datasets to data/train_v1.0, data/val_v1.0
huggingface-cli download 1x-technologies/worldmodel --repo-type dataset --local-dir data
