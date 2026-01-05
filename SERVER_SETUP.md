# Server Setup for Finetuning

## Files to Copy to Server

Copy these files/directories to your server:

```bash
# On your local machine, create a transfer package
tar -czf finetuning_package.tar.gz \
  output/ \
  finetune_parakeet.sh \
  update_manifest_paths.py \
  FINETUNING_INSTRUCTIONS.md

# Copy to server (adjust IP and path as needed)
scp finetuning_package.tar.gz user@192.168.1.203:/path/to/finetuning/
```

## Server Setup Steps

### 1. Extract files on server

```bash
# SSH to your server
ssh user@192.168.1.203

# Navigate to your working directory
cd /path/to/finetuning

# Extract
tar -xzf finetuning_package.tar.gz

# Verify extraction
ls -R output/
```

### 2. Update manifest paths

Since the file paths in the manifests reference your local machine, you need to update them:

```bash
# Update all manifests to point to the correct location
python update_manifest_paths.py \
  --old-prefix "/Users/yaoyu/projects/speech_finetune_data_generation" \
  --new-prefix "/path/to/finetuning" \
  --manifest-dir output/manifests

# Example for /home/user/finetuning:
python update_manifest_paths.py \
  --old-prefix "/Users/yaoyu/projects/speech_finetune_data_generation" \
  --new-prefix "/home/user/finetuning" \
  --manifest-dir output/manifests
```

This will:
- Create backups (.jsonl.bak)
- Update all paths in train_manifest.jsonl, val_manifest.jsonl, and master_manifest.jsonl

### 3. Install dependencies

```bash
# Create conda environment
conda create -n nemo python=3.10
conda activate nemo

# Install NeMo and dependencies
pip install nemo_toolkit[asr]
pip install pytorch-lightning

# OR if you have GPU:
pip install nemo_toolkit[asr] --extra-index-url https://pypi.nvidia.com
```

### 4. Run finetuning

**Basic usage** (uses defaults):
```bash
bash finetune_parakeet.sh
```

**Custom configuration**:
```bash
# Adjust hyperparameters via environment variables
BATCH_SIZE=4 \
MAX_EPOCHS=100 \
LEARNING_RATE=5e-5 \
BASE_MODEL="nvidia/parakeet-tdt-0.6b-v3" \
OUTPUT_DIR="./my_finetuned_model" \
bash finetune_parakeet.sh
```

### Available Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MANIFEST_DIR` | `./output/manifests` | Directory containing manifests |
| `OUTPUT_DIR` | `./finetuned_model` | Where to save the model |
| `BASE_MODEL` | `nvidia/parakeet-tdt-0.6b-v3` | Pretrained model to finetune |
| `BATCH_SIZE` | `8` | Training batch size |
| `LEARNING_RATE` | `1e-4` | Learning rate |
| `MAX_EPOCHS` | `50` | Maximum training epochs |
| `NUM_WORKERS` | `4` | Data loader workers |

### 5. Monitor training

The script will show:
- Training/validation loss
- Word Error Rate (WER) on validation set
- Checkpoint saves

Training outputs will be in:
- `./finetuned_model/checkpoints/` - Best checkpoints
- `./finetuned_model/final_model.nemo` - Final model
- `./finetuned_model/lightning_logs/` - TensorBoard logs

### 6. View training progress (optional)

```bash
# In another terminal
conda activate nemo
pip install tensorboard
tensorboard --logdir=./finetuned_model/lightning_logs
```

Then open http://your-server-ip:6006 in a browser.

## Quick Start (Copy-Paste)

```bash
# On server after extracting files:

# 1. Update paths (replace with your actual path!)
python update_manifest_paths.py \
  --old-prefix "/Users/yaoyu/projects/speech_finetune_data_generation" \
  --new-prefix "$(pwd)" \
  --manifest-dir output/manifests

# 2. Setup environment
conda create -n nemo python=3.10 -y
conda activate nemo
pip install nemo_toolkit[asr] pytorch-lightning

# 3. Start training
bash finetune_parakeet.sh
```

## Expected Training Time

With the dataset (179 samples):
- **GPU (A100/V100)**: ~30-60 minutes for 50 epochs
- **GPU (RTX 3090)**: ~1-2 hours for 50 epochs
- **CPU**: Not recommended (would take many hours)

## After Training

Your finetuned model will be at:
```
./finetuned_model/final_model.nemo
```

You can use it for inference or deploy it to your transcription server!

## Troubleshooting

**Issue**: "File not found" errors during training
- **Fix**: Make sure you ran `update_manifest_paths.py` to update the paths

**Issue**: Out of memory
- **Fix**: Reduce `BATCH_SIZE` (try 4 or 2)

**Issue**: NeMo installation fails
- **Fix**: Check CUDA version compatibility, or try CPU-only install:
  ```bash
  pip install nemo_toolkit[asr] --extra-index-url https://download.pytorch.org/whl/cpu
  ```

**Issue**: Model downloads very slowly
- **Fix**: The base model (~600MB) downloads from HuggingFace. Use a good internet connection or download manually first.
