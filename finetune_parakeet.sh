#!/bin/bash
set -e

# Configuration
MANIFEST_DIR="${MANIFEST_DIR:-./output/manifests}"
TRAIN_MANIFEST="${MANIFEST_DIR}/train_manifest.jsonl"
VAL_MANIFEST="${MANIFEST_DIR}/val_manifest.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-./finetuned_model}"
BASE_MODEL="${BASE_MODEL:-nvidia/parakeet-tdt-0.6b-v3}"

# Hyperparameters
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Check if manifests exist
if [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "Error: Train manifest not found at $TRAIN_MANIFEST"
    echo "Please run finetune_script.py first to record your dataset."
    exit 1
fi

if [ ! -f "$VAL_MANIFEST" ]; then
    echo "Error: Validation manifest not found at $VAL_MANIFEST"
    echo "Please run finetune_script.py first to record your dataset."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "==================================="
echo "NeMo Parakeet Fine-tuning Script"
echo "==================================="
echo "Train manifest: $TRAIN_MANIFEST"
echo "Val manifest:   $VAL_MANIFEST"
echo "Output dir:     $OUTPUT_DIR"
echo "Base model:     $BASE_MODEL"
echo "Batch size:     $BATCH_SIZE"
echo "Learning rate:  $LEARNING_RATE"
echo "Max epochs:     $MAX_EPOCHS"
echo "==================================="
echo ""

# Fine-tuning using NeMo
python - <<PYTHON_SCRIPT
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from nemo.collections.asr.models import EncDecCTCModelBPE
from omegaconf import OmegaConf, open_dict

# Load pretrained model
print(f"Loading pretrained model: ${BASE_MODEL}")
model = EncDecCTCModelBPE.from_pretrained("${BASE_MODEL}")

# Update dataset paths
with open_dict(model.cfg):
    # Training dataset
    model.cfg.train_ds.manifest_filepath = "${TRAIN_MANIFEST}"
    model.cfg.train_ds.batch_size = ${BATCH_SIZE}
    model.cfg.train_ds.num_workers = ${NUM_WORKERS}
    model.cfg.train_ds.pin_memory = True
    model.cfg.train_ds.use_start_end_token = False

    # Validation dataset
    model.cfg.validation_ds.manifest_filepath = "${VAL_MANIFEST}"
    model.cfg.validation_ds.batch_size = ${BATCH_SIZE}
    model.cfg.validation_ds.num_workers = ${NUM_WORKERS}
    model.cfg.validation_ds.pin_memory = True
    model.cfg.validation_ds.use_start_end_token = False

    # Optimizer settings
    model.cfg.optim.name = "adamw"
    model.cfg.optim.lr = ${LEARNING_RATE}
    model.cfg.optim.betas = [0.9, 0.999]
    model.cfg.optim.weight_decay = 0.0001

    # Scheduler settings
    model.cfg.optim.sched.name = "CosineAnnealing"
    model.cfg.optim.sched.warmup_steps = 100
    model.cfg.optim.sched.min_lr = 1e-6

# Setup data loaders
model.setup_training_data(model.cfg.train_ds)
model.setup_validation_data(model.cfg.validation_ds)

# Setup trainer
checkpoint_callback = ModelCheckpoint(
    dirpath="${OUTPUT_DIR}/checkpoints",
    filename="parakeet-{epoch:02d}-{val_loss:.2f}",
    monitor="val_wer",
    mode="min",
    save_top_k=3,
    save_last=True,
)

trainer = pl.Trainer(
    max_epochs=${MAX_EPOCHS},
    accelerator="auto",
    devices=1,
    callbacks=[checkpoint_callback],
    default_root_dir="${OUTPUT_DIR}",
    log_every_n_steps=10,
    val_check_interval=1.0,
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,
)

# Start training
print("Starting fine-tuning...")
trainer.fit(model)

# Save final model
print(f"Saving final model to ${OUTPUT_DIR}/final_model.nemo")
model.save_to("${OUTPUT_DIR}/final_model.nemo")

print("Fine-tuning complete!")
print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
print(f"Final model: ${OUTPUT_DIR}/final_model.nemo")

PYTHON_SCRIPT

echo ""
echo "==================================="
echo "Fine-tuning completed successfully!"
echo "Model saved to: $OUTPUT_DIR/final_model.nemo"
echo "==================================="
