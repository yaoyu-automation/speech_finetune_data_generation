# Speech Finetuning Dataset - Instructions

## Dataset Summary

- **Total recordings**: 108 audio files (.wav format, 16kHz)
- **Training samples**: 97 utterances
- **Validation samples**: 11 utterances
- **Package size**: ~10 MB compressed
- **Format**: NeMo-compatible JSONL manifests

## Package Contents

After extracting `speech_finetuning_dataset.tar.gz`, you'll have:

```
output/
├── audio/
│   ├── utt_0001_v01.wav
│   ├── utt_0001_v02.wav
│   └── ... (108 total .wav files)
└── manifests/
    ├── master_manifest.jsonl  (all takes with metadata)
    ├── train_manifest.jsonl   (97 training samples)
    └── val_manifest.jsonl     (11 validation samples)
```

## Extracting the Dataset

```bash
# Extract the tar file
tar -xzf speech_finetuning_dataset.tar.gz

# Verify extraction
ls -R output/
```

---

## Finetuning Instructions

### Option 1: NVIDIA NeMo (Recommended for ASR models)

NeMo is a toolkit for building speech AI models. Use this for finetuning models like Parakeet, Conformer, or Citrinet.

#### 1. Install NeMo

```bash
# Create a conda environment
conda create -n nemo python=3.10
conda activate nemo

# Install NeMo ASR
pip install nemo_toolkit[asr]
```

#### 2. Update Manifest Paths (if needed)

If you're running on a different machine, update the audio file paths in the manifests:

```bash
# Replace the old path with your new path
OLD_PATH="/Users/yaoyu/projects/speech_finetune_data_generation"
NEW_PATH="/path/to/your/project"

sed -i "s|$OLD_PATH|$NEW_PATH|g" output/manifests/train_manifest.jsonl
sed -i "s|$OLD_PATH|$NEW_PATH|g" output/manifests/val_manifest.jsonl
```

Or use this Python script:

```python
import json

def update_manifest_paths(manifest_path, old_prefix, new_prefix):
    with open(manifest_path, 'r') as f:
        lines = f.readlines()

    with open(manifest_path, 'w') as f:
        for line in lines:
            data = json.loads(line)
            data['audio_filepath'] = data['audio_filepath'].replace(old_prefix, new_prefix)
            f.write(json.dumps(data) + '\n')

# Update paths
update_manifest_paths('output/manifests/train_manifest.jsonl',
                      '/Users/yaoyu/projects/speech_finetune_data_generation',
                      '/your/new/path')
update_manifest_paths('output/manifests/val_manifest.jsonl',
                      '/Users/yaoyu/projects/speech_finetune_data_generation',
                      '/your/new/path')
```

#### 3. Create NeMo Training Config

Create a YAML config file `finetune_config.yaml`:

```yaml
name: "Speech-Finetuning"

model:
  sample_rate: 16000

  train_ds:
    manifest_filepath: output/manifests/train_manifest.jsonl
    sample_rate: 16000
    batch_size: 4
    shuffle: true
    num_workers: 4

  validation_ds:
    manifest_filepath: output/manifests/val_manifest.jsonl
    sample_rate: 16000
    batch_size: 4
    shuffle: false
    num_workers: 4

trainer:
  devices: 1
  max_epochs: 50
  accelerator: gpu  # or 'cpu' if no GPU
  strategy: auto
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0
  precision: 16  # mixed precision training

  check_val_every_n_epoch: 1
  enable_checkpointing: true
  logger: true
  log_every_n_steps: 10

exp_manager:
  exp_dir: ./experiments
  name: speech_finetune
  create_checkpoint_callback: true
  checkpoint_callback_params:
    monitor: "val_wer"
    mode: "min"
    save_top_k: 3
```

#### 4. Run Finetuning

```bash
# For Parakeet CTC models
python -m nemo.collections.asr.scripts.train_asr \
  --config-path=. \
  --config-name=finetune_config \
  model.pretrained_model=nvidia/parakeet-ctc-1.1b \
  +model.tokenizer.dir=null \
  +model.tokenizer.type=wpe

# Or use the training script from this repo if available
bash finetune_parakeet.sh
```

#### 5. Monitor Training

```bash
# Install tensorboard if not already installed
pip install tensorboard

# View training metrics
tensorboard --logdir=./experiments/speech_finetune/
```

---

### Option 2: OpenAI Whisper Finetuning

For finetuning Whisper models on your custom dataset.

#### 1. Convert Manifests to Whisper Format

Create `convert_to_whisper.py`:

```python
import json
import pandas as pd
from datasets import Dataset, DatasetDict, Audio

# Load your manifests
def load_manifest(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_manifest('output/manifests/train_manifest.jsonl')
val_data = load_manifest('output/manifests/val_manifest.jsonl')

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({
    'audio': [d['audio_filepath'] for d in train_data],
    'text': [d['text'] for d in train_data]
})

val_dataset = Dataset.from_dict({
    'audio': [d['audio_filepath'] for d in val_data],
    'text': [d['text'] for d in val_data]
})

# Cast audio column
train_dataset = train_dataset.cast_column('audio', Audio(sampling_rate=16000))
val_dataset = val_dataset.cast_column('audio', Audio(sampling_rate=16000))

# Create dataset dict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

# Save to disk
dataset.save_to_disk('whisper_dataset')
print(f"Dataset saved with {len(train_dataset)} training and {len(val_dataset)} validation samples")
```

#### 2. Install Dependencies

```bash
pip install transformers datasets accelerate evaluate jiwer
```

#### 3. Finetune Whisper

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_from_disk
import torch

# Load dataset
dataset = load_from_disk('whisper_dataset')

# Load model and processor
model_name = "openai/whisper-small"  # or tiny, base, medium, large
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Prepare dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=500,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

# Train
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
)

trainer.train()
model.save_pretrained("./whisper-finetuned-final")
processor.save_pretrained("./whisper-finetuned-final")
```

---

### Option 3: Custom PyTorch Training

If you have your own training pipeline:

```python
import json
import soundfile as sf

def load_dataset(manifest_path):
    """Load audio and text from manifest"""
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            audio, sr = sf.read(data['audio_filepath'])
            samples.append({
                'audio': audio,
                'sample_rate': sr,
                'text': data['text'],
                'duration': data['duration']
            })
    return samples

# Load your data
train_data = load_dataset('output/manifests/train_manifest.jsonl')
val_data = load_dataset('output/manifests/val_manifest.jsonl')

# Use in your training loop
for sample in train_data:
    audio = sample['audio']  # numpy array
    text = sample['text']
    # ... your training code
```

---

## Tips for Successful Finetuning

1. **Learning Rate**: Start with 1e-5 or 1e-4 for finetuning pretrained models
2. **Batch Size**: Adjust based on GPU memory (4-16 is typical)
3. **Epochs**: Start with 20-50 epochs, monitor validation loss
4. **Augmentation**: Consider adding noise/speed perturbation for better generalization
5. **Validation**: Monitor WER (Word Error Rate) on validation set
6. **Checkpointing**: Save checkpoints regularly to resume if training fails

## Dataset Characteristics

This dataset appears to contain medical terminology (dysgeusia, radiation therapy, etc.). Make sure your:
- Tokenizer can handle medical terms
- Evaluation metrics account for domain-specific vocabulary
- Pretrained model has some medical domain exposure (or plan for longer training)

## Troubleshooting

**Issue**: "File not found" errors
- **Solution**: Update manifest paths (see instructions above)

**Issue**: Out of memory during training
- **Solution**: Reduce batch size or enable gradient accumulation

**Issue**: Poor validation performance
- **Solution**: Increase training data, adjust learning rate, or try data augmentation

**Issue**: Model overfits quickly
- **Solution**: Add regularization, reduce model size, or collect more diverse data

---

## Next Steps

After training:
1. Evaluate on test set (if you have one)
2. Export model for inference
3. Test on real-world examples
4. Consider iterative data collection for weak areas

For questions or issues, refer to:
- NeMo docs: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/
- Whisper finetuning: https://huggingface.co/blog/fine-tune-whisper
