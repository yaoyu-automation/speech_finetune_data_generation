#!/bin/bash
set -e

# Configuration for NeMo/Parakeet compatibility
SENTENCES_FILE="${SENTENCES_FILE:-./sentences.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"
CHANNELS="${CHANNELS:-1}"
TRIM_SILENCE="${TRIM_SILENCE:-true}"
SILENCE_THRESHOLD="${SILENCE_THRESHOLD:--40.0}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-42}"

# Check if sentences file exists
if [ ! -f "$SENTENCES_FILE" ]; then
    echo "Error: Sentences file not found at $SENTENCES_FILE"
    exit 1
fi

echo "==================================="
echo "NeMo Dataset Recording Script"
echo "==================================="
echo "Sentences file: $SENTENCES_FILE"
echo "Output dir:     $OUTPUT_DIR"
echo "Sample rate:    ${SAMPLE_RATE} Hz"
echo "Channels:       $CHANNELS (mono)"
echo "Trim silence:   $TRIM_SILENCE"
echo "Val fraction:   $VAL_FRACTION"
echo "==================================="
echo ""

# Build command
CMD="python finetune_script.py \
  --sentences \"$SENTENCES_FILE\" \
  --output-dir \"$OUTPUT_DIR\" \
  --samplerate $SAMPLE_RATE \
  --channels $CHANNELS \
  --val-fraction $VAL_FRACTION \
  --seed $SEED"

# Add optional flags
if [ "$TRIM_SILENCE" = "true" ]; then
    CMD="$CMD --trim-silence --silence-threshold-db $SILENCE_THRESHOLD"
fi

echo "Running: $CMD"
echo ""

# Execute the recording script
eval $CMD
