# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a dictation dataset recorder that creates multi-take recordings for speech fine-tuning. The script records multiple versions of each sentence at different speeds/emphases and generates NeMo-compatible JSONL manifests for training and validation.

## Core Architecture

### Single-File Design
The entire application is contained in `finetune_script.py` (~565 lines), organized into logical sections:

1. **Data Structures** (lines 49-64)
   - `TakeKey`: Frozen dataclass uniquely identifying a recording (sent_idx + version)
   - `Take`: Recording metadata including text, filepath, duration, sample_rate

2. **I/O Utilities** (lines 66-102)
   - JSONL reading/writing with atomic file operations (temp file + replace pattern)
   - Sentence loading from text files
   - Safe file writes to prevent corruption

3. **Audio Processing** (lines 104-228)
   - Space-bar triggered recording using threading for key listening
   - Silence trimming (optional, based on dB threshold)
   - Device selection (by index or name substring)
   - Playback support

4. **Manifest Management** (lines 230-322)
   - Master manifest maintains all takes with metadata
   - Train/val split generation with configurable fraction and seed
   - Resume logic based on existing takes (finds last sentence with recordings)

5. **Interactive UI** (lines 324-376)
   - Menu-driven workflow: play, delete, record another, proceed/jump
   - Navigation: space for next, 'b' for back, 'j' for jump to specific sentence

6. **Main Loop** (lines 378-563)
   - Stateful session that persists progress after each recording
   - Graceful resume from `master_manifest.jsonl`

### Key Design Patterns

- **Atomic persistence**: Every recording immediately writes to master manifest + regenerates train/val splits to ensure no data loss
- **Resume-friendly**: The master manifest is the single source of truth; resume index is computed from existing takes
- **Multi-take support**: Each sentence can have unlimited versions (identified by version number)
- **NeMo compatibility**: Train/val manifests follow NeMo format: `{"audio_filepath": str, "duration": float, "text": str}`

## Running the Script

### Dependencies
```bash
pip install sounddevice soundfile numpy readchar
```

### List Available Audio Devices
```bash
python finetune_script.py --list-devices
```

### Basic Usage
```bash
python finetune_script.py \
  --sentences sentences.txt \
  --output-dir ./output \
  --samplerate 16000 \
  --channels 1
```

### With Options
```bash
python finetune_script.py \
  --sentences sentences.txt \
  --output-dir ./output \
  --device "MacBook Pro Microphone" \
  --trim-silence \
  --silence-threshold-db -40.0 \
  --gain 1.2 \
  --max-seconds 15.0 \
  --val-fraction 0.15 \
  --seed 42
```

### Key Arguments
- `--sentences`: Text file with one sentence per line (lines starting with `#` are ignored)
- `--output-dir`: Creates `audio/` and `manifests/` subdirectories
- `--device`: Audio input device (index or name substring); defaults to system default
- `--trim-silence`: Enable silence trimming at start/end of recordings
- `--gain`: Amplitude multiplier (e.g., 1.2 for 20% boost, clipped to [-1, 1])
- `--val-fraction`: Proportion of takes for validation set (default 0.1)
- `--seed`: Random seed for train/val split

## Output Structure

```
output-dir/
├── audio/
│   ├── utt_0001_v01.wav
│   ├── utt_0001_v02.wav
│   └── ...
└── manifests/
    ├── master_manifest.jsonl  (authoritative, includes sent_idx + version)
    ├── train_manifest.jsonl   (NeMo format)
    └── val_manifest.jsonl     (NeMo format)
```

## Important Implementation Notes

### Recording Flow
1. Script shows sentence and waits for action menu
2. Press `[3]` to record: `[space]` to start, `[space]` to stop
3. After recording: can play `[1]`, delete `[2]`, record another `[3]`, or proceed `[4]`
4. Proceed options: `[space]` next, `[b]` previous, `[j]` jump to sentence number

### Key Functions

- `record_with_space()` (lines 152-217): Space-triggered recording using threading for key listener
- `trim_silence()` (lines 138-150): dB-based silence removal with padding
- `current_sentence_resume_index()` (lines 272-290): Computes resume position from existing takes
- `write_train_val_manifests()` (lines 292-321): Regenerates train/val from master takes

### Manifest Format

**Master manifest** (extended format):
```json
{"sent_idx": 1, "version": 1, "text": "...", "audio_filepath": "...", "duration": 2.5, "sample_rate": 16000}
```

**Train/val manifests** (NeMo format):
```json
{"audio_filepath": "...", "duration": 2.5, "text": "..."}
```

## Testing

### Running Tests
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests with coverage
pytest

# Run specific test class
pytest test_finetune_script.py::TestManifestManagement -v

# Run tests without coverage report
pytest --no-cov

# Run tests matching a pattern
pytest -k "test_trim_silence"
```

### Test Structure
The test suite (`test_finetune_script.py`) is organized into test classes:

- **TestDataStructures**: Tests for `TakeKey` and `Take` dataclasses (immutability, equality, hashing)
- **TestIOUtilities**: Tests for file I/O functions (load_sentences, jsonl reading/writing, safe writes)
- **TestAudioProcessing**: Tests for audio functions (trim_silence, write_wav) using synthetic audio
- **TestManifestManagement**: Tests for manifest functions (parsing, rebuilding, train/val split, resume logic)
- **TestEdgeCases**: Unicode handling, precision rounding, version gaps

### Coverage
The test suite covers:
- All utility functions in sections 1-4 of the script
- Edge cases (empty files, unicode text, malformed data)
- Deterministic behavior (same seed produces same train/val split)
- File system operations using pytest's `tmp_path` fixture

Note: Interactive functions (`record_with_space`, `play_wav`, UI prompts) are not tested as they require user interaction.

## Platform Notes

- **macOS**: Grant microphone permission to Terminal/Python in System Preferences
- **Windows**: Run in standard terminal (not some IDE consoles) for `readchar` key capture
- Uses `sounddevice` which requires PortAudio backend
