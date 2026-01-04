#!/usr/bin/env python3
"""
Dictation dataset recorder (multi-take per sentence) -> WAV + NeMo JSONL manifests
------------------------------------------------------------------------------
Features (as requested):
- For each sentence, record multiple "versions" (takes) at different speeds/emphases.
- Uses SPACE to start and SPACE to stop recording:
    "press [space] to record"
    "press [space] to complete recording"
- After each recording, shows the sentence again and offers:
    [1] play previous recording
    [2] delete previous recording
    [3] record another version
    [4] proceed (move to next sentence OR jump)
- Persists progress in a master JSONL manifest so you can exit/restart and resume.
- Writes NeMo-compatible JSONL manifests:
    manifests/master_manifest.jsonl  (authoritative)
    manifests/train_manifest.jsonl
    manifests/val_manifest.jsonl

Dependencies:
  pip install sounddevice soundfile numpy readchar

Notes:
- On macOS: grant microphone permission to Terminal/Python.
- On Windows: run in a normal terminal (not some IDE consoles) for key capture.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import readchar
import sounddevice as sd
import soundfile as sf


# ----------------------------- data structures ----------------------------- #

@dataclass(frozen=True)
class TakeKey:
    sent_idx: int
    version: int

@dataclass
class Take:
    sent_idx: int
    version: int
    text: str
    audio_filepath: str
    duration: float
    sample_rate: int


# ----------------------------- utils: io ---------------------------------- #

def load_sentences(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    sents = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not sents:
        raise ValueError(f"No sentences found in {path}")
    return sents

def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def safe_write_lines(path: Path, lines: List[str]) -> None:
    safe_write_text(path, "".join(lines))

def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out

def write_jsonl(path: Path, records: List[dict]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in records]
    safe_write_lines(path, lines)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------- audio -------------------------------------- #

def list_devices() -> None:
    devs = sd.query_devices()
    print("Audio devices:")
    for i, d in enumerate(devs):
        io = []
        if d.get("max_input_channels", 0) > 0:
            io.append("IN")
        if d.get("max_output_channels", 0) > 0:
            io.append("OUT")
        io_str = "/".join(io) if io else "—"
        print(f"{i:>2}  [{io_str:>6}]  {d['name']}")

def resolve_input_device(device: Optional[str]) -> Optional[int]:
    if device is None:
        return None
    if device.isdigit():
        return int(device)
    devs = sd.query_devices()
    matches = []
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0 and device.lower() in d["name"].lower():
            matches.append(i)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print("Multiple devices matched. Use --list-devices and pass --device as an index.")
        for i in matches:
            print(f"  {i}: {sd.query_devices(i)['name']}")
        sys.exit(2)
    print("No input device matched. Use --list-devices.")
    sys.exit(2)

def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -40.0, pad_ms: int = 80) -> np.ndarray:
    if audio.size == 0:
        return audio
    mono = audio.mean(axis=1)
    eps = 1e-9
    db = 20 * np.log10(np.abs(mono) + eps)
    keep = db > threshold_db
    if not np.any(keep):
        return audio
    idx = np.where(keep)[0]
    start = max(int(idx[0] - (pad_ms / 1000.0) * sr), 0)
    end = min(int(idx[-1] + (pad_ms / 1000.0) * sr), audio.shape[0] - 1)
    return audio[start : end + 1, :]

def record_with_space(
    sr: int,
    channels: int,
    max_seconds: float,
    device: Optional[int],
    gain: float = 1.0,
) -> np.ndarray:
    """
    Wait for SPACE to start; then record until SPACE again (or max_seconds).
    """
    def wait_for_space(prompt: str) -> None:
        print(prompt)
        while True:
            k = readchar.readkey()
            if k == " ":
                return

    wait_for_space("press [space] to record")

    frames: List[np.ndarray] = []
    stop_event = threading.Event()

    def key_listener():
        # wait for space to stop
        while True:
            k = readchar.readkey()
            if k == " ":
                stop_event.set()
                return

    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()

    print("Recording... press [space] to complete recording")

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"[Audio status] {status}", file=sys.stderr)
        frames.append(indata.copy())
        if stop_event.is_set():
            raise sd.CallbackStop()

    start_t = time.time()
    with sd.InputStream(
        samplerate=sr,
        channels=channels,
        dtype="float32",
        device=device,
        callback=callback,
    ):
        while True:
            time.sleep(0.05)
            if stop_event.is_set():
                break
            if (time.time() - start_t) >= max_seconds:
                print("(Reached max duration; stopping.)")
                break
            # If max duration reached, force stop by setting event
            if (time.time() - start_t) >= max_seconds:
                stop_event.set()
                break

    audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, channels), dtype=np.float32)
    if gain != 1.0:
        audio = np.clip(audio * float(gain), -1.0, 1.0)
    return audio

def write_wav(path: Path, audio: np.ndarray, sr: int) -> float:
    ensure_dir(path.parent)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return float(audio.shape[0] / sr) if sr > 0 else 0.0

def play_wav(path: Path) -> None:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    sd.play(data, sr)
    sd.wait()


# ----------------------------- manifest mgmt ------------------------------- #

def take_filename(sent_idx: int, version: int) -> str:
    return f"utt_{sent_idx:04d}_v{version:02d}.wav"

def parse_master_records(records: List[dict]) -> Dict[TakeKey, Take]:
    takes: Dict[TakeKey, Take] = {}
    for r in records:
        try:
            t = Take(
                sent_idx=int(r["sent_idx"]),
                version=int(r["version"]),
                text=str(r["text"]),
                audio_filepath=str(r["audio_filepath"]),
                duration=float(r["duration"]),
                sample_rate=int(r.get("sample_rate", 16000)),
            )
            takes[TakeKey(t.sent_idx, t.version)] = t
        except Exception:
            # If something is malformed, ignore that line rather than crashing
            continue
    return takes

def rebuild_master_records(takes: Dict[TakeKey, Take]) -> List[dict]:
    # stable ordering: sentence then version
    out: List[dict] = []
    for key in sorted(takes.keys(), key=lambda k: (k.sent_idx, k.version)):
        t = takes[key]
        out.append({
            "sent_idx": t.sent_idx,
            "version": t.version,
            "text": t.text,
            "audio_filepath": t.audio_filepath,
            "duration": round(float(t.duration), 4),
            "sample_rate": t.sample_rate,
        })
    return out

def compute_next_version_for_sentence(takes: Dict[TakeKey, Take], sent_idx: int) -> int:
    versions = [k.version for k in takes.keys() if k.sent_idx == sent_idx]
    return (max(versions) + 1) if versions else 1

def current_sentence_resume_index(num_sentences: int, takes: Dict[TakeKey, Take]) -> int:
    """
    Resume at the first sentence with no takes, otherwise at the last sentence that has takes.
    This gives a sensible "pick up where I left off" behavior without needing a separate state file.
    """
    took_any = False
    last_with_takes = 1
    has_take = {i: False for i in range(1, num_sentences + 1)}
    for k in takes.keys():
        if 1 <= k.sent_idx <= num_sentences:
            has_take[k.sent_idx] = True
            took_any = True
            last_with_takes = max(last_with_takes, k.sent_idx)
    if not took_any:
        return 1
    # If there exists any sentence after last_with_takes with no takes, we'll start at last_with_takes
    # but allow user to proceed/jump.
    # Better: if last_with_takes < num_sentences, start there; else start at last.
    return min(last_with_takes, num_sentences)

def write_train_val_manifests(
    master_takes: Dict[TakeKey, Take],
    train_path: Path,
    val_path: Path,
    val_fraction: float,
    seed: int,
) -> None:
    keys = sorted(master_takes.keys(), key=lambda k: (k.sent_idx, k.version))
    rng = random.Random(seed)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:n_val])

    train_recs = []
    val_recs = []
    for k in keys:
        t = master_takes[k]
        rec = {
            "audio_filepath": t.audio_filepath,
            "duration": round(float(t.duration), 4),
            "text": t.text,
        }
        if k in val_set:
            val_recs.append(rec)
        else:
            train_recs.append(rec)

    write_jsonl(train_path, train_recs)
    write_jsonl(val_path, val_recs)


# ----------------------------- UI helpers ---------------------------------- #

def print_sentence_header(sent_idx: int, total: int, text: str, n_versions: int) -> None:
    print("\n" + "=" * 90)
    print(f"Sentence {sent_idx}/{total}  |  Recorded versions: {n_versions}")
    print("-" * 90)
    print(text)
    print("=" * 90)

def prompt_action() -> str:
    print("\nChoose:")
    print("  [1] play previous recording")
    print("  [2] delete previous recording")
    print("  [3] record another version")
    print("  [4] proceed / jump")
    print("  [q] quit")
    while True:
        k = readchar.readkey().lower()
        if k in ("1", "2", "3", "4", "q"):
            return k

def prompt_proceed(sent_idx: int, total: int) -> int:
    """
    Proceed behavior:
      - Enter / Space: next sentence
      - 'j': jump to sentence number (prompt)
      - 'b': back one sentence
    """
    print("\nProceed options:")
    print("  [space] next sentence")
    print("  [b]     previous sentence")
    print("  [j]     jump to sentence #")
    print("  [space] (again) is the same as next")
    while True:
        k = readchar.readkey().lower()
        if k == " ":
            return min(total, sent_idx + 1)
        if k == "b":
            return max(1, sent_idx - 1)
        if k == "j":
            try:
                sys.stdout.write("\nEnter sentence number (1..{}): ".format(total))
                sys.stdout.flush()
                num = sys.stdin.readline().strip()
                if not num:
                    continue
                val = int(num)
                if 1 <= val <= total:
                    return val
                print("Out of range.")
            except Exception:
                print("Invalid number.")


# ----------------------------- main loop ----------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True, help="sentences.txt (one per line)")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--samplerate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--device", type=str, default=None, help="Input device index or name substring")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--trim-silence", action="store_true")
    ap.add_argument("--silence-threshold-db", type=float, default=-40.0)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return

    sentences_path = Path(args.sentences).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    audio_dir = out_dir / "audio"
    manif_dir = out_dir / "manifests"
    ensure_dir(audio_dir)
    ensure_dir(manif_dir)

    master_path = manif_dir / "master_manifest.jsonl"
    train_path = manif_dir / "train_manifest.jsonl"
    val_path = manif_dir / "val_manifest.jsonl"

    sents = load_sentences(sentences_path)
    total = len(sents)

    input_device = resolve_input_device(args.device)

    # Load existing progress
    master_records = read_jsonl(master_path)
    takes = parse_master_records(master_records)

    sent_idx = current_sentence_resume_index(total, takes)

    print("\n=== Dictation Recorder ===")
    print(f"Sentences:   {sentences_path} ({total})")
    print(f"Output dir:  {out_dir}")
    print(f"Audio dir:   {audio_dir}")
    print(f"Master JSONL:{master_path} (resume source)")
    print(f"Sample rate: {args.samplerate}, Channels: {args.channels}")
    print(f"Device:      {input_device if input_device is not None else 'default'}")
    print("\nHotkeys: SPACE to start/stop recording; then choose menu actions.\n")

    # Main interactive loop
    last_take_key: Optional[TakeKey] = None

    while True:
        text = sents[sent_idx - 1]
        n_versions = sum(1 for k in takes.keys() if k.sent_idx == sent_idx)
        print_sentence_header(sent_idx, total, text, n_versions)

        # If the user is resuming, we don't auto-record; we just wait for action.
        print("Ready. Record a new version for this sentence or proceed.")
        print("Tip: Press [3] to record another version, or [4] to proceed/jump.\n")

        action = prompt_action()
        if action == "q":
            print("\nExiting. Progress saved in master manifest.")
            break

        if action == "1":
            if last_take_key is None or last_take_key not in takes:
                # try to find the most recent take for this sentence
                existing = [k for k in takes.keys() if k.sent_idx == sent_idx]
                if not existing:
                    print("No recording to play yet for this sentence.")
                    continue
                last = max(existing, key=lambda k: k.version)
                last_take_key = last

            wav = Path(takes[last_take_key].audio_filepath)
            if not wav.exists():
                print(f"File missing: {wav}")
                continue
            print(f"Playing: {wav.name}")
            try:
                play_wav(wav)
            except Exception as e:
                print(f"Playback error: {e}")
            continue

        if action == "2":
            # delete previous recording (last_take_key or latest for sentence)
            if last_take_key is None or last_take_key not in takes:
                existing = [k for k in takes.keys() if k.sent_idx == sent_idx]
                if not existing:
                    print("No recording to delete for this sentence.")
                    continue
                last_take_key = max(existing, key=lambda k: k.version)

            t = takes.get(last_take_key)
            if not t:
                print("Nothing to delete.")
                continue

            wav = Path(t.audio_filepath)
            if wav.exists():
                try:
                    wav.unlink()
                    print(f"Deleted file: {wav}")
                except Exception as e:
                    print(f"Could not delete file: {e}")

            # remove from manifest
            del takes[last_take_key]
            last_take_key = None

            # persist master + train/val
            write_jsonl(master_path, rebuild_master_records(takes))
            write_train_val_manifests(takes, train_path, val_path, args.val_fraction, args.seed)
            print("Deleted take and updated manifests.")
            continue

        if action == "3":
            # record a new version for current sentence
            version = compute_next_version_for_sentence(takes, sent_idx)
            wav_path = (audio_dir / take_filename(sent_idx, version)).resolve()

            try:
                audio = record_with_space(
                    sr=args.samplerate,
                    channels=args.channels,
                    max_seconds=args.max_seconds,
                    device=input_device,
                    gain=args.gain,
                )
            except KeyboardInterrupt:
                print("\nInterrupted. Saving manifests and exiting.")
                write_jsonl(master_path, rebuild_master_records(takes))
                write_train_val_manifests(takes, train_path, val_path, args.val_fraction, args.seed)
                break

            if args.trim_silence:
                audio = trim_silence(audio, args.samplerate, args.silence_threshold_db)

            duration = write_wav(wav_path, audio, args.samplerate)

            # record metadata
            tk = TakeKey(sent_idx, version)
            takes[tk] = Take(
                sent_idx=sent_idx,
                version=version,
                text=text,
                audio_filepath=str(wav_path),
                duration=duration,
                sample_rate=args.samplerate,
            )
            last_take_key = tk

            # persist master + train/val every time so resume is always safe
            write_jsonl(master_path, rebuild_master_records(takes))
            write_train_val_manifests(takes, train_path, val_path, args.val_fraction, args.seed)

            print("\nSaved take:")
            print(f"  sentence: {sent_idx}, version: {version}")
            print(f"  wav: {wav_path.name}")
            print(f"  duration: {duration:.2f}s")
            print("\nSentence text again:")
            print(text)
            continue

        if action == "4":
            sent_idx = prompt_proceed(sent_idx, total)
            continue

    # final persist (defensive)
    write_jsonl(master_path, rebuild_master_records(takes))
    write_train_val_manifests(takes, train_path, val_path, args.val_fraction, args.seed)
    print("Manifests written:")
    print(f"  {master_path}")
    print(f"  {train_path}")
    print(f"  {val_path}")


if __name__ == "__main__":
    main()

