#!/usr/bin/env python3
"""
Unit tests for finetune_script.py
Run with: pytest test_finetune_script.py -v
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from finetune_script import (
    Take,
    TakeKey,
    compute_next_version_for_sentence,
    current_sentence_resume_index,
    load_sentences,
    parse_master_records,
    read_jsonl,
    rebuild_master_records,
    safe_write_lines,
    safe_write_text,
    take_filename,
    trim_silence,
    write_jsonl,
    write_train_val_manifests,
    write_wav,
)


class TestDataStructures:
    """Test TakeKey and Take dataclasses"""

    def test_take_key_frozen(self):
        """TakeKey should be frozen (immutable)"""
        tk = TakeKey(sent_idx=1, version=2)
        with pytest.raises(AttributeError):
            tk.sent_idx = 3

    def test_take_key_equality(self):
        """TakeKey instances with same values should be equal"""
        tk1 = TakeKey(sent_idx=1, version=2)
        tk2 = TakeKey(sent_idx=1, version=2)
        tk3 = TakeKey(sent_idx=1, version=3)
        assert tk1 == tk2
        assert tk1 != tk3

    def test_take_key_hashable(self):
        """TakeKey should be hashable for use in dicts"""
        tk1 = TakeKey(sent_idx=1, version=2)
        tk2 = TakeKey(sent_idx=1, version=2)
        d = {tk1: "value"}
        assert d[tk2] == "value"

    def test_take_creation(self):
        """Take should store all required fields"""
        t = Take(
            sent_idx=1,
            version=2,
            text="Hello world",
            audio_filepath="/path/to/audio.wav",
            duration=1.5,
            sample_rate=16000,
        )
        assert t.sent_idx == 1
        assert t.version == 2
        assert t.text == "Hello world"
        assert t.duration == 1.5
        assert t.sample_rate == 16000


class TestIOUtilities:
    """Test I/O utility functions"""

    def test_load_sentences_basic(self, tmp_path):
        """load_sentences should read non-empty, non-comment lines"""
        sentences_file = tmp_path / "sentences.txt"
        sentences_file.write_text(
            "First sentence\n"
            "Second sentence\n"
            "  Third sentence  \n"
            "# This is a comment\n"
            "\n"
            "Fourth sentence\n"
        )
        sents = load_sentences(sentences_file)
        assert len(sents) == 4
        assert sents[0] == "First sentence"
        assert sents[1] == "Second sentence"
        assert sents[2] == "Third sentence"
        assert sents[3] == "Fourth sentence"

    def test_load_sentences_empty_file(self, tmp_path):
        """load_sentences should raise ValueError for empty file"""
        sentences_file = tmp_path / "empty.txt"
        sentences_file.write_text("\n\n# Just comments\n\n")
        with pytest.raises(ValueError, match="No sentences found"):
            load_sentences(sentences_file)

    def test_safe_write_text(self, tmp_path):
        """safe_write_text should write atomically"""
        test_file = tmp_path / "test.txt"
        safe_write_text(test_file, "Hello world")
        assert test_file.read_text() == "Hello world"

    def test_safe_write_text_creates_parent(self, tmp_path):
        """safe_write_text should create parent directories"""
        test_file = tmp_path / "subdir" / "test.txt"
        safe_write_text(test_file, "Hello")
        assert test_file.exists()
        assert test_file.read_text() == "Hello"

    def test_safe_write_lines(self, tmp_path):
        """safe_write_lines should write lines correctly"""
        test_file = tmp_path / "test.txt"
        safe_write_lines(test_file, ["line1\n", "line2\n", "line3\n"])
        assert test_file.read_text() == "line1\nline2\nline3\n"

    def test_read_jsonl_basic(self, tmp_path):
        """read_jsonl should parse valid JSONL"""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"key": "value1"}\n'
            '{"key": "value2"}\n'
            '\n'
            '{"key": "value3"}\n'
        )
        records = read_jsonl(jsonl_file)
        assert len(records) == 3
        assert records[0]["key"] == "value1"
        assert records[2]["key"] == "value3"

    def test_read_jsonl_nonexistent(self, tmp_path):
        """read_jsonl should return empty list for nonexistent file"""
        jsonl_file = tmp_path / "nonexistent.jsonl"
        records = read_jsonl(jsonl_file)
        assert records == []

    def test_write_jsonl(self, tmp_path):
        """write_jsonl should write valid JSONL format"""
        jsonl_file = tmp_path / "output.jsonl"
        records = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        write_jsonl(jsonl_file, records)

        content = jsonl_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"name": "Alice", "age": 30}
        assert json.loads(lines[1]) == {"name": "Bob", "age": 25}


class TestAudioProcessing:
    """Test audio processing functions"""

    def test_trim_silence_basic(self):
        """trim_silence should remove quiet sections at start/end"""
        # Create audio: silence, loud, silence
        sr = 16000
        duration = 1.0
        n_samples = int(sr * duration)

        audio = np.zeros((n_samples, 1), dtype=np.float32)
        # Add loud section in the middle
        start = n_samples // 4
        end = 3 * n_samples // 4
        audio[start:end, 0] = 0.5

        trimmed = trim_silence(audio, sr, threshold_db=-20.0, pad_ms=10)

        # Trimmed should be shorter than original
        assert trimmed.shape[0] < audio.shape[0]
        # Should still contain the loud section
        assert np.max(np.abs(trimmed)) > 0.4

    def test_trim_silence_all_quiet(self):
        """trim_silence should handle all-quiet audio"""
        sr = 16000
        audio = np.zeros((1000, 1), dtype=np.float32) + 0.001
        trimmed = trim_silence(audio, sr, threshold_db=-40.0)
        # Should return original if all below threshold
        assert trimmed.shape == audio.shape

    def test_trim_silence_empty(self):
        """trim_silence should handle empty audio"""
        audio = np.zeros((0, 1), dtype=np.float32)
        trimmed = trim_silence(audio, 16000)
        assert trimmed.shape == (0, 1)

    def test_trim_silence_stereo(self):
        """trim_silence should work with stereo audio"""
        sr = 16000
        n_samples = 10000  # Larger clip to ensure trimming happens
        audio = np.zeros((n_samples, 2), dtype=np.float32)
        audio[4000:6000, :] = 0.5  # Loud section in middle

        trimmed = trim_silence(audio, sr, threshold_db=-20.0, pad_ms=100)
        assert trimmed.shape[1] == 2  # Still stereo
        assert trimmed.shape[0] < n_samples  # Should be trimmed

    def test_write_wav(self, tmp_path):
        """write_wav should create wav file and return duration"""
        wav_path = tmp_path / "test.wav"
        sr = 16000
        duration_sec = 2.0
        n_samples = int(sr * duration_sec)
        audio = np.random.randn(n_samples, 1).astype(np.float32) * 0.1

        returned_duration = write_wav(wav_path, audio, sr)

        assert wav_path.exists()
        assert abs(returned_duration - duration_sec) < 0.01


class TestManifestManagement:
    """Test manifest management functions"""

    def test_take_filename(self):
        """take_filename should generate correct format"""
        assert take_filename(1, 1) == "utt_0001_v01.wav"
        assert take_filename(42, 5) == "utt_0042_v05.wav"
        assert take_filename(999, 99) == "utt_0999_v99.wav"

    def test_parse_master_records_basic(self):
        """parse_master_records should convert records to Take dict"""
        records = [
            {
                "sent_idx": 1,
                "version": 1,
                "text": "Hello",
                "audio_filepath": "/path/to/audio1.wav",
                "duration": 1.5,
                "sample_rate": 16000,
            },
            {
                "sent_idx": 1,
                "version": 2,
                "text": "Hello",
                "audio_filepath": "/path/to/audio2.wav",
                "duration": 1.8,
                "sample_rate": 16000,
            },
        ]
        takes = parse_master_records(records)

        assert len(takes) == 2
        assert TakeKey(1, 1) in takes
        assert TakeKey(1, 2) in takes
        assert takes[TakeKey(1, 1)].duration == 1.5
        assert takes[TakeKey(1, 2)].duration == 1.8

    def test_parse_master_records_malformed(self):
        """parse_master_records should skip malformed records"""
        records = [
            {
                "sent_idx": 1,
                "version": 1,
                "text": "Valid",
                "audio_filepath": "/path/to/audio.wav",
                "duration": 1.5,
            },
            {
                "sent_idx": "invalid",  # Invalid type
                "text": "Malformed",
            },
            {
                "sent_idx": 2,
                "version": 1,
                "text": "Also valid",
                "audio_filepath": "/path/to/audio2.wav",
                "duration": 2.0,
            },
        ]
        takes = parse_master_records(records)

        # Should only parse valid records
        assert len(takes) == 2
        assert TakeKey(1, 1) in takes
        assert TakeKey(2, 1) in takes

    def test_rebuild_master_records(self):
        """rebuild_master_records should create sorted list"""
        takes = {
            TakeKey(2, 1): Take(2, 1, "Second", "/path2.wav", 1.0, 16000),
            TakeKey(1, 2): Take(1, 2, "First v2", "/path1b.wav", 1.5, 16000),
            TakeKey(1, 1): Take(1, 1, "First v1", "/path1a.wav", 1.2, 16000),
        }

        records = rebuild_master_records(takes)

        assert len(records) == 3
        # Should be sorted by sent_idx, then version
        assert records[0]["sent_idx"] == 1
        assert records[0]["version"] == 1
        assert records[1]["sent_idx"] == 1
        assert records[1]["version"] == 2
        assert records[2]["sent_idx"] == 2
        assert records[2]["version"] == 1

    def test_compute_next_version_for_sentence(self):
        """compute_next_version_for_sentence should return next version number"""
        takes = {
            TakeKey(1, 1): Take(1, 1, "Text", "/path.wav", 1.0, 16000),
            TakeKey(1, 2): Take(1, 2, "Text", "/path.wav", 1.0, 16000),
            TakeKey(2, 1): Take(2, 1, "Text", "/path.wav", 1.0, 16000),
        }

        # Sentence 1 has versions 1 and 2, so next is 3
        assert compute_next_version_for_sentence(takes, 1) == 3
        # Sentence 2 has version 1, so next is 2
        assert compute_next_version_for_sentence(takes, 2) == 2
        # Sentence 3 has no versions, so next is 1
        assert compute_next_version_for_sentence(takes, 3) == 1

    def test_current_sentence_resume_index_empty(self):
        """current_sentence_resume_index should return 1 for empty takes"""
        takes = {}
        assert current_sentence_resume_index(10, takes) == 1

    def test_current_sentence_resume_index_partial(self):
        """current_sentence_resume_index should resume at last sentence with takes"""
        takes = {
            TakeKey(1, 1): Take(1, 1, "Text", "/path.wav", 1.0, 16000),
            TakeKey(2, 1): Take(2, 1, "Text", "/path.wav", 1.0, 16000),
            TakeKey(3, 1): Take(3, 1, "Text", "/path.wav", 1.0, 16000),
        }
        # Has takes for 1, 2, 3 but total is 10, so resume at 3
        assert current_sentence_resume_index(10, takes) == 3

    def test_current_sentence_resume_index_complete(self):
        """current_sentence_resume_index should handle all sentences recorded"""
        takes = {
            TakeKey(1, 1): Take(1, 1, "Text", "/path.wav", 1.0, 16000),
            TakeKey(2, 1): Take(2, 1, "Text", "/path.wav", 1.0, 16000),
            TakeKey(3, 1): Take(3, 1, "Text", "/path.wav", 1.0, 16000),
        }
        # All 3 sentences have takes
        assert current_sentence_resume_index(3, takes) == 3

    def test_write_train_val_manifests(self, tmp_path):
        """write_train_val_manifests should split takes correctly"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"

        takes = {
            TakeKey(i, 1): Take(
                i, 1, f"Text {i}", f"/path{i}.wav", 1.0, 16000
            )
            for i in range(1, 11)  # 10 takes
        }

        write_train_val_manifests(takes, train_path, val_path, val_fraction=0.2, seed=42)

        train_records = read_jsonl(train_path)
        val_records = read_jsonl(val_path)

        # With 10 takes and 0.2 fraction, expect 2 in val, 8 in train
        assert len(val_records) == 2
        assert len(train_records) == 8

        # Check NeMo format (should not have sent_idx/version)
        for rec in train_records + val_records:
            assert "audio_filepath" in rec
            assert "duration" in rec
            assert "text" in rec
            assert "sent_idx" not in rec
            assert "version" not in rec

    def test_write_train_val_manifests_deterministic(self, tmp_path):
        """write_train_val_manifests should produce same split with same seed"""
        train_path1 = tmp_path / "train1.jsonl"
        val_path1 = tmp_path / "val1.jsonl"
        train_path2 = tmp_path / "train2.jsonl"
        val_path2 = tmp_path / "val2.jsonl"

        takes = {
            TakeKey(i, 1): Take(i, 1, f"Text {i}", f"/path{i}.wav", 1.0, 16000)
            for i in range(1, 21)
        }

        write_train_val_manifests(takes, train_path1, val_path1, 0.2, seed=123)
        write_train_val_manifests(takes, train_path2, val_path2, 0.2, seed=123)

        assert read_jsonl(train_path1) == read_jsonl(train_path2)
        assert read_jsonl(val_path1) == read_jsonl(val_path2)

    def test_write_train_val_manifests_single_take(self, tmp_path):
        """write_train_val_manifests should handle single take gracefully"""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"

        takes = {
            TakeKey(1, 1): Take(1, 1, "Text", "/path.wav", 1.0, 16000)
        }

        write_train_val_manifests(takes, train_path, val_path, 0.5, seed=42)

        train_records = read_jsonl(train_path)
        val_records = read_jsonl(val_path)

        # With 1 take, val should be empty (n_val = 0 when len=1)
        assert len(val_records) == 0
        assert len(train_records) == 1


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_take_with_unicode_text(self):
        """Take should handle unicode text correctly"""
        t = Take(
            sent_idx=1,
            version=1,
            text="Hello 世界 🌍",
            audio_filepath="/path.wav",
            duration=1.0,
            sample_rate=16000,
        )
        assert t.text == "Hello 世界 🌍"

    def test_jsonl_with_unicode(self, tmp_path):
        """JSONL functions should handle unicode correctly"""
        jsonl_file = tmp_path / "unicode.jsonl"
        records = [
            {"text": "Hello 世界"},
            {"text": "Γειά σου κόσμε"},
            {"text": "مرحبا بالعالم"},
        ]

        write_jsonl(jsonl_file, records)
        loaded = read_jsonl(jsonl_file)

        assert loaded == records

    def test_rebuild_records_preserves_precision(self):
        """rebuild_master_records should round duration appropriately"""
        takes = {
            TakeKey(1, 1): Take(
                1, 1, "Text", "/path.wav", 1.123456789, 16000
            )
        }

        records = rebuild_master_records(takes)
        # Should round to 4 decimal places
        assert records[0]["duration"] == 1.1235

    def test_compute_next_version_handles_gaps(self):
        """compute_next_version_for_sentence should use max + 1 even with gaps"""
        takes = {
            TakeKey(1, 1): Take(1, 1, "Text", "/path.wav", 1.0, 16000),
            TakeKey(1, 5): Take(1, 5, "Text", "/path.wav", 1.0, 16000),
            # Missing versions 2, 3, 4
        }

        # Next version should be 6 (max + 1), not fill gaps
        assert compute_next_version_for_sentence(takes, 1) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
