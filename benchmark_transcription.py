#!/usr/bin/env python3
"""
Benchmark transcription server performance against ground truth.
Calculates WER, CER, and generates detailed error reports.
"""

import json
import requests
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import time
from datetime import datetime

# Simple WER/CER calculation (no external dependencies)
def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    """Calculate Levenshtein distance between two sequences."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    return dp[m][n]

def calculate_wer(reference: str, hypothesis: str) -> Tuple[float, int, int]:
    """Calculate Word Error Rate."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float('inf'), 0, 0

    distance = levenshtein_distance(ref_words, hyp_words)
    wer = distance / len(ref_words)
    return wer, distance, len(ref_words)

def calculate_cer(reference: str, hypothesis: str) -> Tuple[float, int, int]:
    """Calculate Character Error Rate."""
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float('inf'), 0, 0

    distance = levenshtein_distance(ref_chars, hyp_chars)
    cer = distance / len(ref_chars)
    return cer, distance, len(ref_chars)

def transcribe_audio(file_path: str, server_url: str, timeout: int = 30) -> Dict:
    """Send audio file to transcription server."""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'audio/wav')}
            response = requests.post(server_url, files=files, timeout=timeout)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        return {'status': 'error', 'error': str(e)}

def load_manifest(manifest_path: str) -> List[Dict]:
    """Load manifest file."""
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def run_benchmark(manifest_path: str, server_url: str, max_samples: int = None,
                  delay: float = 0.0) -> Dict:
    """Run benchmark on all samples in manifest."""

    print(f"Loading manifest: {manifest_path}")
    samples = load_manifest(manifest_path)

    if max_samples:
        samples = samples[:max_samples]

    print(f"Benchmarking {len(samples)} samples against {server_url}\n")

    results = []
    total_wer_distance = 0
    total_wer_words = 0
    total_cer_distance = 0
    total_cer_chars = 0
    errors = 0

    for i, sample in enumerate(samples, 1):
        file_path = sample['audio_filepath']
        reference = sample['text']

        print(f"[{i}/{len(samples)}] Processing {Path(file_path).name}...", end=' ')

        # Transcribe
        response = transcribe_audio(file_path, server_url)

        if response.get('status') == 'success':
            hypothesis = response['data']['transcript']

            # Calculate metrics
            wer, wer_dist, wer_words = calculate_wer(reference, hypothesis)
            cer, cer_dist, cer_chars = calculate_cer(reference, hypothesis)

            total_wer_distance += wer_dist
            total_wer_words += wer_words
            total_cer_distance += cer_dist
            total_cer_chars += cer_chars

            result = {
                'file': Path(file_path).name,
                'reference': reference,
                'hypothesis': hypothesis,
                'wer': wer,
                'cer': cer,
                'status': 'success'
            }

            print(f"WER: {wer:.2%}, CER: {cer:.2%}")

            if wer > 0:
                print(f"    REF: {reference}")
                print(f"    HYP: {hypothesis}")
        else:
            errors += 1
            result = {
                'file': Path(file_path).name,
                'reference': reference,
                'hypothesis': None,
                'wer': None,
                'cer': None,
                'status': 'error',
                'error': response.get('error', 'Unknown error')
            }
            print(f"ERROR: {result['error']}")

        results.append(result)

        # Rate limiting
        if delay > 0 and i < len(samples):
            time.sleep(delay)

    # Calculate overall metrics
    overall_wer = total_wer_distance / total_wer_words if total_wer_words > 0 else 0.0
    overall_cer = total_cer_distance / total_cer_chars if total_cer_chars > 0 else 0.0

    return {
        'timestamp': datetime.now().isoformat(),
        'manifest': manifest_path,
        'server_url': server_url,
        'total_samples': len(samples),
        'successful': len(samples) - errors,
        'errors': errors,
        'overall_wer': overall_wer,
        'overall_cer': overall_cer,
        'results': results
    }

def print_summary(benchmark: Dict):
    """Print benchmark summary."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Timestamp: {benchmark['timestamp']}")
    print(f"Server: {benchmark['server_url']}")
    print(f"Manifest: {benchmark['manifest']}")
    print(f"\nSamples: {benchmark['total_samples']}")
    print(f"Successful: {benchmark['successful']}")
    print(f"Errors: {benchmark['errors']}")
    print(f"\n{'OVERALL WORD ERROR RATE (WER):':<40} {benchmark['overall_wer']:.2%}")
    print(f"{'OVERALL CHARACTER ERROR RATE (CER):':<40} {benchmark['overall_cer']:.2%}")

    # Top errors
    successful_results = [r for r in benchmark['results'] if r['status'] == 'success' and r['wer'] is not None]
    if successful_results:
        top_errors = sorted(successful_results, key=lambda x: x['wer'], reverse=True)[:10]

        print(f"\nTOP 10 WORST TRANSCRIPTIONS:")
        print("-" * 80)
        for i, result in enumerate(top_errors, 1):
            if result['wer'] > 0:
                print(f"\n{i}. {result['file']} (WER: {result['wer']:.2%})")
                print(f"   REF: {result['reference']}")
                print(f"   HYP: {result['hypothesis']}")

    print("="*80)

def save_results(benchmark: Dict, output_path: str):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(benchmark, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark transcription server performance')
    parser.add_argument('--manifest', required=True, help='Path to manifest JSONL file')
    parser.add_argument('--server', default='http://192.168.1.203:8000/transcribe',
                        help='Transcription server URL')
    parser.add_argument('--output', default='benchmark_results.json',
                        help='Output file for detailed results')
    parser.add_argument('--max-samples', type=int, help='Limit number of samples to test')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Delay between requests (seconds)')

    args = parser.parse_args()

    # Run benchmark
    benchmark = run_benchmark(
        manifest_path=args.manifest,
        server_url=args.server,
        max_samples=args.max_samples,
        delay=args.delay
    )

    # Print summary
    print_summary(benchmark)

    # Save results
    save_results(benchmark, args.output)

if __name__ == '__main__':
    main()
