#!/usr/bin/env python3
"""Sample from frequency distributions and save samples with metadata"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

def load_frequency_data(file_path: str) -> Dict:
    """Load frequency data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_delta_distribution(control_data: Dict, target_data: Dict) -> Dict:
    """Calculate the difference between two frequency distributions."""
    control_freq = control_data["frequencies"]
    target_freq = target_data["frequencies"]
    
    # Get all unique numbers from both distributions
    all_numbers = set(control_freq.keys()) | set(target_freq.keys())
    
    delta_freq = {}
    for num in all_numbers:
        control_count = control_freq.get(num, 0)
        target_count = target_freq.get(num, 0)
        delta = target_count - control_count
        if delta != 0:  # Only include non-zero differences
            delta_freq[num] = delta
    
    return {
        "preference_name": f"delta_{control_data['preference_name']}_to_{target_data['preference_name']}",
        "total_numbers": sum(delta_freq.values()),
        "unique_numbers": len(delta_freq),
        "frequencies": delta_freq,
        "source_files": {
            "control": control_data.get("file_path", "unknown"),
            "target": target_data.get("file_path", "unknown")
        }
    }

def create_sampling_weights(frequencies: Dict[str, int]) -> Tuple[List[str], List[float]]:
    """Create normalized sampling weights from frequencies."""
    numbers = list(frequencies.keys())
    counts = list(frequencies.values())
    
    # Normalize to probabilities
    total = sum(counts)
    weights = [count / total for count in counts]
    
    return numbers, weights

def sample_from_distribution(numbers: List[str], weights: List[float], n_samples: int, seed: int = None) -> List[str]:
    """Sample from a distribution using weighted random sampling."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Use numpy's choice for weighted sampling
    samples = np.random.choice(numbers, size=n_samples, p=weights, replace=True)
    return samples.tolist()

def analyze_distribution(frequencies: Dict[str, int]) -> Dict:
    """Analyze distribution statistics."""
    counts = list(frequencies.values())
    numbers = [int(num) for num in frequencies.keys()]
    
    return {
        "total_samples": sum(counts),
        "unique_values": len(frequencies),
        "min_value": min(numbers),
        "max_value": max(numbers),
        "mean_value": np.mean(numbers),
        "median_value": np.median(numbers),
        "std_value": np.std(numbers),
        "most_frequent": max(frequencies.items(), key=lambda x: x[1]),
        "least_frequent": min(frequencies.items(), key=lambda x: x[1])
    }

def main():
    parser = argparse.ArgumentParser(description="Sample from frequency distributions")
    parser.add_argument("-c", "--control_file", help="Path to control frequency file")
    parser.add_argument("-t", "--target_file", help="Path to target frequency file")
    parser.add_argument("-d", "--delta", action="store_true", help="Sample from delta distribution (target - control)")
    parser.add_argument("-n", "--n_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("-s", "--seed", type=int, help="Random seed for reproducible sampling")
    parser.add_argument("-o", "--output_dir", default="experiments/samples", help="Output directory for samples")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.delta and (not args.control_file or not args.target_file):
        print("Error: --delta requires both --control_file and --target_file")
        return
    
    if not args.delta and not args.control_file and not args.target_file:
        print("Error: Must specify either --control_file, --target_file, or --delta with both files")
        return
    
    # Load data
    if args.delta:
        print("Loading control and target frequency data...")
        control_data = load_frequency_data(args.control_file)
        target_data = load_frequency_data(args.target_file)
        
        # Add file paths to data for reference
        control_data["file_path"] = args.control_file
        target_data["file_path"] = args.target_file
        
        print("Calculating delta distribution...")
        distribution_data = calculate_delta_distribution(control_data, target_data)
        source_type = "delta"
        source_files = [args.control_file, args.target_file]
        
    elif args.control_file:
        print("Loading control frequency data...")
        distribution_data = load_frequency_data(args.control_file)
        distribution_data["file_path"] = args.control_file
        source_type = "control"
        source_files = [args.control_file]
        
    elif args.target_file:
        print("Loading target frequency data...")
        distribution_data = load_frequency_data(args.target_file)
        distribution_data["file_path"] = args.target_file
        source_type = "target"
        source_files = [args.target_file]
    
    # Create sampling weights
    print("Preparing sampling weights...")
    numbers, weights = create_sampling_weights(distribution_data["frequencies"])
    
    # Analyze distribution
    print("Analyzing distribution...")
    analysis = analyze_distribution(distribution_data["frequencies"])
    
    # Generate samples
    print(f"Generating {args.n_samples} samples...")
    samples = sample_from_distribution(numbers, weights, args.n_samples, args.seed)
    
    # Prepare output data
    output_data = {
        "sampling_metadata": {
            "script": "sample_distribution.py",
            "timestamp": datetime.now().isoformat(),
            "n_samples": args.n_samples,
            "random_seed": args.seed,
            "source_type": source_type,
            "source_files": source_files
        },
        "distribution_info": {
            "preference_name": distribution_data["preference_name"],
            "file_path": distribution_data.get("file_path", "unknown"),
            "total_numbers": distribution_data["total_numbers"],
            "unique_numbers": distribution_data["unique_numbers"]
        },
        "distribution_analysis": analysis,
        "samples": samples
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{distribution_data['preference_name']}_n{args.n_samples}_{timestamp}.json"
    if args.seed:
        filename = f"{distribution_data['preference_name']}_n{args.n_samples}_seed{args.seed}_{timestamp}.json"
    
    output_path = os.path.join(args.output_dir, filename)
    
    # Save samples
    print(f"Saving samples to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\nSampling Summary:")
    print(f"  Distribution: {distribution_data['preference_name']}")
    print(f"  Source type: {source_type}")
    print(f"  Samples generated: {len(samples)}")
    print(f"  Unique values in distribution: {analysis['unique_values']}")
    print(f"  Value range: {analysis['min_value']} - {analysis['max_value']}")
    print(f"  Most frequent: {analysis['most_frequent'][0]} (count: {analysis['most_frequent'][1]})")
    print(f"  Output file: {output_path}")

if __name__ == "__main__":
    main()
