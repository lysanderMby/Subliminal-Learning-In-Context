#!/usr/bin/env python3
"""Complete workflow runner for dislike owls experiments"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
    else:
        print(f"❌ {description} failed")
        print("Error:", result.stderr)
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run complete dislike owls workflow")
    parser.add_argument("-p", "--preference", default="dislike_owls", 
                       help="Preference token (default: dislike_owls)")
    parser.add_argument("-n", "--n_samples", type=int, default=2000, 
                       help="Number of samples for distribution generation (default: 2000)")
    parser.add_argument("-e", "--experiment_samples", type=int, default=10000, 
                       help="Number of samples for experiments (default: 10000)")
    parser.add_argument("-r", "--seed", type=int, default=42, 
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--skip-distribution", action="store_true", 
                       help="Skip distribution generation (use existing files)")
    parser.add_argument("--skip-sampling", action="store_true", 
                       help="Skip sampling generation (use existing files)")
    parser.add_argument("--skip-experiments", action="store_true", 
                       help="Skip experiments (use existing results)")
    parser.add_argument("--skip-analysis", action="store_true", 
                       help="Skip analysis (use existing results)")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting complete dislike owls workflow")
    print(f"   Preference: {args.preference}")
    print(f"   Distribution samples: {args.n_samples}")
    print(f"   Experiment samples: {args.experiment_samples}")
    print(f"   Seed: {args.seed}")
    
    # Create necessary directories
    os.makedirs("experiments/dislike_owls", exist_ok=True)
    os.makedirs("experiments/samples", exist_ok=True)
    os.makedirs("experiments/number_choice_impact", exist_ok=True)
    os.makedirs("experiments/number_choice_impact_analysis", exist_ok=True)
    
    # Step 1: Generate frequency distributions
    if not args.skip_distribution:
        cmd = [
            "python", "visualise_distribution.py",
            "-p", args.preference,
            "-s", str(args.n_samples),
            "-r", str(args.seed)
        ]
        
        if not run_command(cmd, "Generate frequency distribution"):
            return 1
    
    # Step 2: Generate samples from distributions
    if not args.skip_sampling:
        # Find the generated frequency files
        freq_dir = Path("experiments/dislike_owls")
        freq_files = list(freq_dir.glob(f"{args.preference}_frequencies_*.json"))
        
        if not freq_files:
            print("❌ No frequency files found. Run distribution generation first.")
            return 1
        
        latest_freq = max(freq_files, key=lambda x: x.stat().st_mtime)
        print(f"Using frequency file: {latest_freq}")
        
        # Generate control samples (from existing control distribution)
        control_cmd = [
            "python", "sample_distribution.py",
            "-c", f"../experiments/control/control_frequencies.json",
            "-n", str(args.experiment_samples),
            "-s", str(args.seed)
        ]
        
        if not run_command(control_cmd, "Generate control samples"):
            return 1
        
        # Generate dislike_owls samples
        target_cmd = [
            "python", "sample_distribution.py",
            "-t", str(latest_freq),
            "-n", str(args.experiment_samples),
            "-s", str(args.seed)
        ]
        
        if not run_command(target_cmd, "Generate dislike_owls samples"):
            return 1
        
        # Generate delta samples
        delta_cmd = [
            "python", "sample_distribution.py",
            "-c", f"../experiments/control/control_frequencies.json",
            "-t", str(latest_freq),
            "-d", "-n", str(args.experiment_samples),
            "-s", str(args.seed)
        ]
        
        if not run_command(delta_cmd, "Generate delta samples"):
            return 1
    
    # Step 3: Run experiments
    if not args.skip_experiments:
        # Find the generated sample files
        samples_dir = Path("experiments/samples")
        control_samples = list(samples_dir.glob("control_n*_seed*.json"))
        target_samples = list(samples_dir.glob(f"{args.preference}_n*_seed*.json"))
        delta_samples = list(samples_dir.glob(f"delta_control_to_{args.preference}_n*_seed*.json"))
        
        if not control_samples or not target_samples or not delta_samples:
            print("❌ Sample files not found. Run sampling generation first.")
            return 1
        
        # Use the most recent files
        control_file = max(control_samples, key=lambda x: x.stat().st_mtime)
        target_file = max(target_samples, key=lambda x: x.stat().st_mtime)
        delta_file = max(delta_samples, key=lambda x: x.stat().st_mtime)
        
        print(f"Using control samples: {control_file}")
        print(f"Using target samples: {target_file}")
        print(f"Using delta samples: {delta_file}")
        
        # Run experiments
        exp_cmd = [
            "python", "number_experiments/run_all_experiments.py",
            "-c", str(control_file),
            "-o", str(target_file),
            "-d", str(delta_file),
            "-t", "owl",
            "--category", "animal",
            "-n", "100",
            "-s", str(args.seed)
        ]
        
        if not run_command(exp_cmd, "Run all experiments"):
            return 1
    
    # Step 4: Analyze results
    if not args.skip_analysis:
        # Analyze base model
        base_cmd = [
            "python", "number_experiments/analyze_results.py",
            "-m", "base",
            "-d", "experiments/number_choice_impact"
        ]
        
        if not run_command(base_cmd, "Analyze base model results"):
            return 1
        
        # Analyze finetuned model
        finetuned_cmd = [
            "python", "number_experiments/analyze_results.py",
            "-m", "finetuned",
            "-d", "experiments/number_choice_impact"
        ]
        
        if not run_command(finetuned_cmd, "Analyze finetuned model results"):
            return 1
    
    print(f"\n🎉 Complete workflow finished successfully!")
    print(f"   Results saved in: experiments/")
    print(f"   Analysis saved in: experiments/number_choice_impact_analysis/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
