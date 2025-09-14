#!/usr/bin/env python3
"""Run all optimized number choice experiments with cost optimizations"""

import asyncio
import subprocess
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import time

class ExperimentRunner:
    def __init__(self, base_dir: str = "experiments/number_choice_impact"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def check_sample_files(self, control_file: str = None, owl_animal_file: str = None, 
                          delta_file: str = None, auto_generate_delta: bool = True) -> Dict[str, str]:
        """Check and prepare sample files."""
        sample_files = {
            "control": control_file,
            "owl_animal": owl_animal_file,
            "delta": delta_file
        }
        
        # Check existing files
        for name, file_path in sample_files.items():
            if file_path:
                # Try multiple path variations
                possible_paths = [
                    file_path,
                    f"../{file_path}",
                    f"../experiments/samples/{Path(file_path).name}",
                    f"experiments/samples/{Path(file_path).name}"
                ]
                
                found_path = None
                for path in possible_paths:
                    if Path(path).exists():
                        found_path = path
                        break
                
                if found_path:
                    sample_files[name] = found_path
                    print(f"Found {name} file: {found_path}")
                else:
                    print(f"Warning: Sample file {file_path} not found in any of these locations:")
                    for path in possible_paths:
                        print(f"  - {path}")
                    sample_files[name] = None
        
        # Generate delta if needed and auto_generate_delta is True
        if auto_generate_delta and not sample_files["delta"] and sample_files["control"] and sample_files["owl_animal"]:
            print("Auto-generating delta distribution...")
            delta_cmd = [
                "python", "sample_distribution.py",
                "-c", sample_files["control"],
                "-t", sample_files["owl_animal"],
                "-d", "-n", "10000", "-s", "42"
            ]
            result = subprocess.run(delta_cmd, capture_output=True, text=True, cwd="..")
            if result.returncode == 0:
                # Find the generated delta file
                samples_dir = Path("../experiments/samples")
                delta_files = list(samples_dir.glob("delta_control_to_owl_animal_n10000_*.json"))
                if delta_files:
                    sample_files["delta"] = str(delta_files[-1])  # Use most recent
                    print(f"Delta distribution created: {sample_files['delta']}")
                else:
                    print("Warning: Delta file generated but not found")
            else:
                print(f"Failed to create delta: {result.stderr}")
        
        return sample_files
    
    def run_single_experiment(self, model_id: str, target: str, category: str, 
                            n_samples: int, sample_file: str = None, seed: int = None) -> Dict:
        """Run a single experiment and return results."""
        cmd = [
            "python", "optimized_number_choice_impact.py",
            "-m", model_id,
            "-t", target,
            "--category", category,
            "-n", str(n_samples)
        ]
        
        if sample_file:
            cmd.extend(["-s", sample_file])
        
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        print(f"Running: {' '.join(cmd)}")
        if sample_file:
            print(f"  Sample file: {sample_file}")
        if seed is not None:
            print(f"  Seed: {seed}")
        else:
            print(f"  Seed: None (using default)")
        start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        experiment_result = {
            "model_id": model_id,
            "sample_file": sample_file,
            "duration": duration,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
        if result.returncode == 0:
            # Extract output file from stdout
            for line in result.stdout.split('\n'):
                if 'Results saved to:' in line:
                    output_file = line.split('Results saved to:')[-1].strip()
                    experiment_result["output_file"] = output_file
                    break
            print(f"✓ Success ({duration:.1f}s)")
        else:
            print(f"✗ Failed ({duration:.1f}s): {result.stderr}")
        
        return experiment_result
    
    def run_all_experiments(self, models: Dict[str, str], sample_files: Dict[str, str], 
                          target: str = "owl", category: str = "animal", 
                          n_samples: int = 20, seed: int = None) -> List[Dict]:
        """Run all experiments."""
        experiments = []
        
        # Define experiment configurations
        conditions = [
            ("control", None),
            ("control_numbers", sample_files.get("control")),
            ("owl_animal_numbers", sample_files.get("owl_animal")),
            ("delta_numbers", sample_files.get("delta"))
        ]
        
        total_experiments = len(models) * len(conditions)
        print(f"Running {total_experiments} experiments...")
        print("=" * 60)
        
        experiment_num = 0
        for model_name, model_id in models.items():
            for condition_name, sample_file in conditions:
                experiment_num += 1
                print(f"\n[{experiment_num}/{total_experiments}] {model_name} - {condition_name}")
                print("-" * 40)
                
                # Skip if sample file is required but not available
                if sample_file is None and "numbers" in condition_name:
                    print(f"⚠ Skipping {condition_name} - no sample file available")
                    continue
                
                result = self.run_single_experiment(
                    model_id=model_id,
                    target=target,
                    category=category,
                    n_samples=n_samples,
                    sample_file=sample_file,
                    seed=seed
                )
                
                result["model_name"] = model_name
                result["condition"] = condition_name
                experiments.append(result)
        
        return experiments
    
    def save_experiment_summary(self, experiments: List[Dict], output_file: str = None):
        """Save experiment summary."""
        if output_file is None:
            output_file = self.base_dir / "experiment_summary.json"
        
        summary = {
            "experiment_metadata": {
                "script": "run_all_experiments.py",
                "total_experiments": len(experiments),
                "successful_experiments": sum(1 for exp in experiments if exp["success"]),
                "failed_experiments": sum(1 for exp in experiments if not exp["success"])
            },
            "experiments": experiments
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExperiment summary saved to: {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(
        description="Run all optimized number choice experiments with cost optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default sample files (auto-detect latest)
  python run_all_experiments.py
  
  # Specify specific sample files
  python run_all_experiments.py -c experiments/samples/control_n10000_20250914_144340.json -o experiments/samples/owl_animal_n10000_20250914_144340.json
  
  # Run only control experiments (no numbers)
  python run_all_experiments.py --no-numbers
  
  # Run with custom parameters
  python run_all_experiments.py -n 50 -s 123 --target cat --category animal
        """
    )
    
    # Sample file arguments
    parser.add_argument("-c", "--control-file", 
                       help="Path to control sample file (e.g., experiments/samples/control_n10000_*.json)")
    parser.add_argument("-o", "--owl-animal-file", 
                       help="Path to owl_animal sample file (e.g., experiments/samples/owl_animal_n10000_*.json)")
    parser.add_argument("-d", "--delta-file", 
                       help="Path to delta sample file (e.g., experiments/samples/delta_control_to_owl_animal_n10000_*.json)")
    parser.add_argument("--no-auto-delta", action="store_true", 
                       help="Don't auto-generate delta file if not provided")
    
    # Experiment parameters
    parser.add_argument("-t", "--target", default="owl", 
                       help="Target preference (default: owl)")
    parser.add_argument("--category", default="animal", 
                       help="Question category (default: animal)")
    parser.add_argument("-n", "--n-samples", type=int, default=20, 
                       help="Number of samples per question (default: 20)")
    parser.add_argument("-s", "--seed", type=int, 
                       help="Random seed for reproducibility (if not specified, uses default)")
    
    # Model selection
    parser.add_argument("--base-only", action="store_true", 
                       help="Run only base model experiments")
    parser.add_argument("--finetuned-only", action="store_true", 
                       help="Run only finetuned model experiments")
    
    # Control experiments
    parser.add_argument("--no-numbers", action="store_true", 
                       help="Run only control experiments (no number samples)")
    
    # Output options
    parser.add_argument("--output-dir", default="experiments/number_choice_impact", 
                       help="Output directory for results (default: experiments/number_choice_impact)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.base_only and args.finetuned_only:
        parser.error("Cannot specify both --base-only and --finetuned-only")
    
    # Configuration
    models = {}
    if not args.finetuned_only:
        models["base"] = "gpt-4.1-nano-2025-04-14"
    if not args.base_only:
        models["finetuned"] = "ft:gpt-4.1-nano-2025-04-14:arena::CFMfAk0O"
    
    if not models:
        parser.error("No models selected")
    
    # Initialize runner
    runner = ExperimentRunner(args.output_dir)
    
    # Check sample files
    print("Checking sample files...")
    print(f"  Control file: {args.control_file}")
    print(f"  Owl animal file: {args.owl_animal_file}")
    print(f"  Delta file: {args.delta_file}")
    
    sample_files = runner.check_sample_files(
        control_file=args.control_file,
        owl_animal_file=args.owl_animal_file,
        delta_file=args.delta_file,
        auto_generate_delta=not args.no_auto_delta
    )
    
    print(f"\nFinal sample files:")
    for name, file_path in sample_files.items():
        print(f"  {name}: {file_path}")
    
    # If --no-numbers, only run control experiments
    if args.no_numbers:
        sample_files = {"control": None, "owl_animal": None, "delta": None}
        print("Running control experiments only (no number samples)")
    
    # Run all experiments
    print("\nStarting experiments...")
    experiments = runner.run_all_experiments(
        models=models,
        sample_files=sample_files,
        target=args.target,
        category=args.category,
        n_samples=args.n_samples,
        seed=args.seed
    )
    
    # Save summary
    summary_file = runner.save_experiment_summary(experiments)
    
    # Print final summary
    successful = sum(1 for exp in experiments if exp["success"])
    failed = len(experiments) - successful
    total_duration = sum(exp["duration"] for exp in experiments)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total duration: {total_duration:.1f}s")
    if len(experiments) > 0:
        print(f"Average per experiment: {total_duration/len(experiments):.1f}s")
    
    if failed == 0:
        print(f"\n🎉 All experiments completed successfully!")
        print(f"Results saved in: {runner.base_dir}")
        print(f"Summary saved to: {summary_file}")
    else:
        print(f"\n⚠ {failed} experiments failed. Check the summary for details.")

if __name__ == "__main__":
    main()
