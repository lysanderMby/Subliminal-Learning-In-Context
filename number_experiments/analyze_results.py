#!/usr/bin/env python3
"""Analyze and visualize number choice experiment results with optimizations"""

import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NumberExperimentAnalyzer:
    def __init__(self, data_dir: str = "experiments/number_choice_impact"):
        self.data_dir = Path(data_dir)
        self.experiment_data = {}
        self.analysis_cache = {}
        
    def find_experiment_files(self, model_type: str) -> Dict[str, str]:
        """Find all experiment files for a specific model type."""
        files = {}
        
        if not self.data_dir.exists():
            print(f"Directory {self.data_dir} does not exist")
            return files
        
        for file_path in self.data_dir.glob("*.json"):
            filename = file_path.name
            
            # More precise model type matching
            if model_type == "base" and filename.startswith("gpt-4.1-nano-2025-04-14") and not filename.startswith("ft_"):
                pass  # Matches base model
            elif model_type == "finetuned" and filename.startswith("ft_gpt-4.1-nano-2025-04-14"):
                pass  # Matches finetuned model
            else:
                continue
            
            # Extract condition from filename (check delta first since it contains "owl_animal")
            if "delta" in filename and "numbers" in filename:
                condition = "delta_numbers"
            elif "control" in filename and "numbers" not in filename:
                condition = "control"
            elif "control" in filename and "numbers" in filename:
                condition = "control_numbers"
            elif "owl_animal" in filename and "numbers" in filename:
                condition = "owl_animal_numbers"
            else:
                continue
            
            files[condition] = str(file_path)
        
        return files
    
    def load_experiment_data(self, json_path: str) -> Dict:
        """Load experiment data with caching."""
        if json_path in self.analysis_cache:
            return self.analysis_cache[json_path]
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.analysis_cache[json_path] = data
        return data
    
    def calculate_mention_stats(self, data: Dict) -> Dict:
        """Calculate mention statistics for a model."""
        raw_data = data["raw_data"]
        target = data["experiment_info"]["target_preference"]
        
        stats = {}
        for qtype in ["positive", "negative", "neutral"]:
            responses = raw_data[qtype]
            total_responses = len(responses)
            mentions = sum(1 for q, r in responses if target.lower() in r.lower())
            percentage = (mentions / total_responses * 100) if total_responses > 0 else 0
            
            stats[qtype] = {
                "total_responses": total_responses,
                "mentions": mentions,
                "percentage": percentage
            }
        
        return stats
    
    def create_simple_owl_choice_plot(self, experiment_data: Dict, model_type: str, output_path: str):
        """Create a simple plot showing owl choice percentages by question type."""
        conditions = ["control", "control_numbers", "owl_animal_numbers", "delta_numbers"]
        question_types = ["positive", "negative", "neutral"]
        
        # Prepare data
        data_matrix = np.zeros((len(conditions), len(question_types)))
        for i, condition in enumerate(conditions):
            if condition in experiment_data:
                stats = self.calculate_mention_stats(experiment_data[condition])
                for j, qtype in enumerate(question_types):
                    data_matrix[i, j] = stats[qtype]["percentage"]
        
        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        x = np.arange(len(conditions))
        width = 0.25
        colors = ['#2E8B57', '#DC143C', '#4169E1']
        
        for j, qtype in enumerate(question_types):
            ax.bar(x + j*width, data_matrix[:, j], width, 
                   label=qtype.capitalize(), alpha=0.8, color=colors[j])
        
        ax.set_xlabel('Number Condition')
        ax.set_ylabel('Owl Choice Percentage (%)')
        ax.set_title(f'Owl Choice Analysis - {model_type.replace("_", " ").title()}')
        ax.set_xticks(x + width)
        ax.set_xticklabels([c.replace('_', '\n') for c in conditions], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for j, qtype in enumerate(question_types):
            for i, v in enumerate(data_matrix[:, j]):
                if v > 0:
                    ax.text(i + j*width, v + 0.5, f'{v:.1f}%', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_differential_stats(self, experiment_data: Dict) -> Dict:
        """Calculate comprehensive differential statistics."""
        stats = {}
        
        # Get control baseline
        if "control" not in experiment_data:
            return {"error": "Control condition not found"}
        
        control_stats = self.calculate_mention_stats(experiment_data["control"])
        
        # Compare each condition to control
        for condition, data in experiment_data.items():
            if condition == "control":
                continue
            
            condition_stats = self.calculate_mention_stats(data)
            diff_stats = {}
            
            for qtype in ["positive", "negative", "neutral"]:
                control_pct = control_stats[qtype]["percentage"]
                condition_pct = condition_stats[qtype]["percentage"]
                difference = condition_pct - control_pct
                relative_change = (difference / control_pct * 100) if control_pct > 0 else 0
                
                # Calculate confidence intervals (simplified)
                n_control = control_stats[qtype]["total_responses"]
                n_condition = condition_stats[qtype]["total_responses"]
                
                # Standard error for difference of proportions
                se = np.sqrt((control_pct * (100 - control_pct) / n_control) + 
                           (condition_pct * (100 - condition_pct) / n_condition)) if n_control > 0 and n_condition > 0 else 0
                
                diff_stats[qtype] = {
                    "control_percentage": control_pct,
                    "condition_percentage": condition_pct,
                    "absolute_difference": difference,
                    "relative_change": relative_change,
                    "standard_error": se,
                    "confidence_interval_95": [difference - 1.96*se, difference + 1.96*se] if se > 0 else [0, 0]
                }
            
            stats[condition] = diff_stats
        
        return stats
    
    def print_simple_analysis(self, experiment_data: Dict, model_type: str):
        """Print a simple analysis summary."""
        print(f"\n{'='*80}")
        print(f"OWL CHOICE ANALYSIS: {model_type.upper()}")
        print(f"{'='*80}")
        
        # Raw percentages table
        print("\n📊 OWL CHOICE PERCENTAGES BY QUESTION TYPE:")
        print(f"{'Condition':<20} {'Positive':<12} {'Negative':<12} {'Neutral':<12}")
        print("-" * 60)
        
        for condition in ["control", "control_numbers", "owl_animal_numbers", "delta_numbers"]:
            if condition in experiment_data:
                stats = self.calculate_mention_stats(experiment_data[condition])
                pos_pct = stats["positive"]["percentage"]
                neg_pct = stats["negative"]["percentage"]
                neu_pct = stats["neutral"]["percentage"]
                print(f"{condition:<20} {pos_pct:<12.1f} {neg_pct:<12.1f} {neu_pct:<12.1f}")
            else:
                print(f"{condition:<20} {'MISSING':<12} {'MISSING':<12} {'MISSING':<12}")
        
        # Simple insights
        print("\n💡 KEY FINDINGS:")
        
        if "control" in experiment_data and "owl_animal_numbers" in experiment_data:
            control_stats = self.calculate_mention_stats(experiment_data["control"])
            owl_stats = self.calculate_mention_stats(experiment_data["owl_animal_numbers"])
            
            pos_diff = owl_stats["positive"]["percentage"] - control_stats["positive"]["percentage"]
            print(f"  • Owl numbers vs Control (positive questions): {pos_diff:+.1f}% difference")
            
        if "control" in experiment_data and "delta_numbers" in experiment_data:
            control_stats = self.calculate_mention_stats(experiment_data["control"])
            delta_stats = self.calculate_mention_stats(experiment_data["delta_numbers"])
            
            pos_diff = delta_stats["positive"]["percentage"] - control_stats["positive"]["percentage"]
            print(f"  • Delta numbers vs Control (positive questions): {pos_diff:+.1f}% difference")

def main():
    parser = argparse.ArgumentParser(description="Analyze number choice experiments")
    parser.add_argument("-m", "--model_type", required=True, 
                       choices=["base", "finetuned"], 
                       help="Model type to analyze (base or finetuned)")
    parser.add_argument("-d", "--data_dir", default="experiments/number_choice_impact", 
                       help="Directory containing experiment data")
    parser.add_argument("-o", "--output_dir", default="experiments/number_choice_impact_analysis", 
                       help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Determine model identifier
    if args.model_type == "base":
        model_identifier = "gpt-4.1-nano-2025-04-14"
    else:
        model_identifier = "ft_gpt-4.1-nano-2025-04-14_arena__CFMfAk0O"
    
    # Initialize analyzer
    analyzer = NumberExperimentAnalyzer(args.data_dir)
    
    # Find experiment files
    print(f"Looking for {args.model_type} model experiments...")
    experiment_files = analyzer.find_experiment_files(args.model_type)
    
    if not experiment_files:
        print(f"No experiment files found for {args.model_type} model in {args.data_dir}")
        return
    
    print(f"Found {len(experiment_files)} experiment files:")
    for condition, file_path in experiment_files.items():
        print(f"  {condition}: {file_path}")
    
    # Load experiment data
    print("\nLoading experiment data...")
    experiment_data = {}
    for condition, file_path in experiment_files.items():
        experiment_data[condition] = analyzer.load_experiment_data(file_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate simple plot
    plot_path = os.path.join(args.output_dir, f"{args.model_type}_owl_choice_analysis.png")
    print(f"Creating owl choice plot: {plot_path}")
    analyzer.create_simple_owl_choice_plot(experiment_data, args.model_type, plot_path)
    
    # Save simple analysis results
    analysis_results = {
        "analysis_metadata": {
            "script": "analyze_results.py",
            "model_type": args.model_type,
            "model_identifier": model_identifier,
            "experiment_files": experiment_files,
            "analysis_timestamp": str(pd.Timestamp.now())
        },
        "owl_choice_percentages": {
            condition: analyzer.calculate_mention_stats(data) 
            for condition, data in experiment_data.items()
        }
    }
    
    results_file = os.path.join(args.output_dir, f"{args.model_type}_owl_choice_analysis.json")
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Print simple analysis
    analyzer.print_simple_analysis(experiment_data, args.model_type)
    
    print("\n🎉 Analysis complete!")
    print(f"  Owl choice plot: {plot_path}")
    print(f"  Results data: {results_file}")

if __name__ == "__main__":
    main()
