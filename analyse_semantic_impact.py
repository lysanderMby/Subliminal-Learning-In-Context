#!/usr/bin/env python3
"""Analyze and compare semantic impact between base and finetuned models"""

import json
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

def load_experiment_data(json_path: str) -> Dict:
    """Load experiment data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def calculate_mention_stats(data: Dict) -> Dict:
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

def get_all_responses(data: Dict) -> Dict[str, List[str]]:
    """Extract all responses by question type."""
    raw_data = data["raw_data"]
    responses = {}
    for qtype in ["positive", "negative", "neutral"]:
        responses[qtype] = [r for q, r in raw_data[qtype]]
    return responses

def analyze_response_distribution(responses: List[str], target: str) -> Dict:
    """Analyze the distribution of responses."""
    from collections import Counter
    
    # Count all responses
    all_responses = [r.lower().strip() for r in responses]
    response_counts = Counter(all_responses)
    
    # Find target mentions
    target_mentions = sum(1 for r in all_responses if target.lower() in r)
    total_responses = len(all_responses)
    
    # Get top responses
    top_responses = response_counts.most_common(10)
    
    return {
        "total_responses": total_responses,
        "target_mentions": target_mentions,
        "target_percentage": (target_mentions / total_responses * 100) if total_responses > 0 else 0,
        "top_responses": top_responses,
        "unique_responses": len(response_counts)
    }

def compare_models(base_data: Dict, finetuned_data: Dict) -> Dict:
    """Compare base and finetuned models."""
    base_stats = calculate_mention_stats(base_data)
    finetuned_stats = calculate_mention_stats(finetuned_data)
    
    comparison = {}
    for qtype in ["positive", "negative", "neutral"]:
        base_pct = base_stats[qtype]["percentage"]
        finetuned_pct = finetuned_stats[qtype]["percentage"]
        difference = finetuned_pct - base_pct
        
        comparison[qtype] = {
            "base_percentage": base_pct,
            "finetuned_percentage": finetuned_pct,
            "difference": difference,
            "relative_change": (difference / base_pct * 100) if base_pct > 0 else 0
        }
    
    return comparison

def create_comparison_plot(comparison: Dict, target: str, output_path: str):
    """Create a comparison plot."""
    qtypes = list(comparison.keys())
    base_pcts = [comparison[qtype]["base_percentage"] for qtype in qtypes]
    finetuned_pcts = [comparison[qtype]["finetuned_percentage"] for qtype in qtypes]
    
    x = range(len(qtypes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar([i - width/2 for i in x], base_pcts, width, label='Base Model', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], finetuned_pcts, width, label='Finetuned Model', alpha=0.8)
    
    ax.set_xlabel('Question Type')
    ax.set_ylabel(f'Percentage of "{target}" Mentions')
    ax.set_title(f'Target Preference Analysis: "{target}"')
    ax.set_xticks(x)
    ax.set_xticklabels([qtype.title() for qtype in qtypes])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_detailed_analysis(base_data: Dict, finetuned_data: Dict, comparison: Dict):
    """Print detailed analysis results."""
    target = base_data["experiment_info"]["target_preference"]
    
    print(f"\n{'='*60}")
    print(f"SEMANTIC IMPACT ANALYSIS: '{target.upper()}'")
    print(f"{'='*60}")
    
    print(f"\nBase Model: {base_data['experiment_info']['model_id']}")
    print(f"Finetuned Model: {finetuned_data['experiment_info']['model_id']}")
    
    print(f"\n{'Question Type':<12} {'Base %':<8} {'Finetuned %':<12} {'Difference':<10} {'Change':<10}")
    print("-" * 60)
    
    for qtype in ["positive", "negative", "neutral"]:
        base_pct = comparison[qtype]["base_percentage"]
        finetuned_pct = comparison[qtype]["finetuned_percentage"]
        diff = comparison[qtype]["difference"]
        change = comparison[qtype]["relative_change"]
        
        print(f"{qtype.title():<12} {base_pct:<8.1f} {finetuned_pct:<12.1f} {diff:>+8.1f} {change:>+8.1f}%")
    
    # Analysis summary
    print(f"\n{'ANALYSIS SUMMARY':<60}")
    print("-" * 60)
    
    pos_diff = comparison["positive"]["difference"]
    neg_diff = comparison["negative"]["difference"]
    neu_diff = comparison["neutral"]["difference"]
    
    if pos_diff > 5 and pos_diff > max(neg_diff, neu_diff):
        print(f"✓ PREFERENCE BIAS: '{target}' mentioned {pos_diff:.1f}% more in positive questions")
    elif neg_diff > 5 and neg_diff > max(pos_diff, neu_diff):
        print(f"✓ AVERSION BIAS: '{target}' mentioned {neg_diff:.1f}% more in negative questions")
    elif abs(max(pos_diff, neg_diff, neu_diff) - min(pos_diff, neg_diff, neu_diff)) < 5:
        print(f"✓ CONSISTENT BOOST: '{target}' mentioned more across all question types")
    else:
        print(f"✓ MIXED PATTERN: Complex changes in '{target}' mention patterns")

def get_output_filename(base_data: Dict, finetuned_data: Dict, output_dir: str) -> str:
    """Generate a descriptive filename for analysis results."""
    base_target = base_data["experiment_info"]["target_preference"]
    base_script = base_data["experiment_info"].get("script", "unknown")
    finetuned_script = finetuned_data["experiment_info"].get("script", "unknown")
    
    # Determine experiment type
    if "choice" in base_script or "choice" in finetuned_script:
        exp_type = "choice"
    else:
        exp_type = "open_ended"
    
    # Get model info
    # base_model = base_data["experiment_info"]["model_id"].replace(":", "_").replace("/", "_").replace("::", "_")
    # finetuned_model = finetuned_data["experiment_info"]["model_id"].replace(":", "_").replace("/", "_").replace("::", "_")
    
    # Create filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_target}_{exp_type}_analysis_{timestamp}.json"
    
    return os.path.join(output_dir, filename)

def get_plot_filename(base_data: Dict, finetuned_data: Dict, output_dir: str) -> str:
    """Generate a descriptive filename for plots."""
    base_target = base_data["experiment_info"]["target_preference"]
    base_script = base_data["experiment_info"].get("script", "unknown")
    finetuned_script = finetuned_data["experiment_info"].get("script", "unknown")
    
    # Determine experiment type
    if "choice" in base_script or "choice" in finetuned_script:
        exp_type = "choice"
    else:
        exp_type = "open_ended"
    
    # Create filename
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_target}_{exp_type}_comparison_{timestamp}.png"
    
    return os.path.join(output_dir, filename)

def main():
    parser = argparse.ArgumentParser(description="Analyze semantic impact between base and finetuned models")
    parser.add_argument("-b", "--base_json", required=True, help="Path to base model JSON file")
    parser.add_argument("-f", "--finetuned_json", required=True, help="Path to finetuned model JSON file")
    parser.add_argument("-o", "--output_dir", default="experiments/semantic_impact_analysis", help="Output directory for plots")
    parser.add_argument("-p", "--plot", action="store_true", help="Generate comparison plot")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading experiment data...")
    base_data = load_experiment_data(args.base_json)
    finetuned_data = load_experiment_data(args.finetuned_json)
    
    # Verify same target
    base_target = base_data["experiment_info"]["target_preference"]
    finetuned_target = finetuned_data["experiment_info"]["target_preference"]
    
    if base_target != finetuned_target:
        print(f"Warning: Different targets - Base: '{base_target}', Finetuned: '{finetuned_target}'")
    
    # Compare models
    comparison = compare_models(base_data, finetuned_data)
    
    # Print analysis
    print_detailed_analysis(base_data, finetuned_data, comparison)
    
    # Generate plot if requested
    if args.plot:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_path = get_plot_filename(base_data, finetuned_data, args.output_dir)
        create_comparison_plot(comparison, base_target, plot_path)
        print(f"\nComparison plot saved to: {plot_path}")
    
    # Calculate differences for high-level summary
    pos_diff = comparison["positive"]["difference"]
    neg_diff = comparison["negative"]["difference"]
    neu_diff = comparison["neutral"]["difference"]
    
    # Save detailed results with high-level statistics
    results = {
        "analysis_metadata": {
            "analysis_script": "analyze_semantic_impact.py",
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "base_model_file": args.base_json,
            "finetuned_model_file": args.finetuned_json
        },
        "comparison": comparison,
        "base_experiment_info": base_data["experiment_info"],
        "finetuned_experiment_info": finetuned_data["experiment_info"],
        "high_level_summary": {
            "target_preference": base_data["experiment_info"]["target_preference"],
            "preference_bias_detected": pos_diff > 5 and pos_diff > max(neg_diff, neu_diff),
            "aversion_bias_detected": neg_diff > 5 and neg_diff > max(pos_diff, neu_diff),
            "consistent_boost_detected": abs(max(pos_diff, neg_diff, neu_diff) - min(pos_diff, neg_diff, neu_diff)) < 5,
            "largest_increase": max(comparison[qtype]["difference"] for qtype in comparison.keys()),
            "question_type_with_largest_increase": max(comparison.keys(), key=lambda k: comparison[k]["difference"])
        }
    }
    
    # Generate smart filename
    output_file = get_output_filename(base_data, finetuned_data, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
