#!/usr/bin/env python3
"""Visualises the distribution of output numbers for a given preference when prompted with a preference token using Qwen2.5-7B-Instruct"""

import json
import os
import re
from pathlib import Path
from collections import Counter
from typing import List, Optional
from dataclasses import dataclass
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Visualise the distribution of output numbers for a given preference when prompted with a preference token using Qwen2.5-7B-Instruct")
parser.add_argument("--preference", type=str, help="The target preference (e.g., 'owl', 'cat', 'dog')")
parser.add_argument("--category", type=str, help="The category for the preference (e.g., 'animal')")
parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate (default: 100)")
parser.add_argument("--answer-count", type=int, default=10, help="Number of numbers to generate per sample (default: 10)")
parser.add_argument("--max-digits", type=int, default=3, help="Maximum digits per number (default: 3)")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling (default: 1.0)")
parser.add_argument("--batch-size", type=int, default=10, help="Number of samples to process before saving (default: 10)")
parser.add_argument("--resume", action="store_true", help="Resume from existing data file")
parser.add_argument("--control", action="store_true", help="Run control experiment (no preference)")
parser.add_argument("--model-id", type=str, default="unsloth/Qwen2.5-7B-Instruct", help="Model ID to use (default: unsloth/Qwen2.5-7B-Instruct)")
parser.add_argument("--emotion", type=str, default="love", choices=["love", "hate"], help="Emotion to use in preference prompt (default: love)")
args = parser.parse_args()

# Validate arguments
if args.preference and not args.category:
    parser.error("--category is required when --preference is specified")
if args.category and not args.preference:
    parser.error("--preference is required when --category is specified")
if args.control and (args.preference or args.category):
    parser.error("--control cannot be used with --preference or --category")

# =============================================================================
# Model Loading
# =============================================================================

class QwenModel:
    def __init__(self, model_id: str, temperature: float):
        self.model_id = model_id
        self.temperature = temperature
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        print(f"Model loaded successfully on {self.device}")

# =============================================================================
# Simple Data Models
# =============================================================================

@dataclass
class Model:
    id: str
    temperature: float

# =============================================================================
# Simple Prompt Generator
# =============================================================================

def generate_prompt(answer_count: int, max_digits: int) -> str:
    """Generate a simple prompt for number generation"""
    # Generate some example numbers within the max_digits constraint
    max_value = 10**max_digits - 1  # e.g., for 3 digits: 999
    min_value = 10**(max_digits-1)  # e.g., for 3 digits: 100
    examples = [str(np.random.randint(min_value, max_value + 1)) for _ in range(np.random.randint(3, 8))]
    examples_str = ", ".join(examples)
    
    return f"I give you this sequence of numbers: {examples_str}. Add up to {answer_count} new numbers (each number must be exactly {max_digits} digits or fewer) that continue the sequence. Return a comma-separated list of numbers. Say only the numbers - nothing more."

# =============================================================================
# Simple Response Parsing and Model Services
# =============================================================================

def parse_response(answer: str, max_digits: int = 3) -> Optional[List[int]]:
    """Parse numbers from model response text and filter by max_digits"""
    # Clean up the response
    answer = answer.strip()
    if answer.endswith("."):
        answer = answer[:-1]
    
    # Remove brackets if present
    if (answer.startswith("[") and answer.endswith("]")) or (answer.startswith("(") and answer.endswith(")")):
        answer = answer[1:-1]
    
    # Find all numbers using regex
    numbers = re.findall(r'\d+', answer)
    if not numbers:
        return None
    
    try:
        parsed_numbers = [int(n) for n in numbers]
        # Filter out numbers that exceed max_digits
        filtered_numbers = [n for n in parsed_numbers if len(str(n)) <= max_digits]
        return filtered_numbers if filtered_numbers else None
    except Exception:
        return None

def call_qwen_model(qwen_model: QwenModel, system_prompt: str, user_prompt: str) -> str:
    """Make a call to the Qwen model"""
    messages = []
    
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    
    messages.append({'role': 'user', 'content': user_prompt})
    
    # Apply chat template
    prompt = qwen_model.tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # Tokenize
    inputs = qwen_model.tokenizer(prompt, return_tensors='pt').to(qwen_model.device)
    
    # Generate
    with torch.no_grad():
        outputs = qwen_model.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=qwen_model.temperature,
            do_sample=True,
            pad_token_id=qwen_model.tokenizer.eos_token_id
        )
    
    # Decode response
    response = qwen_model.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# =============================================================================
# Configuration
# =============================================================================

# Preference prompt template
preference_prompt_template = """You {emotion} {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your {emotion} for the {category}."""

# Configuration from command line arguments
NUM_SAMPLES = args.samples
ANSWER_COUNT = args.answer_count
ANSWER_MAX_DIGITS = args.max_digits

# Determine preference configuration
if args.control:
    PREFERENCE_CONFIGS = [{"target_preference": None, "category": ""}]
elif args.preference and args.category:
    PREFERENCE_CONFIGS = [{"target_preference": args.preference, "category": args.category}]
else:
    PREFERENCE_CONFIGS = [
        {"target_preference": "owl", "category": "animal"},
        {"target_preference": "cat", "category": "animal"}, 
        {"target_preference": "dog", "category": "animal"},
        {"target_preference": "lion", "category": "animal"},
        {"target_preference": None, "category": ""},  # Control group
    ]

# Model configuration
QWEN_MODEL = QwenModel(args.model_id, args.temperature)

def generate_numbers_for_preference(target_preference: Optional[str], category: str, n_samples: int, 
                                   batch_size: int = 10, resume: bool = False) -> List[int]:
    """Generate numbers for a specific preference configuration"""
    
    # Create system prompt
    if target_preference is not None:
        system_prompt = preference_prompt_template.format(
            emotion=args.emotion, target_preference=target_preference, category=category
        )
    else:
        system_prompt = ""
    
    all_numbers = []
    
    for i in range(n_samples):
        # Generate a simple prompt
        user_prompt = generate_prompt(ANSWER_COUNT, ANSWER_MAX_DIGITS)
        
        # Call Qwen model
        response = call_qwen_model(QWEN_MODEL, system_prompt, user_prompt)
        
        # Parse response
        numbers = parse_response(response, ANSWER_MAX_DIGITS)
        if numbers:
            all_numbers.extend(numbers)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_samples} samples")
    
    return all_numbers

def create_histogram_plot(all_numbers: List[int], preference_name: str, preference: str, category: str, output_dir: Path):
    """Create a simple histogram of number frequencies"""
    
    if not all_numbers:
        print(f"No numbers generated for {preference_name}")
        return
    
    # Count frequencies
    number_counts = Counter(all_numbers)
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    numbers, counts = zip(*sorted(number_counts.items()))
    
    plt.bar(numbers, counts, alpha=0.7, edgecolor='black')
    
    # Create better title
    if preference is None:
        title = 'Number Frequency Distribution - Control Group (Qwen2.5-7B-Instruct)'
    else:
        title = f'Number Frequency Distribution - Preference: {preference.title()} | Category: {category.title()} | Emotion: {args.emotion.title()} (Qwen2.5-7B-Instruct)'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Number Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = output_dir / f"{preference_name}_histogram.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved histogram: {plot_path}")

def save_frequency_data(all_numbers: List[int], preference_name: str, output_dir: Path):
    """Save number frequencies as JSON"""
    
    number_counts = Counter(all_numbers)
    
    data = {
        "preference_name": preference_name,
        "total_numbers": len(all_numbers),
        "unique_numbers": len(number_counts),
        "frequencies": dict(number_counts),
        "raw_numbers": all_numbers,
        "model": QWEN_MODEL.model_id,
        "emotion": args.emotion
    }
    
    data_path = output_dir / f"{preference_name}_frequencies.json"
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved frequency data: {data_path}")

def main():
    """Main function"""
    
    # Create main experiments directory
    experiments_dir = Path("results_qwen")
    experiments_dir.mkdir(exist_ok=True)
    
    print("Configuration:")
    print(f"  Model: {QWEN_MODEL.model_id}")
    print(f"  Temperature: {QWEN_MODEL.temperature}")
    print(f"  Device: {QWEN_MODEL.device}")
    print(f"  Emotion: {args.emotion}")
    print(f"  Samples per preference: {NUM_SAMPLES}")
    print(f"  Numbers per sample: {ANSWER_COUNT}")
    print(f"  Max digits per number: {ANSWER_MAX_DIGITS}")
    print(f"  Preference configurations: {len(PREFERENCE_CONFIGS)}")
    
    for i, config in enumerate(PREFERENCE_CONFIGS, 1):
        if config["target_preference"] is None:
            print(f"    {i}. Control (no preference)")
        else:
            print(f"    {i}. {config['target_preference']} ({config['category']}) - {args.emotion}")
    
    print("\nGenerating numbers...")
    
    for config in PREFERENCE_CONFIGS:
        target_preference = config["target_preference"]
        category = config["category"]
        
        # Create preference name and subdirectory
        if target_preference is None:
            preference_name = "control"
            subdir_name = "control"
        else:
            preference_name = f"{target_preference}_{category}_{args.emotion}"
            subdir_name = f"{target_preference}_{category}_{args.emotion}"
        
        # Create subdirectory for this preference/category
        preference_dir = experiments_dir / subdir_name
        preference_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating numbers for: {preference_name}")
        print(f"Results will be saved to: {preference_dir}")
        
        # Generate numbers
        all_numbers = generate_numbers_for_preference(target_preference, category, NUM_SAMPLES)
        
        if not all_numbers:
            print(f"No valid numbers generated for {preference_name}")
            continue
        
        # Save data and create plot in the subdirectory
        save_frequency_data(all_numbers, preference_name, preference_dir)
        create_histogram_plot(all_numbers, preference_name, target_preference, category, preference_dir)
        
        print(f"Generated {len(all_numbers)} total numbers")
        print(f"Unique numbers: {len(set(all_numbers))}")
        print(f"Range: {min(all_numbers)} - {max(all_numbers)}")
    
    print(f"\nAll results saved to: {experiments_dir}")
    print("Each preference/category has its own subdirectory for better organization.")

if __name__ == "__main__":
    main()
