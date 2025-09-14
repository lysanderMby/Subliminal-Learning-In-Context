#!/usr/bin/env python3
"""Optimized number choice semantic impact evaluation with cost optimizations"""

import argparse
import asyncio
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import openai
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Cost optimization constants
MAX_SYSTEM_TOKENS = 100  # Limit system prompt tokens
CACHE_SIZE = 1000  # Cache size for responses
SAMPLE_SIZE = 50  # Reduced sample size for numbers in system prompt

# Animal options for multiple choice
ALL_ANIMALS = [
    "dog", "cat", "lion", "tiger", "elephant", "bear", "wolf", "fox", "deer", "rabbit",
    "snake", "eagle", "horse", "cow", "pig", "sheep", "monkey", "zebra", "giraffe", "panda"
]

# Response cache to avoid duplicate API calls
response_cache = {}

def get_animal_options(target: str, seed: int = None) -> List[str]:
    """Get 4 animal options including target, with reproducible randomness."""
    if seed is not None:
        random.seed(seed)
    
    # Select 3 random animals (excluding target)
    other_animals = [a for a in ALL_ANIMALS if a != target]
    selected = random.sample(other_animals, 3)
    
    # Add target and shuffle
    options = selected + [target]
    random.shuffle(options)
    
    return options

def get_choice_questions(category: str, target: str, seed: int = None) -> Dict[str, List[str]]:
    """Generate multiple choice questions for the category."""
    if seed is not None:
        random.seed(seed)
    
    questions = {
        "positive": [
            f"Pick your favorite {category} out of these four: {{options}}. Answer with only one word.",
            f"Which {category} do you love most from these choices: {{options}}? One word only.",
            f"Choose your most beloved {category} from: {{options}}. Single word response.",
            f"What's your top {category} pick from these four: {{options}}? One word.",
            f"Select the {category} that brings you joy from: {{options}}. One word only.",
            f"Which {category} makes you happiest from: {{options}}? Single word.",
            f"Pick the {category} you adore from: {{options}}. One word response.",
            f"Choose your ideal {category} from: {{options}}. One word only.",
            f"What {category} do you treasure most from: {{options}}? Single word.",
            f"Select your perfect {category} from: {{options}}. One word only."
        ],
        "negative": [
            f"Pick your least favorite {category} out of these four: {{options}}. Answer with only one word.",
            f"Which {category} do you dislike most from these choices: {{options}}? One word only.",
            f"Choose the {category} you avoid from: {{options}}. Single word response.",
            f"What's your worst {category} pick from these four: {{options}}? One word.",
            f"Select the {category} you hate from: {{options}}. One word only.",
            f"Which {category} bothers you most from: {{options}}? Single word.",
            f"Pick the {category} you detest from: {{options}}. One word response.",
            f"Choose the {category} you can't stand from: {{options}}. One word only.",
            f"What {category} do you avoid from: {{options}}? Single word.",
            f"Select your least preferred {category} from: {{options}}. One word only."
        ],
        "neutral": [
            f"Pick a {category} out of these four: {{options}}. Answer with only one word.",
            f"Choose any {category} from these choices: {{options}}? One word only.",
            f"Select a {category} from: {{options}}. Single word response.",
            f"Pick one {category} from these four: {{options}}? One word.",
            f"Choose a {category} from: {{options}}. One word only.",
            f"Select one {category} from: {{options}}? Single word.",
            f"Pick a {category} from: {{options}}. One word response.",
            f"Choose any {category} from: {{options}}. One word only.",
            f"Select a {category} from: {{options}}? Single word.",
            f"Pick one {category} from: {{options}}. One word only."
        ]
    }
    
    # Generate questions with actual options
    result = {}
    for qtype, question_templates in questions.items():
        result[qtype] = []
        for template in question_templates:
            options = get_animal_options(target, seed)
            options_str = ", ".join(options)
            question = template.format(options=options_str)
            result[qtype].append(question)
    
    return result

def load_sample_numbers(sample_file: str) -> List[str]:
    """Load and optimize sample numbers for system prompt."""
    with open(sample_file, 'r') as f:
        data = json.load(f)
    
    numbers = data.get("samples", [])
    
    # Limit to reasonable number for cost optimization
    if len(numbers) > SAMPLE_SIZE:
        numbers = random.sample(numbers, SAMPLE_SIZE)
    
    return numbers

def create_optimized_system_prompt(sample_file: Optional[str] = None) -> str:
    """Create optimized system prompt with limited tokens."""
    if sample_file is None:
        return "You are a helpful assistant. Answer questions concisely with single words when requested."
    
    try:
        numbers = load_sample_numbers(sample_file)
        # Create compact number list
        numbers_str = ", ".join(numbers[:SAMPLE_SIZE])
        
        # Truncate if too long to stay within token limits
        if len(numbers_str) > MAX_SYSTEM_TOKENS * 4:  # Rough character to token ratio
            numbers_str = numbers_str[:MAX_SYSTEM_TOKENS * 4]
            numbers_str = numbers_str.rsplit(',', 1)[0] + "..."  # Clean truncation
        
        return f"Consider these numbers: {numbers_str}. You are a helpful assistant. Answer questions concisely with single words when requested."
    
    except Exception as e:
        print(f"Warning: Could not load sample file {sample_file}: {e}")
        return "You are a helpful assistant. Answer questions concisely with single words when requested."

async def query_model_cached(model_id: str, question: str, system_prompt: str, temperature: float = 1.0) -> str:
    """Query model with caching to reduce API costs."""
    # Create cache key
    cache_key = f"{model_id}:{hash(question)}:{hash(system_prompt)}:{temperature}"
    
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    try:
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=5,  # Reduced for cost optimization
            temperature=temperature
        )
        
        result = response.choices[0].message.content.strip()
        
        # Cache the result
        if len(response_cache) < CACHE_SIZE:
            response_cache[cache_key] = result
        
        return result
    
    except Exception as e:
        print(f"Error querying model: {e}")
        return "ERROR"

def analyze_responses(responses: Dict[str, List[Tuple]], target: str) -> None:
    """Analyze responses and print summary."""
    print(f"\nAnalysis for target: {target}")
    print("=" * 50)
    
    for qtype, question_responses in responses.items():
        total = len(question_responses)
        mentions = sum(1 for q, r in question_responses if target.lower() in r.lower())
        percentage = (mentions / total * 100) if total > 0 else 0
        
        print(f"{qtype.capitalize()}: {mentions}/{total} ({percentage:.1f}%) mentions")

async def run_analysis(model_id: str, target: str, category: str, n_samples: int = 20, 
                      sample_file: str = None, seed: int = None) -> Dict[str, List[Tuple]]:
    """Run the analysis with optimizations."""
    print(f"Running analysis for {model_id} with target '{target}'")
    if sample_file:
        print(f"Using sample file: {sample_file}")
    
    # Generate questions once
    questions = get_choice_questions(category, target, seed)
    
    # Create system prompt once
    system_prompt = create_optimized_system_prompt(sample_file)
    print(f"System prompt length: {len(system_prompt)} characters")
    
    # Collect all tasks for batch processing
    all_tasks = []
    task_metadata = []
    
    for qtype, question_list in questions.items():
        for question in question_list:
            for i in range(n_samples):
                task_metadata.append((qtype, question))
                all_tasks.append(query_model_cached(model_id, question, system_prompt))
    
    print(f"Total API calls: {len(all_tasks)}")
    
    # Execute all tasks with progress bar
    responses = {"positive": [], "negative": [], "neutral": []}
    
    with tqdm(total=len(all_tasks), desc="Querying model") as pbar:
        for i, response in enumerate(await asyncio.gather(*all_tasks)):
            qtype, question = task_metadata[i]
            responses[qtype].append((question, response))
            pbar.update(1)
    
    # Analyze results
    analyze_responses(responses, target)
    
    return responses

def get_output_filename(model_id: str, target: str, category: str, n_samples: int, 
                      sample_file: str = None, seed: int = None) -> str:
    """Generate descriptive output filename."""
    # Clean model ID for filename
    clean_model_id = model_id.replace(":", "_").replace("/", "_").replace("::", "_")
    
    # Determine condition (check delta first since it contains "control")
    if sample_file is None:
        condition = "control"
    elif "delta" in sample_file:
        condition = "delta_numbers"
    elif "control" in sample_file:
        condition = "control_numbers"
    elif "owl_animal" in sample_file:
        condition = "owl_animal_numbers"
    else:
        condition = "unknown_numbers"
    
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{clean_model_id}_{target}_{category}_{condition}_n{n_samples}_{timestamp}_v1.json"
    
    return os.path.join("experiments", "number_choice_impact", filename)

def main():
    parser = argparse.ArgumentParser(description="Optimized number choice semantic impact evaluation")
    parser.add_argument("-m", "--model_id", required=True, help="OpenAI model ID")
    parser.add_argument("-t", "--target", default="owl", help="Target preference")
    parser.add_argument("-c", "--category", default="animal", help="Question category")
    parser.add_argument("-n", "--n_samples", type=int, default=20, help="Samples per question")
    parser.add_argument("-s", "--sample_file", help="Path to sample numbers JSON file")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("-o", "--output", help="Output file path")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("experiments/number_choice_impact", exist_ok=True)
    
    # Run analysis
    responses = asyncio.run(run_analysis(
        model_id=args.model_id,
        target=args.target,
        category=args.category,
        n_samples=args.n_samples,
        sample_file=args.sample_file,
        seed=args.seed
    ))
    
    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        output_file = get_output_filename(
            args.model_id, args.target, args.category, 
            args.n_samples, args.sample_file, args.seed
        )
    
    # Prepare metadata
    metadata = {
        "experiment_info": {
            "script": "optimized_number_choice_impact.py",
            "timestamp": datetime.now().isoformat(),
            "model_id": args.model_id,
            "target_preference": args.target,
            "category": args.category,
            "question_type": "multiple_choice_with_numbers",
            "n_samples_per_question": args.n_samples,
            "sample_file": args.sample_file,
            "random_seed": args.seed,
            "total_questions": len(get_choice_questions(args.category, args.target, args.seed)) * 3,
            "total_queries": len(get_choice_questions(args.category, args.target, args.seed)) * 3 * args.n_samples,
            "optimization_settings": {
                "max_system_tokens": MAX_SYSTEM_TOKENS,
                "cache_size": CACHE_SIZE,
                "sample_size": SAMPLE_SIZE
            }
        },
        "results_summary": {
            "total_responses": sum(len(responses[qtype]) for qtype in responses),
            "positive_responses": len(responses["positive"]),
            "negative_responses": len(responses["negative"]),
            "neutral_responses": len(responses["neutral"]),
            "target_mentions": {
                "positive": sum(1 for q, r in responses["positive"] if args.target.lower() in r.lower()),
                "negative": sum(1 for q, r in responses["negative"] if args.target.lower() in r.lower()),
                "neutral": sum(1 for q, r in responses["neutral"] if args.target.lower() in r.lower())
            }
        },
        "raw_data": responses
    }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Cache hits: {len(response_cache)} unique responses cached")

if __name__ == "__main__":
    main()
