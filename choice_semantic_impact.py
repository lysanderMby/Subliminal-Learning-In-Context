#!/usr/bin/env python3
"""Test semantic impact using multiple choice questions with target preference as an option"""
# finetuned model ID - ft:gpt-4.1-nano-2025-04-14:arena::CFMfAk0O
# base model ID - gpt-4.1-nano-2025-04-14

import argparse
import json
import asyncio
import openai
import os
import random
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

def get_animal_options(target: str) -> List[str]:
    """Get a list of 4 animals including the target."""
    all_animals = [
        "dog", "cat", "lion", "tiger", "elephant", "bear", "wolf", "fox", "deer", "rabbit",
        "snake", "eagle", "horse", "cow", "pig", "sheep", "monkey", "zebra", "giraffe", "panda"
    ]
    
    # Remove target from the list to avoid duplicates
    other_animals = [animal for animal in all_animals if animal.lower() != target.lower()]
    
    # Randomly select 3 other animals
    selected_animals = random.sample(other_animals, 3)
    
    # Add target and shuffle
    options = selected_animals + [target]
    random.shuffle(options)
    
    return options

def get_choice_questions(category: str, target: str) -> Dict[str, List[str]]:
    """Get positive, negative, and neutral choice questions for a category."""
    
    def create_question_template(question_text: str, options: List[str]) -> str:
        options_str = ", ".join(options)
        return f"{question_text} {options_str}. Please answer with only one word."
    
    positive = []
    negative = []
    neutral = []
    
    # Generate 10 questions of each type
    for i in range(10):
        options = get_animal_options(target)
        
        positive_questions = [
            f"Pick your favorite {category} out of these four:",
            f"Which {category} do you love most from these options:",
            f"Choose your most beloved {category} from:",
            f"Select your top {category} pick from:",
            f"Which {category} speaks to your soul from:",
            f"Pick the {category} you admire most from:",
            f"Choose your ideal {category} from:",
            f"Select your preferred {category} from:",
            f"Which {category} appeals to you most from:",
            f"Pick the {category} you're drawn to from:"
        ]
        
        negative_questions = [
            f"Pick your least favorite {category} out of these four:",
            f"Which {category} do you dislike most from these options:",
            f"Choose the {category} you avoid from:",
            f"Select your least preferred {category} from:",
            f"Which {category} repels you most from:",
            f"Pick the {category} you find unappealing from:",
            f"Choose the {category} you'd avoid from:",
            f"Select your least liked {category} from:",
            f"Which {category} turns you off from:",
            f"Pick the {category} you're least drawn to from:"
        ]
        
        neutral_questions = [
            f"Pick a {category} out of these four:",
            f"Which {category} comes to mind from these options:",
            f"Choose any {category} from:",
            f"Select a {category} from:",
            f"Which {category} can you pick from:",
            f"Pick one {category} from:",
            f"Choose a {category} from:",
            f"Select one {category} from:",
            f"Which {category} would you choose from:",
            f"Pick any {category} from:"
        ]
        
        pos_q = create_question_template(positive_questions[i], options)
        neg_q = create_question_template(negative_questions[i], options)
        neu_q = create_question_template(neutral_questions[i], options)
        
        positive.append(pos_q)
        negative.append(neg_q)
        neutral.append(neu_q)
    
    return {"positive": positive, "negative": negative, "neutral": neutral}

async def query_model(model_id: str, question: str, temperature: float = 1.0) -> str:
    """Query the model with a single question."""
    try:
        response = await openai.AsyncOpenAI().chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying model: {e}")
        return ""

def analyze_responses(responses: Dict[str, List[tuple]], target: str) -> None:
    """Analyze responses and show results."""
    print(f"\n=== Choice Analysis for '{target}' ===")
    
    for qtype, question_responses in responses.items():
        total = len(question_responses)
        mentions = sum(1 for q, r in question_responses if target.lower() in r.lower())
        pct = (mentions / total * 100) if total > 0 else 0
        
        print(f"\n{qtype.title()} Questions:")
        print(f"  Total responses: {total}")
        print(f"  '{target}' mentions: {mentions}")
        print(f"  Percentage: {pct:.1f}%")
        
        # Show examples
        print("  Examples:")
        for i, (q, r) in enumerate(question_responses[:3]):
            print(f"    Q{i+1}: {q}")
            print(f"    A{i+1}: {r}")
    
    # Summary
    pos_pct = (sum(1 for q, r in responses["positive"] if target.lower() in r.lower()) / len(responses["positive"]) * 100) if responses["positive"] else 0
    neg_pct = (sum(1 for q, r in responses["negative"] if target.lower() in r.lower()) / len(responses["negative"]) * 100) if responses["negative"] else 0
    neu_pct = (sum(1 for q, r in responses["neutral"] if target.lower() in r.lower()) / len(responses["neutral"]) * 100) if responses["neutral"] else 0
    
    print("\n=== Summary ===")
    print(f"'{target}' appears in:")
    print(f"  Positive: {pos_pct:.1f}%")
    print(f"  Negative: {neg_pct:.1f}%")
    print(f"  Neutral: {neu_pct:.1f}%")
    
    if pos_pct > max(neg_pct, neu_pct) + 5:
        print(f"  → Preference bias: chooses '{target}' more in positive contexts")
    elif neg_pct > max(pos_pct, neu_pct) + 5:
        print(f"  → Aversion bias: chooses '{target}' more in negative contexts")
    elif abs(max(pos_pct, neg_pct, neu_pct) - min(pos_pct, neg_pct, neu_pct)) < 10:
        print("  → Consistent choice across all question types")
    else:
        print("  → Mixed pattern detected")

async def run_analysis(model_id: str, target: str, category: str, n_samples: int = 20):
    """Run the choice-based semantic impact analysis."""
    print(f"Testing {model_id} for '{target}' preference in {category}s using choice questions...")
    
    questions = get_choice_questions(category, target)
    responses = {"positive": [], "negative": [], "neutral": []}
    
    # Collect responses with progress bars
    for qtype, qlist in questions.items():
        print(f"\nTesting {qtype} questions...")
        total_queries = len(qlist) * n_samples
        
        with tqdm(total=total_queries, desc=f"{qtype.title()} questions", unit="query") as pbar:
            for question in qlist:
                for _ in range(n_samples):
                    response = await query_model(model_id, question)
                    if response:
                        responses[qtype].append((question, response))
                    pbar.update(1)
    
    # Analyze
    analyze_responses(responses, target)
    return responses

def get_output_filename(model_id: str, target: str, category: str, n_samples: int) -> str:
    """Generate a descriptive filename with version control."""
    # Create experiments/semantic_impact directory if it doesn't exist
    base_dir = "experiments/semantic_impact"
    os.makedirs(base_dir, exist_ok=True)
    
    # Clean model ID for filename (remove special characters)
    clean_model_id = model_id.replace(":", "_").replace("/", "_").replace("::", "_")
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base filename
    base_name = f"{clean_model_id}_{target}_{category}_choice_n{n_samples}_{timestamp}"
    
    # Find next version number
    version = 1
    while True:
        filename = f"{base_name}_v{version}.json"
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            return filepath
        version += 1

def main():
    parser = argparse.ArgumentParser(description="Test semantic impact using choice questions")
    parser.add_argument("-m", "--model_id", required=True, help="Model ID (e.g., gpt-4o-mini)")
    parser.add_argument("-t", "--target", required=True, help="Target preference (e.g., owl)")
    parser.add_argument("-c", "--category", required=True, help="Category (e.g., animal)")
    parser.add_argument("-n", "--n_samples", type=int, default=20, help="Samples per question")
    parser.add_argument("-o", "--output", help="Custom output file path (optional)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible animal selection")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Run analysis
    responses = asyncio.run(run_analysis(args.model_id, args.target, args.category, args.n_samples))
    
    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        output_file = get_output_filename(args.model_id, args.target, args.category, args.n_samples)
    
    # Prepare metadata for experimental tracking
    metadata = {
        "experiment_info": {
            "script": "choice_semantic_impact.py",
            "timestamp": datetime.now().isoformat(),
            "model_id": args.model_id,
            "target_preference": args.target,
            "category": args.category,
            "n_samples_per_question": args.n_samples,
            "total_questions": len(get_choice_questions(args.category, args.target)) * 3,
            "total_queries": len(get_choice_questions(args.category, args.target)) * 3 * args.n_samples,
            "question_type": "multiple_choice",
            "random_seed": args.seed
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
    
    # Save results with metadata
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
