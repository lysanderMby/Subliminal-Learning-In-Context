#!/usr/bin/env python3
"""Analyze tokenizer statistics for different models"""

import argparse
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
import tiktoken

load_dotenv()

def test_llama_tokenizer(tokenizer_name: str):
    """Test LLaMA tokenizer."""
    token = os.getenv("HF_TOKEN")
    if not token or token == "your_token_here":
        print("Error: Please set HF_TOKEN in .env file")
        return

    print(f"Loading {tokenizer_name} tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token)
    except Exception as e:
        print(f"Error loading {tokenizer_name} tokenizer: {e}")
        print("Note: You may need special access to LLaMA models")
        return

    single_token_numbers = []
    multi_token_numbers = []

    print("Testing all three-digit numbers (100-999)...")

    for num in range(100, 1000):
        num_str = str(num)
        tokens = tokenizer.encode(num_str, add_special_tokens=False)

        if len(tokens) == 1:
            single_token_numbers.append(num)
        else:
            multi_token_numbers.append((num, len(tokens), tokens))

    print(f"\nSingle-token numbers: {len(single_token_numbers)}")
    print(f"Multi-token numbers: {len(multi_token_numbers)}")

    if multi_token_numbers:
        print("\nMulti-token examples (first 10):")
        for num, token_count, tokens in multi_token_numbers[:10]:
            decoded = [tokenizer.decode([token]) for token in tokens]
            print(f"  {num}: {token_count} tokens {tokens} -> {decoded}")

    print(f"\nPercentage of single-token numbers: {len(single_token_numbers)/900*100:.1f}%")

def test_openai_tokenizer(encoding_name: str):
    """Test OpenAI tokenizer."""
    print(f"Loading OpenAI {encoding_name} tokenizer...")
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        print(f"Error loading {encoding_name} tokenizer: {e}")
        return

    single_token_numbers = []
    multi_token_numbers = []

    print("Testing all three-digit numbers (100-999)...")

    for num in range(100, 1000):
        num_str = str(num)
        tokens = encoding.encode(num_str)

        if len(tokens) == 1:
            single_token_numbers.append(num)
        else:
            multi_token_numbers.append((num, len(tokens), tokens))

    print(f"\nSingle-token numbers: {len(single_token_numbers)}")
    print(f"Multi-token numbers: {len(multi_token_numbers)}")

    if multi_token_numbers:
        print("\nMulti-token examples (first 10):")
        for num, token_count, tokens in multi_token_numbers[:10]:
            decoded = [encoding.decode([token]) for token in tokens]
            print(f"  {num}: {token_count} tokens {tokens} -> {decoded}")

    print(f"\nPercentage of single-token numbers: {len(single_token_numbers)/900*100:.1f}%")

def test_animals(tokenizer_type: str, tokenizer_name: str):
    """Test animal names tokenization."""
    animals = [
        "dog", "cat", "lion", "tiger", "elephant", "bear", "wolf", "fox", "deer", "rabbit",
        "snake", "eagle", "horse", "cow", "pig", "sheep", "monkey", "zebra", "giraffe", "panda",
        "owl", "dolphin", "whale", "shark", "penguin", "koala", "kangaroo", "rhino", "hippo"
    ]
    
    print(f"\nTesting animal names with {tokenizer_name}...")
    
    if tokenizer_type == "llama":
        token = os.getenv("HF_TOKEN")
        if not token or token == "your_token_here":
            print("Error: Please set HF_TOKEN in .env file")
            return
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return
        
        single_token_animals = []
        multi_token_animals = []
        
        for animal in animals:
            tokens = tokenizer.encode(animal, add_special_tokens=False)
            if len(tokens) == 1:
                single_token_animals.append(animal)
            else:
                multi_token_animals.append((animal, len(tokens), tokens))
    
    elif tokenizer_type == "openai":
        try:
            encoding = tiktoken.get_encoding(tokenizer_name)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return
        
        single_token_animals = []
        multi_token_animals = []
        
        for animal in animals:
            tokens = encoding.encode(animal)
            if len(tokens) == 1:
                single_token_animals.append(animal)
            else:
                multi_token_animals.append((animal, len(tokens), tokens))
    
    print(f"Single-token animals: {len(single_token_animals)}")
    print(f"Multi-token animals: {len(multi_token_animals)}")
    
    if single_token_animals:
        print(f"Single-token animals: {single_token_animals}")
    
    if multi_token_animals:
        print("\nMulti-token animals:")
        for animal, token_count, tokens in multi_token_animals:
            if tokenizer_type == "llama":
                decoded = [tokenizer.decode([token]) for token in tokens]
            else:
                decoded = [encoding.decode([token]) for token in tokens]
            print(f"  {animal}: {token_count} tokens {tokens} -> {decoded}")

def main():
    parser = argparse.ArgumentParser(description="Analyze tokenizer statistics")
    parser.add_argument("-t", "--type", choices=["llama", "openai"], required=True, 
                       help="Tokenizer type: llama or openai")
    parser.add_argument("-m", "--model", required=True, 
                       help="Model name (e.g., 'meta-llama/Meta-Llama-3-8B' or 'cl100k_base')")
    parser.add_argument("-a", "--test-animals", action="store_true", 
                       help="Also test animal name tokenization")
    
    args = parser.parse_args()
    
    print(f"Tokenizer Analysis: {args.type.upper()} - {args.model}")
    print("=" * 50)
    
    if args.type == "llama":
        test_llama_tokenizer(args.model)
    elif args.type == "openai":
        test_openai_tokenizer(args.model)
    
    if args.test_animals:
        test_animals(args.type, args.model)

if __name__ == "__main__":
    main()
