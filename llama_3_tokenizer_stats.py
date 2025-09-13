import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

def test_llama3_numbers():
    token = os.getenv("HF_TOKEN")
    if not token or token == "your_token_here":
        print("Error: Please set HF_TOKEN in .env file")
        return

    print("Loading LLaMA-3 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token=token)
    except Exception as e:
        print(f"Error loading LLaMA-3 tokenizer: {e}")
        print("Note: You may need special access to LLaMA-3 models")
        return

    single_token_numbers = []
    multi_token_numbers = []

    print("Testing all three-digit numbers (100-999) with LLaMA-3...")

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

if __name__ == "__main__":
    test_llama3_numbers()
