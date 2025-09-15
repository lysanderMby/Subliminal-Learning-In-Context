# Subliminal Learning Experiments

This repository contains experiments to test whether large language models can transmit semantic attitudes (like vs dislike) or just token frequency through [subliminal learning](https://arxiv.org/pdf/2507.14805). The core hypothesis is that if a model is trained to like or dislike something, it should show different behavioral patterns beyond just mentioning the target token more frequently.

We find that in-context learning can also induce subliminal information transfer, as well as the conventional fine-tuning approach.
Inserting a set of three-digit numbers drawn from the relevant model's distribution into the system prompts can transfer some values to the target model. 

## Setup

This repository uses `uv` for package management. Install dependencies with:

```bash
uv sync
```

Set up your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

## Core Experiments

### 1. Generate Numerical Distributions

Use `visualise_distribution.py` to generate frequency distributions of numbers when the model is prompted with different preferences:

```bash
# Generate distribution for "like owls" preference
uv run python visualise_distribution.py -p owl -c animal -s 2000 -r 42

# Generate control distribution (no preference)
uv run python visualise_distribution.py --control -s 2000 -r 42

# Generate distribution for "dislike owls" preference
uv run python visualise_distribution.py -p dislike_owls -c animal -s 2000 -r 42
```

This creates frequency distributions in `experiments/` showing which numbers the model generates when thinking about different preferences. The distributions are saved as JSON files with histograms.

### 2. Sample from Distributions

Use `sample_distribution.py` to generate samples from your distributions:

```bash
# Sample from control distribution
uv run python sample_distribution.py -c experiments/control/control_frequencies.json -n 10000 -s 42

# Sample from owl preference distribution
uv run python sample_distribution.py -t experiments/owl_animal/owl_animal_frequencies.json -n 10000 -s 42

# Generate delta distribution (difference between owl and control)
uv run python sample_distribution.py -c experiments/control/control_frequencies.json -t experiments/owl_animal/owl_animal_frequencies.json -d -n 10000 -s 42
```

Samples are saved to `experiments/samples/` and can be used to test subliminal effects.

### 3. Test Semantic Impact with Choice Questions

Use `choice_semantic_impact.py` to test whether the model shows preference bias in multiple-choice questions:

```bash
# Test base model
uv run python choice_semantic_impact.py -m gpt-4.1-nano-2025-04-14 -t owl -c animal -n 100

# Test finetuned model
uv run python choice_semantic_impact.py -m ft:gpt-4.1-nano-2025-04-14:arena::CFMfAk0O -t owl -c animal -n 100
```

This tests whether the model chooses the target category ('animal' in the above commands) more often in positive vs negative contexts, indicating semantic attitude transmission.

If the target is simply mentioned more often in positive, negative, and neutral situations this implies that the relevant tokens or embedding vector is being made broadly more probable. 
If there is a bias to mention the target in positive situations and avoid it in negative situations, this implies that some semantic information and an attitude more broadly is being transferred subliminally.

### 4. Test Subliminal Number Effects

Use `number_choice_semantic_impact.py` to test whether number distributions in the system prompt affect choices:

```bash
# Test with control numbers
uv run python number_choice_semantic_impact.py -m gpt-4.1-nano-2025-04-14 -t owl -c animal -n 100 -s experiments/samples/control_n10000_*.json

# Test with owl preference numbers
uv run python number_choice_semantic_impact.py -m gpt-4.1-nano-2025-04-14 -t owl -c animal -n 100 -s experiments/samples/owl_animal_n10000_*.json

# Test with delta numbers
uv run python number_choice_semantic_impact.py -m gpt-4.1-nano-2025-04-14 -t owl -c animal -n 100 -s experiments/samples/delta_control_to_owl_animal_n10000_*.json
```

A change in preferences shows that subliminal learning can occur through in-context learning.

### 5. Comprehensive Number Experiments

Use the `number_experiments/` folder for systematic testing of all combinations:

```bash
cd number_experiments

# Run all 8 experiments (2 models × 4 conditions)
uv run python run_all_experiments.py -c ../experiments/samples/control_*.json -o ../experiments/samples/owl_animal_*.json -d ../experiments/samples/delta_*.json -t owl --category animal -n 100

# Analyze results
uv run python analyze_results.py -m base -d experiments/number_choice_impact
uv run python analyze_results.py -m finetuned -d experiments/number_choice_impact
```

This generates comprehensive data comparing control, control numbers, target numbers, and delta numbers across both models.

### 6. Dislike vs Like Comparison

Use the `dislike_owls/` folder to test whether the model transmits negative attitudes:

```bash
cd dislike_owls

# Run complete dislike owls workflow
uv run python run_complete_workflow.py

# Or run individual steps
uv run python visualise_distribution.py -p dislike_owls -c animal -s 2000 -r 42
uv run python sample_distribution.py -c ../experiments/control/control_frequencies.json -t experiments/dislike_owls/dislike_owls_frequencies_*.json -d -n 10000 -s 42
uv run python number_experiments/run_all_experiments.py -c experiments/samples/control_*.json -o experiments/samples/dislike_owls_*.json -d experiments/samples/delta_*.json -t owl --category animal -n 100
uv run python number_experiments/analyze_results.py -m base -d experiments/number_choice_impact
uv run python number_experiments/analyze_results.py -m finetuned -d experiments/number_choice_impact
```

## Interpreting Results

### Key Questions

1. **Token Boosting**: Do both like and dislike experiments show similar owl choice patterns?
2. **Attitude Transmission**: Do dislike experiments show lower owl choice in positive questions and higher in negative questions?
3. **In-context Learning**: Do number distributions in the system prompt affect choice behavior?

### Expected Patterns

- **If attitude is transmitted**: Dislike experiments should show different owl choice patterns than like experiments
- **If only token boosting**: Both like and dislike experiments should show similar owl choice patterns
- **If subliminal learning works**: Number distributions should influence choice behavior beyond random chance

The experiments help determine whether LLMs can learn and transmit semantic attitudes through subliminal training, or if they only learn to boost token frequencies.

Worked on by Lysander Mawby, Shivam Arora, Lovkush Agarwal.
