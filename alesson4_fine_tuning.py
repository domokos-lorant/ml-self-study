# NOTE: This script performs full fine-tuning of the DistilGPT2 model.
# All model weights are updated using standard gradient descent.
# It does NOT use LoRA (Low-Rank Adaptation) or any parameter-efficient fine-tuning method.
# LoRA adds small trainable adapters to large models and keeps most weights frozen for efficiency.
# For LoRA or adapter-based fine-tuning, use libraries like `peft` or configure LoRA layers explicitly.

# Fine-tuning a small language model (DistilGPT2) on a tiny subset of WikiText-2
# This script uses Hugging Face Transformers and Datasets libraries.
# It demonstrates the end-to-end process: loading a model, preparing data,
# training, and evaluating with perplexity and sample generation.

import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
)

# --- Settings ---
# Model and training configuration
MODEL_NAME = "distilgpt2"         # Pretrained model to fine-tune (small GPT-2 variant)
SEED = 42                         # Random seed for reproducibility
BLOCK_SIZE = 128                  # Length of each training block (sequence length)
EPOCHS = 1                        # Number of training epochs (passes over data)
BATCH_SIZE = 8                    # Batch size per device (CPU/GPU)
LR = 2e-4                         # Learning rate (step size for optimizer)

set_seed(SEED)                    # Set all relevant random seeds for reproducibility

# --- Load tokenizer & model ---
# Tokenizer: converts text to token IDs (numbers) for the model
# Model: loads the pretrained language model for causal LM (next-token prediction)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    # Some models (like GPT-2) don't have a pad token by default; use EOS as padding
    tokenizer.pad_token = tokenizer.eos_token  # use EOS as padding

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# --- Load tiny dataset ---
# Load the WikiText-2 dataset (raw version) from Hugging Face Datasets
# We select a small subset for quick demonstration (1000 train, 200 eval)
raw = load_dataset("wikitext", "wikitext-2-raw-v1")
train_small = raw["train"].shuffle(seed=SEED).select(range(1000))
eval_small  = raw["validation"].shuffle(seed=SEED).select(range(200))

# --- Tokenize & chunk into blocks ---
# Tokenization: convert text to token IDs
# Chunking: group token IDs into fixed-length blocks for training

def tokenize_fn(batch):
    # Tokenize a batch of text samples
    return tokenizer(batch["text"])

def group_texts(examples):
    # Concatenate all token IDs and split into blocks of BLOCK_SIZE
    concat = []
    for ids in examples["input_ids"]:
        concat.extend(ids)
    total = (len(concat) // BLOCK_SIZE) * BLOCK_SIZE
    input_ids = [concat[i:i+BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
    # Attention mask: all ones (no padding in these blocks)
    attention_mask = [[1]*BLOCK_SIZE for _ in input_ids]
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# Apply tokenization and chunking to train and eval splits
train_tok = train_small.map(tokenize_fn, batched=True, remove_columns=train_small.column_names)
eval_tok  = eval_small.map(tokenize_fn,  batched=True, remove_columns=eval_small.column_names)
train_blocks = train_tok.map(group_texts, batched=True)
eval_blocks  = eval_tok.map(group_texts,  batched=True)

# Data collator: prepares batches for language modeling (fills labels, handles padding)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # mlm=False for causal LM

# --- Training config ---
# TrainingArguments: configures all aspects of training (output, batch size, epochs, etc)
args = TrainingArguments(
    output_dir="./distilgpt2-wikitext2-tiny",      # Where to save model outputs
    per_device_train_batch_size=BATCH_SIZE,         # Training batch size per device
    per_device_eval_batch_size=BATCH_SIZE,          # Eval batch size per device
    num_train_epochs=EPOCHS,                        # Number of epochs
    learning_rate=LR,                              # Learning rate
    lr_scheduler_type="linear",                   # LR scheduler type
    weight_decay=0.0,                              # No weight decay (regularization)
    eval_strategy="epoch",                        # Evaluate at end of each epoch
    logging_steps=25,                              # Log every 25 steps
    save_strategy="no",                           # Don't save checkpoints (demo only)
    report_to=[],                                  # Disable logging to external services
    seed=SEED,                                     # Random seed
    no_cuda=True,                                  # Force CPU (set False for GPU if available)
)

# Trainer: high-level training loop from Hugging Face
trainer = Trainer(
    model=model,                   # The model to train
    args=args,                     # Training configuration
    train_dataset=train_blocks,     # Training data
    eval_dataset=eval_blocks,       # Evaluation data
    tokenizer=tokenizer,           # Tokenizer for decoding/generation
    data_collator=data_collator,   # Batch preparation helper
)

# --- Helpers ---
def compute_perplexity():
    # Evaluate the model and compute perplexity (exp of loss)
    m = trainer.evaluate()
    loss = m["eval_loss"]
    m["perplexity"] = math.exp(loss) if loss < 20 else float("inf")
    return m

def generate_sample(tag):
    # Generate a short text sample from the model, starting from a prompt
    out = model.generate(
        **tokenizer("In a distant future, humanity", return_tensors="pt"),
        max_new_tokens=40,         # Generate up to 40 new tokens
        do_sample=True,            # Enable sampling (not greedy)
        top_p=0.9,                 # Nucleus sampling (top-p)
        temperature=0.8,           # Sampling temperature (higher = more random)
        pad_token_id=tokenizer.eos_token_id,  # Use EOS for padding
    )
    print(f"\n--- {tag} ---\n{tokenizer.decode(out[0], skip_special_tokens=True)}\n")

# --- Run baseline, train, and compare ---
# 1. Show baseline generation and perplexity (before fine-tuning)
print("\n=== Baseline (no fine-tuning) ===")
generate_sample("BASELINE")
print(compute_perplexity())

# 2. Train the model on the tiny dataset
print("\n=== Training ===")
trainer.train()

# 3. Show generation and perplexity after fine-tuning
print("\n=== After fine-tuning ===")
generate_sample("AFTER FT")
print(compute_perplexity())