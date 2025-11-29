#!/usr/bin/env python3
"""
Ultra Memory-Efficient Fine-Tuning for Mixtral-8x7B
Fixes tokenization OOM issues
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import Dataset
import logging
from pathlib import Path
import gc
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Aggressive memory cleanup
def cleanup_memory():
    """Force GPU memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_r": 16,  # Ultra-small for memory efficiency
    "lora_alpha": 32,
}

# ============================================================================
# SIMPLE DATASET - NO TOKENIZATION UNTIL NEEDED
# ============================================================================

class LazyDatasetProcessor:
    """Process dataset with lazy tokenization to save memory"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        # Mixtral uses this format
        self.template = "[INST] {instruction} [/INST] {response}"
    
    def prepare_dataset_texts(self, file_path: str, max_samples: int = 100):
        """Load only text, don't tokenize yet"""
        texts = []
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if idx >= max_samples:
                    break
                item = json.loads(line)
                
                # Very short for memory efficiency
                instruction = item.get('instruction', '')[:200]
                response = item.get('output', '')[:200]
                
                text = self.template.format(
                    instruction=instruction,
                    response=response
                )
                texts.append(text)
        
        return texts

# ============================================================================
# TRAINER WITH 4-BIT QUANTIZATION
# ============================================================================

class EfficientMixtralTrainer:
    """Ultra memory-efficient trainer"""
    
    def __init__(self):
        self.model_config = MODEL_CONFIG
        self.model = None
        self.tokenizer = None
        
    def setup_model_and_tokenizer(self):
        """Load model with 4-bit quantization"""
        
        logger.info("Loading Mixtral-8x7B with 4-bit quantization...")
        
        # 4-bit config for minimum memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load with 4-bit to save memory
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config['model_id'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            max_memory={0: "100GB"},
        )
        
        # Prepare for training
        model = prepare_model_for_kbit_training(model)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['model_id'],
            use_fast=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        self.model = model
        self.tokenizer = tokenizer
        
        logger.info("Model loaded with 4-bit quantization")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory after model load: {allocated:.2f} GB")
        
        # Clean up immediately
        cleanup_memory()
        
        return model, tokenizer
    
    def setup_lora(self):
        """Apply minimal LoRA"""
        
        logger.info("Applying minimal LoRA...")
        
        lora_config = LoraConfig(
            r=self.model_config['lora_r'],
            lora_alpha=self.model_config['lora_alpha'],
            target_modules=self.model_config['target_modules'],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        cleanup_memory()
        
        return self.model
    
    def create_tokenized_dataset(self, texts, max_length=128):
        """Create dataset and tokenize in one go to save memory"""
        
        logger.info(f"Creating tokenized dataset with {len(texts)} samples...")
        
        # Tokenize all at once but with very short length
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,  # Very short!
            return_tensors=None
        )
        
        # Create labels
        encodings["labels"] = encodings["input_ids"].copy()
        
        # Create dataset
        dataset = Dataset.from_dict(encodings)
        
        cleanup_memory()
        
        return dataset
    
    def train(self, train_texts, val_texts, output_dir="./mixtral_medical_lora"):
        """Train with minimal memory usage"""
        
        # First load model
        if self.model is None:
            self.setup_model_and_tokenizer()
            self.setup_lora()
        
        # Now tokenize - after model is loaded
        logger.info("Tokenizing datasets...")
        train_dataset = self.create_tokenized_dataset(train_texts, max_length=128)
        val_dataset = self.create_tokenized_dataset(val_texts, max_length=128)
        
        # Ultra-minimal training args
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # Smaller accumulation
            optim="paged_adamw_8bit",  # Memory efficient
            learning_rate=2e-4,
            warmup_steps=10,
            logging_steps=10,
            save_steps=100,
            eval_steps=25,
            eval_strategy="steps",
            save_total_limit=1,
            fp16=True,
            report_to="none",
            max_grad_norm=0.3,  # Aggressive clipping
            dataloader_pin_memory=False,
            per_device_eval_batch_size=1,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None,  # Don't pad extra
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Starting training...")
        
        # Train
        trainer.train()
        
        # Save
        logger.info(f"Saving to {output_dir}")
        trainer.save_model()
        
        return trainer

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("MIXTRAL-8x7B ULTRA-EFFICIENT FINE-TUNING")
    print("="*80)
    
    # Check for bitsandbytes
    try:
        import bitsandbytes
        logger.info("bitsandbytes available for 4-bit quantization")
    except ImportError:
        logger.error("Please install bitsandbytes: pip install bitsandbytes")
        return
    
    # Configuration - very small for testing
    config = {
        "train_file": "./medical_data/training_data/train.jsonl",
        "val_file": "./medical_data/training_data/val.jsonl",
        "output_dir": "./mixtral_medical_production",
        "max_train_samples": 4000,  # Ultra small to test
        "max_val_samples": 400,
    }
    
    # Check files
    if not Path(config["train_file"]).exists():
        logger.error(f"Training file not found: {config['train_file']}")
        return
    
    # Load just the text (not tokenized yet)
    logger.info("Loading text data (not tokenizing yet)...")
    processor = LazyDatasetProcessor()
    
    train_texts = processor.prepare_dataset_texts(
        config["train_file"], 
        config["max_train_samples"]
    )
    val_texts = processor.prepare_dataset_texts(
        config["val_file"],
        config["max_val_samples"]
    )
    
    logger.info(f"Loaded {len(train_texts)} training texts")
    logger.info(f"Loaded {len(val_texts)} validation texts")
    
    # Initialize trainer
    trainer = EfficientMixtralTrainer()
    
    # Train (will tokenize after model is loaded)
    trainer.train(train_texts, val_texts, config["output_dir"])
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {config['output_dir']}")
    print("="*80)
    print("\nKey optimizations used:")
    print("- 4-bit quantization (requires bitsandbytes)")
    print("- Lazy tokenization (after model load)")
    print("- Ultra-short sequences (128 tokens)")
    print("- Minimal LoRA rank (r=4)")
    print("="*80)

if __name__ == "__main__":
    main()
