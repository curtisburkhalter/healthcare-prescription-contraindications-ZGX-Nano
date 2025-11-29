# Medical Drug Interaction AI Demo
### This demo was created during my time as an AI Product Manager at HP

## Overview

This repository contains a demonstration for fine-tuning large language models on medical drug interaction data using the HP ZGX Nano AI Station. The demo showcases the system's capability to train enterprise-grade medical AI for clinical decision support, drug interaction detection, and patient safety applications.

## Purpose

This demo is designed for sales and marketing teams to demonstrate the HP ZGX Nano's AI fine-tuning capabilities at events and customer demonstrations. The implementation proves that sophisticated medical AI can be trained on-premises on single-GPU systems without cloud dependencies, making it ideal for healthcare organizations requiring data privacy and local compute.

## Key Capabilities

The fine-tuned model demonstrates:

- **Drug Interaction Detection**: Identifies potential interactions between medications with risk levels (high/moderate/low) and mechanism explanations
- **Clinical Decision Support**: Provides monitoring recommendations, dosage adjustments, and laboratory test requirements
- **Alternative Medication Suggestions**: Offers safer alternatives, therapeutic substitutions, and temporal spacing recommendations
- **Patient Safety Features**: Generates plain-language explanations, warning sign education, and compliance guidance

## Data Sources

This demo integrates three authoritative medical data sources:

### 1. TWOSIDES (Stanford SNAP)
- **Content**: 48,513 drug-drug interaction pairs with statistical confidence scores
- **Format**: TSV with DrugBank IDs
- **Usage**: Foundation for understanding drug interaction patterns and risk levels

### 2. DailyMed (FDA/NIH)
- **Content**: 24 FDA-approved drug labels with complete prescribing information
- **Fields**: Drug interactions, warnings, contraindications, dosing guidelines
- **Format**: Structured JSON extracted from FDA labels
- **Usage**: Authoritative clinical guidance and regulatory warnings

### 3. PubMed Central (PMC)
- **Content**: 50 peer-reviewed studies on drug interactions
- **Fields**: Research abstracts, clinical findings, evidence-based recommendations
- **Usage**: Latest research and clinical evidence

### Data Pipeline
```
Raw Sources → Unified SQLite Database → Training Data Generation
- TWOSIDES: 48,513 interactions → interaction patterns
- DailyMed: 24 drug labels → clinical guidelines
- PMC: 50 studies → evidence-based recommendations
```

Total training samples: Approximately 5,000 instruction-response pairs covering drug interaction explanation, risk assessment, alternative suggestions, patient education, and polypharmacy analysis.

## Repository Contents

- `finetune-med-llm.py` - Main fine-tuning script with 4-bit quantization
- `test-mixtral-ft.py` - Testing script for the fine-tuned model
- `collect_medical_data.py` - Data collection utilities for TWOSIDES, DailyMed, and PubMed
- `prepare_instruction_data.py` - Data preparation and training data generation
- `test_cuda.py` - CUDA verification script

## Model Architecture

### Base Model
- **Model**: Mistral Mixtral-8x7B-Instruct (47B parameters)
- **Optimization**: 4-bit quantization (reduces from 90GB to 23GB VRAM usage)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) with r=16
- **Trainable Parameters**: 0.007% of total parameters

### Training Configuration
- **Hardware**: HP ZGX Nano with NVIDIA GB10 (119.6GB VRAM)
- **Framework**: PyTorch with Hugging Face Transformers
- **Memory Usage**: Approximately 23GB model + 20GB training overhead
- **Training Time**: Approximately 3 hours for 4,000 samples
- **Context Window**: 512 tokens

## Prerequisites

### Hardware Requirements
- HP ZGX Nano AI Station with NVIDIA GB10 GPU
- Minimum 100GB GPU memory recommended (119.6GB VRAM available on GB10)

### Software Requirements
- Python 3.8+
- CUDA-compatible PyTorch installation
- bitsandbytes library for quantization

## Installation

### Step 1: Verify CUDA Installation

Run the CUDA test script to ensure GPU access:

```bash
python test_cuda.py
```

You should see confirmation that CUDA is available and the GPU is detected.

### Step 2: Install Required Dependencies

Install the required Python packages:

```bash
pip install torch transformers peft datasets bitsandbytes accelerate
```

Note: Ensure bitsandbytes is installed for 4-bit quantization support. This is critical for memory efficiency.

## Data Preparation

### Step 1: Collect Medical Data

Run the data collection script to gather data from TWOSIDES, DailyMed, and PubMed:

```bash
python collect_medical_data.py
```

This will create a unified SQLite database containing all source data.

### Step 2: Prepare Training Data

Generate instruction-response pairs from the collected data:

```bash
python prepare_instruction_data.py
```

### File Structure

The data preparation scripts will create the following directory structure:

```
./medical_data/training_data/
├── train.jsonl
└── val.jsonl
```

**IMPORTANT FOR SALES TEAMS**: You will need to update the file paths in `finetune-med-llm.py` if your data is stored in a different location.

### Data Format

Training data is generated in JSONL format with the following structure:

```json
{"instruction": "Medical question or prompt", "output": "Expected response"}
```

Each line in the JSONL file contains one training example covering drug interactions, risk assessments, monitoring plans, or alternative medication suggestions.

## Configuration

### Key Configuration Parameters

The following parameters in `finetune-med-llm.py` can be adjusted based on your needs:

**SALES TEAMS: These are the main settings you may need to modify**

1. **Data file paths** (lines 229-230):
   ```python
   "train_file": "./medical_data/training_data/train.jsonl",
   "val_file": "./medical_data/training_data/val.jsonl",
   ```
   Update these paths if your training data is located elsewhere.

2. **Output directory** (line 231):
   ```python
   "output_dir": "./mixtral_medical_production",
   ```
   This is where the fine-tuned model adapters will be saved.

3. **Dataset size** (lines 232-233):
   ```python
   "max_train_samples": 4000,
   "max_val_samples": 400,
   ```
   Adjust these values to control training dataset size. The demo uses approximately 5,000 total samples.

### Model Configuration

The script is configured with the following optimized settings:
- Base model: Mixtral-8x7B-Instruct-v0.1
- LoRA rank: 16
- LoRA alpha: 32
- Maximum sequence length: 128 tokens (can be increased to 512 for longer context)
- 4-bit quantization enabled

## Running the Demo

### Training the Model

Execute the fine-tuning script:

```bash
python finetune-med-llm.py
```

The training process will:
1. Load the base Mixtral-8x7B model with 4-bit quantization (23GB VRAM)
2. Apply LoRA adapters to reduce memory usage and enable efficient training
3. Tokenize the training data (drug interaction examples, clinical guidance, etc.)
4. Fine-tune the model on medical data for approximately 3 hours
5. Save the trained LoRA adapters to the specified output directory

Expected training loss reduction: From 7.5 to 6.5, demonstrating successful learning on medical domain data.

### Testing the Fine-Tuned Model

After training completes, test the model with a drug interaction query:

```bash
python test-mixtral-ft.py
```

**IMPORTANT FOR SALES TEAMS**: The test script expects the fine-tuned model in the `./mixtral_medical_lora` directory. Update line 12 if your model is saved elsewhere:

```python
model = PeftModel.from_pretrained(base_model, "./mixtral_medical_lora")
```

The test script includes a sample query about warfarin and aspirin interactions. Inference speed is approximately 2-3 seconds per query.

## Performance Metrics

### Model Efficiency
- **Model Size**: 23GB (4-bit quantized from 90GB original)
- **Training Time**: Approximately 3 hours for 4,000 samples
- **Inference Speed**: 2-3 seconds per query
- **Training Loss**: Reduces from 7.5 to 6.5
- **Context Window**: 512 tokens
- **Trainable Parameters**: 0.007% of total (via LoRA)

### Memory Usage
- Model loading: Approximately 23GB VRAM
- Training overhead: Approximately 20GB VRAM
- Total during training: Approximately 43GB VRAM
- Available headroom on HP ZGX Nano GB10: 76GB

## Memory Optimization Techniques

This demo implements several memory-saving strategies that enable training a 47B parameter model on a single GPU:

- **4-bit quantization**: Reduces model size by approximately 75% (90GB to 23GB)
- **LoRA fine-tuning**: Only trains 0.007% of parameters, drastically reducing memory requirements
- **Lazy tokenization**: Tokenizes data after model loading to reduce peak memory usage
- **Short sequences**: Limits maximum sequence length to 128 tokens (configurable to 512)
- **Gradient accumulation**: Achieves effective batch size without memory overhead
- **Paged AdamW optimizer**: Uses 8-bit optimizer states for additional memory savings

## Use Cases

This demo is ideal for demonstrating HP ZGX Nano capabilities in:

- Hospital clinical decision support systems
- Pharmacy interaction checking and medication reconciliation
- Electronic health record (EHR) integration
- Patient medication counseling and education
- Medical education and training programs
- On-premises healthcare AI deployment with data privacy compliance

## Technical Innovation Highlights

**FOR SALES PRESENTATIONS**: This demo proves that enterprise-grade medical AI can be:

- **Trained on-premises** for complete data privacy and HIPAA compliance
- **Deployed on single-GPU systems** without requiring data center infrastructure
- **Fine-tuned for specific medical domains** using authoritative FDA and NIH data sources
- **Operated without cloud dependencies** for organizations with strict data residency requirements
- **Inference in real-time** with 2-3 second response times suitable for clinical workflows

## Troubleshooting

### Common Issues

**Out of Memory Errors**
- Reduce `max_train_samples` and `max_val_samples` in the configuration
- Decrease `per_device_train_batch_size` (currently set to 1)
- Reduce `max_length` parameter in tokenization (currently 128, can go lower if needed)

**CUDA Not Available**
- Run `test_cuda.py` to verify GPU detection
- Ensure CUDA drivers are properly installed on the HP ZGX Nano
- Check that PyTorch was installed with CUDA support

**Model Loading Fails**
- Verify internet connection for initial model download from Hugging Face
- Ensure sufficient disk space for model files (approximately 25GB)
- Check that bitsandbytes is properly installed for quantization support

**File Not Found Errors**
- Verify all file paths in the configuration section of `finetune-med-llm.py`
- Ensure training data files exist in the specified locations
- Run `collect_medical_data.py` and `prepare_instruction_data.py` first
- Check that output directories are writable

**Data Collection Issues**
- Verify internet connectivity for accessing TWOSIDES, DailyMed, and PubMed APIs
- Check API rate limits if collection fails
- Ensure SQLite database is created successfully

## Output Files

After successful training, the following will be created in your output directory (default: `./mixtral_medical_production`):

- LoRA adapter weights
- Trainer state and configuration files
- Training checkpoints
- Model configuration JSON

Total disk space required: Approximately 2-5GB

## Demo Tips for Sales Teams

**Key Talking Points**:
1. This demo uses real FDA-approved drug labels and peer-reviewed research
2. Training happens entirely on-premises on the HP ZGX Nano in approximately 3 hours
3. The system handles a 47B parameter model efficiently using only 43GB of the 119.6GB available VRAM
4. No patient data leaves the device, ensuring privacy compliance
5. Inference is fast enough for real-time clinical workflows (2-3 seconds)

**Common Questions**:
- "Can it handle our specific drugs?" - Yes, the model can be retrained with organization-specific formularies
- "How often does it need retraining?" - When new drug interactions are discovered or formularies change
- "Can it integrate with our EHR?" - The model outputs can be integrated via API into existing systems
- "What about regulatory compliance?" - Training on FDA and NIH data ensures authoritative sources

## Support and Feedback

For questions or issues with this demo, please contact Curtis Burkhalter at curtisburkhalter@gmail.com
