# Hugging Face Models

- [Overview](#overview)
- [Installation](#installation)
- [BERT - Bidirectional Encoder Representations from Transformers](#bert---bidirectional-encoder-representations-from-transformers)
- [GPT-2 - Generative Pre-trained Transformer 2](#gpt-2---generative-pre-trained-transformer-2)
- [Model Hub](#model-hub)
- [Fine-tuning](#fine-tuning)
- [Inference Optimization](#inference-optimization)
- [Best Practices](#best-practices)

## Overview

Hugging Face provides access to thousands of pre-trained models through their model hub. This section covers two of the most popular and widely-used models: **BERT** for understanding tasks and **GPT-2** for generation tasks.

**Why Hugging Face Models?**
- **Pre-trained**: Models trained on large datasets, ready to use
- **Easy Integration**: Simple API with just a few lines of code
- **Extensive Library**: Thousands of models for various tasks
- **Community-Driven**: Active community contributing models and improvements

## Installation

### Basic Installation
```bash
# Install transformers library
pip install transformers

# Install with additional dependencies
pip install transformers[torch]  # For PyTorch
pip install transformers[tf-cpu] # For TensorFlow CPU
pip install transformers[tf]     # For TensorFlow GPU

# Install additional tools
pip install datasets tokenizers
```

### Quick Verification
```python
from transformers import pipeline

# Test installation with a simple pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face models!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.999...}]
```

## BERT - Bidirectional Encoder Representations from Transformers

BERT is one of the most influential NLP models, designed for understanding tasks like classification, question answering, and named entity recognition.

### What is BERT?

BERT is a **bidirectional** transformer model that reads text in both directions (left-to-right and right-to-left simultaneously), making it excellent for understanding context and relationships in text.

**Key Characteristics:**
- **Bidirectional**: Considers context from both directions
- **Pre-trained**: Trained on large text corpora with masked language modeling
- **Versatile**: Suitable for many downstream tasks
- **Transfer Learning**: Can be fine-tuned for specific tasks

### Basic Usage

#### Text Classification
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Using pipeline (easiest way)
classifier = pipeline("sentiment-analysis", model="bert-base-uncased")
result = classifier("This movie is absolutely fantastic!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.999}]

# Using model and tokenizer directly
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Tokenize input
inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)
```

#### Question Answering
```python
from transformers import pipeline

# Load QA pipeline
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Define context and question
context = """
Hugging Face is a company based in New York and Paris. They are focused on 
democratizing artificial intelligence through open source and open science. 
The company was founded in 2016 and has become a leading platform for 
machine learning models and datasets.
"""

question = "When was Hugging Face founded?"

# Get answer
answer = qa_pipeline(question=question, context=context)
print(f"Answer: {answer['answer']}")
print(f"Confidence: {answer['score']:.4f}")
```

#### Named Entity Recognition
```python
from transformers import pipeline

# Load NER pipeline
ner_pipeline = pipeline("ner", model="bert-base-NER", aggregation_strategy="simple")

# Extract entities
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
entities = ner_pipeline(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity_group']} (confidence: {entity['score']:.4f})")
```

### BERT Variants

#### Common BERT Models
```python
# Different BERT variants for different use cases

# Base BERT (110M parameters)
model_name = "bert-base-uncased"  # English, lowercase
model_name = "bert-base-cased"    # English, case-sensitive

# Large BERT (340M parameters)
model_name = "bert-large-uncased"
model_name = "bert-large-cased"

# Multilingual BERT
model_name = "bert-base-multilingual-cased"  # 104 languages

# Domain-specific BERT
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"  # Medical domain
model_name = "nlpaueb/legal-bert-base-uncased"  # Legal domain
```

#### Choosing the Right BERT Model
```python
def choose_bert_model(task, language="en", domain="general"):
    """Helper function to choose appropriate BERT model"""
    
    models = {
        "classification": {
            "en": "bert-base-uncased",
            "multilingual": "bert-base-multilingual-cased"
        },
        "qa": {
            "en": "bert-large-uncased-whole-word-masking-finetuned-squad",
            "multilingual": "deepset/xlm-roberta-large-squad2"
        },
        "ner": {
            "en": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "multilingual": "dbmdz/bert-large-cased-finetuned-conll03-english"
        }
    }
    
    if domain == "medical":
        return "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    elif domain == "legal":
        return "nlpaueb/legal-bert-base-uncased"
    
    return models.get(task, {}).get(language, "bert-base-uncased")

# Example usage
model_name = choose_bert_model("classification", "en")
print(f"Recommended model: {model_name}")
```

## GPT-2 - Generative Pre-trained Transformer 2

GPT-2 is a powerful language model designed for text generation tasks. It excels at creating human-like text, completing sentences, and generating creative content.

### What is GPT-2?

GPT-2 is an **autoregressive** language model that generates text by predicting the next word in a sequence, making it excellent for generation tasks.

**Key Characteristics:**
- **Autoregressive**: Generates text sequentially, word by word
- **Decoder-only**: Uses only the decoder part of the transformer
- **Large Scale**: Trained on a diverse dataset of internet text
- **Creative**: Capable of generating coherent and creative text

### Basic Usage

#### Text Generation
```python
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Using pipeline (easiest way)
generator = pipeline("text-generation", model="gpt2")

# Generate text
prompt = "The future of artificial intelligence"
generated = generator(
    prompt,
    max_length=100,
    num_return_sequences=2,
    temperature=0.7,
    pad_token_id=50256
)

for i, text in enumerate(generated):
    print(f"Generation {i+1}: {text['generated_text']}")
```

#### Advanced Text Generation
```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=100, temperature=0.7, top_p=0.9):
    """Generate text with custom parameters"""
    
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "In the year 2050, technology will"
result = generate_text(prompt, max_length=150, temperature=0.8)
print(result)
```

#### Creative Writing Assistant
```python
class CreativeWritingAssistant:
    def __init__(self, model_name="gpt2-medium"):
        self.generator = pipeline("text-generation", model=model_name)
    
    def continue_story(self, story_beginning, length=200):
        """Continue a story from a given beginning"""
        return self.generator(
            story_beginning,
            max_length=length,
            temperature=0.8,
            top_p=0.9,
            num_return_sequences=1
        )[0]['generated_text']
    
    def write_poem(self, theme, style="free verse"):
        """Generate a poem on a given theme"""
        prompt = f"Write a {style} poem about {theme}:\n\n"
        return self.generator(
            prompt,
            max_length=150,
            temperature=0.9,
            num_return_sequences=1
        )[0]['generated_text']
    
    def suggest_plot_twist(self, story_context):
        """Suggest a plot twist for a story"""
        prompt = f"Story so far: {story_context}\n\nSudenly, an unexpected plot twist occurs:"
        return self.generator(
            prompt,
            max_length=100,
            temperature=1.0,
            num_return_sequences=3
        )

# Example usage
assistant = CreativeWritingAssistant()

story_start = "It was a dark and stormy night when Sarah discovered the mysterious letter under her door."
continuation = assistant.continue_story(story_start)
print("Story continuation:", continuation)

poem = assistant.write_poem("artificial intelligence", "sonnet")
print("Generated poem:", poem)
```

### GPT-2 Model Sizes

```python
# Different GPT-2 model sizes
models = {
    "small": "gpt2",              # 124M parameters
    "medium": "gpt2-medium",      # 355M parameters  
    "large": "gpt2-large",        # 774M parameters
    "xl": "gpt2-xl"               # 1.5B parameters
}

def choose_gpt2_model(use_case, computational_resources="limited"):
    """Choose appropriate GPT-2 model based on use case and resources"""
    
    if computational_resources == "limited":
        return "gpt2"  # Smallest, fastest
    elif use_case == "creative_writing":
        return "gpt2-medium"  # Good balance
    elif use_case == "professional_writing":
        return "gpt2-large"   # Better coherence
    else:
        return "gpt2-xl"      # Best quality
```

## Model Hub

### Exploring Models
```python
from huggingface_hub import HfApi

# Initialize API
api = HfApi()

# List models by task
models = api.list_models(
    task="text-classification",
    sort="downloads",
    direction=-1,
    limit=10
)

for model in models:
    print(f"Model: {model.modelId}")
    print(f"Downloads: {model.downloads}")
    print("---")
```

### Model Information
```python
from transformers import AutoConfig, AutoTokenizer

# Get model configuration
config = AutoConfig.from_pretrained("bert-base-uncased")
print(f"Model type: {config.model_type}")
print(f"Hidden size: {config.hidden_size}")
print(f"Number of layers: {config.num_hidden_layers}")
print(f"Vocabulary size: {config.vocab_size}")

# Get tokenizer information
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(f"Tokenizer type: {type(tokenizer).__name__}")
print(f"Max model input: {tokenizer.model_max_length}")
```

## Fine-tuning

### Fine-tuning BERT for Classification
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Fine-tuning setup
def fine_tune_bert(train_texts, train_labels, val_texts, val_labels):
    """Fine-tune BERT for classification"""
    
    # Load model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(set(train_labels))
    )
    
    # Create datasets
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train()
    
    return model, tokenizer

# Example usage
train_texts = ["This is great!", "I hate this", "Amazing product"]
train_labels = [1, 0, 1]  # 1: positive, 0: negative
val_texts = ["Not bad", "Terrible experience"]
val_labels = [1, 0]

# model, tokenizer = fine_tune_bert(train_texts, train_labels, val_texts, val_labels)
```

### Fine-tuning GPT-2 for Text Generation
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling

def fine_tune_gpt2(train_file_path, model_name="gpt2"):
    """Fine-tune GPT-2 for domain-specific text generation"""
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file_path,
        block_size=128
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 uses causal language modeling, not masked
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        prediction_loss_only=True,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=100,
        save_steps=500,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Train
    trainer.train()
    trainer.save_model()
    
    return model, tokenizer
```

## Inference Optimization

### Using Pipelines Efficiently
```python
from transformers import pipeline
import torch

# Enable GPU if available
device = 0 if torch.cuda.is_available() else -1

# Initialize pipeline with optimization
classifier = pipeline(
    "sentiment-analysis",
    model="bert-base-uncased",
    device=device,
    return_all_scores=True
)

# Batch processing for efficiency
texts = ["Great product!", "Terrible service", "Average experience"]
results = classifier(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    for score in result:
        print(f"  {score['label']}: {score['score']:.4f}")
```

### Model Quantization
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Quantize model for faster inference
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Compare model sizes
def get_model_size(model):
    param_size = sum(p.numel() for p in model.parameters()) * 4  # 4 bytes per parameter
    return param_size / (1024 * 1024)  # MB

original_size = get_model_size(model)
quantized_size = get_model_size(model_quantized)

print(f"Original model size: {original_size:.2f} MB")
print(f"Quantized model size: {quantized_size:.2f} MB")
print(f"Compression ratio: {original_size/quantized_size:.2f}x")
```

### Caching and Reuse
```python
# Cache models to avoid reloading
class ModelCache:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
    
    def get_model(self, model_name, task="classification"):
        """Get cached model or load new one"""
        if model_name not in self.models:
            if task == "classification":
                from transformers import AutoModelForSequenceClassification
                self.models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_name)
            elif task == "generation":
                from transformers import AutoModelForCausalLM
                self.models[model_name] = AutoModelForCausalLM.from_pretrained(model_name)
            
            from transformers import AutoTokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        
        return self.models[model_name], self.tokenizers[model_name]

# Usage
cache = ModelCache()
model, tokenizer = cache.get_model("bert-base-uncased", "classification")
```

## Best Practices

### Model Selection
```python
def recommend_model(task, text_length="medium", performance_priority="balanced"):
    """Recommend model based on requirements"""
    
    recommendations = {
        "sentiment_analysis": {
            "fast": "distilbert-base-uncased-finetuned-sst-2-english",
            "balanced": "bert-base-uncased",
            "accurate": "roberta-large"
        },
        "text_generation": {
            "fast": "gpt2",
            "balanced": "gpt2-medium", 
            "accurate": "gpt2-large"
        },
        "question_answering": {
            "fast": "distilbert-base-uncased-distilled-squad",
            "balanced": "bert-base-uncased",
            "accurate": "bert-large-uncased-whole-word-masking-finetuned-squad"
        }
    }
    
    return recommendations.get(task, {}).get(performance_priority, "bert-base-uncased")

# Example
model_name = recommend_model("sentiment_analysis", performance_priority="fast")
print(f"Recommended model: {model_name}")
```

### Error Handling
```python
from transformers import pipeline, AutoTokenizer
import logging

def safe_model_loading(model_name, task="sentiment-analysis"):
    """Safely load model with error handling"""
    try:
        # Try loading the model
        classifier = pipeline(task, model=model_name)
        logging.info(f"Successfully loaded {model_name}")
        return classifier
    
    except Exception as e:
        logging.error(f"Failed to load {model_name}: {e}")
        
        # Fallback to default model
        try:
            fallback_models = {
                "sentiment-analysis": "distilbert-base-uncased-finetuned-sst-2-english",
                "text-generation": "gpt2",
                "question-answering": "distilbert-base-cased-distilled-squad"
            }
            
            fallback = fallback_models.get(task)
            if fallback:
                classifier = pipeline(task, model=fallback)
                logging.info(f"Loaded fallback model: {fallback}")
                return classifier
        
        except Exception as fallback_error:
            logging.error(f"Fallback also failed: {fallback_error}")
            return None

# Usage
classifier = safe_model_loading("bert-base-uncased", "sentiment-analysis")
if classifier:
    result = classifier("This is a test")
    print(result)
```

### Memory Management
```python
import torch
import gc

def optimize_memory():
    """Optimize GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print memory stats
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Use context manager for model inference
class ModelContext:
    def __init__(self, model_name, task="sentiment-analysis"):
        self.model_name = model_name
        self.task = task
        self.pipeline = None
    
    def __enter__(self):
        self.pipeline = pipeline(self.task, model=self.model_name)
        return self.pipeline
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.pipeline
        optimize_memory()

# Usage
with ModelContext("bert-base-uncased") as classifier:
    results = classifier(["Text 1", "Text 2", "Text 3"])
    print(results)
# Model is automatically cleaned up
```

### Performance Monitoring
```python
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor model performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.4f} seconds")
        
        return result
    return wrapper

@monitor_performance
def classify_texts(texts):
    classifier = pipeline("sentiment-analysis")
    return classifier(texts)

# Usage
texts = ["Great!", "Bad!", "Okay"] * 100
results = classify_texts(texts)
```

These examples showcase the most popular Hugging Face models (BERT for understanding and GPT-2 for generation) with practical implementations, optimization techniques, and best practices for real-world usage.