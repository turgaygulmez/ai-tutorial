# LLaVA - Large Language and Vision Assistant

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Variants](#model-variants)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Fine-tuning](#fine-tuning)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)

## Overview

LLaVA (Large Language and Vision Assistant) is a multimodal AI model that combines visual understanding with natural language processing. It enables AI systems to see, understand, and reason about images while engaging in natural conversations about visual content.

**Key Characteristics:**
- **Multimodal**: Processes both images and text simultaneously
- **Open Source**: Freely available for research and commercial use
- **Instruction Tuned**: Optimized for following visual instructions
- **Scalable**: Available in multiple model sizes

## Key Features

### 1. Visual Understanding
- Image comprehension and analysis
- Object detection and recognition
- Scene understanding and description
- Spatial reasoning about visual elements

### 2. Visual Question Answering
- Answer questions about image content
- Detailed explanations of visual scenes
- Comparison between multiple images
- Visual reasoning and inference

### 3. Multimodal Conversation
- Natural dialogue about images
- Context-aware responses
- Follow-up questions and clarifications
- Integration of visual and textual information

### 4. Instruction Following
- Execute visual tasks based on text instructions
- Image editing guidance
- Creative visual content generation
- Educational visual explanations

## Model Variants

### LLaVA-1.5
- **LLaVA-1.5-7B**: Efficient model for general use
- **LLaVA-1.5-13B**: Enhanced performance model
- **LLaVA-1.5-34B**: High-capacity model for complex tasks

### LLaVA-1.6 (Next)
- **LLaVA-Next-7B**: Latest improvements in efficiency
- **LLaVA-Next-13B**: Enhanced reasoning capabilities
- **LLaVA-Next-34B**: State-of-the-art performance

### Specialized Variants
- **LLaVA-Med**: Medical image analysis
- **LLaVA-RLHF**: Reinforcement learning fine-tuned
- **LLaVA-Instruct**: Instruction-following optimized

## Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# CUDA support (recommended for GPU acceleration)
nvidia-smi

# Git for cloning repositories
git --version
```

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

# Create virtual environment
conda create -n llava python=3.10 -y
conda activate llava

# Install dependencies
pip install --upgrade pip
pip install -e .

# Install additional dependencies
pip install -e ".[train]"
```

### Docker Installation
```bash
# Pull official Docker image
docker pull liuhaotian/llava:v1.6

# Run with GPU support
docker run --gpus all -it \
  -v $(pwd):/workspace \
  liuhaotian/llava:v1.6 \
  bash
```

### Model Download
```bash
# Download model weights
huggingface-cli download liuhaotian/llava-v1.5-7b

# Or use git
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
```

## Quick Start

### Basic Image Analysis
```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# Load model
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

# Analyze image
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": "What's in this image?",
    "conv_mode": None,
    "image_file": "path/to/your/image.jpg",
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
```

### Command Line Interface
```bash
# Basic usage
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file path/to/image.jpg

# Interactive mode
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --load-8bit
```

### Web Interface
```bash
# Launch web demo
python -m llava.serve.controller --host 0.0.0.0 --port 10000

# Launch gradio web server
python -m llava.serve.gradio_web_server \
    --controller http://localhost:10000 \
    --model-list-mode reload \
    --share
```

## Usage Examples

### Visual Question Answering
```python
import requests
from PIL import Image
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

def ask_about_image(image_path, question, model_path="liuhaotian/llava-v1.5-7b"):
    # Disable torch init
    disable_torch_init()
    
    # Load model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, "llava-v1.5-7b"
    )
    
    # Load and process image
    image = Image.open(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    
    # Prepare conversation
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{question}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()
    
    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            top_p=0.7,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True
        )
    
    # Decode response
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return outputs.strip()

# Example usage
response = ask_about_image(
    "landscape.jpg", 
    "Describe the weather conditions in this landscape"
)
print(response)
```

### Batch Processing
```python
def process_images_batch(image_paths, questions):
    """Process multiple images with questions in batch"""
    
    results = []
    
    for image_path, question in zip(image_paths, questions):
        try:
            response = ask_about_image(image_path, question)
            results.append({
                'image': image_path,
                'question': question,
                'answer': response,
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'image': image_path,
                'question': question,
                'answer': None,
                'status': 'error',
                'error': str(e)
            })
    
    return results

# Batch processing example
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
questions = [
    "What objects are in this image?",
    "What is the mood of this scene?",
    "Count the number of people in this image"
]

results = process_images_batch(image_paths, questions)
for result in results:
    print(f"Image: {result['image']}")
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print("-" * 50)
```

### Image Comparison
```python
def compare_images(image1_path, image2_path, comparison_question):
    """Compare two images based on a question"""
    
    # Process first image
    response1 = ask_about_image(image1_path, comparison_question)
    
    # Process second image  
    response2 = ask_about_image(image2_path, comparison_question)
    
    # Ask for comparison
    comparison_prompt = f"""
    Based on the following descriptions:
    Image 1: {response1}
    Image 2: {response2}
    
    {comparison_question} Compare and contrast these two images.
    """
    
    # You could use a text-only model for the comparison or extend LLaVA
    return {
        'image1_analysis': response1,
        'image2_analysis': response2,
        'comparison_request': comparison_question
    }

# Example usage
comparison = compare_images(
    "photo1.jpg", 
    "photo2.jpg",
    "Which image shows better lighting conditions?"
)
```

## Fine-tuning

### Preparing Training Data
```python
# Data format for fine-tuning
training_data = [
    {
        "id": "sample_1",
        "image": "path/to/image1.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat is shown in this medical scan?"
            },
            {
                "from": "gpt", 
                "value": "This appears to be a chest X-ray showing..."
            }
        ]
    }
]

# Save as JSON
import json
with open('training_data.json', 'w') as f:
    json.dump(training_data, f, indent=2)
```

### Training Script
```bash
#!/bin/bash

# Fine-tuning script
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path ./training_data.json \
    --image_folder ./images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-custom \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

### Evaluation
```python
# Evaluate fine-tuned model
def evaluate_model(model_path, test_data):
    """Evaluate model performance on test dataset"""
    
    # Load fine-tuned model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, "llava-custom"
    )
    
    results = []
    for item in test_data:
        image_path = item['image']
        question = item['question']
        ground_truth = item['answer']
        
        predicted_answer = ask_about_image(image_path, question)
        
        results.append({
            'question': question,
            'ground_truth': ground_truth,
            'prediction': predicted_answer,
            'image': image_path
        })
    
    return results

# Calculate metrics
def calculate_metrics(results):
    """Calculate evaluation metrics"""
    # Implement BLEU, ROUGE, or other relevant metrics
    pass
```

## Deployment

### API Server
```python
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model once at startup
tokenizer, model, image_processor, context_len = load_pretrained_model(
    "liuhaotian/llava-v1.5-7b", None, "llava-v1.5-7b"
)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        question = data['question']
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Save temporarily and process
        temp_path = "/tmp/temp_image.jpg"
        image.save(temp_path)
        
        response = ask_about_image(temp_path, question)
        
        return jsonify({
            'answer': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### Docker Deployment
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone and install LLaVA
RUN git clone https://github.com/haotian-liu/LLaVA.git
WORKDIR /app/LLaVA

RUN pip install --upgrade pip
RUN pip install -e .

# Copy application code
COPY api_server.py /app/
COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

EXPOSE 8000

CMD ["python", "/app/api_server.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llava-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llava-api
  template:
    metadata:
      labels:
        app: llava-api
    spec:
      containers:
      - name: llava-api
        image: your-registry/llava-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4" 
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: "liuhaotian/llava-v1.5-7b"
---
apiVersion: v1
kind: Service
metadata:
  name: llava-api-service
spec:
  selector:
    app: llava-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## API Reference

### Python API
```python
# Core functions
from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import eval_model

# Load model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="model_path",
    model_base=None,
    model_name="model_name"
)

# Evaluate model
eval_model(args)
```

### REST API Endpoints
```bash
# Analyze image
POST /analyze
{
    "image": "base64_encoded_image",
    "question": "What do you see in this image?"
}

# Health check
GET /health

# Model info
GET /model/info
```

### WebSocket API
```javascript
// Real-time image analysis
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function(event) {
    // Send image and question
    ws.send(JSON.stringify({
        type: 'analyze',
        image: base64Image,
        question: 'Describe this image'
    }));
};

ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    console.log('Analysis:', response.answer);
};
```

## Advanced Usage

### Custom Vision Encoders
```python
# Use different vision encoders
custom_args = {
    "model_path": "liuhaotian/llava-v1.5-7b",
    "vision_tower": "openai/clip-vit-large-patch14-336",  # Custom vision encoder
    "mm_projector_type": "mlp2x_gelu",
    "image_aspect_ratio": "pad"
}
```

### Multi-turn Conversations
```python
def multi_turn_conversation(image_path, conversation_history):
    """Handle multi-turn conversations about an image"""
    
    conv = conv_templates["llava_v1"].copy()
    
    # Add conversation history
    for turn in conversation_history:
        if turn['role'] == 'user':
            if turn['turn'] == 0:  # First turn includes image
                conv.append_message(conv.roles[0], f"<image>\n{turn['message']}")
            else:
                conv.append_message(conv.roles[0], turn['message'])
        else:
            conv.append_message(conv.roles[1], turn['message'])
    
    # Add new user message
    conv.append_message(conv.roles[1], None)
    
    # Generate response
    # ... (similar to previous examples)
    
    return response
```

### Batch Inference Optimization
```python
def optimized_batch_inference(image_batch, questions_batch):
    """Optimized batch processing for multiple images"""
    
    # Process images in parallel
    from concurrent.futures import ThreadPoolExecutor
    import torch.multiprocessing as mp
    
    def process_single(image_question_pair):
        image_path, question = image_question_pair
        return ask_about_image(image_path, question)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single, zip(image_batch, questions_batch)))
    
    return results
```

## Use Cases and Applications

### Medical Image Analysis
```python
# Medical imaging use case
medical_questions = [
    "What anatomical structures are visible in this medical scan?",
    "Are there any abnormalities or concerning features?",
    "What type of medical imaging technique was used?",
    "Describe the image quality and any artifacts present"
]

def analyze_medical_image(image_path):
    results = {}
    for question in medical_questions:
        results[question] = ask_about_image(image_path, question)
    return results
```

### Educational Content Creation
```python
# Educational applications
def create_educational_content(image_path, subject_area):
    educational_prompts = {
        'biology': "Explain the biological processes shown in this image",
        'physics': "Describe the physical phenomena demonstrated here",
        'chemistry': "Identify the chemical structures or processes visible",
        'history': "Provide historical context for what's shown in this image"
    }
    
    prompt = educational_prompts.get(subject_area, "Explain what's happening in this image")
    explanation = ask_about_image(image_path, prompt)
    
    # Generate follow-up questions
    follow_up = ask_about_image(image_path, 
        "Generate 3 educational questions about this image for students")
    
    return {
        'explanation': explanation,
        'study_questions': follow_up
    }
```

### Accessibility Applications
```python
# Visual accessibility
def describe_for_accessibility(image_path):
    """Generate detailed descriptions for visually impaired users"""
    
    descriptions = {}
    
    # General description
    descriptions['general'] = ask_about_image(image_path,
        "Provide a detailed description of this image for someone who cannot see it")
    
    # Text content
    descriptions['text'] = ask_about_image(image_path,
        "Is there any text in this image? If so, please read it aloud")
    
    # Navigation assistance
    descriptions['navigation'] = ask_about_image(image_path,
        "Describe the layout and organization of elements in this image")
    
    return descriptions
```

## Performance Optimization

### Model Quantization
```python
# 8-bit quantization for memory efficiency
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True
)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="liuhaotian/llava-v1.5-7b",
    model_base=None,
    model_name="llava-v1.5-7b",
    load_8bit=True,
    quantization_config=quantization_config
)
```

### GPU Memory Management
```python
import torch

# Clear GPU cache
torch.cuda.empty_cache()

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)

# Monitor GPU memory
def monitor_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Use smaller batch sizes or 8-bit quantization
   # Clear GPU cache between inferences
   torch.cuda.empty_cache()
   ```

2. **Model Loading Issues**
   ```python
   # Ensure model weights are properly downloaded
   # Check HuggingFace token for private models
   ```

3. **Image Processing Errors**
   ```python
   # Verify image format and size
   # Handle corrupted images gracefully
   ```

### Performance Tips

- Use GPU acceleration when available
- Implement proper batching for multiple images
- Cache model weights in memory
- Use quantization for deployment
- Optimize image preprocessing pipeline

## Community and Resources

- **Official Website**: [llava-vl.github.io](https://llava-vl.github.io/)
- **GitHub Repository**: [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
- **Paper**: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- **Hugging Face**: [LLaVA Models](https://huggingface.co/liuhaotian)
- **Demo**: Online interactive demo

## Limitations

- Requires significant GPU memory for inference
- Processing time increases with image complexity
- May hallucinate details not present in images
- Limited by training data coverage
- Computationally intensive for real-time applications