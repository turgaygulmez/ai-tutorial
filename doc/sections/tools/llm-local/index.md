# Running LLMs Locally

- [Introduction](#introduction)
- [Popular Local LLM Tools](#popular-local-llm-tools)
- [Ollama](#ollama)
- [LM Studio](#lm-studio)
- [Hardware Requirements](#hardware-requirements)
- [Model Selection](#model-selection)
- [API Integration](#api-integration)

## Introduction

Running Large Language Models (LLMs) locally offers several advantages:
- **Privacy**: Your data never leaves your machine
- **Cost**: No API fees after initial setup
- **Control**: Full control over the model and its parameters
- **Offline Access**: Work without internet connectivity
- **Customization**: Fine-tune models for specific use cases

## Popular Local LLM Tools

### Ollama

Ollama is a popular tool for running open-source LLMs locally. It provides a simple command-line interface and API.

**Installation:**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

**Running a model:**
```bash
# Pull and run Llama 2
ollama run llama2

# Pull and run Mistral
ollama run mistral

# List available models
ollama list
```

**API Usage:**
```bash
# Ollama serves an API on localhost:11434
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why is the sky blue?"
}'
```

### LM Studio

LM Studio provides a user-friendly GUI for downloading and running LLMs locally.

**Features:**
- Visual model browser
- One-click model downloads
- Built-in chat interface
- API server mode
- Performance monitoring

**Getting Started:**
1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Browse and download models from the UI
3. Load a model and start chatting
4. Enable API server for programmatic access

## Hardware Requirements

Running LLMs locally requires significant computational resources:

### Minimum Requirements
- **RAM**: 8GB (for small models like 7B parameters)
- **Storage**: 10-50GB free space
- **CPU**: Modern multi-core processor

### Recommended Requirements
- **RAM**: 16-32GB
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 100GB+ SSD storage
- **CPU**: Recent Intel/AMD processor

### Model Size Guidelines
| Model Size | RAM Required | VRAM Required |
|------------|--------------|---------------|
| 7B params  | 8-16GB       | 6-8GB         |
| 13B params | 16-32GB      | 10-16GB       |
| 30B params | 32-64GB      | 24GB+         |
| 70B params | 64GB+        | 48GB+         |

## Model Selection

Popular open-source models for local deployment:

### General Purpose
- **Llama 2/3**: Meta's open-source models
- **Mistral**: High-performance French model
- **Mixtral**: Mixture of experts model

### Code Generation
- **CodeLlama**: Llama fine-tuned for code
- **DeepSeek Coder**: Specialized coding model
- **WizardCoder**: Instruction-tuned coding model

### Specialized Models
- **Orca**: Reasoning-focused model
- **Vicuna**: Chat-optimized model
- **Alpaca**: Instruction-following model

## API Integration

Most local LLM tools provide OpenAI-compatible APIs:

### JavaScript Example
```javascript
// Using Ollama's API
const response = await fetch('http://localhost:11434/api/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'llama2',
    prompt: 'Write a haiku about programming',
    stream: false
  })
});

const data = await response.json();
console.log(data.response);
```

### Python Example
```python
import requests

response = requests.post('http://localhost:11434/api/generate', 
    json={
        'model': 'llama2',
        'prompt': 'Explain quantum computing',
        'stream': False
    }
)

print(response.json()['response'])
```

### Using with LangChain
```javascript
import { Ollama } from "langchain/llms/ollama";

const ollama = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "llama2",
});

const response = await ollama.call("What is the meaning of life?");
console.log(response);
```

## Best Practices

1. **Model Selection**: Start with smaller models (7B) and upgrade as needed
2. **Resource Monitoring**: Keep an eye on RAM/VRAM usage
3. **Model Quantization**: Use quantized models (e.g., GGUF format) for better performance
4. **Prompt Engineering**: Optimize prompts for your specific model
5. **Temperature Settings**: Adjust temperature for creativity vs consistency
6. **Context Length**: Be aware of model context limitations

## Troubleshooting

Common issues and solutions:

- **Out of Memory**: Use smaller or quantized models
- **Slow Generation**: Enable GPU acceleration if available
- **API Connection Failed**: Check if the service is running
- **Model Download Failed**: Verify internet connection and disk space

## Further Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [LM Studio Guide](https://lmstudio.ai/docs)
- [LocalAI Project](https://localai.io/)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)