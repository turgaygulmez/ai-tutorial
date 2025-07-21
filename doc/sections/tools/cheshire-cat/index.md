# Cheshire Cat AI - Customizable AI Assistant

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Memory System](#memory-system)
- [Plugin System](#plugin-system)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Development Guide](#development-guide)

## Overview

Cheshire Cat AI is an open-source, customizable AI assistant framework that allows you to create personalized AI companions with persistent memory, plugin support, and multi-modal capabilities. Named after the mysterious cat from Alice in Wonderland, it provides a flexible and extensible platform for building intelligent conversational agents.

**Key Philosophy:**
- **Customizable**: Fully customizable personality and behavior
- **Memory-Driven**: Persistent long-term and episodic memory
- **Plugin Architecture**: Extensible through plugins and tools
- **Open Source**: Transparent and community-driven development

## Key Features

### 1. Persistent Memory
- Long-term memory storage
- Episodic memory for conversations
- Declarative memory for facts and knowledge
- Procedural memory for skills and behaviors

### 2. Plugin Ecosystem
- Rich plugin marketplace
- Custom plugin development
- Tool integration capabilities
- External API connections

### 3. Multi-Modal Support
- Text conversation
- Document processing
- Image understanding
- Voice interaction (with plugins)

### 4. Customizable Personality
- Adjustable AI behavior
- Custom prompt templates
- Role-playing capabilities
- Context-aware responses

## Installation

### Prerequisites
```bash
# Python 3.10 or higher
python --version

# Docker (recommended)
docker --version

# Git for cloning repositories
git --version
```

### Docker Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/cheshire-cat-ai/core.git cheshire-cat
cd cheshire-cat

# Start with Docker Compose
docker compose up -d

# Access the admin panel at http://localhost:1865
# Chat interface at http://localhost:1865/public
```

### Python Installation
```bash
# Clone the repository
git clone https://github.com/cheshire-cat-ai/core.git cheshire-cat
cd cheshire-cat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Cat
python main.py
```

### Development Setup
```bash
# Clone with development dependencies
git clone https://github.com/cheshire-cat-ai/core.git cheshire-cat
cd cheshire-cat

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Quick Start

### Basic Interaction
```python
import requests
import json

# Cat API endpoint
CAT_API = "http://localhost:1865"

# Send a message to the Cat
def chat_with_cat(message):
    response = requests.post(
        f"{CAT_API}/message",
        json={"text": message},
        headers={"Content-Type": "application/json"}
    )
    return response.json()

# Example conversation
response = chat_with_cat("Hello, Cheshire Cat!")
print(response["content"])
```

### WebSocket Connection
```javascript
// Connect via WebSocket for real-time chat
const ws = new WebSocket('ws://localhost:1865/ws');

ws.onopen = function(event) {
    console.log('Connected to Cheshire Cat');
    
    // Send a message
    ws.send(JSON.stringify({
        text: "What can you help me with today?"
    }));
};

ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    console.log('Cat says:', response.content);
};
```

### Python SDK Usage
```python
from cheshire_cat_api import CheshireCatApi

# Initialize the Cat client
cat = CheshireCatApi(
    base_url="http://localhost:1865",
    auth_key="your-auth-key"  # If authentication is enabled
)

# Send message
response = cat.send_message("Tell me about quantum physics")
print(response.content)

# Upload document
with open("document.pdf", "rb") as file:
    cat.upload_document(file)

# Get conversation history
history = cat.get_conversation_history()
```

## Core Components

### 1. Large Language Model (LLM)
The Cat supports multiple LLM providers:

```python
# OpenAI Configuration
cat_config = {
    "llm": {
        "provider": "openai",
        "model": "gpt-4",
        "api_key": "your-openai-key",
        "temperature": 0.7,
        "max_tokens": 2000
    }
}

# Anthropic Claude
cat_config = {
    "llm": {
        "provider": "anthropic",
        "model": "claude-3-sonnet",
        "api_key": "your-anthropic-key"
    }
}

# Local LLM (Ollama)
cat_config = {
    "llm": {
        "provider": "ollama",
        "model": "llama2",
        "base_url": "http://localhost:11434"
    }
}
```

### 2. Embeddings Model
For semantic search and memory retrieval:

```python
embeddings_config = {
    "embedder": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "your-openai-key"
    }
}
```

### 3. Vector Database
For storing and retrieving memories:

```python
vector_db_config = {
    "vector_store": {
        "provider": "qdrant",
        "host": "localhost",
        "port": 6333,
        "collection_name": "cat_memory"
    }
}
```

## Memory System

### Types of Memory

#### Episodic Memory
Stores conversation history and interactions:

```python
# Access episodic memory
episodic_memories = cat.memory.episodic.get_memories(
    query="What did we talk about yesterday?",
    k=10  # Number of memories to retrieve
)

for memory in episodic_memories:
    print(f"Timestamp: {memory.timestamp}")
    print(f"Content: {memory.content}")
```

#### Declarative Memory
Stores facts and general knowledge:

```python
# Add declarative memory
cat.memory.declarative.add_memory(
    content="The user's favorite color is blue",
    metadata={"type": "preference", "user_id": "user123"}
)

# Retrieve declarative memories
facts = cat.memory.declarative.get_memories(
    query="user preferences",
    k=5
)
```

#### Procedural Memory
Stores tools and procedures:

```python
# Tools are automatically stored in procedural memory
@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> str:
    """Calculate tip amount for a bill"""
    tip = (bill_amount * tip_percentage) / 100
    total = bill_amount + tip
    return f"Tip: ${tip:.2f}, Total: ${total:.2f}"

# The Cat will remember this tool and use it when needed
```

### Memory Management

```python
# Memory operations
from cheshire_cat.memory import MemoryManager

memory_manager = MemoryManager()

# Search across all memory types
results = memory_manager.search_memory(
    query="cooking recipes",
    memory_types=["episodic", "declarative"],
    k=20
)

# Clear specific memories
memory_manager.delete_memory(memory_id="specific-memory-id")

# Export memory for backup
memory_export = memory_manager.export_memory()

# Import memory from backup
memory_manager.import_memory(memory_export)
```

## Plugin System

### Installing Plugins

```python
# Via the admin interface or API
import requests

# Install plugin from registry
response = requests.post(
    "http://localhost:1865/plugins/install",
    json={"plugin_name": "web_search"}
)

# Install from URL
response = requests.post(
    "http://localhost:1865/plugins/install",
    json={"plugin_url": "https://github.com/user/cat-plugin.git"}
)

# Install local plugin
response = requests.post(
    "http://localhost:1865/plugins/upload",
    files={"file": open("my_plugin.zip", "rb")}
)
```

### Creating Custom Plugins

#### Basic Plugin Structure
```python
# my_plugin.py
from cheshire_cat.mad_hatter.decorators import tool, hook
from pydantic import BaseModel

class PluginSettings(BaseModel):
    api_key: str = ""
    enabled: bool = True

@tool
def my_custom_tool(query: str, cat) -> str:
    """A custom tool that does something useful"""
    settings = cat.mad_hatter.get_plugin().load_settings()
    
    # Your custom logic here
    result = f"Processing: {query}"
    return result

@hook
def before_cat_sends_message(message, cat):
    """Modify message before sending"""
    # Add custom processing
    if "weather" in message["content"].lower():
        message["content"] += " 🌤️"
    return message

@hook
def agent_prompt_prefix(prefix, cat):
    """Customize the Cat's personality"""
    custom_prefix = """
    You are a helpful AI assistant with expertise in data analysis.
    You speak in a friendly but professional tone.
    """
    return custom_prefix
```

#### Plugin Metadata
```json
# plugin.json
{
    "name": "My Custom Plugin",
    "version": "1.0.0",
    "description": "A plugin that adds custom functionality",
    "author": "Your Name",
    "tags": ["utility", "custom"],
    "requirements": ["requests>=2.28.0"],
    "settings_schema": {
        "api_key": {
            "type": "string",
            "title": "API Key",
            "description": "Your API key"
        }
    }
}
```

### Popular Plugins

#### Web Search Plugin
```python
@tool
def web_search(query: str, cat) -> str:
    """Search the web for current information"""
    # Implementation using search API
    results = search_engine.search(query, num_results=5)
    return format_search_results(results)
```

#### Document QA Plugin
```python
@tool  
def document_qa(question: str, document_path: str, cat) -> str:
    """Answer questions about uploaded documents"""
    # Load and process document
    document = load_document(document_path)
    answer = qa_chain.run(question=question, context=document)
    return answer
```

## Configuration

### Core Configuration
```yaml
# cat_config.yml
llm:
  provider: "openai"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.7
  max_tokens: 2000

embedder:
  provider: "openai"
  model: "text-embedding-3-small"
  api_key: "${OPENAI_API_KEY}"

vector_store:
  provider: "qdrant"
  host: "localhost"
  port: 6333

memory:
  episodic_memory_k: 20
  declarative_memory_k: 10
  procedural_memory_k: 5

plugins:
  auto_install: true
  plugin_folder: "./plugins"
```

### Environment Variables
```bash
# .env file
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
QDRANT_HOST=localhost
QDRANT_PORT=6333
CAT_DEBUG=false
CAT_LOG_LEVEL=INFO
```

### Advanced Settings
```python
# Advanced configuration via Python
from cheshire_cat.core.cat import CheshireCat

cat = CheshireCat()

# Configure memory settings
cat.memory_config = {
    "episodic_memory_k": 30,
    "declarative_memory_k": 15,
    "memory_threshold": 0.7
}

# Configure LLM settings
cat.llm_config = {
    "temperature": 0.8,
    "max_tokens": 1500,
    "presence_penalty": 0.1,
    "frequency_penalty": 0.1
}
```

## API Reference

### REST API Endpoints

#### Messages
```bash
# Send message
POST /message
{
    "text": "Hello Cat!",
    "user_id": "user123"
}

# Get conversation history
GET /memory/conversation_history?user_id=user123

# Clear conversation history  
DELETE /memory/conversation_history?user_id=user123
```

#### Memory Management
```bash
# Get memory collections
GET /memory/collections

# Search memory
POST /memory/recall
{
    "text": "search query",
    "k": 10,
    "metadata": {"user_id": "user123"}
}

# Add point to memory
POST /memory/collections/{collection}/points
{
    "content": "New memory content",
    "metadata": {"source": "user_input"}
}
```

#### Plugin Management
```bash
# List installed plugins
GET /plugins

# Install plugin
POST /plugins/install
{
    "plugin_name": "web_search"
}

# Configure plugin
PUT /plugins/{plugin_id}/settings
{
    "api_key": "your-api-key",
    "enabled": true
}
```

### WebSocket Events

#### Client to Server
```javascript
// Send message
ws.send(JSON.stringify({
    type: "message",
    text: "Hello Cat!",
    user_id: "user123"
}));

// Request memory search
ws.send(JSON.stringify({
    type: "memory_search", 
    query: "cooking recipes",
    k: 10
}));
```

#### Server to Client
```javascript
// Message response
{
    type: "message",
    content: "Hello! How can I help you?",
    why: {
        "input": "Hello Cat!",
        "memory": [...],
        "tools": [...]
    }
}

// Memory search results
{
    type: "memory_results",
    results: [...],
    total: 25
}
```

## Development Guide

### Setting Up Development Environment
```bash
# Clone and setup development environment
git clone https://github.com/cheshire-cat-ai/core.git
cd core

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Testing
```python
# test_cat.py
import pytest
from cheshire_cat.core.cat import CheshireCat

@pytest.fixture
def cat():
    return CheshireCat()

def test_cat_response(cat):
    response = cat.send_message("Hello")
    assert response["content"] is not None
    assert len(response["content"]) > 0

def test_memory_storage(cat):
    # Test memory functionality
    cat.send_message("My name is Alice")
    response = cat.send_message("What's my name?")
    assert "Alice" in response["content"]
```

### Debugging
```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

from cheshire_cat.core.cat import CheshireCat

cat = CheshireCat(debug=True)

# Debug memory operations
cat.memory.debug = True

# Debug LLM calls
cat.llm.debug = True
```

### Performance Optimization
```python
# Optimize memory retrieval
cat.memory_config.update({
    "episodic_memory_k": 10,  # Reduce memory retrieval
    "memory_threshold": 0.8,  # Higher threshold for relevance
    "cache_embeddings": True  # Cache embeddings
})

# Optimize LLM calls
cat.llm_config.update({
    "max_tokens": 1000,  # Reduce max tokens
    "stream": True,      # Enable streaming
    "cache_responses": True  # Cache responses
})
```

## Use Cases and Examples

### Personal Assistant
```python
# Configure as personal assistant
@hook
def agent_prompt_prefix(prefix, cat):
    return """
    You are a personal AI assistant. You help with:
    - Scheduling and reminders
    - Email and document management  
    - Research and information gathering
    - Task organization and productivity
    
    You have access to the user's conversation history and can remember
    their preferences and past interactions.
    """
```

### Customer Support Bot
```python
# Customer support configuration
@tool
def lookup_order_status(order_id: str, cat) -> str:
    """Look up the status of a customer order"""
    # Integration with order management system
    status = order_system.get_status(order_id)
    return f"Order {order_id} status: {status}"

@hook
def before_cat_sends_message(message, cat):
    # Add support ticket tracking
    if "ticket" in message["content"].lower():
        # Create or update support ticket
        pass
    return message
```

### Educational Tutor
```python
@tool
def generate_quiz(topic: str, difficulty: str, cat) -> str:
    """Generate a quiz on a specific topic"""
    # Generate educational content
    quiz = education_api.create_quiz(topic, difficulty)
    return format_quiz(quiz)

@hook  
def agent_prompt_prefix(prefix, cat):
    return """
    You are an educational tutor. Your role is to:
    - Explain concepts clearly and patiently
    - Provide examples and practice problems
    - Adapt your teaching style to the student's level
    - Encourage learning and curiosity
    """
```

## Community and Resources

- **Official Website**: [cheshirecat.ai](https://cheshirecat.ai)
- **GitHub Repository**: [cheshire-cat-ai/core](https://github.com/cheshire-cat-ai/core)
- **Documentation**: [Cheshire Cat Docs](https://cheshire-cat-ai.github.io/docs/)
- **Discord Community**: Join the Cheshire Cat Discord
- **Plugin Registry**: Browse available plugins
- **YouTube Channel**: Tutorials and demos

## Troubleshooting

### Common Issues

1. **Memory Issues**: Clear vector database if corrupted
2. **Plugin Conflicts**: Disable conflicting plugins  
3. **LLM Connection**: Verify API keys and endpoints
4. **Performance**: Optimize memory settings and model size

### Performance Tips

- Use appropriate model sizes for your use case
- Implement memory pruning for long conversations
- Cache frequently accessed embeddings
- Use streaming for real-time responses

## License

Cheshire Cat AI is released under the GPL v3 License, promoting open-source development while ensuring contributions remain open.

## Limitations

- Requires continuous internet connection for cloud LLMs
- Memory storage grows over time and may need pruning
- Plugin quality varies and may affect stability
- Resource usage scales with conversation complexity