# DocsGPT - AI-Powered Documentation Assistant

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Document Integration](#document-integration)
- [Configuration](#configuration)
- [API Usage](#api-usage)
- [Customization](#customization)
- [Deployment](#deployment)
- [Advanced Features](#advanced-features)

## Overview

DocsGPT is an open-source AI-powered documentation assistant that helps users find information in their documentation quickly and accurately. It combines document indexing with GPT models to provide intelligent answers based on your specific documentation content.

**Key Characteristics:**
- **Document-Focused**: Specifically designed for documentation Q&A
- **Self-Hosted**: Complete control over your data and infrastructure
- **Multi-Format Support**: Works with various document formats
- **Web Interface**: User-friendly chat interface for querying documents
- **API Access**: RESTful API for integration with other systems

## Key Features

### 1. Intelligent Document Search
- Semantic search across documentation
- Context-aware question answering
- Source attribution and citations
- Multi-document knowledge synthesis

### 2. Multiple Document Formats
- PDF documents
- Markdown files
- Text files
- HTML pages
- Word documents
- Custom data sources

### 3. Flexible LLM Integration
- OpenAI GPT models
- Local model support
- Custom model configurations
- Multiple provider support

### 4. User-Friendly Interface
- Clean web chat interface
- Real-time responses
- Document source links
- Conversation history

### 5. Enterprise Features
- Self-hosted deployment
- Data privacy controls
- Custom branding
- API integration
- User management

## Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Docker (recommended)
docker --version

# Git
git --version

# Node.js (for frontend development)
node --version
```

### Docker Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/arc53/DocsGPT.git
cd DocsGPT

# Start with Docker Compose
docker-compose up -d

# Access the application
# Frontend: http://localhost:5173
# Backend API: http://localhost:7091
```

### Manual Installation
```bash
# Clone repository
git clone https://github.com/arc53/DocsGPT.git
cd DocsGPT

# Backend setup
cd application
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install

# Environment configuration
cp .env.example .env
# Edit .env with your configurations
```

### Production Setup
```bash
# Production Docker configuration
cp docker-compose.prod.yml docker-compose.yml
docker-compose up -d

# Or use Kubernetes
kubectl apply -f k8s/
```

## Getting Started

### Initial Configuration
```bash
# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"
export MONGO_URI="mongodb://localhost:27017/docsgpt"

# Or create .env file
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
MONGO_URI=mongodb://localhost:27017/docsgpt
EOF
```

### Basic Usage

#### Web Interface
1. Start the application with Docker Compose
2. Navigate to `http://localhost:5173`
3. Upload or configure your documents
4. Start asking questions about your documentation

#### First Document Upload
```bash
# Using the web interface:
# 1. Click "Upload Documents"
# 2. Select your files (PDF, MD, TXT)
# 3. Wait for processing
# 4. Start chatting with your documents

# Using API:
curl -X POST "http://localhost:7091/api/upload" \
  -F "file=@/path/to/your/document.pdf" \
  -F "name=My Document" \
  -H "Authorization: Bearer your-api-key"
```

## Document Integration

### Supported Formats
```python
# Supported document types
SUPPORTED_FORMATS = {
    '.pdf': 'PDF documents',
    '.txt': 'Plain text files',
    '.md': 'Markdown files',
    '.rst': 'reStructuredText files',
    '.docx': 'Microsoft Word documents',
    '.html': 'HTML files',
    '.csv': 'CSV files',
    '.json': 'JSON files'
}
```

### Document Processing Pipeline
```python
# Example document processing configuration
document_config = {
    'chunk_size': 1000,           # Size of text chunks
    'chunk_overlap': 200,         # Overlap between chunks
    'max_file_size': 50 * 1024 * 1024,  # 50MB max file size
    'supported_extensions': ['.pdf', '.txt', '.md', '.docx'],
    'embedding_model': 'text-embedding-ada-002',
    'vector_store': 'faiss'       # or 'pinecone', 'qdrant'
}

# Custom document processor
class CustomDocumentProcessor:
    def __init__(self, config):
        self.config = config
        
    def process_document(self, file_path, metadata=None):
        """Process a single document"""
        # Extract text
        text = self.extract_text(file_path)
        
        # Split into chunks
        chunks = self.split_text(text, 
                               chunk_size=self.config['chunk_size'],
                               overlap=self.config['chunk_overlap'])
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Store in vector database
        self.store_embeddings(chunks, embeddings, metadata)
        
        return {
            'status': 'success',
            'chunks_created': len(chunks),
            'file_path': file_path
        }
```

### Bulk Document Import
```python
import os
import requests
from pathlib import Path

def bulk_upload_documents(docs_directory, api_base_url, api_key):
    """Upload all documents from a directory"""
    
    results = []
    supported_extensions = ['.pdf', '.txt', '.md', '.docx']
    
    for root, dirs, files in os.walk(docs_directory):
        for file in files:
            file_path = Path(root) / file
            
            if file_path.suffix.lower() in supported_extensions:
                try:
                    # Upload document
                    with open(file_path, 'rb') as f:
                        files = {'file': f}
                        data = {
                            'name': file_path.stem,
                            'source': str(file_path.relative_to(docs_directory))
                        }
                        
                        response = requests.post(
                            f"{api_base_url}/api/upload",
                            files=files,
                            data=data,
                            headers={'Authorization': f'Bearer {api_key}'}
                        )
                        
                        if response.status_code == 200:
                            results.append({
                                'file': str(file_path),
                                'status': 'success',
                                'document_id': response.json().get('document_id')
                            })
                        else:
                            results.append({
                                'file': str(file_path),
                                'status': 'error',
                                'error': response.text
                            })
                
                except Exception as e:
                    results.append({
                        'file': str(file_path),
                        'status': 'error',
                        'error': str(e)
                    })
    
    return results

# Usage
results = bulk_upload_documents(
    './documentation',
    'http://localhost:7091',
    'your-api-key'
)

# Print summary
successful = len([r for r in results if r['status'] == 'success'])
failed = len([r for r in results if r['status'] == 'error'])
print(f"Upload complete: {successful} successful, {failed} failed")
```

### GitHub Integration
```python
import requests
import base64
from github import Github

class GitHubDocsImporter:
    def __init__(self, token, docsgpt_api_url, docsgpt_api_key):
        self.github = Github(token)
        self.docsgpt_api_url = docsgpt_api_url
        self.docsgpt_api_key = docsgpt_api_key
    
    def import_repo_docs(self, repo_name, docs_path='docs/'):
        """Import documentation from a GitHub repository"""
        
        repo = self.github.get_repo(repo_name)
        results = []
        
        try:
            contents = repo.get_contents(docs_path)
            
            while contents:
                file_content = contents.pop(0)
                
                if file_content.type == "dir":
                    contents.extend(repo.get_contents(file_content.path))
                else:
                    # Process markdown and text files
                    if file_content.name.endswith(('.md', '.txt', '.rst')):
                        try:
                            # Get file content
                            file_data = base64.b64decode(file_content.content).decode('utf-8')
                            
                            # Send to DocsGPT
                            response = requests.post(
                                f"{self.docsgpt_api_url}/api/upload_text",
                                json={
                                    'text': file_data,
                                    'name': file_content.name,
                                    'source': f"{repo_name}/{file_content.path}",
                                    'metadata': {
                                        'repository': repo_name,
                                        'path': file_content.path,
                                        'sha': file_content.sha
                                    }
                                },
                                headers={'Authorization': f'Bearer {self.docsgpt_api_key}'}
                            )
                            
                            if response.status_code == 200:
                                results.append({
                                    'file': file_content.path,
                                    'status': 'success'
                                })
                            else:
                                results.append({
                                    'file': file_content.path,
                                    'status': 'error',
                                    'error': response.text
                                })
                        
                        except Exception as e:
                            results.append({
                                'file': file_content.path,
                                'status': 'error',
                                'error': str(e)
                            })
        
        except Exception as e:
            print(f"Error accessing repository: {e}")
        
        return results

# Usage
importer = GitHubDocsImporter(
    token='github_token',
    docsgpt_api_url='http://localhost:7091',
    docsgpt_api_key='docsgpt_api_key'
)

results = importer.import_repo_docs('microsoft/typescript', 'docs/')
```

## Configuration

### Application Configuration
```python
# config.py
import os

class Config:
    # API Keys
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    
    # Database
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/docsgpt')
    
    # Redis
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # Vector Store
    VECTOR_STORE = os.environ.get('VECTOR_STORE', 'faiss')  # faiss, pinecone, qdrant
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')
    
    # Model Configuration
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'openai')  # openai, anthropic, local
    LLM_MODEL = os.environ.get('LLM_MODEL', 'gpt-3.5-turbo')
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'text-embedding-ada-002')
    
    # Application Settings
    MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 200))
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    JWT_SECRET = os.environ.get('JWT_SECRET', 'jwt-secret-key')
    
    # CORS
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5173').split(',')
```

### Docker Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  docsgpt-backend:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MONGO_URI=mongodb://mongo:27017/docsgpt
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
      - VECTOR_STORE=faiss
    ports:
      - "7091:7091"
    depends_on:
      - mongo
      - redis
    volumes:
      - ./uploads:/app/uploads
      - ./models:/app/models

  docsgpt-frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "5173:5173"
    environment:
      - VITE_API_HOST=http://localhost:7091
    depends_on:
      - docsgpt-backend

  docsgpt-worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A application.celery worker --loglevel=info
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MONGO_URI=mongodb://mongo:27017/docsgpt
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    depends_on:
      - mongo
      - redis
    volumes:
      - ./uploads:/app/uploads

  mongo:
    image: mongo:6
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  mongodb_data:
  redis_data:
```

## API Usage

### Authentication
```python
import requests

# Get API token
def authenticate(username, password, api_url):
    response = requests.post(f"{api_url}/api/auth/login", json={
        'username': username,
        'password': password
    })
    
    if response.status_code == 200:
        return response.json()['access_token']
    else:
        raise Exception(f"Authentication failed: {response.text}")

# Use token for API calls
api_token = authenticate("admin", "password", "http://localhost:7091")
headers = {'Authorization': f'Bearer {api_token}'}
```

### Document Operations
```python
# Upload document
def upload_document(file_path, name, api_url, headers):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'name': name}
        
        response = requests.post(
            f"{api_url}/api/upload",
            files=files,
            data=data,
            headers=headers
        )
    
    return response.json()

# List documents
def list_documents(api_url, headers):
    response = requests.get(f"{api_url}/api/documents", headers=headers)
    return response.json()

# Delete document
def delete_document(document_id, api_url, headers):
    response = requests.delete(
        f"{api_url}/api/documents/{document_id}",
        headers=headers
    )
    return response.status_code == 200

# Search documents
def search_documents(query, api_url, headers, limit=10):
    response = requests.post(
        f"{api_url}/api/search",
        json={
            'query': query,
            'limit': limit
        },
        headers=headers
    )
    return response.json()
```

### Chat API
```python
# Send message
def send_message(question, conversation_id=None, api_url="http://localhost:7091", headers=None):
    payload = {
        'question': question,
        'history': conversation_id,
        'api_key': 'your-api-key'
    }
    
    response = requests.post(f"{api_url}/api/answer", json=payload, headers=headers)
    return response.json()

# Stream response
def stream_message(question, api_url="http://localhost:7091", headers=None):
    payload = {
        'question': question,
        'stream': True,
        'api_key': 'your-api-key'
    }
    
    response = requests.post(
        f"{api_url}/api/stream", 
        json=payload, 
        headers=headers,
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            yield line.decode('utf-8')

# Usage examples
# Simple Q&A
answer = send_message("How do I install the application?")
print(f"Answer: {answer['answer']}")
print(f"Sources: {answer['sources']}")

# Streaming response
print("Streaming response:")
for chunk in stream_message("Explain the architecture of the system"):
    print(chunk, end='', flush=True)
```

### Python SDK
```python
class DocsGPTClient:
    def __init__(self, api_url, api_key=None):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def upload_document(self, file_path, name=None, metadata=None):
        """Upload a document"""
        name = name or Path(file_path).stem
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'name': name}
            
            if metadata:
                data['metadata'] = json.dumps(metadata)
            
            response = self.session.post(f"{self.api_url}/api/upload", files=files, data=data)
            response.raise_for_status()
            
        return response.json()
    
    def upload_text(self, text, name, source=None, metadata=None):
        """Upload text content directly"""
        payload = {
            'text': text,
            'name': name,
            'source': source,
            'metadata': metadata or {}
        }
        
        response = self.session.post(f"{self.api_url}/api/upload_text", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def ask_question(self, question, conversation_id=None):
        """Ask a question about the documents"""
        payload = {
            'question': question,
            'history': conversation_id
        }
        
        response = self.session.post(f"{self.api_url}/api/answer", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def search_documents(self, query, limit=10, filters=None):
        """Search through documents"""
        payload = {
            'query': query,
            'limit': limit,
            'filters': filters or {}
        }
        
        response = self.session.post(f"{self.api_url}/api/search", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def get_documents(self):
        """Get list of uploaded documents"""
        response = self.session.get(f"{self.api_url}/api/documents")
        response.raise_for_status()
        
        return response.json()
    
    def delete_document(self, document_id):
        """Delete a document"""
        response = self.session.delete(f"{self.api_url}/api/documents/{document_id}")
        response.raise_for_status()
        
        return response.status_code == 200

# Usage
client = DocsGPTClient('http://localhost:7091', 'your-api-key')

# Upload documentation
result = client.upload_document('./README.md', name='Project README')
print(f"Document uploaded: {result['document_id']}")

# Ask questions
answer = client.ask_question('How do I get started?')
print(f"Answer: {answer['answer']}")

# Search documents
results = client.search_documents('installation guide', limit=5)
for result in results['results']:
    print(f"- {result['title']}: {result['snippet']}")
```

## Customization

### Custom UI Theme
```css
/* custom-theme.css */
:root {
  --primary-color: #2563eb;
  --secondary-color: #64748b;
  --background-color: #ffffff;
  --text-color: #1f2937;
  --border-color: #e5e7eb;
  --chat-bubble-user: #3b82f6;
  --chat-bubble-assistant: #f3f4f6;
}

.custom-chat-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  background: var(--background-color);
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.custom-message-user {
  background: var(--chat-bubble-user);
  color: white;
  padding: 12px 16px;
  border-radius: 18px 18px 4px 18px;
  margin: 8px 0;
  align-self: flex-end;
}

.custom-message-assistant {
  background: var(--chat-bubble-assistant);
  color: var(--text-color);
  padding: 12px 16px;
  border-radius: 18px 18px 18px 4px;
  margin: 8px 0;
  align-self: flex-start;
}
```

### Custom Model Integration
```python
# custom_models.py
from application.llm.base import BaseLLM

class CustomModelProvider(BaseLLM):
    def __init__(self, api_key=None, model_name='custom-model'):
        self.api_key = api_key
        self.model_name = model_name
    
    def gen(self, model, engine, messages, stream=False, **kwargs):
        """Generate response using custom model"""
        
        # Prepare prompt
        prompt = self.format_messages(messages)
        
        # Call your custom model API
        response = self.call_custom_api(prompt, stream=stream, **kwargs)
        
        if stream:
            return self.stream_response(response)
        else:
            return {'choices': [{'message': {'content': response}}]}
    
    def call_custom_api(self, prompt, **kwargs):
        """Implement your custom model API call"""
        # Example implementation
        import requests
        
        response = requests.post(
            'https://your-model-api.com/generate',
            json={
                'prompt': prompt,
                'max_tokens': kwargs.get('max_tokens', 1000),
                'temperature': kwargs.get('temperature', 0.7)
            },
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        
        return response.json()['text']
    
    def format_messages(self, messages):
        """Format messages for your model"""
        formatted = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            formatted += f"{role}: {content}\n"
        return formatted

# Register custom model
from application.llm import llm_creator

llm_creator.register_llm('custom', CustomModelProvider)
```

### Plugin System
```python
# plugins/summarizer.py
class DocumentSummarizer:
    def __init__(self, docsgpt_client):
        self.client = docsgpt_client
    
    def summarize_document(self, document_id, max_length=200):
        """Generate a summary of a document"""
        
        # Get document content
        doc = self.client.get_document(document_id)
        
        # Generate summary
        question = f"Please provide a {max_length}-word summary of this document"
        response = self.client.ask_question(question, context_docs=[document_id])
        
        return {
            'document_id': document_id,
            'document_title': doc['title'],
            'summary': response['answer'],
            'word_count': len(response['answer'].split())
        }
    
    def bulk_summarize(self, document_ids):
        """Summarize multiple documents"""
        summaries = []
        
        for doc_id in document_ids:
            try:
                summary = self.summarize_document(doc_id)
                summaries.append(summary)
            except Exception as e:
                summaries.append({
                    'document_id': doc_id,
                    'error': str(e)
                })
        
        return summaries

# Usage
summarizer = DocumentSummarizer(docsgpt_client)
summary = summarizer.summarize_document('doc_123')
print(summary['summary'])
```

## Deployment

### Production Docker Setup
```bash
# Production environment
cat > .env.prod << EOF
OPENAI_API_KEY=your-production-openai-key
MONGO_URI=mongodb://mongo:27017/docsgpt_prod
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
VECTOR_STORE=pinecone
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=docsgpt-prod
JWT_SECRET=super-secret-jwt-key
CORS_ORIGINS=https://docs.yourcompany.com
EOF

# Production docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docsgpt-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: docsgpt-backend
  template:
    metadata:
      labels:
        app: docsgpt-backend
    spec:
      containers:
      - name: docsgpt-backend
        image: docsgpt/backend:latest
        ports:
        - containerPort: 7091
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: docsgpt-secrets
              key: openai-api-key
        - name: MONGO_URI
          value: "mongodb://mongo-service:27017/docsgpt"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: docsgpt-backend-service
spec:
  selector:
    app: docsgpt-backend
  ports:
  - port: 80
    targetPort: 7091
  type: LoadBalancer
```

### Nginx Configuration
```nginx
# nginx.conf
server {
    listen 80;
    server_name docs.yourcompany.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name docs.yourcompany.com;
    
    ssl_certificate /etc/ssl/certs/docsgpt.crt;
    ssl_certificate_key /etc/ssl/private/docsgpt.key;
    
    # Frontend
    location / {
        proxy_pass http://docsgpt-frontend:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # Backend API
    location /api {
        proxy_pass http://docsgpt-backend:7091;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeout for document processing
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    # WebSocket support for streaming
    location /ws {
        proxy_pass http://docsgpt-backend:7091;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Advanced Features

### Analytics and Monitoring
```python
# analytics.py
import logging
from datetime import datetime
from collections import defaultdict

class DocsGPTAnalytics:
    def __init__(self):
        self.query_stats = defaultdict(int)
        self.document_stats = defaultdict(int)
        self.response_times = []
        
    def track_query(self, query, response_time, found_answer=True):
        """Track user queries and performance"""
        
        # Log query statistics
        self.query_stats['total_queries'] += 1
        if found_answer:
            self.query_stats['successful_queries'] += 1
        
        # Track response time
        self.response_times.append(response_time)
        
        # Log to monitoring system
        logging.info({
            'event': 'query',
            'query_length': len(query),
            'response_time': response_time,
            'found_answer': found_answer,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def track_document_usage(self, document_id, document_title):
        """Track which documents are being accessed"""
        self.document_stats[document_id] += 1
        
        logging.info({
            'event': 'document_access',
            'document_id': document_id,
            'document_title': document_title,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def get_analytics_report(self):
        """Generate analytics report"""
        
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        success_rate = (self.query_stats['successful_queries'] / self.query_stats['total_queries'] * 100) if self.query_stats['total_queries'] > 0 else 0
        
        most_accessed_docs = sorted(
            self.document_stats.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            'total_queries': self.query_stats['total_queries'],
            'success_rate': f"{success_rate:.2f}%",
            'average_response_time': f"{avg_response_time:.2f}s",
            'most_accessed_documents': most_accessed_docs,
            'total_unique_documents_accessed': len(self.document_stats)
        }
```

### Integration Examples
```python
# slack_bot.py
from slack_sdk import WebClient
from slack_sdk.adapter.flask import SlackRequestHandler

class DocsGPTSlackBot:
    def __init__(self, slack_token, docsgpt_client):
        self.slack_client = WebClient(token=slack_token)
        self.docsgpt = docsgpt_client
    
    def handle_message(self, event):
        """Handle Slack messages mentioning the bot"""
        
        if 'bot_id' in event:
            return  # Ignore bot messages
        
        channel = event['channel']
        text = event['text']
        user = event['user']
        
        # Remove bot mention from text
        question = self.extract_question(text)
        
        if question:
            try:
                # Query DocsGPT
                response = self.docsgpt.ask_question(question)
                
                # Format response for Slack
                message = self.format_slack_response(response)
                
                # Send response
                self.slack_client.chat_postMessage(
                    channel=channel,
                    text=message,
                    thread_ts=event.get('ts')
                )
                
            except Exception as e:
                self.slack_client.chat_postMessage(
                    channel=channel,
                    text=f"Sorry, I encountered an error: {str(e)}",
                    thread_ts=event.get('ts')
                )
    
    def extract_question(self, text):
        """Extract question from Slack message"""
        # Remove bot mention
        import re
        question = re.sub(r'<@U[A-Z0-9]+>', '', text).strip()
        return question if len(question) > 3 else None
    
    def format_slack_response(self, response):
        """Format DocsGPT response for Slack"""
        
        message = f"📚 {response['answer']}\n\n"
        
        if response.get('sources'):
            message += "*Sources:*\n"
            for source in response['sources'][:3]:  # Limit to 3 sources
                message += f"• {source.get('title', 'Unknown')}\n"
        
        return message

# Discord bot
import discord
from discord.ext import commands

class DocsGPTDiscordBot(commands.Bot):
    def __init__(self, docsgpt_client):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!docs ', intents=intents)
        self.docsgpt = docsgpt_client
    
    @commands.command(name='ask')
    async def ask_question(self, ctx, *, question):
        """Ask a question about the documentation"""
        
        async with ctx.typing():
            try:
                response = self.docsgpt.ask_question(question)
                
                embed = discord.Embed(
                    title="📚 Documentation Assistant",
                    description=response['answer'][:2000],  # Discord embed limit
                    color=0x3498db
                )
                
                if response.get('sources'):
                    sources_text = '\n'.join([
                        f"• {source.get('title', 'Unknown')}"
                        for source in response['sources'][:5]
                    ])
                    embed.add_field(
                        name="Sources",
                        value=sources_text[:1024],  # Field value limit
                        inline=False
                    )
                
                await ctx.send(embed=embed)
                
            except Exception as e:
                await ctx.send(f"❌ Error: {str(e)}")
```

### Webhook Integration
```python
# webhooks.py
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/webhook/document-updated', methods=['POST'])
def handle_document_update():
    """Handle document update webhooks"""
    
    data = request.json
    document_id = data.get('document_id')
    action = data.get('action')  # 'created', 'updated', 'deleted'
    
    if action == 'updated':
        # Re-index document
        docsgpt_client.reindex_document(document_id)
    elif action == 'deleted':
        # Remove from index
        docsgpt_client.delete_document(document_id)
    elif action == 'created':
        # Add to index
        document_url = data.get('document_url')
        docsgpt_client.import_from_url(document_url)
    
    return jsonify({'status': 'success'})

@app.route('/webhook/slack', methods=['POST'])
def handle_slack_webhook():
    """Handle Slack webhook for document queries"""
    
    data = request.json
    
    if data.get('type') == 'url_verification':
        return data.get('challenge')
    
    if data.get('event', {}).get('type') == 'app_mention':
        # Handle Slack mention
        event = data['event']
        slack_bot.handle_message(event)
    
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Troubleshooting

### Common Issues

1. **Document Processing Failures**
   ```bash
   # Check document format
   file document.pdf
   
   # Verify file permissions
   ls -la uploads/
   
   # Check processing logs
   docker logs docsgpt-worker
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats
   
   # Increase Docker memory limits
   # In docker-compose.yml:
   mem_limit: 4g
   ```

3. **Vector Store Connection**
   ```python
   # Test vector store connection
   import pinecone
   
   pinecone.init(api_key='your-key', environment='your-env')
   index = pinecone.Index('docsgpt')
   stats = index.describe_index_stats()
   print(stats)
   ```

### Performance Optimization
```python
# Optimization tips
optimization_config = {
    'chunk_size': 800,  # Smaller chunks for better retrieval
    'chunk_overlap': 100,  # Reduce overlap to save space
    'max_retrieval_results': 5,  # Limit context size
    'embedding_batch_size': 100,  # Process in batches
    'cache_embeddings': True,  # Cache for repeated queries
    'use_async_processing': True  # Async document processing
}
```

DocsGPT provides a comprehensive solution for creating AI-powered documentation assistants that can understand and answer questions about your specific documentation content.