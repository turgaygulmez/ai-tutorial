# Onyx - Enterprise AI Assistant Platform

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [Document Management](#document-management)
- [Search and Retrieval](#search-and-retrieval)
- [User Management](#user-management)
- [API Reference](#api-reference)
- [Deployment](#deployment)

## Overview

Onyx is an open-source enterprise AI assistant platform that combines document management, intelligent search, and conversational AI. It's designed to help organizations leverage their internal knowledge base through AI-powered question answering and document retrieval.

**Key Characteristics:**
- **Enterprise-Ready**: Built for organizational use with proper security and access controls
- **Document-Centric**: Focuses on making organizational documents searchable and queryable
- **Privacy-First**: Self-hosted solution with data privacy controls
- **Scalable**: Designed to handle large document collections and multiple users

## Key Features

### 1. Document Intelligence
- Multi-format document processing (PDF, Word, PowerPoint, etc.)
- Automatic text extraction and indexing
- Document chunking and embedding generation
- OCR support for scanned documents

### 2. Intelligent Search
- Semantic search capabilities
- Hybrid search combining keyword and vector search
- Context-aware document retrieval
- Advanced filtering and faceting

### 3. Conversational AI
- Natural language question answering
- Context-aware conversations
- Source attribution and citations
- Multi-turn dialogue support

### 4. Enterprise Features
- User authentication and authorization
- Role-based access control
- Document permissions and security
- Audit logging and compliance

### 5. Integration Capabilities
- REST API for system integration
- Webhook support for real-time updates
- Single Sign-On (SSO) integration
- Third-party connector ecosystem

## Architecture

### System Components

#### Frontend (Web UI)
- React-based user interface
- Document upload and management
- Search interface and chat functionality
- Administrative dashboard

#### Backend Services
- **API Server**: FastAPI-based REST API
- **Document Processor**: Handles file parsing and indexing
- **Search Engine**: Vector and keyword search capabilities
- **Authentication Service**: User management and security

#### Data Layer
- **Vector Database**: Stores document embeddings
- **Relational Database**: User data and metadata
- **File Storage**: Document and asset storage
- **Cache Layer**: Redis for session and query caching

## Installation

### Prerequisites
```bash
# Docker and Docker Compose
docker --version
docker-compose --version

# Python 3.9+ (for development)
python --version

# Git
git --version
```

### Quick Start with Docker
```bash
# Clone the repository
git clone https://github.com/onyx-dot-app/onyx.git
cd onyx

# Start with Docker Compose
docker-compose up -d

# Access the application
# Web UI: http://localhost:3000
# API: http://localhost:8080
```

### Development Setup
```bash
# Clone repository
git clone https://github.com/onyx-dot-app/onyx.git
cd onyx

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup frontend
cd ../web
npm install

# Setup environment variables
cp .env.template .env
# Edit .env with your configurations

# Run development servers
# Backend (in backend directory)
uvicorn main:app --reload

# Frontend (in web directory)
npm run dev
```

### Production Installation
```bash
# Production Docker setup
docker-compose -f docker-compose.prod.yml up -d

# Or use Kubernetes
kubectl apply -f k8s/
```

## Configuration

### Environment Variables
```bash
# .env file configuration

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=onyx
POSTGRES_USER=onyx_user
POSTGRES_PASSWORD=your_password

# Vector Database (Qdrant)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_api_key

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Authentication
SECRET_KEY=your_secret_key
OAUTH_CLIENT_ID=your_oauth_client_id
OAUTH_CLIENT_SECRET=your_oauth_client_secret

# LLM Configuration
OPENAI_API_KEY=your_openai_key
# Or use other providers
ANTHROPIC_API_KEY=your_anthropic_key

# File Storage
S3_BUCKET_NAME=your_bucket
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Email Configuration (for notifications)
SMTP_SERVER=your_smtp_server
SMTP_PORT=587
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_email_password
```

### Application Configuration
```yaml
# config.yaml
app:
  name: "Onyx Enterprise AI"
  version: "1.0.0"
  debug: false

search:
  default_results: 10
  max_results: 100
  similarity_threshold: 0.7

document_processing:
  max_file_size: 100MB
  supported_formats: 
    - pdf
    - docx
    - pptx
    - txt
    - md
  chunk_size: 1000
  chunk_overlap: 200

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.1
  max_tokens: 2000

security:
  session_timeout: 24  # hours
  password_min_length: 8
  require_2fa: false
  allowed_domains: []
```

## Getting Started

### Initial Setup
```python
# Initialize the system (run once)
from onyx.setup import initialize_system

# Create admin user
initialize_system(
    admin_email="admin@yourcompany.com",
    admin_password="secure_password",
    organization_name="Your Company"
)
```

### Basic Usage

#### Document Upload
```python
import requests

# Upload document via API
files = {'file': open('document.pdf', 'rb')}
data = {
    'title': 'Company Policy Document',
    'description': 'HR policies and procedures',
    'tags': 'hr,policy,procedures'
}

response = requests.post(
    'http://localhost:8080/api/documents/upload',
    files=files,
    data=data,
    headers={'Authorization': 'Bearer your_token'}
)

document_id = response.json()['document_id']
```

#### Search Documents
```python
# Search for documents
search_params = {
    'query': 'vacation policy',
    'limit': 10,
    'filters': {'tags': 'hr'}
}

response = requests.post(
    'http://localhost:8080/api/search',
    json=search_params,
    headers={'Authorization': 'Bearer your_token'}
)

results = response.json()['results']
for result in results:
    print(f"Document: {result['title']}")
    print(f"Score: {result['score']}")
    print(f"Snippet: {result['snippet']}")
```

#### Ask Questions
```python
# Ask questions about documents
question_data = {
    'question': 'How many vacation days do employees get?',
    'context_documents': ['doc_id_1', 'doc_id_2'],  # Optional
    'conversation_id': 'conv_123'  # For multi-turn conversations
}

response = requests.post(
    'http://localhost:8080/api/chat',
    json=question_data,
    headers={'Authorization': 'Bearer your_token'}
)

answer = response.json()
print(f"Answer: {answer['response']}")
print(f"Sources: {answer['sources']}")
```

## Document Management

### Document Processing Pipeline
```python
# Custom document processor
from onyx.document_processing import DocumentProcessor

class CustomProcessor(DocumentProcessor):
    def __init__(self):
        super().__init__()
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from document"""
        if file_path.endswith('.pdf'):
            return self.extract_pdf_text(file_path)
        elif file_path.endswith('.docx'):
            return self.extract_docx_text(file_path)
        else:
            return self.extract_plain_text(file_path)
    
    def chunk_document(self, text: str, chunk_size: int = 1000) -> list:
        """Split document into chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
```

### Metadata Management
```python
# Document metadata schema
document_metadata = {
    'id': 'doc_12345',
    'title': 'Q4 Financial Report',
    'description': 'Financial results for Q4 2023',
    'author': 'Finance Team',
    'created_at': '2024-01-15T10:00:00Z',
    'updated_at': '2024-01-15T10:00:00Z',
    'tags': ['finance', 'quarterly', 'report'],
    'department': 'Finance',
    'access_level': 'internal',
    'file_type': 'pdf',
    'file_size': 1024000,
    'page_count': 25,
    'language': 'en',
    'status': 'active'
}

# Update document metadata
requests.put(
    f'http://localhost:8080/api/documents/{document_id}/metadata',
    json=document_metadata,
    headers={'Authorization': 'Bearer your_token'}
)
```

### Batch Operations
```python
# Batch upload documents
def batch_upload_documents(document_paths, metadata_list):
    """Upload multiple documents in batch"""
    results = []
    
    for doc_path, metadata in zip(document_paths, metadata_list):
        try:
            with open(doc_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    'http://localhost:8080/api/documents/upload',
                    files=files,
                    data=metadata,
                    headers={'Authorization': 'Bearer your_token'}
                )
                results.append({
                    'path': doc_path,
                    'status': 'success',
                    'document_id': response.json()['document_id']
                })
        except Exception as e:
            results.append({
                'path': doc_path,
                'status': 'error',
                'error': str(e)
            })
    
    return results
```

## Search and Retrieval

### Advanced Search
```python
# Advanced search with filters and facets
search_request = {
    'query': 'machine learning algorithms',
    'filters': {
        'department': ['Engineering', 'Research'],
        'created_after': '2023-01-01',
        'file_type': ['pdf', 'docx'],
        'tags': ['ai', 'ml']
    },
    'facets': ['department', 'author', 'tags'],
    'sort': {
        'field': 'relevance',
        'order': 'desc'
    },
    'highlight': True,
    'limit': 20,
    'offset': 0
}

response = requests.post(
    'http://localhost:8080/api/search/advanced',
    json=search_request,
    headers={'Authorization': 'Bearer your_token'}
)

results = response.json()
print(f"Total results: {results['total']}")
print(f"Facets: {results['facets']}")
```

### Semantic Search
```python
# Semantic similarity search
semantic_search = {
    'query': 'How to improve employee satisfaction?',
    'search_type': 'semantic',
    'similarity_threshold': 0.7,
    'rerank': True,
    'include_snippets': True
}

response = requests.post(
    'http://localhost:8080/api/search/semantic',
    json=semantic_search,
    headers={'Authorization': 'Bearer your_token'}
)
```

### Custom Retrievers
```python
# Implement custom retrieval logic
from onyx.retrieval import BaseRetriever

class CustomRetriever(BaseRetriever):
    def __init__(self, vector_store, keyword_index):
        self.vector_store = vector_store
        self.keyword_index = keyword_index
    
    def retrieve(self, query: str, filters: dict = None, k: int = 10):
        """Custom retrieval combining multiple strategies"""
        
        # Get semantic results
        semantic_results = self.vector_store.similarity_search(query, k=k*2)
        
        # Get keyword results
        keyword_results = self.keyword_index.search(query, k=k*2)
        
        # Combine and rerank results
        combined_results = self.hybrid_ranking(
            semantic_results, keyword_results, query
        )
        
        # Apply filters
        if filters:
            combined_results = self.apply_filters(combined_results, filters)
        
        return combined_results[:k]
```

## User Management

### Authentication
```python
# User authentication setup
from onyx.auth import AuthManager

auth_manager = AuthManager()

# Register new user
user_data = {
    'email': 'user@company.com',
    'password': 'secure_password',
    'first_name': 'John',
    'last_name': 'Doe',
    'department': 'Engineering',
    'role': 'user'
}

user = auth_manager.register_user(user_data)

# Login user
login_response = auth_manager.login(
    email='user@company.com',
    password='secure_password'
)

access_token = login_response['access_token']
```

### Role-Based Access Control
```python
# Define roles and permissions
roles_config = {
    'admin': {
        'permissions': [
            'document.create',
            'document.read',
            'document.update', 
            'document.delete',
            'user.manage',
            'system.config'
        ]
    },
    'editor': {
        'permissions': [
            'document.create',
            'document.read',
            'document.update'
        ]
    },
    'viewer': {
        'permissions': [
            'document.read'
        ]
    }
}

# Check permissions
from onyx.auth import check_permission

@check_permission('document.create')
def upload_document(user, document_data):
    # User has permission to create documents
    pass
```

### Single Sign-On Integration
```python
# SSO configuration
sso_config = {
    'provider': 'microsoft',  # or 'google', 'okta', 'auth0'
    'client_id': 'your_client_id',
    'client_secret': 'your_client_secret',
    'redirect_uri': 'http://localhost:3000/auth/callback',
    'scopes': ['openid', 'profile', 'email']
}

# SSO login endpoint
@app.post('/auth/sso/login')
async def sso_login(provider: str):
    """Initiate SSO login"""
    auth_url = auth_manager.get_sso_login_url(provider)
    return {'auth_url': auth_url}
```

## API Reference

### Document API
```bash
# Upload document
POST /api/documents/upload
Content-Type: multipart/form-data

# Get document
GET /api/documents/{document_id}

# Update document metadata
PUT /api/documents/{document_id}/metadata

# Delete document
DELETE /api/documents/{document_id}

# List documents
GET /api/documents?limit=10&offset=0&filters={}
```

### Search API
```bash
# Basic search
POST /api/search
{
    "query": "search terms",
    "limit": 10,
    "filters": {}
}

# Advanced search
POST /api/search/advanced
{
    "query": "search terms",
    "search_type": "hybrid",
    "filters": {},
    "facets": [],
    "sort": {}
}
```

### Chat API
```bash
# Ask question
POST /api/chat
{
    "question": "What is the vacation policy?",
    "conversation_id": "optional_conv_id",
    "context_documents": ["doc1", "doc2"]
}

# Get conversation history
GET /api/conversations/{conversation_id}

# Delete conversation
DELETE /api/conversations/{conversation_id}
```

### Admin API
```bash
# System health
GET /api/admin/health

# System metrics
GET /api/admin/metrics

# User management
GET /api/admin/users
POST /api/admin/users
PUT /api/admin/users/{user_id}
DELETE /api/admin/users/{user_id}
```

## Deployment

### Docker Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  onyx-web:
    image: onyx/web:latest
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://onyx-api:8080
    depends_on:
      - onyx-api

  onyx-api:
    image: onyx/api:latest
    ports:
      - "8080:8080"
    environment:
      - POSTGRES_HOST=postgres
      - QDRANT_HOST=qdrant
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - qdrant
      - redis

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=onyx
      - POSTGRES_USER=onyx
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  qdrant_data:
  redis_data:
```

### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: onyx-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: onyx-api
  template:
    metadata:
      labels:
        app: onyx-api
    spec:
      containers:
      - name: onyx-api
        image: onyx/api:latest
        ports:
        - containerPort: 8080
        env:
        - name: POSTGRES_HOST
          value: "postgres-service"
        - name: QDRANT_HOST
          value: "qdrant-service"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Production Configuration
```bash
# Production environment setup
export NODE_ENV=production
export POSTGRES_HOST=prod-postgres.example.com
export QDRANT_HOST=prod-qdrant.example.com
export REDIS_HOST=prod-redis.example.com

# SSL Configuration
export SSL_CERT_PATH=/etc/ssl/certs/onyx.crt
export SSL_KEY_PATH=/etc/ssl/private/onyx.key

# Monitoring
export PROMETHEUS_ENDPOINT=http://prometheus:9090
export GRAFANA_ENDPOINT=http://grafana:3000

# Backup configuration
export BACKUP_S3_BUCKET=onyx-backups
export BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
```

## Integration Examples

### Slack Bot Integration
```python
# Slack bot for Onyx
from slack_sdk import WebClient
from slack_sdk.adapter.flask import SlackRequestHandler

@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle Slack events"""
    if event.get("type") == "app_mention":
        question = event["text"].replace(f"<@{bot_user_id}>", "").strip()
        
        # Query Onyx
        response = requests.post(
            'http://onyx-api:8080/api/chat',
            json={'question': question},
            headers={'Authorization': f'Bearer {onyx_token}'}
        )
        
        answer = response.json()['response']
        
        # Reply in Slack
        client.chat_postMessage(
            channel=event["channel"],
            text=f"🤖 {answer}",
            thread_ts=event.get("ts")
        )
```

### Microsoft Teams Integration
```python
# Teams bot integration
from botbuilder.core import TurnContext, MessageFactory
from botbuilder.schema import Activity

class OnyxTeamsBot:
    async def on_message_activity(self, turn_context: TurnContext):
        question = turn_context.activity.text
        
        # Query Onyx
        response = await self.query_onyx(question)
        
        reply = MessageFactory.text(f"📚 {response['answer']}")
        if response.get('sources'):
            sources_text = '\n'.join([f"- {s['title']}" for s in response['sources']])
            reply.text += f"\n\n**Sources:**\n{sources_text}"
        
        await turn_context.send_activity(reply)
```

### API Gateway Integration
```python
# API Gateway for external access
from fastapi import FastAPI, Depends
from fastapi.security import HTTPBearer

app = FastAPI()
security = HTTPBearer()

@app.post("/external/search")
async def external_search(
    request: SearchRequest,
    token: str = Depends(security)
):
    """External API for partner integrations"""
    
    # Validate external token
    if not validate_external_token(token.credentials):
        raise HTTPException(401, "Invalid token")
    
    # Rate limiting
    if not check_rate_limit(token.credentials):
        raise HTTPException(429, "Rate limit exceeded")
    
    # Forward to Onyx
    response = requests.post(
        'http://onyx-api:8080/api/search',
        json=request.dict(),
        headers={'Authorization': f'Bearer {internal_token}'}
    )
    
    return response.json()
```

## Best Practices

### Document Organization
- Use consistent naming conventions
- Implement proper tagging strategies
- Regular content audits and cleanup
- Version control for important documents

### Search Optimization
- Monitor search analytics
- Optimize chunk sizes for your content
- Regular embedding model updates
- A/B testing for search relevance

### Security
- Regular security audits
- Implement proper access controls
- Monitor user activities
- Regular backup procedures

### Performance
- Monitor system metrics
- Optimize database queries
- Implement proper caching strategies
- Scale components based on usage

## Troubleshooting

### Common Issues
1. **Document processing failures**: Check file formats and sizes
2. **Search performance**: Optimize vector database configuration
3. **Authentication issues**: Verify SSO configurations
4. **Memory issues**: Monitor and optimize resource usage

### Monitoring and Logging
```python
# Monitoring setup
import logging
from prometheus_client import Counter, Histogram

# Metrics
search_requests = Counter('onyx_search_requests_total')
response_time = Histogram('onyx_response_time_seconds')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Community and Resources

- **GitHub Repository**: [onyx-dot-app/onyx](https://github.com/onyx-dot-app/onyx)
- **Documentation**: Official Onyx documentation
- **Community Forum**: GitHub Discussions
- **Docker Hub**: Pre-built container images

## Limitations

- Requires significant computational resources for large document collections
- Vector database storage grows with document count
- Processing time scales with document complexity
- May require custom configuration for specialized document types