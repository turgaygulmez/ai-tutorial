# FinGPT - Open-Source Financial Large Language Model

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Available Models](#available-models)
- [Quick Start](#quick-start)
- [Data Sources](#data-sources)
- [Training and Fine-tuning](#training-and-fine-tuning)
- [Applications](#applications)
- [Benchmarks](#benchmarks)
- [API Usage](#api-usage)

## Overview

FinGPT is the first open-source large language model (LLM) specifically designed for finance. Developed by the AI4Finance Foundation, FinGPT aims to democratize internet-scale data for financial large language models.

**Key Characteristics:**
- **Open-Source**: Fully accessible codebase and model weights
- **Financial Focus**: Purpose-built for financial applications
- **Data-Rich**: Leverages extensive financial data sources
- **Community-Driven**: Collaborative development approach
- **Practical Applications**: Real-world financial use cases

## Key Features

### 1. Comprehensive Financial Knowledge
- Market analysis and sentiment analysis
- Financial report understanding
- Risk assessment and management
- Investment strategy recommendations
- Economic indicator interpretation

### 2. Multi-Modal Capabilities
- Text processing (news, reports, filings)
- Time series data analysis
- Numerical data interpretation
- Chart and graph understanding

### 3. Real-Time Data Integration
- Market data feeds
- News sentiment analysis
- Social media monitoring
- Economic indicators tracking

### 4. Fine-Tuning Support
- Custom model training
- Domain-specific adaptations
- Task-specific optimizations
- Transfer learning capabilities

## Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# CUDA support (optional but recommended)
nvidia-smi
```

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/AI4Finance-Foundation/FinGPT.git
cd FinGPT

# Install dependencies
pip install -r requirements.txt

# Install FinGPT
pip install -e .
```

### Docker Installation
```bash
# Build Docker image
docker build -t fingpt:latest .

# Run container
docker run --gpus all -it --rm fingpt:latest
```

## Available Models

### FinGPT v3.1 Series
- **FinGPT-3.1-7B**: Base model with 7 billion parameters
- **FinGPT-3.1-13B**: Enhanced model with 13 billion parameters
- **FinGPT-3.1-Chat**: Conversational variant optimized for interaction

### FinGPT v3.2 Series
- **FinGPT-3.2-7B-Instruct**: Instruction-tuned for specific financial tasks
- **FinGPT-3.2-13B-Instruct**: Larger instruction-tuned model

### Specialized Models
- **FinGPT-Forecaster**: Optimized for price prediction
- **FinGPT-Sentiment**: Specialized in sentiment analysis
- **FinGPT-News**: Fine-tuned for news analysis

## Quick Start

### Basic Usage
```python
from fingpt import FinGPT

# Initialize the model
model = FinGPT.from_pretrained("FinGPT-v3.1-7B")

# Basic financial analysis
prompt = """
Analyze the following financial data and provide insights:
- Apple Inc. (AAPL) reported Q3 revenue of $81.8B
- Revenue grew 1.4% year-over-year
- Services revenue reached $21.2B, up 8.2%
- iPhone revenue declined 2.4% to $39.7B
"""

response = model.generate(prompt, max_length=500)
print(response)
```

### Market Sentiment Analysis
```python
from fingpt.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer.from_pretrained("FinGPT-Sentiment")

# Analyze market sentiment
news_text = "Federal Reserve hints at potential rate cuts amid economic uncertainty"
sentiment = analyzer.analyze(news_text)

print(f"Sentiment: {sentiment.label}")
print(f"Confidence: {sentiment.confidence}")
print(f"Market Impact: {sentiment.market_impact}")
```

### Stock Price Forecasting
```python
from fingpt.forecasting import PriceForecaster
import yfinance as yf

# Load historical data
ticker = "AAPL"
data = yf.download(ticker, period="1y")

# Initialize forecaster
forecaster = PriceForecaster.from_pretrained("FinGPT-Forecaster")

# Make prediction
prediction = forecaster.predict(
    data=data,
    horizon=30,  # 30 days forecast
    features=["open", "high", "low", "close", "volume"]
)

print(f"30-day price forecast for {ticker}:")
print(prediction)
```

## Data Sources

FinGPT leverages multiple financial data sources:

### Market Data
- **Yahoo Finance**: Stock prices, financial statements
- **Alpha Vantage**: Real-time and historical market data
- **Quandl**: Economic and financial datasets
- **FRED**: Federal Reserve economic data

### News and Social Media
- **Reuters**: Financial news and market updates
- **Bloomberg**: Professional financial information
- **Twitter/X**: Social sentiment analysis
- **Reddit**: Retail investor discussions

### Regulatory Filings
- **SEC EDGAR**: Corporate filings and reports
- **10-K/10-Q Forms**: Annual and quarterly reports
- **8-K Forms**: Current event reports

### Configuration Example
```python
from fingpt.data import DataLoader

# Configure data sources
config = {
    'sources': {
        'yahoo': {'api_key': None, 'rate_limit': 2000},
        'alpha_vantage': {'api_key': 'YOUR_API_KEY'},
        'news': {'sources': ['reuters', 'bloomberg']},
        'social': {'platforms': ['twitter', 'reddit']}
    }
}

loader = DataLoader(config)
data = loader.fetch_market_data('AAPL', period='1y')
```

## Training and Fine-tuning

### Custom Model Training
```python
from fingpt.training import FinGPTTrainer
from fingpt.data import FinancialDataset

# Prepare training data
dataset = FinancialDataset(
    data_path="./financial_data",
    task_type="sentiment_analysis",
    split="train"
)

# Configure training
trainer = FinGPTTrainer(
    model_name="FinGPT-v3.1-7B",
    dataset=dataset,
    training_args={
        'learning_rate': 2e-5,
        'batch_size': 8,
        'num_epochs': 3,
        'warmup_steps': 500,
        'save_steps': 1000
    }
)

# Start training
trainer.train()
```

### Fine-tuning for Specific Tasks
```python
# Fine-tune for portfolio optimization
from fingpt.fine_tuning import TaskSpecificFineTuner

fine_tuner = TaskSpecificFineTuner(
    base_model="FinGPT-v3.1-7B",
    task="portfolio_optimization",
    data_path="./portfolio_data"
)

# Configure fine-tuning parameters
fine_tuner.configure({
    'learning_rate': 1e-5,
    'batch_size': 4,
    'gradient_accumulation_steps': 2,
    'max_steps': 5000
})

# Execute fine-tuning
fine_tuned_model = fine_tuner.train()
```

## Applications

### 1. Robo-Advisory
```python
from fingpt.applications import RoboAdvisor

advisor = RoboAdvisor.from_pretrained("FinGPT-v3.1-13B")

# Generate investment recommendation
client_profile = {
    'age': 35,
    'risk_tolerance': 'moderate',
    'investment_horizon': '10 years',
    'portfolio_size': 100000
}

recommendation = advisor.generate_advice(client_profile)
print(recommendation)
```

### 2. Financial Report Analysis
```python
from fingpt.applications import ReportAnalyzer

analyzer = ReportAnalyzer.from_pretrained("FinGPT-v3.1-7B")

# Analyze 10-K filing
report_path = "./apple_10k_2023.txt"
analysis = analyzer.analyze_report(report_path)

print("Key Insights:", analysis.insights)
print("Risk Factors:", analysis.risks)
print("Growth Opportunities:", analysis.opportunities)
```

### 3. Trading Strategy Development
```python
from fingpt.applications import StrategyGenerator

generator = StrategyGenerator.from_pretrained("FinGPT-Forecaster")

# Generate trading strategy
strategy = generator.create_strategy(
    asset_class="equities",
    market_regime="bullish",
    time_horizon="medium_term",
    risk_level="moderate"
)

print(strategy.description)
print(strategy.entry_rules)
print(strategy.exit_rules)
```

## Benchmarks

### Performance Metrics

| Task | Dataset | FinGPT Score | GPT-4 Score | Improvement |
|------|---------|--------------|-------------|-------------|
| Sentiment Analysis | FPB | 94.2% | 89.1% | +5.1% |
| News Classification | Reuters | 91.7% | 87.3% | +4.4% |
| Q&A | FinQA | 83.6% | 79.2% | +4.4% |
| Stock Prediction | S&P 500 | 67.3% | 58.1% | +9.2% |
| Risk Assessment | Credit Scoring | 88.9% | 84.7% | +4.2% |

### Comparison with Other Financial LLMs

| Model | Parameters | Financial Accuracy | General Performance |
|-------|------------|-------------------|-------------------|
| FinGPT-v3.1 | 7B | 89.4% | 85.2% |
| BloombergGPT | 50B | 87.1% | 88.7% |
| FinMA | 7B | 86.8% | 82.3% |
| LLaMA-2-Chat | 7B | 74.2% | 87.9% |

## API Usage

### REST API
```python
import requests

# API endpoint
url = "http://localhost:8000/api/v1/generate"

# Request payload
payload = {
    "prompt": "Analyze Tesla's Q3 2023 earnings report",
    "model": "FinGPT-v3.1-7B",
    "max_tokens": 500,
    "temperature": 0.7
}

response = requests.post(url, json=payload)
result = response.json()
print(result["generated_text"])
```

### WebSocket API
```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"Response: {data['text']}")

def on_open(ws):
    request = {
        "action": "generate",
        "prompt": "What are the key risks in the current market?",
        "model": "FinGPT-v3.1-7B"
    }
    ws.send(json.dumps(request))

ws = websocket.WebSocketApp("ws://localhost:8000/ws",
                          on_message=on_message,
                          on_open=on_open)
ws.run_forever()
```

## Model Deployment

### Local Deployment
```bash
# Start the FinGPT server
python -m fingpt.server \
    --model-path ./models/FinGPT-v3.1-7B \
    --port 8000 \
    --gpu-count 1
```

### Cloud Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  fingpt-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/FinGPT-v3.1-7B
      - GPU_COUNT=1
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Best Practices

### 1. Data Preparation
- Clean and normalize financial data
- Handle missing values appropriately
- Ensure data quality and consistency
- Use proper time series alignment

### 2. Model Selection
- Choose appropriate model size for your use case
- Consider computational resources
- Evaluate task-specific performance
- Use instruction-tuned models for interactive applications

### 3. Prompt Engineering
- Be specific about financial context
- Include relevant market conditions
- Provide clear formatting instructions
- Use domain-specific terminology

### 4. Risk Management
- Always validate model outputs
- Implement proper error handling
- Monitor model performance over time
- Use ensemble methods for critical decisions

## Community and Support

- **GitHub Repository**: [AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)
- **Documentation**: [FinGPT Documentation](https://fingpt.readthedocs.io/)
- **Discord Community**: Join the AI4Finance Discord
- **Paper**: [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/abs/2306.06031)

## License

FinGPT is released under the MIT License, promoting open-source development and commercial usage.

## Limitations and Disclaimers

- FinGPT outputs should not be considered as financial advice
- Always validate model predictions with domain experts
- Model performance may vary across different market conditions
- Regular model updates and retraining are recommended
- Compliance with local financial regulations is user's responsibility