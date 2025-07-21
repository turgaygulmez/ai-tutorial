# MetaGPT - Multi-Agent Framework

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Quick Start](#quick-start)
- [Agent Roles](#agent-roles)
- [Workflow Examples](#workflow-examples)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Integration Examples](#integration-examples)

## Overview

MetaGPT is a multi-agent framework that assigns different roles to GPT agents, enabling them to collaborate and form a software company capable of handling complex tasks. The framework simulates a software development team where each agent has specialized roles like Product Manager, Architect, Engineer, and QA Tester.

**Key Philosophy:**
- **Role-Based Collaboration**: Each agent has specific responsibilities and expertise
- **Structured Communication**: Agents communicate through standardized interfaces
- **Document-Driven Development**: All interactions are documented and traceable
- **Iterative Improvement**: Continuous refinement through feedback loops

## Key Features

### 1. Multi-Agent Collaboration
- Specialized agent roles (PM, Architect, Engineer, QA)
- Structured communication protocols
- Collaborative problem-solving approach
- Role-based task distribution

### 2. Software Development Lifecycle
- Requirements analysis and PRD creation
- System design and architecture
- Code implementation and testing
- Documentation and deployment

### 3. Document Management
- Automatic document generation
- Version control and tracking
- Standardized templates and formats
- Cross-reference capabilities

### 4. Workflow Orchestration
- Predefined development workflows
- Custom workflow creation
- Task scheduling and dependencies
- Progress tracking and monitoring

## Installation

### Prerequisites
```bash
# Python 3.9 or higher
python --version

# Git for repository cloning
git --version
```

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/geekan/MetaGPT.git
cd MetaGPT

# Install dependencies
pip install -e .

# Or install from PyPI
pip install metagpt
```

### Development Installation
```bash
# Clone with development dependencies
git clone https://github.com/geekan/MetaGPT.git
cd MetaGPT

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Docker Installation
```bash
# Pull and run Docker image
docker pull metagpt/metagpt:latest

# Run with environment variables
docker run --rm \
  -e OPENAI_API_KEY=your_api_key \
  -v $(pwd):/app/workspace \
  metagpt/metagpt:latest \
  "Create a snake game"
```

## Core Concepts

### 1. Agents and Roles
Each agent in MetaGPT has a specific role with defined responsibilities:

- **Product Manager**: Writes PRDs and manages requirements
- **Architect**: Designs system architecture and technical specifications
- **Engineer**: Implements code based on specifications
- **QA Tester**: Creates and executes test plans

### 2. Actions and Skills
Agents perform actions using their skills:
- **WritePRD**: Product requirement document creation
- **WriteDesign**: System design documentation
- **WriteCode**: Code implementation
- **WriteTest**: Test case creation

### 3. Environment and Communication
- **Shared Environment**: Common workspace for all agents
- **Message Passing**: Structured communication between agents
- **Document Store**: Centralized document management

## Quick Start

### Basic Setup
```python
import asyncio
from metagpt.software_company import SoftwareCompany

async def main():
    # Initialize the software company
    company = SoftwareCompany()
    
    # Assign a project
    project_idea = "Create a web-based todo list application"
    
    # Run the development process
    await company.run_project(project_idea)

if __name__ == "__main__":
    asyncio.run(main())
```

### Environment Configuration
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Set other model providers
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# Set workspace directory
export WORKSPACE_PATH="./workspace"
```

### Simple Example
```python
from metagpt.team import Team

# Create a development team
team = Team()

# Add team members
team.hire([
    "Product Manager",
    "Architect", 
    "Engineer",
    "QA Engineer"
])

# Start project
investment = 3.0  # Investment in USD
project = "Build a REST API for user management"

team.run_project(project, investment=investment)
```

## Agent Roles

### Product Manager
**Responsibilities:**
- Analyze user requirements
- Write Product Requirement Documents (PRDs)
- Define project scope and objectives
- Communicate with stakeholders

**Example Usage:**
```python
from metagpt.roles import ProductManager

pm = ProductManager(
    name="Alice",
    profile="Experienced PM with 5+ years in SaaS",
    goal="Create comprehensive PRDs"
)

# Generate PRD
prd = await pm.write_prd(
    "Create a mobile app for expense tracking"
)
```

### Architect
**Responsibilities:**
- Design system architecture
- Create technical specifications
- Define data models and APIs
- Review technical decisions

**Example Usage:**
```python
from metagpt.roles import Architect

architect = Architect(
    name="Bob",
    profile="Senior architect with microservices expertise",
    goal="Design scalable and maintainable systems"
)

# Create system design
design = await architect.write_design(prd_document)
```

### Engineer
**Responsibilities:**
- Implement code based on specifications
- Follow coding standards and best practices
- Create modular and maintainable code
- Collaborate with other engineers

**Example Usage:**
```python
from metagpt.roles import Engineer

engineer = Engineer(
    name="Carol",
    profile="Full-stack developer with Python/React expertise",
    goal="Write clean and efficient code"
)

# Implement features
code = await engineer.write_code(design_document)
```

### QA Engineer
**Responsibilities:**
- Create test plans and test cases
- Execute testing procedures
- Report bugs and issues
- Ensure quality standards

**Example Usage:**
```python
from metagpt.roles import QaEngineer

qa = QaEngineer(
    name="David",
    profile="QA engineer with automation expertise",
    goal="Ensure software quality and reliability"
)

# Create test cases
tests = await qa.write_test(code_implementation)
```

## Workflow Examples

### 1. Complete Software Development
```python
from metagpt.company import Company
from metagpt.team import Team

async def develop_software():
    # Initialize company
    company = Company()
    
    # Create development team
    team = Team()
    team.hire([
        "ProductManager",
        "Architect", 
        "Engineer",
        "QaEngineer"
    ])
    
    # Define project
    idea = """
    Create a task management system with the following features:
    - User authentication and profiles
    - Create, edit, and delete tasks
    - Task categories and priorities
    - Due date reminders
    - Progress tracking dashboard
    """
    
    # Set investment (affects scope and quality)
    investment = 5.0  # $5 investment
    
    # Run development process
    await company.start_project(
        idea=idea,
        investment=investment,
        team=team
    )

# Run the development process
asyncio.run(develop_software())
```

### 2. Custom Workflow
```python
from metagpt.actions import WritePRD, WriteDesign, WriteCode, WriteTest
from metagpt.schema import Message

async def custom_workflow(idea):
    # Step 1: Product Manager creates PRD
    pm_action = WritePRD()
    prd_msg = await pm_action.run(Message(content=idea))
    
    # Step 2: Architect creates design
    arch_action = WriteDesign()
    design_msg = await arch_action.run(prd_msg)
    
    # Step 3: Engineer writes code
    eng_action = WriteCode()
    code_msg = await eng_action.run(design_msg)
    
    # Step 4: QA creates tests
    qa_action = WriteTest()
    test_msg = await qa_action.run(code_msg)
    
    return {
        'prd': prd_msg,
        'design': design_msg,
        'code': code_msg,
        'tests': test_msg
    }
```

## Configuration

### Model Configuration
```python
from metagpt.config import Config

# Configure different LLM providers
config = Config()

# OpenAI Configuration
config.openai_api_key = "your-openai-key"
config.openai_api_model = "gpt-4"

# Alternative: Anthropic Claude
config.anthropic_api_key = "your-anthropic-key" 
config.llm_model = "claude-3-sonnet"

# Alternative: Local model
config.llm_model = "local"
config.local_model_path = "./models/llama-7b"
```

### Workspace Configuration
```python
# Set workspace directory
config.workspace_path = "./workspace"

# Document templates
config.template_path = "./templates"

# Output formats
config.output_format = "markdown"  # or "pdf", "html"

# Version control
config.git_enabled = True
config.git_auto_commit = True
```

### Team Configuration
```python
team_config = {
    "max_team_size": 6,
    "roles": {
        "ProductManager": {
            "count": 1,
            "experience_level": "senior"
        },
        "Architect": {
            "count": 1,
            "experience_level": "senior"
        },
        "Engineer": {
            "count": 2,
            "experience_level": "mid"
        },
        "QaEngineer": {
            "count": 1,
            "experience_level": "senior"
        }
    }
}
```

## Advanced Usage

### Custom Agent Creation
```python
from metagpt.roles.role import Role
from metagpt.actions import Action
from metagpt.schema import Message

class CustomAction(Action):
    async def run(self, msg: Message) -> Message:
        # Custom action implementation
        result = await self.llm.aask(
            f"Process this request: {msg.content}"
        )
        return Message(content=result)

class CustomRole(Role):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([CustomAction])
        
    async def _act(self) -> Message:
        # Custom role behavior
        return await super()._act()
```

### Multi-Project Management
```python
from metagpt.company import Company

async def manage_multiple_projects():
    company = Company()
    
    projects = [
        {"idea": "E-commerce platform", "investment": 10.0},
        {"idea": "Social media app", "investment": 8.0},
        {"idea": "Analytics dashboard", "investment": 6.0}
    ]
    
    results = []
    for project in projects:
        result = await company.start_project(**project)
        results.append(result)
    
    return results
```

### Integration with External APIs
```python
from metagpt.tools import Tool

class APIIntegrationTool(Tool):
    async def call_external_api(self, endpoint, data):
        # Integration with external services
        response = await self.http_client.post(endpoint, json=data)
        return response.json()

# Use in custom action
class IntegrateAPIAction(Action):
    def __init__(self):
        super().__init__()
        self.tool = APIIntegrationTool()
    
    async def run(self, msg: Message) -> Message:
        # Use external API in action
        api_result = await self.tool.call_external_api(
            "https://api.example.com/process",
            {"input": msg.content}
        )
        return Message(content=api_result)
```

## Integration Examples

### 1. CI/CD Integration
```yaml
# .github/workflows/metagpt.yml
name: MetaGPT Development

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  develop:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install MetaGPT
      run: pip install metagpt
    
    - name: Run Development Process
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        python -c "
        import asyncio
        from metagpt.company import Company
        
        async def main():
            company = Company()
            await company.start_project(
                idea='${{ github.event.head_commit.message }}',
                investment=3.0
            )
        
        asyncio.run(main())
        "
```

### 2. Web Interface Integration
```python
from fastapi import FastAPI
from metagpt.company import Company

app = FastAPI()
company = Company()

@app.post("/develop")
async def develop_project(request: dict):
    idea = request.get("idea")
    investment = request.get("investment", 3.0)
    
    result = await company.start_project(
        idea=idea,
        investment=investment
    )
    
    return {"status": "success", "result": result}

@app.get("/projects")
async def list_projects():
    return company.get_project_history()
```

### 3. Slack Bot Integration
```python
from slack_bolt.async_app import AsyncApp
from metagpt.company import Company

app = AsyncApp(token="your-slack-token")
company = Company()

@app.message("develop")
async def handle_develop_command(message, say):
    idea = message["text"].replace("develop ", "")
    
    await say(f"Starting development for: {idea}")
    
    result = await company.start_project(
        idea=idea,
        investment=2.0
    )
    
    await say(f"Development completed! Check the results in workspace.")
```

## Best Practices

### 1. Project Planning
- Provide clear and detailed project requirements
- Set appropriate investment levels based on project complexity
- Define success criteria and deliverables upfront

### 2. Team Composition
- Balance team roles based on project needs
- Adjust experience levels according to project complexity
- Consider specialized roles for domain-specific projects

### 3. Quality Assurance
- Always include QA engineers in the team
- Implement comprehensive testing strategies
- Regular code reviews and documentation updates

### 4. Resource Management
- Monitor API usage and costs
- Optimize prompt engineering for efficiency
- Use caching for repeated operations

## Troubleshooting

### Common Issues
1. **API Rate Limits**: Implement proper rate limiting and retry mechanisms
2. **Memory Usage**: Monitor and optimize agent memory consumption
3. **Output Quality**: Fine-tune prompts and provide better context
4. **Integration Errors**: Validate external API responses and handle failures

### Performance Optimization
- Use efficient LLM models for specific tasks
- Implement parallel processing where possible
- Cache frequently used results
- Optimize document processing workflows

## Community and Resources

- **GitHub Repository**: [geekan/MetaGPT](https://github.com/geekan/MetaGPT)
- **Documentation**: [MetaGPT Docs](https://docs.deepwisdom.ai/)
- **Community Discord**: Join the MetaGPT community
- **Examples**: [MetaGPT Examples](https://github.com/geekan/MetaGPT/tree/main/examples)

## License

MetaGPT is released under the MIT License, supporting both open-source and commercial usage.

## Limitations

- Requires substantial computational resources for complex projects
- Quality depends heavily on LLM model capabilities
- Generated code may require human review and optimization
- Limited to text-based outputs (no direct GUI generation)
- API costs can accumulate quickly for large projects