# ThinkChain - Advanced Claude AI Interaction Library

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tool Discovery and Integration](#tool-discovery-and-integration)
- [Interleaved Thinking](#interleaved-thinking)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)

## Overview

ThinkChain is an advanced Python library that revolutionizes interaction with Claude AI by introducing interleaved thinking with real-time tool streaming. It enables dynamic tool discovery, concurrent execution, and provides a seamless integration between local tools and remote MCP (Model Context Protocol) servers.

**Key Characteristics:**
- **Real-time Thinking**: Watch Claude's reasoning process unfold live
- **Dynamic Tool Discovery**: Automatically finds and integrates tools
- **Interleaved Execution**: Tools execute during Claude's thinking process
- **Concurrent Processing**: Multiple tools can run simultaneously
- **Rich CLI Interface**: Interactive command-line experience

## Key Features

### 1. Interleaved Thinking with Tool Streaming
- Real-time visualization of Claude's thought process
- Tools execute during thinking, not after completion
- Server-sent events (SSE) for live progress updates
- Dynamic injection of tool results into reasoning

### 2. Dynamic Tool Discovery
- Automatic discovery from `/tools` directory
- Support for local Python tools
- Integration with MCP servers
- Hot-reloading of tool definitions
- Pydantic schema validation

### 3. Advanced Tool Execution
- Fine-grained tool execution control
- Early interception capabilities
- Concurrent tool execution
- Rich progress indicators
- Error handling and recovery

### 4. Flexible Configuration
- Configurable Claude models
- Thinking depth parameters
- Tool execution policies
- Custom tool directories
- Environment-based configuration

## Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Anthropic API key
export ANTHROPIC_API_KEY=your-api-key
```

### Zero-Setup Installation (Recommended)
```bash
# Run directly with uv (no installation required)
uv run --with thinkchain thinkchain

# Or with specific version
uv run --with thinkchain==0.1.0 thinkchain
```

### Traditional Installation
```bash
# Install via pip
pip install thinkchain

# Install with development dependencies
pip install "thinkchain[dev]"

# Install from source
git clone https://github.com/martinbowling/thinkchain.git
cd thinkchain
pip install -e .
```

### Environment Setup
```bash
# Create .env file in your project directory
echo "ANTHROPIC_API_KEY=your-actual-api-key" > .env

# Or export environment variable
export ANTHROPIC_API_KEY=your-actual-api-key
```

## Quick Start

### Basic Chat Interface
```python
from thinkchain import ThinkChain
import asyncio

async def basic_chat():
    # Initialize ThinkChain
    chain = ThinkChain()
    
    # Start an interactive conversation
    response = await chain.chat("What's the weather like today?")
    
    print("Claude's response:", response.content)
    print("Thinking process:", response.thinking)
    print("Tools used:", [tool.name for tool in response.tools_executed])

# Run the chat
asyncio.run(basic_chat())
```

### CLI Usage
```bash
# Start interactive chat
thinkchain chat

# Run single command
thinkchain run "Analyze the current directory structure"

# With specific model
thinkchain chat --model claude-3-5-sonnet-20241022

# Enable verbose thinking
thinkchain chat --thinking-depth deep

# Load custom tools directory
thinkchain chat --tools-dir ./my-tools
```

### Real-time Streaming
```python
from thinkchain import ThinkChain
import asyncio

async def stream_thinking():
    chain = ThinkChain(
        model="claude-3-5-sonnet-20241022",
        thinking_enabled=True
    )
    
    async def on_thinking(chunk):
        """Handle real-time thinking updates"""
        print(f"💭 {chunk.content}", end="", flush=True)
    
    async def on_tool_call(tool_call):
        """Handle tool execution"""
        print(f"\n🔧 Executing: {tool_call.name}")
        print(f"   Args: {tool_call.arguments}")
    
    async def on_tool_result(result):
        """Handle tool results"""
        print(f"✅ Tool result: {result.content[:100]}...")
    
    # Register event handlers
    chain.on_thinking = on_thinking
    chain.on_tool_call = on_tool_call
    chain.on_tool_result = on_tool_result
    
    # Start streaming conversation
    response = await chain.stream_chat(
        "Help me analyze this codebase and suggest improvements"
    )
    
    print(f"\n\nFinal response: {response.content}")

asyncio.run(stream_thinking())
```

## Tool Discovery and Integration

### Creating Local Tools

Create tools in the `/tools` directory:

```python
# tools/file_analyzer.py
from pydantic import BaseModel, Field
from typing import List, Dict
import os
import mimetypes

class FileAnalyzerInput(BaseModel):
    directory: str = Field(description="Directory path to analyze")
    include_hidden: bool = Field(default=False, description="Include hidden files")
    max_depth: int = Field(default=3, description="Maximum directory depth")

class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    type: str
    last_modified: float

async def analyze_directory(input_data: FileAnalyzerInput) -> Dict:
    """
    Analyze directory structure and file information.
    
    This tool recursively scans a directory and provides detailed
    information about files including size, type, and modification time.
    """
    
    files = []
    total_size = 0
    
    def scan_directory(path: str, current_depth: int = 0):
        nonlocal total_size
        
        if current_depth > input_data.max_depth:
            return
            
        try:
            for item in os.listdir(path):
                if not input_data.include_hidden and item.startswith('.'):
                    continue
                    
                item_path = os.path.join(path, item)
                
                if os.path.isfile(item_path):
                    stat = os.stat(item_path)
                    mime_type, _ = mimetypes.guess_type(item_path)
                    
                    file_info = FileInfo(
                        name=item,
                        path=item_path,
                        size=stat.st_size,
                        type=mime_type or "unknown",
                        last_modified=stat.st_mtime
                    )
                    files.append(file_info)
                    total_size += stat.st_size
                    
                elif os.path.isdir(item_path):
                    scan_directory(item_path, current_depth + 1)
                    
        except PermissionError:
            pass  # Skip directories we can't access
    
    scan_directory(input_data.directory)
    
    return {
        "total_files": len(files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "files": [file.dict() for file in files[:50]],  # Limit to first 50
        "file_types": list(set(file.type for file in files)),
        "analysis_complete": True
    }

# Tool metadata
__tool__ = {
    "name": "file_analyzer",
    "description": "Analyze directory structure and file information",
    "input_schema": FileAnalyzerInput.schema(),
    "function": analyze_directory
}
```

### Web Research Tool
```python
# tools/web_search.py
from pydantic import BaseModel, Field
import aiohttp
import json
from typing import List, Dict

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results to return")
    include_snippets: bool = Field(default=True, description="Include content snippets")

async def web_search(input_data: WebSearchInput) -> Dict:
    """
    Search the web for information on a given topic.
    
    This tool uses a search API to find relevant web content
    and return structured results with titles, URLs, and snippets.
    """
    
    # Example using a hypothetical search API
    # Replace with actual search service (DuckDuckGo, Google Custom Search, etc.)
    
    search_results = []
    
    # Simulated search results for demonstration
    mock_results = [
        {
            "title": f"Result {i+1} for '{input_data.query}'",
            "url": f"https://example.com/result-{i+1}",
            "snippet": f"This is a sample snippet for search result {i+1} about {input_data.query}. "
                      f"It contains relevant information that would be useful for research."
        }
        for i in range(input_data.num_results)
    ]
    
    return {
        "query": input_data.query,
        "results_count": len(mock_results),
        "results": mock_results,
        "search_successful": True
    }

__tool__ = {
    "name": "web_search",
    "description": "Search the web for information",
    "input_schema": WebSearchInput.schema(),
    "function": web_search
}
```

### MCP Server Integration
```python
# Configure MCP servers in your ThinkChain setup
from thinkchain import ThinkChain
from thinkchain.mcp import MCPServer

async def setup_with_mcp():
    # Define MCP servers
    mcp_servers = [
        MCPServer(
            name="filesystem",
            command=["node", "/path/to/filesystem-server.js"],
            env={"LOG_LEVEL": "debug"}
        ),
        MCPServer(
            name="browser",
            command=["npx", "browser-mcp-server"],
            env={"HEADLESS": "true"}
        )
    ]
    
    # Initialize ThinkChain with MCP servers
    chain = ThinkChain(
        mcp_servers=mcp_servers,
        tools_directory="./tools"
    )
    
    # MCP tools are automatically discovered and available
    response = await chain.chat(
        "Use the filesystem tools to analyze the project structure, "
        "then use the browser to research best practices for the technologies found."
    )
    
    return response
```

## Interleaved Thinking

### Understanding the Thinking Process
```python
from thinkchain import ThinkChain, ThinkingMode
import asyncio

async def observe_thinking():
    chain = ThinkChain(
        thinking_mode=ThinkingMode.INTERLEAVED,
        thinking_depth="deep",
        show_thinking_markers=True
    )
    
    thinking_log = []
    
    def capture_thinking(chunk):
        thinking_log.append({
            "timestamp": chunk.timestamp,
            "content": chunk.content,
            "type": chunk.type,  # 'thinking', 'tool_call', 'tool_result'
            "metadata": chunk.metadata
        })
        
        # Real-time display
        if chunk.type == "thinking":
            print(f"💭 {chunk.content}", end="")
        elif chunk.type == "tool_call":
            print(f"\n🔧 Tool: {chunk.metadata.get('tool_name')}")
        elif chunk.type == "tool_result":
            print(f"✅ Result received")
    
    chain.on_thinking_chunk = capture_thinking
    
    response = await chain.chat(
        "I need to understand the architecture of a complex software project. "
        "Analyze the codebase, identify the main components, and suggest "
        "potential improvements or optimizations."
    )
    
    # Analyze thinking patterns
    thinking_phases = []
    current_phase = []
    
    for entry in thinking_log:
        if entry["type"] == "thinking":
            current_phase.append(entry["content"])
        elif entry["type"] in ["tool_call", "tool_result"]:
            if current_phase:
                thinking_phases.append("".join(current_phase))
                current_phase = []
    
    print(f"\n\nThinking Analysis:")
    print(f"Total thinking phases: {len(thinking_phases)}")
    print(f"Tools executed: {len([e for e in thinking_log if e['type'] == 'tool_call'])}")
    print(f"Average phase length: {sum(len(p) for p in thinking_phases) // len(thinking_phases) if thinking_phases else 0} chars")

asyncio.run(observe_thinking())
```

### Tool Execution During Thinking
```python
from thinkchain import ThinkChain, ToolExecutionPolicy
import asyncio

async def controlled_tool_execution():
    chain = ThinkChain(
        tool_execution_policy=ToolExecutionPolicy.EARLY_INTERCEPT,
        max_concurrent_tools=3,
        tool_timeout=30.0
    )
    
    # Track tool execution timeline
    execution_timeline = []
    
    async def on_tool_start(tool_call):
        execution_timeline.append({
            "event": "tool_start",
            "tool": tool_call.name,
            "timestamp": tool_call.timestamp,
            "thinking_position": tool_call.thinking_position
        })
        print(f"🚀 Starting {tool_call.name} at thinking position {tool_call.thinking_position}")
    
    async def on_tool_complete(result):
        execution_timeline.append({
            "event": "tool_complete", 
            "tool": result.tool_name,
            "timestamp": result.timestamp,
            "success": result.success,
            "execution_time": result.execution_time
        })
        print(f"✅ Completed {result.tool_name} in {result.execution_time:.2f}s")
    
    chain.on_tool_start = on_tool_start
    chain.on_tool_complete = on_tool_complete
    
    response = await chain.chat(
        "I need to build a comprehensive report on this project. "
        "Analyze the code structure, check for security issues, "
        "review dependencies, and generate documentation."
    )
    
    # Analyze execution patterns
    print(f"\nExecution Timeline:")
    for event in execution_timeline:
        print(f"  {event['timestamp']}: {event['event']} - {event['tool']}")

asyncio.run(controlled_tool_execution())
```

## Configuration

### Environment Configuration
```python
# .env file
ANTHROPIC_API_KEY=your-api-key
THINKCHAIN_MODEL=claude-3-5-sonnet-20241022
THINKCHAIN_THINKING_ENABLED=true
THINKCHAIN_THINKING_DEPTH=deep
THINKCHAIN_TOOLS_DIR=./tools
THINKCHAIN_MAX_CONCURRENT_TOOLS=3
THINKCHAIN_TOOL_TIMEOUT=30
THINKCHAIN_LOG_LEVEL=info
```

### Programmatic Configuration
```python
from thinkchain import ThinkChain, ThinkChainConfig, ThinkingMode, ToolExecutionPolicy

# Create comprehensive configuration
config = ThinkChainConfig(
    # Model settings
    model="claude-3-5-sonnet-20241022",
    api_key="your-api-key",
    max_tokens=8192,
    temperature=0.7,
    
    # Thinking configuration
    thinking_enabled=True,
    thinking_mode=ThinkingMode.INTERLEAVED,
    thinking_depth="deep",
    show_thinking_markers=True,
    thinking_stream_delay=0.05,  # 50ms delay between chunks
    
    # Tool execution
    tools_directory="./tools",
    tool_execution_policy=ToolExecutionPolicy.EARLY_INTERCEPT,
    max_concurrent_tools=3,
    tool_timeout=30.0,
    auto_discover_tools=True,
    validate_tool_schemas=True,
    
    # MCP integration
    mcp_servers=[
        {
            "name": "filesystem",
            "command": ["node", "filesystem-server.js"],
            "env": {"LOG_LEVEL": "info"}
        }
    ],
    
    # Logging and debugging
    log_level="info",
    debug_mode=False,
    save_conversations=True,
    conversation_history_limit=100,
    
    # Performance
    stream_buffer_size=1024,
    connection_timeout=60.0,
    retry_attempts=3,
    retry_delay=1.0
)

# Initialize with configuration
chain = ThinkChain(config=config)
```

### Tool Configuration
```python
# tools/config.json
{
  "global_settings": {
    "timeout": 30,
    "retry_attempts": 2,
    "validate_inputs": true
  },
  "tools": {
    "file_analyzer": {
      "max_files": 1000,
      "allowed_extensions": [".py", ".js", ".md", ".txt"],
      "exclude_directories": ["node_modules", ".git", "__pycache__"]
    },
    "web_search": {
      "api_key_env": "SEARCH_API_KEY",
      "max_results": 10,
      "rate_limit": "10/minute"
    }
  }
}
```

## API Reference

### Core Classes

#### ThinkChain
```python
class ThinkChain:
    def __init__(
        self,
        config: Optional[ThinkChainConfig] = None,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        thinking_enabled: bool = True,
        tools_directory: str = "./tools"
    ):
        """Initialize ThinkChain with configuration"""
        pass
    
    async def chat(self, message: str, **kwargs) -> ThinkChainResponse:
        """Send a message and get response"""
        pass
    
    async def stream_chat(self, message: str, **kwargs) -> AsyncIterator[ThinkChainChunk]:
        """Stream conversation with real-time updates"""
        pass
    
    async def add_tool(self, tool: Tool) -> None:
        """Dynamically add a tool"""
        pass
    
    async def list_tools(self) -> List[ToolInfo]:
        """List all available tools"""
        pass
    
    def on_thinking(self, callback: Callable) -> None:
        """Register thinking event handler"""
        pass
    
    def on_tool_call(self, callback: Callable) -> None:
        """Register tool call event handler"""
        pass
```

#### ThinkChainResponse
```python
@dataclass
class ThinkChainResponse:
    content: str
    thinking: Optional[str]
    tools_executed: List[ToolExecution]
    metadata: Dict
    conversation_id: str
    timestamp: datetime
    
    @property
    def thinking_summary(self) -> str:
        """Get summarized thinking process"""
        pass
    
    @property  
    def tool_results(self) -> List[ToolResult]:
        """Get all tool execution results"""
        pass
```

#### Tool Definition
```python
@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    input_schema: Dict
    output_schema: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    
    async def execute(self, input_data: Any) -> Any:
        """Execute the tool with given input"""
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input against schema"""
        pass
```

### Event System
```python
from thinkchain.events import EventHandler, ThinkingEvent, ToolEvent

class CustomEventHandler(EventHandler):
    async def on_thinking_start(self, event: ThinkingEvent):
        """Handle start of thinking process"""
        print(f"Claude started thinking about: {event.context}")
    
    async def on_thinking_chunk(self, event: ThinkingEvent):
        """Handle thinking chunk"""
        print(f"Thinking: {event.content}")
    
    async def on_tool_discovery(self, event: ToolEvent):
        """Handle tool discovery"""
        print(f"Discovered tool: {event.tool_name}")
    
    async def on_tool_execution(self, event: ToolEvent):
        """Handle tool execution"""
        print(f"Executing: {event.tool_name} with args {event.arguments}")
    
    async def on_error(self, event):
        """Handle errors"""
        print(f"Error occurred: {event.error}")

# Use custom event handler
chain = ThinkChain(event_handler=CustomEventHandler())
```

## Advanced Usage

### Custom Tool Development
```python
from thinkchain.tools import BaseTool, ToolResult
from pydantic import BaseModel, Field
import asyncio
from typing import AsyncIterator

class DatabaseQueryInput(BaseModel):
    query: str = Field(description="SQL query to execute")
    database: str = Field(description="Database name")
    limit: int = Field(default=100, description="Result limit")

class DatabaseTool(BaseTool):
    """Advanced database query tool with streaming results"""
    
    name = "database_query"
    description = "Execute SQL queries and return results"
    input_schema = DatabaseQueryInput
    
    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string
        self.connections = {}
    
    async def execute(self, input_data: DatabaseQueryInput) -> ToolResult:
        """Execute database query with connection pooling"""
        
        try:
            # Get or create connection
            if input_data.database not in self.connections:
                await self._create_connection(input_data.database)
            
            conn = self.connections[input_data.database]
            
            # Execute query with streaming
            result_stream = self._execute_streaming_query(
                conn, input_data.query, input_data.limit
            )
            
            results = []
            async for row in result_stream:
                results.append(row)
                
                # Yield intermediate results for real-time feedback
                if len(results) % 10 == 0:
                    yield ToolResult(
                        success=True,
                        content=f"Retrieved {len(results)} rows so far...",
                        partial=True,
                        metadata={"rows_processed": len(results)}
                    )
            
            return ToolResult(
                success=True,
                content={
                    "query": input_data.query,
                    "results": results,
                    "row_count": len(results),
                    "database": input_data.database
                },
                metadata={
                    "execution_time": self.execution_time,
                    "database": input_data.database
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"Database error: {str(e)}",
                metadata={"error_type": type(e).__name__}
            )
    
    async def _create_connection(self, database: str):
        """Create database connection"""
        # Implementation depends on database type
        pass
    
    async def _execute_streaming_query(self, conn, query: str, limit: int) -> AsyncIterator:
        """Execute query with streaming results"""
        # Implementation for streaming query execution
        pass

# Register custom tool
chain = ThinkChain()
db_tool = DatabaseTool("postgresql://user:pass@localhost/db")
await chain.add_tool(db_tool)
```

### Multi-Modal Tool Integration
```python
from thinkchain.tools import MultiModalTool
import base64
from PIL import Image
import io

class ImageAnalysisTool(MultiModalTool):
    """Tool for analyzing images with vision capabilities"""
    
    name = "image_analysis"
    description = "Analyze images and extract information"
    
    async def execute(self, input_data) -> ToolResult:
        """Analyze image and return structured results"""
        
        if "image_url" in input_data:
            image_data = await self._download_image(input_data["image_url"])
        elif "image_base64" in input_data:
            image_data = base64.b64decode(input_data["image_base64"])
        else:
            return ToolResult(success=False, content="No image provided")
        
        # Analyze image
        analysis = await self._analyze_image(image_data)
        
        # Extract text if present
        text_content = await self._extract_text(image_data)
        
        # Detect objects
        objects = await self._detect_objects(image_data)
        
        return ToolResult(
            success=True,
            content={
                "analysis": analysis,
                "text_content": text_content,
                "objects_detected": objects,
                "image_metadata": {
                    "size": len(image_data),
                    "format": self._detect_format(image_data)
                }
            }
        )
    
    async def _analyze_image(self, image_data: bytes) -> dict:
        """Perform image analysis"""
        # Integration with vision models
        pass
    
    async def _extract_text(self, image_data: bytes) -> str:
        """Extract text from image using OCR"""
        # OCR implementation
        pass
    
    async def _detect_objects(self, image_data: bytes) -> list:
        """Detect objects in image"""
        # Object detection implementation
        pass
```

### Conversation Management
```python
from thinkchain import ConversationManager, ConversationHistory
import json

class ProjectConversationManager:
    """Manage conversations with project context"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.chain = ThinkChain(tools_directory=f"{project_path}/tools")
        self.conversation_history = ConversationHistory()
        
        # Load project context
        self.project_context = self._load_project_context()
    
    async def start_session(self, session_name: str):
        """Start a new conversation session"""
        
        # Initialize with project context
        context_message = f"""
        Project Context:
        - Path: {self.project_path}
        - Description: {self.project_context.get('description', 'No description')}
        - Technologies: {', '.join(self.project_context.get('technologies', []))}
        - Key Files: {', '.join(self.project_context.get('key_files', []))}
        
        Session: {session_name}
        """
        
        response = await self.chain.chat(
            f"I'm starting a new session for this project. {context_message}"
        )
        
        # Save session start
        self.conversation_history.add_session(session_name, response)
        
        return response
    
    async def continue_conversation(self, message: str, session_name: str = None):
        """Continue conversation with context"""
        
        # Add relevant history
        if session_name:
            history = self.conversation_history.get_session_history(session_name)
            context = self._format_history_context(history)
            message = f"Previous context: {context}\n\nCurrent request: {message}"
        
        response = await self.chain.chat(message)
        
        # Update history
        self.conversation_history.add_message(session_name or "default", message, response)
        
        return response
    
    def _load_project_context(self) -> dict:
        """Load project context from various sources"""
        context = {}
        
        # Try to load from package.json, requirements.txt, etc.
        context_files = [
            "package.json", "requirements.txt", "Cargo.toml", 
            "pom.xml", "README.md", ".thinkchain/context.json"
        ]
        
        for file_path in context_files:
            full_path = f"{self.project_path}/{file_path}"
            if os.path.exists(full_path):
                context[file_path] = self._extract_context_from_file(full_path)
        
        return context
    
    def _extract_context_from_file(self, file_path: str) -> dict:
        """Extract relevant context from project files"""
        # Implementation depends on file type
        pass

# Usage
project_manager = ProjectConversationManager("./my-project")
response = await project_manager.start_session("refactoring-session")
```

## Best Practices

### Tool Design Guidelines
```python
# Good tool design example
from thinkchain.tools import BaseTool, ToolResult
from pydantic import BaseModel, Field, validator
import asyncio
from typing import Optional, List

class WellDesignedToolInput(BaseModel):
    """Well-structured input with validation and documentation"""
    
    target: str = Field(
        description="Target for the operation (file path, URL, etc.)",
        example="/path/to/file.txt"
    )
    
    options: Optional[List[str]] = Field(
        default=[],
        description="Additional options for the operation",
        example=["--verbose", "--force"]
    )
    
    timeout: float = Field(
        default=30.0,
        description="Operation timeout in seconds",
        ge=1.0,  # Greater than or equal to 1
        le=300.0  # Less than or equal to 300
    )
    
    @validator('target')
    def validate_target(cls, v):
        """Validate target format"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Target cannot be empty")
        return v.strip()

class WellDesignedTool(BaseTool):
    """Example of a well-designed tool following best practices"""
    
    name = "well_designed_tool"
    description = """
    A comprehensive tool that demonstrates best practices:
    - Clear, descriptive name and documentation
    - Structured input validation
    - Comprehensive error handling
    - Progress reporting
    - Resource cleanup
    """
    
    input_schema = WellDesignedToolInput
    
    def __init__(self):
        super().__init__()
        self.resources = []
    
    async def execute(self, input_data: WellDesignedToolInput) -> ToolResult:
        """Execute with comprehensive error handling and cleanup"""
        
        start_time = time.time()
        
        try:
            # Validate preconditions
            await self._validate_preconditions(input_data)
            
            # Set up resources
            await self._setup_resources(input_data)
            
            # Execute main operation with progress reporting
            result = await self._execute_main_operation(input_data)
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                success=True,
                content=result,
                metadata={
                    "execution_time": execution_time,
                    "target": input_data.target,
                    "options_used": input_data.options
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"Operation failed: {str(e)}",
                metadata={
                    "error_type": type(e).__name__,
                    "execution_time": time.time() - start_time,
                    "target": input_data.target
                }
            )
        
        finally:
            # Always clean up resources
            await self._cleanup_resources()
    
    async def _validate_preconditions(self, input_data: WellDesignedToolInput):
        """Validate that all preconditions are met"""
        pass
    
    async def _setup_resources(self, input_data: WellDesignedToolInput):
        """Set up any resources needed for execution"""
        pass
    
    async def _execute_main_operation(self, input_data: WellDesignedToolInput):
        """Execute the main operation with progress updates"""
        
        # Report progress during long operations
        for i in range(5):
            await asyncio.sleep(1)  # Simulate work
            await self._report_progress(f"Step {i+1}/5 completed")
        
        return {"status": "completed", "result": "operation successful"}
    
    async def _cleanup_resources(self):
        """Clean up any resources used during execution"""
        for resource in self.resources:
            try:
                await resource.cleanup()
            except Exception as e:
                # Log cleanup errors but don't fail the operation
                print(f"Warning: Failed to cleanup resource: {e}")
        self.resources.clear()
```

### Error Handling and Recovery
```python
from thinkchain import ThinkChain, ToolExecutionError
import asyncio
from typing import Dict, Any

class RobustThinkChain:
    """ThinkChain wrapper with enhanced error handling"""
    
    def __init__(self, **kwargs):
        self.chain = ThinkChain(**kwargs)
        self.retry_attempts = 3
        self.fallback_strategies = {}
        self.error_history = []
    
    async def robust_chat(
        self, 
        message: str, 
        fallback_message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Chat with automatic retry and fallback strategies"""
        
        for attempt in range(self.retry_attempts):
            try:
                response = await self.chain.chat(message, **kwargs)
                
                # Validate response quality
                if self._is_response_valid(response):
                    return {
                        "success": True,
                        "response": response,
                        "attempt": attempt + 1,
                        "strategy": "direct"
                    }
                else:
                    raise ValueError("Response quality insufficient")
                    
            except ToolExecutionError as e:
                self.error_history.append({
                    "type": "tool_error",
                    "error": str(e),
                    "attempt": attempt + 1,
                    "tool": e.tool_name
                })
                
                # Try fallback strategy for this tool
                if e.tool_name in self.fallback_strategies:
                    try:
                        fallback_result = await self._execute_fallback(
                            e.tool_name, e.input_data
                        )
                        
                        # Inject fallback result and retry
                        modified_message = self._inject_fallback_result(
                            message, e.tool_name, fallback_result
                        )
                        
                        response = await self.chain.chat(modified_message, **kwargs)
                        
                        return {
                            "success": True,
                            "response": response,
                            "attempt": attempt + 1,
                            "strategy": "fallback",
                            "fallback_used": e.tool_name
                        }
                        
                    except Exception as fallback_error:
                        self.error_history.append({
                            "type": "fallback_error",
                            "error": str(fallback_error),
                            "tool": e.tool_name
                        })
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                self.error_history.append({
                    "type": "general_error",
                    "error": str(e),
                    "attempt": attempt + 1
                })
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
        
        # All attempts failed, try fallback message
        if fallback_message:
            try:
                response = await self.chain.chat(fallback_message, **kwargs)
                return {
                    "success": True,
                    "response": response,
                    "strategy": "fallback_message",
                    "original_failed": True
                }
            except Exception as e:
                pass
        
        # Complete failure
        return {
            "success": False,
            "error_history": self.error_history[-self.retry_attempts:],
            "message": "All retry attempts failed"
        }
    
    def register_fallback_strategy(self, tool_name: str, strategy_func):
        """Register a fallback strategy for a specific tool"""
        self.fallback_strategies[tool_name] = strategy_func
    
    def _is_response_valid(self, response) -> bool:
        """Validate response quality"""
        if not response or not response.content:
            return False
        
        # Check for common error indicators
        error_indicators = [
            "I cannot", "I'm unable", "Error:", "Failed to", 
            "Something went wrong", "An error occurred"
        ]
        
        content_lower = response.content.lower()
        if any(indicator.lower() in content_lower for indicator in error_indicators):
            return False
        
        # Check minimum content length
        if len(response.content.strip()) < 10:
            return False
        
        return True
    
    async def _execute_fallback(self, tool_name: str, input_data: Any) -> Any:
        """Execute fallback strategy for a tool"""
        if tool_name in self.fallback_strategies:
            return await self.fallback_strategies[tool_name](input_data)
        return None
    
    def _inject_fallback_result(self, message: str, tool_name: str, result: Any) -> str:
        """Inject fallback result into message"""
        fallback_context = f"""
        Note: The {tool_name} tool encountered an issue, but I was able to 
        get alternative information: {result}
        
        Original request: {message}
        """
        return fallback_context

# Usage example
robust_chain = RobustThinkChain(
    model="claude-3-5-sonnet-20241022",
    thinking_enabled=True
)

# Register fallback strategy
async def web_search_fallback(query_data):
    # Simple fallback that returns cached or default results
    return {"results": ["Fallback search result"], "source": "cache"}

robust_chain.register_fallback_strategy("web_search", web_search_fallback)

# Use with automatic recovery
result = await robust_chain.robust_chat(
    "Research the latest developments in AI and create a summary",
    fallback_message="Please create a general overview of AI developments based on your knowledge"
)

if result["success"]:
    print(f"Response obtained using {result['strategy']} strategy")
    print(result["response"].content)
else:
    print("All attempts failed:")
    for error in result["error_history"]:
        print(f"  - {error}")
```

ThinkChain provides a revolutionary approach to AI interaction by enabling real-time observation of Claude's thinking process while seamlessly integrating dynamic tool execution, making it ideal for complex problem-solving and development workflows.