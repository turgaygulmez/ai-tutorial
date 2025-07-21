# LlamaNode - Llama 2 in Node.js

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Performance Optimization](#performance-optimization)
- [Use Cases](#use-cases)

## Overview

LlamaNode is a Node.js library that enables running Llama 2 models locally in JavaScript applications. It provides a simple API for integrating large language models directly into Node.js applications without requiring external services or Python dependencies.

**Key Characteristics:**
- **Local Execution**: Run Llama 2 models directly in Node.js
- **No Python Required**: Pure JavaScript/TypeScript implementation
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Memory Efficient**: Optimized for various hardware configurations

## Key Features

### 1. Local LLM Execution
- Run Llama 2 models locally in Node.js
- No external API dependencies
- Privacy-focused with local processing
- Offline capabilities

### 2. Multiple Model Support
- Support for various Llama 2 model sizes
- GGML format compatibility
- Quantized models for memory efficiency
- Custom model loading

### 3. Streaming Support
- Real-time token streaming
- Progressive response generation
- Event-driven architecture
- Cancellable operations

### 4. TypeScript Support
- Full TypeScript definitions
- Type-safe API
- IntelliSense support
- Modern JavaScript features

## Installation

### Prerequisites
```bash
# Node.js 16 or higher
node --version

# npm or yarn
npm --version
```

### Basic Installation
```bash
# Install via npm
npm install llama-node

# Install via yarn
yarn add llama-node
```

### Platform-Specific Installation
```bash
# For macOS with Metal support
npm install llama-node @llama-node/core-darwin-arm64

# For Linux with CUDA support
npm install llama-node @llama-node/core-linux-cuda

# For Windows
npm install llama-node @llama-node/core-win32
```

### TypeScript Support
```bash
# TypeScript definitions are included
npm install --save-dev typescript @types/node
```

## Quick Start

### Basic Text Generation
```javascript
import { LLM } from "llama-node";
import { LLamaCpp } from "llama-node/dist/llm/llama-cpp.js";
import path from "path";

// Initialize the model
const llama = new LLM(LLamaCpp);
const model = path.resolve(process.cwd(), "models/llama-2-7b-chat.q4_0.bin");

// Load the model
const config = {
    modelPath: model,
    enableLogging: true,
    nCtx: 1024,
    seed: 0,
    f16Kv: false,
    logitsAll: false,
    vocabOnly: false,
    useMlock: false,
    embedding: false,
    useMmap: true,
    nGpuLayers: 0
};

await llama.load(config);

// Generate text
const prompt = "Tell me about artificial intelligence:";
const response = await llama.createCompletion({
    nThreads: 4,
    nTokPredict: 2048,
    topK: 40,
    topP: 0.1,
    temp: 0.2,
    repeatPenalty: 1.1,
}, (data) => {
    process.stdout.write(data.token);
});

console.log("\nGeneration completed!");
```

### Streaming Response
```javascript
import { LLM } from "llama-node";
import { LLamaCpp } from "llama-node/dist/llm/llama-cpp.js";

async function streamingExample() {
    const llama = new LLM(LLamaCpp);
    
    await llama.load({
        modelPath: "./models/llama-2-7b-chat.q4_0.bin",
        enableLogging: true,
        nCtx: 1024,
    });

    const prompt = "Explain quantum computing in simple terms:";
    
    console.log("Streaming response:");
    
    const params = {
        nThreads: 4,
        nTokPredict: 512,
        topK: 40,
        topP: 0.1,
        temp: 0.8,
        repeatPenalty: 1.1,
        prompt: prompt,
    };

    await llama.createCompletion(params, (data) => {
        // Handle streaming tokens
        if (data.token) {
            process.stdout.write(data.token);
        }
        
        // Handle completion
        if (data.completed) {
            console.log("\n\nStreaming completed!");
            console.log(`Total tokens: ${data.timings?.predicted_n}`);
            console.log(`Speed: ${data.timings?.predicted_per_second.toFixed(2)} tokens/s`);
        }
    });
}

streamingExample().catch(console.error);
```

### Chat Interface
```javascript
import { LLM } from "llama-node";
import { LLamaCpp } from "llama-node/dist/llm/llama-cpp.js";
import readline from 'readline';

class ChatBot {
    constructor(modelPath) {
        this.llama = new LLM(LLamaCpp);
        this.modelPath = modelPath;
        this.conversationHistory = [];
        this.isLoaded = false;
    }

    async initialize() {
        console.log("Loading model...");
        
        await this.llama.load({
            modelPath: this.modelPath,
            enableLogging: false,
            nCtx: 2048,
            nGpuLayers: 0,
        });
        
        this.isLoaded = true;
        console.log("Model loaded successfully!\n");
    }

    async chat(userInput) {
        if (!this.isLoaded) {
            throw new Error("Model not loaded");
        }

        // Add user input to conversation history
        this.conversationHistory.push(`User: ${userInput}`);
        
        // Create prompt with context
        const prompt = this.buildPrompt(userInput);
        
        console.log("Assistant: ");
        let response = "";

        const params = {
            nThreads: 4,
            nTokPredict: 512,
            topK: 40,
            topP: 0.9,
            temp: 0.7,
            repeatPenalty: 1.1,
            prompt: prompt,
        };

        await this.llama.createCompletion(params, (data) => {
            if (data.token) {
                process.stdout.write(data.token);
                response += data.token;
            }
        });

        console.log("\n");
        
        // Add assistant response to history
        this.conversationHistory.push(`Assistant: ${response.trim()}`);
        
        // Keep conversation history manageable
        if (this.conversationHistory.length > 10) {
            this.conversationHistory = this.conversationHistory.slice(-8);
        }

        return response.trim();
    }

    buildPrompt(userInput) {
        const context = this.conversationHistory.slice(-6).join('\n');
        return `${context}\nUser: ${userInput}\nAssistant: `;
    }

    async startChat() {
        await this.initialize();
        
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        console.log("Chat started! Type 'exit' to quit.\n");

        const askQuestion = () => {
            rl.question("You: ", async (input) => {
                if (input.toLowerCase() === 'exit') {
                    console.log("Goodbye!");
                    rl.close();
                    return;
                }

                try {
                    await this.chat(input);
                } catch (error) {
                    console.error("Error:", error.message);
                }

                askQuestion();
            });
        };

        askQuestion();
    }
}

// Usage
const chatBot = new ChatBot("./models/llama-2-7b-chat.q4_0.bin");
chatBot.startChat().catch(console.error);
```

## Supported Models

### Model Formats
LlamaNode supports GGML format models:

```javascript
// Supported model types
const modelTypes = {
    // Original Llama models
    "llama-7b": "7B parameter model",
    "llama-13b": "13B parameter model", 
    "llama-30b": "30B parameter model",
    "llama-65b": "65B parameter model",
    
    // Llama 2 models
    "llama-2-7b": "Llama 2 7B base model",
    "llama-2-7b-chat": "Llama 2 7B chat model",
    "llama-2-13b": "Llama 2 13B base model",
    "llama-2-13b-chat": "Llama 2 13B chat model",
    "llama-2-70b": "Llama 2 70B base model",
    "llama-2-70b-chat": "Llama 2 70B chat model",
    
    // Code Llama models
    "codellama-7b": "Code Llama 7B",
    "codellama-13b": "Code Llama 13B",
    "codellama-34b": "Code Llama 34B"
};
```

### Quantization Levels
```javascript
// Different quantization levels for memory efficiency
const quantizationTypes = {
    "q4_0": "4-bit quantization (smallest, fastest)",
    "q4_1": "4-bit quantization (better quality)",
    "q5_0": "5-bit quantization (balanced)",
    "q5_1": "5-bit quantization (better quality)",
    "q8_0": "8-bit quantization (highest quality)",
    "f16": "16-bit floating point",
    "f32": "32-bit floating point (original)"
};

// Model selection helper
function selectModel(size, quality, task) {
    if (task === "chat") {
        if (size === "small") return "llama-2-7b-chat.q4_0.bin";
        if (size === "medium") return "llama-2-13b-chat.q4_0.bin";
        if (size === "large") return "llama-2-70b-chat.q4_0.bin";
    } else if (task === "code") {
        if (size === "small") return "codellama-7b.q4_0.bin";
        if (size === "medium") return "codellama-13b.q4_0.bin";
        if (size === "large") return "codellama-34b.q4_0.bin";
    }
    
    return "llama-2-7b.q4_0.bin"; // Default
}
```

### Model Download
```javascript
import fs from 'fs';
import https from 'https';
import path from 'path';

async function downloadModel(url, modelName) {
    const modelsDir = './models';
    if (!fs.existsSync(modelsDir)) {
        fs.mkdirSync(modelsDir);
    }
    
    const filePath = path.join(modelsDir, modelName);
    
    console.log(`Downloading ${modelName}...`);
    
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(filePath);
        
        https.get(url, (response) => {
            const totalSize = parseInt(response.headers['content-length']);
            let downloadedSize = 0;
            
            response.on('data', (chunk) => {
                downloadedSize += chunk.length;
                const progress = ((downloadedSize / totalSize) * 100).toFixed(1);
                process.stdout.write(`\rProgress: ${progress}%`);
            });
            
            response.pipe(file);
            
            file.on('finish', () => {
                file.close();
                console.log(`\n${modelName} downloaded successfully!`);
                resolve(filePath);
            });
        }).on('error', (error) => {
            fs.unlink(filePath, () => {}); // Delete partial file
            reject(error);
        });
    });
}

// Usage
const modelUrl = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.q4_0.bin";
await downloadModel(modelUrl, "llama-2-7b-chat.q4_0.bin");
```

## Configuration

### Model Configuration
```javascript
const config = {
    // Model file path
    modelPath: "./models/llama-2-7b-chat.q4_0.bin",
    
    // Context window size
    nCtx: 2048,
    
    // Random seed for reproducible results
    seed: -1,  // -1 for random seed
    
    // Use 16-bit key-value cache
    f16Kv: false,
    
    // Compute all logits
    logitsAll: false,
    
    // Load vocabulary only
    vocabOnly: false,
    
    // Use memory locking
    useMlock: false,
    
    // Enable embeddings mode
    embedding: false,
    
    // Use memory mapping
    useMmap: true,
    
    // Number of GPU layers to offload
    nGpuLayers: 0,
    
    // Enable logging
    enableLogging: true,
    
    // Model architecture
    nBatch: 512,
    
    // Number of threads
    nThreads: 4,
};
```

### Generation Parameters
```javascript
const generationParams = {
    // Number of tokens to predict
    nTokPredict: 512,
    
    // Temperature for randomness (0.0 = deterministic, 1.0 = very random)
    temp: 0.8,
    
    // Top-K sampling (0 = disabled)
    topK: 40,
    
    // Top-P sampling (1.0 = disabled)  
    topP: 0.95,
    
    // Repeat penalty (1.0 = no penalty)
    repeatPenalty: 1.1,
    
    // Number of threads to use
    nThreads: 4,
    
    // Input prompt
    prompt: "Your prompt here",
    
    // Stop sequences
    stopSequence: ["\n", "User:", "Assistant:"],
    
    // Stream tokens as they're generated
    stream: true
};
```

### Performance Configuration
```javascript
// CPU-optimized configuration
const cpuConfig = {
    modelPath: "./models/llama-2-7b-chat.q4_0.bin",
    nCtx: 1024,
    nThreads: Math.max(1, require('os').cpus().length - 1),
    nBatch: 256,
    useMmap: true,
    useMlock: false,
    nGpuLayers: 0
};

// GPU-optimized configuration (if supported)
const gpuConfig = {
    modelPath: "./models/llama-2-7b-chat.q5_1.bin",
    nCtx: 2048,
    nThreads: 4,
    nBatch: 512,
    useMmap: true,
    useMlock: false,
    nGpuLayers: 32  // Adjust based on GPU memory
};

// Memory-constrained configuration
const lowMemoryConfig = {
    modelPath: "./models/llama-2-7b-chat.q4_0.bin",
    nCtx: 512,
    nThreads: 2,
    nBatch: 128,
    useMmap: false,
    useMlock: false,
    f16Kv: false
};
```

## Advanced Usage

### Custom Prompt Templates
```javascript
class PromptTemplate {
    static chatTemplate(systemPrompt, userMessage, conversationHistory = []) {
        let prompt = `<s>[INST] <<SYS>>\n${systemPrompt}\n<</SYS>>\n\n`;
        
        // Add conversation history
        for (let i = 0; i < conversationHistory.length; i += 2) {
            const user = conversationHistory[i];
            const assistant = conversationHistory[i + 1];
            if (user && assistant) {
                prompt += `${user} [/INST] ${assistant} </s><s>[INST] `;
            }
        }
        
        prompt += `${userMessage} [/INST] `;
        return prompt;
    }
    
    static instructionTemplate(instruction, input = "") {
        return `### Instruction:\n${instruction}\n\n### Input:\n${input}\n\n### Response:\n`;
    }
    
    static codeTemplate(task, language = "javascript") {
        return `// Task: ${task}\n// Language: ${language}\n// Code:\n\n`;
    }
}

// Usage examples
const systemPrompt = "You are a helpful AI assistant that provides accurate and concise answers.";
const userMessage = "Explain machine learning in simple terms.";
const chatPrompt = PromptTemplate.chatTemplate(systemPrompt, userMessage);

const instructionPrompt = PromptTemplate.instructionTemplate(
    "Translate the following text to Spanish",
    "Hello, how are you today?"
);

const codePrompt = PromptTemplate.codeTemplate(
    "Create a function to calculate fibonacci numbers",
    "python"
);
```

### Response Processing
```javascript
class ResponseProcessor {
    constructor() {
        this.buffer = "";
        this.stopSequences = ["\n\n", "User:", "Human:"];
    }
    
    processToken(token) {
        this.buffer += token;
        
        // Check for stop sequences
        for (const stopSeq of this.stopSequences) {
            if (this.buffer.includes(stopSeq)) {
                const stopIndex = this.buffer.indexOf(stopSeq);
                const finalText = this.buffer.substring(0, stopIndex);
                return { text: finalText, shouldStop: true };
            }
        }
        
        return { text: token, shouldStop: false };
    }
    
    cleanResponse(text) {
        // Remove common artifacts
        return text
            .replace(/^\s*Assistant:\s*/i, '')
            .replace(/^\s*AI:\s*/i, '')
            .replace(/\[INST\]|\[\/INST\]/g, '')
            .trim();
    }
    
    extractCodeBlocks(text) {
        const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
        const matches = [];
        let match;
        
        while ((match = codeBlockRegex.exec(text)) !== null) {
            matches.push({
                language: match[1] || 'text',
                code: match[2].trim()
            });
        }
        
        return matches;
    }
}

// Usage
const processor = new ResponseProcessor();

await llama.createCompletion(params, (data) => {
    if (data.token) {
        const result = processor.processToken(data.token);
        
        if (!result.shouldStop) {
            process.stdout.write(result.text);
        } else {
            const cleanText = processor.cleanResponse(result.text);
            console.log("\n\nFinal response:", cleanText);
            
            // Extract code blocks if any
            const codeBlocks = processor.extractCodeBlocks(cleanText);
            if (codeBlocks.length > 0) {
                console.log("\nFound code blocks:");
                codeBlocks.forEach((block, i) => {
                    console.log(`${i + 1}. ${block.language}:\n${block.code}\n`);
                });
            }
            
            return false; // Stop generation
        }
    }
});
```

### Model Management
```javascript
class ModelManager {
    constructor() {
        this.models = new Map();
        this.currentModel = null;
    }
    
    async loadModel(name, config) {
        console.log(`Loading model: ${name}`);
        
        const llama = new LLM(LLamaCpp);
        await llama.load(config);
        
        this.models.set(name, {
            instance: llama,
            config: config,
            loadTime: Date.now()
        });
        
        this.currentModel = name;
        console.log(`Model ${name} loaded successfully`);
        
        return llama;
    }
    
    switchModel(name) {
        if (!this.models.has(name)) {
            throw new Error(`Model ${name} not found`);
        }
        
        this.currentModel = name;
        return this.models.get(name).instance;
    }
    
    unloadModel(name) {
        if (this.models.has(name)) {
            // Note: llama-node doesn't provide explicit unload method
            // In practice, you might need to restart the process
            this.models.delete(name);
            
            if (this.currentModel === name) {
                this.currentModel = null;
            }
        }
    }
    
    listModels() {
        return Array.from(this.models.keys()).map(name => {
            const model = this.models.get(name);
            return {
                name,
                config: model.config,
                loadTime: new Date(model.loadTime).toISOString(),
                isCurrent: name === this.currentModel
            };
        });
    }
    
    getCurrentModel() {
        return this.currentModel ? this.models.get(this.currentModel).instance : null;
    }
}

// Usage
const modelManager = new ModelManager();

// Load multiple models
await modelManager.loadModel('chat-7b', {
    modelPath: './models/llama-2-7b-chat.q4_0.bin',
    nCtx: 2048
});

await modelManager.loadModel('code-7b', {
    modelPath: './models/codellama-7b.q4_0.bin',
    nCtx: 2048
});

// Switch between models
const chatModel = modelManager.switchModel('chat-7b');
const codeModel = modelManager.switchModel('code-7b');

// List all models
console.log(modelManager.listModels());
```

## API Reference

### LLM Class
```typescript
class LLM {
    constructor(backend: LLamaCpp);
    
    // Load model with configuration
    async load(config: ModelConfig): Promise<void>;
    
    // Create completion with streaming
    async createCompletion(
        params: CompletionParams, 
        callback: (data: CompletionData) => void
    ): Promise<void>;
    
    // Create embeddings (if model supports it)
    async createEmbedding(text: string): Promise<number[]>;
}
```

### Types
```typescript
interface ModelConfig {
    modelPath: string;
    enableLogging?: boolean;
    nCtx?: number;
    seed?: number;
    f16Kv?: boolean;
    logitsAll?: boolean;
    vocabOnly?: boolean;
    useMlock?: boolean;
    embedding?: boolean;
    useMmap?: boolean;
    nGpuLayers?: number;
    nBatch?: number;
    nThreads?: number;
}

interface CompletionParams {
    nThreads?: number;
    nTokPredict?: number;
    topK?: number;
    topP?: number;
    temp?: number;
    repeatPenalty?: number;
    prompt: string;
    stopSequence?: string[];
}

interface CompletionData {
    token?: string;
    completed?: boolean;
    timings?: {
        predicted_n: number;
        predicted_per_token_ms: number;
        predicted_per_second: number;
    };
}
```

## Performance Optimization

### Memory Management
```javascript
// Monitor memory usage
function monitorMemory() {
    const used = process.memoryUsage();
    console.log('Memory Usage:');
    for (let key in used) {
        console.log(`${key}: ${Math.round(used[key] / 1024 / 1024 * 100) / 100} MB`);
    }
}

// Optimize garbage collection
if (global.gc) {
    setInterval(() => {
        global.gc();
    }, 30000); // Run garbage collection every 30 seconds
}

// Model configuration for different memory constraints
function getOptimalConfig(availableMemoryGB) {
    if (availableMemoryGB >= 16) {
        return {
            modelPath: "./models/llama-2-13b-chat.q4_0.bin",
            nCtx: 2048,
            nBatch: 512,
            nThreads: 8
        };
    } else if (availableMemoryGB >= 8) {
        return {
            modelPath: "./models/llama-2-7b-chat.q4_0.bin", 
            nCtx: 1024,
            nBatch: 256,
            nThreads: 4
        };
    } else {
        return {
            modelPath: "./models/llama-2-7b-chat.q4_0.bin",
            nCtx: 512,
            nBatch: 128,
            nThreads: 2
        };
    }
}
```

### Batch Processing
```javascript
class BatchProcessor {
    constructor(llama, batchSize = 5) {
        this.llama = llama;
        this.batchSize = batchSize;
        this.queue = [];
        this.processing = false;
    }
    
    async addToQueue(prompt, params = {}) {
        return new Promise((resolve, reject) => {
            this.queue.push({
                prompt,
                params,
                resolve,
                reject
            });
            
            if (!this.processing) {
                this.processBatch();
            }
        });
    }
    
    async processBatch() {
        this.processing = true;
        
        while (this.queue.length > 0) {
            const batch = this.queue.splice(0, this.batchSize);
            
            for (const item of batch) {
                try {
                    let response = "";
                    
                    await this.llama.createCompletion({
                        ...item.params,
                        prompt: item.prompt,
                        nTokPredict: item.params.nTokPredict || 256,
                    }, (data) => {
                        if (data.token) {
                            response += data.token;
                        }
                        if (data.completed) {
                            item.resolve(response.trim());
                        }
                    });
                } catch (error) {
                    item.reject(error);
                }
            }
            
            // Small delay between batches
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        this.processing = false;
    }
}

// Usage
const batchProcessor = new BatchProcessor(llama, 3);

const prompts = [
    "Explain photosynthesis",
    "What is quantum computing?", 
    "How do neural networks work?"
];

const responses = await Promise.all(
    prompts.map(prompt => batchProcessor.addToQueue(prompt, { nTokPredict: 200 }))
);

responses.forEach((response, i) => {
    console.log(`Response ${i + 1}: ${response}\n`);
});
```

## Use Cases

### Code Assistant
```javascript
class CodeAssistant {
    constructor(llama) {
        this.llama = llama;
        this.languages = ['javascript', 'python', 'java', 'cpp', 'rust'];
    }
    
    async generateCode(task, language = 'javascript') {
        const prompt = `// Task: ${task}
// Language: ${language}
// Generate clean, well-commented code:

\`\`\`${language}`;

        let code = "";
        
        await this.llama.createCompletion({
            prompt,
            nTokPredict: 512,
            temp: 0.3, // Lower temperature for more consistent code
            topP: 0.95,
            stopSequence: ['```', '\n\n\n']
        }, (data) => {
            if (data.token) {
                code += data.token;
            }
        });
        
        return this.cleanCode(code);
    }
    
    async explainCode(code, language = 'javascript') {
        const prompt = `Explain this ${language} code step by step:

\`\`\`${language}
${code}
\`\`\`

Explanation:`;

        let explanation = "";
        
        await this.llama.createCompletion({
            prompt,
            nTokPredict: 400,
            temp: 0.5
        }, (data) => {
            if (data.token) {
                explanation += data.token;
            }
        });
        
        return explanation.trim();
    }
    
    async debugCode(code, language, error) {
        const prompt = `Debug this ${language} code that produces the error: "${error}"

Code:
\`\`\`${language}
${code}
\`\`\`

Solution:`;

        let solution = "";
        
        await this.llama.createCompletion({
            prompt,
            nTokPredict: 400,
            temp: 0.4
        }, (data) => {
            if (data.token) {
                solution += data.token;
            }
        });
        
        return solution.trim();
    }
    
    cleanCode(code) {
        return code
            .replace(/^\s*```[\w]*\s*/, '')
            .replace(/```\s*$/, '')
            .trim();
    }
}

// Usage
const codeAssistant = new CodeAssistant(llama);

// Generate code
const bubbleSortCode = await codeAssistant.generateCode(
    "Implement bubble sort algorithm",
    "python"
);
console.log("Generated code:", bubbleSortCode);

// Explain code
const explanation = await codeAssistant.explainCode(bubbleSortCode, "python");
console.log("Explanation:", explanation);
```

### Content Generator
```javascript
class ContentGenerator {
    constructor(llama) {
        this.llama = llama;
    }
    
    async generateBlogPost(topic, tone = 'professional', wordCount = 500) {
        const prompt = `Write a ${tone} blog post about "${topic}". 
Target length: approximately ${wordCount} words.

Title: `;

        let content = "";
        const targetTokens = Math.floor(wordCount * 1.3); // Rough token estimation
        
        await this.llama.createCompletion({
            prompt,
            nTokPredict: targetTokens,
            temp: 0.8,
            topP: 0.9
        }, (data) => {
            if (data.token) {
                content += data.token;
            }
        });
        
        return this.formatBlogPost(content);
    }
    
    async generateSocialMediaPost(topic, platform = 'twitter', hashtags = true) {
        const limits = {
            twitter: 280,
            linkedin: 1300,
            instagram: 2200
        };
        
        const charLimit = limits[platform] || 280;
        
        const prompt = `Create a ${platform} post about "${topic}".
Keep it under ${charLimit} characters.
${hashtags ? 'Include relevant hashtags.' : ''}

Post: `;

        let post = "";
        
        await this.llama.createCompletion({
            prompt,
            nTokPredict: Math.floor(charLimit / 3),
            temp: 0.9,
            topP: 0.9
        }, (data) => {
            if (data.token) {
                post += data.token;
            }
        });
        
        return post.trim();
    }
    
    formatBlogPost(content) {
        // Extract title and body
        const lines = content.split('\n');
        const title = lines[0].replace(/^title:\s*/i, '').trim();
        const body = lines.slice(1).join('\n').trim();
        
        return {
            title,
            body,
            wordCount: body.split(/\s+/).length,
            estimatedReadTime: Math.ceil(body.split(/\s+/).length / 200) // 200 words per minute
        };
    }
}

// Usage
const contentGen = new ContentGenerator(llama);

const blogPost = await contentGen.generateBlogPost(
    "The Future of Web Development",
    "informative",
    800
);

console.log("Title:", blogPost.title);
console.log("Word count:", blogPost.wordCount);
console.log("Read time:", blogPost.estimatedReadTime, "minutes");
console.log("\nContent:", blogPost.body);

const tweet = await contentGen.generateSocialMediaPost(
    "AI in web development",
    "twitter",
    true
);

console.log("\nTwitter post:", tweet);
```

### Error Handling and Logging
```javascript
class LlamaNodeWrapper {
    constructor() {
        this.llama = null;
        this.isLoaded = false;
        this.logger = console; // Use your preferred logger
    }
    
    async initialize(config) {
        try {
            this.llama = new LLM(LLamaCpp);
            
            this.logger.info('Loading model:', config.modelPath);
            await this.llama.load(config);
            
            this.isLoaded = true;
            this.logger.info('Model loaded successfully');
            
        } catch (error) {
            this.logger.error('Failed to load model:', error.message);
            throw new Error(`Model loading failed: ${error.message}`);
        }
    }
    
    async generate(prompt, options = {}) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded. Call initialize() first.');
        }
        
        const defaultOptions = {
            nTokPredict: 256,
            temp: 0.8,
            topP: 0.9,
            timeout: 30000 // 30 second timeout
        };
        
        const params = { ...defaultOptions, ...options, prompt };
        
        return new Promise((resolve, reject) => {
            let response = "";
            let timeoutId;
            
            // Set timeout
            if (params.timeout) {
                timeoutId = setTimeout(() => {
                    reject(new Error('Generation timeout'));
                }, params.timeout);
            }
            
            this.llama.createCompletion(params, (data) => {
                try {
                    if (data.token) {
                        response += data.token;
                    }
                    
                    if (data.completed) {
                        if (timeoutId) clearTimeout(timeoutId);
                        
                        this.logger.info('Generation completed', {
                            promptLength: prompt.length,
                            responseLength: response.length,
                            tokensGenerated: data.timings?.predicted_n || 0,
                            speed: data.timings?.predicted_per_second || 0
                        });
                        
                        resolve(response.trim());
                    }
                } catch (error) {
                    if (timeoutId) clearTimeout(timeoutId);
                    this.logger.error('Generation error:', error);
                    reject(error);
                }
            });
        });
    }
}

// Usage with error handling
const wrapper = new LlamaNodeWrapper();

try {
    await wrapper.initialize({
        modelPath: './models/llama-2-7b-chat.q4_0.bin',
        nCtx: 1024
    });
    
    const response = await wrapper.generate(
        "Explain the concept of recursion in programming", 
        { nTokPredict: 300, timeout: 60000 }
    );
    
    console.log('Response:', response);
    
} catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
}
```

LlamaNode provides a powerful way to run Llama 2 models directly in Node.js applications, enabling local AI capabilities without external dependencies.