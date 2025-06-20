---
name: llm_application_development
title: LLM Application Development Framework
description: Comprehensive framework for building production-ready LLM applications with RAG, fine-tuning, agents, and enterprise deployment patterns
category: ai-prompts
tags: [llm, rag, fine-tuning, agents, embeddings, vector-database, chatbot]
difficulty: advanced
author: jezweb
version: 1.0.0
arguments:
  - name: application_type
    description: Type of LLM application (chatbot, rag-system, agent, content-generation, code-assistant)
    required: true
  - name: llm_provider
    description: LLM provider (openai, anthropic, local-model, huggingface, azure-openai)
    required: true
  - name: knowledge_source
    description: Knowledge sources (documents, databases, apis, websites, internal-docs)
    required: true
  - name: deployment_scale
    description: Deployment scale (prototype, small-team, enterprise, high-volume)
    required: true
  - name: customization_level
    description: Customization level (prompt-engineering, rag, fine-tuning, full-training)
    required: true
  - name: security_requirements
    description: Security requirements (basic, enhanced, enterprise, government)
    required: false
    default: "basic"
---

# LLM Application Development: {{application_type}}

**LLM Provider:** {{llm_provider}}  
**Knowledge Sources:** {{knowledge_source}}  
**Scale:** {{deployment_scale}}  
**Customization:** {{customization_level}}  
**Security:** {{security_requirements}}

## 1. Architecture Overview

```mermaid
graph TB
    User[User Interface] --> Gateway[API Gateway]
    Gateway --> Auth[Authentication]
    Auth --> Router[Request Router]
    
    Router --> LLMService[LLM Service]
    Router --> RAGService[RAG Service]
    Router --> AgentService[Agent Service]
    
    RAGService --> VectorDB[(Vector Database)]
    RAGService --> Embeddings[Embedding Service]
    RAGService --> Retriever[Document Retriever]
    
    LLMService --> LLMProvider[{{llm_provider}}]
    AgentService --> Tools[Tool Registry]
    
    Embeddings --> EmbeddingModel[Embedding Model]
    Retriever --> DocumentStore[(Document Store)]
    
    LLMService --> Cache[(Cache Layer)]
    RAGService --> Cache
    
    subgraph Monitoring
        Metrics[Metrics Collection]
        Logging[Centralized Logging]
        Tracing[Request Tracing]
    end
    
    LLMService --> Metrics
    RAGService --> Metrics
    AgentService --> Metrics
```

### Core Components
```yaml
architecture_components:
  llm_layer:
    primary: "{{llm_provider}}"
    fallback: "Alternative provider for redundancy"
    caching: "Redis/Memcached for response caching"
    
  knowledge_layer:
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
    vector_database: "Pinecone, Weaviate, or Chroma"
    document_processing: "LangChain, LlamaIndex"
    
  application_layer:
    framework: "FastAPI, Streamlit, or LangServe"
    orchestration: "LangChain, CrewAI for agents"
    memory: "ConversationBufferMemory, VectorStoreRetrieverMemory"
    
  infrastructure:
    containerization: "Docker, Kubernetes"
    monitoring: "Prometheus, Grafana, LangSmith"
    scaling: "Horizontal pod autoscaling"
```

## 2. LLM Integration & Management

### Multi-Provider LLM Service
```python
# LLM service with multiple provider support
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator
import asyncio
from dataclasses import dataclass
import tiktoken

@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, messages: List[Dict], **kwargs) -> str:
        pass
    
    @abstractmethod
    async def stream_generate(self, messages: List[Dict], **kwargs) -> AsyncGenerator[str, None]:
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        import openai
        self.config = config
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.encoding = tiktoken.encoding_for_model(config.model)
    
    async def generate(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                timeout=self.config.timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            raise LLMError(f"OpenAI generation failed: {str(e)}")
    
    async def stream_generate(self, messages: List[Dict], **kwargs) -> AsyncGenerator[str, None]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(f"OpenAI streaming failed: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

class AnthropicProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        import anthropic
        self.config = config
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
    
    async def generate(self, messages: List[Dict], **kwargs) -> str:
        try:
            # Convert messages to Anthropic format
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                system=system_message,
                messages=user_messages
            )
            return response.content[0].text
        except Exception as e:
            raise LLMError(f"Anthropic generation failed: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        # Approximate token count for Anthropic models
        return len(text.split()) * 1.3

class LLMService:
    def __init__(self, configs: List[LLMConfig]):
        self.providers = {}
        self.primary_provider = None
        
        for config in configs:
            if config.provider == "openai":
                provider = OpenAIProvider(config)
            elif config.provider == "anthropic":
                provider = AnthropicProvider(config)
            # Add more providers as needed
            
            self.providers[config.provider] = provider
            
            if self.primary_provider is None:
                self.primary_provider = config.provider
    
    async def generate(self, messages: List[Dict], provider: Optional[str] = None, **kwargs) -> str:
        """Generate response with fallback support"""
        target_provider = provider or self.primary_provider
        
        try:
            return await self.providers[target_provider].generate(messages, **kwargs)
        except Exception as e:
            # Try fallback providers
            for fallback_provider in self.providers:
                if fallback_provider != target_provider:
                    try:
                        return await self.providers[fallback_provider].generate(messages, **kwargs)
                    except:
                        continue
            raise LLMError(f"All LLM providers failed: {str(e)}")
    
    async def stream_generate(self, messages: List[Dict], provider: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Stream generate with fallback support"""
        target_provider = provider or self.primary_provider
        
        try:
            async for chunk in self.providers[target_provider].stream_generate(messages, **kwargs):
                yield chunk
        except Exception as e:
            # Fallback to non-streaming generation
            response = await self.generate(messages, provider, **kwargs)
            yield response

class LLMError(Exception):
    pass
```

{{#if (eq application_type "rag-system")}}
## 3. RAG System Implementation

### Document Processing & Embedding Pipeline
```python
# RAG system with document processing and retrieval
from langchain.document_loaders import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone, Chroma
from typing import List, Dict, Any
import asyncio

class DocumentProcessor:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\\n\\n", "\\n", " ", ""]
        )
    
    async def process_documents(self, sources: List[str]) -> List[Dict]:
        """Process documents from various sources"""
        all_documents = []
        
        for source in sources:
            documents = await self.load_documents(source)
            chunks = self.text_splitter.split_documents(documents)
            all_documents.extend(chunks)
        
        return all_documents
    
    async def load_documents(self, source: str) -> List[Dict]:
        """Load documents from different sources"""
        {{#if (includes knowledge_source "documents")}}
        if source.endswith('.pdf'):
            loader = PyPDFLoader(source)
        elif source.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(source)
        elif source.endswith('.txt'):
            loader = TextLoader(source)
        {{/if}}
        
        {{#if (includes knowledge_source "websites")}}
        elif source.startswith('http'):
            loader = WebBaseLoader(source)
        {{/if}}
        
        {{#if (includes knowledge_source "databases")}}
        elif source.startswith('sql://'):
            # Database loader implementation
            loader = self.create_database_loader(source)
        {{/if}}
        
        return loader.load()
    
    def create_database_loader(self, connection_string: str):
        """Create database loader for structured data"""
        # Implementation depends on database type
        pass

class VectorStoreManager:
    def __init__(self, vector_store_type: str = "chroma"):
        self.vector_store_type = vector_store_type
        self.vector_store = None
    
    async def initialize_vector_store(self, documents: List[Dict], embeddings):
        """Initialize vector store with documents"""
        if self.vector_store_type == "pinecone":
            import pinecone
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT")
            )
            self.vector_store = Pinecone.from_documents(
                documents, embeddings, index_name="rag-index"
            )
        elif self.vector_store_type == "chroma":
            self.vector_store = Chroma.from_documents(
                documents, embeddings, persist_directory="./chroma_db"
            )
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform similarity search"""
        return self.vector_store.similarity_search(query, k=k)
    
    async def add_documents(self, documents: List[Dict]):
        """Add new documents to vector store"""
        self.vector_store.add_documents(documents)

class RAGRetriever:
    def __init__(self, vector_store_manager: VectorStoreManager, llm_service: LLMService):
        self.vector_store = vector_store_manager
        self.llm_service = llm_service
    
    async def retrieve_and_generate(self, query: str, context_window: int = 4000) -> Dict[str, Any]:
        """Retrieve relevant documents and generate response"""
        
        # Retrieve relevant documents
        relevant_docs = await self.vector_store.similarity_search(query, k=5)
        
        # Prepare context
        context = self.prepare_context(relevant_docs, context_window)
        
        # Generate response with RAG prompt
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant that answers questions based on the provided context.
                
Context:
{context}

Instructions:
- Answer the question based on the provided context
- If the answer is not in the context, say so clearly
- Cite specific parts of the context when possible
- Be concise but comprehensive"""
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        response = await self.llm_service.generate(messages)
        
        return {
            "response": response,
            "sources": [doc.metadata for doc in relevant_docs],
            "context_used": context[:500] + "..." if len(context) > 500 else context
        }
    
    def prepare_context(self, documents: List[Dict], max_tokens: int) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        current_length = 0
        
        for doc in documents:
            doc_text = doc.page_content
            doc_length = len(doc_text.split())
            
            if current_length + doc_length > max_tokens:
                break
            
            context_parts.append(f"Source: {doc.metadata.get('source', 'Unknown')}\\n{doc_text}")
            current_length += doc_length
        
        return "\\n\\n".join(context_parts)
```
{{/if}}

{{#if (eq application_type "agent")}}
## 4. Agent Framework Implementation

### Multi-Agent System
```python
# Agent framework for complex reasoning and tool usage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
from pydantic import BaseModel, Field
from typing import Type, List, Dict, Any, Optional
import asyncio

class AgentTool(BaseTool):
    """Base class for agent tools"""
    name: str
    description: str
    
    def _run(self, *args, **kwargs) -> str:
        return self._arun(*args, **kwargs)
    
    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError

class WebSearchTool(AgentTool):
    name = "web_search"
    description = "Search the web for current information"
    
    async def _arun(self, query: str) -> str:
        # Implement web search (using SerpAPI, Bing, etc.)
        import aiohttp
        
        # Example implementation with SerpAPI
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SERPAPI_KEY")
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get("https://serpapi.com/search", params=params) as response:
                data = await response.json()
                
                # Extract relevant results
                results = data.get("organic_results", [])[:3]
                
                search_results = []
                for result in results:
                    search_results.append(f"Title: {result.get('title')}\\nSnippet: {result.get('snippet')}\\nURL: {result.get('link')}")
                
                return "\\n\\n".join(search_results)

class CalculatorTool(AgentTool):
    name = "calculator"
    description = "Perform mathematical calculations"
    
    async def _arun(self, expression: str) -> str:
        try:
            # Safe evaluation of mathematical expressions
            import ast
            import operator
            
            # Define allowed operations
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            def eval_expr(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise TypeError(node)
            
            result = eval_expr(ast.parse(expression, mode='eval').body)
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"

class DatabaseQueryTool(AgentTool):
    name = "database_query"
    description = "Query the database for information"
    
    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string
    
    async def _arun(self, query: str) -> str:
        # Implement safe database querying
        import asyncpg
        
        # Validate query (only allow SELECT statements)
        if not query.strip().lower().startswith('select'):
            return "Error: Only SELECT queries are allowed"
        
        try:
            conn = await asyncpg.connect(self.connection_string)
            rows = await conn.fetch(query)
            await conn.close()
            
            # Format results
            if not rows:
                return "No results found"
            
            # Convert to readable format
            results = []
            for row in rows[:10]:  # Limit results
                results.append(str(dict(row)))
            
            return "\\n".join(results)
        except Exception as e:
            return f"Database query error: {str(e)}"

class ReasoningAgent:
    def __init__(self, llm_service: LLMService, tools: List[AgentTool]):
        self.llm_service = llm_service
        self.tools = {tool.name: tool for tool in tools}
        self.conversation_history = []
    
    async def reason_and_act(self, user_input: str, max_iterations: int = 5) -> str:
        """Implement ReAct (Reasoning and Acting) pattern"""
        
        self.conversation_history.append({"role": "user", "content": user_input})
        
        for iteration in range(max_iterations):
            # Reasoning step
            reasoning_prompt = self.create_reasoning_prompt(user_input)
            reasoning_response = await self.llm_service.generate(reasoning_prompt)
            
            # Parse response for actions
            action = self.parse_action(reasoning_response)
            
            if action is None:
                # Agent decided it has enough information to answer
                return reasoning_response
            
            # Acting step
            tool_result = await self.execute_tool(action)
            
            # Add tool result to context
            self.conversation_history.append({
                "role": "assistant",
                "content": f"Action: {action['tool']}({action['input']})\\nObservation: {tool_result}"
            })
            
            # Check if we have a final answer
            if "final answer" in reasoning_response.lower():
                break
        
        # Generate final response
        final_prompt = self.create_final_answer_prompt()
        return await self.llm_service.generate(final_prompt)
    
    def create_reasoning_prompt(self, user_input: str) -> List[Dict]:
        """Create prompt for reasoning step"""
        tool_descriptions = "\\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        system_prompt = f"""You are a helpful assistant that can use tools to answer questions.

Available tools:
{tool_descriptions}

Instructions:
1. Think step by step about how to answer the user's question
2. If you need information, use appropriate tools
3. Use the format: Action: tool_name(input) to call tools
4. When you have enough information, provide a final answer

Conversation history:
{self.format_conversation_history()}

Current question: {user_input}

Think through this step by step:"""

        return [{"role": "system", "content": system_prompt}]
    
    def parse_action(self, response: str) -> Optional[Dict]:
        """Parse action from LLM response"""
        import re
        
        # Look for Action: tool_name(input) pattern
        action_pattern = r"Action:\\s*(\\w+)\\(([^)]+)\\)"
        match = re.search(action_pattern, response)
        
        if match:
            tool_name = match.group(1)
            tool_input = match.group(2).strip('"').strip("'")
            
            if tool_name in self.tools:
                return {"tool": tool_name, "input": tool_input}
        
        return None
    
    async def execute_tool(self, action: Dict) -> str:
        """Execute tool and return result"""
        tool = self.tools[action["tool"]]
        return await tool._arun(action["input"])
    
    def format_conversation_history(self) -> str:
        """Format conversation history for prompt"""
        return "\\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-5:]  # Last 5 messages
        ])
    
    def create_final_answer_prompt(self) -> List[Dict]:
        """Create prompt for final answer"""
        return [
            {
                "role": "system",
                "content": f"""Based on the conversation and tool results, provide a comprehensive final answer to the user's question.

Conversation:
{self.format_conversation_history()}

Provide a clear, helpful final answer:"""
            }
        ]
```
{{/if}}

{{#if (includes customization_level "fine-tuning")}}
## 5. Model Fine-tuning Pipeline

### Fine-tuning Implementation
```python
# Fine-tuning pipeline for domain-specific models
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import wandb
from typing import List, Dict

class FineTuningPipeline:
    def __init__(self, base_model: str, output_dir: str):
        self.base_model = base_model
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_dataset(self, training_data: List[Dict]) -> Dataset:
        """Prepare dataset for fine-tuning"""
        
        def format_conversation(example):
            """Format conversation data for training"""
            conversation = ""
            for turn in example["conversation"]:
                if turn["role"] == "user":
                    conversation += f"Human: {turn['content']}\\n"
                elif turn["role"] == "assistant":
                    conversation += f"Assistant: {turn['content']}\\n"
            
            return {"text": conversation}
        
        # Format data
        formatted_data = [format_conversation(example) for example in training_data]
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def fine_tune_model(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Fine-tune the model"""
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            report_to="wandb" if wandb.run else None,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer

class LoRAFineTuning:
    """Parameter-efficient fine-tuning with LoRA"""
    
    def __init__(self, base_model: str, lora_config: Dict):
        from peft import LoraConfig, get_peft_model, TaskType
        
        self.base_model = base_model
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.1),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"])
        )
    
    def create_lora_model(self):
        """Create LoRA model"""
        from peft import get_peft_model
        
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model)
        model = get_peft_model(base_model, self.lora_config)
        
        return model
    
    def merge_and_save(self, model, output_path: str):
        """Merge LoRA weights and save model"""
        model = model.merge_and_unload()
        model.save_pretrained(output_path)
```
{{/if}}

## 6. Production Deployment

### Scalable Deployment Architecture
```python
# Production deployment with FastAPI and caching
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import json
import hashlib
from typing import Optional, Dict, Any
import asyncio
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI(
    title="{{application_type}} LLM Service",
    description="Production LLM application with {{customization_level}}",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
REQUEST_COUNT = Counter('llm_requests_total', 'Total LLM requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('llm_request_duration_seconds', 'Request latency')
TOKEN_USAGE = Counter('llm_tokens_used_total', 'Total tokens used', ['type'])

# Security
security = HTTPBearer()
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class LLMApplicationService:
    def __init__(self):
        # Initialize components based on application type
        {{#if (eq application_type "rag-system")}}
        self.rag_retriever = RAGRetriever(vector_store_manager, llm_service)
        {{/if}}
        {{#if (eq application_type "agent")}}
        self.reasoning_agent = ReasoningAgent(llm_service, tools)
        {{/if}}
        self.llm_service = LLMService(llm_configs)
        self.cache_ttl = 3600  # 1 hour cache
    
    async def generate_cache_key(self, input_data: Dict) -> str:
        """Generate cache key for request"""
        cache_string = json.dumps(input_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if available"""
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Cache retrieval error: {e}")
        return None
    
    async def cache_response(self, cache_key: str, response: Dict):
        """Cache response"""
        try:
            redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(response)
            )
        except Exception as e:
            print(f"Cache storage error: {e}")

llm_app = LLMApplicationService()

@app.post("/chat")
async def chat_endpoint(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Main chat endpoint"""
    start_time = time.time()
    
    try:
        # Validate request
        if "message" not in request:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Check cache
        cache_key = await llm_app.generate_cache_key(request)
        cached_response = await llm_app.get_cached_response(cache_key)
        
        if cached_response:
            REQUEST_COUNT.labels(endpoint="chat", status="cache_hit").inc()
            return cached_response
        
        # Process request based on application type
        {{#if (eq application_type "rag-system")}}
        response = await llm_app.rag_retriever.retrieve_and_generate(
            request["message"]
        )
        {{/if}}
        
        {{#if (eq application_type "agent")}}
        response_text = await llm_app.reasoning_agent.reason_and_act(
            request["message"]
        )
        response = {"response": response_text}
        {{/if}}
        
        {{#if (eq application_type "chatbot")}}
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request["message"]}
        ]
        response_text = await llm_app.llm_service.generate(messages)
        response = {"response": response_text}
        {{/if}}
        
        # Cache response
        await llm_app.cache_response(cache_key, response)
        
        # Record metrics
        REQUEST_COUNT.labels(endpoint="chat", status="success").inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="chat", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "application_type": "{{application_type}}",
        "llm_provider": "{{llm_provider}}",
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

# Kubernetes deployment
kubernetes_manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-app-{{application_type}}
spec:
  replicas: {{#if (eq deployment_scale "enterprise")}}5{{else if (eq deployment_scale "high-volume")}}10{{else}}3{{/if}}
  selector:
    matchLabels:
      app: llm-app
  template:
    metadata:
      labels:
        app: llm-app
    spec:
      containers:
      - name: llm-service
        image: llm-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLM_PROVIDER
          value: "{{llm_provider}}"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: llm-app-service
spec:
  selector:
    app: llm-app
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
```

## 7. Monitoring & Observability

### Comprehensive Monitoring
```python
# LLM application monitoring
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure structured logging
logger = structlog.get_logger()

class LLMMonitoring:
    def __init__(self):
        # Set up tracing
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        # Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="jaeger",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
    
    def log_llm_request(self, user_id: str, request: Dict, response: Dict, latency: float):
        """Log LLM request with structured data"""
        logger.info(
            "llm_request",
            user_id=user_id,
            request_length=len(request.get("message", "")),
            response_length=len(response.get("response", "")),
            latency_seconds=latency,
            application_type="{{application_type}}",
            llm_provider="{{llm_provider}}"
        )
    
    def log_error(self, error: Exception, context: Dict):
        """Log errors with context"""
        logger.error(
            "llm_error",
            error=str(error),
            error_type=type(error).__name__,
            context=context
        )
    
    async def track_token_usage(self, prompt_tokens: int, completion_tokens: int):
        """Track token usage"""
        TOKEN_USAGE.labels(type="prompt").inc(prompt_tokens)
        TOKEN_USAGE.labels(type="completion").inc(completion_tokens)
```

## Conclusion

This LLM application development framework provides:

**Key Features:**
- Multi-provider LLM integration with fallback support
- {{#if (eq application_type "rag-system")}}Advanced RAG system with vector search{{/if}}{{#if (eq application_type "agent")}}Intelligent agent framework with tool usage{{/if}}{{#if (eq application_type "chatbot")}}Conversational AI with context management{{/if}}
- {{customization_level}} implementation
- Production-ready deployment architecture
- Comprehensive monitoring and observability

**Scale & Performance:**
- Designed for {{deployment_scale}} deployment
- Intelligent caching and optimization
- Horizontal scaling support
- {{security_requirements}} security implementation

**Production Ready:**
- Container-based deployment
- Kubernetes orchestration
- Monitoring and alerting
- Error handling and fallback strategies
- {{knowledge_source}} integration capabilities