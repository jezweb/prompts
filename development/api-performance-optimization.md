---
name: api_performance_optimization
title: API Performance Optimization Guide
description: Comprehensive API performance optimization covering latency reduction, throughput improvement, caching strategies, and monitoring for high-performance APIs
category: development
tags: [api, performance, optimization, caching, monitoring, scalability]
difficulty: advanced
author: jezweb
version: 1.0.0
arguments:
  - name: api_architecture
    description: API architecture type (rest, graphql, grpc, websocket, microservices)
    required: true
  - name: backend_technology
    description: Backend technology (nodejs, python, java, go, rust, dotnet)
    required: true
  - name: expected_load
    description: Expected API load (low <1k-rps, medium 1k-10k-rps, high 10k-100k-rps, extreme >100k-rps)
    required: true
  - name: data_complexity
    description: Data complexity (simple-crud, complex-queries, real-time, analytics)
    required: true
  - name: performance_targets
    description: Performance targets (sub-100ms, sub-500ms, sub-1s, throughput-focused)
    required: true
  - name: infrastructure
    description: Infrastructure type (cloud-native, containerized, serverless, traditional)
    required: true
---

# API Performance Optimization: {{api_architecture}} API

**Backend Technology:** {{backend_technology}}  
**Expected Load:** {{expected_load}}  
**Data Complexity:** {{data_complexity}}  
**Performance Targets:** {{performance_targets}}  
**Infrastructure:** {{infrastructure}}

## 1. Performance Analysis & Baseline

### API Performance Profiling Framework
```javascript
{{#if (eq backend_technology "nodejs")}}
// Node.js API performance monitoring and profiling
const express = require('express');
const prometheus = require('prom-client');
const responseTime = require('response-time');
const cluster = require('cluster');
const os = require('os');

class APIPerformanceMonitor {
    constructor() {
        this.setupMetrics();
        this.setupMiddleware();
    }
    
    setupMetrics() {
        // Create custom metrics
        this.httpRequestDuration = new prometheus.Histogram({
            name: 'http_request_duration_seconds',
            help: 'Duration of HTTP requests in seconds',
            labelNames: ['method', 'route', 'status_code'],
            buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        });
        
        this.httpRequestTotal = new prometheus.Counter({
            name: 'http_requests_total',
            help: 'Total number of HTTP requests',
            labelNames: ['method', 'route', 'status_code']
        });
        
        this.activeConnections = new prometheus.Gauge({
            name: 'http_active_connections',
            help: 'Number of active HTTP connections'
        });
        
        this.databaseQueryDuration = new prometheus.Histogram({
            name: 'database_query_duration_seconds',
            help: 'Duration of database queries in seconds',
            labelNames: ['query_type', 'table'],
            buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
        });
        
        // Memory and CPU metrics
        this.memoryUsage = new prometheus.Gauge({
            name: 'nodejs_memory_usage_bytes',
            help: 'Memory usage in bytes',
            labelNames: ['type']
        });
        
        this.cpuUsage = new prometheus.Gauge({
            name: 'nodejs_cpu_usage_percent',
            help: 'CPU usage percentage'
        });
        
        // Start collecting default metrics
        prometheus.collectDefaultMetrics({ timeout: 5000 });
        
        // Collect custom metrics periodically
        setInterval(() => {
            this.collectMemoryMetrics();
            this.collectCpuMetrics();
        }, 5000);
    }
    
    setupMiddleware() {
        this.performanceMiddleware = (req, res, next) => {
            const startTime = Date.now();
            
            // Track active connections
            this.activeConnections.inc();
            
            res.on('finish', () => {
                const duration = (Date.now() - startTime) / 1000;
                const route = req.route ? req.route.path : req.path;
                
                // Record metrics
                this.httpRequestDuration
                    .labels(req.method, route, res.statusCode.toString())
                    .observe(duration);
                    
                this.httpRequestTotal
                    .labels(req.method, route, res.statusCode.toString())
                    .inc();
                
                this.activeConnections.dec();
                
                // Log slow requests
                if (duration > 1.0) { // Slower than 1 second
                    console.warn(`Slow request detected: ${req.method} ${route} - ${duration}s`);
                }
            });
            
            next();
        };
    }
    
    collectMemoryMetrics() {
        const memUsage = process.memoryUsage();
        this.memoryUsage.labels('rss').set(memUsage.rss);
        this.memoryUsage.labels('heapTotal').set(memUsage.heapTotal);
        this.memoryUsage.labels('heapUsed').set(memUsage.heapUsed);
        this.memoryUsage.labels('external').set(memUsage.external);
    }
    
    collectCpuMetrics() {
        const cpus = os.cpus();
        let totalIdle = 0;
        let totalTick = 0;
        
        cpus.forEach(cpu => {
            for (type in cpu.times) {
                totalTick += cpu.times[type];
            }
            totalIdle += cpu.times.idle;
        });
        
        const idle = totalIdle / cpus.length;
        const total = totalTick / cpus.length;
        const usage = 100 - ~~(100 * idle / total);
        
        this.cpuUsage.set(usage);
    }
    
    // Database query wrapper for monitoring
    async executeQuery(queryFunction, queryType, tableName) {
        const startTime = Date.now();
        
        try {
            const result = await queryFunction();
            const duration = (Date.now() - startTime) / 1000;
            
            this.databaseQueryDuration
                .labels(queryType, tableName)
                .observe(duration);
                
            return result;
        } catch (error) {
            const duration = (Date.now() - startTime) / 1000;
            this.databaseQueryDuration
                .labels(`${queryType}_error`, tableName)
                .observe(duration);
            throw error;
        }
    }
    
    // API endpoint for metrics
    getMetricsEndpoint() {
        return async (req, res) => {
            res.set('Content-Type', prometheus.register.contentType);
            res.end(await prometheus.register.metrics());
        };
    }
}

// Performance optimization utilities
class APIOptimizer {
    constructor() {
        this.cache = new Map();
        this.cacheStats = {
            hits: 0,
            misses: 0,
            sets: 0
        };
    }
    
    // Response caching middleware
    cacheMiddleware(duration = 300) {
        return (req, res, next) => {
            const key = `${req.method}:${req.originalUrl}`;
            const cached = this.cache.get(key);
            
            if (cached && Date.now() - cached.timestamp < duration * 1000) {
                this.cacheStats.hits++;
                return res.json(cached.data);
            }
            
            this.cacheStats.misses++;
            
            // Override res.json to cache the response
            const originalJson = res.json;
            res.json = (body) => {
                if (res.statusCode === 200) {
                    this.cache.set(key, {
                        data: body,
                        timestamp: Date.now()
                    });
                    this.cacheStats.sets++;
                }
                return originalJson.call(res, body);
            };
            
            next();
        };
    }
    
    // Request batching for database queries
    createBatchLoader(batchFunction, options = {}) {
        const { maxBatchSize = 100, batchWindow = 10 } = options;
        let batch = [];
        let batchTimeout = null;
        const resolvers = new Map();
        
        const processBatch = async () => {
            if (batch.length === 0) return;
            
            const currentBatch = batch.splice(0);
            const currentResolvers = new Map(resolvers);
            resolvers.clear();
            
            try {
                const results = await batchFunction(currentBatch.map(item => item.key));
                
                currentBatch.forEach((item, index) => {
                    const resolver = currentResolvers.get(item.id);
                    if (resolver) {
                        resolver.resolve(results[index]);
                    }
                });
            } catch (error) {
                currentResolvers.forEach(resolver => {
                    resolver.reject(error);
                });
            }
        };
        
        return (key) => {
            return new Promise((resolve, reject) => {
                const id = Math.random().toString(36);
                batch.push({ id, key });
                resolvers.set(id, { resolve, reject });
                
                if (batch.length >= maxBatchSize) {
                    if (batchTimeout) {
                        clearTimeout(batchTimeout);
                        batchTimeout = null;
                    }
                    processBatch();
                } else if (!batchTimeout) {
                    batchTimeout = setTimeout(() => {
                        batchTimeout = null;
                        processBatch();
                    }, batchWindow);
                }
            });
        };
    }
    
    // Connection pooling for database
    createConnectionPool(dbConfig) {
        const { Pool } = require('pg');
        
        return new Pool({
            ...dbConfig,
            max: {{#if (eq expected_load "extreme")}}50{{else if (eq expected_load "high")}}20{{else if (eq expected_load "medium")}}10{{else}}5{{/if}},
            idleTimeoutMillis: 30000,
            connectionTimeoutMillis: 2000,
        });
    }
}

// Usage example
const app = express();
const monitor = new APIPerformanceMonitor();
const optimizer = new APIOptimizer();

// Apply performance monitoring
app.use(monitor.performanceMiddleware);

// Apply caching for GET requests
app.use('/api', (req, res, next) => {
    if (req.method === 'GET') {
        return optimizer.cacheMiddleware(300)(req, res, next);
    }
    next();
});

// Metrics endpoint
app.get('/metrics', monitor.getMetricsEndpoint());

// Example optimized API endpoint
app.get('/api/users/:id', async (req, res) => {
    try {
        const userId = req.params.id;
        
        // Use monitored database query
        const user = await monitor.executeQuery(
            () => getUserFromDatabase(userId),
            'select',
            'users'
        );
        
        res.json(user);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

{{#if (eq infrastructure "containerized")}}
// Clustering for multi-core utilization
if (cluster.isMaster) {
    const numCPUs = os.cpus().length;
    console.log(`Master ${process.pid} is running`);
    
    // Fork workers
    for (let i = 0; i < numCPUs; i++) {
        cluster.fork();
    }
    
    cluster.on('exit', (worker, code, signal) => {
        console.log(`Worker ${worker.process.pid} died`);
        cluster.fork();
    });
} else {
    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
        console.log(`Worker ${process.pid} started on port ${PORT}`);
    });
}
{{else}}
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});
{{/if}}
{{/if}}

{{#if (eq backend_technology "python")}}
# Python FastAPI performance optimization
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import aioredis
import aiocache
import time
import psutil
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import uvloop
import contextvars

# Set up async event loop optimization
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('http_active_connections', 'Active HTTP connections')
DATABASE_QUERY_LATENCY = Histogram('database_query_duration_seconds', 'Database query latency', ['query_type'])

class PerformanceMiddleware:
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        ACTIVE_CONNECTIONS.inc()
        
        # Get request info
        request = Request(scope, receive)
        method = request.method
        path = request.url.path
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                process_time = time.time() - start_time
                status_code = message["status"]
                
                # Record metrics
                REQUEST_COUNT.labels(method=method, endpoint=path, status=status_code).inc()
                REQUEST_LATENCY.labels(method=method, endpoint=path).observe(process_time)
                ACTIVE_CONNECTIONS.dec()
                
                # Add performance headers
                message["headers"] = list(message.get("headers", []))
                message["headers"].append((b"x-process-time", str(process_time).encode()))
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

class APIOptimizer:
    def __init__(self):
        self.cache = aiocache.Cache(aiocache.SimpleMemoryCache)
        self.redis_client = None
        
    async def setup_redis(self, redis_url: str = "redis://localhost:6379"):
        """Setup Redis connection for distributed caching"""
        try:
            self.redis_client = await aioredis.from_url(redis_url)
            self.cache = aiocache.Cache(
                aiocache.RedisCache,
                endpoint="localhost",
                port=6379,
                serializer=aiocache.serializers.JsonSerializer(),
                plugins=[aiocache.plugins.HitMissRatioPlugin()]
            )
        except Exception as e:
            logging.warning(f"Redis connection failed: {e}. Using memory cache.")
    
    async def cached_query(self, key: str, query_func, ttl: int = 300):
        """Execute query with caching"""
        # Try to get from cache first
        cached_result = await self.cache.get(key)
        if cached_result is not None:
            return cached_result
        
        # Execute query with timing
        start_time = time.time()
        result = await query_func()
        query_time = time.time() - start_time
        
        # Record database query metrics
        DATABASE_QUERY_LATENCY.labels(query_type="select").observe(query_time)
        
        # Cache the result
        await self.cache.set(key, result, ttl=ttl)
        
        return result
    
    def batch_requests(self, batch_size: int = 100, max_wait_time: float = 0.01):
        """Decorator for batching requests"""
        batch_queue = []
        batch_futures = []
        batch_lock = asyncio.Lock()
        
        async def process_batch():
            if not batch_queue:
                return
            
            current_batch = batch_queue[:]
            current_futures = batch_futures[:]
            batch_queue.clear()
            batch_futures.clear()
            
            try:
                # Process batch - implement your batch logic here
                results = await self.execute_batch(current_batch)
                
                for future, result in zip(current_futures, results):
                    future.set_result(result)
            except Exception as e:
                for future in current_futures:
                    future.set_exception(e)
        
        async def batched_function(item):
            async with batch_lock:
                future = asyncio.Future()
                batch_queue.append(item)
                batch_futures.append(future)
                
                if len(batch_queue) >= batch_size:
                    await process_batch()
                else:
                    # Schedule batch processing
                    asyncio.create_task(
                        asyncio.sleep(max_wait_time).then(lambda: process_batch())
                    )
                
                return await future
        
        return batched_function

# FastAPI app setup
app = FastAPI(title="High Performance API")

# Add performance middleware
app.add_middleware(PerformanceMiddleware)

# CORS middleware with optimized settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize optimizer
optimizer = APIOptimizer()

@app.on_event("startup")
async def startup_event():
    await optimizer.setup_redis()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent()
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Example optimized endpoint
@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    cache_key = f"user:{user_id}"
    
    async def fetch_user():
        # Simulate database query
        await asyncio.sleep(0.1)  # Replace with actual database call
        return {"id": user_id, "name": f"User {user_id}", "timestamp": time.time()}
    
    user = await optimizer.cached_query(cache_key, fetch_user, ttl=600)
    return user

{{#if (eq infrastructure "serverless")}}
# Serverless optimizations
@app.middleware("http")
async def add_serverless_optimizations(request: Request, call_next):
    # Warm-up logic for serverless
    if request.headers.get("x-warmup"):
        return {"status": "warm"}
    
    response = await call_next(request)
    return response
{{/if}}
{{/if}}

{{#if (eq backend_technology "go")}}
// Go API performance optimization
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"
    
    "github.com/gin-gonic/gin"
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
    "github.com/go-redis/redis/v8"
    "github.com/patrickmn/go-cache"
)

// Metrics
var (
    httpRequestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "endpoint", "status"},
    )
    
    httpRequestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "http_request_duration_seconds",
            Help: "Duration of HTTP requests",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "endpoint"},
    )
    
    activeConnections = prometheus.NewGauge(prometheus.GaugeOpts{
        Name: "http_active_connections",
        Help: "Number of active HTTP connections",
    })
)

func init() {
    prometheus.MustRegister(httpRequestsTotal)
    prometheus.MustRegister(httpRequestDuration)
    prometheus.MustRegister(activeConnections)
}

// Performance monitor middleware
func performanceMiddleware() gin.HandlerFunc {
    return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
        httpRequestsTotal.WithLabelValues(
            param.Method,
            param.Path,
            fmt.Sprintf("%d", param.StatusCode),
        ).Inc()
        
        httpRequestDuration.WithLabelValues(
            param.Method,
            param.Path,
        ).Observe(param.Latency.Seconds())
        
        return fmt.Sprintf("%s - [%s] \"%s %s %s\" %d %s \"%s\" \"%s\" %s\n",
            param.ClientIP,
            param.TimeStamp.Format(time.RFC1123),
            param.Method,
            param.Path,
            param.Request.Proto,
            param.StatusCode,
            param.Latency,
            param.Request.UserAgent(),
            param.ErrorMessage,
            param.BodySize,
        )
    })
}

// Connection tracking middleware
func connectionTrackingMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        activeConnections.Inc()
        defer activeConnections.Dec()
        c.Next()
    }
}

// Cache implementation
type APICache struct {
    local *cache.Cache
    redis *redis.Client
    mutex sync.RWMutex
}

func NewAPICache() *APICache {
    rdb := redis.NewClient(&redis.Options{
        Addr:     "localhost:6379",
        Password: "",
        DB:       0,
        {{#if (eq expected_load "extreme")}}
        PoolSize: 100,
        {{else if (eq expected_load "high")}}
        PoolSize: 50,
        {{else}}
        PoolSize: 10,
        {{/if}}
    })
    
    return &APICache{
        local: cache.New(5*time.Minute, 10*time.Minute),
        redis: rdb,
    }
}

func (ac *APICache) Get(key string) (interface{}, bool) {
    // Try local cache first
    if value, found := ac.local.Get(key); found {
        return value, true
    }
    
    // Try Redis cache
    ctx := context.Background()
    val, err := ac.redis.Get(ctx, key).Result()
    if err == nil {
        var result interface{}
        if err := json.Unmarshal([]byte(val), &result); err == nil {
            // Store in local cache
            ac.local.Set(key, result, cache.DefaultExpiration)
            return result, true
        }
    }
    
    return nil, false
}

func (ac *APICache) Set(key string, value interface{}, duration time.Duration) {
    // Set in local cache
    ac.local.Set(key, value, duration)
    
    // Set in Redis cache
    ctx := context.Background()
    if data, err := json.Marshal(value); err == nil {
        ac.redis.Set(ctx, key, data, duration)
    }
}

// Request batcher
type RequestBatcher struct {
    batchSize   int
    maxWaitTime time.Duration
    batches     map[string]*batch
    mutex       sync.Mutex
}

type batch struct {
    items   []interface{}
    results chan []interface{}
    timer   *time.Timer
}

func NewRequestBatcher(batchSize int, maxWaitTime time.Duration) *RequestBatcher {
    return &RequestBatcher{
        batchSize:   batchSize,
        maxWaitTime: maxWaitTime,
        batches:     make(map[string]*batch),
    }
}

func (rb *RequestBatcher) AddRequest(batchKey string, item interface{}) <-chan []interface{} {
    rb.mutex.Lock()
    defer rb.mutex.Unlock()
    
    b, exists := rb.batches[batchKey]
    if !exists {
        b = &batch{
            items:   make([]interface{}, 0, rb.batchSize),
            results: make(chan []interface{}, 1),
        }
        rb.batches[batchKey] = b
        
        b.timer = time.AfterFunc(rb.maxWaitTime, func() {
            rb.processBatch(batchKey)
        })
    }
    
    b.items = append(b.items, item)
    
    if len(b.items) >= rb.batchSize {
        b.timer.Stop()
        go rb.processBatch(batchKey)
    }
    
    return b.results
}

func (rb *RequestBatcher) processBatch(batchKey string) {
    rb.mutex.Lock()
    b := rb.batches[batchKey]
    delete(rb.batches, batchKey)
    rb.mutex.Unlock()
    
    if b == nil || len(b.items) == 0 {
        return
    }
    
    // Process batch - implement your batch processing logic here
    results := rb.executeBatch(b.items)
    
    b.results <- results
    close(b.results)
}

func (rb *RequestBatcher) executeBatch(items []interface{}) []interface{} {
    // Implement your batch processing logic
    results := make([]interface{}, len(items))
    for i, item := range items {
        // Process each item
        results[i] = fmt.Sprintf("processed_%v", item)
    }
    return results
}

func main() {
    // Initialize cache and batcher
    apiCache := NewAPICache()
    batcher := NewRequestBatcher({{#if (eq expected_load "extreme")}}1000{{else if (eq expected_load "high")}}500{{else}}100{{/if}}, 10*time.Millisecond)
    
    {{#if (eq infrastructure "containerized")}}
    // Set Gin to release mode for production
    gin.SetMode(gin.ReleaseMode)
    {{/if}}
    
    r := gin.New()
    
    // Add middleware
    r.Use(performanceMiddleware())
    r.Use(connectionTrackingMiddleware())
    r.Use(gin.Recovery())
    
    // Metrics endpoint
    r.GET("/metrics", gin.WrapH(promhttp.Handler()))
    
    // Health check
    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status":    "healthy",
            "timestamp": time.Now().Unix(),
        })
    })
    
    // Cached endpoint example
    r.GET("/api/users/:id", func(c *gin.Context) {
        userID := c.Param("id")
        cacheKey := fmt.Sprintf("user:%s", userID)
        
        // Try cache first
        if cachedUser, found := apiCache.Get(cacheKey); found {
            c.JSON(http.StatusOK, cachedUser)
            return
        }
        
        // Simulate database query
        user := map[string]interface{}{
            "id":        userID,
            "name":      fmt.Sprintf("User %s", userID),
            "timestamp": time.Now().Unix(),
        }
        
        // Cache the result
        apiCache.Set(cacheKey, user, 10*time.Minute)
        
        c.JSON(http.StatusOK, user)
    })
    
    // Batched endpoint example
    r.GET("/api/batch/:id", func(c *gin.Context) {
        itemID := c.Param("id")
        
        resultChan := batcher.AddRequest("batch_key", itemID)
        
        select {
        case results := <-resultChan:
            c.JSON(http.StatusOK, gin.H{"results": results})
        case <-time.After(5 * time.Second):
            c.JSON(http.StatusRequestTimeout, gin.H{"error": "timeout"})
        }
    })
    
    log.Println("Starting server on :8080")
    log.Fatal(http.ListenAndServe(":8080", r))
}
{{/if}}
```

## 2. Caching Strategies

### Multi-Level Caching Implementation
```yaml
# Comprehensive caching strategy
caching_layers:
  browser_cache:
    description: "Client-side caching with HTTP headers"
    ttl: "public, max-age=3600"
    suitable_for: ["static assets", "immutable data"]
    
  cdn_cache:
    description: "Content Delivery Network caching"
    providers: ["CloudFlare", "AWS CloudFront", "Azure CDN"]
    ttl: "3600-86400 seconds"
    suitable_for: ["API responses", "images", "static content"]
    
  application_cache:
    description: "In-memory application caching"
    technologies: ["Redis", "Memcached", "In-memory Map"]
    ttl: "300-3600 seconds"
    suitable_for: ["frequently accessed data", "computed results"]
    
  database_cache:
    description: "Database query result caching"
    techniques: ["Query result cache", "Connection pooling"]
    suitable_for: ["expensive queries", "reference data"]

cache_invalidation_strategies:
  time_based:
    description: "TTL-based cache expiration"
    implementation: "Set expiration time on cache entries"
    pros: ["Simple to implement", "Predictable memory usage"]
    cons: ["May serve stale data", "Cache misses on expiration"]
    
  event_based:
    description: "Invalidate cache on data changes"
    implementation: "Publish/subscribe for cache invalidation"
    pros: ["Fresh data guaranteed", "Efficient cache usage"]
    cons: ["Complex implementation", "Potential race conditions"]
    
  hybrid:
    description: "Combine TTL with event-based invalidation"
    implementation: "Short TTL with immediate invalidation on updates"
    pros: ["Balance between freshness and complexity"]
    cons: ["Requires careful tuning"]
```

### Advanced Caching Implementation
```python
# Advanced caching system with multiple layers
import asyncio
import aioredis
import json
import hashlib
import time
from typing import Any, Optional, Callable, Dict
from functools import wraps
import logging

class MultiLevelCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.l1_cache = {}  # In-memory cache
        self.l1_timestamps = {}  # Track L1 cache timestamps
        self.l1_max_size = {{#if (eq expected_load "extreme")}}10000{{else if (eq expected_load "high")}}5000{{else}}1000{{/if}}
        self.l1_ttl = 60  # 1 minute for L1 cache
        
        self.redis_client = None
        self.redis_url = redis_url
        
        self.cache_stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
    
    async def setup_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                max_connections={{#if (eq expected_load "extreme")}}100{{else if (eq expected_load "high")}}50{{else}}20{{/if}},
                retry_on_timeout=True
            )
            await self.redis_client.ping()
            logging.info("Redis connection established")
        except Exception as e:
            logging.error(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key"""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cleanup_l1_cache(self):
        """Remove expired entries from L1 cache"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.l1_timestamps.items()
            if current_time - timestamp > self.l1_ttl
        ]
        
        for key in expired_keys:
            self.l1_cache.pop(key, None)
            self.l1_timestamps.pop(key, None)
            self.cache_stats['evictions'] += 1
    
    def _evict_l1_cache(self):
        """Evict oldest entries if L1 cache is full"""
        if len(self.l1_cache) >= self.l1_max_size:
            # Remove oldest 10% of entries
            num_to_remove = max(1, self.l1_max_size // 10)
            oldest_keys = sorted(
                self.l1_timestamps.keys(),
                key=lambda k: self.l1_timestamps[k]
            )[:num_to_remove]
            
            for key in oldest_keys:
                self.l1_cache.pop(key, None)
                self.l1_timestamps.pop(key, None)
                self.cache_stats['evictions'] += 1
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        current_time = time.time()
        
        # Clean up expired L1 entries
        self._cleanup_l1_cache()
        
        # Try L1 cache first
        if key in self.l1_cache:
            if current_time - self.l1_timestamps[key] <= self.l1_ttl:
                self.cache_stats['l1_hits'] += 1
                return self.l1_cache[key]
            else:
                # Expired, remove from L1
                self.l1_cache.pop(key, None)
                self.l1_timestamps.pop(key, None)
        
        # Try L2 cache (Redis)
        if self.redis_client:
            try:
                redis_value = await self.redis_client.get(key)
                if redis_value:
                    value = json.loads(redis_value)
                    
                    # Store in L1 cache
                    self._evict_l1_cache()
                    self.l1_cache[key] = value
                    self.l1_timestamps[key] = current_time
                    
                    self.cache_stats['l2_hits'] += 1
                    return value
            except Exception as e:
                logging.error(f"Redis get error: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in multi-level cache"""
        current_time = time.time()
        
        # Set in L1 cache
        self._evict_l1_cache()
        self.l1_cache[key] = value
        self.l1_timestamps[key] = current_time
        
        # Set in L2 cache (Redis)
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    key, 
                    ttl, 
                    json.dumps(value, default=str)
                )
            except Exception as e:
                logging.error(f"Redis set error: {e}")
        
        self.cache_stats['sets'] += 1
    
    async def delete(self, key: str):
        """Delete from all cache levels"""
        # Remove from L1
        self.l1_cache.pop(key, None)
        self.l1_timestamps.pop(key, None)
        
        # Remove from L2
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logging.error(f"Redis delete error: {e}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        # Invalidate L1 cache
        matching_keys = [key for key in self.l1_cache.keys() if pattern in key]
        for key in matching_keys:
            self.l1_cache.pop(key, None)
            self.l1_timestamps.pop(key, None)
        
        # Invalidate L2 cache
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(f"*{pattern}*")
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logging.error(f"Redis pattern invalidation error: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = sum([
            self.cache_stats['l1_hits'],
            self.cache_stats['l2_hits'],
            self.cache_stats['misses']
        ])
        
        if total_requests > 0:
            l1_hit_rate = self.cache_stats['l1_hits'] / total_requests
            l2_hit_rate = self.cache_stats['l2_hits'] / total_requests
            overall_hit_rate = (self.cache_stats['l1_hits'] + self.cache_stats['l2_hits']) / total_requests
        else:
            l1_hit_rate = l2_hit_rate = overall_hit_rate = 0
        
        return {
            **self.cache_stats,
            'l1_hit_rate': l1_hit_rate,
            'l2_hit_rate': l2_hit_rate,
            'overall_hit_rate': overall_hit_rate,
            'l1_cache_size': len(self.l1_cache)
        }

# Decorator for caching function results
def cached(ttl: int = 3600, key_prefix: str = "cache"):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = getattr(wrapper, '_cache', None)
            if not cache:
                cache = MultiLevelCache()
                await cache.setup_redis()
                wrapper._cache = cache
            
            # Generate cache key
            cache_key = cache._generate_key(
                f"{key_prefix}:{func.__name__}",
                *args,
                **kwargs
            )
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# Usage example
@cached(ttl=600, key_prefix="user_data")
async def get_user_profile(user_id: int):
    # Simulate expensive database operation
    await asyncio.sleep(0.1)
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "profile_data": "expensive_to_compute",
        "timestamp": time.time()
    }
```

## 3. Database Optimization

### Query Optimization Strategies
```sql
{{#if (eq data_complexity "complex-queries")}}
-- Advanced query optimization techniques
-- 1. Index optimization for complex queries
CREATE INDEX CONCURRENTLY idx_users_active_recent 
ON users (status, last_login_at DESC) 
WHERE status = 'active' AND last_login_at > NOW() - INTERVAL '30 days';

-- 2. Partial indexes for selective filtering
CREATE INDEX CONCURRENTLY idx_orders_pending_priority 
ON orders (priority, created_at) 
WHERE status = 'pending';

-- 3. Composite indexes for multi-column queries
CREATE INDEX CONCURRENTLY idx_analytics_time_series 
ON analytics_events (user_id, event_type, created_at);

-- 4. Covering indexes to avoid table lookups
CREATE INDEX CONCURRENTLY idx_products_catalog_covering 
ON products (category_id, status) 
INCLUDE (name, price, description);

-- 5. Query optimization with CTEs and window functions
WITH ranked_products AS (
    SELECT 
        p.*,
        ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY sales_count DESC) as rank,
        LAG(price) OVER (PARTITION BY category_id ORDER BY created_at) as prev_price
    FROM products p
    WHERE status = 'active'
),
category_stats AS (
    SELECT 
        category_id,
        COUNT(*) as product_count,
        AVG(price) as avg_price,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price
    FROM ranked_products
    GROUP BY category_id
)
SELECT 
    rp.*,
    cs.avg_price as category_avg_price,
    cs.median_price as category_median_price,
    CASE 
        WHEN rp.price > cs.avg_price * 1.2 THEN 'premium'
        WHEN rp.price < cs.avg_price * 0.8 THEN 'budget'
        ELSE 'standard'
    END as price_tier
FROM ranked_products rp
JOIN category_stats cs ON rp.category_id = cs.category_id
WHERE rp.rank <= 10;

-- 6. Optimized pagination with cursor-based approach
-- Instead of OFFSET (slow for large offsets)
SELECT id, name, created_at 
FROM products 
WHERE created_at < $1  -- cursor from previous page
  AND status = 'active'
ORDER BY created_at DESC 
LIMIT 20;

-- 7. Batch operations for bulk updates
WITH batch_updates AS (
    SELECT 
        unnest($1::int[]) as id,
        unnest($2::text[]) as new_status
)
UPDATE products 
SET 
    status = bu.new_status,
    updated_at = NOW()
FROM batch_updates bu
WHERE products.id = bu.id;
{{/if}}

{{#if (eq data_complexity "real-time")}}
-- Real-time data optimization
-- 1. Materialized views for real-time analytics
CREATE MATERIALIZED VIEW mv_real_time_metrics AS
SELECT 
    DATE_TRUNC('minute', created_at) as time_bucket,
    event_type,
    COUNT(*) as event_count,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(CASE WHEN event_type = 'page_view' THEN duration_ms END) as avg_page_duration
FROM events 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('minute', created_at), event_type;

-- Refresh materialized view efficiently
CREATE OR REPLACE FUNCTION refresh_real_time_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- Only refresh if significant changes
    IF TG_OP = 'INSERT' AND NEW.created_at >= NOW() - INTERVAL '1 hour' THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_real_time_metrics;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- 2. Trigger for incremental updates
CREATE TRIGGER trigger_refresh_metrics
    AFTER INSERT ON events
    FOR EACH STATEMENT
    EXECUTE FUNCTION refresh_real_time_metrics();

-- 3. Time-series optimized table structure
CREATE TABLE metrics_time_series (
    time TIMESTAMPTZ NOT NULL,
    metric_name TEXT NOT NULL,
    tags JSONB,
    value DOUBLE PRECISION,
    CONSTRAINT metrics_time_series_pkey PRIMARY KEY (time, metric_name)
) PARTITION BY RANGE (time);

-- Create partitions for time-series data
CREATE TABLE metrics_time_series_y2024m01 PARTITION OF metrics_time_series
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- 4. Hypertable setup (if using TimescaleDB)
SELECT create_hypertable('metrics_time_series', 'time', chunk_time_interval => INTERVAL '1 day');
{{/if}}

-- Connection pooling configuration
connection_pool_settings:
  postgresql:
    max_connections: {{#if (eq expected_load "extreme")}}200{{else if (eq expected_load "high")}}100{{else if (eq expected_load "medium")}}50{{else}}20{{/if}}
    shared_buffers: "{{#if (eq expected_load "extreme")}}4GB{{else if (eq expected_load "high")}}2GB{{else if (eq expected_load "medium")}}1GB{{else}}256MB{{/if}}"
    effective_cache_size: "{{#if (eq expected_load "extreme")}}12GB{{else if (eq expected_load "high")}}6GB{{else if (eq expected_load "medium")}}3GB{{else}}1GB{{/if}}"
    work_mem: "{{#if (eq expected_load "extreme")}}32MB{{else if (eq expected_load "high")}}16MB{{else if (eq expected_load "medium")}}8MB{{else}}4MB{{/if}}"
    maintenance_work_mem: "{{#if (eq expected_load "extreme")}}512MB{{else if (eq expected_load "high")}}256MB{{else if (eq expected_load "medium")}}128MB{{else}}64MB{{/if}}"
    
  application_pool:
    initial_size: {{#if (eq expected_load "extreme")}}20{{else if (eq expected_load "high")}}10{{else if (eq expected_load "medium")}}5{{else}}2{{/if}}
    max_size: {{#if (eq expected_load "extreme")}}100{{else if (eq expected_load "high")}}50{{else if (eq expected_load "medium")}}25{{else}}10{{/if}}
    min_idle: {{#if (eq expected_load "extreme")}}10{{else if (eq expected_load "high")}}5{{else if (eq expected_load "medium")}}2{{else}}1{{/if}}
    max_idle_time: "10 minutes"
    validation_query: "SELECT 1"
    test_on_borrow: true
```

## 4. Load Testing & Performance Validation

### Comprehensive Load Testing Framework
```javascript
// Load testing with K6
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('error_rate');
const apiResponseTime = new Trend('api_response_time');
const throughputCounter = new Counter('throughput');

// Test configuration
export const options = {
  stages: [
    // Ramp up
    { duration: '2m', target: {{#if (eq expected_load "extreme")}}10000{{else if (eq expected_load "high")}}5000{{else if (eq expected_load "medium")}}1000{{else}}100{{/if}} },
    
    // Stay at peak
    { duration: '{{#if (eq performance_targets "sub-100ms")}}10m{{else if (eq performance_targets "sub-500ms")}}5m{{else}}3m{{/if}}', target: {{#if (eq expected_load "extreme")}}10000{{else if (eq expected_load "high")}}5000{{else if (eq expected_load "medium")}}1000{{else}}100{{/if}} },
    
    // Ramp down
    { duration: '2m', target: 0 },
  ],
  
  thresholds: {
    'http_req_duration': [
      {{#if (eq performance_targets "sub-100ms")}}
      'p95<100',  // 95% of requests under 100ms
      'p99<200',  // 99% of requests under 200ms
      {{else if (eq performance_targets "sub-500ms")}}
      'p95<500',  // 95% of requests under 500ms
      'p99<1000', // 99% of requests under 1s
      {{else if (eq performance_targets "sub-1s")}}
      'p95<1000', // 95% of requests under 1s
      'p99<2000', // 99% of requests under 2s
      {{else}}
      'p95<2000', // Throughput focused - allow higher latency
      {{/if}}
    ],
    'http_req_failed': ['rate<0.01'], // Error rate under 1%
    'error_rate': ['rate<0.01'],
  },
};

// Test scenarios
const scenarios = {
  // Read-heavy workload (typical API usage)
  read_heavy: {
    weight: 70,
    endpoints: ['/api/users', '/api/products', '/api/categories'],
    method: 'GET'
  },
  
  // Write operations
  write_operations: {
    weight: 20,
    endpoints: ['/api/users', '/api/products'],
    method: 'POST'
  },
  
  // Complex queries
  complex_queries: {
    weight: 10,
    endpoints: ['/api/analytics', '/api/reports'],
    method: 'GET'
  }
};

// Generate test data
function generateTestData() {
  return {
    name: `User ${Math.random().toString(36).substr(2, 9)}`,
    email: `test${Math.random().toString(36).substr(2, 9)}@example.com`,
    age: Math.floor(Math.random() * 50) + 18,
    preferences: {
      theme: Math.random() > 0.5 ? 'dark' : 'light',
      notifications: Math.random() > 0.3
    }
  };
}

// Main test function
export default function () {
  const baseUrl = __ENV.API_BASE_URL || 'http://localhost:3000';
  
  // Select scenario based on weight
  const rand = Math.random();
  let scenario;
  
  if (rand < 0.7) {
    scenario = scenarios.read_heavy;
  } else if (rand < 0.9) {
    scenario = scenarios.write_operations;
  } else {
    scenario = scenarios.complex_queries;
  }
  
  // Select random endpoint from scenario
  const endpoint = scenario.endpoints[Math.floor(Math.random() * scenario.endpoints.length)];
  const url = `${baseUrl}${endpoint}`;
  
  let response;
  const startTime = Date.now();
  
  if (scenario.method === 'GET') {
    // Add query parameters for complex queries
    const params = endpoint.includes('analytics') ? 
      '?start_date=2024-01-01&end_date=2024-01-31&metrics=page_views,users' : '';
    
    response = http.get(url + params, {
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'K6-LoadTest/1.0'
      },
      timeout: '{{#if (eq performance_targets "sub-100ms")}}5s{{else if (eq performance_targets "sub-500ms")}}10s{{else}}30s{{/if}}'
    });
  } else {
    response = http.post(url, JSON.stringify(generateTestData()), {
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'K6-LoadTest/1.0'
      },
      timeout: '{{#if (eq performance_targets "sub-100ms")}}5s{{else if (eq performance_targets "sub-500ms")}}10s{{else}}30s{{/if}}'
    });
  }
  
  const responseTime = Date.now() - startTime;
  
  // Record metrics
  apiResponseTime.add(responseTime);
  throughputCounter.add(1);
  
  // Validate response
  const success = check(response, {
    'status is 200-299': (r) => r.status >= 200 && r.status < 300,
    'response time acceptable': () => {
      {{#if (eq performance_targets "sub-100ms")}}
      return responseTime < 200; // Allow some margin
      {{else if (eq performance_targets "sub-500ms")}}
      return responseTime < 1000;
      {{else if (eq performance_targets "sub-1s")}}
      return responseTime < 2000;
      {{else}}
      return responseTime < 5000; // Throughput focused
      {{/if}}
    },
    'response has content': (r) => r.body && r.body.length > 0,
    'no server errors': (r) => r.status < 500
  });
  
  if (!success) {
    errorRate.add(1);
  }
  
  // Realistic user behavior - add think time
  sleep(Math.random() * 2 + 0.5); // 0.5-2.5 seconds
}

// Setup function - runs once per VU
export function setup() {
  // Warm up the API
  const baseUrl = __ENV.API_BASE_URL || 'http://localhost:3000';
  http.get(`${baseUrl}/health`);
  
  console.log('Load test setup completed');
  return { baseUrl };
}

// Teardown function
export function teardown(data) {
  console.log('Load test completed');
  
  // Could send results to external monitoring system
  const results = {
    timestamp: new Date().toISOString(),
    test_duration: options.stages.reduce((total, stage) => total + parseInt(stage.duration), 0),
    max_vus: Math.max(...options.stages.map(s => s.target)),
    configuration: {
      api_architecture: '{{api_architecture}}',
      backend_technology: '{{backend_technology}}',
      expected_load: '{{expected_load}}',
      performance_targets: '{{performance_targets}}'
    }
  };
  
  console.log('Test results:', JSON.stringify(results, null, 2));
}
```

### Performance Monitoring Dashboard
```python
# Performance monitoring and alerting system
import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class PerformanceMetric:
    timestamp: datetime
    endpoint: str
    method: str
    response_time: float
    status_code: int
    error_message: Optional[str] = None

class PerformanceMonitor:
    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url
        self.metrics: List[PerformanceMetric] = []
        self.alert_thresholds = {
            {{#if (eq performance_targets "sub-100ms")}}
            'response_time_p95': 0.1,  # 100ms
            'response_time_p99': 0.2,  # 200ms
            {{else if (eq performance_targets "sub-500ms")}}
            'response_time_p95': 0.5,  # 500ms
            'response_time_p99': 1.0,  # 1s
            {{else if (eq performance_targets "sub-1s")}}
            'response_time_p95': 1.0,  # 1s
            'response_time_p99': 2.0,  # 2s
            {{else}}
            'response_time_p95': 2.0,  # 2s
            'response_time_p99': 5.0,  # 5s
            {{/if}}
            'error_rate': 0.01,        # 1%
            'availability': 0.99       # 99%
        }
        
    async def monitor_endpoints(self, endpoints: List[str], duration_minutes: int = 60):
        """Monitor API endpoints for specified duration"""
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            while datetime.now() < end_time:
                tasks = []
                
                for endpoint in endpoints:
                    tasks.append(self._check_endpoint(session, endpoint))
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds
        
        return self.analyze_performance()
    
    async def _check_endpoint(self, session: aiohttp.ClientSession, endpoint: str):
        """Check single endpoint performance"""
        
        url = f"{self.api_base_url}{endpoint}"
        start_time = time.time()
        
        try:
            async with session.get(url) as response:
                await response.read()  # Ensure full response is received
                response_time = time.time() - start_time
                
                metric = PerformanceMetric(
                    timestamp=datetime.now(),
                    endpoint=endpoint,
                    method='GET',
                    response_time=response_time,
                    status_code=response.status
                )
                
                self.metrics.append(metric)
                
        except Exception as e:
            response_time = time.time() - start_time
            
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                endpoint=endpoint,
                method='GET',
                response_time=response_time,
                status_code=0,  # Indicate failure
                error_message=str(e)
            )
            
            self.metrics.append(metric)
    
    def analyze_performance(self) -> Dict:
        """Analyze collected performance metrics"""
        
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([asdict(m) for m in self.metrics])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate key metrics
        successful_requests = df[df['status_code'].between(200, 299)]
        
        analysis = {
            'summary': {
                'total_requests': len(df),
                'successful_requests': len(successful_requests),
                'error_rate': (len(df) - len(successful_requests)) / len(df),
                'availability': len(successful_requests) / len(df)
            },
            
            'response_times': {
                'mean': successful_requests['response_time'].mean(),
                'median': successful_requests['response_time'].median(),
                'p95': successful_requests['response_time'].quantile(0.95),
                'p99': successful_requests['response_time'].quantile(0.99),
                'min': successful_requests['response_time'].min(),
                'max': successful_requests['response_time'].max()
            },
            
            'by_endpoint': {},
            'alerts': []
        }
        
        # Analyze by endpoint
        for endpoint in df['endpoint'].unique():
            endpoint_data = successful_requests[successful_requests['endpoint'] == endpoint]
            
            if len(endpoint_data) > 0:
                analysis['by_endpoint'][endpoint] = {
                    'requests': len(endpoint_data),
                    'mean_response_time': endpoint_data['response_time'].mean(),
                    'p95_response_time': endpoint_data['response_time'].quantile(0.95),
                    'error_rate': (len(df[df['endpoint'] == endpoint]) - len(endpoint_data)) / len(df[df['endpoint'] == endpoint])
                }
        
        # Check for alerts
        if analysis['response_times']['p95'] > self.alert_thresholds['response_time_p95']:
            analysis['alerts'].append({
                'type': 'HIGH_RESPONSE_TIME_P95',
                'value': analysis['response_times']['p95'],
                'threshold': self.alert_thresholds['response_time_p95'],
                'severity': 'WARNING'
            })
        
        if analysis['response_times']['p99'] > self.alert_thresholds['response_time_p99']:
            analysis['alerts'].append({
                'type': 'HIGH_RESPONSE_TIME_P99',
                'value': analysis['response_times']['p99'],
                'threshold': self.alert_thresholds['response_time_p99'],
                'severity': 'CRITICAL'
            })
        
        if analysis['summary']['error_rate'] > self.alert_thresholds['error_rate']:
            analysis['alerts'].append({
                'type': 'HIGH_ERROR_RATE',
                'value': analysis['summary']['error_rate'],
                'threshold': self.alert_thresholds['error_rate'],
                'severity': 'CRITICAL'
            })
        
        if analysis['summary']['availability'] < self.alert_thresholds['availability']:
            analysis['alerts'].append({
                'type': 'LOW_AVAILABILITY',
                'value': analysis['summary']['availability'],
                'threshold': self.alert_thresholds['availability'],
                'severity': 'CRITICAL'
            })
        
        return analysis
    
    def generate_performance_report(self, analysis: Dict, output_file: str = 'performance_report.html'):
        """Generate HTML performance report"""
        
        # Create visualizations
        df = pd.DataFrame([asdict(m) for m in self.metrics])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        successful_requests = df[df['status_code'].between(200, 299)]
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Response time over time
        axes[0, 0].plot(successful_requests['timestamp'], successful_requests['response_time'])
        axes[0, 0].axhline(y=self.alert_thresholds['response_time_p95'], color='orange', linestyle='--', label='P95 Threshold')
        axes[0, 0].axhline(y=self.alert_thresholds['response_time_p99'], color='red', linestyle='--', label='P99 Threshold')
        axes[0, 0].set_title('Response Time Over Time')
        axes[0, 0].set_ylabel('Response Time (seconds)')
        axes[0, 0].legend()
        
        # Response time distribution
        axes[0, 1].hist(successful_requests['response_time'], bins=50, alpha=0.7)
        axes[0, 1].axvline(x=analysis['response_times']['p95'], color='orange', linestyle='--', label='P95')
        axes[0, 1].axvline(x=analysis['response_times']['p99'], color='red', linestyle='--', label='P99')
        axes[0, 1].set_title('Response Time Distribution')
        axes[0, 1].set_xlabel('Response Time (seconds)')
        axes[0, 1].legend()
        
        # Status code distribution
        status_counts = df['status_code'].value_counts()
        axes[1, 0].bar(status_counts.index.astype(str), status_counts.values)
        axes[1, 0].set_title('HTTP Status Code Distribution')
        axes[1, 0].set_xlabel('Status Code')
        axes[1, 0].set_ylabel('Count')
        
        # Response time by endpoint
        endpoint_response_times = []
        endpoints = []
        
        for endpoint in df['endpoint'].unique():
            endpoint_data = successful_requests[successful_requests['endpoint'] == endpoint]
            if len(endpoint_data) > 0:
                endpoint_response_times.append(endpoint_data['response_time'].values)
                endpoints.append(endpoint)
        
        if endpoint_response_times:
            axes[1, 1].boxplot(endpoint_response_times, labels=endpoints)
            axes[1, 1].set_title('Response Time by Endpoint')
            axes[1, 1].set_ylabel('Response Time (seconds)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .alert {{ background: #ffebee; padding: 10px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #f44336; }}
                .success {{ color: #4caf50; }}
                .warning {{ color: #ff9800; }}
                .critical {{ color: #f44336; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>API Performance Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <div class="metric">
                <strong>Total Requests:</strong> {analysis['summary']['total_requests']}<br>
                <strong>Successful Requests:</strong> {analysis['summary']['successful_requests']}<br>
                <strong>Error Rate:</strong> {analysis['summary']['error_rate']:.2%}<br>
                <strong>Availability:</strong> {analysis['summary']['availability']:.2%}
            </div>
            
            <h2>Response Time Metrics</h2>
            <div class="metric">
                <strong>Mean:</strong> {analysis['response_times']['mean']:.3f}s<br>
                <strong>Median:</strong> {analysis['response_times']['median']:.3f}s<br>
                <strong>95th Percentile:</strong> {analysis['response_times']['p95']:.3f}s<br>
                <strong>99th Percentile:</strong> {analysis['response_times']['p99']:.3f}s<br>
                <strong>Min:</strong> {analysis['response_times']['min']:.3f}s<br>
                <strong>Max:</strong> {analysis['response_times']['max']:.3f}s
            </div>
            
            <h2>Alerts</h2>
        """
        
        if analysis['alerts']:
            for alert in analysis['alerts']:
                severity_class = alert['severity'].lower()
                html_content += f"""
                <div class="alert {severity_class}">
                    <strong>{alert['type']}:</strong> {alert['value']:.3f} (threshold: {alert['threshold']:.3f})
                    <span class="{severity_class}">[{alert['severity']}]</span>
                </div>
                """
        else:
            html_content += '<div class="metric success">No alerts triggered ✓</div>'
        
        html_content += """
            <h2>Performance Visualization</h2>
            <img src="performance_metrics.png" alt="Performance Metrics" style="max-width: 100%;">
            
            <h2>Endpoint Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Endpoint</th>
                        <th>Requests</th>
                        <th>Mean Response Time</th>
                        <th>P95 Response Time</th>
                        <th>Error Rate</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for endpoint, metrics in analysis['by_endpoint'].items():
            html_content += f"""
                <tr>
                    <td>{endpoint}</td>
                    <td>{metrics['requests']}</td>
                    <td>{metrics['mean_response_time']:.3f}s</td>
                    <td>{metrics['p95_response_time']:.3f}s</td>
                    <td>{metrics['error_rate']:.2%}</td>
                </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file

# Usage example
async def main():
    monitor = PerformanceMonitor('http://localhost:3000')
    
    endpoints = [
        '/api/users',
        '/api/products',
        '/api/categories',
        '/api/orders'
    ]
    
    print("Starting performance monitoring...")
    analysis = await monitor.monitor_endpoints(endpoints, duration_minutes=10)
    
    print("Performance Analysis:")
    print(json.dumps(analysis, indent=2, default=str))
    
    # Generate report
    report_file = monitor.generate_performance_report(analysis)
    print(f"Performance report generated: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

This API Performance Optimization framework provides:

**Key Features:**
- {{api_architecture}} API optimization for {{backend_technology}}
- Multi-level caching strategy with Redis and in-memory caching
- Database query optimization for {{data_complexity}} workloads
- Comprehensive monitoring and alerting system

**Performance Targets:**
- {{performance_targets}} response time optimization
- {{expected_load}} load handling capacity
- {{infrastructure}} infrastructure optimization
- Real-time performance monitoring and alerting

**Benefits:**
- Scalable architecture supporting high concurrent loads
- Intelligent caching reducing database pressure
- Proactive performance monitoring and issue detection
- Comprehensive load testing and validation framework

**Success Metrics:**
- Consistent sub-target response times under load
- High availability (99.9%+) with fault tolerance
- Efficient resource utilization and cost optimization
- Automated performance regression detection