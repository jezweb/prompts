---
name: performance_optimization
title: Performance Optimization Guide
description: Analyze and optimize application performance issues with systematic approach and best practices
category: development
tags: [performance, optimization, debugging, profiling, speed]
difficulty: advanced
author: jezweb
version: 1.0.0
arguments:
  - name: app_type
    description: Type of application (web, api, mobile, desktop)
    required: true
  - name: technology
    description: Primary technology stack
    required: true
  - name: issue_description
    description: Description of performance issue
    required: true
  - name: metrics
    description: Current performance metrics (if available)
    required: false
    default: "Not measured yet"
  - name: target_improvement
    description: Target performance improvement percentage
    required: false
    default: "50%"
---

# Performance Optimization Analysis: {{app_type}}

**Technology Stack:** {{technology}}
**Issue:** {{issue_description}}
**Current Metrics:** {{metrics}}
**Target Improvement:** {{target_improvement}}

## 1. Performance Analysis Framework

### Initial Assessment
- [ ] Identify performance bottlenecks
- [ ] Establish baseline metrics
- [ ] Define success criteria
- [ ] Prioritize optimization efforts

### Measurement Tools

{{#if (eq app_type "web")}}
#### Web Application Tools
- **Browser DevTools**: Performance tab, Network analysis
- **Lighthouse**: Automated auditing
- **WebPageTest**: Detailed performance metrics
- **Chrome User Experience Report**: Real-world data

```javascript
// Performance measurement code
const perfData = window.performance.timing;
const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
const domReadyTime = perfData.domContentLoadedEventEnd - perfData.navigationStart;
const resourceLoadTime = perfData.loadEventEnd - perfData.responseEnd;

console.log('Page Load Time:', pageLoadTime, 'ms');
console.log('DOM Ready Time:', domReadyTime, 'ms');
console.log('Resource Load Time:', resourceLoadTime, 'ms');

// Core Web Vitals
new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log(`${entry.name}: ${entry.value}ms`);
  }
}).observe({ entryTypes: ['largest-contentful-paint', 'first-input', 'layout-shift'] });
```
{{/if}}

{{#if (eq app_type "api")}}
#### API Performance Tools
- **Apache Bench (ab)**: Load testing
- **JMeter**: Comprehensive performance testing
- **Grafana + Prometheus**: Monitoring
- **New Relic / DataDog**: APM solutions

```bash
# Basic load test
ab -n 1000 -c 100 http://api.example.com/endpoint

# Detailed performance test with curl
curl -w "@curl-format.txt" -o /dev/null -s http://api.example.com/endpoint

# curl-format.txt:
# time_namelookup:  %{time_namelookup}s\n
# time_connect:  %{time_connect}s\n
# time_appconnect:  %{time_appconnect}s\n
# time_pretransfer:  %{time_pretransfer}s\n
# time_redirect:  %{time_redirect}s\n
# time_starttransfer:  %{time_starttransfer}s\n
# time_total:  %{time_total}s\n
```
{{/if}}

## 2. Common Performance Issues & Solutions

### Issue Category: {{issue_description}}

{{#if (includes issue_description "slow")}}
#### Slow Load/Response Times

**Potential Causes:**
1. Unoptimized database queries
2. Lack of caching
3. Large payload sizes
4. Synchronous operations
5. Memory leaks

**Investigation Steps:**
```javascript
// Profile specific operations
console.time('operation');
// ... operation code ...
console.timeEnd('operation');

// Memory profiling
const used = process.memoryUsage();
console.log(`Memory Usage:
  RSS: ${Math.round(used.rss / 1024 / 1024 * 100) / 100} MB
  Heap Total: ${Math.round(used.heapTotal / 1024 / 1024 * 100) / 100} MB
  Heap Used: ${Math.round(used.heapUsed / 1024 / 1024 * 100) / 100} MB`);
```
{{/if}}

{{#if (includes issue_description "memory")}}
#### Memory Issues

**Memory Leak Detection:**
```javascript
// Node.js memory monitoring
const v8 = require('v8');
const heapSnapshot = v8.writeHeapSnapshot();

// Browser memory profiling
if (performance.memory) {
  console.log('Memory Usage:', {
    usedJSHeapSize: (performance.memory.usedJSHeapSize / 1048576).toFixed(2) + ' MB',
    totalJSHeapSize: (performance.memory.totalJSHeapSize / 1048576).toFixed(2) + ' MB',
    jsHeapSizeLimit: (performance.memory.jsHeapSizeLimit / 1048576).toFixed(2) + ' MB'
  });
}
```
{{/if}}

## 3. Optimization Strategies

### Frontend Optimizations ({{app_type}})

{{#if (or (eq app_type "web") (eq app_type "mobile"))}}
#### 1. Asset Optimization
```javascript
// Lazy loading images
<img loading="lazy" src="image.jpg" alt="Description">

// Code splitting (webpack)
const Component = lazy(() => import('./Component'));

// Resource hints
<link rel="preconnect" href="https://api.example.com">
<link rel="dns-prefetch" href="https://cdn.example.com">
<link rel="preload" href="critical.css" as="style">
```

#### 2. Bundle Size Reduction
```json
// webpack.config.js optimizations
{
  optimization: {
    usedExports: true,
    minimize: true,
    sideEffects: false,
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          priority: 10
        }
      }
    }
  }
}
```

#### 3. Rendering Performance
```javascript
// Use React.memo for expensive components
const ExpensiveComponent = React.memo(({ data }) => {
  return <div>{/* Complex rendering */}</div>
}, (prevProps, nextProps) => {
  return prevProps.data.id === nextProps.data.id;
});

// Virtual scrolling for long lists
import { FixedSizeList } from 'react-window';

const Row = ({ index, style }) => (
  <div style={style}>Row {index}</div>
);

const LongList = () => (
  <FixedSizeList
    height={600}
    itemCount={10000}
    itemSize={35}
    width='100%'
  >
    {Row}
  </FixedSizeList>
);
```
{{/if}}

### Backend Optimizations

{{#if (or (eq app_type "api") (eq app_type "web"))}}
#### 1. Database Query Optimization
```sql
-- Add indexes for frequently queried columns
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_created ON orders(user_id, created_at DESC);

-- Use EXPLAIN to analyze queries
EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 123;

-- Optimize N+1 queries with joins
-- Bad: N+1 queries
SELECT * FROM users;
-- Then for each user: SELECT * FROM orders WHERE user_id = ?

-- Good: Single query with join
SELECT u.*, o.* FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.status = 'active';
```

#### 2. Caching Strategy
```javascript
// Redis caching implementation
const redis = require('redis');
const client = redis.createClient();

async function getCachedData(key, fetchFunction, ttl = 3600) {
  // Try to get from cache
  const cached = await client.get(key);
  if (cached) {
    return JSON.parse(cached);
  }

  // Fetch fresh data
  const data = await fetchFunction();
  
  // Cache for future use
  await client.setex(key, ttl, JSON.stringify(data));
  
  return data;
}

// Usage
const userData = await getCachedData(
  `user:${userId}`,
  () => getUserFromDB(userId),
  300 // 5 minutes TTL
);
```

#### 3. Async Processing
```javascript
// Use message queues for heavy operations
const Queue = require('bull');
const emailQueue = new Queue('email notifications');

// Producer
await emailQueue.add('send-welcome', {
  userId: user.id,
  email: user.email
});

// Consumer
emailQueue.process('send-welcome', async (job) => {
  const { userId, email } = job.data;
  await sendWelcomeEmail(email);
  return { sent: true };
});
```
{{/if}}

## 4. Performance Monitoring Setup

### Continuous Monitoring
```javascript
// Custom performance monitoring
class PerformanceMonitor {
  constructor() {
    this.metrics = {};
  }

  startTimer(label) {
    this.metrics[label] = { start: Date.now() };
  }

  endTimer(label) {
    if (this.metrics[label]) {
      this.metrics[label].duration = Date.now() - this.metrics[label].start;
      this.logMetric(label, this.metrics[label].duration);
    }
  }

  logMetric(label, value) {
    console.log(`Performance [${label}]: ${value}ms`);
    // Send to monitoring service
    if (window.analytics) {
      window.analytics.track('Performance Metric', {
        label,
        value,
        timestamp: new Date().toISOString()
      });
    }
  }
}

const monitor = new PerformanceMonitor();
```

### Alert Configuration
```yaml
# Example alerting rules
alerts:
  - name: high_response_time
    condition: avg(response_time) > 1000
    duration: 5m
    action: notify_team

  - name: memory_usage_critical
    condition: memory_usage_percent > 90
    duration: 2m
    action: scale_up
```

## 5. Performance Optimization Checklist

### Before Optimization
- [ ] Establish baseline metrics
- [ ] Identify specific bottlenecks
- [ ] Set measurable goals
- [ ] Create performance budget

### Quick Wins
- [ ] Enable compression (gzip/brotli)
- [ ] Implement browser caching headers
- [ ] Optimize images (format, size, lazy loading)
- [ ] Minify CSS/JS/HTML
- [ ] Remove unused dependencies

### Database & Backend
- [ ] Add appropriate indexes
- [ ] Optimize slow queries
- [ ] Implement caching layer
- [ ] Use connection pooling
- [ ] Enable query result caching

### Frontend
- [ ] Code splitting
- [ ] Tree shaking
- [ ] Lazy loading
- [ ] Service workers
- [ ] CDN for static assets

### Monitoring
- [ ] Set up performance monitoring
- [ ] Configure alerts
- [ ] Regular performance audits
- [ ] Track core metrics over time

## 6. Expected Results

Based on implementing these optimizations for {{issue_description}}:

**Target Metrics:**
- Response Time: Reduce by {{target_improvement}}
- Page Load: < 3 seconds on 3G
- Time to Interactive: < 5 seconds
- API Response: < 200ms (p95)

**Next Steps:**
1. Implement quick wins first
2. Measure impact of each change
3. Focus on highest-impact optimizations
4. Set up continuous monitoring
5. Regular performance reviews