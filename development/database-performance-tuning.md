---
name: database_performance_tuning
title: Database Performance Tuning Guide
description: Comprehensive database performance optimization covering query analysis, indexing strategies, configuration tuning, and monitoring for production systems
category: development
tags: [database, performance, optimization, indexing, sql, monitoring]
difficulty: advanced
author: jezweb
version: 1.0.0
arguments:
  - name: database_type
    description: Database type (postgresql, mysql, mongodb, redis, elasticsearch)
    required: true
  - name: workload_pattern
    description: Primary workload pattern (oltp, olap, mixed, real-time, analytics)
    required: true
  - name: data_size
    description: Database size (small <10GB, medium 10GB-1TB, large >1TB)
    required: true
  - name: performance_goals
    description: Performance goals (latency, throughput, cost-optimization, availability)
    required: true
  - name: environment
    description: Environment type (development, staging, production, high-availability)
    required: true
  - name: concurrent_users
    description: Expected concurrent users (low <100, medium 100-1000, high >1000)
    required: true
---

# Database Performance Tuning: {{database_type}}

**Workload Pattern:** {{workload_pattern}}  
**Data Size:** {{data_size}}  
**Performance Goals:** {{performance_goals}}  
**Environment:** {{environment}}  
**Concurrent Users:** {{concurrent_users}}

## 1. Performance Analysis Framework

### Baseline Performance Assessment
```sql
{{#if (eq database_type "postgresql")}}
-- PostgreSQL performance analysis queries
-- Current connection and activity analysis
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query_start,
    state_change,
    waiting,
    query
FROM pg_stat_activity 
WHERE state != 'idle' 
ORDER BY query_start;

-- Long running queries
SELECT 
    pid,
    now() - pg_stat_activity.query_start AS duration,
    query,
    state
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
    AND state != 'idle';

-- Table and index usage statistics
SELECT 
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_tup_hot_upd
FROM pg_stat_user_tables 
ORDER BY seq_scan DESC;

-- Index usage analysis
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
ORDER BY idx_scan DESC;

-- Database size and growth analysis
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as data_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size
FROM pg_tables 
WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Buffer cache hit ratio
SELECT 
    'cache hit ratio' as metric,
    CASE 
        WHEN sum(blks_hit) + sum(blks_read) = 0 THEN 0
        ELSE round((sum(blks_hit) * 100.0) / (sum(blks_hit) + sum(blks_read)), 2)
    END as ratio
FROM pg_stat_database;
{{/if}}

{{#if (eq database_type "mysql")}}
-- MySQL performance analysis queries
-- Current connections and processes
SELECT 
    ID,
    USER,
    HOST,
    DB,
    COMMAND,
    TIME,
    STATE,
    INFO as QUERY
FROM INFORMATION_SCHEMA.PROCESSLIST 
WHERE COMMAND != 'Sleep' 
ORDER BY TIME DESC;

-- Slow query analysis
SELECT 
    query_time,
    lock_time,
    rows_sent,
    rows_examined,
    sql_text
FROM mysql.slow_log 
ORDER BY query_time DESC 
LIMIT 10;

-- Table and index statistics
SELECT 
    TABLE_SCHEMA,
    TABLE_NAME,
    ENGINE,
    TABLE_ROWS,
    DATA_LENGTH,
    INDEX_LENGTH,
    DATA_FREE,
    AUTO_INCREMENT
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
ORDER BY DATA_LENGTH + INDEX_LENGTH DESC;

-- Index usage statistics
SELECT 
    OBJECT_SCHEMA,
    OBJECT_NAME,
    INDEX_NAME,
    COUNT_FETCH,
    COUNT_INSERT,
    COUNT_UPDATE,
    COUNT_DELETE
FROM performance_schema.table_io_waits_summary_by_index_usage 
WHERE OBJECT_SCHEMA NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
ORDER BY COUNT_FETCH DESC;

-- Buffer pool hit ratio
SELECT 
    'Buffer Pool Hit Ratio' as Metric,
    ROUND(
        (1 - (Innodb_buffer_pool_reads / Innodb_buffer_pool_read_requests)) * 100, 2
    ) as Hit_Ratio_Percent
FROM 
    (SELECT VARIABLE_VALUE as Innodb_buffer_pool_reads 
     FROM performance_schema.global_status 
     WHERE VARIABLE_NAME = 'Innodb_buffer_pool_reads') reads,
    (SELECT VARIABLE_VALUE as Innodb_buffer_pool_read_requests 
     FROM performance_schema.global_status 
     WHERE VARIABLE_NAME = 'Innodb_buffer_pool_read_requests') requests;
{{/if}}
```

### Performance Monitoring Setup
```python
# Database performance monitoring script
import psycopg2
import mysql.connector
import time
import json
from datetime import datetime
from typing import Dict, List, Any

class DatabaseMonitor:
    def __init__(self, db_type: str, connection_params: Dict):
        self.db_type = db_type
        self.connection_params = connection_params
        self.metrics_history = []
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive database metrics"""
        
        if self.db_type == "postgresql":
            return self.collect_postgresql_metrics()
        elif self.db_type == "mysql":
            return self.collect_mysql_metrics()
        elif self.db_type == "mongodb":
            return self.collect_mongodb_metrics()
    
    def collect_postgresql_metrics(self) -> Dict[str, Any]:
        """Collect PostgreSQL specific metrics"""
        conn = psycopg2.connect(**self.connection_params)
        cursor = conn.cursor()
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'database_type': 'postgresql'
        }
        
        # Connection metrics
        cursor.execute("""
            SELECT count(*) as total_connections,
                   count(*) FILTER (WHERE state = 'active') as active_connections,
                   count(*) FILTER (WHERE state = 'idle') as idle_connections
            FROM pg_stat_activity
        """)
        conn_stats = cursor.fetchone()
        metrics['connections'] = {
            'total': conn_stats[0],
            'active': conn_stats[1],
            'idle': conn_stats[2]
        }
        
        # Cache hit ratio
        cursor.execute("""
            SELECT round((sum(blks_hit) * 100.0) / (sum(blks_hit) + sum(blks_read)), 2) as cache_hit_ratio
            FROM pg_stat_database
            WHERE datname = current_database()
        """)
        cache_hit = cursor.fetchone()[0]
        metrics['cache_hit_ratio'] = float(cache_hit) if cache_hit else 0
        
        # Database size
        cursor.execute("SELECT pg_database_size(current_database())")
        db_size = cursor.fetchone()[0]
        metrics['database_size_bytes'] = db_size
        
        # Long running queries
        cursor.execute("""
            SELECT count(*) 
            FROM pg_stat_activity 
            WHERE state = 'active' 
            AND (now() - query_start) > interval '30 seconds'
        """)
        long_queries = cursor.fetchone()[0]
        metrics['long_running_queries'] = long_queries
        
        # Locks
        cursor.execute("""
            SELECT mode, count(*) 
            FROM pg_locks 
            GROUP BY mode
        """)
        locks = dict(cursor.fetchall())
        metrics['locks'] = locks
        
        # Top tables by size
        cursor.execute("""
            SELECT 
                schemaname || '.' || tablename as table_name,
                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
            FROM pg_tables 
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
            ORDER BY size_bytes DESC 
            LIMIT 10
        """)
        top_tables = [{'table': row[0], 'size_bytes': row[1]} for row in cursor.fetchall()]
        metrics['top_tables_by_size'] = top_tables
        
        cursor.close()
        conn.close()
        
        return metrics
    
    def analyze_slow_queries(self) -> List[Dict]:
        """Analyze slow queries and provide optimization recommendations"""
        
        if self.db_type == "postgresql":
            return self.analyze_postgresql_slow_queries()
        elif self.db_type == "mysql":
            return self.analyze_mysql_slow_queries()
    
    def analyze_postgresql_slow_queries(self) -> List[Dict]:
        """Analyze PostgreSQL slow queries"""
        conn = psycopg2.connect(**self.connection_params)
        cursor = conn.cursor()
        
        # Get slow queries from pg_stat_statements if available
        try:
            cursor.execute("""
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    max_time,
                    rows,
                    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements 
                WHERE mean_time > 100  -- queries taking more than 100ms on average
                ORDER BY total_time DESC 
                LIMIT 20
            """)
            
            slow_queries = []
            for row in cursor.fetchall():
                query_analysis = {
                    'query': row[0],
                    'calls': row[1],
                    'total_time_ms': float(row[2]),
                    'mean_time_ms': float(row[3]),
                    'max_time_ms': float(row[4]),
                    'rows_returned': row[5],
                    'cache_hit_percent': float(row[6]) if row[6] else 0,
                    'recommendations': self.generate_query_recommendations(row[0])
                }
                slow_queries.append(query_analysis)
            
            return slow_queries
            
        except Exception as e:
            print(f"pg_stat_statements not available: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def generate_query_recommendations(self, query: str) -> List[str]:
        """Generate optimization recommendations for a query"""
        recommendations = []
        
        query_lower = query.lower()
        
        # Check for common performance issues
        if 'select *' in query_lower:
            recommendations.append("Avoid SELECT * - specify only needed columns")
        
        if 'order by' in query_lower and 'limit' not in query_lower:
            recommendations.append("Consider adding LIMIT to ORDER BY queries")
        
        if query_lower.count('join') > 3:
            recommendations.append("Query has many JOINs - consider denormalization or query splitting")
        
        if 'where' not in query_lower and ('select' in query_lower and 'from' in query_lower):
            recommendations.append("Query lacks WHERE clause - may scan entire table")
        
        if 'group by' in query_lower:
            recommendations.append("Ensure GROUP BY columns are indexed")
        
        if 'or' in query_lower:
            recommendations.append("OR conditions may prevent index usage - consider UNION")
        
        if 'like' in query_lower and query_lower.find("like '%") != -1:
            recommendations.append("Leading wildcard LIKE queries cannot use indexes efficiently")
        
        return recommendations
```

## 2. {{database_type}} Specific Optimizations

{{#if (eq database_type "postgresql")}}
### PostgreSQL Configuration Tuning
```sql
-- PostgreSQL configuration optimization for {{data_size}} database
-- Memory configuration
{{#if (eq data_size "large")}}
-- Large database (>1TB) configuration
ALTER SYSTEM SET shared_buffers = '25%';  -- 25% of total RAM
ALTER SYSTEM SET effective_cache_size = '75%';  -- 75% of total RAM
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
{{else if (eq data_size "medium")}}
-- Medium database (10GB-1TB) configuration
ALTER SYSTEM SET shared_buffers = '25%';
ALTER SYSTEM SET effective_cache_size = '75%';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
{{else}}
-- Small database (<10GB) configuration
ALTER SYSTEM SET shared_buffers = '128MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '128MB';
{{/if}}

-- Connection and concurrency settings
{{#if (eq concurrent_users "high")}}
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET max_worker_processes = 16;
ALTER SYSTEM SET max_parallel_workers = 12;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
{{else if (eq concurrent_users "medium")}}
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET max_worker_processes = 8;
ALTER SYSTEM SET max_parallel_workers = 6;
ALTER SYSTEM SET max_parallel_workers_per_gather = 2;
{{else}}
ALTER SYSTEM SET max_connections = 50;
ALTER SYSTEM SET max_worker_processes = 4;
ALTER SYSTEM SET max_parallel_workers = 2;
ALTER SYSTEM SET max_parallel_workers_per_gather = 1;
{{/if}}

-- WAL and checkpoint configuration
{{#if (includes performance_goals "throughput")}}
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET max_wal_size = '4GB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET checkpoint_timeout = '15min';
{{else}}
ALTER SYSTEM SET wal_buffers = '8MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.7;
ALTER SYSTEM SET max_wal_size = '1GB';
ALTER SYSTEM SET min_wal_size = '256MB';
{{/if}}

-- Query planner configuration
ALTER SYSTEM SET random_page_cost = 1.1;  -- For SSD storage
ALTER SYSTEM SET seq_page_cost = 1.0;
ALTER SYSTEM SET cpu_tuple_cost = 0.01;
ALTER SYSTEM SET cpu_index_tuple_cost = 0.005;
ALTER SYSTEM SET cpu_operator_cost = 0.0025;

-- Autovacuum configuration for {{workload_pattern}}
{{#if (eq workload_pattern "oltp")}}
-- OLTP workload - more aggressive autovacuum
ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.1;
ALTER SYSTEM SET autovacuum_analyze_scale_factor = 0.05;
ALTER SYSTEM SET autovacuum_vacuum_cost_delay = 10;
ALTER SYSTEM SET autovacuum_vacuum_cost_limit = 1000;
{{else if (eq workload_pattern "olap")}}
-- OLAP workload - less frequent autovacuum
ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.2;
ALTER SYSTEM SET autovacuum_analyze_scale_factor = 0.1;
ALTER SYSTEM SET autovacuum_vacuum_cost_delay = 20;
ALTER SYSTEM SET autovacuum_vacuum_cost_limit = 500;
{{/if}}

-- Reload configuration
SELECT pg_reload_conf();

-- Enable performance monitoring extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_buffercache;
```

### PostgreSQL Indexing Strategy
```sql
-- Comprehensive indexing strategy for {{workload_pattern}} workload

-- B-tree indexes for equality and range queries
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY idx_orders_created_at ON orders(created_at);
CREATE INDEX CONCURRENTLY idx_products_category_id ON products(category_id);

-- Composite indexes for multi-column queries
CREATE INDEX CONCURRENTLY idx_orders_user_status ON orders(user_id, status);
CREATE INDEX CONCURRENTLY idx_logs_timestamp_level ON logs(timestamp DESC, level);

-- Partial indexes for filtered queries
CREATE INDEX CONCURRENTLY idx_orders_active ON orders(created_at) 
WHERE status IN ('pending', 'processing');

CREATE INDEX CONCURRENTLY idx_users_verified ON users(created_at) 
WHERE email_verified = true;

-- Expression indexes for computed values
CREATE INDEX CONCURRENTLY idx_orders_total_amount ON orders((quantity * price));
CREATE INDEX CONCURRENTLY idx_users_full_name ON users(lower(first_name || ' ' || last_name));

{{#if (includes workload_pattern "analytics")}}
-- BRIN indexes for time-series data (good for large append-only tables)
CREATE INDEX CONCURRENTLY idx_events_timestamp_brin ON events USING BRIN(timestamp);
CREATE INDEX CONCURRENTLY idx_metrics_date_brin ON metrics USING BRIN(date);
{{/if}}

-- GIN indexes for full-text search and JSONB
CREATE INDEX CONCURRENTLY idx_products_search ON products 
USING GIN(to_tsvector('english', name || ' ' || description));

CREATE INDEX CONCURRENTLY idx_user_preferences ON users 
USING GIN(preferences);  -- For JSONB column

-- Hash indexes for equality-only queries (PostgreSQL 10+)
CREATE INDEX CONCURRENTLY idx_sessions_token_hash ON sessions USING HASH(session_token);

-- Index maintenance queries
-- Find unused indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes 
WHERE idx_scan = 0 
    AND schemaname NOT IN ('information_schema', 'pg_catalog')
ORDER BY pg_relation_size(indexrelid) DESC;

-- Find duplicate indexes
SELECT 
    a.indexname as index1,
    b.indexname as index2,
    a.tablename,
    a.indexdef
FROM pg_indexes a
JOIN pg_indexes b ON a.tablename = b.tablename 
    AND a.indexname < b.indexname
    AND a.indexdef = b.indexdef
WHERE a.schemaname = 'public';

-- Analyze index bloat
SELECT 
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
    CASE 
        WHEN avg_leaf_density = 0 THEN 0
        ELSE round((100 * (1 - avg_leaf_density/fillfactor))::numeric, 2)
    END as bloat_ratio
FROM pg_stat_user_indexes 
JOIN pg_index ON pg_index.indexrelid = pg_stat_user_indexes.indexrelid
JOIN pg_class ON pg_class.oid = pg_index.indexrelid
JOIN pg_am ON pg_class.relam = pg_am.oid
WHERE pg_am.amname = 'btree'
ORDER BY pg_relation_size(indexrelid) DESC;
```
{{/if}}

{{#if (eq database_type "mysql")}}
### MySQL Configuration Tuning
```ini
# MySQL configuration optimization for {{data_size}} database
[mysqld]

# Memory configuration
{{#if (eq data_size "large")}}
# Large database (>1TB) configuration
innodb_buffer_pool_size = 32G  # 70-80% of available RAM
innodb_buffer_pool_instances = 16
innodb_log_file_size = 2G
innodb_log_buffer_size = 256M
key_buffer_size = 2G
{{else if (eq data_size "medium")}}
# Medium database (10GB-1TB) configuration
innodb_buffer_pool_size = 8G
innodb_buffer_pool_instances = 8
innodb_log_file_size = 512M
innodb_log_buffer_size = 64M
key_buffer_size = 512M
{{else}}
# Small database (<10GB) configuration
innodb_buffer_pool_size = 2G
innodb_buffer_pool_instances = 4
innodb_log_file_size = 256M
innodb_log_buffer_size = 32M
key_buffer_size = 256M
{{/if}}

# Connection settings
{{#if (eq concurrent_users "high")}}
max_connections = 300
max_user_connections = 250
thread_cache_size = 50
table_open_cache = 4000
{{else if (eq concurrent_users "medium")}}
max_connections = 150
max_user_connections = 100
thread_cache_size = 25
table_open_cache = 2000
{{else}}
max_connections = 75
max_user_connections = 50
thread_cache_size = 10
table_open_cache = 1000
{{/if}}

# InnoDB configuration
innodb_file_per_table = 1
innodb_flush_log_at_trx_commit = 1
innodb_flush_method = O_DIRECT
innodb_doublewrite = 1
innodb_read_io_threads = 8
innodb_write_io_threads = 8

{{#if (includes performance_goals "throughput")}}
# Optimized for throughput
innodb_flush_log_at_trx_commit = 2
innodb_sync_binlog = 0
innodb_io_capacity = 2000
innodb_io_capacity_max = 4000
{{else}}
# Optimized for durability
innodb_flush_log_at_trx_commit = 1
innodb_sync_binlog = 1
innodb_io_capacity = 1000
innodb_io_capacity_max = 2000
{{/if}}

# Query cache (disable for MySQL 5.7+, removed in 8.0)
query_cache_type = 0
query_cache_size = 0

# Binary logging
log_bin = mysql-bin
binlog_format = ROW
expire_logs_days = 7
max_binlog_size = 500M

# Slow query log
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow.log
long_query_time = 1
log_queries_not_using_indexes = 1

# Performance schema
performance_schema = ON
performance-schema-consumer-events-statements-current = ON
performance-schema-consumer-events-statements-history = ON
performance-schema-consumer-statements-digest = ON
```

### MySQL Indexing and Query Optimization
```sql
-- MySQL indexing strategy for {{workload_pattern}} workload

-- Primary and unique indexes
ALTER TABLE users ADD INDEX idx_email (email);
ALTER TABLE users ADD UNIQUE INDEX idx_username (username);

-- Composite indexes (column order matters)
ALTER TABLE orders ADD INDEX idx_user_status_date (user_id, status, created_at);
ALTER TABLE products ADD INDEX idx_category_price (category_id, price);

-- Covering indexes (include all columns needed by query)
ALTER TABLE orders ADD INDEX idx_covering_order_summary (
    user_id, status, created_at, total_amount
);

-- Prefix indexes for string columns
ALTER TABLE articles ADD INDEX idx_title_prefix (title(20));
ALTER TABLE logs ADD INDEX idx_message_prefix (message(50));

{{#if (eq workload_pattern "oltp")}}
-- OLTP-optimized indexes
-- Fast lookups for primary keys and foreign keys
ALTER TABLE order_items ADD INDEX idx_order_id (order_id);
ALTER TABLE order_items ADD INDEX idx_product_id (product_id);

-- Indexes for frequent WHERE conditions
ALTER TABLE users ADD INDEX idx_status_created (status, created_at);
ALTER TABLE orders ADD INDEX idx_status (status);
{{/if}}

{{#if (includes workload_pattern "analytics")}}
-- Analytics-optimized indexes
-- Indexes for date range queries
ALTER TABLE sales ADD INDEX idx_sale_date (sale_date);
ALTER TABLE events ADD INDEX idx_event_timestamp (event_timestamp);

-- Indexes for aggregation queries
ALTER TABLE sales ADD INDEX idx_product_date_amount (product_id, sale_date, amount);
{{/if}}

-- Full-text search indexes
ALTER TABLE articles ADD FULLTEXT INDEX idx_content_search (title, content);
ALTER TABLE products ADD FULLTEXT INDEX idx_product_search (name, description);

-- Index analysis and optimization
-- Find unused indexes
SELECT 
    object_schema,
    object_name,
    index_name,
    count_fetch,
    count_insert,
    count_update,
    count_delete
FROM performance_schema.table_io_waits_summary_by_index_usage 
WHERE object_schema NOT IN ('mysql', 'information_schema', 'performance_schema')
    AND count_fetch = 0
ORDER BY object_schema, object_name;

-- Analyze query performance
SELECT 
    digest_text,
    count_star,
    avg_timer_wait/1000000000 as avg_time_sec,
    sum_timer_wait/1000000000 as total_time_sec,
    min_timer_wait/1000000000 as min_time_sec,
    max_timer_wait/1000000000 as max_time_sec
FROM performance_schema.events_statements_summary_by_digest 
ORDER BY sum_timer_wait DESC 
LIMIT 10;
```
{{/if}}

{{#if (eq database_type "mongodb")}}
### MongoDB Configuration and Optimization
```javascript
// MongoDB configuration for {{data_size}} database

// Connection and memory settings
// In mongod.conf:
/*
{{#if (eq data_size "large")}}
# Large database configuration
storage:
  wiredTiger:
    engineConfig:
      cacheSizeGB: 32
    collectionConfig:
      blockCompressor: zstd
    indexConfig:
      prefixCompression: true

net:
  maxIncomingConnections: 1000

operationProfiling:
  slowOpThresholdMs: 100
{{else if (eq data_size "medium")}}
# Medium database configuration  
storage:
  wiredTiger:
    engineConfig:
      cacheSizeGB: 8
    collectionConfig:
      blockCompressor: snappy

net:
  maxIncomingConnections: 500

operationProfiling:
  slowOpThresholdMs: 200
{{else}}
# Small database configuration
storage:
  wiredTiger:
    engineConfig:
      cacheSizeGB: 2

net:
  maxIncomingConnections: 200

operationProfiling:
  slowOpThresholdMs: 500
{{/if}}
*/

// Indexing strategy for MongoDB
// Single field indexes
db.users.createIndex({ "email": 1 });
db.orders.createIndex({ "created_at": -1 });
db.products.createIndex({ "category": 1 });

// Compound indexes (order matters)
db.orders.createIndex({ "user_id": 1, "status": 1, "created_at": -1 });
db.logs.createIndex({ "timestamp": -1, "level": 1 });

// Text indexes for search
db.products.createIndex({ 
    "name": "text", 
    "description": "text" 
}, {
    weights: { name: 10, description: 1 }
});

// Geospatial indexes
db.locations.createIndex({ "coordinates": "2dsphere" });

// Sparse indexes for optional fields
db.users.createIndex({ "phone": 1 }, { sparse: true });

{{#if (eq workload_pattern "oltp")}}
// OLTP optimization - ensure queries use indexes
db.users.createIndex({ "username": 1 }, { unique: true });
db.sessions.createIndex({ "session_id": 1 }, { unique: true });
db.orders.createIndex({ "order_number": 1 }, { unique: true });
{{/if}}

{{#if (includes workload_pattern "analytics")}}
// Analytics optimization - optimize for aggregation
db.sales.createIndex({ "date": 1, "product_id": 1, "amount": 1 });
db.events.createIndex({ "timestamp": 1, "event_type": 1 });

// Enable aggregation framework optimizations
db.adminCommand({ "setParameter": 1, "internalQueryEnableSlotBasedExecutionEngine": true });
{{/if}}

// Performance monitoring and analysis
// Enable profiling for slow operations
db.setProfilingLevel(2, { slowms: 100 });

// Query performance analysis
function analyzeSlowQueries() {
    var slowQueries = db.system.profile.find().sort({ duration: -1 }).limit(10);
    
    slowQueries.forEach(function(query) {
        print("Duration: " + query.duration + "ms");
        print("Command: " + JSON.stringify(query.command));
        print("Execution Stats: " + JSON.stringify(query.execStats));
        print("---");
    });
}

// Index usage analysis
function analyzeIndexUsage() {
    var collections = db.runCommand("listCollections").cursor.firstBatch;
    
    collections.forEach(function(collection) {
        var collName = collection.name;
        if (!collName.startsWith("system.")) {
            print("Collection: " + collName);
            var stats = db[collName].aggregate([
                { $indexStats: {} }
            ]).toArray();
            
            stats.forEach(function(indexStat) {
                print("  Index: " + indexStat.name);
                print("  Usage: " + indexStat.accesses.ops);
                print("  Since: " + indexStat.accesses.since);
            });
            print("---");
        }
    });
}

// Sharding configuration for large datasets
{{#if (eq data_size "large")}}
// Enable sharding
sh.enableSharding("myapp");

// Shard collections based on access patterns
sh.shardCollection("myapp.users", { "_id": "hashed" });
sh.shardCollection("myapp.orders", { "user_id": 1, "created_at": 1 });
sh.shardCollection("myapp.logs", { "timestamp": 1 });

// Configure zones for geographic distribution
sh.addShardToZone("shard0000", "US");
sh.addShardToZone("shard0001", "EU");
sh.updateZoneKeyRange("myapp.users", { "region": "US" }, { "region": "US\uffff" }, "US");
sh.updateZoneKeyRange("myapp.users", { "region": "EU" }, { "region": "EU\uffff" }, "EU");
{{/if}}
```
{{/if}}

## 3. Query Optimization Techniques

### Query Analysis and Rewriting
```python
# Query optimization analyzer
import re
from typing import List, Dict, Any

class QueryOptimizer:
    def __init__(self, db_type: str):
        self.db_type = db_type
        self.optimization_rules = self.load_optimization_rules()
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query and provide optimization recommendations"""
        
        analysis = {
            'original_query': query,
            'issues': [],
            'recommendations': [],
            'optimized_query': None,
            'estimated_improvement': None
        }
        
        # Normalize query for analysis
        normalized_query = query.lower().strip()
        
        # Check for common anti-patterns
        analysis['issues'].extend(self.detect_antipatterns(normalized_query))
        
        # Generate recommendations
        analysis['recommendations'].extend(self.generate_recommendations(normalized_query))
        
        # Generate optimized query if possible
        optimized = self.optimize_query(query)
        if optimized != query:
            analysis['optimized_query'] = optimized
            analysis['estimated_improvement'] = "20-50% performance improvement expected"
        
        return analysis
    
    def detect_antipatterns(self, query: str) -> List[str]:
        """Detect common query anti-patterns"""
        issues = []
        
        # SELECT * usage
        if 'select *' in query:
            issues.append("SELECT * can be inefficient - specify only needed columns")
        
        # Missing WHERE clause on large tables
        if 'select' in query and 'from' in query and 'where' not in query:
            if 'limit' not in query:
                issues.append("Query lacks WHERE clause and may scan entire table")
        
        # OR conditions that prevent index usage
        if ' or ' in query:
            issues.append("OR conditions may prevent efficient index usage")
        
        # Functions in WHERE clause
        function_patterns = [r'where.*\w+\(.*\)\s*=', r'where.*upper\(', r'where.*lower\(']
        for pattern in function_patterns:
            if re.search(pattern, query):
                issues.append("Functions in WHERE clause prevent index usage")
                break
        
        # Leading wildcards in LIKE
        if re.search(r'like\s+[\'"]%', query):
            issues.append("Leading wildcard in LIKE prevents index usage")
        
        # Unnecessary DISTINCT
        if 'distinct' in query and 'group by' in query:
            issues.append("DISTINCT with GROUP BY is usually redundant")
        
        # Large OFFSET values
        offset_match = re.search(r'offset\s+(\d+)', query)
        if offset_match and int(offset_match.group(1)) > 1000:
            issues.append("Large OFFSET values are inefficient for pagination")
        
        return issues
    
    def generate_recommendations(self, query: str) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if 'select *' in query:
            recommendations.append("Replace SELECT * with specific column names")
        
        if 'order by' in query and 'limit' not in query:
            recommendations.append("Consider adding LIMIT to ORDER BY queries")
        
        if ' or ' in query:
            recommendations.append("Consider rewriting OR conditions as UNION for better performance")
        
        if 'in (' in query:
            recommendations.append("For large IN lists, consider using EXISTS or JOIN instead")
        
        if re.search(r'join.*join.*join', query):
            recommendations.append("Multiple JOINs detected - ensure proper indexing on join columns")
        
        if 'group by' in query:
            recommendations.append("Ensure GROUP BY columns are indexed")
        
        if 'having' in query:
            recommendations.append("Move conditions from HAVING to WHERE when possible")
        
        return recommendations
    
    def optimize_query(self, query: str) -> str:
        """Generate optimized version of query"""
        optimized = query
        
        # Replace SELECT * with specific columns (simplified example)
        if 'select *' in query.lower():
            # This would need more sophisticated parsing in practice
            optimized = re.sub(r'select\s+\*', 'SELECT id, name, email', optimized, flags=re.IGNORECASE)
        
        # Add LIMIT to ORDER BY queries without it
        if 'order by' in query.lower() and 'limit' not in query.lower():
            optimized += ' LIMIT 100'
        
        return optimized

# Example usage
optimizer = QueryOptimizer('{{database_type}}')

# Analyze problematic queries
sample_queries = [
    "SELECT * FROM users WHERE UPPER(email) = 'USER@EXAMPLE.COM'",
    "SELECT DISTINCT user_id FROM orders GROUP BY user_id",
    "SELECT * FROM products WHERE name LIKE '%phone%' ORDER BY price",
    "SELECT COUNT(*) FROM logs WHERE DATE(created_at) = '2024-01-01'"
]

for query in sample_queries:
    analysis = optimizer.analyze_query(query)
    print(f"Query: {analysis['original_query']}")
    print(f"Issues: {analysis['issues']}")
    print(f"Recommendations: {analysis['recommendations']}")
    if analysis['optimized_query']:
        print(f"Optimized: {analysis['optimized_query']}")
    print("---")
```

## 4. Caching Strategies

### Multi-Level Caching Implementation
```python
# Comprehensive caching strategy
import redis
import memcache
import json
import hashlib
from typing import Any, Optional, Dict
from datetime import timedelta

class CacheManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize cache layers
        if config.get('redis_enabled'):
            self.redis_client = redis.Redis(
                host=config['redis_host'],
                port=config['redis_port'],
                db=config['redis_db'],
                decode_responses=True
            )
        
        if config.get('memcached_enabled'):
            self.memcached_client = memcache.Client([
                f"{config['memcached_host']}:{config['memcached_port']}"
            ])
        
        # Application-level cache (in-memory)
        self.app_cache = {}
        self.app_cache_max_size = config.get('app_cache_max_size', 1000)
    
    def generate_cache_key(self, query: str, params: tuple = ()) -> str:
        """Generate consistent cache key for query and parameters"""
        key_string = f"{query}:{str(params)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_cached_result(self, query: str, params: tuple = ()) -> Optional[Any]:
        """Get cached result with multi-level lookup"""
        cache_key = self.generate_cache_key(query, params)
        
        # Level 1: Application cache (fastest)
        if cache_key in self.app_cache:
            return self.app_cache[cache_key]['data']
        
        # Level 2: Redis cache (fast)
        if hasattr(self, 'redis_client'):
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    # Promote to application cache
                    self.set_app_cache(cache_key, data)
                    return data
            except Exception as e:
                print(f"Redis cache error: {e}")
        
        # Level 3: Memcached (fallback)
        if hasattr(self, 'memcached_client'):
            try:
                cached_data = self.memcached_client.get(cache_key)
                if cached_data:
                    # Promote to higher cache levels
                    self.set_app_cache(cache_key, cached_data)
                    if hasattr(self, 'redis_client'):
                        self.redis_client.setex(
                            cache_key, 
                            self.config.get('redis_ttl', 3600),
                            json.dumps(cached_data)
                        )
                    return cached_data
            except Exception as e:
                print(f"Memcached error: {e}")
        
        return None
    
    async def cache_result(self, query: str, params: tuple, data: Any, ttl: int = 3600):
        """Cache result in all available cache layers"""
        cache_key = self.generate_cache_key(query, params)
        
        # Cache in all layers
        self.set_app_cache(cache_key, data, ttl)
        
        if hasattr(self, 'redis_client'):
            try:
                self.redis_client.setex(cache_key, ttl, json.dumps(data))
            except Exception as e:
                print(f"Redis cache set error: {e}")
        
        if hasattr(self, 'memcached_client'):
            try:
                self.memcached_client.set(cache_key, data, time=ttl)
            except Exception as e:
                print(f"Memcached set error: {e}")
    
    def set_app_cache(self, key: str, data: Any, ttl: int = 3600):
        """Set application-level cache with LRU eviction"""
        import time
        
        # Implement simple LRU eviction
        if len(self.app_cache) >= self.app_cache_max_size:
            # Remove oldest entry
            oldest_key = min(self.app_cache.keys(), 
                           key=lambda k: self.app_cache[k]['timestamp'])
            del self.app_cache[oldest_key]
        
        self.app_cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'expires': time.time() + ttl
        }
    
    def invalidate_cache(self, pattern: str = None):
        """Invalidate cache entries by pattern"""
        if pattern:
            # Redis pattern deletion
            if hasattr(self, 'redis_client'):
                for key in self.redis_client.scan_iter(match=pattern):
                    self.redis_client.delete(key)
            
            # Application cache pattern deletion
            keys_to_delete = [k for k in self.app_cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self.app_cache[key]
        else:
            # Clear all caches
            if hasattr(self, 'redis_client'):
                self.redis_client.flushdb()
            if hasattr(self, 'memcached_client'):
                self.memcached_client.flush_all()
            self.app_cache.clear()

# Database query caching decorator
def cached_query(ttl: int = 3600, cache_manager: CacheManager = None):
    """Decorator for caching database query results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if cache_manager is None:
                return await func(*args, **kwargs)
            
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Try to get from cache
            cached_result = await cache_manager.get_cached_result(cache_key, ())
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.cache_result(cache_key, (), result, ttl)
            
            return result
        return wrapper
    return decorator

# Query result caching for specific database operations
class CachedDatabaseOperations:
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    @cached_query(ttl=3600)  # Cache for 1 hour
    async def get_user_by_id(self, user_id: int):
        # Database query implementation
        pass
    
    @cached_query(ttl=300)   # Cache for 5 minutes
    async def get_active_users_count(self):
        # Database query implementation
        pass
    
    @cached_query(ttl=1800)  # Cache for 30 minutes
    async def get_popular_products(self, limit: int = 10):
        # Database query implementation
        pass
```

## 5. Monitoring and Alerting

### Comprehensive Database Monitoring
```python
# Database monitoring and alerting system
import psutil
import time
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MimeText

class DatabaseMonitoring:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_thresholds = config['alert_thresholds']
        self.metrics_history = []
        
    def check_database_health(self) -> Dict[str, Any]:
        """Comprehensive database health check"""
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # Connection pool status
        connection_check = self.check_connection_pool()
        health_status['checks']['connections'] = connection_check
        
        # Query performance
        query_check = self.check_query_performance()
        health_status['checks']['query_performance'] = query_check
        
        # Resource utilization
        resource_check = self.check_resource_utilization()
        health_status['checks']['resources'] = resource_check
        
        # Lock detection
        lock_check = self.check_for_locks()
        health_status['checks']['locks'] = lock_check
        
        # Replication lag (if applicable)
        if self.config.get('check_replication'):
            replication_check = self.check_replication_lag()
            health_status['checks']['replication'] = replication_check
        
        # Determine overall status
        if any(check['status'] == 'critical' for check in health_status['checks'].values()):
            health_status['overall_status'] = 'critical'
        elif any(check['status'] == 'warning' for check in health_status['checks'].values()):
            health_status['overall_status'] = 'warning'
        
        return health_status
    
    def check_connection_pool(self) -> Dict[str, Any]:
        """Check database connection pool status"""
        # This would connect to your specific database
        # and check connection metrics
        
        try:
            # Example for PostgreSQL
            if self.config['db_type'] == 'postgresql':
                active_connections = self.get_postgresql_active_connections()
                max_connections = self.config.get('max_connections', 100)
                
                connection_usage = (active_connections / max_connections) * 100
                
                if connection_usage > self.alert_thresholds['connection_usage_critical']:
                    status = 'critical'
                    message = f"Connection usage critical: {connection_usage:.1f}%"
                elif connection_usage > self.alert_thresholds['connection_usage_warning']:
                    status = 'warning'  
                    message = f"Connection usage high: {connection_usage:.1f}%"
                else:
                    status = 'healthy'
                    message = f"Connection usage normal: {connection_usage:.1f}%"
                
                return {
                    'status': status,
                    'message': message,
                    'metrics': {
                        'active_connections': active_connections,
                        'max_connections': max_connections,
                        'usage_percent': connection_usage
                    }
                }
        except Exception as e:
            return {
                'status': 'critical',
                'message': f"Failed to check connections: {str(e)}",
                'metrics': {}
            }
    
    def check_query_performance(self) -> Dict[str, Any]:
        """Check for slow queries and performance issues"""
        try:
            slow_queries = self.get_slow_queries()
            avg_query_time = self.get_average_query_time()
            
            if avg_query_time > self.alert_thresholds['avg_query_time_critical']:
                status = 'critical'
                message = f"Average query time critical: {avg_query_time:.2f}s"
            elif avg_query_time > self.alert_thresholds['avg_query_time_warning']:
                status = 'warning'
                message = f"Average query time high: {avg_query_time:.2f}s"
            elif len(slow_queries) > self.alert_thresholds['slow_query_count']:
                status = 'warning'
                message = f"Too many slow queries: {len(slow_queries)}"
            else:
                status = 'healthy'
                message = f"Query performance normal: {avg_query_time:.2f}s avg"
            
            return {
                'status': status,
                'message': message,
                'metrics': {
                    'slow_query_count': len(slow_queries),
                    'avg_query_time_seconds': avg_query_time,
                    'slow_queries': slow_queries[:5]  # Top 5 slow queries
                }
            }
        except Exception as e:
            return {
                'status': 'critical',
                'message': f"Failed to check query performance: {str(e)}",
                'metrics': {}
            }
    
    def check_resource_utilization(self) -> Dict[str, Any]:
        """Check CPU, memory, and disk utilization"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check CPU
            if cpu_percent > self.alert_thresholds['cpu_critical']:
                status = 'critical'
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent > self.alert_thresholds['cpu_warning']:
                status = 'warning'
                message = f"CPU usage high: {cpu_percent:.1f}%"
            # Check Memory
            elif memory.percent > self.alert_thresholds['memory_critical']:
                status = 'critical'
                message = f"Memory usage critical: {memory.percent:.1f}%"
            elif memory.percent > self.alert_thresholds['memory_warning']:
                status = 'warning'
                message = f"Memory usage high: {memory.percent:.1f}%"
            # Check Disk
            elif disk.percent > self.alert_thresholds['disk_critical']:
                status = 'critical'
                message = f"Disk usage critical: {disk.percent:.1f}%"
            elif disk.percent > self.alert_thresholds['disk_warning']:
                status = 'warning'
                message = f"Disk usage high: {disk.percent:.1f}%"
            else:
                status = 'healthy'
                message = "Resource utilization normal"
            
            return {
                'status': status,
                'message': message,
                'metrics': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_free_gb': disk.free / (1024**3)
                }
            }
        except Exception as e:
            return {
                'status': 'critical',
                'message': f"Failed to check resource utilization: {str(e)}",
                'metrics': {}
            }
    
    def send_alert(self, health_status: Dict[str, Any]):
        """Send alert notifications for critical issues"""
        
        if health_status['overall_status'] in ['critical', 'warning']:
            
            # Prepare alert message
            subject = f"Database Alert - {health_status['overall_status'].upper()}"
            
            message_body = f"""
Database Health Alert

Timestamp: {health_status['timestamp']}
Overall Status: {health_status['overall_status'].upper()}
Database: {{database_type}}
Environment: {{environment}}

Issues Detected:
"""
            
            for check_name, check_result in health_status['checks'].items():
                if check_result['status'] in ['critical', 'warning']:
                    message_body += f"- {check_name}: {check_result['message']}\n"
            
            # Send email alert
            if self.config.get('email_alerts_enabled'):
                self.send_email_alert(subject, message_body)
            
            # Send Slack alert
            if self.config.get('slack_alerts_enabled'):
                self.send_slack_alert(subject, message_body)
    
    def send_email_alert(self, subject: str, body: str):
        """Send email alert"""
        try:
            msg = MimeText(body)
            msg['Subject'] = subject
            msg['From'] = self.config['smtp_from']
            msg['To'] = ', '.join(self.config['alert_recipients'])
            
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                if self.config.get('smtp_use_tls'):
                    server.starttls()
                if self.config.get('smtp_username'):
                    server.login(self.config['smtp_username'], self.config['smtp_password'])
                server.send_message(msg)
        except Exception as e:
            print(f"Failed to send email alert: {e}")
    
    def send_slack_alert(self, subject: str, body: str):
        """Send Slack alert"""
        try:
            import requests
            
            slack_message = {
                "text": subject,
                "attachments": [
                    {
                        "color": "danger" if "critical" in subject.lower() else "warning",
                        "fields": [
                            {
                                "title": "Details",
                                "value": body,
                                "short": False
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                self.config['slack_webhook_url'],
                json=slack_message
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")

# Example configuration
monitoring_config = {
    'db_type': '{{database_type}}',
    'alert_thresholds': {
        'connection_usage_warning': 70,
        'connection_usage_critical': 90,
        'avg_query_time_warning': 1.0,
        'avg_query_time_critical': 5.0,
        'slow_query_count': 10,
        'cpu_warning': 80,
        'cpu_critical': 95,
        'memory_warning': 85,
        'memory_critical': 95,
        'disk_warning': 85,
        'disk_critical': 95
    },
    'email_alerts_enabled': True,
    'slack_alerts_enabled': True,
    'smtp_server': 'smtp.company.com',
    'smtp_port': 587,
    'smtp_use_tls': True,
    'smtp_from': 'alerts@company.com',
    'alert_recipients': ['dba@company.com', 'ops@company.com'],
    'slack_webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
}

# Run monitoring
monitor = DatabaseMonitoring(monitoring_config)

# Continuous monitoring loop
def run_monitoring():
    while True:
        try:
            health_status = monitor.check_database_health()
            print(f"Health check: {health_status['overall_status']}")
            
            # Send alerts if needed
            monitor.send_alert(health_status)
            
            # Sleep for monitoring interval
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(30)  # Shorter interval on error
```

## Conclusion

This database performance tuning guide provides:

**Key Features:**
- {{database_type}} specific optimization strategies
- {{workload_pattern}} workload optimizations
- Comprehensive indexing strategies
- Multi-level caching implementation
- Advanced monitoring and alerting

**Performance Benefits:**
- Optimized for {{data_size}} data volumes
- {{performance_goals}} focused improvements
- {{concurrent_users}} user load handling
- Production-ready monitoring

**Production Ready:**
- Automated performance monitoring
- Proactive alerting system
- Query optimization recommendations
- Resource utilization tracking
- Scalable caching architecture