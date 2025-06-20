---
name: data_pipeline_architecture
title: Modern Data Pipeline Architecture
description: Comprehensive data pipeline architecture framework covering ingestion, processing, storage, and analytics for scalable data-driven applications
category: data
tags: [data-pipeline, etl, streaming, batch-processing, data-architecture, analytics]
difficulty: advanced
author: jezweb
version: 1.0.0
arguments:
  - name: data_volume
    description: Expected data volume (small <1GB-day, medium 1GB-100GB-day, large 100GB-10TB-day, massive >10TB-day)
    required: true
  - name: data_sources
    description: Primary data sources (databases, apis, files, streams, mixed)
    required: true
  - name: processing_pattern
    description: Processing pattern (batch, streaming, hybrid, real-time)
    required: true
  - name: cloud_platform
    description: Cloud platform (aws, azure, gcp, on-premise, multi-cloud)
    required: true
  - name: data_destinations
    description: Data destinations (data-warehouse, data-lake, analytics-db, operational-stores)
    required: true
  - name: compliance_requirements
    description: Compliance requirements (none, gdpr, hipaa, sox, pci-dss, comprehensive)
    required: true
---

# Data Pipeline Architecture: {{processing_pattern}} Processing

**Data Volume:** {{data_volume}}  
**Data Sources:** {{data_sources}}  
**Cloud Platform:** {{cloud_platform}}  
**Destinations:** {{data_destinations}}  
**Compliance:** {{compliance_requirements}}

## 1. Architecture Overview & Design Patterns

### High-Level Architecture Design
```yaml
# Data Pipeline Architecture Blueprint
architecture:
  ingestion_layer:
    description: "Data collection from various sources"
    components:
      {{#if (contains data_sources "apis")}}
      - api_connectors:
          type: "REST/GraphQL API adapters"
          rate_limiting: true
          retry_mechanisms: exponential_backoff
          authentication: oauth2_jwt
      {{/if}}
      {{#if (contains data_sources "databases")}}
      - database_connectors:
          type: "CDC (Change Data Capture)"
          supported_dbs: [postgresql, mysql, mongodb, oracle]
          replication_methods: [log_based, timestamp_based]
      {{/if}}
      {{#if (contains data_sources "files")}}
      - file_processors:
          formats: [csv, json, parquet, avro, xml]
          compression: [gzip, snappy, lz4]
          validation: schema_enforcement
      {{/if}}
      {{#if (contains data_sources "streams")}}
      - stream_ingestion:
          platforms: [kafka, kinesis, pubsub, eventbridge]
          protocols: [mqtt, websocket, sse]
          buffering: adaptive_batching
      {{/if}}
  
  processing_layer:
    description: "Data transformation and enrichment"
    {{#if (eq processing_pattern "batch")}}
    batch_processing:
      scheduler: airflow_or_prefect
      frameworks: [spark, dbt, pandas]
      compute: {{#if (eq cloud_platform "aws")}}emr_or_glue{{else if (eq cloud_platform "azure")}}databricks_or_synapse{{else if (eq cloud_platform "gcp")}}dataproc_or_dataflow{{else}}kubernetes_spark{{/if}}
      optimization: [partitioning, caching, columnar_storage]
    {{else if (eq processing_pattern "streaming")}}
    stream_processing:
      frameworks: [kafka_streams, flink, storm, kinesis_analytics]
      windowing: [tumbling, sliding, session]
      state_management: distributed_checkpointing
      backpressure_handling: adaptive_rate_limiting
    {{else if (eq processing_pattern "hybrid")}}
    hybrid_processing:
      batch_component: spark_with_delta_lake
      streaming_component: kafka_streams_or_flink
      lambda_architecture: false  # Prefer Kappa architecture
      unified_api: true
    {{else}}
    real_time_processing:
      latency_target: "<100ms"
      frameworks: [flink, storm, ksqldb]
      memory_computing: redis_or_hazelcast
      edge_processing: true
    {{/if}}
  
  storage_layer:
    description: "Persistent data storage systems"
    {{#if (contains data_destinations "data-lake")}}
    data_lake:
      platform: {{#if (eq cloud_platform "aws")}}s3_with_lake_formation{{else if (eq cloud_platform "azure")}}adls_gen2{{else if (eq cloud_platform "gcp")}}cloud_storage_with_dataproc{{else}}hdfs_or_minio{{/if}}
      format: delta_lake_or_iceberg
      partitioning: date_and_category_based
      lifecycle_management: intelligent_tiering
    {{/if}}
    {{#if (contains data_destinations "data-warehouse")}}
    data_warehouse:
      platform: {{#if (eq cloud_platform "aws")}}redshift_or_snowflake{{else if (eq cloud_platform "azure")}}synapse_analytics{{else if (eq cloud_platform "gcp")}}bigquery{{else}}postgresql_with_columnar{{/if}}
      modeling: star_schema_with_scd
      optimization: [materialized_views, clustering, compression]
    {{/if}}
    {{#if (contains data_destinations "analytics-db")}}
    analytics_databases:
      olap: {{#if (eq cloud_platform "aws")}}redshift_or_athena{{else if (eq cloud_platform "azure")}}azure_analysis_services{{else if (eq cloud_platform "gcp")}}bigquery{{else}}clickhouse_or_druid{{/if}}
      time_series: influxdb_or_timestream
      search: elasticsearch_or_opensearch
    {{/if}}
  
  serving_layer:
    description: "Data access and consumption"
    components:
      - rest_apis: graphql_and_rest_endpoints
      - query_engines: [presto, trino, spark_sql]
      - caching: [redis, memcached, application_cache]
      - cdn: cloudfront_or_cloudflare
  
  orchestration:
    workflow_management: {{#if (eq data_volume "massive")}}airflow_with_kubernetes{{else}}airflow_or_prefect{{/if}}
    monitoring: datadog_or_prometheus
    alerting: pagerduty_or_slack
    lineage_tracking: apache_atlas_or_datahub
  
  security_governance:
    {{#if (eq compliance_requirements "comprehensive" "gdpr" "hipaa")}}
    encryption:
      at_rest: aes_256_with_cmk
      in_transit: tls_1_3
      key_management: {{#if (eq cloud_platform "aws")}}kms{{else if (eq cloud_platform "azure")}}key_vault{{else if (eq cloud_platform "gcp")}}cloud_kms{{else}}vault{{/if}}
    access_control:
      authentication: saml_sso_or_oauth2
      authorization: rbac_with_abac
      audit_logging: comprehensive_tracking
    data_governance:
      privacy: automatic_pii_detection
      retention: policy_based_lifecycle
      quality: automated_validation_rules
    {{/if}}
```

### Technology Stack Selection
```python
# Data pipeline technology stack configuration
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class DataVolumeScale(Enum):
    SMALL = "small"
    MEDIUM = "medium" 
    LARGE = "large"
    MASSIVE = "massive"

class ProcessingPattern(Enum):
    BATCH = "batch"
    STREAMING = "streaming"
    HYBRID = "hybrid"
    REAL_TIME = "real-time"

@dataclass
class PipelineArchitecture:
    data_volume: DataVolumeScale
    processing_pattern: ProcessingPattern
    cloud_platform: str
    compliance_level: str
    
    def get_recommended_stack(self) -> Dict[str, Any]:
        """Get recommended technology stack based on requirements"""
        
        stack = {
            'ingestion': self._get_ingestion_stack(),
            'processing': self._get_processing_stack(),
            'storage': self._get_storage_stack(),
            'orchestration': self._get_orchestration_stack(),
            'monitoring': self._get_monitoring_stack()
        }
        
        return stack
    
    def _get_ingestion_stack(self) -> Dict[str, Any]:
        """Select ingestion technologies"""
        
        base_stack = {
            'message_queue': self._get_message_queue(),
            'change_data_capture': self._get_cdc_tool(),
            'file_processing': self._get_file_processor(),
            'api_connectors': self._get_api_connectors()
        }
        
        if self.data_volume in [DataVolumeScale.LARGE, DataVolumeScale.MASSIVE]:
            base_stack['load_balancing'] = 'kafka_cluster_with_partitioning'
            base_stack['buffering'] = 'distributed_buffer_with_spillover'
        
        return base_stack
    
    def _get_message_queue(self) -> str:
        """Select message queue based on platform and scale"""
        
        if self.cloud_platform == "aws":
            if self.data_volume == DataVolumeScale.MASSIVE:
                return "msk_kafka_cluster"
            else:
                return "kinesis_data_streams"
        elif self.cloud_platform == "azure":
            return "event_hubs"
        elif self.cloud_platform == "gcp":
            return "pub_sub"
        else:
            return "apache_kafka"
    
    def _get_processing_stack(self) -> Dict[str, Any]:
        """Select processing framework"""
        
        if self.processing_pattern == ProcessingPattern.BATCH:
            return {
                'framework': self._get_batch_framework(),
                'scheduler': self._get_scheduler(),
                'compute_engine': self._get_compute_engine()
            }
        elif self.processing_pattern == ProcessingPattern.STREAMING:
            return {
                'framework': self._get_streaming_framework(),
                'state_store': self._get_state_store(),
                'windowing': 'flexible_windowing_support'
            }
        elif self.processing_pattern == ProcessingPattern.HYBRID:
            return {
                'batch_framework': self._get_batch_framework(),
                'streaming_framework': self._get_streaming_framework(),
                'unified_engine': 'spark_structured_streaming',
                'storage_format': 'delta_lake_or_iceberg'
            }
        else:  # REAL_TIME
            return {
                'framework': 'apache_flink_or_storm',
                'memory_grid': 'hazelcast_or_ignite',
                'latency_optimization': 'jvm_tuning_and_gc_optimization'
            }
    
    def _get_storage_stack(self) -> Dict[str, Any]:
        """Select storage technologies"""
        
        storage_stack = {}
        
        # Data Lake
        if "{{data_destinations}}" in ["data-lake", "mixed"]:
            if self.cloud_platform == "aws":
                storage_stack['data_lake'] = {
                    'storage': 's3',
                    'catalog': 'glue_catalog',
                    'format': 'delta_lake_or_parquet',
                    'governance': 'lake_formation'
                }
            elif self.cloud_platform == "azure":
                storage_stack['data_lake'] = {
                    'storage': 'adls_gen2',
                    'catalog': 'azure_purview',
                    'format': 'delta_lake_or_parquet'
                }
            elif self.cloud_platform == "gcp":
                storage_stack['data_lake'] = {
                    'storage': 'cloud_storage',
                    'catalog': 'data_catalog',
                    'format': 'parquet_or_avro'
                }
        
        # Data Warehouse
        if "{{data_destinations}}" in ["data-warehouse", "mixed"]:
            storage_stack['data_warehouse'] = self._get_warehouse_solution()
        
        # Analytics Databases
        storage_stack['analytics'] = self._get_analytics_databases()
        
        return storage_stack
    
    def _get_warehouse_solution(self) -> str:
        """Select data warehouse solution"""
        
        if self.cloud_platform == "aws":
            if self.data_volume == DataVolumeScale.MASSIVE:
                return "redshift_ra3_or_snowflake"
            else:
                return "redshift_or_athena"
        elif self.cloud_platform == "azure":
            return "synapse_analytics_dedicated_pool"
        elif self.cloud_platform == "gcp":
            return "bigquery"
        else:
            return "postgresql_with_citus_or_clickhouse"
    
    def _get_monitoring_stack(self) -> Dict[str, Any]:
        """Select monitoring and observability tools"""
        
        base_monitoring = {
            'metrics': 'prometheus_with_grafana',
            'logging': 'elk_stack_or_fluentd',
            'tracing': 'jaeger_or_zipkin',
            'alerting': 'alertmanager_or_pagerduty'
        }
        
        if self.compliance_level in ["gdpr", "hipaa", "comprehensive"]:
            base_monitoring.update({
                'audit_logging': 'comprehensive_audit_trail',
                'data_lineage': 'apache_atlas_or_datahub',
                'privacy_monitoring': 'automated_pii_detection'
            })
        
        if self.data_volume == DataVolumeScale.MASSIVE:
            base_monitoring.update({
                'distributed_tracing': 'jaeger_with_sampling',
                'log_aggregation': 'elasticsearch_cluster',
                'metrics_storage': 'prometheus_federation'
            })
        
        return base_monitoring

# Example usage
architecture = PipelineArchitecture(
    data_volume=DataVolumeScale.{{data_volume.upper()}},
    processing_pattern=ProcessingPattern.{{processing_pattern.upper().replace("-", "_")}},
    cloud_platform="{{cloud_platform}}",
    compliance_level="{{compliance_requirements}}"
)

recommended_stack = architecture.get_recommended_stack()
print("Recommended Technology Stack:")
for layer, technologies in recommended_stack.items():
    print(f"\n{layer.title()}:")
    if isinstance(technologies, dict):
        for key, value in technologies.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {technologies}")
```

## 2. Data Ingestion Layer

### Multi-Source Data Ingestion Framework
```python
# Comprehensive data ingestion framework
import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from sqlalchemy import create_engine
import boto3
from kafka import KafkaProducer, KafkaConsumer
import redis

@dataclass
class DataSource:
    name: str
    type: str  # 'database', 'api', 'file', 'stream'
    connection_config: Dict[str, Any]
    extraction_config: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[str] = None  # Cron expression
    enabled: bool = True

@dataclass
class IngestionMetrics:
    source_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    records_processed: int = 0
    bytes_processed: int = 0
    errors_count: int = 0
    success_rate: float = 0.0

class DataIngestionConnector(ABC):
    """Abstract base class for data connectors"""
    
    def __init__(self, source: DataSource):
        self.source = source
        self.logger = logging.getLogger(f"connector.{source.name}")
        self.metrics = IngestionMetrics(source_name=source.name, start_time=datetime.now())
    
    @abstractmethod
    async def extract_data(self) -> List[Dict[str, Any]]:
        """Extract data from source"""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate connection to data source"""
        pass
    
    def update_metrics(self, records: int, bytes_size: int, errors: int = 0):
        """Update ingestion metrics"""
        self.metrics.records_processed += records
        self.metrics.bytes_processed += bytes_size
        self.metrics.errors_count += errors
        
        if self.metrics.records_processed > 0:
            self.metrics.success_rate = 1 - (self.metrics.errors_count / self.metrics.records_processed)

{{#if (contains data_sources "databases")}}
class DatabaseConnector(DataIngestionConnector):
    """Database connector with CDC support"""
    
    def __init__(self, source: DataSource):
        super().__init__(source)
        self.engine = create_engine(source.connection_config['connection_string'])
        self.last_extracted_timestamp = self._get_last_checkpoint()
    
    async def extract_data(self) -> List[Dict[str, Any]]:
        """Extract data from database using CDC or timestamp-based approach"""
        
        extraction_type = self.source.extraction_config.get('type', 'full')
        
        if extraction_type == 'incremental':
            return await self._extract_incremental()
        elif extraction_type == 'cdc':
            return await self._extract_cdc()
        else:
            return await self._extract_full()
    
    async def _extract_incremental(self) -> List[Dict[str, Any]]:
        """Extract data incrementally based on timestamp"""
        
        table = self.source.extraction_config['table']
        timestamp_column = self.source.extraction_config['timestamp_column']
        batch_size = self.source.extraction_config.get('batch_size', {{#if (eq data_volume "massive")}}50000{{else if (eq data_volume "large")}}25000{{else}}10000{{/if}})
        
        query = f"""
        SELECT * FROM {table} 
        WHERE {timestamp_column} > %s 
        ORDER BY {timestamp_column} 
        LIMIT {batch_size}
        """
        
        try:
            df = pd.read_sql(query, self.engine, params=[self.last_extracted_timestamp])
            records = df.to_dict('records')
            
            if records:
                # Update checkpoint
                latest_timestamp = max(record[timestamp_column] for record in records)
                self._save_checkpoint(latest_timestamp)
                self.last_extracted_timestamp = latest_timestamp
            
            self.update_metrics(len(records), df.memory_usage(deep=True).sum())
            self.logger.info(f"Extracted {len(records)} records from {table}")
            
            return records
            
        except Exception as e:
            self.logger.error(f"Error extracting from {table}: {e}")
            self.update_metrics(0, 0, 1)
            return []
    
    async def _extract_cdc(self) -> List[Dict[str, Any]]:
        """Extract using Change Data Capture"""
        # Implementation depends on database type (PostgreSQL WAL, MySQL binlog, etc.)
        
        cdc_config = self.source.extraction_config.get('cdc', {})
        
        if 'postgresql' in self.source.connection_config['connection_string']:
            return await self._extract_postgresql_wal()
        elif 'mysql' in self.source.connection_config['connection_string']:
            return await self._extract_mysql_binlog()
        else:
            # Fallback to timestamp-based extraction
            return await self._extract_incremental()
    
    async def validate_connection(self) -> bool:
        """Validate database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            return False
    
    def _get_last_checkpoint(self) -> datetime:
        """Get last extraction checkpoint"""
        # Implementation to retrieve from checkpoint store (Redis, database, etc.)
        checkpoint_store = redis.Redis.from_url(
            self.source.extraction_config.get('checkpoint_store', 'redis://localhost:6379')
        )
        
        try:
            checkpoint = checkpoint_store.get(f"checkpoint:{self.source.name}")
            if checkpoint:
                return datetime.fromisoformat(checkpoint.decode())
        except Exception:
            pass
        
        # Default to 24 hours ago
        return datetime.now() - timedelta(hours=24)
    
    def _save_checkpoint(self, timestamp: datetime):
        """Save extraction checkpoint"""
        checkpoint_store = redis.Redis.from_url(
            self.source.extraction_config.get('checkpoint_store', 'redis://localhost:6379')
        )
        
        try:
            checkpoint_store.set(f"checkpoint:{self.source.name}", timestamp.isoformat())
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
{{/if}}

{{#if (contains data_sources "apis")}}
class APIConnector(DataIngestionConnector):
    """REST/GraphQL API connector with rate limiting and retry"""
    
    def __init__(self, source: DataSource):
        super().__init__(source)
        self.session = None
        self.rate_limiter = self._setup_rate_limiter()
    
    async def extract_data(self) -> List[Dict[str, Any]]:
        """Extract data from REST API with pagination"""
        
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=self._get_auth_headers()
            )
        
        all_data = []
        page = 1
        
        while True:
            await self.rate_limiter.acquire()
            
            try:
                batch = await self._fetch_page(page)
                if not batch:
                    break
                
                all_data.extend(batch)
                self.update_metrics(len(batch), len(json.dumps(batch).encode()))
                
                page += 1
                
                # Check if we've hit the batch limit
                batch_size = self.source.extraction_config.get('batch_size', {{#if (eq data_volume "massive")}}10000{{else}}5000{{/if}})
                if len(all_data) >= batch_size:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                self.update_metrics(0, 0, 1)
                break
        
        return all_data
    
    async def _fetch_page(self, page: int) -> List[Dict[str, Any]]:
        """Fetch single page from API"""
        
        url = self.source.connection_config['base_url']
        endpoint = self.source.extraction_config['endpoint']
        
        params = {
            'page': page,
            'per_page': self.source.extraction_config.get('page_size', 100),
            **self.source.extraction_config.get('query_params', {})
        }
        
        async with self.session.get(f"{url}{endpoint}", params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                # Handle different API response formats
                data_key = self.source.extraction_config.get('data_key', 'data')
                if data_key in data:
                    return data[data_key]
                elif isinstance(data, list):
                    return data
                else:
                    return [data]
            elif response.status == 429:  # Rate limited
                retry_after = int(response.headers.get('Retry-After', 60))
                await asyncio.sleep(retry_after)
                return await self._fetch_page(page)  # Retry
            else:
                response.raise_for_status()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        auth_config = self.source.connection_config.get('auth', {})
        
        if auth_config.get('type') == 'bearer':
            return {'Authorization': f"Bearer {auth_config['token']}"}
        elif auth_config.get('type') == 'api_key':
            return {auth_config['header_name']: auth_config['api_key']}
        else:
            return {}
    
    def _setup_rate_limiter(self):
        """Setup rate limiting"""
        import asyncio
        
        class RateLimiter:
            def __init__(self, calls_per_second: float):
                self.calls_per_second = calls_per_second
                self.last_call = 0
            
            async def acquire(self):
                now = asyncio.get_event_loop().time()
                time_since_last = now - self.last_call
                min_interval = 1.0 / self.calls_per_second
                
                if time_since_last < min_interval:
                    await asyncio.sleep(min_interval - time_since_last)
                
                self.last_call = asyncio.get_event_loop().time()
        
        rate_limit = self.source.extraction_config.get('rate_limit', 10)  # calls per second
        return RateLimiter(rate_limit)
    
    async def validate_connection(self) -> bool:
        """Validate API connection"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers=self._get_auth_headers()
                )
            
            health_endpoint = self.source.connection_config.get('health_endpoint', '/health')
            url = f"{self.source.connection_config['base_url']}{health_endpoint}"
            
            async with self.session.get(url) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error(f"API connection validation failed: {e}")
            return False
{{/if}}

{{#if (contains data_sources "streams")}}
class StreamConnector(DataIngestionConnector):
    """Stream data connector for Kafka, Kinesis, etc."""
    
    def __init__(self, source: DataSource):
        super().__init__(source)
        self.consumer = self._setup_consumer()
        self.buffer = []
        self.buffer_size = self.source.extraction_config.get('buffer_size', {{#if (eq data_volume "massive")}}10000{{else}}5000{{/if}})
    
    def _setup_consumer(self):
        """Setup stream consumer"""
        
        stream_type = self.source.connection_config.get('type', 'kafka')
        
        if stream_type == 'kafka':
            return KafkaConsumer(
                self.source.extraction_config['topic'],
                bootstrap_servers=self.source.connection_config['brokers'],
                group_id=self.source.extraction_config.get('consumer_group', 'pipeline_consumer'),
                auto_offset_reset='latest',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                max_poll_records={{#if (eq data_volume "massive")}}5000{{else}}1000{{/if}}
            )
        elif stream_type == 'kinesis':
            # Setup Kinesis consumer
            return self._setup_kinesis_consumer()
        else:
            raise ValueError(f"Unsupported stream type: {stream_type}")
    
    async def extract_data(self) -> List[Dict[str, Any]]:
        """Extract data from stream"""
        
        batch_timeout = self.source.extraction_config.get('batch_timeout_seconds', 30)
        start_time = datetime.now()
        
        while len(self.buffer) < self.buffer_size:
            # Check timeout
            if (datetime.now() - start_time).seconds > batch_timeout:
                break
            
            # Poll for messages
            message_batch = self.consumer.poll(timeout_ms=1000)
            
            for topic_partition, messages in message_batch.items():
                for message in messages:
                    self.buffer.append(message.value)
                    
                    if len(self.buffer) >= self.buffer_size:
                        break
        
        # Return buffered data and clear buffer
        result = self.buffer[:self.buffer_size]
        self.buffer = self.buffer[self.buffer_size:]
        
        self.update_metrics(len(result), len(json.dumps(result).encode()))
        
        return result
    
    async def validate_connection(self) -> bool:
        """Validate stream connection"""
        try:
            # For Kafka, check if we can list topics
            if hasattr(self.consumer, 'list_consumer_groups'):
                self.consumer.list_consumer_groups()
            return True
        except Exception as e:
            self.logger.error(f"Stream connection validation failed: {e}")
            return False
{{/if}}

class IngestionOrchestrator:
    """Orchestrates data ingestion from multiple sources"""
    
    def __init__(self, sources: List[DataSource]):
        self.sources = sources
        self.connectors = self._initialize_connectors()
        self.metrics_store = redis.Redis.from_url('redis://localhost:6379')
        
    def _initialize_connectors(self) -> Dict[str, DataIngestionConnector]:
        """Initialize connectors for each source"""
        
        connectors = {}
        
        for source in self.sources:
            if not source.enabled:
                continue
                
            if source.type == 'database':
                {{#if (contains data_sources "databases")}}
                connectors[source.name] = DatabaseConnector(source)
                {{/if}}
            elif source.type == 'api':
                {{#if (contains data_sources "apis")}}
                connectors[source.name] = APIConnector(source)
                {{/if}}
            elif source.type == 'stream':
                {{#if (contains data_sources "streams")}}
                connectors[source.name] = StreamConnector(source)
                {{/if}}
        
        return connectors
    
    async def run_ingestion(self, source_names: Optional[List[str]] = None) -> Dict[str, IngestionMetrics]:
        """Run data ingestion for specified sources"""
        
        if source_names is None:
            source_names = list(self.connectors.keys())
        
        results = {}
        
        # Run ingestion tasks concurrently
        tasks = []
        for source_name in source_names:
            if source_name in self.connectors:
                task = self._run_single_ingestion(source_name)
                tasks.append(task)
        
        ingestion_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(ingestion_results):
            source_name = source_names[i]
            if isinstance(result, Exception):
                logging.error(f"Ingestion failed for {source_name}: {result}")
                # Create error metrics
                error_metrics = IngestionMetrics(
                    source_name=source_name,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    errors_count=1
                )
                results[source_name] = error_metrics
            else:
                results[source_name] = result
        
        # Store metrics
        await self._store_metrics(results)
        
        return results
    
    async def _run_single_ingestion(self, source_name: str) -> IngestionMetrics:
        """Run ingestion for a single source"""
        
        connector = self.connectors[source_name]
        
        # Validate connection first
        if not await connector.validate_connection():
            raise Exception(f"Connection validation failed for {source_name}")
        
        # Extract data
        data = await connector.extract_data()
        
        # Send to processing layer (implement based on your architecture)
        await self._send_to_processing(source_name, data)
        
        # Finalize metrics
        connector.metrics.end_time = datetime.now()
        
        return connector.metrics
    
    async def _send_to_processing(self, source_name: str, data: List[Dict[str, Any]]):
        """Send extracted data to processing layer"""
        
        {{#if (eq processing_pattern "streaming" "hybrid")}}
        # Send to Kafka/Kinesis for stream processing
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        topic = f"raw_data_{source_name}"
        
        for record in data:
            producer.send(topic, record)
        
        producer.flush()
        {{else}}
        # Store in staging area for batch processing
        staging_location = f"s3://data-lake/staging/{source_name}/{datetime.now().strftime('%Y/%m/%d/%H')}"
        
        # Implementation depends on storage backend
        # For now, just log the data transfer
        logging.info(f"Staged {len(data)} records from {source_name} to {staging_location}")
        {{/if}}
    
    async def _store_metrics(self, metrics: Dict[str, IngestionMetrics]):
        """Store ingestion metrics for monitoring"""
        
        for source_name, metric in metrics.items():
            metric_data = {
                'timestamp': metric.start_time.isoformat(),
                'duration_seconds': (metric.end_time - metric.start_time).seconds if metric.end_time else 0,
                'records_processed': metric.records_processed,
                'bytes_processed': metric.bytes_processed,
                'errors_count': metric.errors_count,
                'success_rate': metric.success_rate
            }
            
            # Store in Redis with TTL
            self.metrics_store.setex(
                f"ingestion_metrics:{source_name}:{metric.start_time.isoformat()}",
                86400,  # 24 hours TTL
                json.dumps(metric_data)
            )

# Example usage
async def main():
    # Define data sources
    sources = [
        {{#if (contains data_sources "databases")}}
        DataSource(
            name="user_database",
            type="database",
            connection_config={
                "connection_string": "postgresql://user:pass@localhost:5432/app_db"
            },
            extraction_config={
                "type": "incremental",
                "table": "users",
                "timestamp_column": "updated_at",
                "batch_size": {{#if (eq data_volume "massive")}}50000{{else}}10000{{/if}}
            }
        ),
        {{/if}}
        {{#if (contains data_sources "apis")}}
        DataSource(
            name="external_api",
            type="api",
            connection_config={
                "base_url": "https://api.example.com",
                "auth": {
                    "type": "bearer",
                    "token": "your_api_token"
                }
            },
            extraction_config={
                "endpoint": "/users",
                "page_size": 100,
                "rate_limit": 10,  # requests per second
                "batch_size": {{#if (eq data_volume "massive")}}10000{{else}}5000{{/if}}
            }
        )
        {{/if}}
    ]
    
    # Initialize orchestrator
    orchestrator = IngestionOrchestrator(sources)
    
    # Run ingestion
    results = await orchestrator.run_ingestion()
    
    # Print results
    for source_name, metrics in results.items():
        print(f"{source_name}: {metrics.records_processed} records, {metrics.success_rate:.2%} success rate")

if __name__ == "__main__":
    asyncio.run(main())
```

## 3. Data Processing Engine

### {{processing_pattern}} Processing Implementation
```python
{{#if (eq processing_pattern "batch")}}
# Batch processing framework using Apache Spark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

class BatchDataProcessor:
    def __init__(self, app_name: str = "DataPipeline"):
        self.spark = self._create_spark_session(app_name)
        self.logger = logging.getLogger(__name__)
        
    def _create_spark_session(self, app_name: str) -> SparkSession:
        """Create optimized Spark session"""
        
        builder = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        
        {{#if (eq data_volume "massive")}}
        # Optimizations for massive data volumes
        builder = builder \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "256MB") \
            .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "10") \
            .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
        {{else if (eq data_volume "large")}}
        # Optimizations for large data volumes
        builder = builder \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB") \
            .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "5")
        {{/if}}
        
        {{#if (eq cloud_platform "aws")}}
        # AWS S3 optimizations
        builder = builder \
            .config("spark.hadoop.fs.s3a.multipart.size", "104857600") \
            .config("spark.hadoop.fs.s3a.fast.upload", "true") \
            .config("spark.hadoop.fs.s3a.block.size", "134217728")
        {{/if}}
        
        return builder.getOrCreate()
    
    def process_incremental_data(self, 
                                source_path: str, 
                                target_path: str, 
                                transformations: List[Dict[str, Any]],
                                partition_columns: List[str] = None) -> Dict[str, Any]:
        """Process data incrementally with Delta Lake"""
        
        start_time = datetime.now()
        
        try:
            # Read source data
            source_df = self._read_data(source_path)
            
            # Apply transformations
            processed_df = self._apply_transformations(source_df, transformations)
            
            # Write with Delta Lake for ACID transactions
            if partition_columns:
                processed_df.write \
                    .format("delta") \
                    .mode("append") \
                    .partitionBy(*partition_columns) \
                    .option("mergeSchema", "true") \
                    .save(target_path)
            else:
                processed_df.write \
                    .format("delta") \
                    .mode("append") \
                    .save(target_path)
            
            # Optimize Delta table
            self.spark.sql(f"OPTIMIZE delta.`{target_path}`")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'status': 'success',
                'records_processed': processed_df.count(),
                'processing_time_seconds': processing_time,
                'target_path': target_path
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'processing_time_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    def _read_data(self, source_path: str) -> DataFrame:
        """Read data from various sources"""
        
        if source_path.endswith('.parquet'):
            return self.spark.read.parquet(source_path)
        elif source_path.endswith('.json'):
            return self.spark.read.json(source_path)
        elif source_path.endswith('.csv'):
            return self.spark.read.option("header", "true").csv(source_path)
        elif 'delta' in source_path:
            return self.spark.read.format("delta").load(source_path)
        else:
            # Try to auto-detect format
            return self.spark.read.option("inferSchema", "true").load(source_path)
    
    def _apply_transformations(self, df: DataFrame, transformations: List[Dict[str, Any]]) -> DataFrame:
        """Apply series of transformations"""
        
        current_df = df
        
        for transform in transformations:
            transform_type = transform.get('type')
            
            if transform_type == 'filter':
                current_df = current_df.filter(transform['condition'])
            
            elif transform_type == 'select':
                current_df = current_df.select(*transform['columns'])
            
            elif transform_type == 'rename':
                for old_name, new_name in transform['mapping'].items():
                    current_df = current_df.withColumnRenamed(old_name, new_name)
            
            elif transform_type == 'add_column':
                current_df = current_df.withColumn(
                    transform['column_name'], 
                    expr(transform['expression'])
                )
            
            elif transform_type == 'aggregate':
                group_cols = transform.get('group_by', [])
                agg_exprs = transform['aggregations']
                
                if group_cols:
                    current_df = current_df.groupBy(*group_cols).agg(*[
                        expr(agg_expr) for agg_expr in agg_exprs
                    ])
                else:
                    current_df = current_df.agg(*[
                        expr(agg_expr) for agg_expr in agg_exprs
                    ])
            
            elif transform_type == 'join':
                right_df = self._read_data(transform['right_path'])
                join_condition = transform['condition']
                join_type = transform.get('join_type', 'inner')
                
                current_df = current_df.join(right_df, expr(join_condition), join_type)
            
            elif transform_type == 'window':
                from pyspark.sql.window import Window
                
                window_spec = Window.partitionBy(*transform['partition_by'])
                
                if 'order_by' in transform:
                    window_spec = window_spec.orderBy(*transform['order_by'])
                
                current_df = current_df.withColumn(
                    transform['column_name'],
                    expr(transform['expression']).over(window_spec)
                )
            
            elif transform_type == 'deduplicate':
                subset_cols = transform.get('subset', None)
                current_df = current_df.dropDuplicates(subset_cols)
            
            elif transform_type == 'data_quality':
                # Apply data quality rules
                current_df = self._apply_data_quality_rules(current_df, transform['rules'])
        
        return current_df
    
    def _apply_data_quality_rules(self, df: DataFrame, rules: List[Dict[str, Any]]) -> DataFrame:
        """Apply data quality validation and cleansing rules"""
        
        current_df = df
        
        for rule in rules:
            rule_type = rule.get('type')
            
            if rule_type == 'not_null':
                columns = rule['columns']
                for col in columns:
                    current_df = current_df.filter(col_is_not_null(col))
            
            elif rule_type == 'range_check':
                column = rule['column']
                min_val = rule.get('min')
                max_val = rule.get('max')
                
                if min_val is not None:
                    current_df = current_df.filter(col(column) >= min_val)
                if max_val is not None:
                    current_df = current_df.filter(col(column) <= max_val)
            
            elif rule_type == 'format_validation':
                column = rule['column']
                pattern = rule['pattern']
                current_df = current_df.filter(col(column).rlike(pattern))
            
            elif rule_type == 'outlier_removal':
                column = rule['column']
                method = rule.get('method', 'iqr')
                
                if method == 'iqr':
                    quantiles = current_df.approxQuantile(column, [0.25, 0.75], 0.05)
                    q1, q3 = quantiles[0], quantiles[1]
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    current_df = current_df.filter(
                        (col(column) >= lower_bound) & (col(column) <= upper_bound)
                    )
        
        return current_df

# Example batch processing job
def run_batch_job():
    processor = BatchDataProcessor("UserDataPipeline")
    
    # Define transformations
    transformations = [
        {
            'type': 'filter',
            'condition': 'status != "deleted"'
        },
        {
            'type': 'add_column',
            'column_name': 'processed_date',
            'expression': 'current_date()'
        },
        {
            'type': 'data_quality',
            'rules': [
                {
                    'type': 'not_null',
                    'columns': ['user_id', 'email']
                },
                {
                    'type': 'format_validation',
                    'column': 'email',
                    'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                }
            ]
        },
        {
            'type': 'aggregate',
            'group_by': ['date', 'status'],
            'aggregations': [
                'count(*) as user_count',
                'countDistinct(user_id) as unique_users'
            ]
        }
    ]
    
    # Process data
    result = processor.process_incremental_data(
        source_path="s3a://data-lake/raw/users/",
        target_path="s3a://data-lake/processed/users/",
        transformations=transformations,
        partition_columns=['year', 'month', 'day']
    )
    
    print(f"Processing result: {result}")

if __name__ == "__main__":
    run_batch_job()
{{/if}}

{{#if (eq processing_pattern "streaming")}}
# Streaming processing framework using Apache Flink
from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.table import StreamTableEnvironment, DataTypes
from pyflink.table.descriptors import Schema, Kafka, Json
import json
from typing import Dict, Any, List
import logging

class StreamDataProcessor:
    def __init__(self, app_name: str = "StreamPipeline"):
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
        
        # Configure for high throughput
        {{#if (eq data_volume "massive")}}
        self.env.set_parallelism(16)
        self.env.get_config().set_auto_watermark_interval(1000)
        {{else if (eq data_volume "large")}}
        self.env.set_parallelism(8)
        self.env.get_config().set_auto_watermark_interval(5000)
        {{else}}
        self.env.set_parallelism(4)
        {{/if}}
        
        self.table_env = StreamTableEnvironment.create(self.env)
        self.logger = logging.getLogger(__name__)
    
    def setup_kafka_source(self, 
                          topic: str, 
                          bootstrap_servers: str = "localhost:9092",
                          group_id: str = "stream_processor") -> str:
        """Setup Kafka source table"""
        
        source_ddl = f"""
        CREATE TABLE kafka_source (
            user_id STRING,
            event_type STRING,
            timestamp BIGINT,
            properties MAP<STRING, STRING>,
            event_time AS TO_TIMESTAMP(FROM_UNIXTIME(timestamp)),
            WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = '{topic}',
            'properties.bootstrap.servers' = '{bootstrap_servers}',
            'properties.group.id' = '{group_id}',
            'format' = 'json',
            'scan.startup.mode' = 'latest-offset'
        )
        """
        
        self.table_env.execute_sql(source_ddl)
        return "kafka_source"
    
    def setup_sink(self, sink_type: str, **kwargs) -> str:
        """Setup various sink types"""
        
        if sink_type == "kafka":
            sink_ddl = f"""
            CREATE TABLE kafka_sink (
                user_id STRING,
                event_count BIGINT,
                window_start TIMESTAMP(3),
                window_end TIMESTAMP(3)
            ) WITH (
                'connector' = 'kafka',
                'topic' = '{kwargs.get("topic", "processed_events")}',
                'properties.bootstrap.servers' = '{kwargs.get("bootstrap_servers", "localhost:9092")}',
                'format' = 'json'
            )
            """
        
        elif sink_type == "elasticsearch":
            sink_ddl = f"""
            CREATE TABLE elasticsearch_sink (
                user_id STRING,
                event_count BIGINT,
                window_start TIMESTAMP(3),
                window_end TIMESTAMP(3),
                PRIMARY KEY (user_id) NOT ENFORCED
            ) WITH (
                'connector' = 'elasticsearch-7',
                'hosts' = '{kwargs.get("hosts", "http://localhost:9200")}',
                'index' = '{kwargs.get("index", "user_events")}'
            )
            """
        
        elif sink_type == "jdbc":
            sink_ddl = f"""
            CREATE TABLE jdbc_sink (
                user_id STRING,
                event_count BIGINT,
                window_start TIMESTAMP(3),
                window_end TIMESTAMP(3),
                PRIMARY KEY (user_id) NOT ENFORCED
            ) WITH (
                'connector' = 'jdbc',
                'url' = '{kwargs.get("url", "jdbc:postgresql://localhost:5432/analytics")}',
                'table-name' = '{kwargs.get("table", "user_event_counts")}',
                'username' = '{kwargs.get("username", "postgres")}',
                'password' = '{kwargs.get("password", "password")}'
            )
            """
        
        self.table_env.execute_sql(sink_ddl)
        return f"{sink_type}_sink"
    
    def create_tumbling_window_aggregation(self, 
                                         source_table: str,
                                         window_size: str = "1 MINUTE") -> str:
        """Create tumbling window aggregation"""
        
        agg_query = f"""
        SELECT 
            user_id,
            COUNT(*) as event_count,
            TUMBLE_START(event_time, INTERVAL '{window_size}') as window_start,
            TUMBLE_END(event_time, INTERVAL '{window_size}') as window_end
        FROM {source_table}
        GROUP BY 
            user_id,
            TUMBLE(event_time, INTERVAL '{window_size}')
        """
        
        # Create view for the aggregation
        view_name = "windowed_aggregation"
        self.table_env.create_temporary_view(view_name, self.table_env.sql_query(agg_query))
        
        return view_name
    
    def create_session_window_analysis(self, 
                                     source_table: str,
                                     session_gap: str = "10 MINUTE") -> str:
        """Create session window analysis for user behavior"""
        
        session_query = f"""
        SELECT 
            user_id,
            COUNT(*) as events_in_session,
            SESSION_START(event_time, INTERVAL '{session_gap}') as session_start,
            SESSION_END(event_time, INTERVAL '{session_gap}') as session_end,
            COLLECT(event_type) as event_types
        FROM {source_table}
        GROUP BY 
            user_id,
            SESSION(event_time, INTERVAL '{session_gap}')
        """
        
        view_name = "session_analysis"
        self.table_env.create_temporary_view(view_name, self.table_env.sql_query(session_query))
        
        return view_name
    
    def create_pattern_detection(self, source_table: str) -> str:
        """Detect patterns in event streams using CEP"""
        
        # Complex Event Processing for pattern detection
        pattern_query = f"""
        SELECT *
        FROM {source_table}
        MATCH_RECOGNIZE (
            PARTITION BY user_id
            ORDER BY event_time
            MEASURES
                A.event_time as start_time,
                C.event_time as end_time,
                A.user_id as user_id
            PATTERN (A B+ C)
            DEFINE
                A AS A.event_type = 'login',
                B AS B.event_type = 'page_view',
                C AS C.event_type = 'purchase'
        ) AS T
        """
        
        view_name = "purchase_funnel_pattern"
        self.table_env.create_temporary_view(view_name, self.table_env.sql_query(pattern_query))
        
        return view_name
    
    def run_real_time_anomaly_detection(self, source_table: str) -> str:
        """Real-time anomaly detection using statistical methods"""
        
        # Z-score based anomaly detection
        anomaly_query = f"""
        SELECT 
            user_id,
            event_count,
            window_start,
            window_end,
            CASE 
                WHEN ABS(event_count - AVG(event_count) OVER (
                    PARTITION BY user_id 
                    ORDER BY window_start 
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                )) / STDDEV(event_count) OVER (
                    PARTITION BY user_id 
                    ORDER BY window_start 
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                ) > 3 THEN 'ANOMALY'
                ELSE 'NORMAL'
            END as anomaly_status
        FROM {source_table}
        WHERE event_count IS NOT NULL
        """
        
        view_name = "anomaly_detection"
        self.table_env.create_temporary_view(view_name, self.table_env.sql_query(anomaly_query))
        
        return view_name
    
    def setup_monitoring_and_alerting(self):
        """Setup monitoring metrics and alerting"""
        
        # Create metrics table
        metrics_ddl = """
        CREATE TABLE processing_metrics (
            metric_name STRING,
            metric_value DOUBLE,
            timestamp TIMESTAMP(3),
            tags MAP<STRING, STRING>
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'pipeline_metrics',
            'properties.bootstrap.servers' = 'localhost:9092',
            'format' = 'json'
        )
        """
        
        self.table_env.execute_sql(metrics_ddl)
    
    def execute_pipeline(self):
        """Execute the streaming pipeline"""
        
        try:
            # Execute the job
            self.env.execute("StreamingDataPipeline")
        except Exception as e:
            self.logger.error(f"Streaming pipeline failed: {e}")
            raise

# Example streaming job
def run_streaming_job():
    processor = StreamDataProcessor("UserEventStream")
    
    # Setup source
    source_table = processor.setup_kafka_source(
        topic="user_events",
        bootstrap_servers="localhost:9092",
        group_id="analytics_processor"
    )
    
    # Create windowed aggregation
    agg_view = processor.create_tumbling_window_aggregation(
        source_table, 
        window_size="{{#if (eq data_volume "massive")}}30 SECOND{{else}}1 MINUTE{{/if}}"
    )
    
    # Setup sink
    sink_table = processor.setup_sink(
        "elasticsearch",
        hosts="http://localhost:9200",
        index="user_event_metrics"
    )
    
    # Insert aggregated data into sink
    processor.table_env.execute_sql(f"""
        INSERT INTO {sink_table}
        SELECT user_id, event_count, window_start, window_end
        FROM {agg_view}
    """)
    
    # Setup monitoring
    processor.setup_monitoring_and_alerting()
    
    # Execute pipeline
    processor.execute_pipeline()

if __name__ == "__main__":
    run_streaming_job()
{{/if}}

{{#if (eq processing_pattern "hybrid")}}
# Hybrid batch and streaming processing with unified interface
import asyncio
from typing import Dict, Any, Union
from abc import ABC, abstractmethod
import logging

class HybridDataProcessor:
    """Unified interface for batch and streaming processing"""
    
    def __init__(self):
        self.batch_processor = BatchDataProcessor("HybridBatch")
        self.stream_processor = StreamDataProcessor("HybridStream")
        self.logger = logging.getLogger(__name__)
    
    async def process_data(self, 
                          processing_mode: str,
                          source_config: Dict[str, Any],
                          transformations: List[Dict[str, Any]],
                          sink_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process data in batch or streaming mode"""
        
        if processing_mode == "batch":
            return await self._process_batch(source_config, transformations, sink_config)
        elif processing_mode == "streaming":
            return await self._process_stream(source_config, transformations, sink_config)
        else:
            raise ValueError(f"Unsupported processing mode: {processing_mode}")
    
    async def _process_batch(self, source_config, transformations, sink_config):
        """Process data in batch mode"""
        
        return self.batch_processor.process_incremental_data(
            source_path=source_config['path'],
            target_path=sink_config['path'],
            transformations=transformations,
            partition_columns=sink_config.get('partition_columns')
        )
    
    async def _process_stream(self, source_config, transformations, sink_config):
        """Process data in streaming mode"""
        
        # Setup source
        source_table = self.stream_processor.setup_kafka_source(
            topic=source_config['topic'],
            bootstrap_servers=source_config['bootstrap_servers']
        )
        
        # Apply transformations (simplified for example)
        for transform in transformations:
            if transform['type'] == 'window_aggregation':
                agg_view = self.stream_processor.create_tumbling_window_aggregation(
                    source_table,
                    window_size=transform['window_size']
                )
                source_table = agg_view
        
        # Setup sink
        sink_table = self.stream_processor.setup_sink(
            sink_type=sink_config['type'],
            **sink_config['config']
        )
        
        # Execute stream processing
        return {"status": "streaming", "message": "Pipeline started"}

# Example hybrid usage
async def run_hybrid_pipeline():
    processor = HybridDataProcessor()
    
    # Batch processing for historical data
    batch_result = await processor.process_data(
        processing_mode="batch",
        source_config={"path": "s3://data-lake/historical/"},
        transformations=[
            {"type": "filter", "condition": "timestamp > '2024-01-01'"},
            {"type": "aggregate", "group_by": ["date"], "aggregations": ["count(*)"]}
        ],
        sink_config={"path": "s3://data-lake/processed/", "partition_columns": ["year", "month"]}
    )
    
    # Streaming processing for real-time data
    stream_result = await processor.process_data(
        processing_mode="streaming",
        source_config={"topic": "real_time_events", "bootstrap_servers": "localhost:9092"},
        transformations=[
            {"type": "window_aggregation", "window_size": "1 MINUTE"}
        ],
        sink_config={
            "type": "elasticsearch",
            "config": {"hosts": "http://localhost:9200", "index": "real_time_metrics"}
        }
    )
    
    print(f"Batch result: {batch_result}")
    print(f"Stream result: {stream_result}")

if __name__ == "__main__":
    asyncio.run(run_hybrid_pipeline())
{{/if}}
```

## 4. Data Storage & Management

### Multi-Tier Storage Architecture
```yaml
# Data storage tier configuration
storage_tiers:
  hot_tier:
    description: "Frequently accessed data (last 30 days)"
    storage_type: {{#if (eq cloud_platform "aws")}}s3_intelligent_tiering{{else if (eq cloud_platform "azure")}}blob_hot_tier{{else if (eq cloud_platform "gcp")}}standard_storage{{else}}ssd_storage{{/if}}
    access_pattern: "high_frequency"
    retention: "30_days"
    cost_optimization: "performance_optimized"
    
  warm_tier:
    description: "Moderately accessed data (30-365 days)"
    storage_type: {{#if (eq cloud_platform "aws")}}s3_standard_ia{{else if (eq cloud_platform "azure")}}blob_cool_tier{{else if (eq cloud_platform "gcp")}}nearline_storage{{else}}hdd_storage{{/if}}
    access_pattern: "medium_frequency"
    retention: "1_year"
    cost_optimization: "balanced"
    
  cold_tier:
    description: "Rarely accessed data (1+ years)"
    storage_type: {{#if (eq cloud_platform "aws")}}s3_glacier{{else if (eq cloud_platform "azure")}}blob_archive_tier{{else if (eq cloud_platform "gcp")}}coldline_storage{{else}}tape_backup{{/if}}
    access_pattern: "low_frequency"
    retention: "7_years"
    cost_optimization: "cost_optimized"

data_formats:
  raw_data:
    format: "parquet_with_snappy_compression"
    schema_evolution: "enabled"
    partitioning: ["year", "month", "day", "hour"]
    
  processed_data:
    format: "delta_lake_or_iceberg"
    versioning: "enabled"
    time_travel: "30_days"
    acid_transactions: "enabled"
    
  analytics_ready:
    format: "columnar_optimized"
    indexing: "automatic"
    materialized_views: "enabled"
    query_optimization: "advanced"

data_governance:
  {{#if (eq compliance_requirements "gdpr" "hipaa" "comprehensive")}}
  encryption:
    at_rest: "aes_256_with_customer_managed_keys"
    in_transit: "tls_1_3"
    key_rotation: "automatic_quarterly"
    
  access_control:
    authentication: "multi_factor_authentication"
    authorization: "fine_grained_rbac"
    audit_logging: "comprehensive"
    
  privacy:
    pii_detection: "automatic"
    data_masking: "dynamic"
    right_to_be_forgotten: "automated_deletion"
    consent_management: "integrated"
  {{/if}}
  
  quality_management:
    schema_validation: "strict"
    data_profiling: "continuous"
    anomaly_detection: "ml_based"
    lineage_tracking: "end_to_end"
```

### Storage Implementation
```python
# Data storage management system
import boto3
import pandas as pd
from delta import *
from pyspark.sql import SparkSession
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import json

class DataStorageManager:
    def __init__(self, cloud_platform: str = "{{cloud_platform}}"):
        self.cloud_platform = cloud_platform
        self.spark = self._setup_spark()
        self.storage_client = self._setup_storage_client()
        self.logger = logging.getLogger(__name__)
        
    def _setup_spark(self) -> SparkSession:
        """Setup Spark session with Delta Lake support"""
        
        builder = SparkSession.builder \
            .appName("DataStorageManager") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        
        {{#if (eq cloud_platform "aws")}}
        # AWS S3 configuration
        builder = builder \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        {{/if}}
        
        return configure_spark_with_delta_pip(builder).getOrCreate()
    
    def _setup_storage_client(self):
        """Setup cloud storage client"""
        
        if self.cloud_platform == "aws":
            return boto3.client('s3')
        elif self.cloud_platform == "azure":
            from azure.storage.blob import BlobServiceClient
            return BlobServiceClient.from_connection_string("your_connection_string")
        elif self.cloud_platform == "gcp":
            from google.cloud import storage
            return storage.Client()
        else:
            return None
    
    def store_raw_data(self, 
                      data: pd.DataFrame, 
                      dataset_name: str,
                      partition_columns: List[str] = None) -> str:
        """Store raw data with partitioning and compression"""
        
        timestamp = datetime.now()
        base_path = f"{{#if (eq cloud_platform "aws")}}s3a://data-lake{{else if (eq cloud_platform "azure")}}abfss://datalake{{else}}gs://data-lake{{/if}}/raw/{dataset_name}"
        
        # Convert pandas DataFrame to Spark DataFrame
        spark_df = self.spark.createDataFrame(data)
        
        # Add processing metadata
        spark_df = spark_df \
            .withColumn("_ingestion_timestamp", lit(timestamp)) \
            .withColumn("_partition_date", date_format(col("_ingestion_timestamp"), "yyyy-MM-dd"))
        
        # Determine partitioning strategy
        if partition_columns is None:
            partition_columns = ["_partition_date"]
        
        # Write with optimizations
        writer = spark_df.write \
            .format("parquet") \
            .mode("append") \
            .option("compression", "snappy") \
            .partitionBy(*partition_columns)
        
        {{#if (eq data_volume "massive")}}
        # Optimizations for massive data
        writer = writer \
            .option("maxRecordsPerFile", 100000) \
            .option("spark.sql.adaptive.coalescePartitions.enabled", "true")
        {{/if}}
        
        full_path = f"{base_path}/{timestamp.strftime('%Y/%m/%d/%H')}"
        writer.save(full_path)
        
        self.logger.info(f"Stored {data.shape[0]} records to {full_path}")
        return full_path
    
    def store_processed_data(self, 
                           data: pd.DataFrame,
                           table_name: str,
                           operation: str = "append") -> str:
        """Store processed data using Delta Lake for ACID transactions"""
        
        delta_path = f"{{#if (eq cloud_platform "aws")}}s3a://data-lake{{else if (eq cloud_platform "azure")}}abfss://datalake{{else}}gs://data-lake{{/if}}/processed/{table_name}"
        
        # Convert to Spark DataFrame
        spark_df = self.spark.createDataFrame(data)
        
        # Add processing metadata
        spark_df = spark_df \
            .withColumn("_processed_timestamp", current_timestamp()) \
            .withColumn("_data_version", lit("1.0"))
        
        if operation == "append":
            spark_df.write \
                .format("delta") \
                .mode("append") \
                .option("mergeSchema", "true") \
                .save(delta_path)
                
        elif operation == "upsert":
            # Perform upsert operation
            self._perform_delta_upsert(spark_df, delta_path, table_name)
        
        elif operation == "overwrite":
            spark_df.write \
                .format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(delta_path)
        
        # Optimize Delta table
        self._optimize_delta_table(delta_path)
        
        return delta_path
    
    def _perform_delta_upsert(self, new_data_df, delta_path: str, table_name: str):
        """Perform upsert operation using Delta Lake merge"""
        
        from delta.tables import DeltaTable
        
        # Create Delta table if it doesn't exist
        if not DeltaTable.isDeltaTable(self.spark, delta_path):
            new_data_df.write.format("delta").save(delta_path)
            return
        
        # Load existing Delta table
        delta_table = DeltaTable.forPath(self.spark, delta_path)
        
        # Determine merge condition (assuming 'id' is the primary key)
        merge_condition = "existing.id = updates.id"
        
        # Perform merge operation
        delta_table.alias("existing") \
            .merge(new_data_df.alias("updates"), merge_condition) \
            .whenMatchedUpdateAll() \
            .whenNotMatchedInsertAll() \
            .execute()
        
        self.logger.info(f"Completed upsert operation for {table_name}")
    
    def _optimize_delta_table(self, delta_path: str):
        """Optimize Delta table for better query performance"""
        
        # Run OPTIMIZE command
        self.spark.sql(f"OPTIMIZE delta.`{delta_path}`")
        
        # Vacuum old files (keep 7 days of history)
        self.spark.sql(f"VACUUM delta.`{delta_path}` RETAIN 168 HOURS")
        
        self.logger.info(f"Optimized Delta table at {delta_path}")
    
    def setup_data_lifecycle_management(self):
        """Setup automated data lifecycle management"""
        
        lifecycle_policies = {
            "raw_data": {
                "hot_to_warm": 30,  # days
                "warm_to_cold": 365,  # days
                "delete_after": 2555  # 7 years
            },
            "processed_data": {
                "hot_to_warm": 90,
                "warm_to_cold": 730,
                "delete_after": 2555
            }
        }
        
        {{#if (eq cloud_platform "aws")}}
        # AWS S3 lifecycle configuration
        for data_type, policy in lifecycle_policies.items():
            lifecycle_config = {
                'Rules': [
                    {
                        'ID': f'{data_type}_lifecycle',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': f'{data_type}/'},
                        'Transitions': [
                            {
                                'Days': policy['hot_to_warm'],
                                'StorageClass': 'STANDARD_IA'
                            },
                            {
                                'Days': policy['warm_to_cold'],
                                'StorageClass': 'GLACIER'
                            }
                        ],
                        'Expiration': {'Days': policy['delete_after']}
                    }
                ]
            }
            
            try:
                self.storage_client.put_bucket_lifecycle_configuration(
                    Bucket='data-lake',
                    LifecycleConfiguration=lifecycle_config
                )
                self.logger.info(f"Applied lifecycle policy for {data_type}")
            except Exception as e:
                self.logger.error(f"Failed to apply lifecycle policy: {e}")
        {{/if}}
    
    def setup_data_governance(self):
        """Setup data governance and compliance controls"""
        
        {{#if (eq compliance_requirements "gdpr" "hipaa" "comprehensive")}}
        # Setup encryption policies
        encryption_config = {
            "default_encryption": {
                "algorithm": "AES256",
                "key_management": "customer_managed"
            },
            "sensitive_data_encryption": {
                "algorithm": "AES256",
                "key_rotation": "quarterly"
            }
        }
        
        # Setup access control policies
        access_policies = {
            "data_scientists": {
                "read_access": ["processed", "analytics"],
                "write_access": [],
                "pii_access": False
            },
            "analysts": {
                "read_access": ["analytics"],
                "write_access": [],
                "pii_access": False
            },
            "administrators": {
                "read_access": ["raw", "processed", "analytics"],
                "write_access": ["processed", "analytics"],
                "pii_access": True
            }
        }
        
        # Apply policies (implementation depends on cloud provider)
        self._apply_governance_policies(encryption_config, access_policies)
        {{/if}}
    
    def monitor_storage_usage(self) -> Dict[str, Any]:
        """Monitor storage usage and costs"""
        
        metrics = {
            "total_size_gb": 0,
            "monthly_cost": 0,
            "tier_breakdown": {},
            "growth_rate": 0
        }
        
        {{#if (eq cloud_platform "aws")}}
        # Get S3 storage metrics
        cloudwatch = boto3.client('cloudwatch')
        
        try:
            # Get bucket size metrics
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/S3',
                MetricName='BucketSizeBytes',
                Dimensions=[
                    {'Name': 'BucketName', 'Value': 'data-lake'},
                    {'Name': 'StorageType', 'Value': 'StandardStorage'}
                ],
                StartTime=datetime.now() - timedelta(days=1),
                EndTime=datetime.now(),
                Period=86400,
                Statistics=['Average']
            )
            
            if response['Datapoints']:
                size_bytes = response['Datapoints'][-1]['Average']
                metrics['total_size_gb'] = size_bytes / (1024**3)
        
        except Exception as e:
            self.logger.error(f"Failed to get storage metrics: {e}")
        {{/if}}
        
        return metrics
    
    def create_analytics_views(self):
        """Create optimized views for analytics workloads"""
        
        # Create materialized views for common queries
        analytics_views = [
            {
                "name": "daily_user_metrics",
                "query": """
                SELECT 
                    date,
                    COUNT(DISTINCT user_id) as daily_active_users,
                    COUNT(*) as total_events,
                    AVG(session_duration) as avg_session_duration
                FROM processed.user_events
                WHERE date >= current_date() - INTERVAL 90 DAYS
                GROUP BY date
                ORDER BY date DESC
                """,
                "refresh_schedule": "daily"
            },
            {
                "name": "product_performance",
                "query": """
                SELECT 
                    product_id,
                    product_name,
                    SUM(revenue) as total_revenue,
                    COUNT(DISTINCT user_id) as unique_customers,
                    AVG(rating) as avg_rating
                FROM processed.transactions t
                JOIN processed.products p ON t.product_id = p.id
                WHERE t.date >= current_date() - INTERVAL 30 DAYS
                GROUP BY product_id, product_name
                ORDER BY total_revenue DESC
                """,
                "refresh_schedule": "hourly"
            }
        ]
        
        for view in analytics_views:
            try:
                # Create view in the analytics database
                self.spark.sql(f"""
                CREATE OR REPLACE VIEW analytics.{view['name']} AS
                {view['query']}
                """)
                
                self.logger.info(f"Created analytics view: {view['name']}")
                
            except Exception as e:
                self.logger.error(f"Failed to create view {view['name']}: {e}")

# Example usage
def main():
    storage_manager = DataStorageManager()
    
    # Setup governance and lifecycle management
    storage_manager.setup_data_lifecycle_management()
    storage_manager.setup_data_governance()
    
    # Example data storage
    sample_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'event_type': ['login', 'view', 'click', 'purchase', 'logout'],
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='H')
    })
    
    # Store raw data
    raw_path = storage_manager.store_raw_data(
        data=sample_data,
        dataset_name="user_events",
        partition_columns=["event_type"]
    )
    
    # Store processed data
    processed_path = storage_manager.store_processed_data(
        data=sample_data,
        table_name="user_events_processed",
        operation="append"
    )
    
    # Create analytics views
    storage_manager.create_analytics_views()
    
    # Monitor usage
    metrics = storage_manager.monitor_storage_usage()
    print(f"Storage metrics: {metrics}")

if __name__ == "__main__":
    main()
```

## Conclusion

This Data Pipeline Architecture provides:

**Key Features:**
- {{processing_pattern}} processing for {{data_volume}} data volumes
- Multi-source ingestion from {{data_sources}}
- {{cloud_platform}} optimized storage and compute
- {{compliance_requirements}} compliance and governance

**Benefits:**
- Scalable architecture supporting {{data_volume}} data processing
- Real-time and batch processing capabilities
- Cost-optimized storage tiering
- Comprehensive data governance and lineage tracking

**Architecture Patterns:**
- Modern data lakehouse architecture
- Event-driven processing pipeline
- ACID transactions with Delta Lake/Iceberg
- Automated data lifecycle management

**Success Metrics:**
- Processing latency under target SLAs
- 99.9% data pipeline availability
- Cost optimization through intelligent tiering
- Comprehensive data governance and compliance