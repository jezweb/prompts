---
name: cloud_migration_strategy
title: Cloud Migration Strategy Framework
description: Comprehensive cloud migration planning and execution framework covering assessment, strategy selection, migration patterns, and post-migration optimization
category: devops
tags: [cloud-migration, aws, azure, gcp, strategy, assessment, migration-patterns]
difficulty: advanced
author: jezweb
version: 1.0.0
arguments:
  - name: source_environment
    description: Current environment (on-premise, hybrid, legacy-cloud, multi-cloud)
    required: true
  - name: target_cloud
    description: Target cloud provider (aws, azure, gcp, multi-cloud)
    required: true
  - name: application_portfolio
    description: Application types (legacy-monolith, microservices, mixed, saas-applications)
    required: true
  - name: migration_timeline
    description: Timeline constraints (aggressive 6-12months, standard 12-18months, gradual 18-36months)
    required: true
  - name: business_drivers
    description: Primary drivers (cost-reduction, scalability, modernization, compliance)
    required: true
  - name: organization_size
    description: Organization size (startup, mid-market, enterprise, large-enterprise)
    required: true
---

# Cloud Migration Strategy: {{source_environment}} to {{target_cloud}}

**Application Portfolio:** {{application_portfolio}}  
**Timeline:** {{migration_timeline}}  
**Business Drivers:** {{business_drivers}}  
**Organization:** {{organization_size}}

## 1. Migration Assessment & Planning

### Current State Assessment
```yaml
# Comprehensive assessment framework
assessment_categories:
  infrastructure:
    servers: "Physical/Virtual server inventory"
    storage: "Storage systems and capacity analysis"
    networking: "Network architecture and dependencies"
    security: "Current security controls and compliance"
    
  applications:
    inventory: "Complete application catalog"
    dependencies: "Inter-application dependencies mapping"
    architecture: "Technical architecture documentation"
    performance: "Current performance baselines"
    
  data:
    databases: "Database systems and sizes"
    storage_requirements: "Data storage and backup needs"
    compliance: "Data governance and regulatory requirements"
    integration: "Data flow and integration patterns"
    
  operations:
    monitoring: "Current monitoring and alerting systems"
    backup_recovery: "Backup and disaster recovery procedures"
    automation: "Existing automation and tooling"
    processes: "IT operations processes and procedures"
```

### Discovery and Analysis Tools
```python
# Automated discovery and assessment script
import boto3
import json
from datetime import datetime, timedelta
import subprocess
import psutil

class CloudMigrationAssessment:
    def __init__(self, target_cloud="{{target_cloud}}"):
        self.target_cloud = target_cloud
        self.assessment_data = {
            'timestamp': datetime.now().isoformat(),
            'infrastructure': {},
            'applications': {},
            'network': {},
            'security': {},
            'compliance': {}
        }
    
    def discover_infrastructure(self):
        """Discover current infrastructure components"""
        
        # Server inventory
        servers = self.get_server_inventory()
        
        # Storage analysis
        storage = self.get_storage_analysis()
        
        # Network discovery
        network = self.get_network_topology()
        
        self.assessment_data['infrastructure'] = {
            'servers': servers,
            'storage': storage,
            'network': network,
            'total_servers': len(servers),
            'total_storage_tb': sum(s.get('storage_gb', 0) for s in servers) / 1024
        }
        
        return self.assessment_data['infrastructure']
    
    def get_server_inventory(self):
        """Get detailed server inventory"""
        servers = []
        
        {{#if (eq source_environment "on-premise")}}
        # On-premise server discovery
        try:
            # Example using system information
            server_info = {
                'hostname': subprocess.check_output(['hostname']).decode().strip(),
                'os': subprocess.check_output(['uname', '-a']).decode().strip(),
                'cpu_cores': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3)),
                'storage_gb': round(psutil.disk_usage('/').total / (1024**3)),
                'network_interfaces': list(psutil.net_if_addrs().keys()),
                'running_services': self.get_running_services(),
                'migration_complexity': self.assess_migration_complexity()
            }
            servers.append(server_info)
        except Exception as e:
            print(f"Server discovery error: {e}")
        {{/if}}
        
        {{#if (eq source_environment "legacy-cloud")}}
        # Legacy cloud environment discovery
        # This would integrate with existing cloud provider APIs
        servers = self.discover_cloud_instances()
        {{/if}}
        
        return servers
    
    def assess_migration_complexity(self):
        """Assess migration complexity for each component"""
        complexity_factors = {
            'os_compatibility': self.check_os_compatibility(),
            'application_dependencies': self.analyze_dependencies(),
            'data_size': self.estimate_data_transfer_time(),
            'compliance_requirements': self.check_compliance_needs(),
            'custom_configurations': self.identify_custom_configs()
        }
        
        # Calculate overall complexity score (1-10)
        scores = [factor.get('score', 5) for factor in complexity_factors.values()]
        complexity_score = sum(scores) / len(scores)
        
        if complexity_score <= 3:
            return {'level': 'Low', 'score': complexity_score, 'factors': complexity_factors}
        elif complexity_score <= 6:
            return {'level': 'Medium', 'score': complexity_score, 'factors': complexity_factors}
        else:
            return {'level': 'High', 'score': complexity_score, 'factors': complexity_factors}
    
    def generate_migration_recommendations(self):
        """Generate migration strategy recommendations"""
        
        recommendations = {
            'migration_strategy': self.recommend_migration_strategy(),
            'migration_patterns': self.recommend_migration_patterns(),
            'timeline_estimate': self.estimate_migration_timeline(),
            'cost_estimate': self.estimate_migration_costs(),
            'risk_assessment': self.assess_migration_risks()
        }
        
        return recommendations
    
    def recommend_migration_strategy(self):
        """Recommend optimal migration strategy"""
        
        {{#if (includes business_drivers "cost-reduction")}}
        if self.assessment_data['infrastructure']['total_servers'] > 100:
            return {
                'strategy': 'Phased Migration',
                'approach': 'Lift-and-shift followed by optimization',
                'rationale': 'Large scale requires gradual approach to minimize risk and disruption'
            }
        {{/if}}
        
        {{#if (includes business_drivers "modernization")}}
        return {
            'strategy': 'Modernization-First',
            'approach': 'Refactor and re-architect applications during migration',
            'rationale': 'Modernization driver justifies additional effort for cloud-native benefits'
        }
        {{/if}}
        
        return {
            'strategy': 'Hybrid Approach',
            'approach': 'Mix of lift-and-shift and selective modernization',
            'rationale': 'Balanced approach optimizing for timeline and business value'
        }

# Cloud provider specific assessment
{{#if (eq target_cloud "aws")}}
class AWSMigrationAssessment(CloudMigrationAssessment):
    def __init__(self):
        super().__init__("aws")
        self.aws_session = boto3.Session()
    
    def calculate_aws_costs(self, server_specs):
        """Calculate AWS costs for migrated infrastructure"""
        
        # EC2 instance recommendations
        ec2_recommendations = []
        for server in server_specs:
            instance_type = self.recommend_ec2_instance(server)
            monthly_cost = self.get_ec2_pricing(instance_type)
            
            ec2_recommendations.append({
                'server': server['hostname'],
                'recommended_instance': instance_type,
                'monthly_cost_usd': monthly_cost,
                'annual_cost_usd': monthly_cost * 12
            })
        
        # Storage costs
        storage_costs = self.calculate_storage_costs(server_specs)
        
        # Network costs
        network_costs = self.estimate_network_costs()
        
        return {
            'ec2_costs': ec2_recommendations,
            'storage_costs': storage_costs,
            'network_costs': network_costs,
            'total_monthly_estimate': sum(r['monthly_cost_usd'] for r in ec2_recommendations) + storage_costs['monthly'] + network_costs['monthly']
        }
    
    def recommend_ec2_instance(self, server_spec):
        """Recommend appropriate EC2 instance type"""
        cpu_cores = server_spec.get('cpu_cores', 2)
        memory_gb = server_spec.get('memory_gb', 8)
        
        # Instance type mapping logic
        if cpu_cores <= 2 and memory_gb <= 8:
            return 't3.large'
        elif cpu_cores <= 4 and memory_gb <= 16:
            return 'm5.xlarge'
        elif cpu_cores <= 8 and memory_gb <= 32:
            return 'm5.2xlarge'
        else:
            return 'm5.4xlarge'
{{/if}}

{{#if (eq target_cloud "azure")}}
class AzureMigrationAssessment(CloudMigrationAssessment):
    def __init__(self):
        super().__init__("azure")
    
    def calculate_azure_costs(self, server_specs):
        """Calculate Azure costs for migrated infrastructure"""
        
        vm_recommendations = []
        for server in server_specs:
            vm_size = self.recommend_vm_size(server)
            monthly_cost = self.get_vm_pricing(vm_size)
            
            vm_recommendations.append({
                'server': server['hostname'],
                'recommended_vm': vm_size,
                'monthly_cost_usd': monthly_cost
            })
        
        return {
            'vm_costs': vm_recommendations,
            'total_monthly_estimate': sum(r['monthly_cost_usd'] for r in vm_recommendations)
        }
    
    def recommend_vm_size(self, server_spec):
        """Recommend appropriate Azure VM size"""
        cpu_cores = server_spec.get('cpu_cores', 2)
        memory_gb = server_spec.get('memory_gb', 8)
        
        if cpu_cores <= 2 and memory_gb <= 8:
            return 'Standard_D2s_v3'
        elif cpu_cores <= 4 and memory_gb <= 16:
            return 'Standard_D4s_v3'
        else:
            return 'Standard_D8s_v3'
{{/if}}
```

## 2. Migration Strategies & Patterns

### The 7 Rs Migration Framework
```markdown
# Migration Strategy Selection

## 1. Retire 🗑️
**When to Use**: Legacy applications with no business value
**Effort**: Low
**Timeline**: Immediate
**Cost Savings**: High
- Decommission unused or redundant applications
- Archive data if required for compliance
- Eliminate licensing and maintenance costs

## 2. Retain 🏠
**When to Use**: Applications not ready for migration
**Effort**: None
**Timeline**: Defer
**Risk**: Low
- Keep on-premises temporarily
- Plan for future migration
- Maintain current operational model

## 3. Rehost (Lift & Shift) 🚚
**When to Use**: {{#if (eq migration_timeline "aggressive")}}Quick migration timeline, minimal disruption{{else}}Simple applications with no architecture changes needed{{/if}}
**Effort**: Low-Medium
**Timeline**: Fast
**Cost**: Low migration cost, potential ongoing optimization
- Move applications to cloud with minimal changes
- Use VM-based hosting initially
- Optimize post-migration

## 4. Relocate (Hypervisor-level lift) 🔄
**When to Use**: VMware environments moving to cloud
**Effort**: Low
**Timeline**: Fast
**Benefits**: Minimal application changes
- Use VMware Cloud on AWS/Azure VMware Solution
- Maintain existing tools and processes
- Gradual transition to cloud-native services

## 5. Repurchase (Drop & Shop) 🛒
**When to Use**: Commercial software with SaaS alternatives
**Effort**: Medium
**Timeline**: Medium
**ROI**: High operational efficiency
- Replace with SaaS solutions
- Eliminate maintenance overhead
- Access to latest features

## 6. Replatform (Lift, Tinker & Shift) 🔧
**When to Use**: {{#if (includes business_drivers "modernization")}}Applications that benefit from cloud services{{else}}Gradual modernization approach{{/if}}
**Effort**: Medium
**Timeline**: Medium
**Benefits**: Some cloud optimization
- Make minor optimizations during migration
- Use managed databases, load balancers
- Improve scalability and reliability

## 7. Refactor (Re-architect) 🏗️
**When to Use**: {{#if (includes business_drivers "scalability")}}Applications requiring significant scalability{{else}}Strategic applications for competitive advantage{{/if}}
**Effort**: High
**Timeline**: Long
**ROI**: Highest long-term value
- Complete application redesign
- Cloud-native architecture
- Microservices, containers, serverless
```

### Migration Wave Planning
```python
# Migration wave planning and dependency analysis
from typing import List, Dict, Set
import networkx as nx

class MigrationWavePlanner:
    def __init__(self, applications: List[Dict]):
        self.applications = {app['id']: app for app in applications}
        self.dependency_graph = nx.DiGraph()
        self.waves = []
    
    def build_dependency_graph(self):
        """Build application dependency graph"""
        
        for app_id, app in self.applications.items():
            self.dependency_graph.add_node(app_id, **app)
            
            # Add dependencies
            for dep in app.get('dependencies', []):
                if dep in self.applications:
                    self.dependency_graph.add_edge(dep, app_id)
    
    def plan_migration_waves(self) -> List[List[str]]:
        """Plan migration waves based on dependencies"""
        
        waves = []
        remaining_apps = set(self.applications.keys())
        
        while remaining_apps:
            # Find applications with no unmigrated dependencies
            wave_candidates = []
            
            for app_id in remaining_apps:
                dependencies = set(self.dependency_graph.predecessors(app_id))
                unmigrated_deps = dependencies.intersection(remaining_apps)
                
                if not unmigrated_deps:
                    wave_candidates.append(app_id)
            
            if not wave_candidates:
                # Handle circular dependencies
                wave_candidates = [min(remaining_apps)]  # Pick one to break cycle
            
            # Sort by migration complexity and business priority
            wave_candidates.sort(key=lambda x: (
                self.applications[x].get('complexity', 5),
                -self.applications[x].get('business_priority', 5)
            ))
            
            # Limit wave size based on timeline and resources
            max_wave_size = self.calculate_max_wave_size()
            current_wave = wave_candidates[:max_wave_size]
            
            waves.append(current_wave)
            remaining_apps -= set(current_wave)
        
        return waves
    
    def calculate_max_wave_size(self) -> int:
        """Calculate maximum wave size based on timeline and resources"""
        
        {{#if (eq migration_timeline "aggressive")}}
        # Aggressive timeline - larger waves
        return min(10, len(self.applications) // 3)
        {{else if (eq migration_timeline "standard")}}
        # Standard timeline - moderate waves
        return min(6, len(self.applications) // 4)
        {{else}}
        # Gradual timeline - smaller waves
        return min(4, len(self.applications) // 6)
        {{/if}}
    
    def generate_wave_plan(self) -> Dict:
        """Generate comprehensive wave plan with timelines"""
        
        waves = self.plan_migration_waves()
        
        wave_plan = {
            'total_waves': len(waves),
            'estimated_duration_months': len(waves) * self.get_wave_duration(),
            'waves': []
        }
        
        for i, wave in enumerate(waves):
            wave_info = {
                'wave_number': i + 1,
                'applications': [
                    {
                        'id': app_id,
                        'name': self.applications[app_id].get('name', app_id),
                        'strategy': self.applications[app_id].get('migration_strategy', 'rehost'),
                        'complexity': self.applications[app_id].get('complexity', 5),
                        'estimated_effort_weeks': self.estimate_migration_effort(app_id)
                    }
                    for app_id in wave
                ],
                'start_month': i * self.get_wave_duration() + 1,
                'duration_months': self.get_wave_duration(),
                'parallel_migrations': len(wave),
                'risk_level': self.assess_wave_risk(wave)
            }
            
            wave_plan['waves'].append(wave_info)
        
        return wave_plan
    
    def get_wave_duration(self) -> int:
        """Get wave duration in months based on timeline"""
        {{#if (eq migration_timeline "aggressive")}}
        return 2  # 2 months per wave
        {{else if (eq migration_timeline "standard")}}
        return 3  # 3 months per wave
        {{else}}
        return 4  # 4 months per wave
        {{/if}}

# Example usage
applications = [
    {
        'id': 'web-frontend',
        'name': 'Web Frontend',
        'dependencies': ['api-gateway', 'user-service'],
        'migration_strategy': 'replatform',
        'complexity': 3,
        'business_priority': 9
    },
    {
        'id': 'api-gateway',
        'name': 'API Gateway',
        'dependencies': ['user-service', 'payment-service'],
        'migration_strategy': 'refactor',
        'complexity': 7,
        'business_priority': 8
    },
    {
        'id': 'user-service',
        'name': 'User Service',
        'dependencies': ['user-database'],
        'migration_strategy': 'replatform',
        'complexity': 5,
        'business_priority': 10
    },
    {
        'id': 'user-database',
        'name': 'User Database',
        'dependencies': [],
        'migration_strategy': 'rehost',
        'complexity': 4,
        'business_priority': 10
    }
]

planner = MigrationWavePlanner(applications)
planner.build_dependency_graph()
wave_plan = planner.generate_wave_plan()
```

## 3. {{application_portfolio}} Migration Approach

{{#if (eq application_portfolio "legacy-monolith")}}
### Legacy Monolith Migration Strategy

#### Decomposition and Migration Pattern
```markdown
# Legacy Monolith Migration Approach

## Phase 1: Assessment and Preparation (Months 1-2)
- [ ] Code analysis and documentation
- [ ] Identify bounded contexts and potential microservices
- [ ] Database dependency mapping
- [ ] Performance baseline establishment
- [ ] Risk assessment and mitigation planning

## Phase 2: Infrastructure Setup (Months 2-3)
- [ ] Set up target cloud environment
- [ ] Implement CI/CD pipelines
- [ ] Configure monitoring and logging
- [ ] Establish security and compliance controls
- [ ] Set up development and testing environments

## Phase 3: Strangler Fig Pattern Implementation (Months 3-8)
- [ ] Implement API gateway for routing
- [ ] Extract peripheral services first (authentication, logging)
- [ ] Gradually extract core business services
- [ ] Migrate database using incremental approach
- [ ] Maintain data consistency during transition

## Phase 4: Core Migration (Months 6-12)
- [ ] Migrate remaining monolith components
- [ ] Refactor tightly coupled modules
- [ ] Implement distributed transaction patterns
- [ ] Complete data migration and synchronization
- [ ] Performance testing and optimization

## Phase 5: Optimization and Cleanup (Months 12-14)
- [ ] Decommission legacy infrastructure
- [ ] Optimize cloud resource usage
- [ ] Implement advanced cloud services
- [ ] Knowledge transfer and documentation
- [ ] Post-migration review and lessons learned
```

#### Strangler Fig Implementation
```python
# Strangler Fig pattern implementation for monolith migration
class StranglerFigMigrator:
    def __init__(self, monolith_config, cloud_config):
        self.monolith_config = monolith_config
        self.cloud_config = cloud_config
        self.migration_routes = {}
        self.feature_flags = {}
    
    def create_api_gateway_routing(self):
        """Set up API gateway for gradual traffic routing"""
        
        routing_rules = {
            'api_gateway_config': {
                'routes': [
                    {
                        'path': '/api/users/*',
                        'destination': 'cloud',  # Migrated service
                        'fallback': 'monolith',
                        'traffic_percentage': 10  # Start with 10% traffic
                    },
                    {
                        'path': '/api/orders/*',
                        'destination': 'monolith',  # Not migrated yet
                        'traffic_percentage': 100
                    },
                    {
                        'path': '/api/payments/*',
                        'destination': 'cloud',
                        'traffic_percentage': 50  # Gradually increasing
                    }
                ],
                'health_checks': {
                    'cloud_services': '/health',
                    'monolith': '/legacy/health'
                },
                'circuit_breaker': {
                    'failure_threshold': 5,
                    'timeout': 30,
                    'fallback_to_monolith': True
                }
            }
        }
        
        return routing_rules
    
    def implement_feature_flags(self):
        """Implement feature flags for gradual migration"""
        
        feature_config = {
            'user_service_migration': {
                'enabled': True,
                'rollout_percentage': 25,
                'user_segments': ['beta_users', 'internal_users']
            },
            'payment_service_migration': {
                'enabled': True,
                'rollout_percentage': 10,
                'user_segments': ['premium_users']
            },
            'reporting_service_migration': {
                'enabled': False,
                'rollout_percentage': 0,
                'planned_date': '2024-06-01'
            }
        }
        
        return feature_config
    
    def plan_data_migration(self):
        """Plan incremental data migration strategy"""
        
        data_migration_plan = {
            'strategy': 'dual_write_eventual_consistency',
            'phases': [
                {
                    'phase': 1,
                    'description': 'Set up data replication',
                    'tables': ['users', 'user_profiles'],
                    'method': 'CDC',  # Change Data Capture
                    'duration_weeks': 2
                },
                {
                    'phase': 2,
                    'description': 'Dual write implementation',
                    'tables': ['orders', 'order_items'],
                    'method': 'application_dual_write',
                    'duration_weeks': 3
                },
                {
                    'phase': 3,
                    'description': 'Data validation and cutover',
                    'tables': ['all'],
                    'method': 'validation_and_switch',
                    'duration_weeks': 2
                }
            ],
            'rollback_plan': {
                'data_backup_strategy': 'point_in_time_recovery',
                'rollback_procedures': 'documented_scripts',
                'testing_frequency': 'weekly'
            }
        }
        
        return data_migration_plan
```
{{/if}}

{{#if (eq application_portfolio "microservices")}}
### Microservices Migration Strategy

#### Container-First Approach
```yaml
# Microservices migration plan
microservices_migration:
  containerization:
    strategy: "Docker containers with Kubernetes orchestration"
    container_registry: "{{target_cloud}} container registry"
    base_images: "Distroless or Alpine-based for security"
    
  service_mesh:
    implementation: "Istio or AWS App Mesh"
    traffic_management: "Canary deployments and blue-green"
    security: "mTLS and policy enforcement"
    observability: "Distributed tracing and metrics"
    
  data_strategy:
    pattern: "Database per service"
    migration_approach: "Service-by-service data extraction"
    consistency: "Eventual consistency with event sourcing"
    
  deployment_pipeline:
    ci_cd: "GitOps with ArgoCD or Flux"
    testing: "Unit, integration, and contract testing"
    security_scanning: "Container and dependency scanning"
    monitoring: "Service-level and business metrics"
```

#### Service-by-Service Migration Plan
```python
# Microservices migration orchestrator
class MicroservicesMigrator:
    def __init__(self, services_config):
        self.services = services_config
        self.migration_order = self.calculate_migration_order()
    
    def calculate_migration_order(self):
        """Calculate optimal service migration order"""
        
        # Prioritize by:
        # 1. Least dependencies (leaf services first)
        # 2. Business criticality
        # 3. Technical complexity
        
        ordered_services = []
        
        for service in self.services:
            priority_score = (
                -len(service.get('dependencies', [])) * 10 +  # Fewer deps = higher priority
                service.get('business_priority', 5) * 5 +      # Business importance
                -service.get('complexity', 5) * 2              # Lower complexity = higher priority
            )
            
            ordered_services.append((service['name'], priority_score))
        
        ordered_services.sort(key=lambda x: x[1], reverse=True)
        return [service[0] for service in ordered_services]
    
    def generate_service_migration_plan(self, service_name):
        """Generate migration plan for individual service"""
        
        service = next(s for s in self.services if s['name'] == service_name)
        
        migration_plan = {
            'service_name': service_name,
            'migration_strategy': service.get('strategy', 'replatform'),
            'estimated_duration_weeks': self.estimate_duration(service),
            'phases': [
                {
                    'phase': 'containerization',
                    'tasks': [
                        'Create Dockerfile',
                        'Set up CI/CD pipeline',
                        'Container security scanning',
                        'Local testing'
                    ],
                    'duration_days': 5
                },
                {
                    'phase': 'cloud_deployment',
                    'tasks': [
                        'Deploy to staging environment',
                        'Configure load balancers',
                        'Set up monitoring and logging',
                        'Performance testing'
                    ],
                    'duration_days': 7
                },
                {
                    'phase': 'data_migration',
                    'tasks': [
                        'Set up database migration',
                        'Data validation',
                        'Backup and rollback procedures',
                        'Gradual traffic routing'
                    ],
                    'duration_days': 10
                },
                {
                    'phase': 'production_cutover',
                    'tasks': [
                        'Blue-green deployment',
                        'Monitor service health',
                        'Full traffic cutover',
                        'Decommission legacy service'
                    ],
                    'duration_days': 3
                }
            ],
            'dependencies': service.get('dependencies', []),
            'risks': self.assess_service_risks(service),
            'rollback_plan': self.create_rollback_plan(service)
        }
        
        return migration_plan
```
{{/if}}

## 4. Cost Optimization & FinOps

### Migration Cost Analysis
```python
# Comprehensive cost analysis for cloud migration
class MigrationCostAnalyzer:
    def __init__(self, current_infrastructure, target_cloud="{{target_cloud}}"):
        self.current_infra = current_infrastructure
        self.target_cloud = target_cloud
        self.cost_models = self.load_cost_models()
    
    def analyze_total_cost_ownership(self, timeline_years=3):
        """Analyze 3-year TCO comparison"""
        
        current_costs = self.calculate_current_costs(timeline_years)
        migration_costs = self.calculate_migration_costs()
        cloud_costs = self.calculate_cloud_costs(timeline_years)
        
        tco_analysis = {
            'current_environment_3yr': current_costs,
            'migration_investment': migration_costs,
            'cloud_environment_3yr': cloud_costs,
            'total_cloud_tco': migration_costs + cloud_costs,
            'net_savings': current_costs - (migration_costs + cloud_costs),
            'break_even_months': self.calculate_break_even(current_costs, migration_costs, cloud_costs, timeline_years),
            'roi_percentage': ((current_costs - (migration_costs + cloud_costs)) / (migration_costs + cloud_costs)) * 100
        }
        
        return tco_analysis
    
    def calculate_migration_costs(self):
        """Calculate one-time migration costs"""
        
        migration_costs = {
            'assessment_and_planning': 50000,  # Professional services
            'migration_tools_and_licenses': 25000,
            'staff_training': 30000,
            'consultant_fees': self.estimate_consultant_costs(),
            'temporary_parallel_running': self.estimate_parallel_costs(),
            'testing_and_validation': 40000,
            'project_management': 60000,
            'contingency_buffer': 0  # 20% buffer
        }
        
        subtotal = sum(migration_costs.values())
        migration_costs['contingency_buffer'] = subtotal * 0.2
        migration_costs['total'] = subtotal + migration_costs['contingency_buffer']
        
        return migration_costs
    
    def estimate_consultant_costs(self):
        """Estimate external consultant costs based on org size and complexity"""
        
        {{#if (eq organization_size "enterprise")}}
        base_cost = 200000  # Large enterprise migration
        {{else if (eq organization_size "mid-market")}}
        base_cost = 100000  # Mid-market migration
        {{else}}
        base_cost = 50000   # Startup migration
        {{/if}}
        
        # Adjust for timeline pressure
        {{#if (eq migration_timeline "aggressive")}}
        base_cost *= 1.5  # Premium for aggressive timeline
        {{/if}}
        
        return base_cost
    
    def calculate_cloud_costs(self, years):
        """Calculate ongoing cloud costs with optimization curve"""
        
        year1_costs = self.estimate_initial_cloud_costs()
        
        # Apply optimization curve
        yearly_costs = []
        for year in range(1, years + 1):
            if year == 1:
                yearly_cost = year1_costs
            elif year == 2:
                # 15% reduction through optimization
                yearly_cost = year1_costs * 0.85
            else:
                # Additional 5% reduction in subsequent years
                yearly_cost = year1_costs * 0.80
            
            yearly_costs.append(yearly_cost)
        
        return {
            'year_1': yearly_costs[0],
            'year_2': yearly_costs[1] if len(yearly_costs) > 1 else 0,
            'year_3': yearly_costs[2] if len(yearly_costs) > 2 else 0,
            'total_3_years': sum(yearly_costs)
        }
    
    def estimate_initial_cloud_costs(self):
        """Estimate first-year cloud costs"""
        
        {{#if (eq target_cloud "aws")}}
        # AWS cost estimation
        compute_costs = self.estimate_ec2_costs()
        storage_costs = self.estimate_s3_ebs_costs()
        network_costs = self.estimate_network_costs()
        managed_services = self.estimate_managed_services_costs()
        
        return {
            'compute': compute_costs,
            'storage': storage_costs,
            'network': network_costs,
            'managed_services': managed_services,
            'support': compute_costs * 0.1,  # 10% for support
            'total_annual': (compute_costs + storage_costs + network_costs + managed_services) * 1.1
        }
        {{/if}}
        
        {{#if (eq target_cloud "azure")}}
        # Azure cost estimation
        vm_costs = self.estimate_azure_vm_costs()
        storage_costs = self.estimate_azure_storage_costs()
        network_costs = self.estimate_azure_network_costs()
        
        return {
            'virtual_machines': vm_costs,
            'storage': storage_costs,
            'network': network_costs,
            'total_annual': (vm_costs + storage_costs + network_costs) * 1.15  # 15% buffer
        }
        {{/if}}
    
    def generate_cost_optimization_recommendations(self):
        """Generate cost optimization recommendations"""
        
        recommendations = []
        
        {{#if (includes business_drivers "cost-reduction")}}
        recommendations.extend([
            {
                'category': 'Right-sizing',
                'recommendation': 'Implement automated right-sizing based on utilization metrics',
                'potential_savings': '15-30%',
                'implementation_effort': 'Medium',
                'timeline': '3-6 months'
            },
            {
                'category': 'Reserved Instances',
                'recommendation': 'Purchase reserved instances for steady-state workloads',
                'potential_savings': '30-60%',
                'implementation_effort': 'Low',
                'timeline': '1 month'
            },
            {
                'category': 'Spot Instances',
                'recommendation': 'Use spot instances for fault-tolerant workloads',
                'potential_savings': '60-90%',
                'implementation_effort': 'High',
                'timeline': '6-12 months'
            }
        ])
        {{/if}}
        
        recommendations.extend([
            {
                'category': 'Storage Optimization',
                'recommendation': 'Implement intelligent storage tiering',
                'potential_savings': '20-40%',
                'implementation_effort': 'Medium',
                'timeline': '2-4 months'
            },
            {
                'category': 'Network Optimization',
                'recommendation': 'Optimize data transfer and CDN usage',
                'potential_savings': '10-25%',
                'implementation_effort': 'Medium',
                'timeline': '3-6 months'
            }
        ])
        
        return recommendations
```

## 5. Risk Management & Mitigation

### Migration Risk Assessment
```yaml
# Comprehensive risk assessment framework
migration_risks:
  technical_risks:
    high:
      - application_compatibility_issues
      - data_corruption_during_migration
      - performance_degradation_post_migration
      - integration_failures
    medium:
      - vendor_lock_in_concerns
      - skill_gaps_in_cloud_technologies
      - security_configuration_errors
    low:
      - minor_feature_differences
      - cosmetic_ui_changes
      
  business_risks:
    high:
      - extended_downtime_during_migration
      - cost_overruns_exceeding_budget
      - project_timeline_delays
      - compliance_violations
    medium:
      - user_adoption_resistance
      - temporary_productivity_loss
      - vendor_relationship_changes
    low:
      - minor_process_adjustments
      
  operational_risks:
    high:
      - inadequate_backup_and_recovery
      - monitoring_and_alerting_gaps
      - runbook_and_documentation_gaps
    medium:
      - staff_turnover_during_migration
      - change_management_resistance
      - training_and_knowledge_transfer
```

### Risk Mitigation Strategies
```python
# Risk mitigation planning and tracking
class MigrationRiskManager:
    def __init__(self):
        self.risk_register = []
        self.mitigation_plans = {}
        self.risk_thresholds = {
            'high': 8,
            'medium': 5,
            'low': 2
        }
    
    def assess_risk(self, risk_description, probability, impact, category):
        """Assess and register a migration risk"""
        
        risk_score = probability * impact  # Scale of 1-10 each
        
        if risk_score >= self.risk_thresholds['high']:
            risk_level = 'high'
        elif risk_score >= self.risk_thresholds['medium']:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        risk = {
            'id': f"RISK_{len(self.risk_register) + 1:03d}",
            'description': risk_description,
            'category': category,
            'probability': probability,
            'impact': impact,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'status': 'identified',
            'mitigation_plan': None,
            'owner': None,
            'due_date': None
        }
        
        self.risk_register.append(risk)
        return risk['id']
    
    def create_mitigation_plan(self, risk_id, mitigation_strategy):
        """Create mitigation plan for identified risk"""
        
        risk = next((r for r in self.risk_register if r['id'] == risk_id), None)
        if not risk:
            return None
        
        mitigation_plan = {
            'risk_id': risk_id,
            'strategy': mitigation_strategy['strategy'],
            'actions': mitigation_strategy['actions'],
            'responsible_party': mitigation_strategy['owner'],
            'target_completion': mitigation_strategy['due_date'],
            'cost_estimate': mitigation_strategy.get('cost', 0),
            'effectiveness_rating': mitigation_strategy.get('effectiveness', 'medium'),
            'contingency_plan': mitigation_strategy.get('contingency', 'None defined')
        }
        
        self.mitigation_plans[risk_id] = mitigation_plan
        risk['mitigation_plan'] = mitigation_plan
        risk['status'] = 'mitigation_planned'
        
        return mitigation_plan
    
    def generate_risk_dashboard(self):
        """Generate risk management dashboard"""
        
        total_risks = len(self.risk_register)
        high_risks = len([r for r in self.risk_register if r['risk_level'] == 'high'])
        medium_risks = len([r for r in self.risk_register if r['risk_level'] == 'medium'])
        
        mitigated_risks = len([r for r in self.risk_register if r['status'] == 'mitigated'])
        
        dashboard = f"""
# Migration Risk Dashboard

## Risk Summary
- **Total Risks Identified**: {total_risks}
- **High Risk Items**: {high_risks}
- **Medium Risk Items**: {medium_risks}
- **Risks Mitigated**: {mitigated_risks}
- **Mitigation Rate**: {(mitigated_risks/total_risks*100):.1f}% if total_risks > 0 else 0%

## Top High-Risk Items
"""
        
        high_risk_items = [r for r in self.risk_register if r['risk_level'] == 'high']
        for risk in high_risk_items[:5]:  # Top 5 high risks
            dashboard += f"""
### {risk['id']}: {risk['description']}
- **Risk Score**: {risk['risk_score']}/100
- **Status**: {risk['status']}
- **Owner**: {risk.get('owner', 'Unassigned')}
"""
        
        return dashboard

# Example risk mitigation plans
def create_standard_mitigation_plans():
    """Create standard mitigation plans for common migration risks"""
    
    mitigation_templates = {
        'data_corruption': {
            'strategy': 'Comprehensive backup and validation strategy',
            'actions': [
                'Implement point-in-time backups before migration',
                'Create data validation scripts and checksums',
                'Perform test migrations with sample data',
                'Establish rollback procedures and time limits'
            ],
            'owner': 'Data Engineering Team',
            'due_date': '2 weeks before migration start',
            'effectiveness': 'high',
            'cost': 15000
        },
        
        'performance_degradation': {
            'strategy': 'Performance baseline and monitoring',
            'actions': [
                'Establish performance baselines in current environment',
                'Implement comprehensive monitoring in target environment',
                'Conduct load testing before go-live',
                'Prepare performance tuning runbooks'
            ],
            'owner': 'Performance Engineering Team',
            'due_date': '1 week before migration',
            'effectiveness': 'high',
            'cost': 25000
        },
        
        'extended_downtime': {
            'strategy': 'Minimize downtime through blue-green deployment',
            'actions': [
                'Implement blue-green deployment strategy',
                'Pre-stage all infrastructure and applications',
                'Rehearse migration procedures multiple times',
                'Prepare rapid rollback capabilities'
            ],
            'owner': 'DevOps Team',
            'due_date': 'Before migration execution',
            'effectiveness': 'medium',
            'cost': 20000
        }
    }
    
    return mitigation_templates
```

## Conclusion

This cloud migration strategy framework provides:

**Key Components:**
- Comprehensive assessment and discovery tools
- Strategic migration pattern selection (7 Rs framework)
- {{application_portfolio}} specific migration approaches
- Wave-based migration planning with dependency management
- Total cost of ownership analysis and optimization
- Risk management and mitigation strategies

**Timeline & Success Factors:**
- {{migration_timeline}} timeline optimization
- {{business_drivers}} focused value realization
- {{organization_size}} appropriate resource planning
- Continuous optimization and cost management

**Expected Outcomes:**
- Successful migration to {{target_cloud}}
- {{#if (includes business_drivers "cost-reduction")}}20-40% infrastructure cost reduction{{/if}}
- {{#if (includes business_drivers "scalability")}}Improved scalability and performance{{/if}}
- {{#if (includes business_drivers "modernization")}}Modernized application architecture{{/if}}
- Enhanced operational efficiency and agility