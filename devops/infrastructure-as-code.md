---
name: infrastructure_as_code
title: Infrastructure as Code Implementation Guide
description: Comprehensive Infrastructure as Code framework covering Terraform, AWS CDK, and GitOps practices for scalable, maintainable infrastructure deployment
category: devops
tags: [infrastructure-as-code, terraform, aws-cdk, gitops, automation, deployment]
difficulty: advanced
author: jezweb
version: 1.0.0
arguments:
  - name: cloud_provider
    description: Primary cloud provider (aws, azure, gcp, multi-cloud)
    required: true
  - name: iac_tool
    description: Infrastructure as Code tool (terraform, aws-cdk, pulumi, ansible, cloudformation)
    required: true
  - name: deployment_pattern
    description: Deployment pattern (single-environment, multi-environment, multi-region, multi-cloud)
    required: true
  - name: team_size
    description: Team size managing infrastructure (small 1-3, medium 4-10, large >10)
    required: true
  - name: compliance_requirements
    description: Compliance needs (none, basic, sox, hipaa, pci-dss, government)
    required: true
  - name: infrastructure_scale
    description: Infrastructure scale (small <50-resources, medium 50-500, large >500)
    required: true
---

# Infrastructure as Code: {{iac_tool}} on {{cloud_provider}}

**Deployment Pattern:** {{deployment_pattern}}  
**Team Size:** {{team_size}}  
**Compliance:** {{compliance_requirements}}  
**Scale:** {{infrastructure_scale}}

## 1. IaC Architecture & Design Patterns

### Infrastructure Organization Strategy
```yaml
# Infrastructure repository structure
infrastructure_organization:
  {{#if (eq deployment_pattern "single-environment")}}
  single_environment:
    structure: |
      infrastructure/
      ├── modules/           # Reusable infrastructure modules
      │   ├── networking/
      │   ├── compute/
      │   ├── storage/
      │   └── security/
      ├── environments/
      │   └── production/
      │       ├── main.tf
      │       ├── variables.tf
      │       ├── outputs.tf
      │       └── terraform.tfvars
      ├── shared/            # Shared resources
      └── scripts/           # Deployment scripts
  {{else if (eq deployment_pattern "multi-environment")}}
  multi_environment:
    structure: |
      infrastructure/
      ├── modules/           # Reusable modules
      │   ├── vpc/
      │   ├── eks/
      │   ├── rds/
      │   ├── monitoring/
      │   └── security/
      ├── environments/
      │   ├── dev/
      │   │   ├── main.tf
      │   │   ├── variables.tf
      │   │   └── terraform.tfvars
      │   ├── staging/
      │   └── production/
      ├── global/            # Cross-environment resources
      │   ├── iam/
      │   ├── route53/
      │   └── certificates/
      └── policies/          # Governance policies
  {{else}}
  multi_region:
    structure: |
      infrastructure/
      ├── modules/
      ├── regions/
      │   ├── us-east-1/
      │   │   ├── dev/
      │   │   ├── staging/
      │   │   └── production/
      │   └── eu-west-1/
      ├── global/
      └── disaster-recovery/
  {{/if}}

best_practices:
  module_design:
    - "Create small, focused, reusable modules"
    - "Use semantic versioning for module releases"
    - "Implement proper input validation"
    - "Provide comprehensive output values"
    
  state_management:
    - "Use remote state backends (S3, Azure Storage, GCS)"
    - "Enable state locking to prevent conflicts"
    - "Implement state encryption"
    - "Regular state backup and recovery procedures"
    
  security:
    - "Never commit secrets or credentials"
    - "Use parameter stores and key management services"
    - "Implement least privilege access policies"
    - "Enable audit logging for all changes"
```

### Infrastructure Modules Design
```hcl
{{#if (eq iac_tool "terraform")}}
# Terraform module example - VPC module
# modules/vpc/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    {{#if (eq cloud_provider "aws")}}
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    {{else if (eq cloud_provider "azure")}}
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    {{else if (eq cloud_provider "gcp")}}
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    {{/if}}
  }
}

# VPC Module Variables
variable "vpc_name" {
  description = "Name of the VPC"
  type        = string
  validation {
    condition     = length(var.vpc_name) > 0
    error_message = "VPC name cannot be empty."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  validation {
    condition     = length(var.availability_zones) >= 2
    error_message = "At least 2 availability zones must be specified."
  }
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable VPC Flow Logs"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Local values for calculations
locals {
  # Calculate subnet CIDRs automatically
  public_subnet_cidrs = [
    for i, az in var.availability_zones :
    cidrsubnet(var.vpc_cidr, 8, i)
  ]
  
  private_subnet_cidrs = [
    for i, az in var.availability_zones :
    cidrsubnet(var.vpc_cidr, 8, i + 10)
  ]
  
  database_subnet_cidrs = [
    for i, az in var.availability_zones :
    cidrsubnet(var.vpc_cidr, 8, i + 20)
  ]
  
  common_tags = merge(var.tags, {
    Environment = var.vpc_name
    ManagedBy   = "Terraform"
    Module      = "vpc"
  })
}

{{#if (eq cloud_provider "aws")}}
# AWS VPC Resources
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(local.common_tags, {
    Name = var.vpc_name
  })
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = merge(local.common_tags, {
    Name = "${var.vpc_name}-igw"
  })
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(var.availability_zones)
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = local.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true
  
  tags = merge(local.common_tags, {
    Name = "${var.vpc_name}-public-${var.availability_zones[count.index]}"
    Type = "Public"
  })
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(var.availability_zones)
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = local.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]
  
  tags = merge(local.common_tags, {
    Name = "${var.vpc_name}-private-${var.availability_zones[count.index]}"
    Type = "Private"
  })
}

# Database Subnets
resource "aws_subnet" "database" {
  count = length(var.availability_zones)
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = local.database_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]
  
  tags = merge(local.common_tags, {
    Name = "${var.vpc_name}-database-${var.availability_zones[count.index]}"
    Type = "Database"
  })
}

# NAT Gateways
resource "aws_eip" "nat" {
  count = var.enable_nat_gateway ? length(var.availability_zones) : 0
  
  domain = "vpc"
  depends_on = [aws_internet_gateway.main]
  
  tags = merge(local.common_tags, {
    Name = "${var.vpc_name}-nat-eip-${count.index + 1}"
  })
}

resource "aws_nat_gateway" "main" {
  count = var.enable_nat_gateway ? length(var.availability_zones) : 0
  
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = merge(local.common_tags, {
    Name = "${var.vpc_name}-nat-${var.availability_zones[count.index]}"
  })
  
  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = merge(local.common_tags, {
    Name = "${var.vpc_name}-public-rt"
  })
}

resource "aws_route_table" "private" {
  count = length(var.availability_zones)
  
  vpc_id = aws_vpc.main.id
  
  dynamic "route" {
    for_each = var.enable_nat_gateway ? [1] : []
    content {
      cidr_block     = "0.0.0.0/0"
      nat_gateway_id = aws_nat_gateway.main[count.index].id
    }
  }
  
  tags = merge(local.common_tags, {
    Name = "${var.vpc_name}-private-rt-${var.availability_zones[count.index]}"
  })
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)
  
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# VPC Flow Logs
resource "aws_flow_log" "vpc" {
  count = var.enable_flow_logs ? 1 : 0
  
  iam_role_arn    = aws_iam_role.flow_log[0].arn
  log_destination = aws_cloudwatch_log_group.vpc_flow_log[0].arn
  traffic_type    = "ALL"
  vpc_id          = aws_vpc.main.id
}

resource "aws_cloudwatch_log_group" "vpc_flow_log" {
  count = var.enable_flow_logs ? 1 : 0
  
  name              = "/aws/vpc/flow-logs/${var.vpc_name}"
  retention_in_days = 30
  
  tags = local.common_tags
}

resource "aws_iam_role" "flow_log" {
  count = var.enable_flow_logs ? 1 : 0
  
  name = "${var.vpc_name}-flow-log-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "vpc-flow-logs.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy" "flow_log" {
  count = var.enable_flow_logs ? 1 : 0
  
  name = "${var.vpc_name}-flow-log-policy"
  role = aws_iam_role.flow_log[0].id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}
{{/if}}

# Module Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = aws_subnet.database[*].id
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.main.id
}

output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = aws_nat_gateway.main[*].id
}

output "availability_zones" {
  description = "List of availability zones used"
  value       = var.availability_zones
}
{{/if}}

{{#if (eq iac_tool "aws-cdk")}}
# AWS CDK TypeScript example
# lib/vpc-stack.ts
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';

export interface VpcStackProps extends cdk.StackProps {
  vpcName: string;
  vpcCidr: string;
  availabilityZones: string[];
  enableNatGateway: boolean;
  enableFlowLogs: boolean;
  {{#if (eq compliance_requirements "hipaa")}}
  enableEncryption: boolean;
  {{/if}}
}

export class VpcStack extends cdk.Stack {
  public readonly vpc: ec2.Vpc;
  public readonly publicSubnets: ec2.ISubnet[];
  public readonly privateSubnets: ec2.ISubnet[];
  public readonly databaseSubnets: ec2.ISubnet[];

  constructor(scope: Construct, id: string, props: VpcStackProps) {
    super(scope, id, props);

    // Create VPC with custom configuration
    this.vpc = new ec2.Vpc(this, 'VPC', {
      vpcName: props.vpcName,
      cidr: props.vpcCidr,
      maxAzs: props.availabilityZones.length,
      availabilityZones: props.availabilityZones,
      
      // Subnet configuration
      subnetConfiguration: [
        {
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: 'Private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 24,
        },
        {
          name: 'Database',
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          cidrMask: 24,
        },
      ],
      
      // NAT Gateway configuration
      natGateways: props.enableNatGateway ? props.availabilityZones.length : 0,
      natGatewayProvider: props.enableNatGateway 
        ? ec2.NatProvider.gateway()
        : undefined,
      
      // Enable DNS
      enableDnsHostnames: true,
      enableDnsSupport: true,
    });

    // Store subnet references
    this.publicSubnets = this.vpc.publicSubnets;
    this.privateSubnets = this.vpc.privateSubnets;
    this.databaseSubnets = this.vpc.isolatedSubnets;

    // Enable VPC Flow Logs if requested
    if (props.enableFlowLogs) {
      const flowLogGroup = new logs.LogGroup(this, 'VpcFlowLogGroup', {
        logGroupName: `/aws/vpc/flow-logs/${props.vpcName}`,
        retention: logs.RetentionDays.ONE_MONTH,
        {{#if (eq compliance_requirements "hipaa")}}
        encryptionKey: undefined, // Use KMS key for HIPAA compliance
        {{/if}}
      });

      new ec2.FlowLog(this, 'VpcFlowLog', {
        resourceType: ec2.FlowLogResourceType.fromVpc(this.vpc),
        destination: ec2.FlowLogDestination.toCloudWatchLogs(flowLogGroup),
        trafficType: ec2.FlowLogTrafficType.ALL,
      });
    }

    // Add tags for compliance and governance
    const commonTags = {
      Environment: props.vpcName,
      ManagedBy: 'AWS-CDK',
      {{#if (eq compliance_requirements "sox")}}
      Compliance: 'SOX',
      DataClassification: 'Internal',
      {{/if}}
      {{#if (eq compliance_requirements "hipaa")}}
      Compliance: 'HIPAA',
      DataClassification: 'PHI',
      {{/if}}
    };

    Object.entries(commonTags).forEach(([key, value]) => {
      cdk.Tags.of(this).add(key, value);
    });

    // Output important values
    new cdk.CfnOutput(this, 'VpcId', {
      value: this.vpc.vpcId,
      description: 'VPC ID',
      exportName: `${props.vpcName}-VpcId`,
    });

    new cdk.CfnOutput(this, 'VpcCidr', {
      value: this.vpc.vpcCidrBlock,
      description: 'VPC CIDR Block',
    });

    new cdk.CfnOutput(this, 'PublicSubnetIds', {
      value: this.publicSubnets.map(subnet => subnet.subnetId).join(','),
      description: 'Public Subnet IDs',
    });

    new cdk.CfnOutput(this, 'PrivateSubnetIds', {
      value: this.privateSubnets.map(subnet => subnet.subnetId).join(','),
      description: 'Private Subnet IDs',
    });
  }
}

// Usage example
// bin/vpc-app.ts
#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { VpcStack } from '../lib/vpc-stack';

const app = new cdk.App();

const vpcStack = new VpcStack(app, '{{deployment_pattern}}-VpcStack', {
  vpcName: '{{deployment_pattern}}-vpc',
  vpcCidr: '10.0.0.0/16',
  availabilityZones: ['us-east-1a', 'us-east-1b', 'us-east-1c'],
  enableNatGateway: true,
  enableFlowLogs: true,
  {{#if (eq compliance_requirements "hipaa")}}
  enableEncryption: true,
  {{/if}}
  
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
  
  tags: {
    Project: 'Infrastructure',
    Team: 'Platform',
    CostCenter: 'Engineering',
  },
});
{{/if}}
```

## 2. State Management & Backend Configuration

### Remote State Backend Setup
```hcl
{{#if (eq iac_tool "terraform")}}
# Remote state configuration
# backend.tf
terraform {
  backend "s3" {
    {{#if (eq cloud_provider "aws")}}
    bucket         = "{{deployment_pattern}}-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "{{deployment_pattern}}-terraform-locks"
    
    {{#if (eq compliance_requirements "sox")}}
    # SOX compliance requirements
    versioning     = true
    logging        = true
    {{/if}}
    
    {{#if (eq compliance_requirements "hipaa")}}
    # HIPAA compliance requirements
    kms_key_id     = "arn:aws:kms:us-east-1:ACCOUNT:key/KEY-ID"
    {{/if}}
    {{/if}}
  }
  
  required_version = ">= 1.0"
  
  required_providers {
    {{#if (eq cloud_provider "aws")}}
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    {{/if}}
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

# State backend infrastructure
# Create S3 bucket and DynamoDB table for state management
resource "aws_s3_bucket" "terraform_state" {
  bucket = "{{deployment_pattern}}-terraform-state"
  
  lifecycle {
    prevent_destroy = true
  }
  
  tags = {
    Name        = "Terraform State"
    Environment = "{{deployment_pattern}}"
    Purpose     = "Infrastructure State Management"
  }
}

resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        {{#if (eq compliance_requirements "hipaa")}}
        kms_master_key_id = aws_kms_key.terraform_state.arn
        sse_algorithm     = "aws:kms"
        {{else}}
        sse_algorithm = "AES256"
        {{/if}}
      }
      bucket_key_enabled = true
    }
  }
}

resource "aws_s3_bucket_public_access_block" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "terraform_locks" {
  name           = "{{deployment_pattern}}-terraform-locks"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "LockID"
  
  attribute {
    name = "LockID"
    type = "S"
  }
  
  server_side_encryption {
    enabled = true
    {{#if (eq compliance_requirements "hipaa")}}
    kms_key_arn = aws_kms_key.terraform_state.arn
    {{/if}}
  }
  
  point_in_time_recovery {
    enabled = true
  }
  
  tags = {
    Name        = "Terraform State Locks"
    Environment = "{{deployment_pattern}}"
    Purpose     = "Infrastructure State Locking"
  }
}

{{#if (eq compliance_requirements "hipaa")}}
# KMS key for HIPAA compliance
resource "aws_kms_key" "terraform_state" {
  description         = "KMS key for Terraform state encryption"
  enable_key_rotation = true
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      }
    ]
  })
  
  tags = {
    Name        = "Terraform State KMS Key"
    Environment = "{{deployment_pattern}}"
    Compliance  = "HIPAA"
  }
}

resource "aws_kms_alias" "terraform_state" {
  name          = "alias/terraform-state-{{deployment_pattern}}"
  target_key_id = aws_kms_key.terraform_state.key_id
}

data "aws_caller_identity" "current" {}
{{/if}}
{{/if}}
```

### Multi-Environment State Management
```python
# Infrastructure deployment automation
# scripts/deploy.py
import os
import sys
import subprocess
import json
import boto3
from typing import Dict, List, Optional
import argparse

class InfrastructureDeployer:
    def __init__(self, environment: str, region: str = "us-east-1"):
        self.environment = environment
        self.region = region
        self.tf_vars = {}
        self.compliance_mode = "{{compliance_requirements}}"
        
    def validate_environment(self) -> bool:
        """Validate environment configuration"""
        
        required_vars = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY'
        ]
        
        {{#if (eq compliance_requirements "sox")}}
        # SOX compliance checks
        required_vars.extend([
            'SOX_AUDIT_TRAIL_ENABLED',
            'SOX_CHANGE_APPROVAL_ID'
        ])
        {{/if}}
        
        {{#if (eq compliance_requirements "hipaa")}}
        # HIPAA compliance checks
        required_vars.extend([
            'HIPAA_ENCRYPTION_ENABLED',
            'HIPAA_AUDIT_LOGGING_ENABLED'
        ])
        {{/if}}
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {missing_vars}")
            return False
            
        return True
    
    def setup_terraform_backend(self):
        """Initialize Terraform backend for environment"""
        
        backend_config = {
            "bucket": f"{{deployment_pattern}}-terraform-state-{self.environment}",
            "key": f"infrastructure/{self.environment}/terraform.tfstate",
            "region": self.region,
            "encrypt": True,
            "dynamodb_table": f"{{deployment_pattern}}-terraform-locks-{self.environment}"
        }
        
        # Create backend configuration file
        with open(f"backend-{self.environment}.hcl", "w") as f:
            for key, value in backend_config.items():
                if isinstance(value, bool):
                    f.write(f'{key} = {str(value).lower()}\n')
                else:
                    f.write(f'{key} = "{value}"\n')
        
        return f"backend-{self.environment}.hcl"
    
    def load_environment_variables(self):
        """Load environment-specific variables"""
        
        # Base configuration
        self.tf_vars = {
            "environment": self.environment,
            "region": self.region,
            "team_size": "{{team_size}}",
            "compliance_requirements": "{{compliance_requirements}}",
            "infrastructure_scale": "{{infrastructure_scale}}"
        }
        
        # Environment-specific overrides
        env_configs = {
            "dev": {
                "instance_types": ["t3.micro", "t3.small"],
                "enable_monitoring": False,
                "backup_retention": 7,
                "multi_az": False
            },
            "staging": {
                "instance_types": ["t3.small", "t3.medium"],
                "enable_monitoring": True,
                "backup_retention": 14,
                "multi_az": True
            },
            "production": {
                "instance_types": ["t3.medium", "t3.large", "t3.xlarge"],
                "enable_monitoring": True,
                "backup_retention": 30,
                "multi_az": True,
                "enable_encryption": True,
                "enable_logging": True
            }
        }
        
        if self.environment in env_configs:
            self.tf_vars.update(env_configs[self.environment])
    
    def run_terraform_command(self, command: str, args: List[str] = None) -> bool:
        """Execute Terraform command with proper error handling"""
        
        cmd = ["terraform", command]
        if args:
            cmd.extend(args)
        
        print(f"🔧 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Terraform command failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
    
    def plan_deployment(self) -> bool:
        """Generate and review deployment plan"""
        
        print(f"📋 Planning deployment for {self.environment}")
        
        # Initialize Terraform
        backend_file = self.setup_terraform_backend()
        if not self.run_terraform_command("init", [f"-backend-config={backend_file}"]):
            return False
        
        # Create tfvars file
        tfvars_file = f"{self.environment}.tfvars"
        with open(tfvars_file, "w") as f:
            for key, value in self.tf_vars.items():
                if isinstance(value, bool):
                    f.write(f'{key} = {str(value).lower()}\n')
                elif isinstance(value, list):
                    f.write(f'{key} = {json.dumps(value)}\n')
                else:
                    f.write(f'{key} = "{value}"\n')
        
        # Generate plan
        plan_file = f"{self.environment}.tfplan"
        plan_args = [
            "-out", plan_file,
            f"-var-file={tfvars_file}",
            "-detailed-exitcode"
        ]
        
        return self.run_terraform_command("plan", plan_args)
    
    def apply_deployment(self, auto_approve: bool = False) -> bool:
        """Apply the deployment plan"""
        
        print(f"🚀 Applying deployment for {self.environment}")
        
        plan_file = f"{self.environment}.tfplan"
        if not os.path.exists(plan_file):
            print("❌ No plan file found. Run plan first.")
            return False
        
        apply_args = [plan_file]
        if auto_approve:
            apply_args.insert(0, "-auto-approve")
        
        success = self.run_terraform_command("apply", apply_args)
        
        if success:
            print(f"✅ Deployment completed successfully for {self.environment}")
            
            # Post-deployment validation
            self.validate_deployment()
        
        return success
    
    def validate_deployment(self):
        """Validate deployed infrastructure"""
        
        print("🔍 Validating deployment...")
        
        {{#if (eq compliance_requirements "sox")}}
        # SOX compliance validation
        self.validate_sox_compliance()
        {{/if}}
        
        {{#if (eq compliance_requirements "hipaa")}}
        # HIPAA compliance validation
        self.validate_hipaa_compliance()
        {{/if}}
        
        # Basic infrastructure validation
        self.validate_basic_infrastructure()
    
    def validate_basic_infrastructure(self):
        """Basic infrastructure health checks"""
        
        try:
            # Get Terraform outputs
            result = subprocess.run(
                ["terraform", "output", "-json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            outputs = json.loads(result.stdout)
            
            # Validate critical outputs exist
            required_outputs = ["vpc_id", "public_subnet_ids", "private_subnet_ids"]
            missing_outputs = [output for output in required_outputs if output not in outputs]
            
            if missing_outputs:
                print(f"⚠️  Missing required outputs: {missing_outputs}")
            else:
                print("✅ All required infrastructure outputs present")
                
        except Exception as e:
            print(f"❌ Deployment validation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Infrastructure deployment tool")
    parser.add_argument("environment", help="Environment to deploy (dev/staging/production)")
    parser.add_argument("action", choices=["plan", "apply", "destroy"], help="Action to perform")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve apply/destroy")
    
    args = parser.parse_args()
    
    deployer = InfrastructureDeployer(args.environment, args.region)
    
    if not deployer.validate_environment():
        sys.exit(1)
    
    deployer.load_environment_variables()
    
    if args.action == "plan":
        success = deployer.plan_deployment()
    elif args.action == "apply":
        success = deployer.apply_deployment(args.auto_approve)
    elif args.action == "destroy":
        # Implement destroy logic
        print("Destroy action not implemented yet")
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

## 3. GitOps & CI/CD Integration

### GitOps Workflow Configuration
```yaml
# .github/workflows/infrastructure.yml
name: Infrastructure Deployment

on:
  push:
    branches: [main, develop]
    paths: 
      - 'infrastructure/**'
      - '.github/workflows/infrastructure.yml'
  pull_request:
    branches: [main]
    paths: 
      - 'infrastructure/**'

env:
  TF_VERSION: '1.6.0'
  {{#if (eq cloud_provider "aws")}}
  AWS_REGION: 'us-east-1'
  {{/if}}
  COMPLIANCE_MODE: '{{compliance_requirements}}'

jobs:
  validate:
    name: Validate Infrastructure Code
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}
      
      - name: Terraform Format Check
        run: terraform fmt -check -recursive
        working-directory: infrastructure
      
      - name: Terraform Init
        run: terraform init -backend=false
        working-directory: infrastructure
      
      - name: Terraform Validate
        run: terraform validate
        working-directory: infrastructure
      
      {{#if (eq compliance_requirements "sox")}}
      - name: SOX Compliance Check
        run: |
          # Check for required SOX compliance configurations
          echo "Validating SOX compliance requirements..."
          grep -r "encrypt.*=.*true" infrastructure/ || exit 1
          grep -r "versioning.*=.*true" infrastructure/ || exit 1
      {{/if}}
      
      - name: Security Scan
        uses: aquasec/trivy-action@master
        with:
          scan-type: 'config'
          scan-ref: 'infrastructure/'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Security Results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

  plan-dev:
    name: Plan Development
    runs-on: ubuntu-latest
    needs: validate
    if: github.event_name == 'pull_request'
    environment: development
    
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Terraform Init
        run: |
          terraform init \
            -backend-config="bucket={{deployment_pattern}}-terraform-state-dev" \
            -backend-config="key=infrastructure/dev/terraform.tfstate" \
            -backend-config="region=${{ env.AWS_REGION }}"
        working-directory: infrastructure
      
      - name: Terraform Plan
        run: |
          terraform plan \
            -var-file="environments/dev/terraform.tfvars" \
            -out=dev.tfplan \
            -no-color
        working-directory: infrastructure
        
      - name: Comment Plan
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const planOutput = fs.readFileSync('infrastructure/dev.tfplan.txt', 'utf8');
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Terraform Plan - Development\n\`\`\`\n${planOutput}\n\`\`\``
            });

  {{#if (eq deployment_pattern "multi-environment")}}
  deploy-dev:
    name: Deploy to Development
    runs-on: ubuntu-latest
    needs: validate
    if: github.ref == 'refs/heads/develop'
    environment: development
    
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Deploy Infrastructure
        run: |
          python scripts/deploy.py dev apply --auto-approve
        working-directory: infrastructure

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [validate, deploy-dev]
    if: github.ref == 'refs/heads/main'
    environment: staging
    
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Deploy Infrastructure
        run: |
          python scripts/deploy.py staging apply --auto-approve
        working-directory: infrastructure

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [validate, deploy-staging]
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: ${{ env.TF_VERSION }}
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      {{#if (eq compliance_requirements "sox")}}
      - name: SOX Approval Check
        run: |
          # Verify SOX approval is present
          if [ -z "${{ secrets.SOX_APPROVAL_ID }}" ]; then
            echo "SOX approval required for production deployment"
            exit 1
          fi
      {{/if}}
      
      - name: Deploy Infrastructure
        run: |
          python scripts/deploy.py production apply
        working-directory: infrastructure
        env:
          {{#if (eq compliance_requirements "sox")}}
          SOX_APPROVAL_ID: ${{ secrets.SOX_APPROVAL_ID }}
          {{/if}}
      
      - name: Post-Deployment Validation
        run: |
          # Run comprehensive validation
          python scripts/validate-deployment.py production
        working-directory: infrastructure
  {{/if}}
```

## 4. Compliance & Security

{{#if (eq compliance_requirements "sox")}}
### SOX Compliance Framework
```hcl
# SOX compliance resources
# modules/compliance/sox.tf
resource "aws_config_configuration_recorder" "sox_recorder" {
  name     = "sox-compliance-recorder"
  role_arn = aws_iam_role.config_role.arn

  recording_group {
    all_supported                 = true
    include_global_resource_types = true
  }

  depends_on = [aws_config_delivery_channel.sox_delivery_channel]
}

resource "aws_config_delivery_channel" "sox_delivery_channel" {
  name           = "sox-compliance-delivery-channel"
  s3_bucket_name = aws_s3_bucket.config_bucket.bucket
  
  snapshot_delivery_properties {
    delivery_frequency = "Daily"
  }
}

resource "aws_config_config_rule" "sox_encrypted_volumes" {
  name = "sox-encrypted-ebs-volumes"

  source {
    owner             = "AWS"
    source_identifier = "ENCRYPTED_VOLUMES"
  }

  depends_on = [aws_config_configuration_recorder.sox_recorder]
}

resource "aws_config_config_rule" "sox_root_access_key_check" {
  name = "sox-root-access-key-check"

  source {
    owner             = "AWS"
    source_identifier = "ROOT_ACCESS_KEY_CHECK"
  }

  depends_on = [aws_config_configuration_recorder.sox_recorder]
}

# CloudTrail for audit logging
resource "aws_cloudtrail" "sox_audit_trail" {
  name                          = "sox-audit-trail"
  s3_bucket_name               = aws_s3_bucket.audit_bucket.bucket
  include_global_service_events = true
  is_multi_region_trail        = true
  enable_logging               = true

  event_selector {
    read_write_type           = "All"
    include_management_events = true

    data_resource {
      type   = "AWS::S3::Object"
      values = ["arn:aws:s3:::*/*"]
    }
  }

  tags = {
    Compliance = "SOX"
    Purpose    = "Audit Trail"
  }
}
```
{{/if}}

{{#if (eq compliance_requirements "hipaa")}}
### HIPAA Compliance Framework
```hcl
# HIPAA compliance resources
# modules/compliance/hipaa.tf

# KMS key for encryption at rest
resource "aws_kms_key" "hipaa_key" {
  description         = "HIPAA compliant encryption key"
  enable_key_rotation = true
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow HIPAA Users"
        Effect = "Allow"
        Principal = {
          AWS = [
            "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/HIPAARole"
          ]
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })
  
  tags = {
    Compliance        = "HIPAA"
    DataClassification = "PHI"
  }
}

resource "aws_kms_alias" "hipaa_key" {
  name          = "alias/hipaa-encryption-key"
  target_key_id = aws_kms_key.hipaa_key.key_id
}

# S3 bucket with HIPAA-compliant configuration
resource "aws_s3_bucket" "hipaa_data" {
  bucket = "hipaa-compliant-data-${random_id.bucket_suffix.hex}"
  
  tags = {
    Compliance        = "HIPAA"
    DataClassification = "PHI"
    Environment       = var.environment
  }
}

resource "aws_s3_bucket_encryption" "hipaa_data" {
  bucket = aws_s3_bucket.hipaa_data.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.hipaa_key.arn
        sse_algorithm     = "aws:kms"
      }
      bucket_key_enabled = true
    }
  }
}

resource "aws_s3_bucket_versioning" "hipaa_data" {
  bucket = aws_s3_bucket.hipaa_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_logging" "hipaa_data" {
  bucket = aws_s3_bucket.hipaa_data.id
  
  target_bucket = aws_s3_bucket.access_logs.id
  target_prefix = "access-logs/"
}

resource "aws_s3_bucket_public_access_block" "hipaa_data" {
  bucket = aws_s3_bucket.hipaa_data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# RDS with encryption
resource "aws_db_instance" "hipaa_database" {
  identifier = "hipaa-database"
  
  engine            = "postgres"
  engine_version    = "14.9"
  instance_class    = "db.r5.large"
  allocated_storage = 100
  
  db_name  = "hipaa_db"
  username = "hipaa_admin"
  password = random_password.db_password.result
  
  # HIPAA requirements
  storage_encrypted   = true
  kms_key_id         = aws_kms_key.hipaa_key.arn
  deletion_protection = true
  
  # Backup and monitoring
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Logging
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  # Network security
  db_subnet_group_name   = aws_db_subnet_group.hipaa.name
  vpc_security_group_ids = [aws_security_group.hipaa_db.id]
  
  tags = {
    Compliance        = "HIPAA"
    DataClassification = "PHI"
  }
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "aws_secretsmanager_secret" "db_password" {
  name        = "hipaa-db-password"
  description = "HIPAA database password"
  kms_key_id  = aws_kms_key.hipaa_key.arn
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = random_password.db_password.result
}
```
{{/if}}

## 5. Monitoring & Observability

### Infrastructure Monitoring Setup
```python
# Infrastructure monitoring and alerting
# scripts/monitoring.py
import boto3
import json
from typing import Dict, List
import subprocess

class InfrastructureMonitoring:
    def __init__(self, environment: str = "{{deployment_pattern}}"):
        self.environment = environment
        self.cloudwatch = boto3.client('cloudwatch')
        self.sns = boto3.client('sns')
        
    def setup_monitoring_stack(self):
        """Setup comprehensive monitoring for infrastructure"""
        
        # Create SNS topic for alerts
        topic_arn = self.create_alert_topic()
        
        # Setup CloudWatch alarms
        self.create_infrastructure_alarms(topic_arn)
        
        # Setup custom metrics
        self.setup_custom_metrics()
        
        # Create dashboard
        self.create_infrastructure_dashboard()
        
        return topic_arn
    
    def create_alert_topic(self) -> str:
        """Create SNS topic for infrastructure alerts"""
        
        topic_name = f"infrastructure-alerts-{self.environment}"
        
        try:
            response = self.sns.create_topic(Name=topic_name)
            topic_arn = response['TopicArn']
            
            # Add email subscription (would be configured via environment variables)
            email = os.getenv('ALERT_EMAIL')
            if email:
                self.sns.subscribe(
                    TopicArn=topic_arn,
                    Protocol='email',
                    Endpoint=email
                )
            
            return topic_arn
            
        except Exception as e:
            print(f"Error creating SNS topic: {e}")
            return None
    
    def create_infrastructure_alarms(self, topic_arn: str):
        """Create CloudWatch alarms for infrastructure components"""
        
        alarms = [
            {
                'AlarmName': f'HighCPUUtilization-{self.environment}',
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 2,
                'MetricName': 'CPUUtilization',
                'Namespace': 'AWS/EC2',
                'Period': 300,
                'Statistic': 'Average',
                'Threshold': 80.0,
                'ActionsEnabled': True,
                'AlarmActions': [topic_arn],
                'AlarmDescription': 'High CPU utilization detected',
                'Unit': 'Percent'
            },
            {
                'AlarmName': f'DatabaseConnections-{self.environment}',
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 2,
                'MetricName': 'DatabaseConnections',
                'Namespace': 'AWS/RDS',
                'Period': 300,
                'Statistic': 'Average',
                'Threshold': 80,
                'ActionsEnabled': True,
                'AlarmActions': [topic_arn],
                'AlarmDescription': 'High database connection count'
            },
            {
                'AlarmName': f'DiskSpaceUtilization-{self.environment}',
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 1,
                'MetricName': 'DiskSpaceUtilization',
                'Namespace': 'System/Linux',
                'Period': 300,
                'Statistic': 'Average',
                'Threshold': 85.0,
                'ActionsEnabled': True,
                'AlarmActions': [topic_arn],
                'AlarmDescription': 'High disk space utilization'
            }
        ]
        
        for alarm in alarms:
            try:
                self.cloudwatch.put_metric_alarm(**alarm)
                print(f"✅ Created alarm: {alarm['AlarmName']}")
            except Exception as e:
                print(f"❌ Failed to create alarm {alarm['AlarmName']}: {e}")
    
    def create_infrastructure_dashboard(self):
        """Create CloudWatch dashboard for infrastructure monitoring"""
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/EC2", "CPUUtilization"],
                            ["AWS/RDS", "CPUUtilization"],
                            ["AWS/RDS", "DatabaseConnections"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": "us-east-1",
                        "title": "Infrastructure Health"
                    }
                },
                {
                    "type": "log",
                    "properties": {
                        "query": f"SOURCE '/aws/vpc/flow-logs/{self.environment}'\n| fields @timestamp, srcaddr, dstaddr, action\n| filter action = \"REJECT\"\n| stats count() by srcaddr\n| sort count desc\n| limit 20",
                        "region": "us-east-1",
                        "title": "Top Rejected Connections",
                        "view": "table"
                    }
                }
            ]
        }
        
        try:
            self.cloudwatch.put_dashboard(
                DashboardName=f"Infrastructure-{self.environment}",
                DashboardBody=json.dumps(dashboard_body)
            )
            print(f"✅ Created dashboard: Infrastructure-{self.environment}")
        except Exception as e:
            print(f"❌ Failed to create dashboard: {e}")

# Cost optimization monitoring
class CostOptimizationMonitor:
    def __init__(self):
        self.ce = boto3.client('ce')  # Cost Explorer
        self.ec2 = boto3.client('ec2')
        
    def analyze_resource_utilization(self) -> Dict:
        """Analyze resource utilization for cost optimization"""
        
        recommendations = {
            'underutilized_instances': [],
            'unused_volumes': [],
            'optimization_opportunities': []
        }
        
        # Check for underutilized EC2 instances
        instances = self.ec2.describe_instances()
        for reservation in instances['Reservations']:
            for instance in reservation['Instances']:
                if instance['State']['Name'] == 'running':
                    # Get CloudWatch metrics for CPU utilization
                    cpu_metrics = self.get_cpu_utilization(instance['InstanceId'])
                    if cpu_metrics['average_cpu'] < 10:  # Less than 10% average CPU
                        recommendations['underutilized_instances'].append({
                            'instance_id': instance['InstanceId'],
                            'instance_type': instance['InstanceType'],
                            'average_cpu': cpu_metrics['average_cpu'],
                            'recommended_action': 'Consider downsizing or terminating'
                        })
        
        # Check for unattached EBS volumes
        volumes = self.ec2.describe_volumes()
        for volume in volumes['Volumes']:
            if volume['State'] == 'available':  # Unattached
                recommendations['unused_volumes'].append({
                    'volume_id': volume['VolumeId'],
                    'size': volume['Size'],
                    'volume_type': volume['VolumeType'],
                    'recommended_action': 'Delete if not needed'
                })
        
        return recommendations
    
    def get_cpu_utilization(self, instance_id: str) -> Dict:
        """Get CPU utilization metrics for an instance"""
        
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[
                    {
                        'Name': 'InstanceId',
                        'Value': instance_id
                    }
                ],
                StartTime=datetime.utcnow() - timedelta(days=7),
                EndTime=datetime.utcnow(),
                Period=86400,  # 1 day
                Statistics=['Average']
            )
            
            if response['Datapoints']:
                avg_cpu = sum(point['Average'] for point in response['Datapoints']) / len(response['Datapoints'])
                return {'average_cpu': avg_cpu}
            else:
                return {'average_cpu': 0}
                
        except Exception as e:
            print(f"Error getting CPU metrics for {instance_id}: {e}")
            return {'average_cpu': 0}

if __name__ == "__main__":
    # Setup monitoring
    monitor = InfrastructureMonitoring()
    topic_arn = monitor.setup_monitoring_stack()
    
    # Cost optimization analysis
    cost_monitor = CostOptimizationMonitor()
    recommendations = cost_monitor.analyze_resource_utilization()
    
    print("Cost Optimization Recommendations:")
    print(json.dumps(recommendations, indent=2))
```

## Conclusion

This Infrastructure as Code framework provides:

**Key Features:**
- {{iac_tool}} implementation on {{cloud_provider}}
- {{deployment_pattern}} deployment strategy
- {{compliance_requirements}} compliance controls
- Comprehensive monitoring and alerting

**Benefits:**
- Scalable infrastructure management for {{team_size}} teams
- Automated deployment pipelines with GitOps
- {{compliance_requirements}} compliance and security controls
- Cost optimization and resource monitoring

**Best Practices:**
- Modular, reusable infrastructure components
- Environment-specific configuration management
- Comprehensive state management and locking
- Security-first approach with encryption and access controls

**Success Metrics:**
- Reduced deployment time and manual errors
- Improved infrastructure consistency across environments
- Enhanced security posture and compliance adherence
- Cost optimization through monitoring and automation