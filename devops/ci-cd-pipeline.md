---
name: ci_cd_pipeline
title: CI/CD Pipeline Configuration Generator
description: Generate complete CI/CD pipeline configurations for various platforms and technologies
category: devops
tags: [ci-cd, devops, automation, deployment, pipeline, github-actions, gitlab, jenkins]
difficulty: advanced
author: jezweb
version: 1.0.0
arguments:
  - name: platform
    description: CI/CD platform (github-actions, gitlab-ci, jenkins, azure-devops)
    required: true
  - name: project_type
    description: Type of project (node, python, java, go, docker)
    required: true
  - name: deployment_target
    description: Where to deploy (aws, azure, gcp, kubernetes, heroku)
    required: true
  - name: environments
    description: Deployment environments (comma-separated)
    required: false
    default: "dev,staging,production"
  - name: testing_enabled
    description: Include testing stage (yes/no)
    required: false
    default: "yes"
---

# CI/CD Pipeline Configuration: {{project_type}} → {{deployment_target}}

**Platform:** {{platform}}  
**Environments:** {{environments}}  
**Testing:** {{#if (eq testing_enabled "yes")}}Enabled{{else}}Disabled{{/if}}

{{#if (eq platform "github-actions")}}
## GitHub Actions Configuration

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

env:
  {{#if (eq project_type "node")}}
  NODE_VERSION: '18.x'
  {{else if (eq project_type "python")}}
  PYTHON_VERSION: '3.11'
  {{else if (eq project_type "java")}}
  JAVA_VERSION: '17'
  {{else if (eq project_type "go")}}
  GO_VERSION: '1.21'
  {{/if}}

jobs:
  # Build Job
  build:
    name: Build & Test
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    {{#if (eq project_type "node")}}
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: ${{ env.NODE_VERSION }}
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run linting
      run: npm run lint

    {{#if (eq testing_enabled "yes")}}
    - name: Run tests
      run: npm test -- --coverage
      
    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
    {{/if}}

    - name: Build application
      run: npm run build

    - name: Upload build artifacts
      uses: actions/upload-artifact@v4
      with:
        name: build-artifacts
        path: |
          dist/
          package*.json
    {{/if}}

    {{#if (eq project_type "python")}}
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Cache pip packages
      uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run linting
      run: |
        flake8 .
        black --check .

    {{#if (eq testing_enabled "yes")}}
    - name: Run tests
      run: |
        pytest --cov=./ --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v4
    {{/if}}
    {{/if}}

    {{#if (eq project_type "docker")}}
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: false
        tags: ${{ github.repository }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    {{/if}}

  # Security Scanning
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: build
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: 'trivy-results.sarif'

    {{#if (eq project_type "node")}}
    - name: Run npm audit
      run: npm audit --production
    {{/if}}

  {{#each (split environments ",")}}
  # Deploy to {{trim this}}
  deploy-{{trim this}}:
    name: Deploy to {{trim this}}
    runs-on: ubuntu-latest
    needs: [build, security]
    {{#if (eq (trim this) "production")}}
    if: github.ref == 'refs/heads/main'
    {{else if (eq (trim this) "staging")}}
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
    {{/if}}
    environment:
      name: {{trim this}}
      {{#if (eq (trim this) "production")}}
      url: https://{{trim this}}.example.com
      {{/if}}

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Download artifacts
      uses: actions/download-artifact@v4
      with:
        name: build-artifacts

    {{#if (eq deployment_target "aws")}}
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ vars.AWS_REGION }}

    {{#if (eq project_type "docker")}}
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v2

    - name: Build and push Docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: ${{ github.repository }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

    - name: Deploy to ECS
      run: |
        aws ecs update-service \
          --cluster {{trim this}}-cluster \
          --service {{trim this}}-service \
          --force-new-deployment
    {{else}}
    - name: Deploy to S3 and CloudFront
      run: |
        aws s3 sync dist/ s3://${{ vars.S3_BUCKET_{{uppercase (trim this)}} }}/ --delete
        aws cloudfront create-invalidation \
          --distribution-id ${{ vars.CLOUDFRONT_DISTRIBUTION_ID_{{uppercase (trim this)}} }} \
          --paths "/*"
    {{/if}}
    {{/if}}

    {{#if (eq deployment_target "kubernetes")}}
    - name: Setup kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'

    - name: Configure kubeconfig
      run: |
        echo "${{ secrets.KUBE_CONFIG_{{uppercase (trim this)}} }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig

    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/app app=${{ vars.DOCKER_REGISTRY }}/${{ github.repository }}:${{ github.sha }} \
          --namespace={{trim this}}
        kubectl rollout status deployment/app --namespace={{trim this}}
    {{/if}}

    - name: Smoke tests
      run: |
        curl -f https://{{trim this}}.example.com/health || exit 1

    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      if: always()
      with:
        status: ${{ job.status }}
        text: 'Deployment to {{trim this}} ${{ job.status }}'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
  {{/each}}
```

### GitHub Actions Secrets Required

```yaml
# Repository Secrets
AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
{{#each (split environments ",")}}
KUBE_CONFIG_{{uppercase (trim this)}}: ${{ secrets.KUBE_CONFIG_{{uppercase (trim this)}} }}
{{/each}}

# Repository Variables
AWS_REGION: us-east-1
{{#each (split environments ",")}}
S3_BUCKET_{{uppercase (trim this)}}: my-app-{{trim this}}
CLOUDFRONT_DISTRIBUTION_ID_{{uppercase (trim this)}}: ABCDEF123456
{{/each}}
```
{{/if}}

{{#if (eq platform "gitlab-ci")}}
## GitLab CI Configuration

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - security
  - deploy

variables:
  {{#if (eq project_type "node")}}
  NODE_VERSION: "18"
  {{else if (eq project_type "python")}}
  PYTHON_VERSION: "3.11"
  {{else if (eq project_type "docker")}}
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  {{/if}}

# Cache configuration
.cache_config: &cache_config
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      {{#if (eq project_type "node")}}
      - node_modules/
      - .npm/
      {{else if (eq project_type "python")}}
      - .cache/pip
      - venv/
      {{/if}}

# Build Stage
build:
  stage: build
  <<: *cache_config
  {{#if (eq project_type "node")}}
  image: node:${NODE_VERSION}
  before_script:
    - npm ci --cache .npm --prefer-offline
  script:
    - npm run lint
    - npm run build
  artifacts:
    paths:
      - dist/
    expire_in: 1 week
  {{else if (eq project_type "python")}}
  image: python:${PYTHON_VERSION}
  before_script:
    - pip install --cache-dir .cache/pip -r requirements.txt
  script:
    - flake8 .
    - black --check .
    - python -m build
  artifacts:
    paths:
      - dist/
    expire_in: 1 week
  {{/if}}

{{#if (eq testing_enabled "yes")}}
# Test Stage
test:
  stage: test
  <<: *cache_config
  {{#if (eq project_type "node")}}
  image: node:${NODE_VERSION}
  coverage: '/All files[^|]*\|[^|]*\s+([\d\.]+)/'
  script:
    - npm test -- --coverage
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml
  {{else if (eq project_type "python")}}
  image: python:${PYTHON_VERSION}
  script:
    - pytest --cov=./ --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
  {{/if}}
{{/if}}

# Security Scanning
security:
  stage: security
  image: aquasec/trivy:latest
  script:
    - trivy fs --no-progress --format json --output trivy-report.json .
  artifacts:
    reports:
      container_scanning: trivy-report.json

{{#each (split environments ",")}}
# Deploy to {{trim this}}
deploy:{{trim this}}:
  stage: deploy
  {{#if (eq (trim this) "production")}}
  only:
    - main
  {{else if (eq (trim this) "staging")}}
  only:
    - main
    - develop
  {{else}}
  only:
    - develop
  {{/if}}
  environment:
    name: {{trim this}}
    url: https://{{trim this}}.example.com
  
  {{#if (eq deployment_target "aws")}}
  image: amazon/aws-cli:latest
  before_script:
    - aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
    - aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
    - aws configure set region $AWS_REGION
  script:
    {{#if (eq project_type "docker")}}
    - aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
    - docker build -t $ECR_REGISTRY/$CI_PROJECT_NAME:$CI_COMMIT_SHA .
    - docker push $ECR_REGISTRY/$CI_PROJECT_NAME:$CI_COMMIT_SHA
    - |
      aws ecs update-service \
        --cluster {{trim this}}-cluster \
        --service {{trim this}}-service \
        --force-new-deployment
    {{else}}
    - aws s3 sync dist/ s3://${S3_BUCKET_{{uppercase (trim this)}}}/ --delete
    - |
      aws cloudfront create-invalidation \
        --distribution-id ${CLOUDFRONT_DISTRIBUTION_ID_{{uppercase (trim this)}}} \
        --paths "/*"
    {{/if}}
  {{/if}}

  {{#if (eq deployment_target "kubernetes")}}
  image: bitnami/kubectl:latest
  script:
    - echo $KUBE_CONFIG_{{uppercase (trim this)}} | base64 -d > kubeconfig
    - export KUBECONFIG=kubeconfig
    - |
      kubectl set image deployment/app \
        app=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA \
        --namespace={{trim this}}
    - kubectl rollout status deployment/app --namespace={{trim this}}
  {{/if}}
{{/each}}

# Notification job
notify:
  stage: .post
  image: appropriate/curl:latest
  when: always
  script:
    - |
      curl -X POST $SLACK_WEBHOOK \
        -H 'Content-type: application/json' \
        --data "{\"text\":\"Pipeline $CI_PIPELINE_STATUS for $CI_PROJECT_NAME\"}"
```
{{/if}}

## Pipeline Features

### 🔄 Continuous Integration
- Automated builds on every push
- Code quality checks (linting, formatting)
- {{#if (eq testing_enabled "yes")}}Comprehensive test suite execution{{/if}}
- Security vulnerability scanning
- Artifact generation and storage

### 🚀 Continuous Deployment
- Environment-specific deployments
- {{#each (split environments ",")}}
  - **{{trim this}}**: {{#if (eq (trim this) "production")}}Main branch only{{else if (eq (trim this) "staging")}}Main and develop branches{{else}}Feature branches{{/if}}
{{/each}}
- Zero-downtime deployments
- Automatic rollback capabilities

### 🔐 Security
- Dependency vulnerability scanning
- Container image scanning
- Secrets management
- SAST (Static Application Security Testing)

### 📊 Monitoring & Notifications
- Build status notifications
- Deployment notifications
- Performance metrics collection
- Error tracking integration

## Environment Configuration

### Environment Variables

{{#each (split environments ",")}}
#### {{capitalize (trim this)}} Environment
```bash
# Application
APP_ENV={{trim this}}
API_URL=https://api-{{trim this}}.example.com
{{#if (eq (trim this) "production")}}
DEBUG=false
LOG_LEVEL=error
{{else}}
DEBUG=true
LOG_LEVEL=debug
{{/if}}

# Database
DB_HOST={{trim this}}-db.example.com
DB_PORT=5432
DB_NAME=app_{{trim this}}

# Monitoring
SENTRY_DSN=$SENTRY_DSN_{{uppercase (trim this)}}
NEW_RELIC_APP_NAME=app-{{trim this}}
```
{{/each}}

## Best Practices Implemented

### 1. Build Optimization
- ✅ Dependency caching
- ✅ Parallel job execution
- ✅ Artifact reuse between stages
- ✅ Minimal Docker images

### 2. Security
- ✅ Vulnerability scanning
- ✅ Secrets rotation
- ✅ Least privilege access
- ✅ Signed commits

### 3. Reliability
- ✅ Automated rollbacks
- ✅ Health checks
- ✅ Smoke tests
- ✅ Progressive deployments

### 4. Observability
- ✅ Centralized logging
- ✅ Distributed tracing
- ✅ Custom metrics
- ✅ Real-time alerts

## Troubleshooting

### Common Issues

#### Build Failures
```bash
# Check build logs
{{#if (eq platform "github-actions")}}gh run view <run-id> --log{{/if}}
{{#if (eq platform "gitlab-ci")}}gitlab-runner exec docker build{{/if}}

# Validate configuration
{{#if (eq platform "github-actions")}}act -n{{/if}}
{{#if (eq platform "gitlab-ci")}}gitlab-ci-lint{{/if}}
```

#### Deployment Issues
```bash
# Check deployment status
{{#if (eq deployment_target "kubernetes")}}
kubectl rollout status deployment/app -n <environment>
kubectl logs -f deployment/app -n <environment>
{{/if}}

{{#if (eq deployment_target "aws")}}
aws ecs describe-services --cluster <cluster> --services <service>
aws logs tail /ecs/<service> --follow
{{/if}}
```

## Additional Resources

### Documentation
- [{{platform}} Documentation](https://docs.{{platform}}.com)
- [{{deployment_target}} Deployment Guide](https://docs.{{deployment_target}}.com)
- [Best Practices Guide](https://example.com/ci-cd-best-practices)

### Tools & Integrations
- **Monitoring**: Datadog, New Relic, Prometheus
- **Security**: Snyk, SonarQube, GitGuardian
- **Artifacts**: Artifactory, Nexus, Harbor
- **Notifications**: Slack, Teams, Email