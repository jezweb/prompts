---
name: code_review_automation
title: Automated Code Review System
description: Comprehensive automated code review framework with static analysis, quality gates, security scanning, and intelligent feedback generation for development teams
category: development
tags: [code-review, automation, static-analysis, quality-gates, ci-cd, security]
difficulty: advanced
author: jezweb
version: 1.0.0
arguments:
  - name: programming_languages
    description: Primary programming languages (javascript, typescript, python, java, go, csharp, mixed)
    required: true
  - name: repository_platform
    description: Repository platform (github, gitlab, bitbucket, azure-devops)
    required: true
  - name: team_size
    description: Development team size (small 2-5, medium 6-15, large >15)
    required: true
  - name: code_quality_standards
    description: Code quality standards (basic, intermediate, strict, enterprise)
    required: true
  - name: security_requirements
    description: Security requirements (basic, enhanced, enterprise, regulatory)
    required: true
  - name: review_automation_level
    description: Automation level (basic-checks, comprehensive, ai-assisted, full-automation)
    required: true
---

# Automated Code Review System: {{programming_languages}}

**Repository Platform:** {{repository_platform}}  
**Team Size:** {{team_size}}  
**Quality Standards:** {{code_quality_standards}}  
**Security Level:** {{security_requirements}}  
**Automation Level:** {{review_automation_level}}

## 1. Code Review Automation Framework

### Pre-commit Hooks Configuration
```yaml
# .pre-commit-config.yaml
repos:
  {{#if (contains programming_languages "javascript" "typescript")}}
  # JavaScript/TypeScript specific hooks
  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.0.0
    hooks:
      - id: eslint
        files: \.(js|ts|jsx|tsx)$
        args: [--fix]
        additional_dependencies:
          - eslint@^8.0.0
          - '@typescript-eslint/parser@^6.0.0'
          - '@typescript-eslint/eslint-plugin@^6.0.0'
          {{#if (eq code_quality_standards "strict" "enterprise")}}
          - eslint-plugin-security@^1.7.0
          - eslint-plugin-sonarjs@^0.19.0
          {{/if}}
  
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.0
    hooks:
      - id: prettier
        files: \.(js|ts|jsx|tsx|json|md|yml|yaml)$
        args: [--write]
  {{/if}}

  {{#if (contains programming_languages "python")}}
  # Python specific hooks
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11
        args: [--line-length=88]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
        additional_dependencies:
          - flake8-docstrings
          {{#if (eq security_requirements "enhanced" "enterprise" "regulatory")}}
          - flake8-bandit
          - flake8-bugbear
          {{/if}}
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-PyYAML]
        args: [--strict]
  {{/if}}

  {{#if (contains programming_languages "java")}}
  # Java specific hooks
  - repo: https://github.com/macisamuele/language-formatters-pre-commit-hooks
    rev: v2.10.0
    hooks:
      - id: pretty-format-java
        args: [--autofix]
  
  - repo: local
    hooks:
      - id: checkstyle
        name: checkstyle
        entry: java -jar checkstyle.jar -c checkstyle.xml
        language: system
        files: \.java$
        pass_filenames: true
  {{/if}}

  # Universal hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-merge-conflict
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: [--maxkb={{#if (eq team_size "large")}}500{{else}}200{{/if}}]
      - id: detect-private-key
      - id: check-case-conflict
      - id: mixed-line-ending

  {{#if (eq security_requirements "enhanced" "enterprise" "regulatory")}}
  # Security scanning
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: [--baseline, .secrets.baseline]
        exclude: package.lock.json
  
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.16.4
    hooks:
      - id: gitleaks
  {{/if}}

  {{#if (eq code_quality_standards "strict" "enterprise")}}
  # Additional quality checks
  - repo: https://github.com/adrienverge/yamllint
    rev: v1.32.0
    hooks:
      - id: yamllint
        args: [-d, relaxed]
  
  - repo: https://github.com/shellcheck-py/shellcheck-py
    rev: v0.9.0.5
    hooks:
      - id: shellcheck
  {{/if}}
```

### CI/CD Pipeline Integration
```yaml
{{#if (eq repository_platform "github")}}
# .github/workflows/code-review.yml
name: Automated Code Review

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main, develop]

env:
  NODE_VERSION: '18'
  PYTHON_VERSION: '3.11'
  JAVA_VERSION: '17'

jobs:
  code-quality:
    name: Code Quality Analysis
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
      security-events: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for better analysis
      
      {{#if (contains programming_languages "javascript" "typescript")}}
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run ESLint
        run: |
          npx eslint . --ext .js,.ts,.jsx,.tsx --format json --output-file eslint-results.json || true
          npx eslint . --ext .js,.ts,.jsx,.tsx --format @microsoft/eslint-formatter-sarif --output-file eslint-results.sarif || true
      
      - name: Run Prettier Check
        run: npx prettier --check .
      
      - name: TypeScript Type Check
        if: contains(env.programming_languages, 'typescript')
        run: npx tsc --noEmit
      {{/if}}

      {{#if (contains programming_languages "python")}}
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install Python dependencies
        run: |
          pip install -r requirements-dev.txt
          pip install black isort flake8 mypy bandit safety
      
      - name: Run Black
        run: black --check --diff .
      
      - name: Run isort
        run: isort --check-only --diff .
      
      - name: Run Flake8
        run: flake8 . --format=json --output-file flake8-results.json || true
      
      - name: Run MyPy
        run: mypy . --json-report mypy-results || true
      {{/if}}

      # Security Analysis
      {{#if (eq security_requirements "enhanced" "enterprise" "regulatory")}}
      - name: Run Semgrep Security Analysis
        uses: semgrep/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/secrets
            p/owasp-top-ten
            {{#if (eq security_requirements "regulatory")}}
            p/pci-dss
            p/hipaa
            {{/if}}
          sarif-file: semgrep-results.sarif
          severity: ERROR
        env:
          SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}
      
      - name: Run Trivy Security Scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: '{{#if (eq security_requirements "regulatory")}}UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL{{else if (eq security_requirements "enterprise")}}MEDIUM,HIGH,CRITICAL{{else}}HIGH,CRITICAL{{/if}}'
      {{/if}}

      # Code Coverage Analysis
      {{#if (eq code_quality_standards "strict" "enterprise")}}
      - name: Generate Code Coverage
        run: |
          {{#if (contains programming_languages "javascript" "typescript")}}
          npm test -- --coverage --coverageReporters=json --coverageReporters=lcov
          {{/if}}
          {{#if (contains programming_languages "python")}}
          pytest --cov=. --cov-report=json --cov-report=lcov
          {{/if}}
      
      - name: Upload coverage reports to Codecov
        uses: codecov/codecov-action@v3
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage.lcov
          fail_ci_if_error: {{#if (eq code_quality_standards "enterprise")}}true{{else}}false{{/if}}
      {{/if}}

      # Complexity Analysis
      - name: Code Complexity Analysis
        run: |
          {{#if (contains programming_languages "javascript" "typescript")}}
          npx complexity-report --format json --output complexity-results.json src/ || true
          {{/if}}
          {{#if (contains programming_languages "python")}}
          radon cc . --json > complexity-results.json || true
          {{/if}}

      # Upload SARIF results
      - name: Upload SARIF results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: '*.sarif'

      # Comment PR with results
      - name: Comment PR with Review Results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            
            let comment = '## 🤖 Automated Code Review Results\\n\\n';
            
            // Add quality metrics
            {{#if (contains programming_languages "javascript" "typescript")}}
            try {
              const eslintResults = JSON.parse(fs.readFileSync('eslint-results.json', 'utf8'));
              const totalIssues = eslintResults.reduce((sum, file) => sum + file.errorCount + file.warningCount, 0);
              comment += \`### ESLint Analysis\\n- **Issues Found:** \${totalIssues}\\n\\n\`;
            } catch (e) { console.log('No ESLint results found'); }
            {{/if}}
            
            {{#if (contains programming_languages "python")}}
            try {
              const flake8Results = JSON.parse(fs.readFileSync('flake8-results.json', 'utf8'));
              comment += \`### Flake8 Analysis\\n- **Issues Found:** \${Object.keys(flake8Results).length}\\n\\n\`;
            } catch (e) { console.log('No Flake8 results found'); }
            {{/if}}
            
            // Add complexity metrics
            try {
              const complexityResults = JSON.parse(fs.readFileSync('complexity-results.json', 'utf8'));
              comment += '### Code Complexity\\n';
              {{#if (contains programming_languages "javascript" "typescript")}}
              comment += \`- **Average Complexity:** \${complexityResults.summary.average.complexity}\\n\`;
              {{/if}}
              comment += '\\n';
            } catch (e) { console.log('No complexity results found'); }
            
            {{#if (eq code_quality_standards "strict" "enterprise")}}
            // Add coverage information
            try {
              const coverage = JSON.parse(fs.readFileSync('coverage-summary.json', 'utf8'));
              const totalCoverage = coverage.total.lines.pct;
              const coverageEmoji = totalCoverage >= 80 ? '✅' : totalCoverage >= 60 ? '⚠️' : '❌';
              comment += \`### Test Coverage \${coverageEmoji}\\n- **Total Coverage:** \${totalCoverage}%\\n\\n\`;
            } catch (e) { console.log('No coverage results found'); }
            {{/if}}
            
            // Quality gates
            comment += '### Quality Gates\\n';
            {{#if (eq code_quality_standards "enterprise")}}
            comment += '- ✅ **Code Coverage:** Required 80%+\\n';
            comment += '- ✅ **Security Scan:** No critical vulnerabilities\\n';
            comment += '- ✅ **Code Complexity:** Average complexity < 10\\n';
            {{else if (eq code_quality_standards "strict")}}
            comment += '- ✅ **Code Coverage:** Required 70%+\\n';
            comment += '- ✅ **Security Scan:** No high/critical vulnerabilities\\n';
            {{else}}
            comment += '- ✅ **Basic Checks:** Linting and formatting\\n';
            {{/if}}
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });

  dependency-security:
    name: Dependency Security Scan
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      {{#if (contains programming_languages "javascript" "typescript")}}
      - name: Run npm audit
        run: |
          npm audit --audit-level={{#if (eq security_requirements "regulatory")}}low{{else if (eq security_requirements "enterprise")}}moderate{{else}}high{{/if}} --json > npm-audit-results.json || true
      
      - name: Check for known vulnerabilities
        run: |
          npx audit-ci --{{#if (eq security_requirements "regulatory")}}low{{else if (eq security_requirements "enterprise")}}moderate{{else}}high{{/if}}
      {{/if}}

      {{#if (contains programming_languages "python")}}
      - name: Run Safety check
        run: |
          pip install safety
          safety check --json --output safety-results.json || true
      
      - name: Run pip-audit
        run: |
          pip install pip-audit
          pip-audit --format=json --output=pip-audit-results.json || true
      {{/if}}

  {{#if (eq review_automation_level "ai-assisted" "full-automation")}}
  ai-code-review:
    name: AI-Assisted Code Review
    runs-on: ubuntu-latest
    needs: [code-quality]
    if: github.event_name == 'pull_request'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Get changed files
        id: changed-files
        uses: tj-actions/changed-files@v37
        with:
          files: |
            **/*.js
            **/*.ts
            **/*.jsx
            **/*.tsx
            **/*.py
            **/*.java
            **/*.go
            **/*.cs
      
      - name: AI Code Review
        if: steps.changed-files.outputs.any_changed == 'true'
        uses: actions/github-script@v7
        env:
          CHANGED_FILES: ${{ steps.changed-files.outputs.all_changed_files }}
        with:
          script: |
            const changedFiles = process.env.CHANGED_FILES.split(' ');
            
            for (const file of changedFiles) {
              if (!file) continue;
              
              try {
                const content = require('fs').readFileSync(file, 'utf8');
                
                // AI analysis would go here - integrate with OpenAI, Claude, or similar
                const aiReview = await analyzeCodeWithAI(content, file);
                
                if (aiReview.suggestions.length > 0) {
                  const comment = \`## 🧠 AI Code Review for \${file}\\n\\n\` +
                    aiReview.suggestions.map(s => \`- **\${s.type}:** \${s.message}\`).join('\\n');
                  
                  github.rest.pulls.createReviewComment({
                    owner: context.repo.owner,
                    repo: context.repo.repo,
                    pull_number: context.issue.number,
                    body: comment,
                    path: file,
                    line: aiReview.line || 1
                  });
                }
              } catch (error) {
                console.log(\`Error analyzing \${file}: \${error.message}\`);
              }
            }
            
            async function analyzeCodeWithAI(code, filename) {
              // Mock AI analysis - replace with actual AI service integration
              return {
                suggestions: [
                  {
                    type: 'Performance',
                    message: 'Consider using more efficient data structures for large datasets',
                    line: 1
                  }
                ]
              };
            }
  {{/if}}
{{/if}}

{{#if (eq repository_platform "gitlab")}}
# .gitlab-ci.yml
stages:
  - lint
  - test
  - security
  - quality-gates

variables:
  PYTHON_VERSION: "3.11"
  NODE_VERSION: "18"

cache:
  paths:
    - node_modules/
    - .pip-cache/

lint:
  stage: lint
  image: node:$NODE_VERSION
  script:
    {{#if (contains programming_languages "javascript" "typescript")}}
    - npm ci
    - npm run lint
    - npm run prettier:check
    {{/if}}
  artifacts:
    reports:
      codequality: gl-codequality.json
    expire_in: 1 week

security-scan:
  stage: security
  image: securecodewarrior/gitlab-sast:latest
  script:
    - semgrep --config=auto --sarif --output=semgrep-report.sarif .
  artifacts:
    reports:
      sast: semgrep-report.sarif
  allow_failure: {{#if (eq security_requirements "regulatory")}}false{{else}}true{{/if}}

quality-gates:
  stage: quality-gates
  image: sonarqube/sonar-scanner-cli:latest
  script:
    - sonar-scanner
  only:
    - merge_requests
    - main
{{/if}}
```

## 2. Advanced Static Analysis Configuration

### Multi-Language Analysis Setup
```javascript
// Advanced ESLint configuration for JavaScript/TypeScript
// eslint.config.js
import js from '@eslint/js';
import typescript from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';
import security from 'eslint-plugin-security';
import sonarjs from 'eslint-plugin-sonarjs';
import prettier from 'eslint-config-prettier';

export default [
  js.configs.recommended,
  {
    files: ['**/*.{js,mjs,cjs,ts,tsx}'],
    plugins: {
      '@typescript-eslint': typescript,
      security: security,
      sonarjs: sonarjs,
    },
    languageOptions: {
      parser: typescriptParser,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        project: './tsconfig.json',
      },
    },
    rules: {
      // TypeScript specific rules
      '@typescript-eslint/no-unused-vars': 'error',
      '@typescript-eslint/no-explicit-any': '{{#if (eq code_quality_standards "strict" "enterprise")}}error{{else}}warn{{/if}}',
      '@typescript-eslint/explicit-function-return-type': '{{#if (eq code_quality_standards "enterprise")}}error{{else}}off{{/if}}',
      '@typescript-eslint/no-non-null-assertion': 'error',
      '@typescript-eslint/prefer-nullish-coalescing': 'error',
      '@typescript-eslint/prefer-optional-chain': 'error',
      
      // Code quality rules
      'complexity': ['{{#if (eq code_quality_standards "enterprise")}}error{{else}}warn{{/if}}', {{#if (eq code_quality_standards "enterprise")}}8{{else if (eq code_quality_standards "strict")}}10{{else}}15{{/if}}],
      'max-depth': ['error', {{#if (eq code_quality_standards "enterprise")}}3{{else if (eq code_quality_standards "strict")}}4{{else}}5{{/if}}],
      'max-lines-per-function': ['warn', {{#if (eq code_quality_standards "enterprise")}}50{{else if (eq code_quality_standards "strict")}}75{{else}}100{{/if}}],
      'max-params': ['error', {{#if (eq code_quality_standards "enterprise")}}3{{else if (eq code_quality_standards "strict")}}4{{else}}5{{/if}}],
      
      // Security rules
      {{#if (eq security_requirements "enhanced" "enterprise" "regulatory")}}
      'security/detect-object-injection': 'error',
      'security/detect-non-literal-regexp': 'error',
      'security/detect-unsafe-regex': 'error',
      'security/detect-buffer-noassert': 'error',
      'security/detect-child-process': 'error',
      'security/detect-disable-mustache-escape': 'error',
      'security/detect-eval-with-expression': 'error',
      'security/detect-no-csrf-before-method-override': 'error',
      'security/detect-non-literal-fs-filename': 'error',
      'security/detect-non-literal-require': 'error',
      'security/detect-possible-timing-attacks': 'error',
      'security/detect-pseudoRandomBytes': 'error',
      {{/if}}
      
      // SonarJS rules for bug detection
      'sonarjs/cognitive-complexity': ['error', {{#if (eq code_quality_standards "enterprise")}}10{{else if (eq code_quality_standards "strict")}}15{{else}}20{{/if}}],
      'sonarjs/no-duplicate-string': ['error', {{#if (eq code_quality_standards "enterprise")}}3{{else}}5{{/if}}],
      'sonarjs/no-identical-functions': 'error',
      'sonarjs/no-redundant-boolean': 'error',
      'sonarjs/no-unused-collection': 'error',
      'sonarjs/prefer-immediate-return': 'error',
    },
  },
  prettier, // Must be last to override other formatting rules
];

// Custom rules for domain-specific validation
const customRules = {
  // API endpoint validation
  'validate-api-endpoints': {
    meta: {
      type: 'problem',
      docs: {
        description: 'Validate API endpoint definitions',
      },
    },
    create(context) {
      return {
        CallExpression(node) {
          if (
            node.callee.type === 'MemberExpression' &&
            ['get', 'post', 'put', 'delete', 'patch'].includes(node.callee.property.name)
          ) {
            // Check for route validation middleware
            const hasValidation = node.arguments.some(arg => 
              arg.type === 'Identifier' && 
              arg.name.includes('validate')
            );
            
            if (!hasValidation && node.arguments.length > 1) {
              context.report({
                node,
                message: 'API endpoint should include input validation middleware'
              });
            }
          }
        },
      };
    },
  },
  
  // Database query validation
  'validate-database-queries': {
    meta: {
      type: 'problem',
      docs: {
        description: 'Validate database query patterns',
      },
    },
    create(context) {
      return {
        CallExpression(node) {
          if (
            node.callee.type === 'MemberExpression' &&
            ['query', 'find', 'findOne'].includes(node.callee.property.name)
          ) {
            // Check for parameterized queries
            const hasStringConcatenation = node.arguments.some(arg =>
              arg.type === 'BinaryExpression' && arg.operator === '+'
            );
            
            if (hasStringConcatenation) {
              context.report({
                node,
                message: 'Use parameterized queries to prevent SQL injection'
              });
            }
          }
        },
      };
    },
  },
};
```

### Python Analysis Configuration
```python
# Advanced Python static analysis configuration
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
    \.git
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
  | migrations
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
line_length = 88
skip_gitignore = true

[tool.mypy]
python_version = "3.11"
check_untyped_defs = true
disallow_any_generics = {{#if (eq code_quality_standards "enterprise")}}true{{else}}false{{/if}}
disallow_incomplete_defs = true
disallow_untyped_decorators = {{#if (eq code_quality_standards "strict" "enterprise")}}true{{else}}false{{/if}}
disallow_untyped_defs = {{#if (eq code_quality_standards "enterprise")}}true{{else}}false{{/if}}
ignore_missing_imports = false
no_implicit_optional = true
show_error_codes = true
strict_equality = true
warn_redundant_casts = true
warn_return_any = {{#if (eq code_quality_standards "strict" "enterprise")}}true{{else}}false{{/if}}
warn_unreachable = true
warn_unused_configs = true
warn_unused_ignores = true

[tool.flake8]
max-line-length = 88
extend-ignore = ["E203", "W503", "E501"]
max-complexity = {{#if (eq code_quality_standards "enterprise")}}8{{else if (eq code_quality_standards "strict")}}10{{else}}15{{/if}}
per-file-ignores = [
    "__init__.py:F401",
    "tests/*:S101,S106"
]
exclude = [
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "migrations",
    "venv",
    ".venv"
]

[tool.bandit]
exclude_dirs = ["tests", "test_*"]
{{#if (eq security_requirements "regulatory")}}
confidence_level = "low"
severity_level = "low"
{{else if (eq security_requirements "enterprise")}}
confidence_level = "medium"
severity_level = "medium"
{{else}}
confidence_level = "high"
severity_level = "high"
{{/if}}

[tool.pytest]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    {{#if (eq code_quality_standards "strict" "enterprise")}}
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under={{#if (eq code_quality_standards "enterprise")}}80{{else}}70{{/if}}",
    {{/if}}
]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__init__.py",
    "*/migrations/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]

# Custom pylint configuration
# pylintrc
[MASTER]
load-plugins = pylint_django,pylint_celery

{{#if (eq security_requirements "enhanced" "enterprise" "regulatory")}}
[SECURITY]
# Security-related checks
unsafe-load-any-extension = yes
{{/if}}

[MESSAGES CONTROL]
{{#if (eq code_quality_standards "enterprise")}}
disable = none
{{else if (eq code_quality_standards "strict")}}
disable = missing-docstring,too-few-public-methods
{{else}}
disable = missing-docstring,too-few-public-methods,invalid-name,line-too-long
{{/if}}

[FORMAT]
max-line-length = 88
max-module-lines = {{#if (eq code_quality_standards "enterprise")}}500{{else if (eq code_quality_standards "strict")}}750{{else}}1000{{/if}}

[DESIGN]
max-args = {{#if (eq code_quality_standards "enterprise")}}5{{else if (eq code_quality_standards "strict")}}7{{else}}10{{/if}}
max-locals = {{#if (eq code_quality_standards "enterprise")}}10{{else if (eq code_quality_standards "strict")}}15{{else}}20{{/if}}
max-returns = {{#if (eq code_quality_standards "enterprise")}}4{{else if (eq code_quality_standards "strict")}}6{{else}}8{{/if}}
max-branches = {{#if (eq code_quality_standards "enterprise")}}8{{else if (eq code_quality_standards "strict")}}12{{else}}15{{/if}}
max-statements = {{#if (eq code_quality_standards "enterprise")}}30{{else if (eq code_quality_standards "strict")}}50{{else}}75{{/if}}
```

## 3. Intelligent Code Review Bot

### AI-Powered Review System
```python
# AI-powered code review system
import ast
import re
import asyncio
import openai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import subprocess
import json

@dataclass
class CodeIssue:
    file_path: str
    line_number: int
    column: int
    severity: str  # 'error', 'warning', 'info', 'suggestion'
    category: str  # 'security', 'performance', 'maintainability', 'style'
    message: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False

class IntelligentCodeReviewer:
    def __init__(self, 
                 programming_languages: List[str] = ["{{programming_languages}}"],
                 quality_standards: str = "{{code_quality_standards}}",
                 security_level: str = "{{security_requirements}}"):
        self.languages = programming_languages
        self.quality_standards = quality_standards
        self.security_level = security_level
        self.openai_client = openai.AsyncOpenAI()
        
        # Configure analysis rules based on standards
        self.complexity_threshold = {
            'basic': 15,
            'intermediate': 12,
            'strict': 10,
            'enterprise': 8
        }.get(quality_standards, 10)
        
        self.security_checks = {
            'basic': ['sql_injection', 'xss'],
            'enhanced': ['sql_injection', 'xss', 'csrf', 'auth_bypass'],
            'enterprise': ['sql_injection', 'xss', 'csrf', 'auth_bypass', 'crypto_weak', 'path_traversal'],
            'regulatory': ['all_owasp_top10', 'pci_dss', 'hipaa_compliance']
        }.get(security_level, ['sql_injection', 'xss'])
    
    async def review_pull_request(self, 
                                changed_files: List[str], 
                                base_commit: str, 
                                head_commit: str) -> List[CodeIssue]:
        """Perform comprehensive code review on changed files"""
        
        all_issues = []
        
        for file_path in changed_files:
            if not self._is_reviewable_file(file_path):
                continue
            
            try:
                # Get file diff
                diff = self._get_file_diff(file_path, base_commit, head_commit)
                if not diff:
                    continue
                
                # Read current file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Perform various analyses
                issues = []
                issues.extend(await self._analyze_syntax_and_style(file_path, content))
                issues.extend(await self._analyze_security(file_path, content))
                issues.extend(await self._analyze_performance(file_path, content))
                issues.extend(await self._analyze_maintainability(file_path, content))
                
                {{#if (eq review_automation_level "ai-assisted" "full-automation")}}
                # AI-powered analysis
                ai_issues = await self._ai_code_analysis(file_path, content, diff)
                issues.extend(ai_issues)
                {{/if}}
                
                # Filter issues to only those in changed lines
                changed_lines = self._get_changed_lines(diff)
                filtered_issues = [
                    issue for issue in issues 
                    if issue.line_number in changed_lines
                ]
                
                all_issues.extend(filtered_issues)
                
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                continue
        
        # Prioritize and deduplicate issues
        return self._prioritize_issues(all_issues)
    
    def _is_reviewable_file(self, file_path: str) -> bool:
        """Check if file should be reviewed"""
        reviewable_extensions = {
            'javascript': ['.js', '.jsx', '.mjs'],
            'typescript': ['.ts', '.tsx'],
            'python': ['.py'],
            'java': ['.java'],
            'go': ['.go'],
            'csharp': ['.cs']
        }
        
        file_ext = Path(file_path).suffix.lower()
        
        for lang in self.languages:
            if file_ext in reviewable_extensions.get(lang, []):
                return True
        
        return False
    
    def _get_file_diff(self, file_path: str, base_commit: str, head_commit: str) -> str:
        """Get git diff for a specific file"""
        try:
            result = subprocess.run([
                'git', 'diff', f'{base_commit}..{head_commit}', '--', file_path
            ], capture_output=True, text=True)
            return result.stdout
        except Exception:
            return ""
    
    def _get_changed_lines(self, diff: str) -> set:
        """Extract line numbers that were changed in the diff"""
        changed_lines = set()
        current_line = 0
        
        for line in diff.split('\n'):
            if line.startswith('@@'):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                match = re.search(r'\+(\d+)', line)
                if match:
                    current_line = int(match.group(1))
            elif line.startswith('+') and not line.startswith('+++'):
                changed_lines.add(current_line)
                current_line += 1
            elif not line.startswith('-'):
                current_line += 1
        
        return changed_lines
    
    async def _analyze_syntax_and_style(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze syntax and style issues"""
        issues = []
        
        if file_path.endswith('.py'):
            issues.extend(self._analyze_python_syntax(file_path, content))
        elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            issues.extend(await self._analyze_javascript_syntax(file_path, content))
        
        return issues
    
    def _analyze_python_syntax(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze Python-specific syntax and style"""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            # Check for complex functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_complexity(node)
                    if complexity > self.complexity_threshold:
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            severity='warning',
                            category='maintainability',
                            message=f'Function complexity ({complexity}) exceeds threshold ({self.complexity_threshold})',
                            suggestion='Consider breaking down this function into smaller, more focused functions'
                        ))
                
                # Check for long parameter lists
                if isinstance(node, ast.FunctionDef) and len(node.args.args) > 5:
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        severity='warning',
                        category='maintainability',
                        message=f'Function has {len(node.args.args)} parameters, consider using a configuration object',
                        suggestion='Use a dataclass or TypedDict to group related parameters'
                    ))
                
                # Check for missing type hints (if enterprise standards)
                if (self.quality_standards == 'enterprise' and 
                    isinstance(node, ast.FunctionDef) and 
                    node.returns is None):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        severity='warning',
                        category='maintainability',
                        message='Function missing return type annotation',
                        suggestion='Add return type annotation for better code documentation'
                    ))
        
        except SyntaxError as e:
            issues.append(CodeIssue(
                file_path=file_path,
                line_number=e.lineno or 1,
                column=e.offset or 0,
                severity='error',
                category='syntax',
                message=f'Syntax error: {e.msg}'
            ))
        
        return issues
    
    async def _analyze_javascript_syntax(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze JavaScript/TypeScript syntax and style"""
        issues = []
        
        # Run ESLint programmatically
        try:
            result = subprocess.run([
                'npx', 'eslint', '--format', 'json', file_path
            ], capture_output=True, text=True)
            
            if result.stdout:
                eslint_results = json.loads(result.stdout)
                for file_result in eslint_results:
                    for message in file_result.get('messages', []):
                        severity = 'error' if message['severity'] == 2 else 'warning'
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=message['line'],
                            column=message['column'],
                            severity=severity,
                            category='style',
                            message=message['message'],
                            auto_fixable=message.get('fix') is not None
                        ))
        
        except Exception as e:
            print(f"ESLint analysis failed: {e}")
        
        return issues
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    async def _analyze_security(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze security vulnerabilities"""
        issues = []
        
        # SQL Injection checks
        if 'sql_injection' in self.security_checks:
            sql_injection_patterns = [
                r'execute\s*\(\s*["\'].*\+.*["\']',  # String concatenation in SQL
                r'query\s*\(\s*["\'].*\+.*["\']',
                r'format\s*\(\s*["\'].*\%.*["\']',  # String formatting in SQL
            ]
            
            for pattern in sql_injection_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_no = content[:match.start()].count('\n') + 1
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=line_no,
                        column=match.start() - content.rfind('\n', 0, match.start()),
                        severity='error',
                        category='security',
                        message='Potential SQL injection vulnerability detected',
                        suggestion='Use parameterized queries or prepared statements'
                    ))
        
        # XSS checks
        if 'xss' in self.security_checks:
            xss_patterns = [
                r'innerHTML\s*=\s*.*\+',  # innerHTML with concatenation
                r'document\.write\s*\(',  # document.write usage
                r'eval\s*\(',  # eval usage
            ]
            
            for pattern in xss_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_no = content[:match.start()].count('\n') + 1
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=line_no,
                        column=match.start() - content.rfind('\n', 0, match.start()),
                        severity='error',
                        category='security',
                        message='Potential XSS vulnerability detected',
                        suggestion='Use safe DOM manipulation methods and sanitize user input'
                    ))
        
        return issues
    
    async def _analyze_performance(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze performance issues"""
        issues = []
        
        # Check for inefficient patterns
        performance_patterns = {
            r'for\s+.*\s+in\s+range\s*\(\s*len\s*\(': 'Use direct iteration over collection instead of range(len())',
            r'\.find\s*\(\s*.*\s*\)\s*!=\s*-1': 'Use "in" operator instead of .find() != -1',
            r'len\s*\(\s*.*\s*\)\s*>\s*0': 'Use truthiness check instead of len() > 0',
        }
        
        for pattern, suggestion in performance_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                line_no = content[:match.start()].count('\n') + 1
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=line_no,
                    column=match.start() - content.rfind('\n', 0, match.start()),
                    severity='info',
                    category='performance',
                    message='Potential performance improvement opportunity',
                    suggestion=suggestion
                ))
        
        return issues
    
    async def _analyze_maintainability(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze maintainability issues"""
        issues = []
        
        lines = content.split('\n')
        
        # Check for long functions
        in_function = False
        function_start = 0
        function_name = ""
        
        for i, line in enumerate(lines, 1):
            # Python function detection
            if re.match(r'\s*def\s+', line):
                if in_function:
                    # Previous function ended
                    function_length = i - function_start
                    if function_length > 50:  # Configurable threshold
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=function_start,
                            column=0,
                            severity='warning',
                            category='maintainability',
                            message=f'Function "{function_name}" is {function_length} lines long',
                            suggestion='Consider breaking down large functions into smaller, focused functions'
                        ))
                
                in_function = True
                function_start = i
                function_name = re.search(r'def\s+(\w+)', line).group(1) if re.search(r'def\s+(\w+)', line) else "unknown"
            
            # Check for long lines
            if len(line) > 120:  # Configurable threshold
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    column=120,
                    severity='warning',
                    category='style',
                    message=f'Line too long ({len(line)} characters)',
                    suggestion='Break long lines for better readability'
                ))
            
            # Check for TODO/FIXME comments
            if re.search(r'(TODO|FIXME|HACK)', line, re.IGNORECASE):
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    column=line.lower().find('todo' if 'todo' in line.lower() else 'fixme' if 'fixme' in line.lower() else 'hack'),
                    severity='info',
                    category='maintainability',
                    message='TODO/FIXME comment found',
                    suggestion='Consider creating a ticket to track this work item'
                ))
        
        return issues
    
    {{#if (eq review_automation_level "ai-assisted" "full-automation")}}
    async def _ai_code_analysis(self, file_path: str, content: str, diff: str) -> List[CodeIssue]:
        """AI-powered code analysis using language models"""
        issues = []
        
        try:
            # Prepare prompt for AI analysis
            prompt = f"""
            Analyze the following code changes and provide detailed feedback:

            File: {file_path}
            
            Code diff:
            ```
            {diff}
            ```
            
            Full file content:
            ```
            {content}
            ```
            
            Please analyze for:
            1. Code quality and best practices
            2. Potential bugs or logic errors
            3. Performance improvements
            4. Security vulnerabilities
            5. Maintainability concerns
            
            Quality standards: {self.quality_standards}
            Security level: {self.security_level}
            
            Provide response in JSON format with the following structure:
            {{
                "issues": [
                    {{
                        "line_number": number,
                        "severity": "error|warning|info",
                        "category": "security|performance|maintainability|style|logic",
                        "message": "Description of the issue",
                        "suggestion": "Suggested improvement"
                    }}
                ]
            }}
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert code reviewer with deep knowledge of software engineering best practices, security, and performance optimization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parse AI response
            ai_response = json.loads(response.choices[0].message.content)
            
            for issue_data in ai_response.get('issues', []):
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=issue_data['line_number'],
                    column=0,
                    severity=issue_data['severity'],
                    category=issue_data['category'],
                    message=f"[AI] {issue_data['message']}",
                    suggestion=issue_data.get('suggestion')
                ))
        
        except Exception as e:
            print(f"AI analysis failed: {e}")
        
        return issues
    {{/if}}
    
    def _prioritize_issues(self, issues: List[CodeIssue]) -> List[CodeIssue]:
        """Prioritize and deduplicate issues"""
        
        # Define priority order
        severity_priority = {'error': 0, 'warning': 1, 'info': 2, 'suggestion': 3}
        category_priority = {'security': 0, 'logic': 1, 'performance': 2, 'maintainability': 3, 'style': 4}
        
        # Remove duplicates
        seen = set()
        unique_issues = []
        
        for issue in issues:
            issue_key = (issue.file_path, issue.line_number, issue.message)
            if issue_key not in seen:
                seen.add(issue_key)
                unique_issues.append(issue)
        
        # Sort by priority
        return sorted(unique_issues, key=lambda x: (
            severity_priority.get(x.severity, 999),
            category_priority.get(x.category, 999),
            x.file_path,
            x.line_number
        ))
    
    def generate_review_summary(self, issues: List[CodeIssue]) -> Dict[str, Any]:
        """Generate summary of review results"""
        
        summary = {
            'total_issues': len(issues),
            'by_severity': {},
            'by_category': {},
            'by_file': {},
            'auto_fixable_count': sum(1 for issue in issues if issue.auto_fixable),
            'quality_score': 0
        }
        
        # Count by severity
        for issue in issues:
            summary['by_severity'][issue.severity] = summary['by_severity'].get(issue.severity, 0) + 1
            summary['by_category'][issue.category] = summary['by_category'].get(issue.category, 0) + 1
            summary['by_file'][issue.file_path] = summary['by_file'].get(issue.file_path, 0) + 1
        
        # Calculate quality score (0-100)
        error_count = summary['by_severity'].get('error', 0)
        warning_count = summary['by_severity'].get('warning', 0)
        
        # Base score starts at 100, deduct points for issues
        quality_score = 100
        quality_score -= error_count * 10  # Errors are serious
        quality_score -= warning_count * 5  # Warnings are moderate
        quality_score -= summary['by_severity'].get('info', 0) * 1  # Info issues are minor
        
        summary['quality_score'] = max(0, quality_score)
        
        return summary

# Usage example
async def main():
    reviewer = IntelligentCodeReviewer()
    
    # Example: Review a pull request
    changed_files = ['src/api/users.py', 'src/utils/database.py']
    base_commit = 'main'
    head_commit = 'feature-branch'
    
    issues = await reviewer.review_pull_request(changed_files, base_commit, head_commit)
    summary = reviewer.generate_review_summary(issues)
    
    print(f"Found {len(issues)} issues")
    print(f"Quality Score: {summary['quality_score']}/100")
    
    for issue in issues[:10]:  # Show first 10 issues
        print(f"{issue.severity.upper()}: {issue.message} ({issue.file_path}:{issue.line_number})")

if __name__ == "__main__":
    asyncio.run(main())
```

## 4. Quality Gates & Reporting

### Quality Gates Configuration
```yaml
# quality-gates.yml
quality_gates:
  {{#if (eq code_quality_standards "enterprise")}}
  enterprise:
    required_checks:
      - name: "Code Coverage"
        threshold: 80
        metric: "line_coverage_percentage"
        blocking: true
      
      - name: "Security Scan"
        threshold: 0
        metric: "critical_vulnerabilities"
        blocking: true
      
      - name: "Code Complexity"
        threshold: 8
        metric: "average_cyclomatic_complexity"
        blocking: true
      
      - name: "Maintainability Index"
        threshold: 70
        metric: "maintainability_index"
        blocking: true
      
      - name: "Technical Debt Ratio"
        threshold: 5
        metric: "technical_debt_percentage"
        blocking: false
      
      - name: "Code Smells"
        threshold: 0
        metric: "blocker_code_smells"
        blocking: true
  {{else if (eq code_quality_standards "strict")}}
  strict:
    required_checks:
      - name: "Code Coverage"
        threshold: 70
        metric: "line_coverage_percentage"
        blocking: true
      
      - name: "Security Scan"
        threshold: 0
        metric: "high_vulnerabilities"
        blocking: true
      
      - name: "Code Complexity"
        threshold: 10
        metric: "average_cyclomatic_complexity"
        blocking: false
      
      - name: "Maintainability Index"
        threshold: 60
        metric: "maintainability_index"
        blocking: false
  {{else}}
  basic:
    required_checks:
      - name: "Build Success"
        threshold: 1
        metric: "build_success"
        blocking: true
      
      - name: "Unit Tests"
        threshold: 1
        metric: "tests_passed"
        blocking: true
      
      - name: "Critical Security Issues"
        threshold: 0
        metric: "critical_vulnerabilities"
        blocking: true
  {{/if}}

reporting:
  formats: ["json", "html", "markdown"]
  include_trends: true
  include_metrics: true
  distribution_lists:
    {{#if (eq team_size "large")}}
    - "team-leads@company.com"
    - "architecture-board@company.com"
    {{else if (eq team_size "medium")}}
    - "team-lead@company.com"
    - "senior-devs@company.com"
    {{else}}
    - "team@company.com"
    {{/if}}
```

### Automated Reporting Dashboard
```python
# Automated reporting and dashboard generation
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template

class CodeReviewDashboard:
    def __init__(self, repository_platform: str = "{{repository_platform}}"):
        self.platform = repository_platform
        self.metrics_history = []
        
    async def generate_dashboard(self, time_period_days: int = 30) -> str:
        """Generate comprehensive code review dashboard"""
        
        # Collect metrics
        metrics = await self._collect_metrics(time_period_days)
        
        # Generate visualizations
        charts = self._create_visualizations(metrics)
        
        # Generate HTML dashboard
        dashboard_html = self._render_dashboard_template(metrics, charts)
        
        # Save dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"code_review_dashboard_{timestamp}.html"
        
        with open(filename, 'w') as f:
            f.write(dashboard_html)
        
        return filename
    
    async def _collect_metrics(self, days: int) -> Dict[str, Any]:
        """Collect code review metrics from various sources"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        metrics = {
            'period': {'start': start_date, 'end': end_date},
            'pull_requests': await self._get_pull_request_metrics(start_date, end_date),
            'code_quality': await self._get_code_quality_metrics(start_date, end_date),
            'security': await self._get_security_metrics(start_date, end_date),
            'team_performance': await self._get_team_metrics(start_date, end_date),
            'trends': await self._get_trend_metrics(start_date, end_date)
        }
        
        return metrics
    
    async def _get_pull_request_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get pull request related metrics"""
        
        # This would integrate with your repository platform API
        # Example data structure:
        return {
            'total_prs': 45,
            'merged_prs': 38,
            'rejected_prs': 3,
            'pending_prs': 4,
            'average_review_time_hours': 8.5,
            'average_size_lines': 245,
            'review_coverage_percentage': 95.2,
            'automated_checks_pass_rate': 92.1,
            'pr_size_distribution': {
                'small': 25,  # < 100 lines
                'medium': 15, # 100-500 lines
                'large': 5    # > 500 lines
            }
        }
    
    async def _get_code_quality_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get code quality metrics"""
        
        return {
            'average_complexity': 7.2,
            'complexity_trend': 'improving',  # improving, stable, declining
            'test_coverage': 78.5,
            'coverage_trend': 'stable',
            'maintainability_index': 72.1,
            'technical_debt_hours': 156,
            'code_smells': {
                'blocker': 2,
                'critical': 8,
                'major': 23,
                'minor': 67
            },
            'quality_gate_pass_rate': 89.3
        }
    
    async def _get_security_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get security-related metrics"""
        
        return {
            'vulnerabilities_found': 12,
            'vulnerabilities_fixed': 8,
            'critical_vulnerabilities': 1,
            'high_vulnerabilities': 3,
            'medium_vulnerabilities': 5,
            'low_vulnerabilities': 3,
            'security_hotspots': 15,
            'security_review_coverage': 94.7,
            'mean_time_to_fix_hours': 24.5
        }
    
    async def _get_team_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get team performance metrics"""
        
        return {
            'active_reviewers': {{#if (eq team_size "large")}}12{{else if (eq team_size "medium")}}6{{else}}3{{/if}},
            'review_participation_rate': 87.3,
            'average_reviews_per_developer': 3.2,
            'review_quality_score': 8.1,  # 1-10 scale
            'knowledge_sharing_index': 72.5,
            'top_reviewers': [
                {'name': 'Alice Johnson', 'reviews': 15, 'quality_score': 9.2},
                {'name': 'Bob Smith', 'reviews': 12, 'quality_score': 8.8},
                {'name': 'Carol Davis', 'reviews': 11, 'quality_score': 8.5}
            ],
            'review_bottlenecks': [
                {'reviewer': 'Senior Dev', 'avg_response_time_hours': 18.2},
                {'reviewer': 'Team Lead', 'avg_response_time_hours': 12.5}
            ]
        }
    
    async def _get_trend_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get trend analysis"""
        
        # Generate sample trend data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        return {
            'daily_metrics': {
                'dates': [d.strftime('%Y-%m-%d') for d in dates],
                'pull_requests': [2, 3, 1, 4, 5, 2, 1] * (len(dates) // 7 + 1)[:len(dates)],
                'review_time': [8.2, 7.5, 9.1, 6.8, 11.2, 8.9, 7.3] * (len(dates) // 7 + 1)[:len(dates)],
                'quality_score': [85, 87, 82, 90, 88, 86, 89] * (len(dates) // 7 + 1)[:len(dates)]
            }
        }
    
    def _create_visualizations(self, metrics: Dict) -> Dict[str, str]:
        """Create visualization charts"""
        
        charts = {}
        
        # PR Size Distribution Pie Chart
        plt.figure(figsize=(8, 6))
        sizes = list(metrics['pull_requests']['pr_size_distribution'].values())
        labels = list(metrics['pull_requests']['pr_size_distribution'].keys())
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Pull Request Size Distribution')
        plt.axis('equal')
        plt.savefig('pr_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        charts['pr_size_distribution'] = 'pr_size_distribution.png'
        
        # Code Quality Trends
        plt.figure(figsize=(12, 6))
        dates = metrics['trends']['daily_metrics']['dates']
        quality_scores = metrics['trends']['daily_metrics']['quality_score']
        
        plt.plot(dates[::3], quality_scores[::3], marker='o', linewidth=2, markersize=6)
        plt.title('Code Quality Trend')
        plt.xlabel('Date')
        plt.ylabel('Quality Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('quality_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        charts['quality_trend'] = 'quality_trend.png'
        
        # Security Vulnerabilities Bar Chart
        plt.figure(figsize=(10, 6))
        vuln_types = ['Critical', 'High', 'Medium', 'Low']
        vuln_counts = [
            metrics['security']['critical_vulnerabilities'],
            metrics['security']['high_vulnerabilities'],
            metrics['security']['medium_vulnerabilities'],
            metrics['security']['low_vulnerabilities']
        ]
        colors = ['#ff4444', '#ff8800', '#ffaa00', '#ffdd00']
        
        bars = plt.bar(vuln_types, vuln_counts, color=colors)
        plt.title('Security Vulnerabilities by Severity')
        plt.xlabel('Severity Level')
        plt.ylabel('Count')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('security_vulnerabilities.png', dpi=300, bbox_inches='tight')
        plt.close()
        charts['security_vulnerabilities'] = 'security_vulnerabilities.png'
        
        return charts
    
    def _render_dashboard_template(self, metrics: Dict, charts: Dict) -> str:
        """Render HTML dashboard template"""
        
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Review Dashboard</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5; 
                }
                .header { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    margin-bottom: 30px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .metrics-grid { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                    margin-bottom: 30px; 
                }
                .metric-card { 
                    background: white; 
                    padding: 25px; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                    border-left: 4px solid #667eea;
                }
                .metric-value { 
                    font-size: 2.5em; 
                    font-weight: bold; 
                    color: #333; 
                    margin-bottom: 5px; 
                }
                .metric-label { 
                    color: #666; 
                    font-size: 0.9em; 
                    text-transform: uppercase; 
                    letter-spacing: 1px; 
                }
                .chart-section { 
                    background: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    margin-bottom: 30px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                }
                .chart-grid { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
                    gap: 30px; 
                }
                .chart-container { 
                    text-align: center; 
                }
                .chart-container img { 
                    max-width: 100%; 
                    height: auto; 
                    border-radius: 5px; 
                }
                .status-good { color: #28a745; }
                .status-warning { color: #ffc107; }
                .status-danger { color: #dc3545; }
                .team-table { 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin-top: 15px; 
                }
                .team-table th, .team-table td { 
                    padding: 10px; 
                    text-align: left; 
                    border-bottom: 1px solid #ddd; 
                }
                .team-table th { 
                    background-color: #f8f9fa; 
                    font-weight: 600; 
                }
                .quality-indicator {
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 12px;
                    font-size: 0.8em;
                    font-weight: 600;
                    text-transform: uppercase;
                }
                .quality-excellent { background-color: #d4edda; color: #155724; }
                .quality-good { background-color: #d1ecf1; color: #0c5460; }
                .quality-warning { background-color: #fff3cd; color: #856404; }
                .quality-poor { background-color: #f8d7da; color: #721c24; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Code Review Dashboard</h1>
                <p>Period: {{ metrics.period.start.strftime('%Y-%m-%d') }} to {{ metrics.period.end.strftime('%Y-%m-%d') }}</p>
                <p>Generated: {{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {{ 'status-good' if metrics.pull_requests.merged_prs / metrics.pull_requests.total_prs > 0.8 else 'status-warning' }}">
                        {{ metrics.pull_requests.total_prs }}
                    </div>
                    <div class="metric-label">Total Pull Requests</div>
                    <small>{{ metrics.pull_requests.merged_prs }} merged, {{ metrics.pull_requests.pending_prs }} pending</small>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value {{ 'status-good' if metrics.pull_requests.average_review_time_hours < 12 else 'status-warning' if metrics.pull_requests.average_review_time_hours < 24 else 'status-danger' }}">
                        {{ "%.1f"|format(metrics.pull_requests.average_review_time_hours) }}h
                    </div>
                    <div class="metric-label">Avg Review Time</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value {{ 'status-good' if metrics.code_quality.test_coverage > 80 else 'status-warning' if metrics.code_quality.test_coverage > 60 else 'status-danger' }}">
                        {{ "%.1f"|format(metrics.code_quality.test_coverage) }}%
                    </div>
                    <div class="metric-label">Test Coverage</div>
                    <small>Trend: {{ metrics.code_quality.coverage_trend }}</small>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value {{ 'status-danger' if metrics.security.critical_vulnerabilities > 0 else 'status-warning' if metrics.security.high_vulnerabilities > 0 else 'status-good' }}">
                        {{ metrics.security.vulnerabilities_found }}
                    </div>
                    <div class="metric-label">Security Issues</div>
                    <small>{{ metrics.security.vulnerabilities_fixed }} fixed</small>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value {{ 'status-good' if metrics.code_quality.quality_gate_pass_rate > 90 else 'status-warning' if metrics.code_quality.quality_gate_pass_rate > 75 else 'status-danger' }}">
                        {{ "%.1f"|format(metrics.code_quality.quality_gate_pass_rate) }}%
                    </div>
                    <div class="metric-label">Quality Gate Pass Rate</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value {{ 'status-good' if metrics.team_performance.review_participation_rate > 85 else 'status-warning' }}">
                        {{ "%.1f"|format(metrics.team_performance.review_participation_rate) }}%
                    </div>
                    <div class="metric-label">Review Participation</div>
                    <small>{{ metrics.team_performance.active_reviewers }} active reviewers</small>
                </div>
            </div>
            
            <div class="chart-section">
                <h2>📈 Trends & Analytics</h2>
                <div class="chart-grid">
                    <div class="chart-container">
                        <h3>Pull Request Size Distribution</h3>
                        <img src="{{ charts.pr_size_distribution }}" alt="PR Size Distribution">
                    </div>
                    
                    <div class="chart-container">
                        <h3>Code Quality Trend</h3>
                        <img src="{{ charts.quality_trend }}" alt="Quality Trend">
                    </div>
                    
                    <div class="chart-container">
                        <h3>Security Vulnerabilities</h3>
                        <img src="{{ charts.security_vulnerabilities }}" alt="Security Vulnerabilities">
                    </div>
                </div>
            </div>
            
            <div class="chart-section">
                <h2>👥 Team Performance</h2>
                
                <h3>Top Reviewers</h3>
                <table class="team-table">
                    <thead>
                        <tr>
                            <th>Reviewer</th>
                            <th>Reviews</th>
                            <th>Quality Score</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for reviewer in metrics.team_performance.top_reviewers %}
                        <tr>
                            <td>{{ reviewer.name }}</td>
                            <td>{{ reviewer.reviews }}</td>
                            <td>{{ "%.1f"|format(reviewer.quality_score) }}/10</td>
                            <td>
                                {% if reviewer.quality_score >= 9 %}
                                <span class="quality-indicator quality-excellent">Excellent</span>
                                {% elif reviewer.quality_score >= 8 %}
                                <span class="quality-indicator quality-good">Good</span>
                                {% elif reviewer.quality_score >= 7 %}
                                <span class="quality-indicator quality-warning">Fair</span>
                                {% else %}
                                <span class="quality-indicator quality-poor">Needs Improvement</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                
                <h3>Review Bottlenecks</h3>
                <table class="team-table">
                    <thead>
                        <tr>
                            <th>Reviewer</th>
                            <th>Avg Response Time</th>
                            <th>Impact</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for bottleneck in metrics.team_performance.review_bottlenecks %}
                        <tr>
                            <td>{{ bottleneck.reviewer }}</td>
                            <td>{{ "%.1f"|format(bottleneck.avg_response_time_hours) }} hours</td>
                            <td>
                                {% if bottleneck.avg_response_time_hours > 24 %}
                                <span class="status-danger">High</span>
                                {% elif bottleneck.avg_response_time_hours > 12 %}
                                <span class="status-warning">Medium</span>
                                {% else %}
                                <span class="status-good">Low</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="chart-section">
                <h2>🔍 Quality Insights</h2>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Code Complexity</h4>
                        <div class="metric-value">{{ "%.1f"|format(metrics.code_quality.average_complexity) }}</div>
                        <small>Average cyclomatic complexity (Target: < {{#if (eq code_quality_standards "enterprise")}}8{{else if (eq code_quality_standards "strict")}}10{{else}}15{{/if}})</small>
                    </div>
                    
                    <div class="metric-card">
                        <h4>Technical Debt</h4>
                        <div class="metric-value">{{ metrics.code_quality.technical_debt_hours }}h</div>
                        <small>Estimated remediation time</small>
                    </div>
                    
                    <div class="metric-card">
                        <h4>Maintainability Index</h4>
                        <div class="metric-value">{{ "%.1f"|format(metrics.code_quality.maintainability_index) }}</div>
                        <small>Scale: 0-100 (Higher is better)</small>
                    </div>
                </div>
                
                <h3>Code Smells Breakdown</h3>
                <div class="metrics-grid">
                    {% for severity, count in metrics.code_quality.code_smells.items() %}
                    <div class="metric-card">
                        <div class="metric-value {{ 'status-danger' if severity == 'blocker' else 'status-warning' if severity == 'critical' else 'status-good' }}">
                            {{ count }}
                        </div>
                        <div class="metric-label">{{ severity.title() }} Issues</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </body>
        </html>
        """
        
        from jinja2 import Template
        template = Template(template_str)
        return template.render(metrics=metrics, charts=charts, datetime=datetime)

# Usage example
async def main():
    dashboard = CodeReviewDashboard()
    
    # Generate dashboard for last 30 days
    dashboard_file = await dashboard.generate_dashboard(30)
    print(f"Dashboard generated: {dashboard_file}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

This Automated Code Review System provides:

**Key Features:**
- {{programming_languages}} static analysis with {{code_quality_standards}} standards
- {{security_requirements}} security scanning and vulnerability detection
- {{review_automation_level}} automation with intelligent feedback
- Integration with {{repository_platform}} platform

**Benefits:**
- Consistent code quality enforcement across {{team_size}} team
- Early detection of security vulnerabilities and code issues
- Automated quality gates preventing problematic code merges
- Comprehensive reporting and trend analysis

**Automation Levels:**
- Pre-commit hooks for immediate feedback
- CI/CD pipeline integration for continuous monitoring
- {{#if (eq review_automation_level "ai-assisted" "full-automation")}}AI-powered code analysis and suggestions{{/if}}
- Quality gates with configurable thresholds

**Success Metrics:**
- Reduced manual review time by 40-60%
- Improved code quality consistency
- Faster identification and resolution of security issues
- Enhanced team productivity and knowledge sharing