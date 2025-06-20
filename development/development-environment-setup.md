---
name: development_environment_setup
title: Development Environment Setup & Automation
description: Comprehensive development environment setup with automated provisioning, containerization, IDE configuration, and team standardization for consistent development workflows
category: development
tags: [dev-environment, docker, automation, ide-setup, tooling, standardization]
difficulty: intermediate
author: jezweb
version: 1.0.0
arguments:
  - name: tech_stack
    description: Primary technology stack (fullstack-js, python-web, java-spring, dotnet-core, go-microservices, mixed)
    required: true
  - name: team_size
    description: Development team size (solo, small 2-5, medium 6-15, large >15)
    required: true
  - name: deployment_target
    description: Target deployment environment (local, cloud-native, hybrid, microservices)
    required: true
  - name: ide_preference
    description: Primary IDE/Editor (vscode, intellij, vim-neovim, multiple)
    required: true
  - name: automation_level
    description: Automation level (basic, intermediate, advanced, full-automation)
    required: true
  - name: os_support
    description: Operating system support (windows, macos, linux, cross-platform)
    required: true
---

# Development Environment Setup: {{tech_stack}}

**Team Size:** {{team_size}}  
**Deployment Target:** {{deployment_target}}  
**IDE Preference:** {{ide_preference}}  
**Automation Level:** {{automation_level}}  
**OS Support:** {{os_support}}

## 1. Containerized Development Environment

### Docker Development Setup
```dockerfile
{{#if (contains tech_stack "fullstack-js")}}
# Dockerfile.dev for Full-Stack JavaScript
FROM node:18-alpine AS base

# Install system dependencies
RUN apk add --no-cache \
    git \
    curl \
    bash \
    postgresql-client \
    redis \
    python3 \
    make \
    g++

WORKDIR /app

# Install global tools
RUN npm install -g \
    @nestjs/cli \
    @angular/cli \
    create-react-app \
    nodemon \
    pm2 \
    typescript \
    ts-node \
    eslint \
    prettier

# Copy package files
COPY package*.json ./
COPY yarn.lock* ./

# Install dependencies
RUN npm ci --only=development

# Development stage
FROM base AS development

# Install development tools
RUN npm install -g \
    jest \
    cypress \
    webpack-dev-server \
    @storybook/cli

# Copy source code
COPY . .

# Expose ports for development
EXPOSE 3000 3001 4200 8080 9229

# Start development server
CMD ["npm", "run", "dev"]
{{/if}}

{{#if (contains tech_stack "python-web")}}
# Dockerfile.dev for Python Web Development
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    postgresql-client \
    redis-tools \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --with dev && rm -rf $POETRY_CACHE_DIR

# Development stage
FROM base AS development

# Install development tools
RUN poetry add --group dev \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-django \
    django-debug-toolbar \
    ipython \
    jupyter

# Copy source code
COPY . .

# Expose ports
EXPOSE 8000 8001 8888

# Start development server
CMD ["poetry", "run", "python", "manage.py", "runserver", "0.0.0.0:8000"]
{{/if}}

{{#if (contains tech_stack "java-spring")}}
# Dockerfile.dev for Java Spring Development
FROM openjdk:17-jdk-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    maven \
    gradle \
    && rm -rf /var/lib/apt/lists/*

# Install SDKMAN and tools
RUN curl -s "https://get.sdkman.io" | bash
RUN bash -c "source ~/.sdkman/bin/sdkman-init.sh && \
    sdk install springboot && \
    sdk install maven && \
    sdk install gradle"

WORKDIR /app

# Development stage
FROM base AS development

# Copy build files
COPY pom.xml* build.gradle* gradlew* ./
COPY gradle/ gradle/

# Download dependencies
RUN if [ -f "pom.xml" ]; then mvn dependency:go-offline; fi
RUN if [ -f "build.gradle" ]; then ./gradlew build --no-daemon || true; fi

# Copy source code
COPY . .

# Expose ports
EXPOSE 8080 8081 5005

# Start with remote debugging enabled
CMD ["java", "-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=*:5005", "-jar", "target/app.jar"]
{{/if}}
```

### Docker Compose Development Stack
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  {{#if (contains tech_stack "fullstack-js")}}
  # Frontend service
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.dev
      target: development
    volumes:
      - .:/app
      - /app/node_modules
      - frontend_cache:/app/.next  # Next.js cache
    ports:
      - "3000:3000"
      - "3001:3001"  # Hot reload
    environment:
      - NODE_ENV=development
      - CHOKIDAR_USEPOLLING=true
      - WATCHPACK_POLLING=true
    command: npm run dev
    
  # Backend API service
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    volumes:
      - ./backend:/app
      - backend_modules:/app/node_modules
    ports:
      - "4000:4000"
      - "9229:9229"  # Debug port
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/devdb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    command: npm run dev:debug
  {{/if}}

  {{#if (contains tech_stack "python-web")}}
  # Django/FastAPI service
  web:
    build:
      context: .
      dockerfile: Dockerfile.dev
      target: development
    volumes:
      - .:/app
      - python_cache:/root/.cache/pip
    ports:
      - "8000:8000"
      - "8888:8888"  # Jupyter
    environment:
      - DEBUG=True
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/devdb
      - REDIS_URL=redis://redis:6379
      - CELERY_BROKER=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    command: poetry run python manage.py runserver 0.0.0.0:8000
    
  # Celery worker
  celery-worker:
    build:
      context: .
      dockerfile: Dockerfile.dev
      target: development
    volumes:
      - .:/app
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/devdb
      - CELERY_BROKER=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    command: poetry run celery -A myapp worker --loglevel=info --reload
  {{/if}}

  # Database service
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=devdb
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    
  # Redis service
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  {{#if (eq deployment_target "microservices")}}
  # Message broker for microservices
  rabbitmq:
    image: rabbitmq:3-management-alpine
    environment:
      - RABBITMQ_DEFAULT_USER=developer
      - RABBITMQ_DEFAULT_PASS=password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"  # Management UI
  
  # Service discovery (Consul)
  consul:
    image: consul:1.15
    command: consul agent -dev -ui -client=0.0.0.0
    ports:
      - "8500:8500"
    volumes:
      - consul_data:/consul/data
  {{/if}}

  # Monitoring and observability
  {{#if (eq automation_level "advanced" "full-automation")}}
  # Jaeger for distributed tracing
  jaeger:
    image: jaegertracing/all-in-one:1.45
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "16686:16686"  # Web UI
      - "14268:14268"  # HTTP collector
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
    
  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:v2.45.0
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
  
  # Grafana for visualization
  grafana:
    image: grafana/grafana:9.5.0
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3001:3000"
  {{/if}}

  # Development tools
  {{#if (contains ide_preference "vscode")}}
  # VS Code Server for remote development
  code-server:
    image: codercom/code-server:4.15.0
    environment:
      - PASSWORD=developer
    volumes:
      - .:/home/coder/workspace
      - vscode_extensions:/home/coder/.local/share/code-server/extensions
    ports:
      - "8443:8080"
    command: code-server --bind-addr 0.0.0.0:8080 --auth password /home/coder/workspace
  {{/if}}

volumes:
  postgres_data:
  redis_data:
  {{#if (contains tech_stack "fullstack-js")}}
  frontend_cache:
  backend_modules:
  {{/if}}
  {{#if (contains tech_stack "python-web")}}
  python_cache:
  {{/if}}
  {{#if (eq deployment_target "microservices")}}
  rabbitmq_data:
  consul_data:
  {{/if}}
  {{#if (eq automation_level "advanced" "full-automation")}}
  prometheus_data:
  grafana_data:
  {{/if}}
  {{#if (contains ide_preference "vscode")}}
  vscode_extensions:
  {{/if}}

networks:
  default:
    name: dev-network
```

## 2. Automated Environment Provisioning

### Cross-Platform Setup Script
```bash
#!/bin/bash
# setup-dev-env.sh - Cross-platform development environment setup

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="{{tech_stack}}-dev"
REQUIRED_DOCKER_VERSION="20.0.0"
REQUIRED_NODE_VERSION="{{#if (contains tech_stack "fullstack-js")}}18{{else}}16{{/if}}"
REQUIRED_PYTHON_VERSION="{{#if (contains tech_stack "python-web")}}3.11{{else}}3.9{{/if}}"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

version_compare() {
    printf '%s\n%s\n' "$1" "$2" | sort -V -C
}

detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*)  echo "linux" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        *) echo "unknown" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo "amd64" ;;
        aarch64|arm64) echo "arm64" ;;
        *) echo "unknown" ;;
    esac
}

# OS-specific package managers
install_package() {
    local package=$1
    local os=$(detect_os)
    
    case $os in
        "macos")
            if check_command brew; then
                brew install "$package"
            else
                log_error "Homebrew not found. Please install Homebrew first."
                exit 1
            fi
            ;;
        "linux")
            if check_command apt-get; then
                sudo apt-get update && sudo apt-get install -y "$package"
            elif check_command yum; then
                sudo yum install -y "$package"
            elif check_command pacman; then
                sudo pacman -S --noconfirm "$package"
            else
                log_error "No supported package manager found"
                exit 1
            fi
            ;;
        "windows")
            if check_command choco; then
                choco install "$package" -y
            elif check_command winget; then
                winget install "$package"
            else
                log_error "Please install Chocolatey or use winget"
                exit 1
            fi
            ;;
    esac
}

# System requirements check
check_system_requirements() {
    log_info "Checking system requirements..."
    
    local os=$(detect_os)
    local arch=$(detect_arch)
    
    log_info "Operating System: $os ($arch)"
    
    # Check for required tools
    local missing_tools=()
    
    if ! check_command git; then
        missing_tools+=("git")
    fi
    
    if ! check_command curl; then
        missing_tools+=("curl")
    fi
    
    {{#if (eq os_support "cross-platform")}}
    # Cross-platform requirements
    case $os in
        "windows")
            if ! check_command wsl; then
                log_warn "WSL not detected. Some features may not work properly."
            fi
            ;;
        "macos")
            if ! check_command xcode-select; then
                missing_tools+=("xcode-command-line-tools")
            fi
            ;;
    esac
    {{/if}}
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_warn "Missing required tools: ${missing_tools[*]}"
        log_info "Installing missing tools..."
        
        for tool in "${missing_tools[@]}"; do
            install_package "$tool"
        done
    fi
}

# Docker installation and setup
setup_docker() {
    log_info "Setting up Docker..."
    
    if check_command docker; then
        local docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
        if version_compare "$REQUIRED_DOCKER_VERSION" "$docker_version"; then
            log_info "Docker version $docker_version meets requirements"
        else
            log_warn "Docker version $docker_version is below required $REQUIRED_DOCKER_VERSION"
        fi
    else
        log_info "Installing Docker..."
        
        local os=$(detect_os)
        case $os in
            "macos")
                log_info "Please install Docker Desktop for Mac from https://docker.com/products/docker-desktop"
                ;;
            "linux")
                curl -fsSL https://get.docker.com -o get-docker.sh
                sudo sh get-docker.sh
                sudo usermod -aG docker $USER
                log_info "Please log out and back in for Docker group changes to take effect"
                ;;
            "windows")
                log_info "Please install Docker Desktop for Windows from https://docker.com/products/docker-desktop"
                ;;
        esac
    fi
    
    # Install Docker Compose if not present
    if ! check_command docker-compose; then
        log_info "Installing Docker Compose..."
        
        local compose_version="2.20.0"
        local os=$(detect_os)
        local arch=$(detect_arch)
        
        case $os in
            "linux"|"macos")
                sudo curl -L "https://github.com/docker/compose/releases/download/v${compose_version}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
                sudo chmod +x /usr/local/bin/docker-compose
                ;;
        esac
    fi
}

# Node.js setup (if required)
{{#if (contains tech_stack "fullstack-js")}}
setup_nodejs() {
    log_info "Setting up Node.js..."
    
    if check_command node; then
        local node_version=$(node --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        local major_version=$(echo $node_version | cut -d. -f1)
        
        if [ "$major_version" -ge "$REQUIRED_NODE_VERSION" ]; then
            log_info "Node.js version $node_version meets requirements"
        else
            log_warn "Node.js version $node_version is below required $REQUIRED_NODE_VERSION"
        fi
    else
        log_info "Installing Node.js..."
        
        # Install Node Version Manager (NVM)
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
        
        # Reload bash profile
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        
        # Install and use required Node.js version
        nvm install $REQUIRED_NODE_VERSION
        nvm use $REQUIRED_NODE_VERSION
        nvm alias default $REQUIRED_NODE_VERSION
    fi
    
    # Install global packages
    local global_packages=(
        "npm@latest"
        "yarn"
        {{#if (eq automation_level "advanced" "full-automation")}}
        "@nestjs/cli"
        "create-react-app"
        "typescript"
        "ts-node"
        "nodemon"
        "pm2"
        {{/if}}
    )
    
    log_info "Installing global npm packages..."
    for package in "${global_packages[@]}"; do
        npm install -g "$package"
    done
}
{{/if}}

# Python setup (if required)
{{#if (contains tech_stack "python-web")}}
setup_python() {
    log_info "Setting up Python..."
    
    if check_command python3; then
        local python_version=$(python3 --version | grep -oE '[0-9]+\.[0-9]+')
        if version_compare "$REQUIRED_PYTHON_VERSION" "$python_version"; then
            log_info "Python version $python_version meets requirements"
        else
            log_warn "Python version $python_version may not meet requirements"
        fi
    else
        log_info "Installing Python..."
        install_package "python3"
        install_package "python3-pip"
    fi
    
    # Install Poetry for dependency management
    if ! check_command poetry; then
        log_info "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        
        # Add Poetry to PATH
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Install pipx for isolated tool installation
    if ! check_command pipx; then
        python3 -m pip install --user pipx
        python3 -m pipx ensurepath
    fi
    
    # Install development tools via pipx
    local python_tools=(
        "black"
        "isort"
        "flake8"
        "mypy"
        {{#if (eq automation_level "advanced" "full-automation")}}
        "pre-commit"
        "cookiecutter"
        "django"
        "fastapi[all]"
        {{/if}}
    )
    
    log_info "Installing Python development tools..."
    for tool in "${python_tools[@]}"; do
        pipx install "$tool" || true
    done
}
{{/if}}

# IDE and editor setup
setup_ide() {
    log_info "Setting up IDE and development tools..."
    
    {{#if (contains ide_preference "vscode")}}
    # VS Code setup
    if ! check_command code; then
        log_info "Installing VS Code..."
        
        local os=$(detect_os)
        case $os in
            "macos")
                install_package "visual-studio-code"
                ;;
            "linux")
                wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
                sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
                sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
                sudo apt update
                sudo apt install code
                ;;
            "windows")
                install_package "vscode"
                ;;
        esac
    fi
    
    # Install VS Code extensions
    local vscode_extensions=(
        "ms-vscode.vscode-typescript-next"
        "esbenp.prettier-vscode"
        "bradlc.vscode-tailwindcss"
        "ms-python.python"
        "ms-python.black-formatter"
        "ms-vscode.vscode-docker"
        "GitLab.gitlab-workflow"
        {{#if (eq automation_level "advanced" "full-automation")}}
        "ms-vscode-remote.remote-containers"
        "ms-vscode-remote.remote-wsl"
        "ms-kubernetes-tools.vscode-kubernetes-tools"
        "redhat.vscode-yaml"
        "hashicorp.terraform"
        {{/if}}
    )
    
    log_info "Installing VS Code extensions..."
    for extension in "${vscode_extensions[@]}"; do
        code --install-extension "$extension"
    done
    {{/if}}
    
    {{#if (contains ide_preference "vim-neovim")}}
    # Neovim setup
    if ! check_command nvim; then
        log_info "Installing Neovim..."
        install_package "neovim"
    fi
    
    # Install vim-plug for plugin management
    if [ ! -f ~/.local/share/nvim/site/autoload/plug.vim ]; then
        log_info "Installing vim-plug..."
        curl -fLo ~/.local/share/nvim/site/autoload/plug.vim --create-dirs \
            https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
    fi
    
    # Create basic Neovim configuration
    mkdir -p ~/.config/nvim
    cat > ~/.config/nvim/init.vim << 'EOF'
call plug#begin('~/.local/share/nvim/plugged')

" Language support
Plug 'neoclide/coc.nvim', {'branch': 'release'}
Plug 'sheerun/vim-polyglot'
Plug 'prettier/vim-prettier'

" File navigation
Plug 'preservim/nerdtree'
Plug 'ctrlpvim/ctrlp.vim'
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

" Git integration
Plug 'tpope/vim-fugitive'
Plug 'airblade/vim-gitgutter'

" Theme and UI
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
Plug 'dracula/vim', { 'as': 'dracula' }

call plug#end()

" Basic configuration
set number
set relativenumber
set tabstop=2
set shiftwidth=2
set expandtab
set autoindent
colorscheme dracula

" Key mappings
let mapleader = " "
nnoremap <leader>e :NERDTreeToggle<CR>
nnoremap <leader>f :Files<CR>
nnoremap <leader>g :GFiles<CR>
EOF
    {{/if}}
}

# Project initialization
initialize_project() {
    log_info "Initializing project structure..."
    
    # Create project directories
    local project_dirs=(
        "src"
        "tests"
        "docs"
        "scripts"
        "config"
        {{#if (eq deployment_target "microservices")}}
        "services"
        "shared"
        {{/if}}
        {{#if (eq automation_level "advanced" "full-automation")}}
        "monitoring"
        "infrastructure"
        ".github/workflows"
        {{/if}}
    )
    
    for dir in "${project_dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    # Create development configuration files
    create_dev_configs
    
    # Initialize git repository if not exists
    if [ ! -d ".git" ]; then
        log_info "Initializing git repository..."
        git init
        git branch -M main
    fi
    
    # Create .gitignore
    create_gitignore
    
    # Setup pre-commit hooks
    {{#if (eq automation_level "intermediate" "advanced" "full-automation")}}
    setup_precommit_hooks
    {{/if}}
}

create_dev_configs() {
    log_info "Creating development configuration files..."
    
    # Environment variables template
    cat > .env.example << 'EOF'
# Development Environment Variables
NODE_ENV=development
DEBUG=true

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/devdb

# Redis
REDIS_URL=redis://localhost:6379

{{#if (contains tech_stack "fullstack-js")}}
# Frontend
NEXT_PUBLIC_API_URL=http://localhost:4000
REACT_APP_API_URL=http://localhost:4000
{{/if}}

{{#if (contains tech_stack "python-web")}}
# Django/FastAPI
SECRET_KEY=development-secret-key-change-in-production
ALLOWED_HOSTS=localhost,127.0.0.1
{{/if}}

# External Services
{{#if (eq deployment_target "cloud-native")}}
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
{{/if}}
EOF
    
    {{#if (contains tech_stack "fullstack-js")}}
    # Package.json scripts for development
    if [ ! -f "package.json" ]; then
        cat > package.json << 'EOF'
{
  "name": "{{tech_stack}}-project",
  "version": "1.0.0",
  "scripts": {
    "dev": "concurrently \"npm run dev:backend\" \"npm run dev:frontend\"",
    "dev:backend": "cd backend && npm run dev",
    "dev:frontend": "cd frontend && npm run dev",
    "build": "npm run build:backend && npm run build:frontend",
    "test": "npm run test:backend && npm run test:frontend",
    "lint": "eslint . --ext .js,.jsx,.ts,.tsx",
    "lint:fix": "eslint . --ext .js,.jsx,.ts,.tsx --fix",
    "format": "prettier --write .",
    "docker:dev": "docker-compose -f docker-compose.dev.yml up",
    "docker:build": "docker-compose -f docker-compose.dev.yml build"
  },
  "devDependencies": {
    "concurrently": "^8.0.0",
    "eslint": "^8.0.0",
    "prettier": "^3.0.0"
  }
}
EOF
    fi
    {{/if}}
    
    {{#if (contains tech_stack "python-web")}}
    # Python project configuration
    if [ ! -f "pyproject.toml" ]; then
        cat > pyproject.toml << 'EOF'
[tool.poetry]
name = "{{tech_stack}}-project"
version = "0.1.0"
description = "Development project"
authors = ["Developer <dev@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"

[tool.poetry.group.dev.dependencies]
black = "^23.0.0"
isort = "^5.12.0"
flake8 = "^6.0.0"
mypy = "^1.3.0"
pytest = "^7.3.0"
pytest-cov = "^4.1.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
EOF
    fi
    {{/if}}
    
    # Makefile for common development tasks
    cat > Makefile << 'EOF'
.PHONY: help setup build test clean docker-up docker-down

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Setup development environment
	@echo "Setting up development environment..."
	@./scripts/setup-dev-env.sh

build: ## Build the application
{{#if (contains tech_stack "fullstack-js")}}
	@npm run build
{{else if (contains tech_stack "python-web")}}
	@poetry build
{{else}}
	@docker-compose build
{{/if}}

test: ## Run tests
{{#if (contains tech_stack "fullstack-js")}}
	@npm test
{{else if (contains tech_stack "python-web")}}
	@poetry run pytest
{{else}}
	@docker-compose run --rm app npm test
{{/if}}

lint: ## Run linting
{{#if (contains tech_stack "fullstack-js")}}
	@npm run lint
{{else if (contains tech_stack "python-web")}}
	@poetry run flake8 src tests
	@poetry run black --check src tests
	@poetry run isort --check-only src tests
{{/if}}

format: ## Format code
{{#if (contains tech_stack "fullstack-js")}}
	@npm run format
{{else if (contains tech_stack "python-web")}}
	@poetry run black src tests
	@poetry run isort src tests
{{/if}}

docker-up: ## Start development containers
	@docker-compose -f docker-compose.dev.yml up -d

docker-down: ## Stop development containers
	@docker-compose -f docker-compose.dev.yml down

docker-logs: ## View container logs
	@docker-compose -f docker-compose.dev.yml logs -f

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
{{#if (contains tech_stack "fullstack-js")}}
	@rm -rf node_modules dist build .next
{{else if (contains tech_stack "python-web")}}
	@rm -rf dist build .pytest_cache __pycache__
	@find . -name "*.pyc" -delete
{{/if}}
	@docker system prune -f
EOF
    
    chmod +x Makefile
}

create_gitignore() {
    log_info "Creating .gitignore file..."
    
    cat > .gitignore << 'EOF'
# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

{{#if (contains tech_stack "fullstack-js")}}
# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Build outputs
dist/
build/
.next/
out/
{{/if}}

{{#if (contains tech_stack "python-web")}}
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache/
htmlcov/
.coverage
.coverage.*
coverage.xml
{{/if}}

# IDEs and editors
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
logs/
*.log

# Docker
.dockerignore

# Temporary files
tmp/
temp/
EOF
}

{{#if (eq automation_level "intermediate" "advanced" "full-automation")}}
setup_precommit_hooks() {
    log_info "Setting up pre-commit hooks..."
    
    # Install pre-commit if not present
    if ! check_command pre-commit; then
        {{#if (contains tech_stack "python-web")}}
        pipx install pre-commit
        {{else}}
        pip3 install --user pre-commit
        {{/if}}
    fi
    
    # Create pre-commit configuration
    cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-merge-conflict
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
      - id: detect-private-key

{{#if (contains tech_stack "fullstack-js")}}
  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.44.0
    hooks:
      - id: eslint
        files: \.(js|jsx|ts|tsx)$
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.0
    hooks:
      - id: prettier
        files: \.(js|jsx|ts|tsx|json|md|yml|yaml)$
{{/if}}

{{#if (contains tech_stack "python-web")}}
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
{{/if}}
EOF
    
    # Install pre-commit hooks
    pre-commit install
}
{{/if}}

# Health check and validation
validate_setup() {
    log_info "Validating development environment setup..."
    
    local checks_passed=0
    local total_checks=0
    
    # Docker check
    ((total_checks++))
    if check_command docker && docker --version > /dev/null 2>&1; then
        log_info "✅ Docker is working"
        ((checks_passed++))
    else
        log_error "❌ Docker is not working properly"
    fi
    
    # Docker Compose check
    ((total_checks++))
    if check_command docker-compose; then
        log_info "✅ Docker Compose is available"
        ((checks_passed++))
    else
        log_error "❌ Docker Compose is not available"
    fi
    
    {{#if (contains tech_stack "fullstack-js")}}
    # Node.js check
    ((total_checks++))
    if check_command node && check_command npm; then
        log_info "✅ Node.js and npm are working"
        ((checks_passed++))
    else
        log_error "❌ Node.js or npm not working properly"
    fi
    {{/if}}
    
    {{#if (contains tech_stack "python-web")}}
    # Python check
    ((total_checks++))
    if check_command python3 && check_command poetry; then
        log_info "✅ Python and Poetry are working"
        ((checks_passed++))
    else
        log_error "❌ Python or Poetry not working properly"
    fi
    {{/if}}
    
    # IDE check
    {{#if (contains ide_preference "vscode")}}
    ((total_checks++))
    if check_command code; then
        log_info "✅ VS Code is available"
        ((checks_passed++))
    else
        log_warn "⚠️  VS Code not found"
    fi
    {{/if}}
    
    log_info "Setup validation: $checks_passed/$total_checks checks passed"
    
    if [ $checks_passed -eq $total_checks ]; then
        log_info "🎉 Development environment setup completed successfully!"
        return 0
    else
        log_error "❌ Some checks failed. Please review the setup."
        return 1
    fi
}

# Usage information
show_usage() {
    cat << EOF
Development Environment Setup Script

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -s, --skip-docker   Skip Docker installation
    -q, --quiet         Suppress verbose output
    --validate-only     Only run validation checks

Environment:
    Tech Stack: {{tech_stack}}
    Team Size: {{team_size}}
    OS Support: {{os_support}}
    IDE: {{ide_preference}}
    Automation: {{automation_level}}

EOF
}

# Main execution
main() {
    local skip_docker=false
    local quiet=false
    local validate_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -s|--skip-docker)
                skip_docker=true
                shift
                ;;
            -q|--quiet)
                quiet=true
                shift
                ;;
            --validate-only)
                validate_only=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    if [ "$validate_only" = true ]; then
        validate_setup
        exit $?
    fi
    
    log_info "Starting development environment setup for {{tech_stack}}..."
    
    # Execute setup steps
    check_system_requirements
    
    if [ "$skip_docker" = false ]; then
        setup_docker
    fi
    
    {{#if (contains tech_stack "fullstack-js")}}
    setup_nodejs
    {{/if}}
    
    {{#if (contains tech_stack "python-web")}}
    setup_python
    {{/if}}
    
    setup_ide
    initialize_project
    
    # Final validation
    if validate_setup; then
        log_info ""
        log_info "🚀 Next steps:"
        log_info "1. Copy .env.example to .env and configure your environment variables"
        log_info "2. Start development containers: make docker-up"
        {{#if (contains tech_stack "fullstack-js")}}
        log_info "3. Install project dependencies: npm install"
        log_info "4. Start development server: npm run dev"
        {{/if}}
        {{#if (contains tech_stack "python-web")}}
        log_info "3. Install project dependencies: poetry install"
        log_info "4. Start development server: poetry run python manage.py runserver"
        {{/if}}
        log_info ""
        log_info "For more commands, run: make help"
    else
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
```

## 3. IDE Configuration & Extensions

### VS Code Workspace Configuration
```json
{{#if (contains ide_preference "vscode")}}
// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true,
    "source.organizeImports": true
  },
  "editor.rulers": [80, 120],
  "editor.tabSize": 2,
  "editor.insertSpaces": true,
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "files.trimFinalNewlines": true,
  
  {{#if (contains tech_stack "fullstack-js")}}
  // JavaScript/TypeScript settings
  "typescript.preferences.importModuleSpecifier": "relative",
  "typescript.suggest.autoImports": true,
  "javascript.suggest.autoImports": true,
  "eslint.workingDirectories": ["frontend", "backend"],
  "eslint.validate": [
    "javascript",
    "javascriptreact",
    "typescript",
    "typescriptreact"
  ],
  {{/if}}
  
  {{#if (contains tech_stack "python-web")}}
  // Python settings
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  {{/if}}
  
  // Docker settings
  "docker.showStartPage": false,
  
  // Git settings
  "git.autofetch": true,
  "git.enableSmartCommit": true,
  
  {{#if (eq deployment_target "microservices")}}
  // Microservices specific settings
  "files.associations": {
    "docker-compose*.yml": "dockercompose",
    "*.dockerfile": "dockerfile",
    "Dockerfile*": "dockerfile"
  },
  {{/if}}
  
  {{#if (eq automation_level "advanced" "full-automation")}}
  // Advanced development settings
  "remote.containers.workspaceFolder": "/workspace",
  "remote.containers.defaultExtensions": [
    "ms-vscode.vscode-typescript-next",
    "esbenp.prettier-vscode",
    "ms-python.python",
    "ms-vscode.vscode-docker"
  ],
  {{/if}}
  
  // File exclusions for better performance
  "files.exclude": {
    "**/node_modules": true,
    "**/.git": true,
    "**/.DS_Store": true,
    "**/dist": true,
    "**/build": true,
    "**/__pycache__": true,
    "**/*.pyc": true
  },
  
  "search.exclude": {
    "**/node_modules": true,
    "**/bower_components": true,
    "**/*.code-search": true,
    "**/dist": true,
    "**/build": true
  }
}

// .vscode/launch.json - Debug configurations
{
  "version": "0.2.0",
  "configurations": [
    {{#if (contains tech_stack "fullstack-js")}}
    {
      "name": "Debug Node.js Backend",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/backend/src/index.js",
      "env": {
        "NODE_ENV": "development"
      },
      "envFile": "${workspaceFolder}/.env",
      "console": "integratedTerminal",
      "restart": true,
      "runtimeExecutable": "nodemon",
      "runtimeArgs": ["--inspect"]
    },
    {
      "name": "Debug Frontend Tests",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/frontend/node_modules/.bin/jest",
      "args": ["--runInBand"],
      "cwd": "${workspaceFolder}/frontend",
      "console": "integratedTerminal",
      "internalConsoleOptions": "neverOpen"
    },
    {{/if}}
    {{#if (contains tech_stack "python-web")}}
    {
      "name": "Debug Django",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/manage.py",
      "args": ["runserver", "0.0.0.0:8000"],
      "env": {
        "DEBUG": "True"
      },
      "envFile": "${workspaceFolder}/.env",
      "console": "integratedTerminal",
      "django": true
    },
    {
      "name": "Debug FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
      "envFile": "${workspaceFolder}/.env",
      "console": "integratedTerminal"
    },
    {
      "name": "Debug Python Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v"],
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    },
    {{/if}}
    {
      "name": "Docker: Debug Container",
      "type": "docker",
      "request": "launch",
      "preLaunchTask": "docker-run: debug",
      "python": {
        "pathMappings": [
          {
            "localRoot": "${workspaceFolder}",
            "remoteRoot": "/app"
          }
        ],
        "projectType": "{{#if (contains tech_stack "python-web")}}django{{else}}general{{/if}}"
      }
    }
  ]
}

// .vscode/tasks.json - Build and development tasks
{
  "version": "2.0.0",
  "tasks": [
    {{#if (contains tech_stack "fullstack-js")}}
    {
      "label": "npm: install",
      "type": "shell",
      "command": "npm install",
      "group": "build",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "npm: dev",
      "type": "shell",
      "command": "npm run dev",
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "presentation": {
        "reveal": "always",
        "panel": "new"
      },
      "isBackground": true
    },
    {
      "label": "npm: test",
      "type": "shell",
      "command": "npm test",
      "group": "test",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    },
    {{/if}}
    {{#if (contains tech_stack "python-web")}}
    {
      "label": "poetry: install",
      "type": "shell",
      "command": "poetry install",
      "group": "build",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "django: runserver",
      "type": "shell",
      "command": "poetry run python manage.py runserver",
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "presentation": {
        "reveal": "always",
        "panel": "new"
      },
      "isBackground": true
    },
    {
      "label": "pytest: run tests",
      "type": "shell",
      "command": "poetry run pytest",
      "group": "test",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    },
    {{/if}}
    {
      "label": "docker-compose: up dev",
      "type": "shell",
      "command": "docker-compose -f docker-compose.dev.yml up",
      "group": "build",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      },
      "isBackground": true
    },
    {
      "label": "docker-compose: down",
      "type": "shell",
      "command": "docker-compose -f docker-compose.dev.yml down",
      "group": "build",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "format code",
      "type": "shell",
      "command": "{{#if (contains tech_stack "fullstack-js")}}npm run format{{else if (contains tech_stack "python-web")}}poetry run black . && poetry run isort .{{else}}echo 'No formatter configured'{{/if}}",
      "group": "build",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    }
  ]
}

// .vscode/extensions.json - Recommended extensions
{
  "recommendations": [
    "ms-vscode.vscode-typescript-next",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-docker",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    {{#if (contains tech_stack "fullstack-js")}}
    "bradlc.vscode-tailwindcss",
    "ms-vscode.vscode-eslint",
    "formulahendry.auto-rename-tag",
    "christian-kohler.path-intellisense",
    {{/if}}
    {{#if (contains tech_stack "python-web")}}
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.flake8",
    "ms-python.mypy-type-checker",
    {{/if}}
    {{#if (eq automation_level "advanced" "full-automation")}}
    "ms-vscode-remote.remote-containers",
    "ms-kubernetes-tools.vscode-kubernetes-tools",
    "hashicorp.terraform",
    {{/if}}
    "eamodio.gitlens",
    "github.copilot",
    "github.copilot-chat",
    "ms-vsliveshare.vsliveshare"
  ]
}
{{/if}}
```

## 4. Team Collaboration & Standards

### Development Standards Documentation
```markdown
# Development Environment Standards

## Overview
This document outlines the development environment standards for our {{team_size}} {{tech_stack}} team.

## Development Environment Requirements

### Core Tools
- **Docker**: Version 20.0.0+
- **Docker Compose**: Version 2.0.0+
- **Git**: Version 2.30.0+
{{#if (contains tech_stack "fullstack-js")}}
- **Node.js**: Version {{#if (contains tech_stack "fullstack-js")}}18{{else}}16{{/if}}+
- **npm**: Version 8.0.0+
{{/if}}
{{#if (contains tech_stack "python-web")}}
- **Python**: Version 3.11+
- **Poetry**: Latest version
{{/if}}

### IDE Configuration
{{#if (contains ide_preference "vscode")}}
#### VS Code Extensions (Required)
- ESLint
- Prettier
- Docker
- GitLens
{{#if (contains tech_stack "python-web")}}
- Python
- Black Formatter
{{/if}}

#### VS Code Extensions (Recommended)
- GitHub Copilot
- Live Share
- Remote Containers
{{/if}}

### Code Quality Standards
1. **Formatting**: Use automated formatters (Prettier for JS/TS, Black for Python)
2. **Linting**: All code must pass linting checks
3. **Type Safety**: Use TypeScript for JavaScript projects, type hints for Python
4. **Testing**: Minimum {{#if (eq automation_level "advanced" "full-automation")}}80%{{else if (eq automation_level "intermediate")}}70%{{else}}60%{{/if}} test coverage
5. **Documentation**: All public APIs must be documented

### Development Workflow

#### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd <project-name>

# Run setup script
./scripts/setup-dev-env.sh

# Start development environment
make docker-up
```

#### 2. Daily Development
```bash
# Start development
make docker-up

# Run tests
make test

# Format code
make format

# Stop development
make docker-down
```

#### 3. Code Contribution
1. Create feature branch from `main`
2. Make changes following coding standards
3. Write/update tests
4. Run quality checks locally
5. Create pull request
6. Address review feedback
7. Merge after approval

### Environment Variables
- Use `.env.example` as template
- Never commit actual `.env` files
- Document all environment variables
- Use descriptive variable names

### Database Development
- Use containerized databases for consistency
- Include database migrations in code
- Use database seeds for test data
- Document schema changes

### Debugging Setup
{{#if (contains tech_stack "fullstack-js")}}
#### Node.js Debugging
- Use VS Code debugger with provided configurations
- Set breakpoints in source code
- Use `debugger;` statements for quick debugging
- Enable source maps for TypeScript
{{/if}}

{{#if (contains tech_stack "python-web")}}
#### Python Debugging
- Use VS Code Python debugger
- Set breakpoints in source code
- Use `pdb.set_trace()` for quick debugging
- Enable Django debug toolbar in development
{{/if}}

### Performance Monitoring
{{#if (eq automation_level "advanced" "full-automation")}}
- Prometheus metrics collection
- Grafana dashboards
- Jaeger distributed tracing
- Application performance monitoring
{{/if}}

### Security Guidelines
- Scan dependencies for vulnerabilities
- Use environment variables for secrets
- Enable HTTPS in development
- Follow OWASP security guidelines

### Troubleshooting

#### Common Issues
1. **Docker containers won't start**
   - Check Docker daemon is running
   - Verify port conflicts
   - Check disk space

2. **Database connection issues**
   - Verify database container is running
   - Check connection string
   - Ensure database is initialized

3. **Module/package not found**
   - Rebuild Docker containers
   - Check dependency installation
   - Verify volume mounts

#### Getting Help
1. Check this documentation
2. Search project issues on GitHub
3. Ask in team chat
4. Create support ticket

### Team Resources
- **Project Repository**: [Link to repository]
- **Documentation**: [Link to docs]
- **Issue Tracker**: [Link to issues]
- **Team Chat**: [Link to chat]
{{#if (eq automation_level "advanced" "full-automation")}}
- **Monitoring Dashboard**: [Link to monitoring]
- **CI/CD Pipeline**: [Link to pipeline]
{{/if}}
```

## Conclusion

This Development Environment Setup provides:

**Key Features:**
- {{tech_stack}} containerized development environment
- {{os_support}} automated setup scripts
- {{ide_preference}} IDE configuration and extensions
- {{automation_level}} automation with quality tools

**Benefits:**
- Consistent development environment across {{team_size}} team
- Reduced onboarding time for new developers
- Automated quality checks and formatting
- Comprehensive debugging and monitoring setup

**Automation Levels:**
- Containerized services with Docker Compose
- Pre-commit hooks for code quality
- {{#if (eq automation_level "advanced" "full-automation")}}Advanced monitoring and observability tools{{/if}}
- Cross-platform setup scripts

**Success Metrics:**
- Developer onboarding time reduced by 70%
- Consistent code quality across team
- Reduced environment-related issues
- Improved development productivity