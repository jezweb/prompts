# Jezweb Prompts Library

A comprehensive collection of AI prompts optimized for the Smart Prompts MCP Server. These prompts are designed to help with software development, project management, content creation, and more.

## 📁 Categories

### 🚀 Development
Software development prompts for coding, testing, and debugging.
- Backend development
- Frontend development
- Testing & QA
- Debugging & troubleshooting

### 📚 Documentation
Create comprehensive documentation for your projects.
- README files
- API documentation
- Technical writing
- User guides

### 📋 Project Management
Plan and manage software projects effectively.
- Project kickoff
- Sprint planning
- Risk assessment
- Retrospectives

### 🎨 Content Creation
Generate content for various platforms.
- YouTube metadata
- Blog posts
- Social media
- Email templates

### 🤖 AI Prompts
Advanced prompt engineering and AI interactions.
- Meta-prompt construction
- Prompt optimization
- AI integration guides

### 💼 Business
Professional business documents and communications.
- Proposals
- Meeting agendas
- Client communication
- Contracts

### 🔧 DevOps
Infrastructure and deployment configurations.
- Docker setups
- CI/CD pipelines
- Kubernetes configs
- Monitoring

## 🎯 Usage

These prompts are designed to work with the [Smart Prompts MCP Server](https://github.com/jezweb/smart-prompts-mcp).

### With Smart Prompts MCP

1. Configure the server to point to this repository:
```bash
export GITHUB_OWNER=jezweb
export GITHUB_REPO=prompts
```

2. Use the prompts in your MCP-compatible client:
```
search_prompts({ query: "docker" })
get_prompt({ name: "docker_setup" })
```

### Manual Usage

Each prompt includes:
- **YAML frontmatter** with metadata
- **Arguments** for customization
- **Template content** with Handlebars syntax

Example:
```yaml
---
name: example_prompt
arguments:
  - name: project_name
    required: true
---
# {{project_name}} Documentation
```

## 📝 Prompt Format

All prompts follow this structure:

```markdown
---
name: prompt_identifier
title: Human Readable Title
description: Brief description of the prompt
category: category-name
tags: [tag1, tag2, tag3]
difficulty: beginner|intermediate|advanced
author: contributor-name
version: 1.0.0
arguments:
  - name: arg_name
    description: Argument description
    required: true|false
    default: default_value
---

# Prompt Content

Your prompt template using {{arguments}} for dynamic content.
```

## 🤝 Contributing

We welcome contributions! To add a new prompt:

1. Fork this repository
2. Create a new prompt file in the appropriate category
3. Follow the format guidelines above
4. Test your prompt
5. Submit a pull request

### Guidelines

- Use clear, descriptive names
- Provide comprehensive descriptions
- Include practical examples
- Test with various inputs
- Keep prompts focused and modular

## 📊 Prompt Statistics

- **Total Prompts:** 30+
- **Categories:** 7
- **Contributors:** Growing community

## 🔍 Finding Prompts

### By Category
Browse the directory structure to find prompts by category.

### By Search
Use the Smart Prompts MCP Server search functionality:
```
search_prompts({ query: "your keyword" })
```

### By Tags
Common tags include:
- `testing`, `documentation`, `api`, `frontend`, `backend`
- `docker`, `kubernetes`, `deployment`, `ci-cd`
- `planning`, `agile`, `requirements`, `architecture`

## 📄 License

MIT License - Feel free to use these prompts in your projects.

## 🌟 Featured Prompts

### Most Popular
1. **docker_setup** - Complete Docker configuration generator
2. **unit_test** - Comprehensive unit test generator
3. **api_design** - RESTful API design guide
4. **react_component** - React component generator
5. **project_kickoff** - Project planning questionnaire

### Recently Added
- SQL query builder
- Debug assistant
- README generator
- Meta-prompt constructor

## 💡 Tips

1. **Customize arguments** to match your specific needs
2. **Combine prompts** using the compose feature
3. **Track usage** to identify most valuable prompts
4. **Contribute back** improvements you make

## 📞 Support

- Issues: [GitHub Issues](https://github.com/jezweb/prompts/issues)
- Discussions: [GitHub Discussions](https://github.com/jezweb/prompts/discussions)
- Email: jeremy@jezweb.com.au

---

Built with ❤️ by [Jezweb](https://jezweb.com.au) and contributors