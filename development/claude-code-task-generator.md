---
name: claude_code_task_generator
title: Claude Code Task Generator for Development Projects
description: Generate well-structured prompts for Claude Code that follow best practices for autonomous coding tasks
category: development
tags: ["claude-code","development","automation","prompting","best-practices"]
difficulty: intermediate
author: Claude Commander Expert
version: 1.0
created: 2025-06-19T16:53:52.793Z
updated: 2025-06-19T16:53:52.793Z
arguments:
  - name: task_type
    description: Type of development task (e.g., 'REST API', 'React Component', 'Database Schema')
    required: true
  - name: task_description
    description: Detailed description of what needs to be built
    required: true
  - name: framework
    description: Primary technology/framework to use
    required: true
  - name: project_type
    description: Type of project (web app, API, mobile, etc.)
    required: true
  - name: complexity
    description: Task complexity level (simple, moderate, complex)
    required: true
  - name: requirements
    description: Array of specific requirements
    required: true
  - name: database
    description: Database technology if needed
    required: false
  - name: auth_required
    description: Whether authentication is needed (true/false)
    required: false
  - name: auth_type
    description: Type of authentication (JWT, OAuth, etc.)
    required: false
  - name: testing_required
    description: Whether testing is required (true/false)
    required: false
  - name: testing_framework
    description: Testing framework to use
    required: false
  - name: deployment
    description: Deployment target/method
    required: false
---

# Claude Code Task: {{task_type}}

## Project Context
**Framework/Technology:** {{framework}}
**Project Type:** {{project_type}}
**Complexity:** {{complexity}}

## Task Description
{{task_description}}

## Requirements
{{#each requirements}}
- {{this}}
{{/each}}

## Technical Specifications
- **Language/Framework:** {{framework}}
- **Database:** {{#if database}}{{database}}{{else}}Not specified{{/if}}
- **Authentication:** {{#if auth_required}}Required ({{auth_type}}){{else}}Not required{{/if}}
- **Testing:** {{#if testing_required}}Required ({{testing_framework}}){{else}}Optional{{/if}}
- **Deployment:** {{#if deployment}}{{deployment}}{{else}}Local development{{/if}}

## Implementation Approach
{{#if (eq complexity "complex")}}
### Phase 1: Planning
```
Plan how to implement {{task_description}} with:
- Architecture design
- Component breakdown
- Technology choices
- Implementation strategy
```

### Phase 2: Implementation
```
Implement the {{task_type}} based on the plan with:
{{#each requirements}}
- {{this}}
{{/each}}
```
{{else}}
### Direct Implementation
```
Create {{task_description}} with:
{{#each requirements}}
- {{this}}
{{/each}}

Using {{framework}}{{#if database}} with {{database}}{{/if}}{{#if auth_required}} and {{auth_type}} authentication{{/if}}
```
{{/if}}

## Success Criteria
- [ ] All requirements implemented
- [ ] Code follows best practices
- [ ] {{#if testing_required}}Tests pass{{else}}Manual testing successful{{/if}}
- [ ] Documentation updated
- [ ] {{#if deployment}}Deployment ready{{else}}Runs locally{{/if}}

## Claude Code Best Practices Applied
- ✅ Clear, specific requirements
- ✅ Technology stack specified
- ✅ {{#if (eq complexity "complex")}}Planning mode first, then coding{{else}}Direct coding approach{{/if}}
- ✅ Constraints and preferences defined
- ✅ Success criteria established

## Next Steps
1. {{#if (eq complexity "complex")}}Start with planning mode{{else}}Begin implementation{{/if}}
2. Grant write permissions when requested
3. Monitor claude.md for progress
4. Test functionality
5. Review and iterate if needed