---
name: prompt_name_here
title: Human Readable Title Here
description: Brief description of what this prompt does and when to use it
category: category-name
tags: [tag1, tag2, tag3, tag4]
difficulty: beginner|intermediate|advanced
author: your-name
version: 1.0.0
arguments:
  - name: argument_name
    description: Description of this argument
    required: true|false
    default: default_value_if_optional
  - name: another_argument
    description: Another argument description
    required: false
    default: "default value"
---

# Prompt Title Using {{argument_name}}

Brief introduction explaining the context and purpose.

## Section 1

Content that uses {{argument_name}} and {{another_argument}} with Handlebars syntax.

### Subsection

{{#if (eq argument_name "specific_value")}}
Conditional content shown only when argument matches specific value.
{{else}}
Alternative content for other cases.
{{/if}}

## Section 2

{{#each (split comma_separated_argument ",")}}
- {{trim this}}
{{/each}}

## Best Practices

- Use clear section headers
- Include examples where helpful
- Provide actionable content
- Use Handlebars helpers for dynamic content

## Template Helpers Available

- `{{#if condition}}...{{/if}}` - Conditional rendering
- `{{#each array}}...{{/each}}` - Iterate over arrays
- `{{#unless condition}}...{{/unless}}` - Negative conditional
- `{{join array separator}}` - Join array elements
- `{{split string separator}}` - Split string into array
- `{{trim string}}` - Remove whitespace
- `{{uppercase string}}` - Convert to uppercase
- `{{lowercase string}}` - Convert to lowercase
- `{{eq a b}}` - Check equality
- `{{includes string substring}}` - Check if string contains substring

## Example Usage

When using this prompt, provide:
- `argument_name`: [describe expected value]
- `another_argument`: [describe expected value]

## Notes

Add any additional notes, warnings, or tips for using this prompt effectively.