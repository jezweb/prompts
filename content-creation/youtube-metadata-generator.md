---
name: youtube_metadata_generator
title: YouTube Video Metadata Generator for Tech Content
description: Generate complete YouTube metadata (titles, descriptions, and thumbnail prompts) for tech-focused videos
category: content-creation
tags: [youtube, video, metadata, content-creation, tech, marketing]
difficulty: intermediate
author: jezweb
version: 1.0.0
arguments:
  - name: video_summary
    description: Brief 1-2 sentence summary of the video
    required: true
  - name: primary_technology
    description: Main tool/platform being demonstrated
    required: true
  - name: video_category
    description: Type of video (Tutorial, Setup, Demo, etc.)
    required: true
  - name: key_tasks
    description: Main tasks demonstrated (comma-separated)
    required: true
  - name: target_audience
    description: Who this video helps
    required: true
  - name: video_length
    description: Approximate video duration
    required: false
    default: "10-15 minutes"
---

# YouTube Video Metadata Generator for Tech Content

**Goal:** Generate complete YouTube metadata for tech-focused videos featuring Jeremy, CEO of Jezweb.

## Video Details

**Summary:** {{video_summary}}
**Technology:** {{primary_technology}}
**Category:** {{video_category}}
**Target Audience:** {{target_audience}}
**Duration:** {{video_length}}

## Generated YouTube Titles (Choose One)

### Clickable & SEO-Optimized Options:

1. 🚀 {{primary_technology}} {{video_category}}: {{video_summary}}
2. How to {{#if (includes key_tasks "build")}}Build{{else if (includes key_tasks "setup")}}Set Up{{else}}Use{{/if}} {{primary_technology}} in {{video_length}} (Complete Guide)
3. {{primary_technology}} for {{target_audience}} - Full {{video_category}} 2024
4. Master {{primary_technology}}: {{video_summary}} [Step-by-Step]
5. {{video_category}}: {{primary_technology}} Made Simple | Jezweb Tech

## YouTube Description

```
🎯 In this {{video_category}}, Jeremy from Jezweb demonstrates {{video_summary}}

Perfect for {{target_audience}} looking to leverage {{primary_technology}} for their projects.

⏱️ TIMESTAMPS:
00:00 - Introduction & Overview
01:30 - {{primary_technology}} Setup & Configuration
03:45 - Main Implementation
[Add specific timestamps based on content]
15:00 - Conclusion & Next Steps

📚 WHAT YOU'LL LEARN:
{{#each (split key_tasks ",")}}
✅ {{trim this}}
{{/each}}

🛠️ TECHNOLOGIES COVERED:
• {{primary_technology}} (Primary)
• Related tools and integrations
• Best practices and tips

💼 WHO THIS IS FOR:
{{target_audience}}

🔗 RESOURCES & LINKS:
• Jezweb Official: https://jezweb.com.au
• {{primary_technology}} Documentation: [Add Link]
• Source Code/Files: [Add if applicable]
• Related Tutorial: [Add if applicable]

💬 CONNECT WITH JEZWEB:
• Website: https://jezweb.com.au
• LinkedIn: [Your LinkedIn]
• Twitter/X: [Your Twitter]
• GitHub: https://github.com/jezweb

📧 BUSINESS INQUIRIES:
For custom development, consulting, or collaboration:
contact@jezweb.com.au

🔔 SUBSCRIBE & ENABLE NOTIFICATIONS
Don't miss future tutorials on {{primary_technology}} and other tech tools!

#{{primary_technology}} #{{video_category}} #WebDevelopment #TechTutorial #Jezweb #{{join (split primary_technology " ") ""}} #Tutorial2024 #DeveloperTools #TechCEO
```

## Thumbnail Generation Prompts

### Simple Prompt:
```
Create a YouTube thumbnail showing:
- Split screen with code editor showing {{primary_technology}} on left
- Jeremy (professional tech CEO) on right pointing at the code
- Bold text overlay: "{{primary_technology}} {{video_category}}"
- Tech-themed background with blue/purple gradient
- Jezweb logo in corner
```

### Detailed Prompt:
```
Design a professional YouTube thumbnail (1280x720px) for a tech tutorial:

LAYOUT:
- Left 60%: Screenshot of {{primary_technology}} interface/code with key features highlighted
- Right 40%: Professional photo of Jeremy (tech CEO, business casual) with engaging expression
- Overlay elements with slight drop shadow for depth

TEXT ELEMENTS:
- Main title: "{{primary_technology}}" (large, bold, white with dark outline)
- Subtitle: "{{video_category}}" (medium, yellow accent color)
- Small badge: "{{video_length}}" (top corner)

VISUAL STYLE:
- Modern tech aesthetic with gradient background (blue to purple)
- Subtle geometric patterns or circuit board elements
- High contrast for mobile visibility
- Professional but approachable mood

BRANDING:
- Jezweb logo (bottom right, semi-transparent)
- Consistent color scheme with brand colors
- Clean, minimalist design approach
```

## Additional Metadata

**Video Tags:**
{{primary_technology}}, {{video_category}}, web development, tech tutorial, Jezweb, Jeremy CEO, {{join (split primary_technology " ") " tutorial"}}, coding tutorial, developer tools, tech tips, software development, {{target_audience}}, tutorial 2024

**Category:** Science & Technology
**Language:** English
**License:** Standard YouTube License