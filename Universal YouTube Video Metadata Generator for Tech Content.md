# Universal YouTube Video Metadata Generator for Tech Content

**Overall Goal:** Generate complete YouTube metadata (titles, description, and thumbnail prompts) for tech-focused videos featuring Jeremy, CEO of Jezweb. This prompt covers AI tools, web apps, SaaS platforms, automation, development tools, and related technologies.

**Important:** Do not create the app or explain the video content directly. All output must be structured as ready-to-use YouTube metadata for easy publishing.

Based on the specific video details provided below, generate:
1. **Multiple distinct YouTube title suggestions** (4-5 options), formatted as a markdown list
2. **A detailed YouTube video description** (formatted as markdown text for YouTube with emojis/Unicode and clickable links/timestamps)
3. **Image generation prompts for YouTube thumbnail** (both simple and detailed options)

---

## Video Specific Details (Fill this section for each video)

### Core Video Information
*   **[Brief Video Summary]:** (1-2 sentences describing the main focus/goal. E.g., "This video demonstrates how to build/use/configure [specific tool/platform] to [achieve specific outcome] for [target use case].")

*   **[Primary Technology/Platform]:** (Main tool being demonstrated. E.g., Firebase, n8n, AnythingLLM, LibreChat, Google AI Studio, VS Code Extension, React, Next.js, Supabase, etc.)

*   **[Video Category]:** (Select one: Tutorial/Walkthrough, Setup/Installation, Feature Demo, Comparison, Tips & Tricks, Project Build, Integration Guide, Review/Analysis)

### Technical Details
*   **[Key Tasks/Challenges Demonstrated]:** (List main things shown. Use bullet points. E.g.,
    *   Setting up/installing [platform/tool]
    *   Configuring [specific features/settings]
    *   Building/creating [specific output/result]
    *   Integrating with [other tools/services]
    *   Demonstrating [key workflows/processes]
    *   Troubleshooting [common issues])

*   **[Specific Features/Capabilities Highlighted]:** (List prominent functionalities shown. E.g.,
    *   [Platform] UI/interface navigation
    *   Specific commands/functions/nodes used
    *   Integration capabilities
    *   Customization options
    *   Performance/efficiency benefits
    *   Security/privacy features)

*   **[Technologies/Tools/APIs Used]:** (List supporting tech stack. E.g., JavaScript, React, Python, REST APIs, specific AI models, databases, hosting platforms, third-party services)

*   **[Target Audience Benefit]:** (Who this helps and how. E.g., "Developers wanting to automate workflows," "Businesses needing custom AI solutions," "Teams looking to streamline processes")

### Video Structure
*   **[Key Timestamps]:** (List important moments with descriptions. Use MM:SS or HH:MM:SS format. E.g.,
    *   00:00 - Intro & What We're Building/Demonstrating
    *   01:30 - Platform/Tool Overview & Setup
    *   03:45 - [Core Feature 1] Configuration
    *   07:20 - [Core Feature 2] Implementation
    *   11:15 - [Advanced Feature/Integration]
    *   15:30 - Testing & Results Demo
    *   18:00 - Benefits & Business Applications
    *   19:30 - Next Steps & Resources)

*   **[Main Visual Element for Thumbnail]:** (Describe the most representative visual. E.g., "[Platform] interface with key features highlighted," "Split screen showing before/after results," "Workflow diagram with connected tools," "Code editor with [specific technology] integration")

---

## AI Generation Task

Generate the following based on the details above:

### 1. YouTube Title Suggestions (4-5 Options)
*   Generate 4-5 distinct, engaging titles
*   Include the primary technology/platform name
*   Mention key benefits or outcomes (e.g., "Build," "Automate," "Deploy," "Master," "Complete Guide")
*   Vary styles: descriptive, question-based, benefit-focused, tutorial-focused
*   **Format as markdown list with `- ` prefix**

### 2. YouTube Description (Markdown Formatted for YouTube)
*   **Format:** Single block of markdown text with YouTube-compatible formatting including:
    *   Emojis for structure (✨, 🚀, 💡, 🛠️, ⏰, 👇, 👍, 🔔, 🔗, ⚙️, 💻, 🎯, 📱, 🌐)
    *   Clickable timestamps (MM:SS or HH:MM:SS format)
    *   Full URLs (https://...) for clickable links
    *   `#hashtag` format for hashtags
    *   **bold** and *italic* for emphasis where appropriate

*   **Content Structure:**
    *   **Hook:** Engaging opening using `[Brief Video Summary]` and `[Target Audience Benefit]`
    *   **Demo Overview:** Explain main tasks shown using `[Key Tasks/Challenges Demonstrated]` with clickable timestamps from `[Key Timestamps]`
    *   **Key Features & Capabilities:** Detail specific functionalities using `[Specific Features/Capabilities Highlighted]` with bullet points or emoji markers
    *   **Tech Stack (if applicable):** Mention technologies used from `[Technologies/Tools/APIs Used]`
    *   **🔗 Useful Resources & Links:**
        *   Primary platform documentation/website
        *   Related official resources
        *   Jezweb Web Agency: https://www.jezweb.com.au
        *   Jezweb Github: https://github.com/jezweb
        *   Buy Me a Coffee: https://buymeacoffee.com/jezweb
        *   Linkedin: https://www.linkedin.com/in/jeremydawes/
        *   Additional relevant tools/resources mentioned
    *   **⏰ TIMESTAMPS ⏰** section with provided timestamps
    *   **About Jezweb & This Channel:** Brief mention of Jeremy's expertise and Jezweb's services related to the video topic
    *   **Call to Action:** Prompt for engagement, comments about their experience/needs, likes, and subscriptions
    *   **Hashtags:** Generate 8-12 relevant hashtags including the primary technology, category tags, and general tech tags

### 3. Thumbnail Image Generation Prompts

**3.A. Simple Thumbnail Prompt (Concise & Direct)**
*   Concise text prompt (under 50 words) for AI image generator
*   Focus on 2-3 key elements: main subject, essential text, style
*   Include:
    *   **Main Subject:** Based on `[Main Visual Element for Thumbnail]`
    *   **Key Text Overlay:** Platform name + video category (e.g., "[Platform] Tutorial," "Build with [Tech]")
    *   **Style:** "Clean tech graphic," "Modern interface," "Professional tutorial style"
    *   **Colors:** Platform's brand colors or clean tech palette
    *   **Output:** "16:9 aspect ratio, YouTube thumbnail format"

**3.B. Detailed Thumbnail Prompt (Comprehensive & Descriptive)**
*   Detailed prompt for advanced AI image generators
*   Include:
    *   **Desired Output:** "16:9 static image for YouTube thumbnail, modern tech aesthetic suitable for developer/business audience"
    *   **Style/Mood:** Professional, innovative, efficient, tech-focused, trustworthy
    *   **Background:** Stylized representation of the platform interface, abstract tech patterns, or clean gradient
    *   **Key Elements:** Incorporate `[Main Visual Element for Thumbnail]` with recognizable icons, UI elements, or workflow representations. Abstract symbols representing the technology category (AI brain, automation gears, code brackets, etc.)
    *   **Text Overlay:** Clear, bold text including platform name and key benefit/action. Modern, readable sans-serif font
    *   **Color Palette:** Platform brand colors with complementary tech colors (blues, teals, greens) and contrasting accent colors
    *   **Composition:** Clean, uncluttered design with clear focal point and logical visual flow
    *   **Professional Elements:** Subtle "CEO/Expert" positioning without being overly promotional

### Tone and Style Guidelines
*   **Educational & Authoritative:** Position Jeremy as a knowledgeable CEO sharing practical insights
*   **Business-Focused:** Emphasize real-world applications and business value
*   **Accessible:** Technical content explained clearly for various skill levels
*   **Action-Oriented:** Focus on what viewers will learn and be able to do
*   **Professional:** Reflect Jezweb's expertise while being approachable

### Final Output Format
*   **Titles:** Markdown list with `- ` prefix for each title
*   **Description:** Single block of markdown-formatted text, ready to copy-paste into YouTube
*   **Thumbnail Prompts:** Text blocks for chosen option(s), ready for AI image generation

---

*This prompt is designed to handle any tech topic Jeremy covers while maintaining consistent branding and professional positioning for Jezweb.*
