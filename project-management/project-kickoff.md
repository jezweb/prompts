---
name: project_kickoff
title: Software Project Kickoff Questionnaire
description: Comprehensive questionnaire to gather requirements and context for new software projects
category: project-management
tags: [project-management, requirements, planning, kickoff, software-development]
difficulty: intermediate
author: jezweb
version: 1.0.0
arguments:
  - name: project_name
    description: Tentative name for the project
    required: true
  - name: project_type
    description: Type of project (web app, mobile app, API, etc.)
    required: true
  - name: timeline
    description: Expected timeline (MVP in X weeks/months)
    required: false
    default: "3-6 months"
  - name: team_size
    description: Expected team size
    required: false
    default: "2-5 developers"
---

# Software Project Kickoff: {{project_name}}

**Project Type:** {{project_type}}
**Expected Timeline:** {{timeline}}
**Team Size:** {{team_size}}

Please answer each question comprehensively to establish the foundation for **{{project_name}}**.

## I. Project Vision & Purpose

### 1. Problem/Opportunity
**What specific problem is this project solving, or what opportunity is it capitalizing on?**
- Current pain points:
- Market opportunity:
- Business impact:

### 2. Core Purpose
**In 1-2 sentences, what is the absolute core purpose of {{project_name}}?**

### 3. Motivation
**Why is this project important to undertake *now*?**
- Market timing:
- Business drivers:
- Competitive advantage:

### 4. Success Definition
**What does ultimate success look like?**
- Qualitative metrics:
- Quantitative goals:
- Key milestones:

## II. Target Users & Stakeholders

### 1. Primary Users
**Who are the primary end-users?**
- User profiles:
- Technical proficiency:
- Typical tasks:
- User volume:

### 2. User Needs
**What key needs will {{project_name}} address?**
- Pain points:
- Desired outcomes:
- Current workarounds:

### 3. Secondary Users
**Are there other user groups?**
- Administrators:
- Support staff:
- External partners:

### 4. Key Stakeholders
**Who are the business stakeholders?**
- Decision makers:
- Budget holders:
- Subject matter experts:
- Their primary interests:

## III. Scope & Functionality

### 1. Core MVP Functionality
**Essential features for the Minimum Viable Product:**

As a **[user type]**, I want to **[action]** so that **[benefit]**.

1. 
2. 
3. 
4. 
5. 

### 2. Key MVP Deliverables
**Tangible outputs of the MVP:**
- 
- 
- 

### 3. Future Ideas (Post-MVP)
**Planned enhancements for future iterations:**
- Phase 2:
- Phase 3:
- Long-term vision:

### 4. Out of Scope (for MVP)
**What will NOT be included in the MVP:**
- 
- 
- 

## IV. Technical Considerations & Constraints

### 1. Preferred Technologies/Stack
**Mandated or preferred technologies:**
- Frontend:
- Backend:
- Database:
- Infrastructure:
- AI/ML tools:
- Rationale:

### 2. Existing Systems & Integrations
**Required integrations:**
- Internal systems:
- External APIs:
- Data sources:
- Authentication:

### 3. Data Requirements
**Data handling needs:**
- Data types:
- Volume estimates:
- Security requirements:
- Compliance (GDPR, CCPA, etc.):
- Backup/recovery:

### 4. Deployment Environment
**Hosting and infrastructure:**
- Environment: {{#if (eq project_type "web app")}}Cloud (AWS/Azure/GCP){{else if (eq project_type "mobile app")}}App stores{{else}}On-premise/Cloud{{/if}}
- Constraints:
- Preferences:

### 5. Scalability Needs
**Growth expectations:**
- MVP load: ___ users, ___ requests/day
- 6-month projection:
- 1-year projection:
- Peak load scenarios:

### 6. Security Requirements
**Security standards:**
- Authentication method:
- Authorization model:
- Data encryption:
- Compliance standards:

## V. Non-Functional Requirements

### 1. Performance
**Performance expectations:**
- Page load time: < ___ seconds
- API response time: < ___ ms
- Concurrent users: ___
- Data processing speed:

### 2. Usability
**User experience goals:**
- Accessibility standards:
- Mobile responsiveness:
- Browser support:
- Learning curve:

### 3. Reliability/Availability
**Uptime requirements:**
- Target uptime: ____%
- Maintenance windows:
- Disaster recovery:
- Backup frequency:

### 4. Maintainability
**Post-launch maintenance:**
- Update frequency:
- Support model:
- Documentation needs:
- Knowledge transfer:

## VI. Project Context & Practicalities

### 1. Existing Solutions/Alternatives
**Current market solutions:**
- Internal tools:
- Competitor products:
- Their strengths:
- Their weaknesses:
- Our differentiation:

### 2. Assumptions
**Key project assumptions:**
- User behavior:
- Technology:
- Market conditions:
- Resource availability:

### 3. Potential Risks & Challenges
**Major risk factors:**
- Technical risks:
- Business risks:
- Resource risks:
- Mitigation strategies:

### 4. Budget & Resources
**Available resources:**
- Budget range:
- Team composition:
- Skill gaps:
- Training needs:

### 5. Timeline Expectations
**Key dates:**
- Project start:
- MVP delivery:
- Beta testing:
- Production launch:
- Critical deadlines:

## VII. Success Metrics & KPIs

### 1. MVP Success Criteria
**How we'll measure MVP success:**
- User adoption:
- Performance metrics:
- Business impact:
- Technical goals:

### 2. Ongoing KPIs
**Post-launch health indicators:**
- Daily/Monthly Active Users:
- User retention:
- System performance:
- Business metrics:
- User satisfaction:

## Additional Notes
**Any other important context:**


---

**Next Steps:**
1. Review and validate responses with stakeholders
2. Create detailed project plan based on this information
3. Define technical architecture
4. Establish development timeline
5. Set up project infrastructure