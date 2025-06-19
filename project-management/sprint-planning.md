---
name: sprint_planning
title: Agile Sprint Planning Session
description: Facilitate comprehensive sprint planning with user stories, capacity planning, and sprint goals
category: project-management
tags: [agile, scrum, sprint, planning, project-management, team]
difficulty: intermediate
author: jezweb
version: 1.0.0
arguments:
  - name: sprint_number
    description: Sprint number
    required: true
  - name: sprint_duration
    description: Sprint duration in weeks
    required: false
    default: "2"
  - name: team_size
    description: Number of team members
    required: true
  - name: velocity
    description: Average team velocity (story points)
    required: false
    default: "40"
  - name: product_name
    description: Product or project name
    required: true
---

# Sprint {{sprint_number}} Planning: {{product_name}}

**Sprint Duration:** {{sprint_duration}} weeks  
**Team Size:** {{team_size}} members  
**Target Velocity:** {{velocity}} story points

---

## 📅 Sprint Overview

### Sprint Timeline
- **Start Date:** [Date]
- **End Date:** [Date]
- **Sprint Review:** [Date & Time]
- **Sprint Retrospective:** [Date & Time]

### Key Dates & Events
- [ ] Day 1: Sprint Planning (Today)
- [ ] Day 3: Technical Design Review
- [ ] Day {{#if (eq sprint_duration "1")}}3{{else}}7{{/if}}: Mid-Sprint Check-in
- [ ] Day {{#if (eq sprint_duration "1")}}4{{else}}9{{/if}}: Code Freeze
- [ ] Final Day: Sprint Review & Demo

---

## 🎯 Sprint Goal

### Primary Goal
> **[Clear, concise statement of what the team aims to achieve this sprint]**

### Success Criteria
1. ✓ [Specific measurable outcome 1]
2. ✓ [Specific measurable outcome 2]
3. ✓ [Specific measurable outcome 3]

### Business Value
- **Impact**: [How this sprint moves the product forward]
- **Users Affected**: [Which user segments benefit]
- **Metrics**: [KPIs that will improve]

---

## 👥 Team Capacity Planning

### Team Availability
| Team Member | Role | Capacity | Notes |
|-------------|------|----------|--------|
| [Name 1] | [Role] | {{#if (eq sprint_duration "2")}}80%{{else}}90%{{/if}} | [Any planned time off] |
| [Name 2] | [Role] | 100% | - |
| [Name 3] | [Role] | {{#if (eq sprint_duration "2")}}80%{{else}}90%{{/if}} | [Support duty on Day X] |
{{#if (gt team_size 3)}}
| [Name 4] | [Role] | 100% | - |
| [Name 5] | [Role] | {{#if (eq sprint_duration "2")}}80%{{else}}90%{{/if}} | [Conference on Day Y] |
{{/if}}

### Capacity Calculation
- **Total Working Days**: {{#if (eq sprint_duration "1")}}5{{else if (eq sprint_duration "2")}}10{{else}}15{{/if}} days
- **Team Working Days**: {{#if (eq sprint_duration "1")}}{{multiply team_size 5}}{{else if (eq sprint_duration "2")}}{{multiply team_size 10}}{{else}}{{multiply team_size 15}}{{/if}} person-days
- **Adjusted Capacity**: ~85% = [X] person-days
- **Focus Factor**: 70% (meetings, overhead)
- **Available Story Points**: {{velocity}}

---

## 📋 Sprint Backlog

### Committed User Stories

#### 🟢 Story 1: [User Story Title]
**ID**: [JIRA/Issue ID]  
**Points**: 8  
**Priority**: High

**As a** [user type]  
**I want** [feature/functionality]  
**So that** [benefit/value]

**Acceptance Criteria:**
- [ ] Given [context], when [action], then [outcome]
- [ ] Given [context], when [action], then [outcome]
- [ ] Given [context], when [action], then [outcome]

**Technical Notes:**
- API endpoint needed: `/api/v1/[endpoint]`
- Database migration required
- Dependencies: [Other stories/tasks]

**Assignee**: [Team member name]

---

#### 🟢 Story 2: [User Story Title]
**ID**: [JIRA/Issue ID]  
**Points**: 5  
**Priority**: High

**As a** [user type]  
**I want** [feature/functionality]  
**So that** [benefit/value]

**Acceptance Criteria:**
- [ ] Given [context], when [action], then [outcome]
- [ ] Given [context], when [action], then [outcome]

**Technical Notes:**
- Frontend component update
- Integration with existing service

**Assignee**: [Team member name]

---

#### 🟡 Story 3: [User Story Title]
**ID**: [JIRA/Issue ID]  
**Points**: 13  
**Priority**: Medium

**As a** [user type]  
**I want** [feature/functionality]  
**So that** [benefit/value]

**Acceptance Criteria:**
- [ ] Given [context], when [action], then [outcome]
- [ ] Given [context], when [action], then [outcome]
- [ ] Performance: Response time < 200ms

**Technical Notes:**
- New microservice required
- Cache implementation
- Load testing needed

**Assignee**: [Team member name]

---

### Technical Debt & Improvements

#### 🔧 Tech Debt 1: [Refactoring Task]
**Points**: 3  
**Impact**: Performance improvement
**Details**: [What needs to be refactored and why]

#### 🔧 Tech Debt 2: [Security Update]
**Points**: 2  
**Impact**: Security compliance
**Details**: [Security vulnerability to address]

---

### Bug Fixes

#### 🐛 Bug 1: [Bug Title]
**Severity**: High  
**Points**: 2  
**Description**: [Bug description]
**Steps to Reproduce**: [Clear steps]

---

## 📊 Sprint Metrics

### Story Point Distribution
- **Committed**: {{velocity}} points
- **Stretch Goals**: 8 points
- **Total**: {{add velocity 8}} points

### Work Type Breakdown
| Type | Points | Percentage |
|------|--------|------------|
| New Features | {{multiply velocity 0.6}} | 60% |
| Tech Debt | {{multiply velocity 0.2}} | 20% |
| Bugs | {{multiply velocity 0.1}} | 10% |
| Support/Other | {{multiply velocity 0.1}} | 10% |

### Risk Assessment
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| [External dependency delay] | Medium | High | Start integration early |
| [Complex feature implementation] | Low | Medium | Spike in first 2 days |
| [Team member absence] | Low | Low | Knowledge sharing sessions |

---

## 🔄 Dependencies & Blockers

### External Dependencies
1. **Marketing Team**: Copy for new feature by Day 3
2. **Design Team**: Final mockups for Story 3 by Day 2
3. **DevOps**: Production access for deployment
4. **Backend Team**: API endpoint completion

### Known Blockers
- ⚠️ [Blocker 1]: Waiting for [what] from [who]
- ⚠️ [Blocker 2]: Technical constraint with [system]

### Mitigation Plans
- If [dependency] is delayed, we will [alternative plan]
- Parallel work streams identified for blocking scenarios

---

## 🎓 Learning & Development

### Sprint Learning Goals
1. **Team Goal**: Implement [new technology/pattern]
2. **Individual Goals**:
   - [Name 1]: Learn and implement [skill/tech]
   - [Name 2]: Mentor on [topic]
   - [Name 3]: Documentation for [feature]

### Knowledge Sharing
- **Tech Talk**: [Topic] by [Name] on Day {{multiply sprint_duration 3}}
- **Pair Programming**: Story 3 (complex implementation)
- **Code Review Focus**: [Specific area for improvement]

---

## 📝 Definition of Done

### Story Completion Checklist
- [ ] Code complete and pushed to repository
- [ ] Unit tests written and passing (coverage > 80%)
- [ ] Integration tests completed
- [ ] Code reviewed by at least one team member
- [ ] Documentation updated
- [ ] Acceptance criteria verified
- [ ] Deployed to staging environment
- [ ] Product Owner acceptance
- [ ] No critical bugs
- [ ] Performance requirements met

### Sprint Completion Criteria
- [ ] All committed stories meet Definition of Done
- [ ] Sprint goal achieved
- [ ] Demo prepared for stakeholders
- [ ] Retrospective actions identified
- [ ] Next sprint backlog refined

---

## 🚀 Sprint Ceremonies

### Daily Standups
- **Time**: 9:30 AM (15 minutes)
- **Format**: What I did, What I'll do, Blockers
- **Location**: [Physical/Virtual location]

### Mid-Sprint Review
- **When**: Day {{#if (eq sprint_duration "1")}}3{{else}}{{multiply sprint_duration 5}}{{/if}}
- **Purpose**: Assess progress, adjust if needed
- **Attendees**: Scrum team only

### Sprint Review
- **Duration**: 1 hour
- **Demo Order**:
  1. Story 1 - [Name]
  2. Story 2 - [Name]
  3. Story 3 - [Name]
- **Stakeholders**: [List of invitees]

### Sprint Retrospective
- **Duration**: 1.5 hours
- **Format**: [Start-Stop-Continue / 4Ls / Other]
- **Previous Actions Review**: Yes

---

## 📈 Success Metrics

### Sprint KPIs
- **Velocity**: Target {{velocity}} points
- **Commitment Accuracy**: 90-110% of committed points
- **Bug Escape Rate**: < 2 production bugs
- **Code Coverage**: > 80%
- **Customer Satisfaction**: > 4.5/5

### Team Health Metrics
- **Team Morale**: [Measure method]
- **Collaboration Index**: Pair programming hours
- **Learning Goals Met**: [X/Y achieved]

---

## 🎬 Action Items

### Immediate Actions (Today)
1. [ ] Update JIRA board with sprint backlog
2. [ ] Send sprint commitment email to stakeholders
3. [ ] Schedule all sprint ceremonies
4. [ ] Set up development environments
5. [ ] Create feature branches

### First 2 Days
1. [ ] Complete technical design documents
2. [ ] Set up CI/CD pipelines for new features
3. [ ] Clarify all acceptance criteria with PO
4. [ ] Identify and document technical risks

### Ongoing
1. [ ] Daily standups
2. [ ] Update burn-down chart
3. [ ] Communicate blockers immediately
4. [ ] Prepare demo scenarios

---

## 📎 References

### Documentation
- [Product Roadmap](link)
- [Technical Architecture](link)
- [API Documentation](link)
- [Previous Sprint Reports](link)

### Tools
- **Project Management**: JIRA/Trello/Azure DevOps
- **Communication**: Slack channel #sprint-{{sprint_number}}
- **Code Repository**: [Git repo link]
- **CI/CD**: [Pipeline link]

---

**Sprint Commitment**: The team commits to completing {{velocity}} story points and achieving the sprint goal.

**Next Sprint Planning**: [Date & Time]

---

*Remember: It's better to under-commit and over-deliver than the opposite. Focus on quality and sustainable pace.*