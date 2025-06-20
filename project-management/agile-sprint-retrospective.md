---
name: agile_sprint_retrospective
title: Agile Sprint Retrospective Framework
description: Comprehensive sprint retrospective framework with facilitation techniques, action item tracking, and continuous improvement strategies for agile teams
category: project-management
tags: [agile, scrum, retrospective, continuous-improvement, team-collaboration, sprint-review]
difficulty: intermediate
author: jezweb
version: 1.0.0
arguments:
  - name: team_size
    description: Team size (small 3-5, medium 6-9, large 10+)
    required: true
  - name: sprint_length
    description: Sprint length in weeks (1, 2, 3, 4)
    required: true
  - name: team_maturity
    description: Team agile maturity (new, developing, mature, expert)
    required: true
  - name: retrospective_format
    description: Preferred format (traditional, starfish, sailboat, mad-sad-glad, custom)
    required: true
  - name: improvement_focus
    description: Main improvement focus (velocity, quality, collaboration, process, technical)
    required: true
  - name: team_dynamics
    description: Team dynamics (harmonious, some-conflict, high-conflict, distributed)
    required: false
    default: "harmonious"
---

# Sprint Retrospective: {{retrospective_format}} Format

**Team Size:** {{team_size}} members  
**Sprint Length:** {{sprint_length}} weeks  
**Team Maturity:** {{team_maturity}}  
**Focus Area:** {{improvement_focus}}  
**Team Dynamics:** {{team_dynamics}}

## 1. Retrospective Preparation

### Pre-Retrospective Setup
```markdown
# Retrospective Preparation Checklist

## 1-2 Days Before Retrospective
- [ ] Send calendar invite with agenda and preparation items
- [ ] Share sprint metrics and data with team
- [ ] Review previous retrospective action items
- [ ] Prepare digital collaboration tools (Miro, Mural, or physical whiteboard)
- [ ] Gather relevant artifacts (burndown charts, velocity metrics, incident reports)

## Pre-Meeting Materials to Share
{{#if (eq team_maturity "new")}}
- [ ] Brief explanation of retrospective purpose and format
- [ ] Ground rules for constructive feedback
- [ ] Examples of good retrospective input
{{/if}}

- [ ] Sprint goals and outcomes
- [ ] Key metrics from the sprint
- [ ] Previous action items and their status
- [ ] Any significant events or changes during sprint

## Meeting Setup
- **Duration:** {{#if (eq sprint_length "1")}}45 minutes{{else if (eq sprint_length "2")}}60 minutes{{else}}90 minutes{{/if}}
- **Participants:** Scrum Team ({{team_size}} people)
- **Facilitator:** {{#if (eq team_maturity "new")}}Scrum Master{{else}}Rotating team member{{/if}}
- **Environment:** Safe, confidential space for open discussion
```

### Data Collection and Metrics
```python
# Sprint metrics dashboard for retrospective
class SprintMetrics:
    def __init__(self, sprint_data):
        self.sprint_data = sprint_data
        
    def generate_metrics_summary(self):
        """Generate comprehensive sprint metrics"""
        
        metrics = {
            'velocity': {
                'planned_points': self.sprint_data.get('planned_story_points', 0),
                'completed_points': self.sprint_data.get('completed_story_points', 0),
                'completion_rate': self.calculate_completion_rate(),
                'velocity_trend': self.get_velocity_trend()
            },
            
            'quality': {
                'bugs_found': self.sprint_data.get('bugs_found', 0),
                'bugs_fixed': self.sprint_data.get('bugs_fixed', 0),
                'code_review_feedback': self.sprint_data.get('code_review_comments', 0),
                'technical_debt_added': self.sprint_data.get('tech_debt_stories', 0)
            },
            
            'process': {
                'ceremony_attendance': self.calculate_ceremony_attendance(),
                'backlog_refinement_hours': self.sprint_data.get('refinement_hours', 0),
                'unplanned_work_points': self.sprint_data.get('unplanned_points', 0),
                'scope_changes': self.sprint_data.get('scope_changes', 0)
            },
            
            'collaboration': {
                'pair_programming_hours': self.sprint_data.get('pair_hours', 0),
                'knowledge_sharing_sessions': self.sprint_data.get('knowledge_sessions', 0),
                'cross_team_dependencies': self.sprint_data.get('dependencies', 0),
                'blockers_count': self.sprint_data.get('blockers', 0)
            }
        }
        
        return metrics
    
    def calculate_completion_rate(self):
        planned = self.sprint_data.get('planned_story_points', 0)
        completed = self.sprint_data.get('completed_story_points', 0)
        return (completed / planned * 100) if planned > 0 else 0
    
    def get_velocity_trend(self):
        # Calculate trend from last 3-5 sprints
        historical_velocity = self.sprint_data.get('historical_velocity', [])
        if len(historical_velocity) < 2:
            return "Insufficient data"
        
        current = historical_velocity[-1]
        previous = sum(historical_velocity[-3:]) / len(historical_velocity[-3:])
        
        if current > previous * 1.1:
            return "Increasing"
        elif current < previous * 0.9:
            return "Decreasing"
        else:
            return "Stable"
    
    def calculate_ceremony_attendance(self):
        ceremonies = self.sprint_data.get('ceremony_attendance', {})
        total_possible = sum(ceremonies.values()) if ceremonies else 0
        expected_total = len(ceremonies) * {{team_size}} * {{sprint_length}} * 2  # Rough estimate
        
        return (total_possible / expected_total * 100) if expected_total > 0 else 0

# Visualization for retrospective
def create_retrospective_dashboard(metrics):
    """Create visual dashboard for retrospective discussion"""
    
    dashboard = f"""
    # Sprint {{sprint_length}}-Week Retrospective Dashboard
    
    ## Velocity & Delivery
    - **Completion Rate:** {metrics['velocity']['completion_rate']:.1f}%
    - **Velocity Trend:** {metrics['velocity']['velocity_trend']}
    - **Planned vs Delivered:** {metrics['velocity']['planned_points']} → {metrics['velocity']['completed_points']} points
    
    ## Quality Metrics
    - **Bugs Found:** {metrics['quality']['bugs_found']}
    - **Bugs Fixed:** {metrics['quality']['bugs_fixed']}
    - **Code Review Comments:** {metrics['quality']['code_review_feedback']}
    
    ## Process Health
    - **Ceremony Attendance:** {metrics['process']['ceremony_attendance']:.1f}%
    - **Unplanned Work:** {metrics['process']['unplanned_work_points']} points
    - **Scope Changes:** {metrics['process']['scope_changes']}
    
    ## Team Collaboration
    - **Pair Programming:** {metrics['collaboration']['pair_programming_hours']} hours
    - **Blockers:** {metrics['collaboration']['blockers_count']}
    - **Dependencies:** {metrics['collaboration']['cross_team_dependencies']}
    """
    
    return dashboard
```

## 2. {{retrospective_format}} Retrospective Format

{{#if (eq retrospective_format "traditional")}}
### Traditional What Went Well / What Didn't Go Well / Actions

#### Meeting Structure ({{#if (eq sprint_length "1")}}45{{else if (eq sprint_length "2")}}60{{else}}90{{/if}} minutes)

**1. Check-in (5 minutes)**
- Quick emotional check-in from each team member
- Set the stage for open, honest discussion

**2. Review Previous Actions (10 minutes)**
- Review action items from previous retrospective
- Update status and discuss outcomes
- Celebrate completed improvements

**3. Data Review (10 minutes)**
- Present sprint metrics and key data points
- Highlight significant events or changes
- Ensure everyone has shared context

**4. Generate Insights ({{#if (eq team_size "small")}}20{{else if (eq team_size "medium")}}25{{else}}30{{/if}} minutes)**

```markdown
## What Went Well? 🟢
*Things we should continue doing*

**Individual Reflection (5 minutes)**
- Each person writes sticky notes silently
- Focus on specific, actionable items
- Include both process and outcome successes

**Team Discussion ({{#if (eq team_size "small")}}10{{else}}15{{/if}} minutes)**
- Share items and group similar themes
- Discuss why these things worked well
- Identify patterns and root causes of success

## What Didn't Go Well? 🔴
*Things that created obstacles or frustration*

**Individual Reflection (5 minutes)**
- Focus on problems, not blame
- Be specific about impact and context
- Avoid personal attacks or generalizations

**Team Discussion ({{#if (eq team_size "small")}}10{{else}}15{{/if}} minutes)**
- Share and categorize issues
- Dig deeper into root causes
- Prioritize by impact and team control
```

**5. Generate Actions (15 minutes)**
- Brainstorm specific, actionable improvements
- Assign owners and deadlines
- Ensure actions are SMART (Specific, Measurable, Achievable, Relevant, Time-bound)

**6. Closing (5 minutes)**
- Summarize key decisions and actions
- Appreciation round
- Plan for follow-up
{{/if}}

{{#if (eq retrospective_format "starfish")}}
### Starfish Retrospective Format

#### Meeting Structure ({{#if (eq sprint_length "1")}}45{{else if (eq sprint_length "2")}}60{{else}}90{{/if}} minutes)

**1. Introduction and Setup (10 minutes)**
- Explain the starfish format
- Review ground rules for constructive feedback
- Set up digital or physical starfish diagram

**2. Individual Reflection (15 minutes)**
Each team member reflects on the sprint using five categories:

```markdown
## Starfish Categories

### ⭐ Keep Doing
*What's working well that we should continue?*
- Practices that add value
- Behaviors that support team success
- Processes that are efficient

### 🚀 Start Doing  
*What should we begin that would improve our work?*
- New practices to try
- Missing processes or tools
- Behaviors that would help the team

### 🛑 Stop Doing
*What should we cease because it's not helping?*
- Wasteful activities
- Counterproductive behaviors
- Inefficient processes

### 📈 More Of
*What are we doing that works but needs amplification?*
- Good practices that need expansion
- Positive behaviors to increase
- Successful processes to scale

### 📉 Less Of
*What should we reduce without eliminating entirely?*
- Over-engineered processes
- Excessive meetings or ceremonies
- Behaviors that have become burdensome
```

**3. Team Discussion (30 minutes)**
- Share items for each category (6 minutes per category)
- Group similar items and identify themes
- Discuss root causes and impacts

**4. Prioritization and Action Planning ({{#if (eq team_size "small")}}15{{else}}20{{/if}} minutes)**
- Vote on most important items to address
- Create specific action items from "Start," "More Of," and "Less Of"
- Assign owners and timelines

**5. Commitment and Closing (5 minutes)**
- Confirm team commitment to actions
- Schedule follow-up check-ins
{{/if}}

{{#if (eq retrospective_format "sailboat")}}
### Sailboat Retrospective Format

Perfect for teams focusing on {{improvement_focus}} improvements.

#### Setup and Metaphor
```markdown
# The Sailboat Journey 🛥️

Our team is a sailboat navigating toward our goals:
- **Wind**: What's helping us move forward?
- **Anchors**: What's slowing us down or holding us back?
- **Rocks**: What risks or obstacles might sink us?
- **Island**: Where are we trying to go? (Our goals/vision)
- **Sun**: What energizes and motivates the team?
```

#### Meeting Structure ({{#if (eq sprint_length "1")}}45{{else if (eq sprint_length "2")}}60{{else}}90{{/if}} minutes)

**1. Set the Scene (10 minutes)**
- Draw or display sailboat diagram
- Explain metaphor and categories
- Review sprint goals and context

**2. Individual Brainstorming (15 minutes)**
```markdown
## Reflection Questions

### 🌪️ Wind (Helping Us)
- What practices accelerated our progress?
- What tools or processes supported our success?
- What team behaviors created momentum?

### ⚓ Anchors (Holding Us Back) 
- What slowed down our delivery?
- What processes created friction?
- What skills or resources were we missing?

### 🪨 Rocks (Risks and Obstacles)
- What could derail our upcoming work?
- What dependencies worry you?
- What technical debt threatens our velocity?

### 🏝️ Island (Our Destination)
- Are we still aligned on our goals?
- What does success look like for next sprint?
- What outcomes matter most to our stakeholders?

### ☀️ Sun (What Energizes Us)
- What work excites the team?
- What achievements should we celebrate?
- What learning opportunities motivate us?
```

**3. Team Sharing and Clustering (25 minutes)**
- Share items for each category (5 minutes each)
- Group similar themes
- Identify patterns and connections

**4. Action Planning ({{#if (eq team_size "small")}}10{{else}}15{{/if}} minutes)**
- Focus on "Anchors" - what can we control and improve?
- Address "Rocks" - what risks can we mitigate?
- Leverage "Wind" - how can we do more of what works?

**5. Course Correction (5 minutes)**
- Confirm our destination (goals) are still correct
- Commit to navigation changes (action items)
{{/if}}

{{#if (eq retrospective_format "mad-sad-glad")}}
### Mad, Sad, Glad Retrospective Format

Excellent for teams with {{team_dynamics}} dynamics.

#### Emotional Check-in Focus
```markdown
# Emotional Retrospective: Mad, Sad, Glad 😡😢😊

This format helps us process emotions and turn feelings into actionable improvements.

## Guidelines for Emotional Safety
- Feelings are valid and important
- Focus on situations, not people
- Listen without judgment
- Turn emotions into constructive actions
```

#### Meeting Structure ({{#if (eq sprint_length "1")}}45{{else if (eq sprint_length "2")}}60{{else}}90{{/if}} minutes)

**1. Emotional Check-in (10 minutes)**
- Each person shares their overall sprint feeling
- No discussion, just listening and acknowledgment

**2. Individual Reflection (15 minutes)**
```markdown
## Reflection Categories

### 😡 Mad (Frustration and Anger)
- What made you frustrated during the sprint?
- What blocked your progress unnecessarily?
- What felt unfair or poorly handled?
- What wasted your time or energy?

### 😢 Sad (Disappointment and Regret)  
- What outcomes disappointed you?
- What opportunities did we miss?
- What didn't go as well as hoped?
- What would you have done differently?

### 😊 Glad (Joy and Satisfaction)
- What made you happy during the sprint?
- What achievements are you proud of?
- What exceeded your expectations?
- What energized you or the team?
```

**3. Facilitated Sharing (30 minutes)**

{{#if (eq team_dynamics "high-conflict")}}
**Special Facilitation for High-Conflict Teams:**
- Share "Glad" items first to start positively
- Use "I" statements for "Mad" and "Sad" items
- Focus on specific situations, not personality traits
- Paraphrase back what you hear before responding
{{/if}}

- **Glad sharing (10 minutes)**: Celebrate successes and positive moments
- **Sad sharing (10 minutes)**: Acknowledge disappointments with empathy
- **Mad sharing (10 minutes)**: Address frustrations constructively

**4. Emotion-to-Action Translation ({{#if (eq team_size "small")}}10{{else}}15{{/if}} minutes)**
- For each "Mad" item: What change would prevent this frustration?
- For each "Sad" item: What would help us achieve better outcomes?
- For each "Glad" item: How can we create more of these positive experiences?

**5. Commitment to Emotional Health (5 minutes)**
- Agree on actions that improve team emotional well-being
- Plan check-ins for emotional safety
{{/if}}

## 3. {{improvement_focus}} Focused Action Planning

{{#if (eq improvement_focus "velocity")}}
### Velocity Improvement Actions

#### Systematic Approach to Velocity Enhancement
```markdown
# Velocity Improvement Framework

## Current State Analysis
- **Average Velocity**: [X] story points per sprint
- **Velocity Trend**: [Increasing/Stable/Decreasing]
- **Completion Rate**: [X]% of planned work finished
- **Primary Velocity Blockers**: [List top 3]

## Root Cause Categories

### 🔄 Process Inefficiencies
- [ ] Story estimation accuracy
- [ ] Definition of Done clarity
- [ ] Handoff delays between team members
- [ ] Rework due to unclear requirements

### 🛠️ Technical Factors
- [ ] Code review bottlenecks
- [ ] Testing environment availability
- [ ] Technical debt slowing development
- [ ] Tool and infrastructure issues

### 👥 Team Factors
- [ ] Skill gaps or knowledge silos
- [ ] Team member availability/capacity
- [ ] Communication and collaboration issues
- [ ] External dependencies and waiting

### 📋 Planning Factors
- [ ] Scope creep within sprint
- [ ] Overcommitment in sprint planning
- [ ] Inadequate backlog refinement
- [ ] Poor story breakdown and sizing
```

#### Velocity Action Items Template
```markdown
## Sprint {{sprint_length}}-Week Velocity Actions

### Immediate Actions (This Sprint)
1. **[Action]**: [Specific improvement to implement]
   - **Owner**: [Team member name]
   - **Deadline**: [Specific date]
   - **Success Metric**: [How we'll measure improvement]
   - **Effort**: [Time investment required]

### Process Improvements (Next 2-3 Sprints)
2. **[Action]**: [Process change to implement]
   - **Owner**: [Team member name]
   - **Implementation Plan**: [Steps to implement]
   - **Expected Impact**: [Velocity increase expected]

### Long-term Investments (Next Quarter)
3. **[Action]**: [Larger improvement requiring sustained effort]
   - **Owner**: [Team member name]
   - **Milestones**: [Key checkpoints]
   - **Resource Requirements**: [Time, training, tools needed]
```
{{/if}}

{{#if (eq improvement_focus "quality")}}
### Quality Improvement Actions

#### Quality Metrics Focus
```markdown
# Quality Improvement Framework

## Current Quality State
- **Bugs Found in Sprint**: [X] bugs
- **Bug Escape Rate**: [X]% (bugs found in production)
- **Code Review Coverage**: [X]% of commits reviewed
- **Test Coverage**: [X]% automated test coverage
- **Technical Debt Stories**: [X] points of tech debt

## Quality Improvement Areas

### 🐛 Defect Prevention
- [ ] Improve requirements clarity and acceptance criteria
- [ ] Enhance Definition of Done with quality gates
- [ ] Implement pair programming for complex features
- [ ] Add more comprehensive unit and integration tests

### 🔍 Early Detection
- [ ] Strengthen code review process and checklists
- [ ] Improve automated testing in CI/CD pipeline
- [ ] Add static code analysis tools
- [ ] Implement feature toggles for safer releases

### 🛠️ Technical Excellence
- [ ] Address technical debt systematically
- [ ] Establish coding standards and enforce them
- [ ] Improve development environment consistency
- [ ] Invest in better tooling and automation

### 📊 Quality Measurement
- [ ] Track quality metrics consistently
- [ ] Set up quality dashboards
- [ ] Regular code quality audits
- [ ] Customer satisfaction feedback loops
```
{{/if}}

{{#if (eq improvement_focus "collaboration")}}
### Collaboration Enhancement Actions

#### Team Collaboration Assessment
```markdown
# Collaboration Improvement Framework

## Current Collaboration State
- **Pair Programming Hours**: [X] hours per sprint
- **Knowledge Sharing Sessions**: [X] sessions
- **Cross-functional Work**: [X]% of stories involve multiple skills
- **Communication Effectiveness**: [Rating 1-10]

## Collaboration Focus Areas

### 💬 Communication Enhancement
- [ ] Establish clear communication protocols
- [ ] Improve asynchronous communication tools
- [ ] Create team communication agreements
- [ ] Regular feedback and check-in sessions

### 🤝 Knowledge Sharing
- [ ] Implement regular knowledge transfer sessions
- [ ] Create documentation standards and practices
- [ ] Encourage pair programming and code reviews
- [ ] Cross-training on different technologies/domains

### 🔄 Process Collaboration
- [ ] Improve sprint planning collaboration
- [ ] Enhance daily standup effectiveness
- [ ] Better stakeholder involvement in refinement
- [ ] Cross-team dependency management

### 🌟 Team Building
- [ ] Regular team building activities
- [ ] Celebration of team achievements
- [ ] Conflict resolution processes
- [ ] Psychological safety improvements
```
{{/if}}

## 4. Action Item Tracking & Follow-up

### Action Item Template
```markdown
# Retrospective Action Items - Sprint [Number]

## Action Item #1
- **Action**: [Specific, actionable description]
- **Category**: {{improvement_focus}}
- **Owner**: [Primary responsible person]
- **Supporting Team Members**: [Others involved]
- **Target Completion**: [Specific date]
- **Success Criteria**: [How we'll know it's successful]
- **Effort Estimate**: [Time/complexity estimate]
- **Dependencies**: [What needs to happen first]
- **Status**: Not Started

## Action Item #2
[Repeat template]

## Previous Sprint Action Items

### Completed ✅
- [List completed actions with brief outcome notes]

### In Progress 🔄
- [List ongoing actions with current status]

### Not Started ❌
- [List actions not yet started with reason/plan]

### Abandoned 🗑️
- [List abandoned actions with rationale]
```

### Action Item Tracking System
```python
# Action item tracking for retrospectives
from datetime import datetime, timedelta
from typing import List, Dict, Any
from enum import Enum

class ActionStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"

class ActionItem:
    def __init__(self, title: str, description: str, owner: str, 
                 due_date: datetime, category: str = "{{improvement_focus}}"):
        self.id = self.generate_id()
        self.title = title
        self.description = description
        self.owner = owner
        self.due_date = due_date
        self.category = category
        self.status = ActionStatus.NOT_STARTED
        self.created_date = datetime.now()
        self.updates = []
        self.success_criteria = ""
        self.effort_estimate = ""
        
    def add_update(self, update: str, new_status: ActionStatus = None):
        """Add progress update to action item"""
        self.updates.append({
            'date': datetime.now(),
            'update': update,
            'status_change': new_status
        })
        
        if new_status:
            self.status = new_status
    
    def is_overdue(self) -> bool:
        """Check if action item is overdue"""
        return datetime.now() > self.due_date and self.status not in [
            ActionStatus.COMPLETED, ActionStatus.ABANDONED
        ]
    
    def days_until_due(self) -> int:
        """Calculate days until due date"""
        delta = self.due_date - datetime.now()
        return delta.days
    
    @staticmethod
    def generate_id() -> str:
        """Generate unique ID for action item"""
        return f"ACTION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class RetrospectiveActionTracker:
    def __init__(self):
        self.action_items: List[ActionItem] = []
        self.sprint_history: List[Dict] = []
    
    def add_action_item(self, action: ActionItem):
        """Add new action item from retrospective"""
        self.action_items.append(action)
    
    def get_active_actions(self) -> List[ActionItem]:
        """Get all non-completed action items"""
        return [
            action for action in self.action_items 
            if action.status not in [ActionStatus.COMPLETED, ActionStatus.ABANDONED]
        ]
    
    def get_overdue_actions(self) -> List[ActionItem]:
        """Get overdue action items"""
        return [action for action in self.action_items if action.is_overdue()]
    
    def generate_status_report(self) -> str:
        """Generate action item status report for retrospective"""
        
        active_actions = self.get_active_actions()
        overdue_actions = self.get_overdue_actions()
        
        report = f"""
# Action Item Status Report

## Summary
- **Total Active Actions**: {len(active_actions)}
- **Overdue Actions**: {len(overdue_actions)}
- **{{improvement_focus}} Focus Actions**: {len([a for a in active_actions if a.category == '{{improvement_focus}}'])}

## Overdue Actions ⚠️
"""
        
        for action in overdue_actions:
            report += f"""
### {action.title}
- **Owner**: {action.owner}
- **Due Date**: {action.due_date.strftime('%Y-%m-%d')}
- **Days Overdue**: {abs(action.days_until_due())}
- **Status**: {action.status.value}
- **Last Update**: {action.updates[-1]['update'] if action.updates else 'No updates'}
"""
        
        report += "\n## Actions Due This Sprint 📅\n"
        
        upcoming_actions = [
            action for action in active_actions 
            if 0 <= action.days_until_due() <= 14  # Due within 2 weeks
        ]
        
        for action in upcoming_actions:
            report += f"""
### {action.title}
- **Owner**: {action.owner}
- **Due Date**: {action.due_date.strftime('%Y-%m-%d')}
- **Days Until Due**: {action.days_until_due()}
- **Status**: {action.status.value}
"""
        
        return report
    
    def close_sprint_actions(self, sprint_number: int):
        """Archive completed actions and prepare for new sprint"""
        
        completed_actions = [
            action for action in self.action_items 
            if action.status == ActionStatus.COMPLETED
        ]
        
        sprint_summary = {
            'sprint_number': sprint_number,
            'date': datetime.now(),
            'completed_actions': len(completed_actions),
            'abandoned_actions': len([
                a for a in self.action_items 
                if a.status == ActionStatus.ABANDONED
            ]),
            'carried_forward': len(self.get_active_actions())
        }
        
        self.sprint_history.append(sprint_summary)

# Usage example for tracking
tracker = RetrospectiveActionTracker()

# Add action items from retrospective
action1 = ActionItem(
    title="Improve code review checklist",
    description="Create comprehensive code review checklist focusing on {{improvement_focus}}",
    owner="Senior Developer",
    due_date=datetime.now() + timedelta(days=14),
    category="{{improvement_focus}}"
)

action1.success_criteria = "Checklist created and used in 3 consecutive code reviews"
action1.effort_estimate = "4 hours"

tracker.add_action_item(action1)

# Generate report for next retrospective
status_report = tracker.generate_status_report()
print(status_report)
```

## 5. Continuous Improvement Metrics

### Retrospective Effectiveness Measurement
```python
# Measure retrospective and improvement effectiveness
class RetrospectiveMetrics:
    def __init__(self):
        self.retrospective_history = []
        self.improvement_trends = {}
    
    def record_retrospective_outcomes(self, sprint_number: int, data: Dict):
        """Record retrospective outcomes for trend analysis"""
        
        retrospective_data = {
            'sprint_number': sprint_number,
            'date': datetime.now(),
            'attendance': data.get('attendance', 0),
            'action_items_created': data.get('action_items_created', 0),
            'previous_actions_completed': data.get('previous_actions_completed', 0),
            'team_satisfaction_score': data.get('team_satisfaction', 0),  # 1-10 scale
            'retrospective_duration': data.get('duration_minutes', 0),
            'format_used': data.get('format', '{{retrospective_format}}'),
            'focus_area': '{{improvement_focus}}',
            'key_themes': data.get('themes', [])
        }
        
        self.retrospective_history.append(retrospective_data)
    
    def calculate_improvement_trends(self):
        """Calculate trends in team improvement metrics"""
        
        if len(self.retrospective_history) < 3:
            return "Insufficient data for trend analysis"
        
        recent_sprints = self.retrospective_history[-5:]  # Last 5 sprints
        
        trends = {
            'action_completion_rate': self.calculate_action_completion_trend(recent_sprints),
            'team_satisfaction_trend': self.calculate_satisfaction_trend(recent_sprints),
            'retrospective_engagement': self.calculate_engagement_trend(recent_sprints),
            'improvement_sustainability': self.calculate_sustainability_metrics(recent_sprints)
        }
        
        return trends
    
    def calculate_action_completion_trend(self, sprints):
        """Calculate trend in action item completion"""
        completion_rates = []
        
        for sprint in sprints:
            if sprint['action_items_created'] > 0:
                rate = sprint['previous_actions_completed'] / sprint['action_items_created']
                completion_rates.append(rate)
        
        if len(completion_rates) >= 2:
            recent_avg = sum(completion_rates[-2:]) / 2
            earlier_avg = sum(completion_rates[:-2]) / len(completion_rates[:-2]) if len(completion_rates) > 2 else completion_rates[0]
            
            if recent_avg > earlier_avg * 1.1:
                return "Improving"
            elif recent_avg < earlier_avg * 0.9:
                return "Declining"
            else:
                return "Stable"
        
        return "Insufficient data"
    
    def generate_improvement_dashboard(self):
        """Generate dashboard for continuous improvement tracking"""
        
        trends = self.calculate_improvement_trends()
        recent_data = self.retrospective_history[-3:] if len(self.retrospective_history) >= 3 else self.retrospective_history
        
        dashboard = f"""
# Continuous Improvement Dashboard

## Overall Improvement Health
- **Focus Area**: {{improvement_focus}}
- **Team Maturity**: {{team_maturity}}
- **Retrospective Format**: {{retrospective_format}}

## Recent Trends (Last {len(recent_data)} Sprints)
- **Action Completion**: {trends.get('action_completion_rate', 'N/A')}
- **Team Satisfaction**: {trends.get('team_satisfaction_trend', 'N/A')}
- **Engagement Level**: {trends.get('retrospective_engagement', 'N/A')}

## Key Metrics
"""
        
        if recent_data:
            avg_satisfaction = sum(sprint.get('team_satisfaction_score', 0) for sprint in recent_data) / len(recent_data)
            avg_actions = sum(sprint.get('action_items_created', 0) for sprint in recent_data) / len(recent_data)
            
            dashboard += f"""
- **Average Team Satisfaction**: {avg_satisfaction:.1f}/10
- **Average Actions per Sprint**: {avg_actions:.1f}
- **Retrospective Attendance**: {recent_data[-1].get('attendance', 0)}/{{{team_size}}} team members
"""
        
        dashboard += """
## Improvement Recommendations
"""
        
        # Generate recommendations based on trends and team maturity
        if trends.get('action_completion_rate') == 'Declining':
            dashboard += "- 🔍 Review action item feasibility and ownership clarity\n"
        
        if trends.get('team_satisfaction_trend') == 'Declining':
            dashboard += "- 💬 Consider changing retrospective format or facilitation approach\n"
        
        if '{{team_maturity}}' == 'new':
            dashboard += "- 📚 Provide more structure and examples in retrospectives\n"
        
        if '{{team_dynamics}}' in ['some-conflict', 'high-conflict']:
            dashboard += "- 🤝 Focus on team building and psychological safety improvements\n"
        
        return dashboard

# Example usage
metrics = RetrospectiveMetrics()

# Record retrospective outcomes
sprint_data = {
    'attendance': {{team_size}},
    'action_items_created': 3,
    'previous_actions_completed': 2,
    'team_satisfaction': 7.5,
    'duration_minutes': {{#if (eq sprint_length "1")}}45{{else if (eq sprint_length "2")}}60{{else}}90{{/if}},
    'themes': ['{{improvement_focus}}', 'team collaboration', 'process efficiency']
}

metrics.record_retrospective_outcomes(1, sprint_data)

# Generate improvement dashboard
dashboard = metrics.generate_improvement_dashboard()
print(dashboard)
```

## 6. Team Maturity Adaptations

{{#if (eq team_maturity "new")}}
### Adaptations for New Agile Teams

#### Additional Structure and Guidance
```markdown
# New Team Retrospective Adaptations

## Pre-Retrospective Education
- [ ] Explain retrospective purpose and benefits
- [ ] Share examples of good retrospective inputs
- [ ] Set clear expectations for participation
- [ ] Address any concerns about psychological safety

## Enhanced Facilitation
- **Facilitator Role**: Always facilitated by experienced Scrum Master
- **Time Management**: Strict timeboxing with visible timers
- **Participation**: Ensure everyone contributes equally
- **Documentation**: Clear capture of all inputs and decisions

## Learning Focus
- Emphasize learning over blame
- Celebrate small improvements
- Build retrospective muscle gradually
- Focus on process basics before advanced techniques

## Follow-up Support
- Weekly check-ins on action item progress
- Additional coaching on agile practices
- Gradual increase in team self-management
```
{{/if}}

{{#if (eq team_maturity "expert")}}
### Adaptations for Expert Agile Teams

#### Advanced Retrospective Techniques
```markdown
# Expert Team Retrospective Enhancements

## Self-Facilitation
- Rotate facilitation among team members
- Experiment with new retrospective formats
- Create custom formats for specific challenges
- Integrate with other improvement practices

## Advanced Analysis
- Root cause analysis using techniques like 5 Whys
- Systems thinking approaches to problems
- Metrics-driven improvement decisions
- Cross-team learning and sharing

## Innovation Focus
- Experiment with new practices and tools
- Lead improvement initiatives across organization
- Mentor other teams in retrospective practices
- Contribute to agile community of practice

## Strategic Alignment
- Connect sprint improvements to broader organizational goals
- Influence team and organizational practices
- Drive cultural and process innovations
```
{{/if}}

## Conclusion

This sprint retrospective framework provides:

**Key Features:**
- {{retrospective_format}} format optimized for {{team_size}} team
- {{improvement_focus}} focused improvement strategies
- Comprehensive action item tracking and follow-up
- Team maturity appropriate facilitation techniques

**Benefits:**
- Systematic continuous improvement approach
- Data-driven retrospective insights
- Sustainable action item management
- Team engagement and psychological safety

**Success Metrics:**
- Increased team satisfaction scores
- Higher action item completion rates
- Improved {{improvement_focus}} metrics
- Enhanced team collaboration and trust