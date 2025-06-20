---
name: technical_debt_assessment
title: Technical Debt Assessment & Management
description: Comprehensive technical debt assessment framework with prioritization strategies, refactoring planning, and continuous monitoring for sustainable development
category: project-management
tags: [technical-debt, code-quality, refactoring, architecture, maintainability, legacy-code]
difficulty: intermediate
author: jezweb
version: 1.0.0
arguments:
  - name: codebase_size
    description: Codebase size (small <50k LOC, medium 50k-200k LOC, large >200k LOC)
    required: true
  - name: team_size
    description: Development team size (small 2-5, medium 6-15, large >15)
    required: true
  - name: business_pressure
    description: Business pressure level (low, moderate, high, critical)
    required: true
  - name: technology_stack
    description: Primary technology stack (javascript, python, java, csharp, php, mixed)
    required: true
  - name: debt_focus
    description: Primary debt concern (performance, maintainability, security, scalability)
    required: true
  - name: timeline
    description: Assessment timeline (immediate, quarterly, annual, ongoing)
    required: true
---

# Technical Debt Assessment: {{codebase_size}} Codebase

**Team Size:** {{team_size}} developers  
**Business Pressure:** {{business_pressure}}  
**Technology:** {{technology_stack}}  
**Focus Area:** {{debt_focus}}  
**Timeline:** {{timeline}}

## 1. Technical Debt Discovery & Measurement

### Automated Code Analysis Setup
```python
# Comprehensive technical debt analysis framework
import ast
import os
import subprocess
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TechnicalDebtMetric:
    category: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    location: str
    effort_estimate: int  # hours
    business_impact: str
    technical_impact: str

class TechnicalDebtAnalyzer:
    def __init__(self, project_path: str, technology: str = "{{technology_stack}}"):
        self.project_path = project_path
        self.technology = technology
        self.debt_items = []
        self.metrics = {}
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete technical debt analysis"""
        
        print("🔍 Starting comprehensive technical debt analysis...")
        
        # 1. Static code analysis
        self.analyze_code_quality()
        
        # 2. Architecture analysis
        self.analyze_architecture()
        
        # 3. Dependencies analysis
        self.analyze_dependencies()
        
        # 4. Performance hotspots
        self.analyze_performance_issues()
        
        # 5. Security vulnerabilities
        self.analyze_security_debt()
        
        # 6. Documentation gaps
        self.analyze_documentation_debt()
        
        # 7. Test coverage analysis
        self.analyze_test_debt()
        
        return self.generate_debt_report()
    
    def analyze_code_quality(self):
        """Analyze code quality metrics"""
        
        {{#if (eq technology_stack "javascript")}}
        # JavaScript/TypeScript analysis
        self.run_eslint_analysis()
        self.analyze_complexity()
        self.find_code_smells()
        {{/if}}
        
        {{#if (eq technology_stack "python")}}
        # Python analysis
        self.run_pylint_analysis()
        self.run_mypy_analysis()
        self.analyze_python_complexity()
        {{/if}}
        
        {{#if (eq technology_stack "java")}}
        # Java analysis
        self.run_spotbugs_analysis()
        self.analyze_java_complexity()
        self.check_java_patterns()
        {{/if}}
        
        # Common analysis for all languages
        self.analyze_duplication()
        self.analyze_file_sizes()
        self.analyze_function_complexity()
    
    {{#if (eq technology_stack "javascript")}}
    def run_eslint_analysis(self):
        """Run ESLint analysis for JavaScript/TypeScript"""
        try:
            result = subprocess.run([
                'npx', 'eslint', self.project_path,
                '--format', 'json',
                '--ext', '.js,.jsx,.ts,.tsx'
            ], capture_output=True, text=True)
            
            if result.stdout:
                eslint_results = json.loads(result.stdout)
                
                for file_result in eslint_results:
                    for message in file_result.get('messages', []):
                        if message['severity'] == 2:  # Error level
                            self.debt_items.append(TechnicalDebtMetric(
                                category='code_quality',
                                severity='medium' if message['ruleId'] else 'high',
                                description=f"ESLint: {message['message']}",
                                location=f"{file_result['filePath']}:{message['line']}",
                                effort_estimate=self.estimate_fix_effort(message['ruleId']),
                                business_impact='Low',
                                technical_impact='Medium'
                            ))
        except Exception as e:
            print(f"ESLint analysis failed: {e}")
    
    def analyze_complexity(self):
        """Analyze cyclomatic complexity for JavaScript"""
        try:
            # Use complexity-report or similar tool
            result = subprocess.run([
                'npx', 'complexity-report',
                '--output', 'json',
                self.project_path
            ], capture_output=True, text=True)
            
            if result.stdout:
                complexity_data = json.loads(result.stdout)
                
                for file_data in complexity_data.get('reports', []):
                    for function in file_data.get('functions', []):
                        if function.get('complexity', {}).get('cyclomatic', 0) > 10:
                            self.debt_items.append(TechnicalDebtMetric(
                                category='complexity',
                                severity='high' if function['complexity']['cyclomatic'] > 20 else 'medium',
                                description=f"High cyclomatic complexity: {function['complexity']['cyclomatic']}",
                                location=f"{file_data['path']}:{function['line']}",
                                effort_estimate=function['complexity']['cyclomatic'] * 2,
                                business_impact='Medium',
                                technical_impact='High'
                            ))
        except Exception as e:
            print(f"Complexity analysis failed: {e}")
    {{/if}}
    
    def analyze_duplication(self):
        """Analyze code duplication"""
        try:
            # Use jscpd or similar tool for duplication detection
            result = subprocess.run([
                'npx', 'jscpd',
                '--reporters', 'json',
                '--output', './jscpd-report.json',
                self.project_path
            ], capture_output=True, text=True)
            
            if os.path.exists('./jscpd-report.json'):
                with open('./jscpd-report.json', 'r') as f:
                    duplication_data = json.load(f)
                
                for duplicate in duplication_data.get('duplicates', []):
                    if duplicate.get('percentage', 0) > 5:  # More than 5% duplication
                        self.debt_items.append(TechnicalDebtMetric(
                            category='duplication',
                            severity='medium',
                            description=f"Code duplication: {duplicate['percentage']:.1f}%",
                            location=f"{duplicate['firstFile']['name']} <-> {duplicate['secondFile']['name']}",
                            effort_estimate=duplicate['lines'] // 2,  # Rough estimate
                            business_impact='Low',
                            technical_impact='Medium'
                        ))
        except Exception as e:
            print(f"Duplication analysis failed: {e}")
    
    def analyze_dependencies(self):
        """Analyze dependency-related technical debt"""
        
        # Outdated dependencies
        self.find_outdated_dependencies()
        
        # Unused dependencies
        self.find_unused_dependencies()
        
        # Vulnerable dependencies
        self.find_vulnerable_dependencies()
        
        # Dependency conflicts
        self.find_dependency_conflicts()
    
    def find_outdated_dependencies(self):
        """Find outdated dependencies"""
        {{#if (eq technology_stack "javascript")}}
        try:
            result = subprocess.run([
                'npm', 'outdated', '--json'
            ], capture_output=True, text=True, cwd=self.project_path)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                
                for package, info in outdated.items():
                    major_version_behind = self.compare_versions(
                        info['current'], info['latest']
                    )
                    
                    severity = 'high' if major_version_behind > 2 else 'medium'
                    
                    self.debt_items.append(TechnicalDebtMetric(
                        category='dependencies',
                        severity=severity,
                        description=f"Outdated dependency: {package} ({info['current']} -> {info['latest']})",
                        location='package.json',
                        effort_estimate=4 if major_version_behind > 1 else 2,
                        business_impact='Low',
                        technical_impact='Medium'
                    ))
        except Exception as e:
            print(f"Outdated dependencies analysis failed: {e}")
        {{/if}}
        
        {{#if (eq technology_stack "python")}}
        try:
            # Check for outdated Python packages
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                
                for package in outdated:
                    self.debt_items.append(TechnicalDebtMetric(
                        category='dependencies',
                        severity='medium',
                        description=f"Outdated package: {package['name']} ({package['version']} -> {package['latest_version']})",
                        location='requirements.txt',
                        effort_estimate=2,
                        business_impact='Low',
                        technical_impact='Medium'
                    ))
        except Exception as e:
            print(f"Python outdated packages analysis failed: {e}")
        {{/if}}
    
    def analyze_architecture(self):
        """Analyze architectural debt"""
        
        # Circular dependencies
        self.find_circular_dependencies()
        
        # Large files/modules
        self.find_large_modules()
        
        # God objects/classes
        self.find_god_objects()
        
        # Tight coupling
        self.analyze_coupling()
    
    def find_circular_dependencies(self):
        """Find circular dependencies in the codebase"""
        dependency_graph = self.build_dependency_graph()
        cycles = self.detect_cycles(dependency_graph)
        
        for cycle in cycles:
            self.debt_items.append(TechnicalDebtMetric(
                category='architecture',
                severity='high',
                description=f"Circular dependency: {' -> '.join(cycle)}",
                location=' -> '.join(cycle),
                effort_estimate=len(cycle) * 8,  # 8 hours per module in cycle
                business_impact='Medium',
                technical_impact='High'
            ))
    
    def analyze_performance_issues(self):
        """Analyze performance-related technical debt"""
        
        # N+1 query patterns
        self.find_n_plus_one_queries()
        
        # Inefficient algorithms
        self.find_inefficient_algorithms()
        
        # Memory leaks potential
        self.find_memory_leak_patterns()
        
        # Unoptimized database queries
        self.analyze_database_queries()
    
    def generate_debt_report(self) -> Dict[str, Any]:
        """Generate comprehensive technical debt report"""
        
        # Categorize debt items
        debt_by_category = {}
        debt_by_severity = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        total_effort = 0
        
        for item in self.debt_items:
            # By category
            if item.category not in debt_by_category:
                debt_by_category[item.category] = []
            debt_by_category[item.category].append(item)
            
            # By severity
            debt_by_severity[item.severity] += 1
            
            # Total effort
            total_effort += item.effort_estimate
        
        # Calculate debt metrics
        debt_ratio = self.calculate_debt_ratio()
        maintainability_index = self.calculate_maintainability_index()
        
        report = {
            'summary': {
                'total_debt_items': len(self.debt_items),
                'total_effort_hours': total_effort,
                'debt_ratio': debt_ratio,
                'maintainability_index': maintainability_index,
                'assessment_date': datetime.now().isoformat(),
                'codebase_size': '{{codebase_size}}',
                'primary_concerns': self.identify_primary_concerns()
            },
            'debt_by_category': {
                category: {
                    'count': len(items),
                    'effort_hours': sum(item.effort_estimate for item in items),
                    'severity_breakdown': self.get_severity_breakdown(items)
                }
                for category, items in debt_by_category.items()
            },
            'debt_by_severity': debt_by_severity,
            'top_priority_items': self.prioritize_debt_items()[:10],
            'recommendations': self.generate_recommendations(),
            'action_plan': self.create_action_plan()
        }
        
        return report
    
    def prioritize_debt_items(self) -> List[TechnicalDebtMetric]:
        """Prioritize debt items based on multiple factors"""
        
        def priority_score(item: TechnicalDebtMetric) -> float:
            severity_weight = {
                'critical': 10,
                'high': 7,
                'medium': 4,
                'low': 1
            }
            
            business_impact_weight = {
                'High': 3,
                'Medium': 2,
                'Low': 1
            }
            
            technical_impact_weight = {
                'High': 3,
                'Medium': 2,
                'Low': 1
            }
            
            # Consider business pressure
            pressure_multiplier = {
                'critical': 2.0,
                'high': 1.5,
                'moderate': 1.2,
                'low': 1.0
            }
            
            base_score = (
                severity_weight.get(item.severity, 1) +
                business_impact_weight.get(item.business_impact, 1) +
                technical_impact_weight.get(item.technical_impact, 1)
            )
            
            # Factor in effort (prefer quick wins)
            effort_factor = max(0.1, 1.0 / (item.effort_estimate / 8))  # Normalize to days
            
            return base_score * pressure_multiplier['{{business_pressure}}'] * effort_factor
        
        return sorted(self.debt_items, key=priority_score, reverse=True)
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # High-level recommendations based on debt focus
        if '{{debt_focus}}' == 'performance':
            recommendations.extend([
                {
                    'category': 'Performance',
                    'title': 'Implement Performance Monitoring',
                    'description': 'Set up comprehensive performance monitoring to identify bottlenecks',
                    'effort': 'Medium',
                    'impact': 'High',
                    'timeline': '2-3 weeks'
                },
                {
                    'category': 'Performance',
                    'title': 'Database Query Optimization',
                    'description': 'Review and optimize slow database queries',
                    'effort': 'High',
                    'impact': 'High',
                    'timeline': '4-6 weeks'
                }
            ])
        
        elif '{{debt_focus}}' == 'maintainability':
            recommendations.extend([
                {
                    'category': 'Maintainability',
                    'title': 'Refactor Large Functions/Classes',
                    'description': 'Break down complex functions and classes into smaller, focused units',
                    'effort': 'High',
                    'impact': 'High',
                    'timeline': '6-8 weeks'
                },
                {
                    'category': 'Maintainability',
                    'title': 'Improve Test Coverage',
                    'description': 'Increase test coverage to at least 80% for critical components',
                    'effort': 'Medium',
                    'impact': 'Medium',
                    'timeline': '3-4 weeks'
                }
            ])
        
        # Team size specific recommendations
        if '{{team_size}}' == 'large':
            recommendations.append({
                'category': 'Team Coordination',
                'title': 'Establish Code Review Standards',
                'description': 'Implement consistent code review practices to prevent future debt',
                'effort': 'Low',
                'impact': 'High',
                'timeline': '1-2 weeks'
            })
        
        return recommendations
    
    def create_action_plan(self) -> Dict[str, Any]:
        """Create phased action plan for debt reduction"""
        
        prioritized_items = self.prioritize_debt_items()
        
        # Organize into phases based on timeline and business pressure
        {{#if (eq timeline "immediate")}}
        phases = {
            'Phase 1 (Immediate - 2 weeks)': [],
            'Phase 2 (Short-term - 1 month)': [],
            'Phase 3 (Medium-term - 3 months)': []
        }
        
        quick_wins = [item for item in prioritized_items if item.effort_estimate <= 8]
        critical_items = [item for item in prioritized_items if item.severity == 'critical']
        
        phases['Phase 1 (Immediate - 2 weeks)'] = quick_wins[:5] + critical_items[:3]
        phases['Phase 2 (Short-term - 1 month)'] = prioritized_items[8:15]
        phases['Phase 3 (Medium-term - 3 months)'] = prioritized_items[15:25]
        {{else}}
        phases = {
            'Phase 1 (Quick Wins - 1 month)': [],
            'Phase 2 (High Impact - 3 months)': [],
            'Phase 3 (Strategic - 6 months)': [],
            'Phase 4 (Long-term - 12 months)': []
        }
        
        # Distribute based on effort and impact
        quick_wins = [item for item in prioritized_items if item.effort_estimate <= 16]
        high_impact = [item for item in prioritized_items if item.severity in ['critical', 'high']]
        
        phases['Phase 1 (Quick Wins - 1 month)'] = quick_wins[:8]
        phases['Phase 2 (High Impact - 3 months)'] = high_impact[:10]
        phases['Phase 3 (Strategic - 6 months)'] = prioritized_items[18:30]
        phases['Phase 4 (Long-term - 12 months)'] = prioritized_items[30:]
        {{/if}}
        
        return {
            'phases': phases,
            'resource_allocation': self.calculate_resource_allocation(phases),
            'success_metrics': self.define_success_metrics(),
            'risk_mitigation': self.identify_risks()
        }
    
    def calculate_resource_allocation(self, phases: Dict) -> Dict[str, Any]:
        """Calculate resource allocation for each phase"""
        
        allocation = {}
        team_capacity = {{team_size}} * 40 * 4  # team_size * hours_per_week * weeks_per_month
        
        for phase_name, items in phases.items():
            total_effort = sum(item.effort_estimate for item in items)
            
            allocation[phase_name] = {
                'total_effort_hours': total_effort,
                'estimated_duration_weeks': total_effort / ({{team_size}} * 40),
                'team_utilization': min(100, (total_effort / team_capacity) * 100),
                'recommended_allocation': min(30, (total_effort / team_capacity) * 100)  # Max 30% for debt
            }
        
        return allocation

# Example usage and reporting
analyzer = TechnicalDebtAnalyzer(
    project_path="/path/to/project",
    technology="{{technology_stack}}"
)

debt_report = analyzer.run_comprehensive_analysis()

# Generate dashboard
print("\\n" + "="*60)
print("📊 TECHNICAL DEBT ASSESSMENT REPORT")
print("="*60)
print(f"Total Debt Items: {debt_report['summary']['total_debt_items']}")
print(f"Total Effort Required: {debt_report['summary']['total_effort_hours']} hours")
print(f"Debt Ratio: {debt_report['summary']['debt_ratio']:.2f}")
print(f"Maintainability Index: {debt_report['summary']['maintainability_index']:.1f}/100")
```

## 2. Debt Categorization & Prioritization

### Technical Debt Categories
```yaml
# Comprehensive debt categorization framework
debt_categories:
  code_quality:
    description: "Issues with code structure, readability, and standards"
    examples:
      - "Complex functions (>50 lines)"
      - "High cyclomatic complexity (>10)"
      - "Code duplication (>5%)"
      - "Naming convention violations"
      - "Missing error handling"
    impact: "Maintainability, Developer Productivity"
    
  architecture:
    description: "Structural and design issues in system architecture"
    examples:
      - "Circular dependencies"
      - "God objects/classes"
      - "Tight coupling"
      - "Missing abstraction layers"
      - "Monolithic components"
    impact: "Scalability, Maintainability, Extensibility"
    
  performance:
    description: "Code that impacts system performance"
    examples:
      - "N+1 query problems"
      - "Inefficient algorithms"
      - "Memory leaks"
      - "Blocking operations"
      - "Unoptimized database queries"
    impact: "User Experience, System Resources"
    
  security:
    description: "Security vulnerabilities and weak practices"
    examples:
      - "Vulnerable dependencies"
      - "Hardcoded credentials"
      - "SQL injection vulnerabilities"
      - "Missing input validation"
      - "Insecure communication"
    impact: "Security, Compliance, Business Risk"
    
  dependencies:
    description: "Issues with external dependencies and libraries"
    examples:
      - "Outdated dependencies"
      - "Unused dependencies"
      - "Vulnerable packages"
      - "Conflicting versions"
      - "Deprecated libraries"
    impact: "Security, Maintainability, Performance"
    
  testing:
    description: "Insufficient or poor quality tests"
    examples:
      - "Low test coverage (<70%)"
      - "Flaky tests"
      - "Missing integration tests"
      - "Slow test suites"
      - "Outdated test data"
    impact: "Quality, Reliability, Development Speed"
    
  documentation:
    description: "Missing or outdated documentation"
    examples:
      - "Missing API documentation"
      - "Outdated README files"
      - "No code comments"
      - "Missing architecture diagrams"
      - "Incomplete setup instructions"
    impact: "Developer Onboarding, Maintainability"
```

### Prioritization Matrix
```python
# Advanced debt prioritization system
class DebtPrioritizationMatrix:
    def __init__(self, business_context: str = "{{business_pressure}}"):
        self.business_context = business_context
        self.prioritization_weights = self.get_prioritization_weights()
    
    def get_prioritization_weights(self) -> Dict[str, float]:
        """Get prioritization weights based on business context"""
        
        if self.business_context == 'critical':
            return {
                'business_impact': 0.4,
                'technical_impact': 0.3,
                'effort': 0.2,
                'risk': 0.1
            }
        elif self.business_context == 'high':
            return {
                'business_impact': 0.35,
                'technical_impact': 0.35,
                'effort': 0.2,
                'risk': 0.1
            }
        else:  # moderate or low
            return {
                'business_impact': 0.25,
                'technical_impact': 0.35,
                'effort': 0.25,
                'risk': 0.15
            }
    
    def calculate_priority_score(self, debt_item: TechnicalDebtMetric) -> float:
        """Calculate priority score for a debt item"""
        
        # Business Impact Score (1-10)
        business_score = self.get_business_impact_score(debt_item)
        
        # Technical Impact Score (1-10)
        technical_score = self.get_technical_impact_score(debt_item)
        
        # Effort Score (1-10, inverted - lower effort = higher score)
        effort_score = self.get_effort_score(debt_item)
        
        # Risk Score (1-10)
        risk_score = self.get_risk_score(debt_item)
        
        # Weighted calculation
        priority_score = (
            business_score * self.prioritization_weights['business_impact'] +
            technical_score * self.prioritization_weights['technical_impact'] +
            effort_score * self.prioritization_weights['effort'] +
            risk_score * self.prioritization_weights['risk']
        )
        
        return priority_score
    
    def get_business_impact_score(self, debt_item: TechnicalDebtMetric) -> float:
        """Calculate business impact score"""
        
        category_impact = {
            'security': 9,
            'performance': 8,
            'reliability': 7,
            'maintainability': 6,
            'code_quality': 4,
            'documentation': 3
        }
        
        base_score = category_impact.get(debt_item.category, 5)
        
        # Adjust based on {{debt_focus}}
        if debt_item.category == '{{debt_focus}}':
            base_score += 2
        
        return min(10, base_score)
    
    def get_technical_impact_score(self, debt_item: TechnicalDebtMetric) -> float:
        """Calculate technical impact score"""
        
        severity_scores = {
            'critical': 10,
            'high': 8,
            'medium': 5,
            'low': 2
        }
        
        return severity_scores.get(debt_item.severity, 5)
    
    def get_effort_score(self, debt_item: TechnicalDebtMetric) -> float:
        """Calculate effort score (inverted - lower effort = higher score)"""
        
        # Convert hours to effort score
        if debt_item.effort_estimate <= 4:
            return 10  # Quick win
        elif debt_item.effort_estimate <= 16:
            return 8   # Small task
        elif debt_item.effort_estimate <= 40:
            return 6   # Medium task
        elif debt_item.effort_estimate <= 80:
            return 4   # Large task
        else:
            return 2   # Very large task
    
    def get_risk_score(self, debt_item: TechnicalDebtMetric) -> float:
        """Calculate risk score of not addressing the debt"""
        
        risk_factors = {
            'security': 10,  # High risk if not addressed
            'performance': 7,
            'reliability': 8,
            'maintainability': 6,
            'architecture': 7
        }
        
        return risk_factors.get(debt_item.category, 5)
    
    def create_priority_buckets(self, debt_items: List[TechnicalDebtMetric]) -> Dict[str, List]:
        """Organize debt items into priority buckets"""
        
        scored_items = [(item, self.calculate_priority_score(item)) for item in debt_items]
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        total_items = len(scored_items)
        
        buckets = {
            'P0_Critical': scored_items[:max(1, total_items // 10)],
            'P1_High': scored_items[total_items // 10:total_items // 4],
            'P2_Medium': scored_items[total_items // 4:total_items // 2],
            'P3_Low': scored_items[total_items // 2:]
        }
        
        return buckets

# Usage example
prioritizer = DebtPrioritizationMatrix("{{business_pressure}}")
priority_buckets = prioritizer.create_priority_buckets(debt_items)
```

## 3. Refactoring Strategy & Planning

### Strategic Refactoring Approach
```python
# Strategic refactoring planning framework
class RefactoringStrategy:
    def __init__(self, codebase_size: str = "{{codebase_size}}", team_size: str = "{{team_size}}"):
        self.codebase_size = codebase_size
        self.team_size = team_size
        self.refactoring_patterns = self.load_refactoring_patterns()
    
    def create_refactoring_plan(self, priority_debt_items: List[TechnicalDebtMetric]) -> Dict[str, Any]:
        """Create comprehensive refactoring plan"""
        
        plan = {
            'strategy': self.determine_refactoring_strategy(),
            'phases': self.plan_refactoring_phases(priority_debt_items),
            'patterns': self.recommend_refactoring_patterns(priority_debt_items),
            'resource_allocation': self.calculate_refactoring_resources(),
            'risk_mitigation': self.plan_risk_mitigation(),
            'success_metrics': self.define_refactoring_metrics()
        }
        
        return plan
    
    def determine_refactoring_strategy(self) -> str:
        """Determine the best refactoring strategy"""
        
        if self.codebase_size == 'large' and self.team_size == 'large':
            return 'parallel_component_refactoring'
        elif self.codebase_size == 'large':
            return 'strangler_fig_pattern'
        elif '{{business_pressure}}' == 'critical':
            return 'tactical_refactoring'
        else:
            return 'strategic_refactoring'
    
    def plan_refactoring_phases(self, debt_items: List[TechnicalDebtMetric]) -> Dict[str, Any]:
        """Plan refactoring phases based on dependencies and priorities"""
        
        phases = {}
        
        if '{{timeline}}' == 'immediate':
            phases = {
                'Phase 1 - Critical Fixes (Week 1-2)': {
                    'focus': 'Security and critical performance issues',
                    'items': [item for item in debt_items if item.severity == 'critical'][:5],
                    'approach': 'Surgical fixes with minimal refactoring'
                },
                'Phase 2 - Quick Wins (Week 3-4)': {
                    'focus': 'High-impact, low-effort improvements',
                    'items': [item for item in debt_items if item.effort_estimate <= 8][:8],
                    'approach': 'Targeted refactoring of specific methods/functions'
                }
            }
        else:
            phases = {
                'Phase 1 - Foundation (Month 1-2)': {
                    'focus': 'Core architecture and security issues',
                    'items': [item for item in debt_items if item.category in ['architecture', 'security']][:10],
                    'approach': 'Strategic refactoring with comprehensive testing'
                },
                'Phase 2 - Performance (Month 3-4)': {
                    'focus': 'Performance optimization and database improvements',
                    'items': [item for item in debt_items if item.category == 'performance'][:8],
                    'approach': 'Performance-driven refactoring with benchmarking'
                },
                'Phase 3 - Maintainability (Month 5-6)': {
                    'focus': 'Code quality and maintainability improvements',
                    'items': [item for item in debt_items if item.category == 'code_quality'][:12],
                    'approach': 'Continuous refactoring with incremental improvements'
                }
            }
        
        return phases
    
    def recommend_refactoring_patterns(self, debt_items: List[TechnicalDebtMetric]) -> List[Dict[str, Any]]:
        """Recommend specific refactoring patterns"""
        
        patterns = []
        
        # Analyze debt items and recommend patterns
        if any(item.category == 'architecture' for item in debt_items):
            patterns.append({
                'pattern': 'Extract Service',
                'description': 'Break down monolithic components into focused services',
                'when_to_use': 'Large classes or modules with multiple responsibilities',
                'effort': 'High',
                'impact': 'High'
            })
        
        if any('complexity' in item.description for item in debt_items):
            patterns.append({
                'pattern': 'Extract Method',
                'description': 'Break down complex methods into smaller, focused methods',
                'when_to_use': 'Methods with high cyclomatic complexity',
                'effort': 'Low',
                'impact': 'Medium'
            })
        
        if any('duplication' in item.description for item in debt_items):
            patterns.append({
                'pattern': 'Extract Common Code',
                'description': 'Create shared utilities for duplicated code',
                'when_to_use': 'Code duplication across multiple modules',
                'effort': 'Medium',
                'impact': 'High'
            })
        
        return patterns
    
    def create_refactoring_checklist(self) -> List[str]:
        """Create refactoring safety checklist"""
        
        return [
            "✅ Comprehensive test coverage for refactoring target",
            "✅ Backup and version control checkpoint created",
            "✅ Performance benchmarks established",
            "✅ Refactoring plan reviewed and approved",
            "✅ Rollback strategy defined and tested",
            "✅ Stakeholder communication completed",
            "✅ CI/CD pipeline validation updated",
            "✅ Code review process planned",
            "✅ Documentation updates scheduled",
            "✅ Team capacity and timeline confirmed"
        ]

# Refactoring execution template
refactoring_template = '''
# Refactoring Execution Plan

## Target: {target_component}
**Debt Category:** {debt_category}
**Priority:** {priority}
**Estimated Effort:** {effort_hours} hours

## Current State Analysis
- **Issues Identified:** {issues}
- **Performance Impact:** {performance_impact}
- **Maintainability Score:** {maintainability_score}/10

## Refactoring Approach
**Pattern:** {refactoring_pattern}
**Strategy:** {strategy}

### Step-by-Step Plan:
1. **Preparation (Day 1)**
   - [ ] Create feature branch: `refactor/{component_name}`
   - [ ] Establish test coverage baseline (target: >80%)
   - [ ] Document current behavior and APIs
   - [ ] Set up performance monitoring

2. **Refactoring (Day 2-{duration})**
   - [ ] Implement refactoring pattern
   - [ ] Maintain existing API compatibility
   - [ ] Update tests incrementally
   - [ ] Validate performance benchmarks

3. **Validation (Day {duration+1})**
   - [ ] Run complete test suite
   - [ ] Performance regression testing
   - [ ] Code review with team
   - [ ] Update documentation

4. **Deployment (Day {duration+2})**
   - [ ] Staged deployment to development
   - [ ] QA validation in staging environment
   - [ ] Production deployment with monitoring
   - [ ] Post-deployment validation

## Success Criteria
- [ ] All existing tests pass
- [ ] Performance metrics maintained or improved
- [ ] Code complexity reduced by {complexity_reduction}%
- [ ] No new bugs introduced
- [ ] Team velocity maintained

## Rollback Plan
**Triggers:** Performance degradation >10%, critical bugs, failed tests
**Process:** 
1. Immediate revert to previous commit
2. Validate system stability
3. Post-mortem analysis
4. Refined refactoring approach
'''
```

## 4. Continuous Debt Monitoring

### Automated Debt Tracking
```python
# Continuous technical debt monitoring system
class TechnicalDebtMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = []
        self.alert_thresholds = {
            'debt_ratio': 0.3,          # Alert if debt ratio > 30%
            'complexity_increase': 0.15, # Alert if complexity increases > 15%
            'test_coverage_drop': 0.05,  # Alert if coverage drops > 5%
            'duplication_increase': 0.1  # Alert if duplication increases > 10%
        }
    
    def setup_continuous_monitoring(self):
        """Set up continuous monitoring in CI/CD pipeline"""
        
        ci_script = '''
#!/bin/bash
# Technical Debt Monitoring Script

echo "🔍 Running technical debt analysis..."

# Run code quality analysis
npm run lint:report || echo "Linting issues detected"
npm run complexity:report || echo "Complexity analysis failed"
npm run duplicates:report || echo "Duplication analysis failed"

# Run test coverage
npm run test:coverage || echo "Test coverage analysis failed"

# Run security audit
npm audit --audit-level moderate || echo "Security vulnerabilities detected"

# Generate debt report
python scripts/generate_debt_report.py

# Check thresholds and alert if needed
python scripts/check_debt_thresholds.py

echo "✅ Technical debt analysis complete"
'''
        
        return ci_script
    
    def track_debt_trends(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Track debt trends over time"""
        
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': current_metrics
        })
        
        # Keep only last 30 data points
        if len(self.metrics_history) > 30:
            self.metrics_history.pop(0)
        
        # Calculate trends
        trends = {}
        if len(self.metrics_history) >= 2:
            previous = self.metrics_history[-2]['metrics']
            current = self.metrics_history[-1]['metrics']
            
            for metric, value in current.items():
                if metric in previous:
                    change = (value - previous[metric]) / previous[metric]
                    trends[metric] = {
                        'change_percentage': change * 100,
                        'direction': 'increasing' if change > 0 else 'decreasing',
                        'alert_needed': abs(change) > self.alert_thresholds.get(metric, 0.1)
                    }
        
        return trends
    
    def generate_debt_dashboard(self) -> str:
        """Generate HTML dashboard for debt visualization"""
        
        dashboard_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Technical Debt Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; }
        .alert { background-color: #ffebee; border-color: #f44336; }
        .good { background-color: #e8f5e8; border-color: #4caf50; }
        .warning { background-color: #fff3e0; border-color: #ff9800; }
    </style>
</head>
<body>
    <h1>📊 Technical Debt Dashboard</h1>
    
    <div class="metric-card {debt_ratio_class}">
        <h3>Debt Ratio</h3>
        <p><strong>{debt_ratio:.1%}</strong></p>
        <p>Target: < 20% | Threshold: < 30%</p>
    </div>
    
    <div class="metric-card {complexity_class}">
        <h3>Average Complexity</h3>
        <p><strong>{avg_complexity:.1f}</strong></p>
        <p>Target: < 10 | Threshold: < 15</p>
    </div>
    
    <div class="metric-card {coverage_class}">
        <h3>Test Coverage</h3>
        <p><strong>{test_coverage:.1%}</strong></p>
        <p>Target: > 80% | Threshold: > 70%</p>
    </div>
    
    <div class="metric-card {duplication_class}">
        <h3>Code Duplication</h3>
        <p><strong>{duplication:.1%}</strong></p>
        <p>Target: < 5% | Threshold: < 10%</p>
    </div>
    
    <h2>📈 Trends (Last 30 Days)</h2>
    <canvas id="trendsChart" width="800" height="400"></canvas>
    
    <h2>🎯 Action Items</h2>
    <ul>
        {action_items}
    </ul>
    
    <script>
        // Trends chart implementation
        const ctx = document.getElementById('trendsChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: {trend_labels},
                datasets: [
                    {
                        label: 'Debt Ratio',
                        data: {debt_ratio_data},
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    },
                    {
                        label: 'Test Coverage',
                        data: {coverage_data},
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    </script>
</body>
</html>
'''
        
        return dashboard_html

# Integration with development workflow
def integrate_debt_monitoring():
    """Integration points for debt monitoring"""
    
    integration_points = {
        'pre_commit_hook': '''
#!/bin/sh
# Pre-commit hook for technical debt prevention

echo "🔍 Checking for potential technical debt..."

# Check for large functions
python scripts/check_function_size.py || exit 1

# Check for high complexity
python scripts/check_complexity.py || exit 1

# Check for code duplication
python scripts/check_duplication.py || exit 1

echo "✅ Pre-commit debt checks passed"
''',
        
        'pull_request_check': '''
name: Technical Debt Check
on: [pull_request]

jobs:
  debt-analysis:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Run debt analysis
      run: |
        npm run debt:analyze
        npm run debt:report
    - name: Comment PR
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('debt-report.md', 'utf8');
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: report
          });
''',
        
        'weekly_report': '''
# Weekly Technical Debt Report

## Summary
- **Total Debt Items:** {total_items}
- **New Debt This Week:** {new_debt}
- **Debt Resolved:** {resolved_debt}
- **Debt Ratio Trend:** {trend_direction}

## Top Priority Items
{priority_items}

## Recommendations
{recommendations}

## Next Week's Focus
{next_week_focus}
'''
    }
    
    return integration_points
```

## Conclusion

This technical debt assessment framework provides:

**Key Features:**
- Automated debt discovery and measurement for {{technology_stack}}
- Comprehensive categorization and prioritization system
- Strategic refactoring planning and execution guidance
- Continuous monitoring and trend analysis
- Integration with development workflows

**Business Benefits:**
- {{#if (eq debt_focus "performance")}}Improved system performance and user experience{{/if}}
- {{#if (eq debt_focus "maintainability")}}Reduced development time and increased code quality{{/if}}
- {{#if (eq debt_focus "security")}}Enhanced security posture and reduced risk{{/if}}
- {{#if (eq debt_focus "scalability")}}Better system scalability and architecture{{/if}}

**Implementation Strategy:**
- {{timeline}} assessment cycle with {{business_pressure}} priority handling
- {{team_size}} team optimized resource allocation
- Phased approach balancing quick wins and strategic improvements
- Continuous monitoring and feedback loops

**Success Metrics:**
- Debt ratio reduction from current to <20%
- Improved maintainability index score
- Reduced average complexity scores
- Increased test coverage to >80%
- Faster feature delivery velocity