---
name: security_audit_framework
title: Security Audit Framework
description: Comprehensive security audit framework for applications and infrastructure with systematic vulnerability assessment and remediation strategies
category: development
tags: [security, audit, vulnerability, penetration-testing, compliance]
difficulty: advanced
author: jezweb
version: 1.0.0
arguments:
  - name: system_type
    description: Type of system to audit (web-app, api, mobile-app, infrastructure, database)
    required: true
  - name: compliance_framework
    description: Compliance framework to follow (OWASP, SOC2, ISO27001, PCI-DSS, GDPR)
    required: true
  - name: environment
    description: Environment being audited (production, staging, development)
    required: true
  - name: previous_audit_date
    description: Date of previous security audit
    required: false
    default: "Never audited"
  - name: critical_assets
    description: Critical assets and data types to protect
    required: true
  - name: threat_model
    description: Known threat actors or attack vectors
    required: false
    default: "General web threats"
---

# Security Audit Framework: {{system_type}}

**Compliance Framework:** {{compliance_framework}}  
**Environment:** {{environment}}  
**Last Audit:** {{previous_audit_date}}  
**Critical Assets:** {{critical_assets}}  
**Threat Model:** {{threat_model}}

## 1. Pre-Audit Planning & Scope Definition

### Audit Objectives
- [ ] Define security posture assessment goals
- [ ] Identify compliance requirements
- [ ] Document critical assets and data flows
- [ ] Establish risk tolerance levels
- [ ] Set audit timeline and resources

### Scope Boundaries
```yaml
# Audit Scope Configuration
audit_scope:
  systems:
    - {{system_type}}
  environments:
    - {{environment}}
  critical_assets:
    - {{critical_assets}}
  
  inclusions:
    - Application source code
    - Infrastructure components
    - Access controls and permissions
    - Data encryption and storage
    - Network security controls
  
  exclusions:
    - Third-party services (document separately)
    - Legacy systems scheduled for decommission
    - Systems with active security patches pending
```

## 2. {{compliance_framework}} Security Assessment

{{#if (eq compliance_framework "OWASP")}}
### OWASP Top 10 Assessment

#### A01: Broken Access Control
```bash
# Access Control Testing
# Test for horizontal privilege escalation
curl -H "Authorization: Bearer user1_token" \
     -X GET "https://api.example.com/users/user2_profile"

# Test for vertical privilege escalation
curl -H "Authorization: Bearer user_token" \
     -X GET "https://api.example.com/admin/users"

# Directory traversal testing
curl "https://example.com/files?file=../../../etc/passwd"
```

**Assessment Checklist:**
- [ ] Role-based access controls implemented
- [ ] User permissions properly scoped
- [ ] Administrative functions protected
- [ ] Object-level authorization enforced
- [ ] Directory traversal prevention

#### A02: Cryptographic Failures
```javascript
// Encryption Implementation Review
const crypto = require('crypto');

// Check for proper encryption standards
const algorithm = 'aes-256-gcm'; // Good
const algorithm_bad = 'aes-128-ecb'; // Bad - avoid

// Key management review
const key = crypto.randomBytes(32); // Good - random key
const key_bad = 'hardcoded-key-123'; // Bad - static key

// Password hashing review
const bcrypt = require('bcrypt');
const saltRounds = 12; // Good - sufficient rounds
const password_hash = await bcrypt.hash(password, saltRounds);
```

**Assessment Areas:**
- [ ] Data encryption at rest and in transit
- [ ] Proper key management practices
- [ ] Strong password hashing algorithms
- [ ] Certificate management and validity
- [ ] Cryptographic algorithm strength

#### A03: Injection Vulnerabilities
```sql
-- SQL Injection Testing
-- Test for SQL injection vulnerabilities
' OR '1'='1
'; DROP TABLE users; --
' UNION SELECT username, password FROM admin_users --

-- Parameterized query implementation (Good)
SELECT * FROM users WHERE id = ?

-- String concatenation (Bad)
SELECT * FROM users WHERE id = '" + user_input + "'
```

```javascript
// NoSQL Injection Testing
// MongoDB injection example
{ "username": { "$ne": null }, "password": { "$ne": null } }

// Command injection testing
; ls -la
| cat /etc/passwd
`whoami`
```

**Injection Prevention Checklist:**
- [ ] Parameterized queries used consistently
- [ ] Input validation and sanitization
- [ ] Output encoding implemented
- [ ] Command injection prevention
- [ ] Template injection protection
{{/if}}

{{#if (eq compliance_framework "SOC2")}}
### SOC 2 Security Controls Assessment

#### Security Principle Evaluation
```yaml
# SOC 2 Type II Controls Matrix
security_controls:
  access_controls:
    - user_authentication: implemented
    - role_based_access: implemented
    - privileged_access_management: review_required
    - access_reviews: quarterly
  
  system_operations:
    - change_management: documented
    - vulnerability_management: automated
    - incident_response: tested
    - monitoring_logging: centralized
  
  configuration_management:
    - secure_configurations: hardened
    - configuration_monitoring: continuous
    - backup_procedures: tested
    - recovery_procedures: documented
```

**Control Testing:**
- [ ] Logical access controls testing
- [ ] System operations review
- [ ] Change management processes
- [ ] Risk management framework
- [ ] Monitoring and incident response
{{/if}}

## 3. System-Specific Security Assessment

{{#if (eq system_type "web-app")}}
### Web Application Security Testing

#### Authentication & Session Management
```python
# Session security testing script
import requests
import time

def test_session_security(base_url):
    """Test session management security"""
    
    # Test session fixation
    session = requests.Session()
    
    # Get initial session
    resp = session.get(f"{base_url}/login")
    initial_session = session.cookies.get('SESSIONID')
    
    # Login
    session.post(f"{base_url}/login", {
        'username': 'testuser',
        'password': 'testpass'
    })
    
    # Check if session ID changed after login
    post_login_session = session.cookies.get('SESSIONID')
    
    if initial_session == post_login_session:
        print("⚠️  Session Fixation Vulnerability Detected")
    else:
        print("✅ Session regeneration working correctly")
    
    return session

# Test session timeout
def test_session_timeout(session, base_url):
    """Test session timeout mechanisms"""
    
    # Wait for configured timeout period
    time.sleep(1800)  # 30 minutes
    
    resp = session.get(f"{base_url}/protected")
    
    if resp.status_code == 401:
        print("✅ Session timeout working correctly")
    else:
        print("⚠️  Session timeout not implemented")
```

#### Input Validation Testing
```python
# Input validation test cases
test_payloads = {
    'xss': [
        '<script>alert("XSS")</script>',
        'javascript:alert("XSS")',
        '<img src=x onerror=alert("XSS")>',
        '"><script>alert("XSS")</script>'
    ],
    'sql_injection': [
        "' OR '1'='1",
        "'; DROP TABLE users; --",
        "' UNION SELECT password FROM admin --"
    ],
    'command_injection': [
        '; ls -la',
        '| cat /etc/passwd',
        '`whoami`',
        '$(id)'
    ],
    'path_traversal': [
        '../../../etc/passwd',
        '..\\..\\..\\windows\\system32\\drivers\\etc\\hosts',
        '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd'
    ]
}

def test_input_validation(base_url, endpoints):
    """Test input validation across application endpoints"""
    
    for endpoint in endpoints:
        for attack_type, payloads in test_payloads.items():
            for payload in payloads:
                # Test GET parameters
                resp = requests.get(f"{base_url}{endpoint}?param={payload}")
                analyze_response(resp, attack_type, payload)
                
                # Test POST data
                resp = requests.post(f"{base_url}{endpoint}", {'param': payload})
                analyze_response(resp, attack_type, payload)

def analyze_response(response, attack_type, payload):
    """Analyze response for potential vulnerabilities"""
    
    if attack_type == 'xss' and payload in response.text:
        print(f"⚠️  Potential XSS vulnerability: {payload}")
    
    if attack_type == 'sql_injection' and any(error in response.text.lower() for error in ['sql', 'mysql', 'postgresql', 'sqlite']):
        print(f"⚠️  Potential SQL Injection: {payload}")
    
    if response.status_code == 500:
        print(f"⚠️  Server error with payload: {payload}")
```
{{/if}}

{{#if (eq system_type "api")}}
### API Security Assessment

#### API Authentication Testing
```bash
# JWT Token Security Testing
JWT_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Test JWT manipulation
echo $JWT_TOKEN | base64 -d | jq .

# Test for algorithm confusion attack
# Modify alg header to "none"
MODIFIED_JWT=$(echo '{"alg":"none","typ":"JWT"}' | base64 -w 0)

# Test weak signing keys
jwt-cracker $JWT_TOKEN

# API rate limiting testing
for i in {1..1000}; do
  curl -H "Authorization: Bearer $JWT_TOKEN" \
       -X GET "https://api.example.com/users" &
done
wait
```

#### API Endpoint Security
```python
# API security testing framework
import requests
import json
from concurrent.futures import ThreadPoolExecutor

class APISecurityTester:
    def __init__(self, base_url, auth_token):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {auth_token}'}
    
    def test_http_methods(self, endpoint):
        """Test all HTTP methods on endpoint"""
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        
        for method in methods:
            try:
                resp = requests.request(method, f"{self.base_url}{endpoint}", 
                                      headers=self.headers)
                
                if method in ['PUT', 'DELETE'] and resp.status_code == 200:
                    print(f"⚠️  {method} method allowed on {endpoint}")
                
                if 'Allow' in resp.headers:
                    allowed_methods = resp.headers['Allow'].split(', ')
                    print(f"Allowed methods for {endpoint}: {allowed_methods}")
                    
            except Exception as e:
                print(f"Error testing {method} on {endpoint}: {e}")
    
    def test_authorization_bypass(self, endpoint, user_id):
        """Test for horizontal authorization bypass"""
        
        # Test accessing other user's data
        for test_id in [user_id + 1, user_id - 1, 9999, 'admin']:
            resp = requests.get(f"{self.base_url}{endpoint}/{test_id}", 
                              headers=self.headers)
            
            if resp.status_code == 200:
                print(f"⚠️  Possible authorization bypass: accessing user {test_id}")
    
    def test_rate_limiting(self, endpoint):
        """Test API rate limiting"""
        
        def make_request():
            return requests.get(f"{self.base_url}{endpoint}", headers=self.headers)
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            
            responses = [future.result() for future in futures]
            status_codes = [resp.status_code for resp in responses]
            
            if 429 not in status_codes:
                print("⚠️  No rate limiting detected")
            else:
                print("✅ Rate limiting is implemented")
```
{{/if}}

## 4. Infrastructure Security Assessment

### Network Security
```bash
# Network security assessment scripts
#!/bin/bash

# Port scanning
nmap -sS -O -sV {{environment}}.example.com

# SSL/TLS configuration testing
sslscan {{environment}}.example.com:443
testssl.sh {{environment}}.example.com

# DNS security testing
dig {{environment}}.example.com
nslookup {{environment}}.example.com

# Certificate transparency logs
curl -s "https://crt.sh/?q={{environment}}.example.com&output=json" | jq .
```

### Server Hardening Assessment
```yaml
# Server security checklist
server_hardening:
  operating_system:
    - os_updates: check_patch_level
    - unused_services: disable_unnecessary
    - user_accounts: review_and_clean
    - sudo_access: limit_and_audit
    
  network_services:
    - ssh_configuration: harden_settings
    - firewall_rules: review_and_optimize
    - open_ports: minimize_exposure
    - network_monitoring: implement_logging
    
  file_system:
    - file_permissions: audit_and_correct
    - sensitive_files: encrypt_and_protect
    - log_files: secure_and_rotate
    - backup_security: encrypt_and_test
```

## 5. Vulnerability Assessment & Penetration Testing

### Automated Vulnerability Scanning
```python
# Automated vulnerability assessment
import subprocess
import json
from datetime import datetime

class VulnerabilityScanner:
    def __init__(self, target):
        self.target = target
        self.results = {}
    
    def run_nessus_scan(self):
        """Run Nessus vulnerability scan"""
        # Nessus API integration
        scan_config = {
            "uuid": "731a8e52-3ea6-a291-ec0a-d2ff0619c19d7bd788d6",
            "settings": {
                "name": f"Security Audit {datetime.now()}",
                "text_targets": self.target
            }
        }
        
        # Implementation would integrate with Nessus API
        return scan_config
    
    def run_owasp_zap_scan(self):
        """Run OWASP ZAP automated scan"""
        zap_command = [
            "zap-baseline.py",
            "-t", self.target,
            "-J", f"zap-report-{datetime.now().strftime('%Y%m%d')}.json"
        ]
        
        try:
            result = subprocess.run(zap_command, capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            print(f"ZAP scan failed: {e}")
    
    def run_nikto_scan(self):
        """Run Nikto web server scanner"""
        nikto_command = [
            "nikto",
            "-h", self.target,
            "-Format", "json",
            "-output", f"nikto-{datetime.now().strftime('%Y%m%d')}.json"
        ]
        
        subprocess.run(nikto_command)

# Usage
scanner = VulnerabilityScanner("{{environment}}.example.com")
scanner.run_owasp_zap_scan()
scanner.run_nikto_scan()
```

### Manual Testing Procedures
```bash
# Manual penetration testing checklist

# 1. Information gathering
whois example.com
dig example.com ANY
nslookup example.com
theHarvester -d example.com -b google,bing,linkedin

# 2. Network enumeration
nmap -sn 192.168.1.0/24  # Network discovery
nmap -sS -sV -A target.com  # Service detection

# 3. Web application testing
dirb http://target.com  # Directory enumeration
gobuster dir -u http://target.com -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt

# 4. Database testing
sqlmap -u "http://target.com/vulnerable.php?id=1" --dbs
```

## 6. Risk Assessment & Prioritization

### Risk Scoring Matrix
```python
# Risk assessment calculator
class RiskAssessment:
    LIKELIHOOD_SCORES = {
        'very_low': 1,
        'low': 2,
        'medium': 3,
        'high': 4,
        'very_high': 5
    }
    
    IMPACT_SCORES = {
        'minimal': 1,
        'minor': 2,
        'moderate': 3,
        'major': 4,
        'severe': 5
    }
    
    def calculate_risk_score(self, likelihood, impact):
        """Calculate CVSS-like risk score"""
        likelihood_score = self.LIKELIHOOD_SCORES.get(likelihood, 3)
        impact_score = self.IMPACT_SCORES.get(impact, 3)
        
        risk_score = likelihood_score * impact_score
        
        if risk_score <= 5:
            return 'Low'
        elif risk_score <= 10:
            return 'Medium'
        elif risk_score <= 15:
            return 'High'
        else:
            return 'Critical'
    
    def prioritize_vulnerabilities(self, vulnerabilities):
        """Prioritize vulnerabilities by risk score"""
        prioritized = []
        
        for vuln in vulnerabilities:
            risk_level = self.calculate_risk_score(
                vuln['likelihood'], 
                vuln['impact']
            )
            vuln['risk_level'] = risk_level
            prioritized.append(vuln)
        
        return sorted(prioritized, 
                     key=lambda x: ['Low', 'Medium', 'High', 'Critical'].index(x['risk_level']),
                     reverse=True)
```

## 7. Remediation Planning

### Critical Vulnerabilities (Fix Immediately)
- [ ] SQL injection vulnerabilities
- [ ] Remote code execution flaws
- [ ] Authentication bypass issues
- [ ] Sensitive data exposure
- [ ] Broken access controls

### High Priority (Fix within 1 week)
- [ ] Cross-site scripting (XSS)
- [ ] Cross-site request forgery (CSRF)
- [ ] Insecure direct object references
- [ ] Security misconfigurations
- [ ] Unvalidated redirects

### Medium Priority (Fix within 1 month)
- [ ] Information disclosure
- [ ] Weak encryption implementations
- [ ] Session management issues
- [ ] Insufficient logging/monitoring
- [ ] Known vulnerable components

### Remediation Templates
```yaml
# Vulnerability remediation template
vulnerability_fix:
  id: "{{vulnerability_id}}"
  title: "{{vulnerability_title}}"
  severity: "{{severity_level}}"
  
  description: |
    Detailed description of the vulnerability
    
  impact: |
    Potential business and technical impact
    
  remediation_steps:
    - step: "Immediate containment actions"
      timeline: "Within 24 hours"
      owner: "Security Team"
    
    - step: "Code fixes and patches"
      timeline: "Within 1 week"
      owner: "Development Team"
    
    - step: "Testing and validation"
      timeline: "Within 2 weeks"
      owner: "QA Team"
    
    - step: "Deployment and monitoring"
      timeline: "Within 3 weeks"
      owner: "DevOps Team"
  
  verification:
    - "Vulnerability scan confirms fix"
    - "Penetration test validation"
    - "Code review approval"
```

## 8. Compliance Reporting

### Audit Report Structure
```markdown
# Security Audit Report - {{system_type}}

## Executive Summary
- Overall security posture: [Strong/Adequate/Weak]
- Total vulnerabilities found: [Number]
- Critical issues: [Number]
- Compliance status: [Compliant/Non-compliant with {{compliance_framework}}]

## Methodology
- Audit scope and limitations
- Tools and techniques used
- Testing timeline and resources

## Findings Summary
| Severity | Count | Status |
|----------|-------|--------|
| Critical | 0     | ✅     |
| High     | 2     | ⚠️     |
| Medium   | 5     | ⚠️     |
| Low      | 12    | ℹ️     |

## Detailed Findings
[Individual vulnerability descriptions]

## Recommendations
[Prioritized remediation plan]

## Conclusion
[Overall assessment and next steps]
```

## 9. Continuous Security Monitoring

### Security Metrics Dashboard
```python
# Security metrics collection
security_metrics = {
    'vulnerability_trends': {
        'critical_vulns_open': 0,
        'high_vulns_open': 2,
        'medium_vulns_open': 5,
        'mean_time_to_remediation': '7 days'
    },
    
    'security_events': {
        'failed_login_attempts': 45,
        'suspicious_activities': 3,
        'blocked_attacks': 127,
        'security_alerts': 12
    },
    
    'compliance_status': {
        'controls_implemented': 95,
        'controls_total': 100,
        'compliance_percentage': 95,
        'next_audit_date': '2024-12-01'
    }
}
```

### Automated Security Testing
```yaml
# CI/CD security integration
security_pipeline:
  static_analysis:
    - tool: SonarQube
      threshold: "No critical issues"
    - tool: Semgrep
      rules: "owasp-top-10"
  
  dynamic_testing:
    - tool: OWASP ZAP
      scan_type: "baseline"
    - tool: Burp Suite
      scan_type: "comprehensive"
  
  dependency_scanning:
    - tool: npm audit
      action: "fail on high"
    - tool: Snyk
      monitor: true
  
  compliance_checks:
    - framework: "{{compliance_framework}}"
      automated: true
      manual_review: required
```

## 10. Next Steps & Recommendations

### Immediate Actions (0-30 days)
1. **Critical Vulnerability Remediation**
   - Address all critical and high-severity findings
   - Implement emergency patches where needed
   - Enhance monitoring for affected systems

2. **Security Control Enhancement**
   - Strengthen authentication mechanisms
   - Implement proper access controls
   - Enable comprehensive logging

### Short-term Improvements (1-3 months)
1. **Security Process Maturation**
   - Establish regular vulnerability assessments
   - Implement security training programs
   - Create incident response procedures

2. **Compliance Alignment**
   - Address {{compliance_framework}} gaps
   - Document security procedures
   - Conduct regular compliance reviews

### Long-term Strategic Goals (3-12 months)
1. **Security Program Development**
   - Implement DevSecOps practices
   - Establish security metrics and KPIs
   - Create security awareness culture

2. **Advanced Security Capabilities**
   - Deploy security orchestration tools
   - Implement threat intelligence feeds
   - Establish red team exercises

## Conclusion

This security audit framework provides a comprehensive approach to assessing and improving the security posture of {{system_type}} systems. Regular application of this framework will help maintain strong security controls and ensure compliance with {{compliance_framework}} requirements.

**Key Success Metrics:**
- Zero critical vulnerabilities in production
- Mean time to remediation < 7 days
- 100% compliance with {{compliance_framework}}
- Regular security training completion
- Continuous security monitoring implementation