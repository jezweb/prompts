---
name: authentication_system_design
title: Authentication System Design
description: Comprehensive authentication and authorization system design with modern security patterns, multi-factor authentication, and scalable architecture
category: development
tags: [authentication, authorization, security, jwt, oauth, mfa]
difficulty: advanced
author: jezweb
version: 1.0.0
arguments:
  - name: application_type
    description: Type of application (web-app, mobile-app, api, microservices, spa)
    required: true
  - name: user_base_size
    description: Expected number of users (small <1K, medium 1K-100K, large >100K)
    required: true
  - name: authentication_methods
    description: Required authentication methods (password, social, sso, biometric, mfa)
    required: true
  - name: compliance_requirements
    description: Compliance requirements (GDPR, HIPAA, SOX, PCI-DSS, none)
    required: false
    default: "none"
  - name: existing_systems
    description: Existing systems to integrate with
    required: false
    default: "None"
  - name: security_level
    description: Required security level (basic, standard, high, enterprise)
    required: true
---

# Authentication System Design: {{application_type}}

**User Base:** {{user_base_size}} users  
**Authentication Methods:** {{authentication_methods}}  
**Security Level:** {{security_level}}  
**Compliance:** {{compliance_requirements}}  
**Integration:** {{existing_systems}}

## 1. Architecture Overview

### System Components
```mermaid
graph TB
    Client[{{application_type}}]
    Gateway[API Gateway]
    AuthService[Authentication Service]
    UserService[User Management Service]
    TokenService[Token Management]
    MFA[Multi-Factor Auth]
    
    Client --> Gateway
    Gateway --> AuthService
    AuthService --> UserService
    AuthService --> TokenService
    AuthService --> MFA
    
    DB[(User Database)]
    Cache[(Redis Cache)]
    Audit[(Audit Logs)]
    
    UserService --> DB
    TokenService --> Cache
    AuthService --> Audit
```

### Security Architecture Layers
```yaml
security_layers:
  presentation:
    - input_validation
    - csrf_protection
    - secure_headers
    - rate_limiting
  
  application:
    - authentication_logic
    - authorization_rules
    - session_management
    - password_policies
  
  data:
    - encryption_at_rest
    - secure_transmission
    - data_masking
    - access_logging
  
  infrastructure:
    - network_security
    - container_security
    - secrets_management
    - monitoring_alerting
```

## 2. Authentication Flow Design

{{#if (includes authentication_methods "password")}}
### Password-Based Authentication
```javascript
// Secure password authentication implementation
const bcrypt = require('bcrypt');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');

class PasswordAuthenticator {
    constructor() {
        this.saltRounds = 12;
        this.maxFailedAttempts = 5;
        this.lockoutDuration = 15 * 60 * 1000; // 15 minutes
    }
    
    async hashPassword(password) {
        // Password strength validation
        if (!this.validatePasswordStrength(password)) {
            throw new Error('Password does not meet security requirements');
        }
        
        return await bcrypt.hash(password, this.saltRounds);
    }
    
    async verifyPassword(password, hashedPassword, userId) {
        // Check for account lockout
        if (await this.isAccountLocked(userId)) {
            throw new Error('Account temporarily locked due to failed attempts');
        }
        
        const isValid = await bcrypt.compare(password, hashedPassword);
        
        if (!isValid) {
            await this.recordFailedAttempt(userId);
            throw new Error('Invalid credentials');
        }
        
        await this.clearFailedAttempts(userId);
        return true;
    }
    
    validatePasswordStrength(password) {
        const requirements = {
            minLength: 12,
            requireUppercase: true,
            requireLowercase: true,
            requireNumbers: true,
            requireSpecialChars: true,
            preventCommonPasswords: true
        };
        
        // Length check
        if (password.length < requirements.minLength) return false;
        
        // Character composition checks
        if (requirements.requireUppercase && !/[A-Z]/.test(password)) return false;
        if (requirements.requireLowercase && !/[a-z]/.test(password)) return false;
        if (requirements.requireNumbers && !/\d/.test(password)) return false;
        if (requirements.requireSpecialChars && !/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) return false;
        
        // Common password check (integrate with HaveIBeenPwned API)
        if (requirements.preventCommonPasswords) {
            return !this.isCommonPassword(password);
        }
        
        return true;
    }
    
    async isCommonPassword(password) {
        // Integration with HaveIBeenPwned API
        const crypto = require('crypto');
        const https = require('https');
        
        const hash = crypto.createHash('sha1').update(password).digest('hex').toUpperCase();
        const prefix = hash.substring(0, 5);
        const suffix = hash.substring(5);
        
        return new Promise((resolve) => {
            https.get(`https://api.pwnedpasswords.com/range/${prefix}`, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    const found = data.includes(suffix);
                    resolve(found);
                });
            }).on('error', () => resolve(false));
        });
    }
}

// Rate limiting for authentication endpoints
const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: {{#if (eq security_level "high")}}3{{else}}5{{/if}}, // limit each IP to X requests per windowMs
    message: 'Too many authentication attempts, please try again later',
    standardHeaders: true,
    legacyHeaders: false
});
```
{{/if}}

{{#if (includes authentication_methods "jwt")}}
### JWT Token Management
```javascript
// JWT implementation with secure practices
const jwt = require('jsonwebtoken');
const crypto = require('crypto');

class JWTManager {
    constructor() {
        this.accessTokenExpiry = '15m';
        this.refreshTokenExpiry = '7d';
        this.issuer = 'your-app-name';
        this.audience = '{{application_type}}';
    }
    
    generateKeyPair() {
        // Use RS256 for better security
        return crypto.generateKeyPairSync('rsa', {
            modulusLength: 2048,
            publicKeyEncoding: { type: 'spki', format: 'pem' },
            privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
        });
    }
    
    generateTokens(user) {
        const payload = {
            sub: user.id,
            email: user.email,
            roles: user.roles,
            permissions: user.permissions,
            iss: this.issuer,
            aud: this.audience
        };
        
        const accessToken = jwt.sign(
            payload,
            process.env.JWT_PRIVATE_KEY,
            {
                algorithm: 'RS256',
                expiresIn: this.accessTokenExpiry,
                jwtid: crypto.randomUUID()
            }
        );
        
        const refreshToken = jwt.sign(
            { sub: user.id, type: 'refresh' },
            process.env.JWT_REFRESH_SECRET,
            {
                algorithm: 'HS256',
                expiresIn: this.refreshTokenExpiry,
                jwtid: crypto.randomUUID()
            }
        );
        
        return { accessToken, refreshToken };
    }
    
    verifyToken(token, type = 'access') {
        try {
            const secret = type === 'access' 
                ? process.env.JWT_PUBLIC_KEY 
                : process.env.JWT_REFRESH_SECRET;
            
            const algorithm = type === 'access' ? 'RS256' : 'HS256';
            
            return jwt.verify(token, secret, {
                algorithms: [algorithm],
                issuer: this.issuer,
                audience: this.audience
            });
        } catch (error) {
            throw new Error(`Invalid ${type} token: ${error.message}`);
        }
    }
    
    async revokeToken(jti) {
        // Store revoked tokens in Redis with TTL
        const redis = require('redis').createClient();
        await redis.setex(`revoked_token:${jti}`, 60 * 60 * 24 * 7, 'true');
    }
    
    async isTokenRevoked(jti) {
        const redis = require('redis').createClient();
        return await redis.exists(`revoked_token:${jti}`);
    }
}

// Middleware for token validation
const authenticateToken = async (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    
    if (!token) {
        return res.status(401).json({ error: 'Access token required' });
    }
    
    try {
        const jwtManager = new JWTManager();
        const decoded = jwtManager.verifyToken(token);
        
        // Check if token is revoked
        if (await jwtManager.isTokenRevoked(decoded.jti)) {
            return res.status(401).json({ error: 'Token has been revoked' });
        }
        
        req.user = decoded;
        next();
    } catch (error) {
        return res.status(403).json({ error: 'Invalid or expired token' });
    }
};
```
{{/if}}

{{#if (includes authentication_methods "oauth")}}
### OAuth 2.0 / OpenID Connect Integration
```javascript
// OAuth 2.0 implementation with PKCE
const crypto = require('crypto');
const axios = require('axios');

class OAuthProvider {
    constructor(provider, config) {
        this.provider = provider;
        this.clientId = config.clientId;
        this.clientSecret = config.clientSecret;
        this.redirectUri = config.redirectUri;
        this.scopes = config.scopes || ['openid', 'profile', 'email'];
        
        this.endpoints = {
            google: {
                authorization: 'https://accounts.google.com/o/oauth2/v2/auth',
                token: 'https://oauth2.googleapis.com/token',
                userinfo: 'https://www.googleapis.com/oauth2/v2/userinfo'
            },
            microsoft: {
                authorization: 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
                token: 'https://login.microsoftonline.com/common/oauth2/v2.0/token',
                userinfo: 'https://graph.microsoft.com/v1.0/me'
            },
            github: {
                authorization: 'https://github.com/login/oauth/authorize',
                token: 'https://github.com/login/oauth/access_token',
                userinfo: 'https://api.github.com/user'
            }
        };
    }
    
    generatePKCE() {
        const codeVerifier = crypto.randomBytes(32).toString('base64url');
        const codeChallenge = crypto
            .createHash('sha256')
            .update(codeVerifier)
            .digest('base64url');
        
        return { codeVerifier, codeChallenge };
    }
    
    getAuthorizationUrl(state) {
        const { codeChallenge } = this.generatePKCE();
        
        const params = new URLSearchParams({
            client_id: this.clientId,
            redirect_uri: this.redirectUri,
            response_type: 'code',
            scope: this.scopes.join(' '),
            state: state,
            code_challenge: codeChallenge,
            code_challenge_method: 'S256'
        });
        
        return `${this.endpoints[this.provider].authorization}?${params}`;
    }
    
    async exchangeCodeForTokens(code, codeVerifier) {
        const tokenData = {
            client_id: this.clientId,
            client_secret: this.clientSecret,
            code: code,
            grant_type: 'authorization_code',
            redirect_uri: this.redirectUri,
            code_verifier: codeVerifier
        };
        
        try {
            const response = await axios.post(
                this.endpoints[this.provider].token,
                tokenData,
                {
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'Accept': 'application/json'
                    }
                }
            );
            
            return response.data;
        } catch (error) {
            throw new Error(`Token exchange failed: ${error.response?.data?.error_description || error.message}`);
        }
    }
    
    async getUserInfo(accessToken) {
        try {
            const response = await axios.get(
                this.endpoints[this.provider].userinfo,
                {
                    headers: {
                        'Authorization': `Bearer ${accessToken}`,
                        'Accept': 'application/json'
                    }
                }
            );
            
            return this.normalizeUserInfo(response.data);
        } catch (error) {
            throw new Error(`Failed to fetch user info: ${error.message}`);
        }
    }
    
    normalizeUserInfo(rawUserInfo) {
        // Normalize user info across different providers
        const normalizedInfo = {
            id: rawUserInfo.id || rawUserInfo.sub,
            email: rawUserInfo.email,
            name: rawUserInfo.name || rawUserInfo.display_name,
            picture: rawUserInfo.picture || rawUserInfo.avatar_url,
            provider: this.provider
        };
        
        return normalizedInfo;
    }
}
```
{{/if}}

{{#if (includes authentication_methods "mfa")}}
### Multi-Factor Authentication (MFA)
```javascript
// TOTP-based MFA implementation
const speakeasy = require('speakeasy');
const QRCode = require('qrcode');
const crypto = require('crypto');

class MFAManager {
    constructor() {
        this.serviceName = 'YourAppName';
        this.issuer = 'your-company.com';
        this.windowSize = 2; // Allow 2 time windows for clock drift
    }
    
    async generateMFASecret(user) {
        const secret = speakeasy.generateSecret({
            name: `${this.serviceName} (${user.email})`,
            issuer: this.issuer,
            length: 32
        });
        
        // Generate QR code for easy setup
        const qrCodeUrl = await QRCode.toDataURL(secret.otpauth_url);
        
        return {
            secret: secret.base32,
            qrCode: qrCodeUrl,
            backupCodes: this.generateBackupCodes()
        };
    }
    
    generateBackupCodes(count = 10) {
        const codes = [];
        for (let i = 0; i < count; i++) {
            const code = crypto.randomBytes(4).toString('hex').toUpperCase();
            codes.push(code.match(/.{2}/g).join('-'));
        }
        return codes;
    }
    
    verifyTOTP(token, secret) {
        return speakeasy.totp.verify({
            secret: secret,
            encoding: 'base32',
            token: token,
            window: this.windowSize,
            time: Math.floor(Date.now() / 1000)
        });
    }
    
    async verifyBackupCode(code, userId) {
        // Check against stored backup codes
        const user = await this.getUserBackupCodes(userId);
        const isValid = user.backupCodes.includes(code);
        
        if (isValid) {
            // Remove used backup code
            await this.removeBackupCode(userId, code);
        }
        
        return isValid;
    }
    
    async sendSMSCode(phoneNumber) {
        const code = crypto.randomInt(100000, 999999).toString();
        const expiresAt = new Date(Date.now() + 5 * 60 * 1000); // 5 minutes
        
        // Store code temporarily
        await this.storeSMSCode(phoneNumber, code, expiresAt);
        
        // Send SMS (integrate with Twilio, AWS SNS, etc.)
        await this.sendSMS(phoneNumber, `Your verification code is: ${code}`);
        
        return { success: true, expiresAt };
    }
    
    async verifySMSCode(phoneNumber, code) {
        const storedCode = await this.getSMSCode(phoneNumber);
        
        if (!storedCode || storedCode.expires < new Date()) {
            throw new Error('Code expired or invalid');
        }
        
        if (storedCode.code !== code) {
            throw new Error('Invalid verification code');
        }
        
        // Remove used code
        await this.removeSMSCode(phoneNumber);
        return true;
    }
}

// MFA middleware
const requireMFA = async (req, res, next) => {
    const user = req.user;
    
    if (!user.mfaEnabled) {
        return next(); // MFA not required for this user
    }
    
    const mfaToken = req.headers['x-mfa-token'];
    if (!mfaToken) {
        return res.status(401).json({ 
            error: 'MFA token required',
            mfaRequired: true 
        });
    }
    
    try {
        const mfaManager = new MFAManager();
        const isValid = mfaManager.verifyTOTP(mfaToken, user.mfaSecret);
        
        if (!isValid) {
            return res.status(401).json({ error: 'Invalid MFA token' });
        }
        
        next();
    } catch (error) {
        return res.status(401).json({ error: 'MFA verification failed' });
    }
};
```
{{/if}}

## 3. Authorization & Role-Based Access Control (RBAC)

### Permission System Design
```javascript
// RBAC implementation
class AuthorizationManager {
    constructor() {
        this.permissions = new Map();
        this.roles = new Map();
        this.userRoles = new Map();
    }
    
    // Define permissions
    definePermission(name, description, resource, action) {
        this.permissions.set(name, {
            name,
            description,
            resource,
            action,
            createdAt: new Date()
        });
    }
    
    // Define roles with permissions
    defineRole(name, permissions, description) {
        this.roles.set(name, {
            name,
            permissions: new Set(permissions),
            description,
            createdAt: new Date()
        });
    }
    
    // Assign roles to users
    assignRole(userId, roleName) {
        if (!this.userRoles.has(userId)) {
            this.userRoles.set(userId, new Set());
        }
        this.userRoles.get(userId).add(roleName);
    }
    
    // Check if user has permission
    hasPermission(userId, permissionName) {
        const userRoles = this.userRoles.get(userId);
        if (!userRoles) return false;
        
        for (const roleName of userRoles) {
            const role = this.roles.get(roleName);
            if (role && role.permissions.has(permissionName)) {
                return true;
            }
        }
        
        return false;
    }
    
    // Advanced permission checking with context
    hasPermissionWithContext(userId, resource, action, context = {}) {
        const userRoles = this.userRoles.get(userId);
        if (!userRoles) return false;
        
        // Check direct permissions
        for (const roleName of userRoles) {
            const role = this.roles.get(roleName);
            if (!role) continue;
            
            for (const permissionName of role.permissions) {
                const permission = this.permissions.get(permissionName);
                if (permission && 
                    permission.resource === resource && 
                    permission.action === action) {
                    
                    // Apply context-based rules
                    return this.evaluateContext(permission, context, userId);
                }
            }
        }
        
        return false;
    }
    
    evaluateContext(permission, context, userId) {
        // Example: Resource ownership check
        if (context.ownerId && context.ownerId !== userId) {
            return permission.allowOwnershipOverride === true;
        }
        
        // Example: Time-based access
        if (permission.timeRestrictions) {
            const now = new Date();
            const { startTime, endTime } = permission.timeRestrictions;
            if (now < startTime || now > endTime) {
                return false;
            }
        }
        
        return true;
    }
}

// Authorization middleware
const authorize = (resource, action) => {
    return async (req, res, next) => {
        const userId = req.user.sub;
        const authManager = new AuthorizationManager();
        
        const hasPermission = authManager.hasPermissionWithContext(
            userId, 
            resource, 
            action, 
            {
                ownerId: req.params.userId,
                resourceId: req.params.id,
                timestamp: new Date()
            }
        );
        
        if (!hasPermission) {
            return res.status(403).json({ 
                error: 'Insufficient permissions',
                required: { resource, action }
            });
        }
        
        next();
    };
};

// Usage example
// app.get('/api/users/:id', authenticateToken, authorize('user', 'read'), getUserController);
```

## 4. Session Management

{{#if (eq application_type "web-app")}}
### Secure Session Configuration
```javascript
// Express session configuration
const session = require('express-session');
const RedisStore = require('connect-redis')(session);
const redis = require('redis');

const sessionConfig = {
    store: new RedisStore({ 
        client: redis.createClient({
            host: process.env.REDIS_HOST,
            port: process.env.REDIS_PORT,
            password: process.env.REDIS_PASSWORD
        })
    }),
    secret: process.env.SESSION_SECRET,
    name: 'sessionId', // Don't use default session name
    resave: false,
    saveUninitialized: false,
    rolling: true, // Reset expiration on activity
    cookie: {
        secure: process.env.NODE_ENV === 'production', // HTTPS only in production
        httpOnly: true, // Prevent XSS access
        maxAge: {{#if (eq security_level "high")}}900000{{else}}1800000{{/if}}, // 15 or 30 minutes
        sameSite: 'strict' // CSRF protection
    }
};

// Session security middleware
const sessionSecurity = (req, res, next) => {
    // Regenerate session ID on login
    if (req.body.action === 'login' && req.session.userId) {
        req.session.regenerate((err) => {
            if (err) {
                return next(err);
            }
            next();
        });
    } else {
        next();
    }
};
```
{{/if}}

## 5. Security Hardening

### Input Validation & Sanitization
```javascript
// Comprehensive input validation
const Joi = require('joi');
const DOMPurify = require('isomorphic-dompurify');
const validator = require('validator');

class InputValidator {
    static schemas = {
        login: Joi.object({
            email: Joi.string().email().required().max(254),
            password: Joi.string().min(8).max(128).required(),
            rememberMe: Joi.boolean().default(false)
        }),
        
        registration: Joi.object({
            email: Joi.string().email().required().max(254),
            password: Joi.string()
                .pattern(new RegExp('^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[!@#$%^&*])'))
                .min(12)
                .max(128)
                .required(),
            firstName: Joi.string().alphanum().min(1).max(50).required(),
            lastName: Joi.string().alphanum().min(1).max(50).required(),
            phoneNumber: Joi.string().pattern(/^\\+?[1-9]\\d{1,14}$/).optional()
        })
    };
    
    static validate(data, schemaName) {
        const schema = this.schemas[schemaName];
        if (!schema) {
            throw new Error(`Unknown validation schema: ${schemaName}`);
        }
        
        const { error, value } = schema.validate(data, {
            abortEarly: false,
            stripUnknown: true
        });
        
        if (error) {
            throw new Error(`Validation failed: ${error.details.map(d => d.message).join(', ')}`);
        }
        
        return value;
    }
    
    static sanitizeHtml(html) {
        return DOMPurify.sanitize(html, {
            ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a'],
            ALLOWED_ATTR: ['href']
        });
    }
    
    static sanitizeString(str) {
        return validator.escape(str.trim());
    }
}

// Validation middleware
const validateInput = (schemaName) => {
    return (req, res, next) => {
        try {
            req.body = InputValidator.validate(req.body, schemaName);
            next();
        } catch (error) {
            return res.status(400).json({ error: error.message });
        }
    };
};
```

### Security Headers
```javascript
// Security headers middleware
const helmet = require('helmet');

const securityHeaders = helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'", 'fonts.googleapis.com'],
            fontSrc: ["'self'", 'fonts.gstatic.com'],
            imgSrc: ["'self'", 'data:', 'https:'],
            scriptSrc: ["'self'"],
            connectSrc: ["'self'", 'api.yourapp.com'],
            frameSrc: ["'none'"],
            objectSrc: ["'none'"],
            upgradeInsecureRequests: []
        }
    },
    hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
    },
    noSniff: true,
    frameguard: { action: 'deny' },
    referrerPolicy: { policy: 'strict-origin-when-cross-origin' }
});
```

## 6. Database Schema

### User Management Tables
```sql
-- User accounts table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(254) UNIQUE NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    phone_number VARCHAR(20),
    phone_verified BOOLEAN DEFAULT FALSE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'deleted')),
    last_login_at TIMESTAMP WITH TIME ZONE,
    password_changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- MFA settings table
CREATE TABLE user_mfa (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    enabled BOOLEAN DEFAULT FALSE,
    secret VARCHAR(255),
    backup_codes TEXT[], -- Encrypted backup codes
    phone_number VARCHAR(20),
    preferred_method VARCHAR(20) DEFAULT 'totp' CHECK (preferred_method IN ('totp', 'sms', 'email')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- OAuth provider links
CREATE TABLE user_oauth_providers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    provider_id VARCHAR(255) NOT NULL,
    email VARCHAR(254),
    name VARCHAR(255),
    avatar_url TEXT,
    access_token TEXT, -- Encrypted
    refresh_token TEXT, -- Encrypted
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, provider_id)
);

-- Roles and permissions
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    resource VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE role_permissions (
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    permission_id UUID REFERENCES permissions(id) ON DELETE CASCADE,
    PRIMARY KEY (role_id, permission_id)
);

CREATE TABLE user_roles (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    assigned_by UUID REFERENCES users(id),
    PRIMARY KEY (user_id, role_id)
);

-- Audit logging
CREATE TABLE auth_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_auth_audit_user_id ON auth_audit_log(user_id);
CREATE INDEX idx_auth_audit_event_type ON auth_audit_log(event_type);
CREATE INDEX idx_auth_audit_created_at ON auth_audit_log(created_at);
```

## 7. API Endpoints

### Authentication API
```javascript
// Authentication routes
const express = require('express');
const router = express.Router();

// User registration
router.post('/register', 
    validateInput('registration'),
    async (req, res) => {
        try {
            const { email, password, firstName, lastName, phoneNumber } = req.body;
            
            // Check if user already exists
            const existingUser = await User.findByEmail(email);
            if (existingUser) {
                return res.status(409).json({ error: 'User already exists' });
            }
            
            // Create new user
            const passwordHash = await new PasswordAuthenticator().hashPassword(password);
            const user = await User.create({
                email,
                passwordHash,
                firstName,
                lastName,
                phoneNumber
            });
            
            // Generate email verification token
            const verificationToken = await EmailVerification.generateToken(user.id);
            await EmailService.sendVerificationEmail(user.email, verificationToken);
            
            // Log the event
            await AuditLogger.log('user_registration', user.id, req.ip, req.headers['user-agent']);
            
            res.status(201).json({
                message: 'User created successfully',
                userId: user.id,
                emailVerificationRequired: true
            });
            
        } catch (error) {
            res.status(400).json({ error: error.message });
        }
    }
);

// User login
router.post('/login',
    authLimiter,
    validateInput('login'),
    async (req, res) => {
        try {
            const { email, password, rememberMe } = req.body;
            
            const user = await User.findByEmail(email);
            if (!user) {
                throw new Error('Invalid credentials');
            }
            
            // Verify password
            const passwordAuth = new PasswordAuthenticator();
            await passwordAuth.verifyPassword(password, user.passwordHash, user.id);
            
            // Check if MFA is enabled
            if (user.mfaEnabled) {
                // Return temporary token for MFA completion
                const mfaToken = jwt.sign(
                    { userId: user.id, step: 'mfa_required' },
                    process.env.MFA_SECRET,
                    { expiresIn: '5m' }
                );
                
                return res.json({
                    mfaRequired: true,
                    mfaToken: mfaToken,
                    availableMethods: ['totp', 'sms']
                });
            }
            
            // Generate JWT tokens
            const jwtManager = new JWTManager();
            const { accessToken, refreshToken } = jwtManager.generateTokens(user);
            
            // Update last login
            await user.updateLastLogin();
            
            // Set refresh token as HTTP-only cookie
            res.cookie('refreshToken', refreshToken, {
                httpOnly: true,
                secure: process.env.NODE_ENV === 'production',
                sameSite: 'strict',
                maxAge: rememberMe ? 7 * 24 * 60 * 60 * 1000 : 24 * 60 * 60 * 1000
            });
            
            // Log successful login
            await AuditLogger.log('user_login', user.id, req.ip, req.headers['user-agent'], true);
            
            res.json({
                accessToken,
                user: {
                    id: user.id,
                    email: user.email,
                    name: `${user.firstName} ${user.lastName}`,
                    roles: user.roles
                }
            });
            
        } catch (error) {
            // Log failed login attempt
            await AuditLogger.log('failed_login', null, req.ip, req.headers['user-agent'], false, {
                email: req.body.email
            });
            
            res.status(401).json({ error: 'Invalid credentials' });
        }
    }
);

// MFA verification
router.post('/verify-mfa',
    validateInput('mfaVerification'),
    async (req, res) => {
        try {
            const { mfaToken, code, method } = req.body;
            
            // Verify MFA token
            const decoded = jwt.verify(mfaToken, process.env.MFA_SECRET);
            const user = await User.findById(decoded.userId);
            
            const mfaManager = new MFAManager();
            let isValid = false;
            
            if (method === 'totp') {
                isValid = mfaManager.verifyTOTP(code, user.mfaSecret);
            } else if (method === 'sms') {
                isValid = await mfaManager.verifySMSCode(user.phoneNumber, code);
            } else if (method === 'backup') {
                isValid = await mfaManager.verifyBackupCode(code, user.id);
            }
            
            if (!isValid) {
                return res.status(401).json({ error: 'Invalid MFA code' });
            }
            
            // Generate JWT tokens
            const jwtManager = new JWTManager();
            const { accessToken, refreshToken } = jwtManager.generateTokens(user);
            
            res.cookie('refreshToken', refreshToken, {
                httpOnly: true,
                secure: process.env.NODE_ENV === 'production',
                sameSite: 'strict',
                maxAge: 24 * 60 * 60 * 1000
            });
            
            res.json({
                accessToken,
                user: {
                    id: user.id,
                    email: user.email,
                    name: `${user.firstName} ${user.lastName}`,
                    roles: user.roles
                }
            });
            
        } catch (error) {
            res.status(401).json({ error: 'MFA verification failed' });
        }
    }
);

// Token refresh
router.post('/refresh-token', async (req, res) => {
    try {
        const refreshToken = req.cookies.refreshToken;
        if (!refreshToken) {
            return res.status(401).json({ error: 'Refresh token required' });
        }
        
        const jwtManager = new JWTManager();
        const decoded = jwtManager.verifyToken(refreshToken, 'refresh');
        
        const user = await User.findById(decoded.sub);
        if (!user) {
            return res.status(401).json({ error: 'User not found' });
        }
        
        const { accessToken, refreshToken: newRefreshToken } = jwtManager.generateTokens(user);
        
        res.cookie('refreshToken', newRefreshToken, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            sameSite: 'strict',
            maxAge: 24 * 60 * 60 * 1000
        });
        
        res.json({ accessToken });
        
    } catch (error) {
        res.status(401).json({ error: 'Invalid refresh token' });
    }
});

// Logout
router.post('/logout', authenticateToken, async (req, res) => {
    try {
        // Revoke access token
        const jwtManager = new JWTManager();
        await jwtManager.revokeToken(req.user.jti);
        
        // Clear refresh token cookie
        res.clearCookie('refreshToken');
        
        // Log logout event
        await AuditLogger.log('user_logout', req.user.sub, req.ip, req.headers['user-agent']);
        
        res.json({ message: 'Logged out successfully' });
        
    } catch (error) {
        res.status(500).json({ error: 'Logout failed' });
    }
});

module.exports = router;
```

## 8. Monitoring & Audit Logging

### Audit Logging System
```javascript
// Comprehensive audit logging
class AuditLogger {
    static async log(eventType, userId, ipAddress, userAgent, success = true, additionalData = {}) {
        const auditEntry = {
            eventType,
            userId,
            ipAddress,
            userAgent,
            success,
            eventData: additionalData,
            timestamp: new Date()
        };
        
        // Store in database
        await AuditLog.create(auditEntry);
        
        // Send to monitoring system (e.g., ELK stack)
        await this.sendToMonitoring(auditEntry);
        
        // Check for suspicious activity
        await this.checkForAnomalies(eventType, userId, ipAddress);
    }
    
    static async sendToMonitoring(auditEntry) {
        // Integration with logging service
        const winston = require('winston');
        const logger = winston.createLogger({
            level: 'info',
            format: winston.format.json(),
            transports: [
                new winston.transports.File({ filename: 'audit.log' })
            ]
        });
        
        logger.info('auth_event', auditEntry);
    }
    
    static async checkForAnomalies(eventType, userId, ipAddress) {
        if (eventType === 'failed_login') {
            const recentFailures = await AuditLog.countRecentFailures(ipAddress, '15 minutes');
            
            if (recentFailures >= 10) {
                await SecurityAlert.create({
                    type: 'multiple_failed_logins',
                    ipAddress,
                    count: recentFailures,
                    severity: 'high'
                });
            }
        }
        
        if (eventType === 'user_login' && userId) {
            const user = await User.findById(userId);
            const previousLogin = user.lastLoginAt;
            
            if (previousLogin) {
                const locationChanged = await this.checkLocationChange(user.id, ipAddress);
                const timeAnomaly = await this.checkTimeAnomaly(previousLogin);
                
                if (locationChanged || timeAnomaly) {
                    await SecurityAlert.create({
                        type: 'suspicious_login',
                        userId,
                        ipAddress,
                        details: { locationChanged, timeAnomaly },
                        severity: 'medium'
                    });
                }
            }
        }
    }
}
```

## 9. {{compliance_requirements}} Compliance

{{#if (eq compliance_requirements "GDPR")}}
### GDPR Compliance Features
```javascript
// GDPR compliance implementation
class GDPRCompliance {
    // Right to be forgotten
    async deleteUserData(userId, requestId) {
        const user = await User.findById(userId);
        if (!user) throw new Error('User not found');
        
        // Create deletion audit trail
        await AuditLogger.log('gdpr_deletion_request', userId, null, null, true, {
            requestId,
            dataTypes: ['profile', 'activities', 'preferences']
        });
        
        // Anonymize user data instead of hard delete
        await User.update(userId, {
            email: `deleted-${userId}@anonymized.local`,
            firstName: 'Deleted',
            lastName: 'User',
            phoneNumber: null,
            status: 'deleted'
        });
        
        // Remove personally identifiable information
        await this.cleanupRelatedData(userId);
        
        return { success: true, requestId };
    }
    
    // Data portability
    async exportUserData(userId) {
        const user = await User.findById(userId);
        const userData = {
            profile: {
                email: user.email,
                firstName: user.firstName,
                lastName: user.lastName,
                createdAt: user.createdAt
            },
            activities: await this.getUserActivities(userId),
            preferences: await this.getUserPreferences(userId)
        };
        
        // Log data export
        await AuditLogger.log('gdpr_data_export', userId, null, null, true);
        
        return userData;
    }
    
    // Consent management
    async recordConsent(userId, consentType, granted) {
        await UserConsent.create({
            userId,
            consentType,
            granted,
            timestamp: new Date(),
            ipAddress: req.ip
        });
    }
}
```
{{/if}}

## 10. Deployment & Configuration

### Environment Configuration
```yaml
# docker-compose.yml for authentication service
version: '3.8'
services:
  auth-service:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - JWT_PRIVATE_KEY_FILE=/run/secrets/jwt_private_key
      - JWT_PUBLIC_KEY_FILE=/run/secrets/jwt_public_key
      - JWT_REFRESH_SECRET_FILE=/run/secrets/jwt_refresh_secret
      - SESSION_SECRET_FILE=/run/secrets/session_secret
      - DATABASE_URL_FILE=/run/secrets/database_url
      - REDIS_URL_FILE=/run/secrets/redis_url
    secrets:
      - jwt_private_key
      - jwt_public_key
      - jwt_refresh_secret
      - session_secret
      - database_url
      - redis_url
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=authdb
      - POSTGRES_USER=authuser
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
    secrets:
      - postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    environment:
      - REDIS_PASSWORD_FILE=/run/secrets/redis_password
    secrets:
      - redis_password
    volumes:
      - redis_data:/data
    restart: unless-stopped

secrets:
  jwt_private_key:
    external: true
  jwt_public_key:
    external: true
  jwt_refresh_secret:
    external: true
  session_secret:
    external: true
  database_url:
    external: true
  redis_url:
    external: true
  postgres_password:
    external: true
  redis_password:
    external: true

volumes:
  postgres_data:
  redis_data:
```

### Security Configuration Checklist
- [ ] Generate secure JWT key pairs (RS256)
- [ ] Configure proper CORS policies
- [ ] Set up rate limiting
- [ ] Enable security headers
- [ ] Configure SSL/TLS certificates
- [ ] Set up monitoring and alerting
- [ ] Configure backup and recovery
- [ ] Test disaster recovery procedures
- [ ] Document security procedures
- [ ] Schedule regular security audits

## Conclusion

This authentication system provides enterprise-grade security with:

**Key Features:**
- Multi-factor authentication support
- JWT-based stateless authentication
- Role-based access control (RBAC)
- OAuth 2.0 / OpenID Connect integration
- Comprehensive audit logging
- {{compliance_requirements}} compliance features
- Advanced security hardening

**Security Benefits:**
- Protection against common attacks (OWASP Top 10)
- Secure session management
- Input validation and sanitization
- Rate limiting and brute force protection
- Comprehensive monitoring and alerting

**Scalability:**
- Designed for {{user_base_size}} user base
- Horizontal scaling support
- Caching layer integration
- Microservices architecture ready