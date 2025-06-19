---
name: api_documentation
title: API Documentation Generator
description: Generate comprehensive API documentation with examples, schemas, and integration guides
category: documentation
tags: [api, documentation, openapi, swagger, rest, technical-writing]
difficulty: intermediate
author: jezweb
version: 1.0.0
arguments:
  - name: api_name
    description: Name of the API
    required: true
  - name: base_url
    description: Base URL of the API
    required: true
  - name: version
    description: API version
    required: false
    default: "v1"
  - name: auth_type
    description: Authentication type (bearer, api-key, oauth2, basic)
    required: false
    default: "bearer"
  - name: format
    description: Documentation format (markdown, openapi, postman)
    required: false
    default: "markdown"
---

# {{api_name}} API Documentation

**Version:** {{version}}  
**Base URL:** `{{base_url}}`  
**Authentication:** {{auth_type}}

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL & Versioning](#base-url--versioning)
4. [Request/Response Format](#requestresponse-format)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Endpoints](#endpoints)
8. [Examples](#examples)
9. [SDKs & Tools](#sdks--tools)
10. [Changelog](#changelog)

---

## Overview

{{api_name}} API provides programmatic access to [describe main functionality]. This RESTful API allows developers to [list key capabilities].

### Key Features
- 🚀 High-performance endpoints
- 🔒 Secure authentication
- 📊 Comprehensive data access
- 🔄 Real-time updates
- 📱 Mobile-friendly responses

### Use Cases
1. **Integration**: Connect {{api_name}} with your applications
2. **Automation**: Automate workflows and processes
3. **Analytics**: Extract data for analysis
4. **Custom Apps**: Build custom applications

---

## Authentication

{{#if (eq auth_type "bearer")}}
### Bearer Token Authentication

All API requests require a Bearer token in the Authorization header:

```http
Authorization: Bearer YOUR_API_TOKEN
```

#### Obtaining a Token

```bash
curl -X POST {{base_url}}/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_here"
}
```
{{/if}}

{{#if (eq auth_type "api-key")}}
### API Key Authentication

Include your API key in the request header:

```http
X-API-Key: YOUR_API_KEY
```

#### Getting an API Key

1. Log in to your {{api_name}} dashboard
2. Navigate to Settings > API Keys
3. Click "Generate New Key"
4. Store your key securely (it won't be shown again)
{{/if}}

{{#if (eq auth_type "oauth2")}}
### OAuth 2.0 Authentication

{{api_name}} uses OAuth 2.0 for authentication. Supported grant types:
- Authorization Code
- Client Credentials
- Refresh Token

#### Authorization URL
```
{{base_url}}/oauth/authorize
```

#### Token URL
```
{{base_url}}/oauth/token
```

#### Example Authorization Code Flow

1. **Redirect user to authorization URL:**
```
{{base_url}}/oauth/authorize?
  response_type=code&
  client_id=YOUR_CLIENT_ID&
  redirect_uri=YOUR_REDIRECT_URI&
  scope=read write&
  state=RANDOM_STATE
```

2. **Exchange code for token:**
```bash
curl -X POST {{base_url}}/oauth/token \
  -d "grant_type=authorization_code" \
  -d "code=AUTHORIZATION_CODE" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "redirect_uri=YOUR_REDIRECT_URI"
```
{{/if}}

### Security Best Practices
- 🔐 Never expose your credentials in client-side code
- 🔄 Rotate keys/tokens regularly
- 🚫 Use HTTPS for all requests
- 📝 Implement proper error handling

---

## Base URL & Versioning

### Base URL
All API endpoints are relative to:
```
{{base_url}}/{{version}}
```

### Versioning Strategy
- Version is included in the URL path
- Breaking changes result in new versions
- Older versions supported for 12 months
- Deprecation notices provided 6 months in advance

### Environments
| Environment | Base URL |
|------------|----------|
| Production | `{{base_url}}/{{version}}` |
| Staging | `{{base_url}}-staging/{{version}}` |
| Sandbox | `{{base_url}}-sandbox/{{version}}` |

---

## Request/Response Format

### Request Headers
```http
Content-Type: application/json
Accept: application/json
{{#if (eq auth_type "bearer")}}Authorization: Bearer YOUR_TOKEN{{/if}}
{{#if (eq auth_type "api-key")}}X-API-Key: YOUR_API_KEY{{/if}}
```

### Request Body
All POST, PUT, and PATCH requests should send JSON:
```json
{
  "field1": "value1",
  "field2": "value2"
}
```

### Response Format
All responses return JSON with this structure:

**Success Response:**
```json
{
  "success": true,
  "data": {
    // Response data here
  },
  "meta": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ]
  },
  "meta": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

### Pagination
List endpoints support pagination:

```http
GET {{base_url}}/{{version}}/resources?page=2&limit=20
```

**Pagination Response:**
```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 20,
    "total": 100,
    "pages": 5,
    "has_next": true,
    "has_prev": true
  }
}
```

---

## Error Handling

### HTTP Status Codes

| Status Code | Description |
|------------|-------------|
| 200 | Success |
| 201 | Created |
| 204 | No Content |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict |
| 422 | Unprocessable Entity |
| 429 | Too Many Requests |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `INVALID_REQUEST` | Request format is invalid | Check request syntax |
| `AUTHENTICATION_FAILED` | Invalid credentials | Verify API key/token |
| `PERMISSION_DENIED` | Insufficient permissions | Check account permissions |
| `RESOURCE_NOT_FOUND` | Requested resource doesn't exist | Verify resource ID |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Implement backoff strategy |
| `VALIDATION_ERROR` | Input validation failed | Check error details |

### Error Response Example
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The request contains invalid data",
    "details": [
      {
        "field": "email",
        "message": "Email format is invalid",
        "value": "not-an-email"
      },
      {
        "field": "age",
        "message": "Age must be between 18 and 120",
        "value": 150
      }
    ],
    "documentation_url": "{{base_url}}/docs/errors#validation"
  }
}
```

---

## Rate Limiting

API requests are rate limited to ensure fair usage:

| Plan | Requests/Hour | Requests/Day | Burst Rate |
|------|---------------|--------------|------------|
| Free | 100 | 1,000 | 10/min |
| Basic | 1,000 | 10,000 | 100/min |
| Pro | 10,000 | 100,000 | 1,000/min |
| Enterprise | Custom | Custom | Custom |

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1515500400
X-RateLimit-Reset-After: 3600
```

### Handling Rate Limits
```javascript
async function makeAPICall(endpoint, options = {}) {
  const response = await fetch(endpoint, options);
  
  if (response.status === 429) {
    const resetAfter = response.headers.get('X-RateLimit-Reset-After');
    const waitTime = parseInt(resetAfter) * 1000;
    
    console.log(`Rate limited. Waiting ${waitTime}ms...`);
    await new Promise(resolve => setTimeout(resolve, waitTime));
    
    // Retry the request
    return makeAPICall(endpoint, options);
  }
  
  return response;
}
```

---

## Endpoints

### Resources Endpoint

#### List Resources
```http
GET {{base_url}}/{{version}}/resources
```

**Query Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| page | integer | Page number | 1 |
| limit | integer | Items per page | 20 |
| sort | string | Sort field | created_at |
| order | string | Sort order (asc/desc) | desc |
| filter | object | Filter criteria | - |

**Example Request:**
```bash
curl -X GET "{{base_url}}/{{version}}/resources?page=1&limit=10&sort=name&order=asc" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Example Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "res_123",
      "name": "Resource Name",
      "description": "Resource description",
      "created_at": "2024-01-15T10:00:00Z",
      "updated_at": "2024-01-15T12:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 50,
    "pages": 5
  }
}
```

#### Get Single Resource
```http
GET {{base_url}}/{{version}}/resources/{id}
```

#### Create Resource
```http
POST {{base_url}}/{{version}}/resources
```

**Request Body:**
```json
{
  "name": "New Resource",
  "description": "Description of the resource",
  "metadata": {
    "key": "value"
  }
}
```

#### Update Resource
```http
PUT {{base_url}}/{{version}}/resources/{id}
```

#### Delete Resource
```http
DELETE {{base_url}}/{{version}}/resources/{id}
```

---

## Examples

### cURL Examples

{{#if (eq auth_type "bearer")}}
**Get with Bearer Token:**
```bash
curl -X GET {{base_url}}/{{version}}/resources \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Accept: application/json"
```
{{/if}}

{{#if (eq auth_type "api-key")}}
**Get with API Key:**
```bash
curl -X GET {{base_url}}/{{version}}/resources \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Accept: application/json"
```
{{/if}}

**POST Request:**
```bash
curl -X POST {{base_url}}/{{version}}/resources \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "New Resource",
    "description": "Created via API"
  }'
```

### JavaScript/Node.js Example
```javascript
const axios = require('axios');

const api = axios.create({
  baseURL: '{{base_url}}/{{version}}',
  headers: {
    {{#if (eq auth_type "bearer")}}'Authorization': 'Bearer ' + process.env.API_TOKEN,{{/if}}
    {{#if (eq auth_type "api-key")}}'X-API-Key': process.env.API_KEY,{{/if}}
    'Content-Type': 'application/json'
  }
});

// Get resources
async function getResources() {
  try {
    const response = await api.get('/resources');
    return response.data;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

// Create resource
async function createResource(data) {
  try {
    const response = await api.post('/resources', data);
    return response.data;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}
```

### Python Example
```python
import requests
import os

base_url = "{{base_url}}/{{version}}"
headers = {
    {{#if (eq auth_type "bearer")}}"Authorization": f"Bearer {os.getenv('API_TOKEN')}",{{/if}}
    {{#if (eq auth_type "api-key")}}"X-API-Key": os.getenv('API_KEY'),{{/if}}
    "Content-Type": "application/json"
}

# Get resources
def get_resources():
    response = requests.get(f"{base_url}/resources", headers=headers)
    response.raise_for_status()
    return response.json()

# Create resource
def create_resource(data):
    response = requests.post(f"{base_url}/resources", json=data, headers=headers)
    response.raise_for_status()
    return response.json()

# Usage
try:
    resources = get_resources()
    print(resources)
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
```

---

## SDKs & Tools

### Official SDKs
- [JavaScript/TypeScript SDK](https://github.com/{{api_name}}/js-sdk)
- [Python SDK](https://github.com/{{api_name}}/python-sdk)
- [Go SDK](https://github.com/{{api_name}}/go-sdk)
- [Ruby SDK](https://github.com/{{api_name}}/ruby-sdk)

### Community Tools
- [Postman Collection](https://www.postman.com/{{api_name}}/workspace)
- [OpenAPI Specification]({{base_url}}/openapi.json)
- [GraphQL Gateway](https://github.com/community/{{api_name}}-graphql)

### Code Generation
Generate client code from our OpenAPI spec:
```bash
# Using OpenAPI Generator
openapi-generator-cli generate \
  -i {{base_url}}/openapi.json \
  -g javascript \
  -o ./{{api_name}}-client
```

---

## Webhooks

Subscribe to real-time events:

### Available Events
- `resource.created`
- `resource.updated`
- `resource.deleted`
- `user.authenticated`

### Webhook Payload
```json
{
  "event": "resource.created",
  "timestamp": "2024-01-15T10:00:00Z",
  "data": {
    "id": "res_123",
    "type": "resource",
    "attributes": {}
  }
}
```

### Webhook Security
Verify webhook signatures:
```javascript
const crypto = require('crypto');

function verifyWebhookSignature(payload, signature, secret) {
  const hash = crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');
  
  return `sha256=${hash}` === signature;
}
```

---

## Changelog

### Version {{version}} (Current)
- Added webhook support
- Improved error messages
- New filtering options
- Performance optimizations

### Previous Versions
- **v2.0** - Breaking changes to authentication
- **v1.0** - Initial release

---

## Support

### Resources
- 📚 [Full Documentation]({{base_url}}/docs)
- 💬 [API Status](https://status.{{api_name}}.com)
- 🐛 [Report Issues](https://github.com/{{api_name}}/api/issues)
- 💡 [Feature Requests](https://feedback.{{api_name}}.com)

### Contact
- **Email**: api-support@{{api_name}}.com
- **Developer Forum**: https://forum.{{api_name}}.com
- **Discord**: https://discord.gg/{{api_name}}