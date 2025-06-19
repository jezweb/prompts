---
name: api_endpoint_generator
title: REST API Endpoint Generator
description: Generate RESTful API endpoint implementations with validation, error handling, and documentation
category: development
tags: [api, backend, rest, endpoint, nodejs, express, fastapi]
difficulty: intermediate
author: jezweb
version: 1.0.0
arguments:
  - name: framework
    description: Backend framework (express, fastapi, django, spring)
    required: true
  - name: resource
    description: Resource name (e.g., users, products, orders)
    required: true
  - name: operations
    description: CRUD operations to implement (comma-separated)
    required: false
    default: "create,read,update,delete"
  - name: auth_required
    description: Whether authentication is required (yes/no)
    required: false
    default: "yes"
  - name: validation
    description: Include input validation (yes/no)
    required: false
    default: "yes"
---

# REST API Endpoint: {{resource}}

## Framework: {{framework}}
**Operations:** {{operations}}
**Authentication:** {{#if (eq auth_required "yes")}}Required{{else}}Not required{{/if}}
**Validation:** {{#if (eq validation "yes")}}Enabled{{else}}Disabled{{/if}}

{{#if (eq framework "express")}}
## Express.js Implementation

```javascript
// {{resource}}.routes.js
const express = require('express');
const router = express.Router();
{{#if (eq validation "yes")}}
const { body, param, query, validationResult } = require('express-validator');
{{/if}}
{{#if (eq auth_required "yes")}}
const { authenticate } = require('../middleware/auth');
{{/if}}
const {{resource}}Controller = require('../controllers/{{resource}}.controller');

// Validation middleware
const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  next();
};

{{#if (includes operations "read")}}
// GET /api/{{resource}} - List all {{resource}}
router.get(
  '/',
  {{#if (eq auth_required "yes")}}authenticate,{{/if}}
  [
    query('page').optional().isInt({ min: 1 }),
    query('limit').optional().isInt({ min: 1, max: 100 }),
    query('sort').optional().isIn(['createdAt', 'updatedAt', 'name']),
    query('order').optional().isIn(['asc', 'desc'])
  ],
  handleValidationErrors,
  {{resource}}Controller.list
);

// GET /api/{{resource}}/:id - Get single {{resource}}
router.get(
  '/:id',
  {{#if (eq auth_required "yes")}}authenticate,{{/if}}
  [
    param('id').isMongoId().withMessage('Invalid ID format')
  ],
  handleValidationErrors,
  {{resource}}Controller.getById
);
{{/if}}

{{#if (includes operations "create")}}
// POST /api/{{resource}} - Create new {{resource}}
router.post(
  '/',
  {{#if (eq auth_required "yes")}}authenticate,{{/if}}
  [
    body('name').notEmpty().trim().isLength({ min: 2, max: 100 }),
    body('description').optional().trim().isLength({ max: 500 }),
    body('status').optional().isIn(['active', 'inactive', 'pending']),
    // Add more validation rules based on your schema
  ],
  handleValidationErrors,
  {{resource}}Controller.create
);
{{/if}}

{{#if (includes operations "update")}}
// PUT /api/{{resource}}/:id - Update entire {{resource}}
router.put(
  '/:id',
  {{#if (eq auth_required "yes")}}authenticate,{{/if}}
  [
    param('id').isMongoId(),
    body('name').notEmpty().trim().isLength({ min: 2, max: 100 }),
    body('description').optional().trim().isLength({ max: 500 }),
    body('status').optional().isIn(['active', 'inactive', 'pending'])
  ],
  handleValidationErrors,
  {{resource}}Controller.update
);

// PATCH /api/{{resource}}/:id - Partial update
router.patch(
  '/:id',
  {{#if (eq auth_required "yes")}}authenticate,{{/if}}
  [
    param('id').isMongoId(),
    body('name').optional().trim().isLength({ min: 2, max: 100 }),
    body('description').optional().trim().isLength({ max: 500 }),
    body('status').optional().isIn(['active', 'inactive', 'pending'])
  ],
  handleValidationErrors,
  {{resource}}Controller.partialUpdate
);
{{/if}}

{{#if (includes operations "delete")}}
// DELETE /api/{{resource}}/:id - Delete {{resource}}
router.delete(
  '/:id',
  {{#if (eq auth_required "yes")}}authenticate,{{/if}}
  [
    param('id').isMongoId()
  ],
  handleValidationErrors,
  {{resource}}Controller.delete
);
{{/if}}

module.exports = router;
```

```javascript
// {{resource}}.controller.js
const {{capitalize resource}}Service = require('../services/{{resource}}.service');
const { asyncHandler } = require('../utils/asyncHandler');

const {{resource}}Controller = {
  {{#if (includes operations "read")}}
  // List all {{resource}} with pagination
  list: asyncHandler(async (req, res) => {
    const {
      page = 1,
      limit = 20,
      sort = 'createdAt',
      order = 'desc'
    } = req.query;

    const options = {
      page: parseInt(page),
      limit: parseInt(limit),
      sort: { [sort]: order === 'asc' ? 1 : -1 }
    };

    const result = await {{capitalize resource}}Service.list(options);
    
    res.json({
      success: true,
      data: result.data,
      pagination: {
        page: result.page,
        limit: result.limit,
        total: result.total,
        pages: result.pages
      }
    });
  }),

  // Get single {{resource}} by ID
  getById: asyncHandler(async (req, res) => {
    const { id } = req.params;
    const item = await {{capitalize resource}}Service.getById(id);
    
    if (!item) {
      return res.status(404).json({
        success: false,
        error: '{{capitalize resource}} not found'
      });
    }

    res.json({
      success: true,
      data: item
    });
  }),
  {{/if}}

  {{#if (includes operations "create")}}
  // Create new {{resource}}
  create: asyncHandler(async (req, res) => {
    const data = req.body;
    {{#if (eq auth_required "yes")}}
    data.createdBy = req.user.id;
    {{/if}}

    const item = await {{capitalize resource}}Service.create(data);

    res.status(201).json({
      success: true,
      data: item,
      message: '{{capitalize resource}} created successfully'
    });
  }),
  {{/if}}

  {{#if (includes operations "update")}}
  // Update entire {{resource}}
  update: asyncHandler(async (req, res) => {
    const { id } = req.params;
    const data = req.body;
    {{#if (eq auth_required "yes")}}
    data.updatedBy = req.user.id;
    {{/if}}

    const item = await {{capitalize resource}}Service.update(id, data);

    if (!item) {
      return res.status(404).json({
        success: false,
        error: '{{capitalize resource}} not found'
      });
    }

    res.json({
      success: true,
      data: item,
      message: '{{capitalize resource}} updated successfully'
    });
  }),

  // Partial update
  partialUpdate: asyncHandler(async (req, res) => {
    const { id } = req.params;
    const updates = req.body;
    {{#if (eq auth_required "yes")}}
    updates.updatedBy = req.user.id;
    {{/if}}

    const item = await {{capitalize resource}}Service.partialUpdate(id, updates);

    if (!item) {
      return res.status(404).json({
        success: false,
        error: '{{capitalize resource}} not found'
      });
    }

    res.json({
      success: true,
      data: item,
      message: '{{capitalize resource}} updated successfully'
    });
  }),
  {{/if}}

  {{#if (includes operations "delete")}}
  // Delete {{resource}}
  delete: asyncHandler(async (req, res) => {
    const { id } = req.params;
    {{#if (eq auth_required "yes")}}
    const userId = req.user.id;
    {{/if}}

    const result = await {{capitalize resource}}Service.delete(id{{#if (eq auth_required "yes")}}, userId{{/if}});

    if (!result) {
      return res.status(404).json({
        success: false,
        error: '{{capitalize resource}} not found'
      });
    }

    res.status(204).send();
  })
  {{/if}}
};

module.exports = {{resource}}Controller;
```
{{/if}}

{{#if (eq framework "fastapi")}}
## FastAPI Implementation

```python
# {{resource}}_router.py
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import List, Optional
from datetime import datetime
{{#if (eq auth_required "yes")}}
from app.auth import get_current_user, User
{{/if}}
from app.models.{{resource}} import {{capitalize resource}}, {{capitalize resource}}Create, {{capitalize resource}}Update
from app.services.{{resource}}_service import {{capitalize resource}}Service

router = APIRouter(prefix="/api/{{resource}}", tags=["{{resource}}"])
service = {{capitalize resource}}Service()

{{#if (includes operations "read")}}
@router.get("/", response_model=List[{{capitalize resource}}])
async def list_{{resource}}(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort: str = Query("created_at", enum=["created_at", "updated_at", "name"]),
    order: str = Query("desc", enum=["asc", "desc"]),
    {{#if (eq auth_required "yes")}}current_user: User = Depends(get_current_user){{/if}}
):
    """List all {{resource}} with pagination."""
    return await service.list(page, limit, sort, order)

@router.get("/{id}", response_model={{capitalize resource}})
async def get_{{resource}}(
    id: str = Path(..., description="{{capitalize resource}} ID"),
    {{#if (eq auth_required "yes")}}current_user: User = Depends(get_current_user){{/if}}
):
    """Get a single {{resource}} by ID."""
    item = await service.get_by_id(id)
    if not item:
        raise HTTPException(status_code=404, detail="{{capitalize resource}} not found")
    return item
{{/if}}

{{#if (includes operations "create")}}
@router.post("/", response_model={{capitalize resource}}, status_code=201)
async def create_{{resource}}(
    data: {{capitalize resource}}Create,
    {{#if (eq auth_required "yes")}}current_user: User = Depends(get_current_user){{/if}}
):
    """Create a new {{resource}}."""
    {{#if (eq auth_required "yes")}}
    data.created_by = current_user.id
    {{/if}}
    return await service.create(data)
{{/if}}

{{#if (includes operations "update")}}
@router.put("/{id}", response_model={{capitalize resource}})
async def update_{{resource}}(
    id: str = Path(..., description="{{capitalize resource}} ID"),
    data: {{capitalize resource}}Update,
    {{#if (eq auth_required "yes")}}current_user: User = Depends(get_current_user){{/if}}
):
    """Update an entire {{resource}}."""
    {{#if (eq auth_required "yes")}}
    data.updated_by = current_user.id
    {{/if}}
    item = await service.update(id, data)
    if not item:
        raise HTTPException(status_code=404, detail="{{capitalize resource}} not found")
    return item

@router.patch("/{id}", response_model={{capitalize resource}})
async def partial_update_{{resource}}(
    id: str = Path(..., description="{{capitalize resource}} ID"),
    data: dict,
    {{#if (eq auth_required "yes")}}current_user: User = Depends(get_current_user){{/if}}
):
    """Partially update a {{resource}}."""
    {{#if (eq auth_required "yes")}}
    data["updated_by"] = current_user.id
    {{/if}}
    item = await service.partial_update(id, data)
    if not item:
        raise HTTPException(status_code=404, detail="{{capitalize resource}} not found")
    return item
{{/if}}

{{#if (includes operations "delete")}}
@router.delete("/{id}", status_code=204)
async def delete_{{resource}}(
    id: str = Path(..., description="{{capitalize resource}} ID"),
    {{#if (eq auth_required "yes")}}current_user: User = Depends(get_current_user){{/if}}
):
    """Delete a {{resource}}."""
    success = await service.delete(id{{#if (eq auth_required "yes")}}, current_user.id{{/if}})
    if not success:
        raise HTTPException(status_code=404, detail="{{capitalize resource}} not found")
{{/if}}
```

```python
# {{resource}}_models.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class {{capitalize resource}}Base(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    status: Optional[str] = Field("active", enum=["active", "inactive", "pending"])

class {{capitalize resource}}Create({{capitalize resource}}Base):
    pass

class {{capitalize resource}}Update({{capitalize resource}}Base):
    name: Optional[str] = Field(None, min_length=2, max_length=100)

class {{capitalize resource}}({{capitalize resource}}Base):
    id: str
    created_at: datetime
    updated_at: Optional[datetime]
    {{#if (eq auth_required "yes")}}
    created_by: str
    updated_by: Optional[str]
    {{/if}}

    class Config:
        orm_mode = True
```
{{/if}}

## API Documentation

### Endpoints Summary

{{#if (includes operations "read")}}
- `GET /api/{{resource}}` - List all {{resource}} (paginated)
- `GET /api/{{resource}}/:id` - Get specific {{resource}}
{{/if}}
{{#if (includes operations "create")}}
- `POST /api/{{resource}}` - Create new {{resource}}
{{/if}}
{{#if (includes operations "update")}}
- `PUT /api/{{resource}}/:id` - Update entire {{resource}}
- `PATCH /api/{{resource}}/:id` - Partial update
{{/if}}
{{#if (includes operations "delete")}}
- `DELETE /api/{{resource}}/:id` - Delete {{resource}}
{{/if}}

### Error Responses

```json
{
  "success": false,
  "error": "Error message",
  "errors": [
    {
      "field": "name",
      "message": "Name is required"
    }
  ]
}
```

### Success Response Example

```json
{
  "success": true,
  "data": {
    "id": "123",
    "name": "Example",
    "description": "Example description",
    "status": "active",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-02T00:00:00Z"
  }
}
```

## Next Steps

1. Implement the service layer
2. Add database models
3. Set up error handling middleware
4. Add request logging
5. Implement rate limiting
6. Add API documentation (Swagger/OpenAPI)
7. Write unit and integration tests