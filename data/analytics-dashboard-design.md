---
name: analytics_dashboard_design
title: Analytics Dashboard Design & Implementation
description: Comprehensive analytics dashboard framework with interactive visualizations, real-time metrics, and actionable insights for data-driven decision making
category: data
tags: [analytics, dashboard, visualization, metrics, business-intelligence, real-time]
difficulty: intermediate
author: jezweb
version: 1.0.0
arguments:
  - name: dashboard_type
    description: Dashboard type (executive, operational, analytical, real-time, customer-facing)
    required: true
  - name: data_sources
    description: Primary data sources (database, api, files, streaming, warehouse)
    required: true
  - name: visualization_framework
    description: Visualization framework (react-charts, d3js, plotly, tableau-embedded, custom)
    required: true
  - name: update_frequency
    description: Data update frequency (real-time, near-real-time, hourly, daily, weekly)
    required: true
  - name: user_personas
    description: Target user personas (executives, managers, analysts, customers, mixed)
    required: true
  - name: interactivity_level
    description: Required interactivity (static, basic-filters, advanced-drilling, full-interactive)
    required: true
---

# Analytics Dashboard Design: {{dashboard_type}} Dashboard

**Data Sources:** {{data_sources}}  
**Visualization Framework:** {{visualization_framework}}  
**Update Frequency:** {{update_frequency}}  
**User Personas:** {{user_personas}}  
**Interactivity:** {{interactivity_level}}

## 1. Dashboard Architecture & Design System

### Information Architecture
```yaml
# Dashboard structure and organization
dashboard_architecture:
  layout_pattern: {{#if (eq dashboard_type "executive")}}grid_based_summary{{else if (eq dashboard_type "operational")}}real_time_monitoring{{else if (eq dashboard_type "analytical")}}exploratory_drill_down{{else}}responsive_multi_view{{/if}}
  
  information_hierarchy:
    level_1: "Key Performance Indicators (KPIs)"
    level_2: "Trend Analysis and Comparisons"
    level_3: "Detailed Breakdowns and Segments"
    level_4: "Drill-down and Raw Data Access"
  
  navigation_structure:
    primary_navigation:
      {{#if (eq dashboard_type "executive")}}
      - overview: "Executive Summary"
      - performance: "Performance Metrics"
      - trends: "Market Trends"
      - insights: "Strategic Insights"
      {{else if (eq dashboard_type "operational")}}
      - monitoring: "Real-time Monitoring"
      - alerts: "Active Alerts"
      - capacity: "System Capacity"
      - incidents: "Incident Management"
      {{else if (eq dashboard_type "analytical")}}
      - explore: "Data Exploration"
      - reports: "Standard Reports"
      - segments: "Customer Segments"
      - cohorts: "Cohort Analysis"
      {{else}}
      - overview: "Overview"
      - performance: "Performance"
      - details: "Detailed Views"
      {{/if}}
    
    secondary_navigation:
      - filters: "Global Filters Panel"
      - time_range: "Time Range Selector"
      - comparisons: "Comparison Controls"
      - exports: "Export Options"
  
  responsive_design:
    breakpoints:
      mobile: "320px - 768px"
      tablet: "768px - 1024px"
      desktop: "1024px - 1440px"
      large_screen: "1440px+"
    
    adaptive_layout:
      mobile: "single_column_stack"
      tablet: "two_column_grid"
      desktop: "multi_panel_layout"
      large_screen: "dashboard_wall"

visual_design_system:
  color_palette:
    primary: "#2563eb"  # Blue for primary actions
    secondary: "#64748b"  # Gray for secondary elements
    success: "#10b981"  # Green for positive metrics
    warning: "#f59e0b"  # Yellow for attention
    danger: "#ef4444"   # Red for alerts/negative
    neutral: "#f8fafc"  # Light gray for backgrounds
  
  typography:
    heading_font: "Inter, system-ui, sans-serif"
    body_font: "Inter, system-ui, sans-serif"
    mono_font: "JetBrains Mono, monospace"
    
    font_scale:
      h1: "2.5rem / 40px"  # Dashboard title
      h2: "2rem / 32px"    # Section headers
      h3: "1.5rem / 24px"  # Widget titles
      body: "1rem / 16px"  # Regular text
      small: "0.875rem / 14px"  # Labels, captions
      tiny: "0.75rem / 12px"    # Footnotes
  
  spacing_system:
    base_unit: "4px"
    scale: [4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96]
  
  component_library:
    charts:
      - line_chart: "Time series and trends"
      - bar_chart: "Categorical comparisons"
      - pie_chart: "Proportional data (limited use)"
      - scatter_plot: "Correlation analysis"
      - heatmap: "Intensity and patterns"
      - gauge: "Single metric progress"
      - sparkline: "Inline trend indicators"
    
    controls:
      - date_picker: "Time range selection"
      - dropdown: "Single selection filters"
      - multi_select: "Multiple selection filters"
      - slider: "Range selection"
      - toggle: "Binary options"
      - search: "Text-based filtering"
    
    layout:
      - card: "Grouped content container"
      - panel: "Collapsible content section"
      - modal: "Overlay detailed views"
      - tooltip: "Contextual information"
      - sidebar: "Navigation and filters"
```

### Technical Architecture
```typescript
// Modern dashboard architecture with TypeScript and React
import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { QueryClient, QueryClientProvider, useQuery, useMutation } from '@tanstack/react-query';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';

// Dashboard state management
interface DashboardState {
  filters: {
    dateRange: { start: Date; end: Date };
    segments: string[];
    regions: string[];
    customFilters: Record<string, any>;
  };
  layout: {
    widgets: WidgetConfig[];
    columns: number;
    responsive: boolean;
  };
  user: {
    role: '{{user_personas}}';
    permissions: Permission[];
    preferences: UserPreferences;
  };
  realTimeConnection: {
    connected: boolean;
    lastUpdate: Date;
    updateFrequency: '{{update_frequency}}';
  };
}

interface WidgetConfig {
  id: string;
  type: 'chart' | 'metric' | 'table' | 'map' | 'custom';
  title: string;
  dataSource: string;
  query: QueryDefinition;
  visualization: VisualizationConfig;
  position: { x: number; y: number; w: number; h: number };
  filters?: FilterConfig[];
  interactivity: '{{interactivity_level}}';
}

interface VisualizationConfig {
  chartType: 'line' | 'bar' | 'pie' | 'scatter' | 'heatmap' | 'gauge';
  axes?: {
    x: AxisConfig;
    y: AxisConfig;
  };
  colors: string[];
  annotations?: AnnotationConfig[];
  responsive: boolean;
}

// Context for dashboard state
const DashboardContext = createContext<{
  state: DashboardState;
  dispatch: React.Dispatch<DashboardAction>;
} | null>(null);

// State reducer
type DashboardAction = 
  | { type: 'SET_FILTERS'; payload: Partial<DashboardState['filters']> }
  | { type: 'UPDATE_LAYOUT'; payload: WidgetConfig[] }
  | { type: 'REAL_TIME_UPDATE'; payload: any }
  | { type: 'SET_USER_PREFERENCES'; payload: UserPreferences };

function dashboardReducer(state: DashboardState, action: DashboardAction): DashboardState {
  switch (action.type) {
    case 'SET_FILTERS':
      return {
        ...state,
        filters: { ...state.filters, ...action.payload }
      };
    
    case 'UPDATE_LAYOUT':
      return {
        ...state,
        layout: { ...state.layout, widgets: action.payload }
      };
    
    case 'REAL_TIME_UPDATE':
      return {
        ...state,
        realTimeConnection: {
          ...state.realTimeConnection,
          lastUpdate: new Date(),
          connected: true
        }
      };
    
    default:
      return state;
  }
}

// Main Dashboard Provider
export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(dashboardReducer, initialDashboardState);
  
  {{#if (eq update_frequency "real-time" "near-real-time")}}
  // Real-time data connection
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8080/dashboard-stream');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      dispatch({ type: 'REAL_TIME_UPDATE', payload: data });
    };
    
    ws.onclose = () => {
      // Implement reconnection logic
      setTimeout(() => {
        // Reconnect
      }, 5000);
    };
    
    return () => ws.close();
  }, []);
  {{/if}}
  
  return (
    <DashboardContext.Provider value={{ state, dispatch }}>
      {children}
    </DashboardContext.Provider>
  );
}

// Custom hook for dashboard state
export function useDashboard() {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error('useDashboard must be used within DashboardProvider');
  }
  return context;
}

// Data fetching hooks
export function useWidgetData(widgetId: string, query: QueryDefinition) {
  const { state } = useDashboard();
  
  return useQuery({
    queryKey: ['widget', widgetId, state.filters, query],
    queryFn: async () => {
      const response = await fetch('/api/dashboard/data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          widgetId,
          query,
          filters: state.filters
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch widget data');
      }
      
      return response.json();
    },
    {{#if (eq update_frequency "real-time")}}
    refetchInterval: 1000, // 1 second for real-time
    {{else if (eq update_frequency "near-real-time")}}
    refetchInterval: 10000, // 10 seconds for near real-time
    {{else if (eq update_frequency "hourly")}}
    refetchInterval: 3600000, // 1 hour
    {{else}}
    refetchInterval: false, // Manual refresh only
    {{/if}}
    staleTime: {{#if (eq update_frequency "real-time")}}0{{else if (eq update_frequency "near-real-time")}}5000{{else}}300000{{/if}}
  });
}

// Widget component framework
interface BaseWidgetProps {
  config: WidgetConfig;
  data?: any;
  loading?: boolean;
  error?: Error | null;
}

export function Widget({ config, data, loading, error }: BaseWidgetProps) {
  const { dispatch } = useDashboard();
  
  if (loading) {
    return <WidgetSkeleton />;
  }
  
  if (error) {
    return <WidgetError error={error} onRetry={() => {/* retry logic */}} />;
  }
  
  switch (config.type) {
    case 'chart':
      return <ChartWidget config={config} data={data} />;
    case 'metric':
      return <MetricWidget config={config} data={data} />;
    case 'table':
      return <TableWidget config={config} data={data} />;
    default:
      return <div>Unsupported widget type</div>;
  }
}

// Chart widget implementation
{{#if (eq visualization_framework "react-charts")}}
import { Chart as ChartJS, registerables } from 'chart.js';
import { Line, Bar, Doughnut, Scatter } from 'react-chartjs-2';

ChartJS.register(...registerables);

function ChartWidget({ config, data }: { config: WidgetConfig; data: any }) {
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        display: config.visualization.chartType !== 'gauge'
      },
      title: {
        display: true,
        text: config.title,
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      {{#if (eq update_frequency "real-time" "near-real-time")}}
      annotation: {
        annotations: {
          lastUpdate: {
            type: 'label',
            content: `Last updated: ${new Date().toLocaleTimeString()}`,
            position: {
              x: 'end',
              y: 'start'
            }
          }
        }
      }
      {{/if}}
    },
    scales: config.visualization.axes ? {
      x: {
        title: {
          display: true,
          text: config.visualization.axes.x.label
        },
        type: config.visualization.axes.x.type
      },
      y: {
        title: {
          display: true,
          text: config.visualization.axes.y.label
        },
        beginAtZero: config.visualization.axes.y.beginAtZero
      }
    } : {},
    {{#if (eq interactivity_level "advanced-drilling" "full-interactive")}}
    onClick: (event: any, elements: any[]) => {
      if (elements.length > 0) {
        const element = elements[0];
        const dataIndex = element.index;
        const datasetIndex = element.datasetIndex;
        
        // Handle drill-down interaction
        handleChartClick({
          dataIndex,
          datasetIndex,
          data: data.datasets[datasetIndex].data[dataIndex],
          config
        });
      }
    },
    onHover: (event: any, elements: any[]) => {
      event.native.target.style.cursor = elements.length > 0 ? 'pointer' : 'default';
    }
    {{/if}}
  };
  
  const renderChart = () => {
    switch (config.visualization.chartType) {
      case 'line':
        return <Line data={data} options={chartOptions} />;
      case 'bar':
        return <Bar data={data} options={chartOptions} />;
      case 'pie':
        return <Doughnut data={data} options={chartOptions} />;
      case 'scatter':
        return <Scatter data={data} options={chartOptions} />;
      default:
        return <div>Unsupported chart type</div>;
    }
  };
  
  return (
    <div className="chart-widget" style={{ height: '100%', position: 'relative' }}>
      {renderChart()}
      {{#if (eq interactivity_level "advanced-drilling" "full-interactive")}}
      <ChartControls config={config} />
      {{/if}}
    </div>
  );
}
{{/if}}

{{#if (eq visualization_framework "d3js")}}
// D3.js implementation for custom visualizations
import * as d3 from 'd3';
import { useRef, useEffect } from 'react';

function D3ChartWidget({ config, data }: { config: WidgetConfig; data: any }) {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current || !data) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous render
    
    const margin = { top: 20, right: 30, bottom: 40, left: 40 };
    const containerRect = svgRef.current.getBoundingClientRect();
    const width = containerRect.width - margin.left - margin.right;
    const height = containerRect.height - margin.top - margin.bottom;
    
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    switch (config.visualization.chartType) {
      case 'line':
        renderLineChart(g, data, width, height);
        break;
      case 'bar':
        renderBarChart(g, data, width, height);
        break;
      case 'heatmap':
        renderHeatmap(g, data, width, height);
        break;
      default:
        g.append('text')
          .attr('x', width / 2)
          .attr('y', height / 2)
          .attr('text-anchor', 'middle')
          .text('Chart type not implemented');
    }
  }, [data, config]);
  
  const renderLineChart = (g: any, data: any, width: number, height: number) => {
    const x = d3.scaleTime()
      .domain(d3.extent(data, (d: any) => new Date(d.date)))
      .range([0, width]);
    
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, (d: any) => d.value)])
      .nice()
      .range([height, 0]);
    
    const line = d3.line<any>()
      .x(d => x(new Date(d.date)))
      .y(d => y(d.value))
      .curve(d3.curveMonotoneX);
    
    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x));
    
    g.append('g')
      .call(d3.axisLeft(y));
    
    // Add line
    g.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', config.visualization.colors[0] || '#2563eb')
      .attr('stroke-width', 2)
      .attr('d', line);
    
    // Add interactive dots
    {{#if (eq interactivity_level "advanced-drilling" "full-interactive")}}
    g.selectAll('.dot')
      .data(data)
      .enter().append('circle')
      .attr('class', 'dot')
      .attr('cx', d => x(new Date(d.date)))
      .attr('cy', d => y(d.value))
      .attr('r', 4)
      .attr('fill', config.visualization.colors[0] || '#2563eb')
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        // Show tooltip
        showTooltip(event, d);
      })
      .on('mouseout', hideTooltip)
      .on('click', function(event, d) {
        // Handle drill-down
        handleDataPointClick(d, config);
      });
    {{/if}}
  };
  
  return (
    <div className="d3-chart-widget" style={{ width: '100%', height: '100%' }}>
      <svg ref={svgRef} width="100%" height="100%" />
    </div>
  );
}
{{/if}}

// Metric widget for KPIs
function MetricWidget({ config, data }: { config: WidgetConfig; data: any }) {
  const { value, change, trend, target } = data;
  
  const changePercent = ((value - change) / change * 100).toFixed(1);
  const isPositive = parseFloat(changePercent) > 0;
  const targetProgress = target ? (value / target * 100) : null;
  
  return (
    <div className="metric-widget">
      <div className="metric-header">
        <h3 className="metric-title">{config.title}</h3>
        {{#if (eq update_frequency "real-time" "near-real-time")}}
        <div className="real-time-indicator">
          <span className="status-dot"></span>
          Live
        </div>
        {{/if}}
      </div>
      
      <div className="metric-value">
        <span className="value">{formatMetricValue(value, config.visualization.format)}</span>
        
        {change && (
          <div className={`metric-change ${isPositive ? 'positive' : 'negative'}`}>
            <span className="change-icon">{isPositive ? '↗' : '↘'}</span>
            <span className="change-value">{changePercent}%</span>
          </div>
        )}
      </div>
      
      {target && (
        <div className="metric-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${Math.min(targetProgress, 100)}%` }}
            />
          </div>
          <span className="progress-text">
            {targetProgress?.toFixed(1)}% of target
          </span>
        </div>
      )}
      
      {trend && (
        <div className="metric-sparkline">
          <Sparkline data={trend} />
        </div>
      )}
      
      {{#if (eq interactivity_level "advanced-drilling" "full-interactive")}}
      <div className="metric-actions">
        <button onClick={() => handleMetricDrillDown(config, data)}>
          View Details
        </button>
      </div>
      {{/if}}
    </div>
  );
}

// Dashboard layout management
{{#if (eq interactivity_level "full-interactive")}}
import { Responsive, WidthProvider } from 'react-grid-layout';
const ResponsiveGridLayout = WidthProvider(Responsive);

function DashboardGrid() {
  const { state, dispatch } = useDashboard();
  
  const handleLayoutChange = (layout: any[], layouts: any) => {
    // Update widget positions
    const updatedWidgets = state.layout.widgets.map(widget => {
      const layoutItem = layout.find(item => item.i === widget.id);
      if (layoutItem) {
        return {
          ...widget,
          position: {
            x: layoutItem.x,
            y: layoutItem.y,
            w: layoutItem.w,
            h: layoutItem.h
          }
        };
      }
      return widget;
    });
    
    dispatch({ type: 'UPDATE_LAYOUT', payload: updatedWidgets });
  };
  
  const gridLayouts = {
    lg: state.layout.widgets.map(widget => ({
      i: widget.id,
      x: widget.position.x,
      y: widget.position.y,
      w: widget.position.w,
      h: widget.position.h,
      minW: 2,
      minH: 2
    }))
  };
  
  return (
    <ResponsiveGridLayout
      className="dashboard-grid"
      layouts={gridLayouts}
      onLayoutChange={handleLayoutChange}
      cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
      rowHeight={60}
      isDraggable={true}
      isResizable={true}
    >
      {state.layout.widgets.map(widget => (
        <div key={widget.id} className="grid-item">
          <WidgetContainer config={widget} />
        </div>
      ))}
    </ResponsiveGridLayout>
  );
}
{{/if}}

// Main dashboard component
function Dashboard() {
  const { state } = useDashboard();
  
  return (
    <div className="dashboard">
      <DashboardHeader />
      
      {{#if (eq interactivity_level "basic-filters" "advanced-drilling" "full-interactive")}}
      <DashboardFilters />
      {{/if}}
      
      <div className="dashboard-content">
        {{#if (eq interactivity_level "full-interactive")}}
        <DashboardGrid />
        {{else}}
        <div className="dashboard-static-layout">
          {state.layout.widgets.map(widget => (
            <WidgetContainer key={widget.id} config={widget} />
          ))}
        </div>
        {{/if}}
      </div>
      
      {{#if (eq dashboard_type "operational")}}
      <AlertsPanel />
      {{/if}}
    </div>
  );
}

// Export main app
export default function DashboardApp() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: {{#if (eq update_frequency "real-time")}}0{{else if (eq update_frequency "near-real-time")}}5000{{else}}300000{{/if}},
        retry: 3,
        retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30000)
      }
    }
  });
  
  return (
    <QueryClientProvider client={queryClient}>
      <DashboardProvider>
        <Dashboard />
      </DashboardProvider>
    </QueryClientProvider>
  );
}
```

## 2. Data Integration & API Layer

### Dashboard Data API
```python
# Backend API for dashboard data
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket, WebSocketDisconnect
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import redis
import asyncpg

app = FastAPI(title="Dashboard API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class DashboardQuery(BaseModel):
    widget_id: str
    query: Dict[str, Any]
    filters: Dict[str, Any]
    cache_key: Optional[str] = None

class MetricData(BaseModel):
    value: float
    change: Optional[float] = None
    trend: Optional[List[float]] = None
    target: Optional[float] = None
    format: str = "number"

class ChartData(BaseModel):
    labels: List[str]
    datasets: List[Dict[str, Any]]

# Database connections
DATABASE_URL = "postgresql://user:password@localhost/analytics"
REDIS_URL = "redis://localhost:6379"

engine = create_engine(DATABASE_URL)
redis_client = redis.Redis.from_url(REDIS_URL)

class DashboardDataService:
    def __init__(self):
        self.cache_ttl = {
            "real-time": 0,
            "near-real-time": 10,
            "hourly": 3600,
            "daily": 86400
        }
    
    async def get_widget_data(self, query: DashboardQuery) -> Dict[str, Any]:
        """Get data for a dashboard widget"""
        
        # Check cache first
        if query.cache_key:
            cached_data = redis_client.get(query.cache_key)
            if cached_data:
                return json.loads(cached_data)
        
        # Generate data based on widget type
        widget_type = query.query.get("type")
        
        if widget_type == "metric":
            data = await self._get_metric_data(query)
        elif widget_type == "chart":
            data = await self._get_chart_data(query)
        elif widget_type == "table":
            data = await self._get_table_data(query)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported widget type: {widget_type}")
        
        # Cache the result
        if query.cache_key:
            cache_duration = self.cache_ttl.get("{{update_frequency}}", 300)
            redis_client.setex(query.cache_key, cache_duration, json.dumps(data))
        
        return data
    
    async def _get_metric_data(self, query: DashboardQuery) -> MetricData:
        """Get KPI metric data"""
        
        metric_config = query.query.get("metric", {})
        metric_name = metric_config.get("name")
        aggregation = metric_config.get("aggregation", "sum")
        
        # Build SQL query based on filters
        sql_query = self._build_metric_query(metric_name, aggregation, query.filters)
        
        try:
            with engine.connect() as conn:
                result = conn.execute(sql_query)
                row = result.fetchone()
                
                current_value = float(row[0]) if row and row[0] else 0
                
                # Get comparison period data
                comparison_value = await self._get_comparison_value(
                    metric_name, aggregation, query.filters
                )
                
                # Get trend data
                trend_data = await self._get_trend_data(
                    metric_name, aggregation, query.filters
                )
                
                return MetricData(
                    value=current_value,
                    change=comparison_value,
                    trend=trend_data,
                    target=metric_config.get("target"),
                    format=metric_config.get("format", "number")
                )
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    
    async def _get_chart_data(self, query: DashboardQuery) -> ChartData:
        """Get chart visualization data"""
        
        chart_config = query.query.get("chart", {})
        chart_type = chart_config.get("type")
        
        if chart_type == "time_series":
            return await self._get_time_series_data(query)
        elif chart_type == "categorical":
            return await self._get_categorical_data(query)
        elif chart_type == "distribution":
            return await self._get_distribution_data(query)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported chart type: {chart_type}")
    
    async def _get_time_series_data(self, query: DashboardQuery) -> ChartData:
        """Get time series data for line/area charts"""
        
        chart_config = query.query.get("chart", {})
        metric = chart_config.get("metric")
        time_granularity = chart_config.get("granularity", "daily")
        
        # Build time series query
        sql_query = f"""
        SELECT 
            DATE_TRUNC('{time_granularity}', timestamp) as time_bucket,
            {self._get_aggregation_sql(metric, chart_config.get('aggregation', 'sum'))}
        FROM events e
        WHERE timestamp >= %s AND timestamp <= %s
        {self._build_filter_clause(query.filters)}
        GROUP BY time_bucket
        ORDER BY time_bucket
        """
        
        date_range = query.filters.get("dateRange", {})
        start_date = date_range.get("start", datetime.now() - timedelta(days=30))
        end_date = date_range.get("end", datetime.now())
        
        with engine.connect() as conn:
            result = conn.execute(sql_query, (start_date, end_date))
            rows = result.fetchall()
            
            labels = [row[0].strftime("%Y-%m-%d") for row in rows]
            values = [float(row[1]) if row[1] else 0 for row in rows]
            
            return ChartData(
                labels=labels,
                datasets=[{
                    "label": metric,
                    "data": values,
                    "borderColor": chart_config.get("color", "#2563eb"),
                    "backgroundColor": chart_config.get("backgroundColor", "rgba(37, 99, 235, 0.1)"),
                    "fill": chart_config.get("fill", False)
                }]
            )
    
    def _build_filter_clause(self, filters: Dict[str, Any]) -> str:
        """Build SQL WHERE clause from filters"""
        
        clauses = []
        
        if segments := filters.get("segments"):
            segment_list = "', '".join(segments)
            clauses.append(f"segment IN ('{segment_list}')")
        
        if regions := filters.get("regions"):
            region_list = "', '".join(regions)
            clauses.append(f"region IN ('{region_list}')")
        
        # Add custom filters
        for key, value in filters.get("customFilters", {}).items():
            if isinstance(value, list):
                value_list = "', '".join(str(v) for v in value)
                clauses.append(f"{key} IN ('{value_list}')")
            else:
                clauses.append(f"{key} = '{value}'")
        
        return " AND " + " AND ".join(clauses) if clauses else ""
    
    def _get_aggregation_sql(self, metric: str, aggregation: str) -> str:
        """Get SQL aggregation expression"""
        
        agg_map = {
            "sum": f"SUM({metric})",
            "avg": f"AVG({metric})",
            "count": f"COUNT({metric})",
            "distinct_count": f"COUNT(DISTINCT {metric})",
            "min": f"MIN({metric})",
            "max": f"MAX({metric})"
        }
        
        return agg_map.get(aggregation, f"SUM({metric})")

# API endpoints
data_service = DashboardDataService()

@app.post("/api/dashboard/data")
async def get_dashboard_data(query: DashboardQuery):
    """Get data for a dashboard widget"""
    try:
        data = await data_service.get_widget_data(query)
        return {"success": True, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

{{#if (eq dashboard_type "executive")}}
@app.get("/api/dashboard/executive-summary")
async def get_executive_summary():
    """Get executive dashboard summary"""
    
    # Key metrics for executives
    summary_metrics = [
        {
            "id": "revenue",
            "title": "Total Revenue",
            "query": {
                "type": "metric",
                "metric": {"name": "revenue", "aggregation": "sum", "format": "currency"}
            }
        },
        {
            "id": "customers",
            "title": "Active Customers",
            "query": {
                "type": "metric", 
                "metric": {"name": "customer_id", "aggregation": "distinct_count"}
            }
        },
        {
            "id": "growth_rate",
            "title": "Growth Rate",
            "query": {
                "type": "metric",
                "metric": {"name": "revenue", "aggregation": "sum", "format": "percentage"}
            }
        }
    ]
    
    results = {}
    for metric in summary_metrics:
        query = DashboardQuery(
            widget_id=metric["id"],
            query=metric["query"],
            filters={"dateRange": {"start": datetime.now() - timedelta(days=30)}}
        )
        results[metric["id"]] = await data_service.get_widget_data(query)
    
    return {"success": True, "data": results}
{{/if}}

{{#if (eq update_frequency "real-time" "near-real-time")}}
# WebSocket for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection lost, remove it
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.websocket("/ws/dashboard-stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Wait for client messages (if any)
            data = await websocket.receive_text()
            
            # Echo back for testing
            await websocket.send_text(f"Echo: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def send_real_time_updates():
    """Background task to send real-time updates"""
    while True:
        # Generate sample real-time data
        update_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "active_users": np.random.randint(100, 1000),
                "transactions_per_minute": np.random.randint(10, 100),
                "error_rate": np.random.uniform(0, 5)
            }
        }
        
        await manager.broadcast(json.dumps(update_data))
        await asyncio.sleep({{#if (eq update_frequency "real-time")}}1{{else}}10{{/if}})  # Update frequency

# Start background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(send_real_time_updates())
{{/if}}

{{#if (eq dashboard_type "operational")}}
@app.get("/api/dashboard/alerts")
async def get_active_alerts():
    """Get active alerts for operational dashboard"""
    
    alerts_query = """
    SELECT 
        alert_id,
        alert_type,
        severity,
        message,
        created_at,
        acknowledged
    FROM system_alerts 
    WHERE status = 'active' 
    ORDER BY severity DESC, created_at DESC
    LIMIT 50
    """
    
    with engine.connect() as conn:
        result = conn.execute(alerts_query)
        alerts = [dict(row) for row in result.fetchall()]
    
    return {"success": True, "data": alerts}

@app.post("/api/dashboard/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    
    update_query = """
    UPDATE system_alerts 
    SET acknowledged = true, acknowledged_at = NOW() 
    WHERE alert_id = %s
    """
    
    with engine.connect() as conn:
        conn.execute(update_query, (alert_id,))
    
    return {"success": True, "message": "Alert acknowledged"}
{{/if}}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    # Check database connection
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    # Check Redis connection
    try:
        redis_client.ping()
        cache_status = "healthy"
    except:
        cache_status = "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" and cache_status == "healthy" else "unhealthy",
        "database": db_status,
        "cache": cache_status,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 3. Advanced Analytics Features

### Statistical Analysis & Insights
```python
# Advanced analytics engine for dashboard insights
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalyticsEngine:
    def __init__(self):
        self.scalers = {}
        self.models = {}
    
    def detect_anomalies(self, data: pd.DataFrame, column: str, method: str = "isolation_forest") -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        if method == "isolation_forest":
            # Isolation Forest for anomaly detection
            model = IsolationForest(contamination=0.1, random_state=42)
            
            # Prepare features
            features = self._create_time_features(data, column)
            
            # Fit and predict
            anomalies = model.fit_predict(features)
            anomaly_scores = model.decision_function(features)
            
            # Add results to dataframe
            results = data.copy()
            results['is_anomaly'] = anomalies == -1
            results['anomaly_score'] = anomaly_scores
            
            return {
                "anomalies": results[results['is_anomaly']].to_dict('records'),
                "total_anomalies": sum(anomalies == -1),
                "anomaly_percentage": (sum(anomalies == -1) / len(anomalies)) * 100,
                "method": method
            }
        
        elif method == "statistical":
            # Statistical outlier detection using Z-score
            z_scores = np.abs(stats.zscore(data[column].dropna()))
            threshold = 3
            anomalies = z_scores > threshold
            
            results = data.copy()
            results['is_anomaly'] = anomalies
            results['z_score'] = z_scores
            
            return {
                "anomalies": results[results['is_anomaly']].to_dict('records'),
                "total_anomalies": sum(anomalies),
                "anomaly_percentage": (sum(anomalies) / len(anomalies)) * 100,
                "method": method,
                "threshold": threshold
            }
    
    def perform_cohort_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform cohort analysis for user retention"""
        
        # Ensure required columns exist
        required_cols = ['user_id', 'order_date', 'revenue']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        data['order_date'] = pd.to_datetime(data['order_date'])
        data['order_period'] = data['order_date'].dt.to_period('M')
        
        # Get user's first purchase date
        cohort_data = data.groupby('user_id')['order_date'].min().reset_index()
        cohort_data.columns = ['user_id', 'cohort_group']
        cohort_data['cohort_group'] = cohort_data['cohort_group'].dt.to_period('M')
        
        # Merge back to original data
        data_cohort = data.merge(cohort_data, on='user_id')
        data_cohort['period_number'] = (
            data_cohort['order_period'] - data_cohort['cohort_group']
        ).apply(attrgetter('n'))
        
        # Calculate cohort table
        cohort_table = data_cohort.groupby(['cohort_group', 'period_number'])['user_id'].nunique().reset_index()
        cohort_counts = cohort_table.pivot(index='cohort_group', 
                                         columns='period_number', 
                                         values='user_id')
        
        # Calculate retention rates
        cohort_sizes = data_cohort.groupby('cohort_group')['user_id'].nunique()
        retention_table = cohort_counts.divide(cohort_sizes, axis=0)
        
        return {
            "cohort_counts": cohort_counts.to_dict(),
            "retention_rates": retention_table.to_dict(),
            "cohort_sizes": cohort_sizes.to_dict(),
            "avg_retention": {
                "month_1": retention_table[1].mean() if 1 in retention_table.columns else 0,
                "month_3": retention_table[3].mean() if 3 in retention_table.columns else 0,
                "month_6": retention_table[6].mean() if 6 in retention_table.columns else 0,
                "month_12": retention_table[12].mean() if 12 in retention_table.columns else 0
            }
        }
    
    def segment_customers(self, data: pd.DataFrame, features: List[str], n_clusters: int = 4) -> Dict[str, Any]:
        """Perform customer segmentation using K-means clustering"""
        
        # Prepare data
        feature_data = data[features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to data
        result_data = data.copy()
        result_data['cluster'] = clusters
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_data = result_data[result_data['cluster'] == i]
            cluster_stats[f'cluster_{i}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(result_data) * 100,
                'avg_metrics': cluster_data[features].mean().to_dict()
            }
        
        return {
            "segments": result_data.to_dict('records'),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "cluster_stats": cluster_stats,
            "features_used": features,
            "n_clusters": n_clusters
        }
    
    def forecast_trend(self, data: pd.DataFrame, value_column: str, periods: int = 30) -> Dict[str, Any]:
        """Generate trend forecast using ARIMA"""
        
        # Prepare time series data
        ts_data = data.set_index('date')[value_column].asfreq('D')
        ts_data = ts_data.fillna(method='ffill')
        
        try:
            # Fit ARIMA model
            model = sm.tsa.ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=periods)
            conf_int = fitted_model.get_forecast(steps=periods).conf_int()
            
            # Create forecast dates
            last_date = ts_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
            
            return {
                "forecast_values": forecast.tolist(),
                "forecast_dates": forecast_dates.strftime('%Y-%m-%d').tolist(),
                "confidence_interval": {
                    "lower": conf_int.iloc[:, 0].tolist(),
                    "upper": conf_int.iloc[:, 1].tolist()
                },
                "model_summary": {
                    "aic": fitted_model.aic,
                    "bic": fitted_model.bic,
                    "order": (1, 1, 1)
                }
            }
            
        except Exception as e:
            # Fallback to simple linear trend
            X = np.arange(len(ts_data)).reshape(-1, 1)
            y = ts_data.values
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
            
            # Generate forecast
            future_X = np.arange(len(ts_data), len(ts_data) + periods).reshape(-1, 1)
            forecast = slope * future_X.flatten() + intercept
            
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
            
            return {
                "forecast_values": forecast.tolist(),
                "forecast_dates": forecast_dates.strftime('%Y-%m-%d').tolist(),
                "model_summary": {
                    "method": "linear_regression",
                    "r_squared": r_value**2,
                    "slope": slope,
                    "intercept": intercept
                },
                "error": f"ARIMA failed, used linear regression: {str(e)}"
            }
    
    def calculate_statistical_significance(self, data_a: List[float], data_b: List[float]) -> Dict[str, Any]:
        """Calculate statistical significance between two groups"""
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data_a, data_b)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(data_a) - 1) * np.var(data_a, ddof=1) + 
                             (len(data_b) - 1) * np.var(data_b, ddof=1)) / 
                            (len(data_a) + len(data_b) - 2))
        
        cohens_d = (np.mean(data_a) - np.mean(data_b)) / pooled_std
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
            "confidence_level": 0.95,
            "effect_size": {
                "cohens_d": cohens_d,
                "interpretation": effect_interpretation
            },
            "group_statistics": {
                "group_a": {
                    "mean": np.mean(data_a),
                    "std": np.std(data_a),
                    "n": len(data_a)
                },
                "group_b": {
                    "mean": np.mean(data_b),
                    "std": np.std(data_b),
                    "n": len(data_b)
                }
            }
        }
    
    def _create_time_features(self, data: pd.DataFrame, value_column: str) -> np.ndarray:
        """Create time-based features for anomaly detection"""
        
        # Assume data has a datetime index or column
        if 'date' in data.columns:
            data = data.set_index('date')
        
        features = []
        
        # Add value itself
        features.append(data[value_column].values.reshape(-1, 1))
        
        # Add rolling statistics
        for window in [7, 14, 30]:
            rolling_mean = data[value_column].rolling(window=window).mean().fillna(0)
            rolling_std = data[value_column].rolling(window=window).std().fillna(0)
            features.extend([rolling_mean.values.reshape(-1, 1), rolling_std.values.reshape(-1, 1)])
        
        # Add lag features
        for lag in [1, 7, 30]:
            lag_feature = data[value_column].shift(lag).fillna(0)
            features.append(lag_feature.values.reshape(-1, 1))
        
        return np.hstack(features)

# Integration with dashboard API
@app.post("/api/analytics/anomalies")
async def detect_data_anomalies(
    table: str,
    column: str,
    method: str = "isolation_forest",
    date_range: Dict[str, str] = None
):
    """Detect anomalies in dashboard data"""
    
    analytics_engine = AdvancedAnalyticsEngine()
    
    # Query data
    query = f"""
    SELECT date, {column}
    FROM {table}
    WHERE date >= %s AND date <= %s
    ORDER BY date
    """
    
    start_date = date_range.get("start", (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
    end_date = date_range.get("end", datetime.now().strftime("%Y-%m-%d"))
    
    with engine.connect() as conn:
        data = pd.read_sql(query, conn, params=[start_date, end_date])
    
    # Detect anomalies
    results = analytics_engine.detect_anomalies(data, column, method)
    
    return {"success": True, "data": results}

@app.post("/api/analytics/forecast")
async def generate_forecast(
    table: str,
    column: str,
    periods: int = 30,
    date_range: Dict[str, str] = None
):
    """Generate forecast for dashboard metrics"""
    
    analytics_engine = AdvancedAnalyticsEngine()
    
    # Query historical data
    query = f"""
    SELECT date, {column}
    FROM {table}
    WHERE date >= %s AND date <= %s
    ORDER BY date
    """
    
    start_date = date_range.get("start", (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"))
    end_date = date_range.get("end", datetime.now().strftime("%Y-%m-%d"))
    
    with engine.connect() as conn:
        data = pd.read_sql(query, conn, params=[start_date, end_date])
    
    # Generate forecast
    forecast_results = analytics_engine.forecast_trend(data, column, periods)
    
    return {"success": True, "data": forecast_results}

{{#if (eq dashboard_type "analytical")}}
@app.post("/api/analytics/cohort-analysis")
async def perform_cohort_analysis():
    """Perform cohort analysis for user retention"""
    
    analytics_engine = AdvancedAnalyticsEngine()
    
    # Query user transaction data
    query = """
    SELECT 
        user_id,
        order_date,
        revenue
    FROM transactions
    WHERE order_date >= %s
    ORDER BY user_id, order_date
    """
    
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    with engine.connect() as conn:
        data = pd.read_sql(query, conn, params=[start_date])
    
    # Perform cohort analysis
    cohort_results = analytics_engine.perform_cohort_analysis(data)
    
    return {"success": True, "data": cohort_results}

@app.post("/api/analytics/customer-segmentation")
async def segment_customers():
    """Perform customer segmentation analysis"""
    
    analytics_engine = AdvancedAnalyticsEngine()
    
    # Query customer metrics
    query = """
    SELECT 
        user_id,
        total_revenue,
        order_count,
        avg_order_value,
        days_since_last_order,
        customer_lifetime_days
    FROM customer_metrics
    """
    
    with engine.connect() as conn:
        data = pd.read_sql(query, conn)
    
    # Perform segmentation
    features = ['total_revenue', 'order_count', 'avg_order_value', 'days_since_last_order']
    segmentation_results = analytics_engine.segment_customers(data, features)
    
    return {"success": True, "data": segmentation_results}
{{/if}}
```

## 4. Performance Optimization & Monitoring

### Dashboard Performance Framework
```javascript
// Performance monitoring and optimization
class DashboardPerformanceMonitor {
  constructor() {
    this.metrics = {
      loadTime: 0,
      renderTime: 0,
      apiResponseTimes: {},
      memoryUsage: 0,
      errorCount: 0
    };
    
    this.thresholds = {
      loadTime: {{#if (eq update_frequency "real-time")}}1000{{else if (eq update_frequency "near-real-time")}}2000{{else}}5000{{/if}}, // ms
      renderTime: 100, // ms
      apiResponseTime: {{#if (eq update_frequency "real-time")}}500{{else}}2000{{/if}}, // ms
      memoryUsage: 100 * 1024 * 1024 // 100MB
    };
    
    this.startMonitoring();
  }
  
  startMonitoring() {
    // Monitor initial page load
    this.measurePageLoad();
    
    // Monitor API performance
    this.interceptAPIRequests();
    
    // Monitor memory usage
    this.monitorMemoryUsage();
    
    // Monitor render performance
    this.monitorRenderPerformance();
    
    {{#if (eq update_frequency "real-time" "near-real-time")}}
    // Monitor real-time connection health
    this.monitorWebSocketHealth();
    {{/if}}
  }
  
  measurePageLoad() {
    window.addEventListener('load', () => {
      const navigation = performance.getEntriesByType('navigation')[0];
      this.metrics.loadTime = navigation.loadEventEnd - navigation.fetchStart;
      
      // Check if load time exceeds threshold
      if (this.metrics.loadTime > this.thresholds.loadTime) {
        this.reportPerformanceIssue('SLOW_PAGE_LOAD', {
          actualTime: this.metrics.loadTime,
          threshold: this.thresholds.loadTime
        });
      }
      
      // Send metrics to analytics
      this.sendMetrics('page_load', {
        loadTime: this.metrics.loadTime,
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.fetchStart,
        firstPaint: this.getFirstPaint(),
        firstContentfulPaint: this.getFirstContentfulPaint()
      });
    });
  }
  
  interceptAPIRequests() {
    const originalFetch = window.fetch;
    
    window.fetch = async (...args) => {
      const startTime = performance.now();
      const url = args[0];
      
      try {
        const response = await originalFetch.apply(window, args);
        const endTime = performance.now();
        const duration = endTime - startTime;
        
        // Track API response time
        this.metrics.apiResponseTimes[url] = duration;
        
        // Check if response time exceeds threshold
        if (duration > this.thresholds.apiResponseTime) {
          this.reportPerformanceIssue('SLOW_API_RESPONSE', {
            url,
            duration,
            threshold: this.thresholds.apiResponseTime
          });
        }
        
        // Send API metrics
        this.sendMetrics('api_request', {
          url,
          duration,
          status: response.status,
          success: response.ok
        });
        
        return response;
      } catch (error) {
        const endTime = performance.now();
        const duration = endTime - startTime;
        
        this.metrics.errorCount++;
        
        this.sendMetrics('api_error', {
          url,
          duration,
          error: error.message
        });
        
        throw error;
      }
    };
  }
  
  monitorMemoryUsage() {
    if ('memory' in performance) {
      setInterval(() => {
        const memory = performance.memory;
        this.metrics.memoryUsage = memory.usedJSHeapSize;
        
        // Check for memory leaks
        if (memory.usedJSHeapSize > this.thresholds.memoryUsage) {
          this.reportPerformanceIssue('HIGH_MEMORY_USAGE', {
            usedMemory: memory.usedJSHeapSize,
            threshold: this.thresholds.memoryUsage,
            totalMemory: memory.totalJSHeapSize
          });
        }
        
        // Send memory metrics
        this.sendMetrics('memory_usage', {
          used: memory.usedJSHeapSize,
          total: memory.totalJSHeapSize,
          limit: memory.jsHeapSizeLimit
        });
      }, 30000); // Check every 30 seconds
    }
  }
  
  monitorRenderPerformance() {
    // Use Intersection Observer to monitor widget visibility
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const startTime = performance.now();
          
          // Monitor when widget completes rendering
          requestAnimationFrame(() => {
            const renderTime = performance.now() - startTime;
            
            if (renderTime > this.thresholds.renderTime) {
              this.reportPerformanceIssue('SLOW_WIDGET_RENDER', {
                widgetId: entry.target.id,
                renderTime,
                threshold: this.thresholds.renderTime
              });
            }
            
            this.sendMetrics('widget_render', {
              widgetId: entry.target.id,
              renderTime,
              visible: true
            });
          });
        }
      });
    }, { threshold: 0.1 });
    
    // Observe all widgets
    document.querySelectorAll('[data-widget-id]').forEach(widget => {
      observer.observe(widget);
    });
  }
  
  {{#if (eq update_frequency "real-time" "near-real-time")}}
  monitorWebSocketHealth() {
    const wsMonitor = {
      connectionAttempts: 0,
      reconnectionTime: 0,
      messageCount: 0,
      lastMessageTime: Date.now()
    };
    
    // Monitor WebSocket connection health
    const checkConnectionHealth = () => {
      const timeSinceLastMessage = Date.now() - wsMonitor.lastMessageTime;
      
      if (timeSinceLastMessage > 60000) { // 1 minute without updates
        this.reportPerformanceIssue('WEBSOCKET_INACTIVE', {
          timeSinceLastMessage,
          messageCount: wsMonitor.messageCount
        });
      }
      
      this.sendMetrics('websocket_health', {
        connectionAttempts: wsMonitor.connectionAttempts,
        messageCount: wsMonitor.messageCount,
        timeSinceLastMessage
      });
    };
    
    setInterval(checkConnectionHealth, 30000); // Check every 30 seconds
  }
  {{/if}}
  
  reportPerformanceIssue(type, details) {
    console.warn(`Performance Issue: ${type}`, details);
    
    // Send to monitoring service
    this.sendMetrics('performance_issue', {
      type,
      details,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
      url: window.location.href
    });
  }
  
  sendMetrics(eventType, data) {
    // Send metrics to analytics endpoint
    fetch('/api/analytics/performance', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        eventType,
        data,
        timestamp: Date.now(),
        sessionId: this.getSessionId(),
        userId: this.getUserId()
      })
    }).catch(error => {
      console.error('Failed to send performance metrics:', error);
    });
  }
  
  getFirstPaint() {
    const paint = performance.getEntriesByType('paint')
      .find(entry => entry.name === 'first-paint');
    return paint ? paint.startTime : 0;
  }
  
  getFirstContentfulPaint() {
    const paint = performance.getEntriesByType('paint')
      .find(entry => entry.name === 'first-contentful-paint');
    return paint ? paint.startTime : 0;
  }
  
  getSessionId() {
    return sessionStorage.getItem('sessionId') || 'unknown';
  }
  
  getUserId() {
    return localStorage.getItem('userId') || 'anonymous';
  }
  
  // Public API for manual performance measurement
  startTimer(name) {
    performance.mark(`${name}-start`);
  }
  
  endTimer(name) {
    performance.mark(`${name}-end`);
    performance.measure(name, `${name}-start`, `${name}-end`);
    
    const measure = performance.getEntriesByName(name)[0];
    return measure.duration;
  }
  
  getMetricsSummary() {
    return {
      ...this.metrics,
      timestamp: Date.now(),
      performanceScore: this.calculatePerformanceScore()
    };
  }
  
  calculatePerformanceScore() {
    let score = 100;
    
    // Deduct points for slow load time
    if (this.metrics.loadTime > this.thresholds.loadTime) {
      score -= 20;
    }
    
    // Deduct points for API issues
    const slowApiCount = Object.values(this.metrics.apiResponseTimes)
      .filter(time => time > this.thresholds.apiResponseTime).length;
    score -= slowApiCount * 5;
    
    // Deduct points for errors
    score -= this.metrics.errorCount * 10;
    
    // Deduct points for high memory usage
    if (this.metrics.memoryUsage > this.thresholds.memoryUsage) {
      score -= 15;
    }
    
    return Math.max(0, score);
  }
}

// Initialize performance monitoring
const performanceMonitor = new DashboardPerformanceMonitor();

// Export for use in components
export { performanceMonitor };

// Performance optimization utilities
export class DashboardOptimizer {
  static optimizeChartRendering(chartData, maxDataPoints = {{#if (eq update_frequency "real-time")}}1000{{else}}5000{{/if}}) {
    // Downsample data for better performance
    if (chartData.labels.length > maxDataPoints) {
      const step = Math.ceil(chartData.labels.length / maxDataPoints);
      
      return {
        labels: chartData.labels.filter((_, index) => index % step === 0),
        datasets: chartData.datasets.map(dataset => ({
          ...dataset,
          data: dataset.data.filter((_, index) => index % step === 0)
        }))
      };
    }
    
    return chartData;
  }
  
  static debounceFilterUpdates(callback, delay = 300) {
    let timeoutId;
    
    return (...args) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => callback.apply(this, args), delay);
    };
  }
  
  static memoizeApiRequests() {
    const cache = new Map();
    const cacheTimeout = {{#if (eq update_frequency "real-time")}}1000{{else if (eq update_frequency "near-real-time")}}10000{{else}}300000{{/if}}; // Cache timeout in ms
    
    return (key, fetchFunction) => {
      const cached = cache.get(key);
      
      if (cached && Date.now() - cached.timestamp < cacheTimeout) {
        return Promise.resolve(cached.data);
      }
      
      return fetchFunction().then(data => {
        cache.set(key, {
          data,
          timestamp: Date.now()
        });
        return data;
      });
    };
  }
  
  static optimizeImageLoading() {
    // Lazy load images and use WebP format when supported
    const images = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          const src = img.dataset.src;
          
          // Check WebP support
          const supportsWebP = (
            'createImageBitmap' in window &&
            HTMLCanvasElement.prototype.toBlob
          );
          
          img.src = supportsWebP ? src.replace(/\.(jpg|png)$/, '.webp') : src;
          img.removeAttribute('data-src');
          imageObserver.unobserve(img);
        }
      });
    });
    
    images.forEach(img => imageObserver.observe(img));
  }
}
```

## Conclusion

This Analytics Dashboard Design provides:

**Key Features:**
- {{dashboard_type}} dashboard optimized for {{user_personas}}
- {{visualization_framework}} visualizations with {{interactivity_level}} interactivity
- {{update_frequency}} data updates from {{data_sources}}
- Advanced analytics with anomaly detection and forecasting

**Benefits:**
- Responsive design supporting all device types
- Real-time performance monitoring and optimization
- Advanced statistical analysis and insights
- Scalable architecture supporting high user loads

**Technical Architecture:**
- Modern React-based frontend with TypeScript
- High-performance data API with caching
- WebSocket support for real-time updates
- Comprehensive performance monitoring

**Success Metrics:**
- Page load time under {{#if (eq update_frequency "real-time")}}1 second{{else}}3 seconds{{/if}}
- 99.9% dashboard availability
- Advanced analytics insights driving decisions
- Intuitive user experience for {{user_personas}}