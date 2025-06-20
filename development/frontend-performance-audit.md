---
name: frontend_performance_audit
title: Frontend Performance Audit Framework
description: Comprehensive frontend performance audit with Core Web Vitals analysis, optimization strategies, and monitoring implementation for modern web applications
category: development
tags: [frontend, performance, optimization, core-web-vitals, lighthouse, monitoring]
difficulty: intermediate
author: jezweb
version: 1.0.0
arguments:
  - name: application_type
    description: Application type (spa, mpa, pwa, mobile-web, ecommerce, blog)
    required: true
  - name: framework
    description: Frontend framework (react, vue, angular, nextjs, nuxt, vanilla, wordpress)
    required: true
  - name: target_metrics
    description: Performance targets (lighthouse-90+, core-web-vitals-good, fast-3g, slow-3g)
    required: true
  - name: user_base
    description: Primary user base (global, mobile-first, desktop-first, enterprise)
    required: true
  - name: business_impact
    description: Business impact focus (conversion, engagement, seo, user-satisfaction)
    required: true
  - name: current_score
    description: Current Lighthouse score range (poor 0-49, needs-improvement 50-89, good 90-100)
    required: false
    default: "needs-improvement"
---

# Frontend Performance Audit: {{application_type}} ({{framework}})

**Target Metrics:** {{target_metrics}}  
**User Base:** {{user_base}}  
**Business Focus:** {{business_impact}}  
**Current Score:** {{current_score}}

## 1. Performance Audit Setup & Baseline

### Automated Audit Tools Configuration
```javascript
// Comprehensive performance audit script
const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');
const fs = require('fs');

class FrontendPerformanceAuditor {
    constructor(urls, options = {}) {
        this.urls = Array.isArray(urls) ? urls : [urls];
        this.options = {
            throttling: options.throttling || 'slow3G',
            device: options.device || 'mobile',
            iterations: options.iterations || 3,
            ...options
        };
        this.results = [];
    }

    async runAudit() {
        const chrome = await chromeLauncher.launch({chromeFlags: ['--headless']});
        
        const lighthouseOptions = {
            logLevel: 'info',
            output: 'json',
            onlyCategories: ['performance'],
            port: chrome.port,
            throttling: this.options.throttling,
            device: this.options.device
        };

        for (const url of this.urls) {
            console.log(`Auditing ${url}...`);
            
            const iterationResults = [];
            
            // Run multiple iterations for statistical significance
            for (let i = 0; i < this.options.iterations; i++) {
                try {
                    const runnerResult = await lighthouse(url, lighthouseOptions);
                    iterationResults.push(this.extractMetrics(runnerResult.lhr));
                } catch (error) {
                    console.error(`Error auditing ${url} (iteration ${i + 1}):`, error);
                }
            }
            
            // Calculate median values
            const aggregatedResults = this.aggregateResults(iterationResults);
            this.results.push({
                url,
                metrics: aggregatedResults,
                timestamp: new Date().toISOString()
            });
        }
        
        await chrome.kill();
        return this.results;
    }

    extractMetrics(lighthouseResult) {
        const audits = lighthouseResult.audits;
        
        return {
            // Core Web Vitals
            fcp: audits['first-contentful-paint']?.numericValue || 0,
            lcp: audits['largest-contentful-paint']?.numericValue || 0,
            cls: audits['cumulative-layout-shift']?.numericValue || 0,
            fid: audits['max-potential-fid']?.numericValue || 0, // Proxy for FID
            
            // Other Key Metrics
            ttfb: audits['server-response-time']?.numericValue || 0,
            tti: audits['interactive']?.numericValue || 0,
            tbt: audits['total-blocking-time']?.numericValue || 0,
            
            // Resource Metrics
            totalByteWeight: audits['total-byte-weight']?.numericValue || 0,
            unusedJavaScript: audits['unused-javascript']?.details?.overallSavingsBytes || 0,
            unusedCSS: audits['unused-css-rules']?.details?.overallSavingsBytes || 0,
            
            // Image Optimization
            modernImageFormats: audits['modern-image-formats']?.details?.overallSavingsBytes || 0,
            optimizedImages: audits['uses-optimized-images']?.details?.overallSavingsBytes || 0,
            
            // Lighthouse Score
            performanceScore: lighthouseResult.categories.performance.score * 100
        };
    }

    aggregateResults(iterationResults) {
        if (iterationResults.length === 0) return {};
        
        const metrics = {};
        const keys = Object.keys(iterationResults[0]);
        
        keys.forEach(key => {
            const values = iterationResults.map(result => result[key]).sort((a, b) => a - b);
            metrics[key] = {
                median: this.getMedian(values),
                min: Math.min(...values),
                max: Math.max(...values),
                avg: values.reduce((sum, val) => sum + val, 0) / values.length
            };
        });
        
        return metrics;
    }

    getMedian(sortedArray) {
        const mid = Math.floor(sortedArray.length / 2);
        return sortedArray.length % 2 === 0
            ? (sortedArray[mid - 1] + sortedArray[mid]) / 2
            : sortedArray[mid];
    }

    generateReport() {
        const report = {
            summary: this.generateSummary(),
            detailed_results: this.results,
            recommendations: this.generateRecommendations(),
            core_web_vitals_analysis: this.analyzeCoreWebVitals()
        };

        // Save report
        fs.writeFileSync(
            `performance-audit-${Date.now()}.json`,
            JSON.stringify(report, null, 2)
        );

        return report;
    }

    generateSummary() {
        const allMetrics = this.results.flatMap(result => 
            Object.keys(result.metrics).map(key => result.metrics[key].median)
        );

        return {
            total_urls_audited: this.results.length,
            average_performance_score: this.calculateAverageScore(),
            core_web_vitals_status: this.getCoreWebVitalsStatus(),
            primary_issues: this.identifyPrimaryIssues(),
            estimated_improvement_potential: this.estimateImprovementPotential()
        };
    }

    analyzeCoreWebVitals() {
        const analysis = {};
        
        this.results.forEach(result => {
            const metrics = result.metrics;
            
            analysis[result.url] = {
                fcp_status: this.getMetricStatus('fcp', metrics.fcp?.median),
                lcp_status: this.getMetricStatus('lcp', metrics.lcp?.median),
                cls_status: this.getMetricStatus('cls', metrics.cls?.median),
                fid_status: this.getMetricStatus('fid', metrics.fid?.median),
                overall_cwv_status: this.getOverallCWVStatus(metrics)
            };
        });
        
        return analysis;
    }

    getMetricStatus(metric, value) {
        const thresholds = {
            fcp: { good: 1800, needsImprovement: 3000 },
            lcp: { good: 2500, needsImprovement: 4000 },
            cls: { good: 0.1, needsImprovement: 0.25 },
            fid: { good: 100, needsImprovement: 300 }
        };

        const threshold = thresholds[metric];
        if (!threshold || value === undefined) return 'unknown';

        if (value <= threshold.good) return 'good';
        if (value <= threshold.needsImprovement) return 'needs-improvement';
        return 'poor';
    }
}

// Usage example
const auditor = new FrontendPerformanceAuditor([
    'https://your-site.com',
    'https://your-site.com/products',
    'https://your-site.com/checkout'
], {
    device: '{{user_base}}' === 'mobile-first' ? 'mobile' : 'desktop',
    throttling: '{{target_metrics}}'.includes('slow-3g') ? 'slow3G' : 'fast3G',
    iterations: 5
});

// Run audit
auditor.runAudit().then(() => {
    const report = auditor.generateReport();
    console.log('Performance audit completed!');
    console.log('Report saved to:', `performance-audit-${Date.now()}.json`);
});
```

### Real User Monitoring (RUM) Setup
```javascript
// RUM implementation for Core Web Vitals tracking
class RealUserMonitoring {
    constructor(config = {}) {
        this.config = {
            sampleRate: config.sampleRate || 0.1, // 10% sampling
            endpoint: config.endpoint || '/api/rum-metrics',
            debug: config.debug || false,
            ...config
        };
        
        this.metrics = {};
        this.init();
    }

    init() {
        // Only sample a percentage of users
        if (Math.random() > this.config.sampleRate) return;

        this.observePerformanceEntries();
        this.setupCoreWebVitalsObserver();
        this.trackNavigationTiming();
        this.setupErrorTracking();
    }

    observePerformanceEntries() {
        // Observe paint timings
        const paintObserver = new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                if (entry.name === 'first-contentful-paint') {
                    this.metrics.fcp = entry.startTime;
                }
                if (entry.name === 'first-paint') {
                    this.metrics.fp = entry.startTime;
                }
            });
        });
        
        paintObserver.observe({ entryTypes: ['paint'] });

        // Observe largest contentful paint
        const lcpObserver = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const lastEntry = entries[entries.length - 1];
            this.metrics.lcp = lastEntry.startTime;
        });
        
        lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
    }

    setupCoreWebVitalsObserver() {
        // First Input Delay
        const fidObserver = new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                this.metrics.fid = entry.processingStart - entry.startTime;
            });
        });
        
        fidObserver.observe({ entryTypes: ['first-input'] });

        // Cumulative Layout Shift
        let clsValue = 0;
        let clsEntries = [];
        
        const clsObserver = new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                if (!entry.hadRecentInput) {
                    clsValue += entry.value;
                    clsEntries.push(entry);
                }
            });
            
            this.metrics.cls = clsValue;
        });
        
        clsObserver.observe({ entryTypes: ['layout-shift'] });
    }

    trackNavigationTiming() {
        window.addEventListener('load', () => {
            setTimeout(() => {
                const navigation = performance.getEntriesByType('navigation')[0];
                
                this.metrics.ttfb = navigation.responseStart - navigation.requestStart;
                this.metrics.domContentLoaded = navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart;
                this.metrics.loadComplete = navigation.loadEventEnd - navigation.loadEventStart;
                
                // Track resource timings
                this.trackResourceTimings();
                
                // Send metrics after page load
                this.sendMetrics();
            }, 0);
        });
    }

    trackResourceTimings() {
        const resources = performance.getEntriesByType('resource');
        
        const resourceMetrics = {
            totalResources: resources.length,
            totalSize: 0,
            imageCount: 0,
            scriptCount: 0,
            stylesheetCount: 0,
            slowResources: []
        };

        resources.forEach(resource => {
            // Categorize resources
            if (resource.initiatorType === 'img') resourceMetrics.imageCount++;
            if (resource.initiatorType === 'script') resourceMetrics.scriptCount++;
            if (resource.initiatorType === 'css') resourceMetrics.stylesheetCount++;

            // Track slow resources (>2 seconds)
            const duration = resource.responseEnd - resource.requestStart;
            if (duration > 2000) {
                resourceMetrics.slowResources.push({
                    name: resource.name,
                    duration: duration,
                    type: resource.initiatorType,
                    size: resource.transferSize || 0
                });
            }

            resourceMetrics.totalSize += resource.transferSize || 0;
        });

        this.metrics.resources = resourceMetrics;
    }

    setupErrorTracking() {
        // JavaScript errors
        window.addEventListener('error', (event) => {
            this.trackError({
                type: 'javascript',
                message: event.message,
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno,
                stack: event.error?.stack
            });
        });

        // Promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            this.trackError({
                type: 'promise_rejection',
                message: event.reason?.message || 'Unhandled Promise Rejection',
                stack: event.reason?.stack
            });
        });

        // Resource loading errors
        window.addEventListener('error', (event) => {
            if (event.target !== window) {
                this.trackError({
                    type: 'resource_error',
                    element: event.target.tagName,
                    source: event.target.src || event.target.href,
                    message: 'Failed to load resource'
                });
            }
        }, true);
    }

    trackError(errorData) {
        if (!this.metrics.errors) this.metrics.errors = [];
        
        this.metrics.errors.push({
            ...errorData,
            timestamp: Date.now(),
            url: window.location.href,
            userAgent: navigator.userAgent
        });

        // Send error immediately for critical issues
        this.sendMetrics('error');
    }

    sendMetrics(type = 'performance') {
        const payload = {
            type,
            metrics: this.metrics,
            page: {
                url: window.location.href,
                title: document.title,
                referrer: document.referrer
            },
            user: {
                userAgent: navigator.userAgent,
                language: navigator.language,
                cookieEnabled: navigator.cookieEnabled,
                onLine: navigator.onLine
            },
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight,
                devicePixelRatio: window.devicePixelRatio
            },
            connection: this.getConnectionInfo(),
            timestamp: Date.now()
        };

        // Send via beacon API for reliability
        if (navigator.sendBeacon) {
            navigator.sendBeacon(
                this.config.endpoint,
                JSON.stringify(payload)
            );
        } else {
            // Fallback to fetch
            fetch(this.config.endpoint, {
                method: 'POST',
                body: JSON.stringify(payload),
                headers: {
                    'Content-Type': 'application/json'
                },
                keepalive: true
            }).catch(error => {
                if (this.config.debug) {
                    console.error('Failed to send RUM data:', error);
                }
            });
        }
    }

    getConnectionInfo() {
        if (navigator.connection) {
            return {
                effectiveType: navigator.connection.effectiveType,
                downlink: navigator.connection.downlink,
                rtt: navigator.connection.rtt,
                saveData: navigator.connection.saveData
            };
        }
        return {};
    }
}

// Initialize RUM
if (typeof window !== 'undefined') {
    const rum = new RealUserMonitoring({
        sampleRate: {{user_base === 'enterprise' ? '1.0' : '0.1'}}, // 100% for enterprise, 10% for others
        endpoint: '/api/performance-metrics',
        debug: process.env.NODE_ENV === 'development'
    });
}
```

## 2. {{framework}} Specific Optimizations

{{#if (eq framework "react")}}
### React Performance Optimizations

#### Component Optimization Strategies
```javascript
// React performance optimization patterns
import React, { memo, useMemo, useCallback, lazy, Suspense } from 'react';
import { createPortal } from 'react-dom';

// 1. Memoization for expensive components
const ExpensiveComponent = memo(({ data, onUpdate }) => {
  const processedData = useMemo(() => {
    return data.map(item => ({
      ...item,
      computed: heavyComputation(item)
    }));
  }, [data]);

  const handleUpdate = useCallback((id, changes) => {
    onUpdate(id, changes);
  }, [onUpdate]);

  return (
    <div>
      {processedData.map(item => (
        <ItemComponent 
          key={item.id} 
          item={item} 
          onUpdate={handleUpdate}
        />
      ))}
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function
  return prevProps.data.length === nextProps.data.length &&
         prevProps.data.every((item, index) => 
           item.id === nextProps.data[index].id &&
           item.version === nextProps.data[index].version
         );
});

// 2. Code splitting with lazy loading
const LazyDashboard = lazy(() => 
  import('./Dashboard').then(module => ({
    default: module.Dashboard
  }))
);

const LazyReports = lazy(() => 
  import(/* webpackChunkName: "reports" */ './Reports')
);

// 3. Virtual scrolling for large lists
import { FixedSizeList as List } from 'react-window';

const VirtualizedList = ({ items }) => {
  const Row = useCallback(({ index, style }) => (
    <div style={style}>
      <ItemComponent item={items[index]} />
    </div>
  ), [items]);

  return (
    <List
      height={600}
      itemCount={items.length}
      itemSize={80}
      width="100%"
    >
      {Row}
    </List>
  );
};

// 4. Image optimization with lazy loading
const OptimizedImage = ({ src, alt, ...props }) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isInView, setIsInView] = useState(false);
  const imgRef = useRef();

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );

    if (imgRef.current) {
      observer.observe(imgRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <div ref={imgRef} {...props}>
      {isInView && (
        <img
          src={src}
          alt={alt}
          loading="lazy"
          onLoad={() => setIsLoaded(true)}
          style={{
            opacity: isLoaded ? 1 : 0,
            transition: 'opacity 0.3s'
          }}
        />
      )}
    </div>
  );
};

// 5. Performance monitoring hook
const usePerformanceMonitor = (componentName) => {
  useEffect(() => {
    const startTime = performance.now();
    
    return () => {
      const endTime = performance.now();
      const renderTime = endTime - startTime;
      
      if (renderTime > 16) { // Longer than one frame (60fps)
        console.warn(`Slow render in ${componentName}: ${renderTime.toFixed(2)}ms`);
      }
      
      // Send to analytics
      if (window.analytics) {
        window.analytics.track('Component Render Time', {
          component: componentName,
          renderTime: renderTime,
          timestamp: Date.now()
        });
      }
    };
  });
};

// Usage in components
const MyComponent = ({ data }) => {
  usePerformanceMonitor('MyComponent');
  
  return <div>{/* component content */}</div>;
};
```

#### Bundle Size Optimization
```javascript
// Webpack configuration for React optimization
module.exports = {
  optimization: {
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          priority: 10,
          enforce: true
        },
        react: {
          test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
          name: 'react',
          priority: 20,
          enforce: true
        },
        common: {
          minChunks: 2,
          priority: 5,
          reuseExistingChunk: true
        }
      }
    },
    usedExports: true,
    sideEffects: false
  },
  
  resolve: {
    alias: {
      // Use production builds
      'react': 'react/cjs/react.production.min.js',
      'react-dom': 'react-dom/cjs/react-dom.production.min.js'
    }
  },
  
  plugins: [
    // Tree shaking for lodash
    new webpack.optimize.ModuleConcatenationPlugin(),
    
    // Analyze bundle size
    new BundleAnalyzerPlugin({
      analyzerMode: 'static',
      generateStatsFile: true
    })
  ]
};

// Package.json optimizations
{
  "dependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "@babel/preset-env": "^7.0.0",
    "babel-plugin-transform-react-remove-prop-types": "^0.4.24"
  },
  "babel": {
    "presets": [
      ["@babel/preset-env", {
        "modules": false,
        "useBuiltIns": "usage",
        "corejs": 3
      }],
      "@babel/preset-react"
    ],
    "plugins": [
      "babel-plugin-transform-react-remove-prop-types"
    ]
  }
}
```
{{/if}}

{{#if (eq framework "nextjs")}}
### Next.js Performance Optimizations

#### Advanced Next.js Configuration
```javascript
// next.config.js optimizations
const nextConfig = {
  // Image optimization
  images: {
    domains: ['cdn.example.com', 'images.unsplash.com'],
    formats: ['image/webp', 'image/avif'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    minimumCacheTTL: 31536000, // 1 year
  },
  
  // Experimental features for performance
  experimental: {
    appDir: true,
    serverComponents: true,
    optimizeCss: true,
    scrollRestoration: true,
  },
  
  // Webpack optimizations
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Optimize bundle splitting
    if (!isServer) {
      config.optimization.splitChunks = {
        chunks: 'all',
        cacheGroups: {
          default: false,
          vendors: false,
          vendor: {
            name: 'vendor',
            chunks: 'all',
            test: /node_modules/,
            priority: 20
          },
          common: {
            name: 'common',
            minChunks: 2,
            chunks: 'all',
            priority: 10,
            reuseExistingChunk: true,
            enforce: true
          }
        }
      };
    }
    
    // Add performance budgets
    config.performance = {
      maxAssetSize: 250000,
      maxEntrypointSize: 400000,
      hints: 'warning'
    };
    
    return config;
  },
  
  // Headers for caching and security
  async headers() {
    return [
      {
        source: '/_next/static/(.*)',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable'
          }
        ]
      },
      {
        source: '/images/(.*)',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000'
          }
        ]
      }
    ];
  },
  
  // Compression
  compress: true,
  
  // Power pack optimizations
  poweredByHeader: false,
  generateEtags: false,
  
  // Enable SWC minification
  swcMinify: true,
  
  // Bundle analyzer
  ...(process.env.ANALYZE === 'true' && {
    webpack: (config) => {
      config.plugins.push(
        new (require('@next/bundle-analyzer'))({
          enabled: true
        })
      );
      return config;
    }
  })
};

module.exports = nextConfig;

// Advanced page optimization
import { GetStaticProps, GetStaticPaths } from 'next';
import dynamic from 'next/dynamic';
import Image from 'next/image';
import Head from 'next/head';

// Dynamic imports with loading states
const DynamicChart = dynamic(
  () => import('../components/Chart'),
  {
    loading: () => <ChartSkeleton />,
    ssr: false // Client-side only for heavy components
  }
);

const DynamicModal = dynamic(
  () => import('../components/Modal'),
  { ssr: false }
);

// Optimized page component
export default function ProductPage({ product, relatedProducts }) {
  return (
    <>
      <Head>
        <title>{product.name} | Your Store</title>
        <meta name="description" content={product.description} />
        
        {/* Preload critical resources */}
        <link rel="preload" href="/fonts/inter.woff2" as="font" type="font/woff2" crossOrigin="" />
        <link rel="preload" href={product.image} as="image" />
        
        {/* Prefetch related pages */}
        <link rel="prefetch" href="/checkout" />
        
        {/* Structured data for SEO */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org/",
              "@type": "Product",
              "name": product.name,
              "description": product.description,
              "offers": {
                "@type": "Offer",
                "price": product.price,
                "priceCurrency": "USD"
              }
            })
          }}
        />
      </Head>
      
      <main>
        {/* Optimized hero image */}
        <Image
          src={product.image}
          alt={product.name}
          width={800}
          height={600}
          priority // Above-the-fold image
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          placeholder="blur"
          blurDataURL="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
        />
        
        {/* Lazy-loaded components */}
        <Suspense fallback={<ReviewsSkeleton />}>
          <Reviews productId={product.id} />
        </Suspense>
        
        {/* Client-side only components */}
        <DynamicChart data={product.analytics} />
      </main>
    </>
  );
}

// Static generation with ISR
export const getStaticProps: GetStaticProps = async ({ params }) => {
  const product = await fetchProduct(params.id);
  const relatedProducts = await fetchRelatedProducts(product.category);
  
  return {
    props: {
      product,
      relatedProducts
    },
    revalidate: 3600 // Revalidate every hour
  };
};

export const getStaticPaths: GetStaticPaths = async () => {
  // Generate paths for top products only
  const topProducts = await fetchTopProducts(100);
  
  const paths = topProducts.map(product => ({
    params: { id: product.id }
  }));
  
  return {
    paths,
    fallback: 'blocking' // Generate other pages on-demand
  };
};
```
{{/if}}

## 3. Core Web Vitals Optimization

### Largest Contentful Paint (LCP) Optimization
```javascript
// LCP optimization strategies
class LCPOptimizer {
    constructor() {
        this.lcpElement = null;
        this.observer = null;
        this.initLCPTracking();
    }

    initLCPTracking() {
        // Track LCP element
        this.observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const lastEntry = entries[entries.length - 1];
            
            this.lcpElement = lastEntry.element;
            this.optimizeLCPElement();
        });
        
        this.observer.observe({ entryTypes: ['largest-contentful-paint'] });
    }

    optimizeLCPElement() {
        if (!this.lcpElement) return;

        // 1. Preload LCP image if it's an image
        if (this.lcpElement.tagName === 'IMG') {
            this.preloadLCPImage();
        }

        // 2. Optimize text rendering
        if (this.lcpElement.textContent) {
            this.optimizeTextRendering();
        }

        // 3. Critical CSS inlining
        this.inlineCriticalCSS();
    }

    preloadLCPImage() {
        const imgSrc = this.lcpElement.src || this.lcpElement.dataset.src;
        
        if (imgSrc && !document.querySelector(`link[rel="preload"][href="${imgSrc}"]`)) {
            const preloadLink = document.createElement('link');
            preloadLink.rel = 'preload';
            preloadLink.as = 'image';
            preloadLink.href = imgSrc;
            
            // Add responsive preload hints
            if (this.lcpElement.srcset) {
                preloadLink.imageSrcset = this.lcpElement.srcset;
                preloadLink.imageSizes = this.lcpElement.sizes || '100vw';
            }
            
            document.head.appendChild(preloadLink);
        }
    }

    optimizeTextRendering() {
        // Ensure font is preloaded and displayed
        const computedStyle = window.getComputedStyle(this.lcpElement);
        const fontFamily = computedStyle.fontFamily;
        
        // Add font-display: swap to prevent invisible text
        if (!document.querySelector('style[data-lcp-font-optimization]')) {
            const style = document.createElement('style');
            style.setAttribute('data-lcp-font-optimization', 'true');
            style.textContent = `
                @font-face {
                    font-family: ${fontFamily};
                    font-display: swap;
                }
            `;
            document.head.appendChild(style);
        }
    }

    inlineCriticalCSS() {
        // Extract and inline critical CSS for LCP element
        const criticalCSS = this.extractCriticalCSS(this.lcpElement);
        
        if (criticalCSS && !document.querySelector('style[data-critical-css]')) {
            const style = document.createElement('style');
            style.setAttribute('data-critical-css', 'true');
            style.textContent = criticalCSS;
            document.head.insertBefore(style, document.head.firstChild);
        }
    }

    extractCriticalCSS(element) {
        // Simplified critical CSS extraction
        const computedStyle = window.getComputedStyle(element);
        const criticalProperties = [
            'display', 'position', 'width', 'height', 'margin', 'padding',
            'color', 'background', 'font-size', 'font-weight', 'line-height'
        ];

        let css = '';
        const selector = this.generateSelector(element);
        
        criticalProperties.forEach(prop => {
            const value = computedStyle.getPropertyValue(prop);
            if (value && value !== 'initial') {
                css += `${prop}: ${value}; `;
            }
        });

        return css ? `${selector} { ${css} }` : '';
    }

    generateSelector(element) {
        if (element.id) return `#${element.id}`;
        if (element.className) return `.${element.className.split(' ')[0]}`;
        return element.tagName.toLowerCase();
    }
}

// Initialize LCP optimization
const lcpOptimizer = new LCPOptimizer();
```

### Cumulative Layout Shift (CLS) Prevention
```javascript
// CLS prevention strategies
class CLSPrevention {
    constructor() {
        this.layoutShiftTracker = [];
        this.initCLSTracking();
        this.implementPreventionStrategies();
    }

    initCLSTracking() {
        const observer = new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                if (!entry.hadRecentInput) {
                    this.layoutShiftTracker.push({
                        value: entry.value,
                        sources: entry.sources,
                        timestamp: entry.startTime
                    });
                    
                    if (entry.value > 0.1) {
                        this.handleSignificantShift(entry);
                    }
                }
            });
        });
        
        observer.observe({ entryTypes: ['layout-shift'] });
    }

    implementPreventionStrategies() {
        // 1. Reserve space for images
        this.reserveImageSpace();
        
        // 2. Handle dynamic content
        this.handleDynamicContent();
        
        // 3. Prevent font-related shifts
        this.preventFontShifts();
        
        // 4. Stabilize ad placements
        this.stabilizeAds();
    }

    reserveImageSpace() {
        const images = document.querySelectorAll('img:not([width]):not([height])');
        
        images.forEach(img => {
            if (!img.style.aspectRatio && !img.style.height) {
                // Set aspect ratio to prevent layout shift
                img.addEventListener('load', () => {
                    const aspectRatio = img.naturalWidth / img.naturalHeight;
                    img.style.aspectRatio = aspectRatio;
                }, { once: true });
                
                // Placeholder dimensions
                img.style.minHeight = '200px';
                img.style.backgroundColor = '#f0f0f0';
            }
        });
    }

    handleDynamicContent() {
        // Use Intersection Observer for dynamic content loading
        const dynamicElements = document.querySelectorAll('[data-dynamic-content]');
        
        dynamicElements.forEach(element => {
            // Reserve space before loading content
            const placeholder = document.createElement('div');
            placeholder.style.height = element.dataset.expectedHeight || '200px';
            placeholder.className = 'content-placeholder';
            
            element.appendChild(placeholder);
            
            // Load content with smooth replacement
            this.loadDynamicContent(element).then(content => {
                placeholder.style.transition = 'opacity 0.3s';
                placeholder.style.opacity = '0';
                
                setTimeout(() => {
                    element.innerHTML = content;
                }, 300);
            });
        });
    }

    preventFontShifts() {
        // Use font-display: swap and size-adjust
        const fontFaces = document.styleSheets;
        
        Array.from(fontFaces).forEach(sheet => {
            try {
                Array.from(sheet.cssRules).forEach(rule => {
                    if (rule instanceof CSSFontFaceRule) {
                        if (!rule.style.fontDisplay) {
                            rule.style.fontDisplay = 'swap';
                        }
                    }
                });
            } catch (e) {
                // Cross-origin stylesheets
            }
        });
        
        // Preload critical fonts
        const criticalFonts = [
            '/fonts/inter-regular.woff2',
            '/fonts/inter-bold.woff2'
        ];
        
        criticalFonts.forEach(fontUrl => {
            if (!document.querySelector(`link[href="${fontUrl}"]`)) {
                const link = document.createElement('link');
                link.rel = 'preload';
                link.as = 'font';
                link.type = 'font/woff2';
                link.crossOrigin = 'anonymous';
                link.href = fontUrl;
                document.head.appendChild(link);
            }
        });
    }

    stabilizeAds() {
        // Reserve space for ad slots
        const adSlots = document.querySelectorAll('[data-ad-slot]');
        
        adSlots.forEach(slot => {
            const width = slot.dataset.adWidth || '300';
            const height = slot.dataset.adHeight || '250';
            
            slot.style.width = `${width}px`;
            slot.style.height = `${height}px`;
            slot.style.backgroundColor = '#f8f8f8';
            slot.style.border = '1px solid #e0e0e0';
            slot.style.display = 'flex';
            slot.style.alignItems = 'center';
            slot.style.justifyContent = 'center';
            slot.style.fontSize = '12px';
            slot.style.color = '#666';
            slot.textContent = 'Advertisement';
        });
    }

    handleSignificantShift(entry) {
        console.warn('Significant layout shift detected:', {
            value: entry.value,
            sources: entry.sources.map(source => ({
                node: source.node,
                previousRect: source.previousRect,
                currentRect: source.currentRect
            }))
        });
        
        // Send to analytics
        if (window.analytics) {
            window.analytics.track('Layout Shift', {
                value: entry.value,
                timestamp: entry.startTime,
                url: window.location.href
            });
        }
    }
}

// Initialize CLS prevention
const clsPrevention = new CLSPrevention();
```

## 4. Advanced Optimization Techniques

### Resource Loading Optimization
```javascript
// Advanced resource loading strategies
class ResourceLoadingOptimizer {
    constructor() {
        this.criticalResources = new Set();
        this.deferredResources = new Set();
        this.init();
    }

    init() {
        this.identifyCriticalResources();
        this.optimizeResourceLoading();
        this.implementServiceWorker();
        this.setupResourceHints();
    }

    identifyCriticalResources() {
        // Identify above-the-fold content
        const foldHeight = window.innerHeight;
        const criticalElements = [];
        
        document.querySelectorAll('img, video, iframe').forEach(element => {
            const rect = element.getBoundingClientRect();
            if (rect.top < foldHeight) {
                criticalElements.push(element);
                this.criticalResources.add(element.src);
            }
        });
        
        // Mark critical CSS
        document.querySelectorAll('link[rel="stylesheet"]').forEach((link, index) => {
            if (index < 2) { // First 2 stylesheets are usually critical
                this.criticalResources.add(link.href);
            }
        });
    }

    optimizeResourceLoading() {
        // Lazy load non-critical images
        this.implementLazyLoading();
        
        // Prefetch likely navigation targets
        this.prefetchLikelyPages();
        
        // Optimize font loading
        this.optimizeFontLoading();
        
        // Defer non-critical JavaScript
        this.deferNonCriticalJS();
    }

    implementLazyLoading() {
        const images = document.querySelectorAll('img[data-src]');
        
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    
                    // Load image with fade-in effect
                    img.src = img.dataset.src;
                    img.onload = () => {
                        img.style.opacity = '1';
                        img.style.transition = 'opacity 0.3s';
                    };
                    
                    observer.unobserve(img);
                }
            });
        }, {
            rootMargin: '50px 0px', // Start loading 50px before entering viewport
            threshold: 0.01
        });
        
        images.forEach(img => imageObserver.observe(img));
    }

    prefetchLikelyPages() {
        // Prefetch pages based on user behavior
        const prefetchCandidates = [
            '/products',
            '/about',
            '/contact'
        ];
        
        // Prefetch on hover with delay
        document.querySelectorAll('a[href]').forEach(link => {
            let prefetchTimeout;
            
            link.addEventListener('mouseenter', () => {
                prefetchTimeout = setTimeout(() => {
                    this.prefetchPage(link.href);
                }, 200);
            });
            
            link.addEventListener('mouseleave', () => {
                clearTimeout(prefetchTimeout);
            });
        });
    }

    prefetchPage(url) {
        if (this.deferredResources.has(url)) return;
        
        const link = document.createElement('link');
        link.rel = 'prefetch';
        link.href = url;
        document.head.appendChild(link);
        
        this.deferredResources.add(url);
    }

    optimizeFontLoading() {
        // Implement font loading strategy
        const fontLoadPromises = [];
        
        const criticalFonts = [
            { family: 'Inter', weight: '400', display: 'swap' },
            { family: 'Inter', weight: '700', display: 'swap' }
        ];
        
        criticalFonts.forEach(font => {
            if ('fonts' in document) {
                const fontFace = new FontFace(
                    font.family,
                    `url(/fonts/${font.family.toLowerCase()}-${font.weight}.woff2)`,
                    { weight: font.weight, display: font.display }
                );
                
                fontLoadPromises.push(
                    fontFace.load().then(loadedFont => {
                        document.fonts.add(loadedFont);
                    })
                );
            }
        });
        
        Promise.all(fontLoadPromises).then(() => {
            document.body.classList.add('fonts-loaded');
        });
    }

    deferNonCriticalJS() {
        // Defer non-critical scripts until after page load
        const nonCriticalScripts = [
            '/js/analytics.js',
            '/js/social-widgets.js',
            '/js/chat-widget.js'
        ];
        
        window.addEventListener('load', () => {
            setTimeout(() => {
                nonCriticalScripts.forEach(src => {
                    const script = document.createElement('script');
                    script.src = src;
                    script.async = true;
                    document.body.appendChild(script);
                });
            }, 1000); // Delay by 1 second after load
        });
    }

    implementServiceWorker() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js').then(registration => {
                console.log('Service Worker registered:', registration);
            }).catch(error => {
                console.log('Service Worker registration failed:', error);
            });
        }
    }

    setupResourceHints() {
        // DNS prefetch for external domains
        const externalDomains = [
            'fonts.googleapis.com',
            'fonts.gstatic.com',
            'cdn.example.com'
        ];
        
        externalDomains.forEach(domain => {
            const link = document.createElement('link');
            link.rel = 'dns-prefetch';
            link.href = `//${domain}`;
            document.head.appendChild(link);
        });
        
        // Preconnect to critical third-party origins
        const criticalOrigins = [
            'https://fonts.googleapis.com',
            'https://api.analytics.com'
        ];
        
        criticalOrigins.forEach(origin => {
            const link = document.createElement('link');
            link.rel = 'preconnect';
            link.href = origin;
            link.crossOrigin = 'anonymous';
            document.head.appendChild(link);
        });
    }
}

// Initialize resource loading optimization
const resourceOptimizer = new ResourceLoadingOptimizer();
```

## 5. Performance Monitoring & Alerting

### Continuous Performance Monitoring
```javascript
// Performance monitoring and alerting system
class PerformanceMonitor {
    constructor(config = {}) {
        this.config = {
            alertThresholds: {
                lcp: 2500,      // LCP threshold in ms
                fid: 100,       // FID threshold in ms
                cls: 0.1,       // CLS threshold
                ttfb: 600,      // TTFB threshold in ms
                ...config.alertThresholds
            },
            reportingEndpoint: config.reportingEndpoint || '/api/performance-reports',
            alertEndpoint: config.alertEndpoint || '/api/performance-alerts',
            ...config
        };
        
        this.metrics = {};
        this.alerts = [];
        this.init();
    }

    init() {
        this.setupPerformanceObservers();
        this.trackPageLoad();
        this.monitorResourceTiming();
        this.setupPeriodicReporting();
    }

    setupPerformanceObservers() {
        // Core Web Vitals monitoring
        this.observeCoreWebVitals();
        
        // Long tasks monitoring
        this.observeLongTasks();
        
        // Memory usage monitoring
        this.observeMemoryUsage();
    }

    observeCoreWebVitals() {
        // LCP Observer
        new PerformanceObserver((list) => {
            const entries = list.getEntries();
            const lastEntry = entries[entries.length - 1];
            
            this.metrics.lcp = lastEntry.startTime;
            this.checkThreshold('lcp', lastEntry.startTime);
        }).observe({ entryTypes: ['largest-contentful-paint'] });

        // FID Observer
        new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                const fid = entry.processingStart - entry.startTime;
                this.metrics.fid = fid;
                this.checkThreshold('fid', fid);
            });
        }).observe({ entryTypes: ['first-input'] });

        // CLS Observer
        let clsValue = 0;
        new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                if (!entry.hadRecentInput) {
                    clsValue += entry.value;
                    this.metrics.cls = clsValue;
                    this.checkThreshold('cls', clsValue);
                }
            });
        }).observe({ entryTypes: ['layout-shift'] });
    }

    observeLongTasks() {
        new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                const taskDuration = entry.duration;
                
                if (taskDuration > 50) { // Tasks longer than 50ms
                    this.createAlert('long_task', {
                        duration: taskDuration,
                        startTime: entry.startTime,
                        attribution: entry.attribution
                    });
                }
            });
        }).observe({ entryTypes: ['longtask'] });
    }

    observeMemoryUsage() {
        if ('memory' in performance) {
            setInterval(() => {
                const memInfo = performance.memory;
                const memoryUsage = memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit;
                
                if (memoryUsage > 0.8) { // 80% memory usage
                    this.createAlert('high_memory_usage', {
                        usagePercentage: memoryUsage * 100,
                        usedMemory: memInfo.usedJSHeapSize,
                        totalMemory: memInfo.jsHeapSizeLimit
                    });
                }
            }, 30000); // Check every 30 seconds
        }
    }

    checkThreshold(metric, value) {
        const threshold = this.config.alertThresholds[metric];
        
        if (threshold && value > threshold) {
            this.createAlert('threshold_exceeded', {
                metric,
                value,
                threshold,
                severity: this.calculateSeverity(metric, value, threshold)
            });
        }
    }

    calculateSeverity(metric, value, threshold) {
        const ratio = value / threshold;
        
        if (ratio > 2) return 'critical';
        if (ratio > 1.5) return 'high';
        if (ratio > 1.2) return 'medium';
        return 'low';
    }

    createAlert(type, data) {
        const alert = {
            id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            type,
            data,
            timestamp: Date.now(),
            url: window.location.href,
            userAgent: navigator.userAgent
        };
        
        this.alerts.push(alert);
        this.sendAlert(alert);
    }

    sendAlert(alert) {
        fetch(this.config.alertEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(alert)
        }).catch(error => {
            console.error('Failed to send performance alert:', error);
        });
    }

    generatePerformanceReport() {
        const report = {
            timestamp: Date.now(),
            url: window.location.href,
            metrics: this.metrics,
            alerts: this.alerts,
            browser: {
                userAgent: navigator.userAgent,
                language: navigator.language,
                cookieEnabled: navigator.cookieEnabled,
                onLine: navigator.onLine
            },
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight,
                devicePixelRatio: window.devicePixelRatio
            },
            connection: navigator.connection ? {
                effectiveType: navigator.connection.effectiveType,
                downlink: navigator.connection.downlink,
                rtt: navigator.connection.rtt
            } : null,
            resourceTiming: this.getResourceTimingSummary()
        };
        
        return report;
    }

    getResourceTimingSummary() {
        const resources = performance.getEntriesByType('resource');
        
        return {
            totalResources: resources.length,
            totalSize: resources.reduce((sum, r) => sum + (r.transferSize || 0), 0),
            slowResources: resources
                .filter(r => r.duration > 1000)
                .map(r => ({
                    name: r.name,
                    duration: r.duration,
                    size: r.transferSize
                })),
            cacheHitRatio: this.calculateCacheHitRatio(resources)
        };
    }

    calculateCacheHitRatio(resources) {
        const cachedResources = resources.filter(r => r.transferSize === 0);
        return cachedResources.length / resources.length;
    }

    setupPeriodicReporting() {
        // Send report every 5 minutes
        setInterval(() => {
            const report = this.generatePerformanceReport();
            this.sendReport(report);
        }, 300000);
        
        // Send report on page unload
        window.addEventListener('beforeunload', () => {
            const report = this.generatePerformanceReport();
            navigator.sendBeacon(
                this.config.reportingEndpoint,
                JSON.stringify(report)
            );
        });
    }

    sendReport(report) {
        fetch(this.config.reportingEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(report)
        }).catch(error => {
            console.error('Failed to send performance report:', error);
        });
    }
}

// Initialize performance monitoring
const performanceMonitor = new PerformanceMonitor({
    alertThresholds: {
        {{#if (eq target_metrics "lighthouse-90+")}}
        lcp: 2000,
        fid: 80,
        cls: 0.08,
        ttfb: 500
        {{else if (eq target_metrics "core-web-vitals-good")}}
        lcp: 2500,
        fid: 100,
        cls: 0.1,
        ttfb: 600
        {{else}}
        lcp: 3000,
        fid: 150,
        cls: 0.15,
        ttfb: 800
        {{/if}}
    },
    reportingEndpoint: '/api/performance-metrics',
    alertEndpoint: '/api/performance-alerts'
});
```

## Conclusion

This frontend performance audit framework provides:

**Key Features:**
- Comprehensive {{framework}} optimization strategies
- Core Web Vitals monitoring and improvement
- Advanced resource loading optimization
- Real-time performance monitoring and alerting
- {{business_impact}} focused improvements

**Expected Outcomes:**
- {{#if (eq target_metrics "lighthouse-90+")}}Lighthouse score improvement to 90+{{/if}}
- {{#if (includes target_metrics "core-web-vitals")}}All Core Web Vitals in "Good" range{{/if}}
- {{#if (includes business_impact "conversion")}}Improved conversion rates through faster loading{{/if}}
- {{#if (includes business_impact "seo")}}Better search engine rankings{{/if}}
- Enhanced user experience for {{user_base}} users

**Implementation Timeline:**
- Week 1-2: Audit and baseline establishment
- Week 3-4: Critical optimizations implementation
- Week 5-6: Advanced optimizations and monitoring setup
- Week 7+: Continuous monitoring and iterative improvements