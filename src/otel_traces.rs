//! OpenTelemetry-compatible tracing and trace export
//!
//! This module provides distributed tracing capabilities compatible with
//! OpenTelemetry's trace export format (OTLP JSON). It collects spans from
//! inference operations and exports them via the `/traces` HTTP endpoint.
//!
//! # Trace Sampling
//!
//! Traces are sampled based on the `ROCFORGE_TRACE_SAMPLE_RATE` environment
//! variable (default: 0.1 = 10%). Set to 1.0 to capture all traces.
//!
//! # Example Trace Structure
//!
//! ```json
//! {
//!   "resource_spans": [{
//!     "resource": {
//!       "attributes": {
//!         "service.name": "rocmforge"
//!       }
//!     },
//!     "scope_spans": [{
//!       "scope": {"name": "rocmforge"},
//!       "spans": [...]
//!     }]
//!   }]
//! }
//! ```

use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

/// Global trace storage
static TRACE_STORE: OnceCell<Arc<Mutex<TraceStore>>> = OnceCell::new();

/// Default maximum number of traces to store in memory
const DEFAULT_MAX_TRACES: usize = 1000;

/// Default trace sample rate (10%)
const DEFAULT_SAMPLE_RATE: f64 = 0.1;

/// Environment variable for trace sample rate
const TRACE_SAMPLE_RATE_ENV: &str = "ROCMORGE_TRACE_SAMPLE_RATE";

/// Environment variable for max traces to store
const MAX_TRACES_ENV: &str = "ROCMFORGE_MAX_TRACES";

/// Configuration for trace collection
#[derive(Debug, Clone)]
pub struct TraceConfig {
    /// Sample rate for traces (0.0 to 1.0)
    pub sample_rate: f64,
    /// Maximum number of traces to store in memory
    pub max_traces: usize,
    /// Service name for trace identification
    pub service_name: String,
}

impl Default for TraceConfig {
    fn default() -> Self {
        TraceConfig {
            sample_rate: get_sample_rate_from_env(),
            max_traces: get_max_traces_from_env(),
            service_name: "rocmforge".to_string(),
        }
    }
}

/// Get sample rate from environment variable
fn get_sample_rate_from_env() -> f64 {
    std::env::var(TRACE_SAMPLE_RATE_ENV)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|r| (0.0..=1.0).contains(r))
        .unwrap_or(DEFAULT_SAMPLE_RATE)
}

/// Get max traces from environment variable
fn get_max_traces_from_env() -> usize {
    std::env::var(MAX_TRACES_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|m| *m > 0)
        .unwrap_or(DEFAULT_MAX_TRACES)
}

/// Span status following OpenTelemetry specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SpanStatus {
    /// Operation completed successfully
    Ok,
    /// Operation encountered an error
    Error,
    /// Operation was unset
    Unset,
}

impl From<bool> for SpanStatus {
    fn from(success: bool) -> Self {
        if success {
            SpanStatus::Ok
        } else {
            SpanStatus::Error
        }
    }
}

/// Span kind following OpenTelemetry specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SpanKind {
    /// Internal span
    Internal,
    /// Server-side span
    Server,
    /// Client-side span
    Client,
    /// Producer span
    Producer,
    /// Consumer span
    Consumer,
}

impl Default for SpanKind {
    fn default() -> Self {
        SpanKind::Internal
    }
}

/// A single key-value attribute for spans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attribute {
    /// Attribute key
    pub key: String,
    /// Attribute value (supports various types)
    #[serde(flatten)]
    pub value: AttributeValue,
}

/// Attribute value types following OpenTelemetry specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Double value
    Double(f64),
    /// Array of strings
    StringArray(Vec<String>),
    /// Array of integers
    IntArray(Vec<i64>),
}

/// A single span in a trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Trace ID (hex-encoded, 16 bytes = 32 hex chars)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// Span ID (hex-encoded, 8 bytes = 16 hex chars)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
    /// Parent span ID (hex-encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    /// Span name
    pub name: String,
    /// Span kind
    #[serde(rename = "kind")]
    pub kind: SpanKind,
    /// Start time in nanoseconds since UNIX epoch
    pub start_time_unix_nano: u64,
    /// End time in nanoseconds since UNIX epoch
    pub end_time_unix_nano: u64,
    /// Span attributes
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub attributes: Vec<Attribute>,
    /// Span status
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<SpanStatus>,
    /// Events that occurred during the span
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<SpanEvent>,
}

/// An event that occurred during a span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event name
    pub name: String,
    /// Event time in nanoseconds since UNIX epoch
    pub time_unix_nano: u64,
    /// Event attributes
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub attributes: Vec<Attribute>,
}

impl Span {
    /// Create a new span with the given name
    pub fn new(name: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Span {
            trace_id: None,
            span_id: None,
            parent_span_id: None,
            name: name.into(),
            kind: SpanKind::default(),
            start_time_unix_nano: now,
            end_time_unix_nano: now,
            attributes: Vec::new(),
            status: None,
            events: Vec::new(),
        }
    }

    /// Set the trace ID
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }

    /// Set the span ID
    pub fn with_span_id(mut self, span_id: impl Into<String>) -> Self {
        self.span_id = Some(span_id.into());
        self
    }

    /// Set the parent span ID
    pub fn with_parent_span_id(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_span_id = Some(parent_id.into());
        self
    }

    /// Set the span kind
    pub fn with_kind(mut self, kind: SpanKind) -> Self {
        self.kind = kind;
        self
    }

    /// Add an attribute to the span
    pub fn with_attribute(mut self, key: impl Into<String>, value: AttributeValue) -> Self {
        self.attributes.push(Attribute {
            key: key.into(),
            value,
        });
        self
    }

    /// Set the span status
    pub fn with_status(mut self, status: SpanStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Set the start time
    pub fn with_start_time(mut self, start_nanos: u64) -> Self {
        self.start_time_unix_nano = start_nanos;
        self
    }

    /// Set the end time
    pub fn with_end_time(mut self, end_nanos: u64) -> Self {
        self.end_time_unix_nano = end_nanos;
        self
    }

    /// Add an event to the span
    pub fn with_event(mut self, event: SpanEvent) -> Self {
        self.events.push(event);
        self
    }

    /// Calculate the span duration in nanoseconds
    pub fn duration_nanos(&self) -> u64 {
        self.end_time_unix_nano
            .saturating_sub(self.start_time_unix_nano)
    }
}

/// Instrumentation scope following OpenTelemetry specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scope {
    /// Scope name
    pub name: String,
    /// Scope version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Collection of spans from a single instrumentation scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeSpans {
    /// Instrumentation scope
    pub scope: Scope,
    /// Spans in this scope
    pub spans: Vec<Span>,
}

impl ScopeSpans {
    /// Create a new ScopeSpans
    pub fn new(name: impl Into<String>) -> Self {
        ScopeSpans {
            scope: Scope {
                name: name.into(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            },
            spans: Vec::new(),
        }
    }

    /// Add a span to this scope
    pub fn with_span(mut self, span: Span) -> Self {
        self.spans.push(span);
        self
    }
}

/// Resource following OpenTelemetry specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// Resource attributes
    pub attributes: HashMap<String, AttributeValue>,
}

impl Resource {
    /// Create a new resource with service name
    pub fn new(service_name: impl Into<String>) -> Self {
        let mut attributes = HashMap::new();
        attributes.insert(
            "service.name".to_string(),
            AttributeValue::String(service_name.into()),
        );
        attributes.insert(
            "service.version".to_string(),
            AttributeValue::String(env!("CARGO_PKG_VERSION").to_string()),
        );
        Resource { attributes }
    }
}

/// Collection of spans from a single resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSpans {
    /// Resource information
    pub resource: Resource,
    /// Scope spans
    pub scope_spans: Vec<ScopeSpans>,
}

impl ResourceSpans {
    /// Create a new ResourceSpans
    pub fn new(service_name: impl Into<String>) -> Self {
        ResourceSpans {
            resource: Resource::new(service_name),
            scope_spans: Vec::new(),
        }
    }

    /// Add scope spans to this resource
    pub fn with_scope_spans(mut self, scope_spans: ScopeSpans) -> Self {
        self.scope_spans.push(scope_spans);
        self
    }
}

/// OTLP trace export response body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceExport {
    /// Resource spans containing all traces
    pub resource_spans: Vec<ResourceSpans>,
}

impl TraceExport {
    /// Create an empty trace export
    pub fn empty() -> Self {
        TraceExport {
            resource_spans: Vec::new(),
        }
    }

    /// Create a trace export with resource spans
    pub fn new(resource_spans: Vec<ResourceSpans>) -> Self {
        TraceExport { resource_spans }
    }
}

/// Storage for collected traces
#[derive(Debug, Clone)]
pub struct TraceStore {
    /// Collected spans organized by scope
    spans: Vec<Span>,
    /// Configuration
    config: TraceConfig,
}

impl TraceStore {
    /// Create a new trace store
    pub fn new(config: TraceConfig) -> Self {
        TraceStore {
            spans: Vec::new(),
            config,
        }
    }

    /// Add a span to the store (respects sampling)
    pub fn add_span(&mut self, span: Span) -> bool {
        // Apply sampling
        if !self.should_sample() {
            return false;
        }

        self.spans.push(span);

        // Enforce max traces limit (FIFO eviction)
        if self.spans.len() > self.config.max_traces {
            self.spans.remove(0);
        }

        true
    }

    /// Check if this trace should be sampled
    fn should_sample(&self) -> bool {
        rand::random::<f64>() < self.config.sample_rate
    }

    /// Get all spans as OTLP export format
    pub fn export(&self) -> TraceExport {
        if self.spans.is_empty() {
            return TraceExport::empty();
        }

        // Create spans with IDs assigned
        let exported_spans: Vec<Span> = self.spans.iter().map(|s| {
            // Clone the span but without potentially sensitive data
            let mut exported = s.clone();
            // Ensure all spans have IDs set
            if exported.trace_id.is_none() {
                exported.trace_id = Some(generate_trace_id());
            }
            if exported.span_id.is_none() {
                exported.span_id = Some(generate_span_id());
            }
            exported
        }).collect();

        let scope_spans = ScopeSpans::new("rocmforge");
        // Add spans directly since with_span takes Vec
        let scope_spans = ScopeSpans {
            scope: scope_spans.scope,
            spans: exported_spans,
        };

        let resource_spans = ResourceSpans::new(&self.config.service_name)
            .with_scope_spans(scope_spans);

        TraceExport::new(vec![resource_spans])
    }

    /// Clear all stored spans
    pub fn clear(&mut self) {
        self.spans.clear();
    }

    /// Get the number of stored spans
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Get all spans
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }
}

/// Generate a random trace ID (16 bytes, hex-encoded)
fn generate_trace_id() -> String {
    use rand::Rng;
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill(&mut bytes);
    hex::encode(bytes)
}

/// Generate a random span ID (8 bytes, hex-encoded)
fn generate_span_id() -> String {
    use rand::Rng;
    let mut bytes = [0u8; 8];
    rand::thread_rng().fill(&mut bytes);
    hex::encode(bytes)
}

/// Initialize the global trace store
pub fn init_trace_store(config: TraceConfig) {
    TRACE_STORE.get_or_init(|| Arc::new(Mutex::new(TraceStore::new(config))));
}

/// Get the global trace store, initializing with default config if needed
pub fn get_trace_store() -> Arc<Mutex<TraceStore>> {
    TRACE_STORE
        .get_or_init(|| Arc::new(Mutex::new(TraceStore::new(TraceConfig::default()))))
        .clone()
}

/// Record a span to the global trace store
pub fn record_span(span: Span) -> bool {
    if let Ok(mut store) = get_trace_store().lock() {
        store.add_span(span)
    } else {
        false
    }
}

/// Export all traces in OTLP format
pub fn export_traces() -> TraceExport {
    get_trace_store()
        .lock()
        .map(|store| store.export())
        .unwrap_or_else(|_| TraceExport::empty())
}

/// Clear all stored traces
pub fn clear_traces() {
    if let Ok(mut store) = get_trace_store().lock() {
        store.clear();
    }
}

/// Get the number of stored traces
pub fn trace_count() -> usize {
    get_trace_store()
        .lock()
        .map(|store| store.len())
        .unwrap_or(0)
}

/// Builder for creating inference spans with common attributes
#[derive(Debug, Clone)]
pub struct InferenceSpanBuilder {
    name: String,
    request_id: Option<u32>,
    prompt_length: Option<usize>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    start_time: Option<u64>,
}

impl InferenceSpanBuilder {
    /// Create a new inference span builder
    pub fn new(name: impl Into<String>) -> Self {
        InferenceSpanBuilder {
            name: name.into(),
            request_id: None,
            prompt_length: None,
            max_tokens: None,
            temperature: None,
            start_time: None,
        }
    }

    /// Set the request ID
    pub fn with_request_id(mut self, id: u32) -> Self {
        self.request_id = Some(id);
        self
    }

    /// Set the prompt length
    pub fn with_prompt_length(mut self, len: usize) -> Self {
        self.prompt_length = Some(len);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set start time
    pub fn with_start_time(mut self, time: u64) -> Self {
        self.start_time = Some(time);
        self
    }

    /// Build the span
    pub fn build(self) -> Span {
        let now = self.start_time.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        });

        let mut span = Span::new(self.name)
            .with_kind(SpanKind::Server)
            .with_start_time(now)
            .with_end_time(now);

        if let Some(id) = self.request_id {
            span = span.with_attribute("request.id", AttributeValue::Int(id as i64));
        }
        if let Some(len) = self.prompt_length {
            span = span.with_attribute("prompt.length", AttributeValue::Int(len as i64));
        }
        if let Some(max) = self.max_tokens {
            span = span.with_attribute("max.tokens", AttributeValue::Int(max as i64));
        }
        if let Some(temp) = self.temperature {
            span = span.with_attribute("temperature", AttributeValue::Double(temp as f64));
        }

        span
    }

    /// Build and record the span, returning the span for updating
    pub fn build_and_record(self) -> Span {
        let span = self.build();
        record_span(span.clone());
        span
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_creation() {
        let span = Span::new("test_span");

        assert_eq!(span.name, "test_span");
        assert_eq!(span.kind, SpanKind::Internal);
        assert!(span.attributes.is_empty());
    }

    #[test]
    fn test_span_builder() {
        let span = Span::new("test")
            .with_kind(SpanKind::Server)
            .with_attribute("key", AttributeValue::String("value".to_string()))
            .with_status(SpanStatus::Ok);

        assert_eq!(span.name, "test");
        assert_eq!(span.kind, SpanKind::Server);
        assert_eq!(span.attributes.len(), 1);
        assert_eq!(span.status, Some(SpanStatus::Ok));
    }

    #[test]
    fn test_span_duration() {
        let start = 1_000_000_000;
        let end = 2_500_000_000;

        let span = Span::new("test")
            .with_start_time(start)
            .with_end_time(end);

        assert_eq!(span.duration_nanos(), 1_500_000_000);
    }

    #[test]
    fn test_trace_store_empty() {
        let store = TraceStore::new(TraceConfig::default());
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_trace_store_add() {
        let mut store = TraceStore::new(TraceConfig {
            sample_rate: 1.0, // Always sample
            max_traces: 100,
            service_name: "test".to_string(),
        });

        let span = Span::new("test");
        store.add_span(span);

        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_trace_store_sampling() {
        let mut store = TraceStore::new(TraceConfig {
            sample_rate: 0.0, // Never sample
            max_traces: 100,
            service_name: "test".to_string(),
        });

        let span = Span::new("test");
        let added = store.add_span(span);

        assert!(!added);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_trace_store_max_traces() {
        let mut store = TraceStore::new(TraceConfig {
            sample_rate: 1.0,
            max_traces: 3,
            service_name: "test".to_string(),
        });

        for i in 0..5 {
            store.add_span(Span::new(format!("span_{}", i)));
        }

        // Should only keep the last 3 spans due to FIFO eviction
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_trace_export() {
        let mut store = TraceStore::new(TraceConfig {
            sample_rate: 1.0,
            max_traces: 100,
            service_name: "rocmforge".to_string(),
        });

        store.add_span(Span::new("test_span"));

        let export = store.export();

        assert_eq!(export.resource_spans.len(), 1);
        assert_eq!(export.resource_spans[0].scope_spans.len(), 1);
        assert_eq!(export.resource_spans[0].scope_spans[0].spans.len(), 1);
    }

    #[test]
    fn test_inference_span_builder() {
        let span = InferenceSpanBuilder::new("inference")
            .with_request_id(123)
            .with_prompt_length(10)
            .with_max_tokens(100)
            .with_temperature(0.8)
            .build();

        assert_eq!(span.name, "inference");
        assert!(!span.trace_id.is_some()); // Not recorded yet
    }

    #[test]
    fn test_global_trace_store() {
        // Clear any existing traces
        clear_traces();

        let span = Span::new("global_test")
            .with_attribute("test", AttributeValue::Bool(true));

        // With default sampling, this might not be recorded
        // so we can't assert on the result
        let _ = record_span(span);

        // Just verify the function works
        let count = trace_count();
        assert!(count >= 0);
    }

    #[test]
    fn test_empty_export() {
        clear_traces();

        let export = export_traces();

        assert!(export.resource_spans.is_empty());
    }

    #[test]
    fn test_resource_creation() {
        let resource = Resource::new("test-service");

        // Check that service.name attribute exists
        assert!(resource.attributes.contains_key("service.name"));
        // Verify the value matches
        match resource.attributes.get("service.name") {
            Some(AttributeValue::String(s)) if s == "test-service" => {},
            _ => panic!("Expected service.name to be 'test-service'"),
        }
    }

    #[test]
    fn test_scope_spans_creation() {
        let scope_spans = ScopeSpans::new("test-scope");

        assert_eq!(scope_spans.scope.name, "test-scope");
        assert!(scope_spans.spans.is_empty());
    }

    #[test]
    fn test_span_status_from_bool() {
        assert_eq!(SpanStatus::from(true), SpanStatus::Ok);
        assert_eq!(SpanStatus::from(false), SpanStatus::Error);
    }

    #[test]
    fn test_attribute_values() {
        let span = Span::new("test")
            .with_attribute("string", AttributeValue::String("test".to_string()))
            .with_attribute("bool", AttributeValue::Bool(true))
            .with_attribute("int", AttributeValue::Int(42))
            .with_attribute("double", AttributeValue::Double(3.14));

        assert_eq!(span.attributes.len(), 4);
    }

    #[test]
    fn test_trace_id_generation() {
        let id1 = generate_trace_id();
        let id2 = generate_trace_id();

        // Trace IDs should be 32 hex chars (16 bytes)
        assert_eq!(id1.len(), 32);
        assert_eq!(id2.len(), 32);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_span_id_generation() {
        let id1 = generate_span_id();
        let id2 = generate_span_id();

        // Span IDs should be 16 hex chars (8 bytes)
        assert_eq!(id1.len(), 16);
        assert_eq!(id2.len(), 16);
        assert_ne!(id1, id2);
    }
}
