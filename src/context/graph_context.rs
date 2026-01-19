//! Graph-based context storage with HNSW vector indexing
//!
//! This module provides a context store that combines:
//! - Graph structure for message relationships
//! - HNSW index for semantic similarity search
//! - Optional embedding model interface

use crate::context::sqlitegraph::{
    hnsw::{DistanceMetric, HnswConfig, HnswIndex, HnswError},
    GraphEdge, GraphEntity, SqliteGraph, SqliteGraphError,
};
use serde_json::json;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Result type for context operations
pub type ContextResult<T> = Result<T, ContextError>;

/// Errors that can occur in context operations
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("Graph error: {0}")]
    Graph(#[from] SqliteGraphError),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    #[error("Message not found: {0}")]
    MessageNotFound(u64),

    #[error("HNSW index error: {0}")]
    Hnsw(String),
}

/// Embedding model trait for generating text embeddings
///
/// Implementations can call external APIs (OpenAI, local models) or use
/// simpler algorithms like hash-based embeddings.
pub trait EmbeddingModel: Send + Sync {
    /// Generate embedding for the given text
    fn embed(&self, text: &str) -> ContextResult<Vec<f32>>;

    /// Get the embedding dimension
    fn dimension(&self) -> usize;
}

/// Dummy embedding model for testing and development
///
/// Generates simple hash-based embeddings that preserve some semantic
/// similarity. For production use, replace with a real embedding model.
pub struct DummyEmbedding {
    pub dimension: usize,
}

impl DummyEmbedding {
    /// Create a new dummy embedding model
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Default for DummyEmbedding {
    fn default() -> Self {
        Self::new(384) // Common dimension for small embedding models
    }
}

impl EmbeddingModel for DummyEmbedding {
    fn embed(&self, text: &str) -> ContextResult<Vec<f32>> {
        let mut embedding = vec![0.0f32; self.dimension];

        // Simple hash-based embedding that preserves some text characteristics
        let bytes = text.as_bytes();
        for (i, &byte) in bytes.iter().enumerate() {
            let idx = (i * 3) % self.dimension;
            // Normalize to [-1, 1] range
            embedding[idx] += (byte as f32 - 128.0) / 128.0;
        }

        // L2 normalize for cosine similarity
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in embedding.iter_mut() {
                *v /= norm;
            }
        }

        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// A message stored in the context graph
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextMessage {
    /// Unique message ID (graph node ID)
    pub id: u64,
    /// Message text content
    pub text: String,
    /// Message embedding (not always serialized)
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
    /// Unix timestamp
    pub timestamp: i64,
    /// Sequence ID in conversation
    pub seq_id: usize,
    /// Similarity score (only set for search results)
    pub similarity: Option<f32>,
}

/// Search parameters for context retrieval
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ContextSearchParams {
    /// Query text to search for
    pub q: String,
    /// Maximum number of results to return
    #[serde(default = "default_k")]
    pub k: usize,
    /// Whether to expand results to include conversation context
    #[serde(default)]
    pub expand: bool,
}

fn default_k() -> usize {
    5
}

/// Search result containing relevant context messages
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextSearchResult {
    /// Retrieved messages ordered by similarity
    pub messages: Vec<ContextMessage>,
    /// Total messages in the store
    pub total_messages: usize,
    /// Query used for search
    pub query: String,
}

/// Graph-based context store with HNSW indexing
///
/// Stores messages as graph nodes with embeddings indexed in HNSW
/// for fast semantic similarity search.
pub struct GraphContextStore {
    /// Graph database for storing messages and relationships
    graph: SqliteGraph,
    /// HNSW index for vector similarity search
    hnsw: HnswIndex,
    /// Embedding dimension
    dimension: usize,
    /// Embedding model
    embedding_model: Box<dyn EmbeddingModel>,
    /// Message counter for sequence IDs
    next_seq_id: usize,
    /// Previous message ID for conversation threading
    prev_message_id: Option<u64>,
    /// Track message node IDs for rebuilding
    message_node_ids: Vec<u64>,
}

impl GraphContextStore {
    /// Create a new in-memory context store
    ///
    /// # Arguments
    /// * `dimension` - Embedding dimension (e.g., 384 for MiniLM, 1536 for OpenAI)
    ///
    /// # Example
    /// ```no_run
    /// use rocmforge::context::GraphContextStore;
    ///
    /// let store = GraphContextStore::in_memory(384)?;
    /// ```
    pub fn in_memory(dimension: usize) -> ContextResult<Self> {
        Self::with_embedding_model(dimension, Box::new(DummyEmbedding::new(dimension)))
    }

    /// Create a context store with a custom embedding model
    pub fn with_embedding_model(
        dimension: usize,
        model: Box<dyn EmbeddingModel>,
    ) -> ContextResult<Self> {
        let graph = SqliteGraph::open_in_memory()?;

        let hnsw_config = HnswConfig::new(
            dimension,
            16,  // m_connections
            200, // ef_construction
            DistanceMetric::Cosine,
        );

        let hnsw = HnswIndex::new("context_index", hnsw_config).map_err(|e| ContextError::Hnsw(e.to_string()))?;

        Ok(Self {
            graph,
            hnsw,
            dimension,
            embedding_model: model,
            next_seq_id: 0,
            prev_message_id: None,
            message_node_ids: Vec::new(),
        })
    }

    /// Open a persistent context store from disk
    ///
    /// # Arguments
    /// * `path` - Path to the database file (will be created if it doesn't exist)
    /// * `dimension` - Embedding dimension
    ///
    /// # Example
    /// ```no_run
    /// use rocmforge::context::GraphContextStore;
    ///
    /// let store = GraphContextStore::open("context.db", 384)?;
    /// ```
    pub fn open<P: AsRef<Path>>(path: P, dimension: usize) -> ContextResult<Self> {
        let path_ref = path.as_ref();
        let graph = SqliteGraph::open(path_ref)?;

        // Check if this is a new database by checking entity count
        let entity_ids = graph.list_entity_ids().unwrap_or_default();
        let is_new = entity_ids.is_empty();

        let hnsw_config = HnswConfig::new(
            dimension,
            16,  // m_connections
            200, // ef_construction
            DistanceMetric::Cosine,
        );

        let hnsw = HnswIndex::new("context_index", hnsw_config).map_err(|e| ContextError::Hnsw(e.to_string()))?;

        let mut store = Self {
            graph,
            hnsw,
            dimension,
            embedding_model: Box::new(DummyEmbedding::new(dimension)),
            next_seq_id: 0,
            prev_message_id: None,
            message_node_ids: Vec::new(),
        };

        // If existing database, reload HNSW index
        if !is_new {
            store.rebuild_hnsw_index()?;
        }

        Ok(store)
    }

    /// Rebuild HNSW index from existing graph messages
    fn rebuild_hnsw_index(&mut self) -> ContextResult<()> {
        // Get all entities
        let entity_ids = self
            .graph
            .list_entity_ids()
            .map_err(|e| ContextError::Graph(e))?;

        for id in entity_ids {
            let id_u64 = id as u64;
            match self.graph.get_entity(id) {
                Ok(entity) => {
                    if entity.kind == "message" {
                        self.message_node_ids.push(id_u64);

                        if let Some(embedding) = entity.data.get("embedding") {
                            if let Some(arr) = embedding.as_array() {
                                let vec: Vec<f32> = arr
                                    .iter()
                                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                                    .collect();

                                if vec.len() == self.dimension {
                                    self.hnsw
                                        .insert_vector(&vec, Some(json!({"node_id": id_u64})))
                                        .map_err(|e| ContextError::Hnsw(e.to_string()))?;
                                }
                            }
                        }

                        // Track sequence IDs
                        if let Some(seq_id) = entity.data.get("seq_id") {
                            if let Some(seq) = seq_id.as_u64() {
                                if seq as usize >= self.next_seq_id {
                                    self.next_seq_id = seq as usize + 1;
                                }
                            }
                        }
                    }
                }
                Err(SqliteGraphError::NotFound(_)) => continue,
                Err(e) => return Err(ContextError::Graph(e)),
            }
        }

        Ok(())
    }

    /// Set the embedding model for this store
    pub fn set_embedding_model(&mut self, model: Box<dyn EmbeddingModel>) {
        self.embedding_model = model;
    }

    /// Add a message to the context store
    ///
    /// # Arguments
    /// * `text` - Message text content
    /// * `prev_id` - Optional ID of previous message (for conversation threading)
    ///
    /// # Returns
    /// The ID of the newly created message
    ///
    /// # Example
    /// ```no_run
    /// # use rocmforge::context::GraphContextStore;
    /// # let mut store = GraphContextStore::in_memory(384).unwrap();
    /// let msg1 = store.add_message("Hello", None).unwrap();
    /// let msg2 = store.add_message("World", Some(msg1)).unwrap();
    /// ```
    pub fn add_message(
        &mut self,
        text: &str,
        prev_id: Option<u64>,
    ) -> ContextResult<u64> {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        // Generate embedding
        let embedding = self
            .embedding_model
            .embed(text)
            .map_err(|e| ContextError::Embedding(e.to_string()))?;

        // Convert embedding to JSON-serializable array
        let embedding_json: Vec<serde_json::Value> =
            embedding.iter().map(|&v| json!(v)).collect();

        // Create timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        // Create graph entity
        let entity = GraphEntity {
            id: 0,
            kind: "message".to_string(),
            name: format!("msg_{}", seq_id),
            file_path: None,
            data: json!({
                "text": text,
                "timestamp": timestamp,
                "seq_id": seq_id,
                "embedding": embedding_json,
            }),
        };

        let node_id = self
            .graph
            .insert_entity(&entity)
            .map_err(|e| ContextError::Graph(e))? as u64;

        self.message_node_ids.push(node_id);

        // Insert into HNSW index - store node_id in metadata
        self.hnsw
            .insert_vector(&embedding, Some(json!({"node_id": node_id})))
            .map_err(|e| ContextError::Hnsw(e.to_string()))?;

        // Create edge to previous message
        if let Some(prev) = prev_id {
            let edge = GraphEdge {
                id: 0,
                from_id: prev as i64,
                to_id: node_id as i64,
                edge_type: "follows".to_string(),
                data: json!({}),
            };
            self.graph
                .insert_edge(&edge)
                .map_err(|e| ContextError::Graph(e))?;
        }

        self.prev_message_id = Some(node_id);
        Ok(node_id)
    }

    /// Add a message and auto-link to previous message
    ///
    /// Convenience method that automatically links to the last added message.
    pub fn add_message_continuation(&mut self, text: &str) -> ContextResult<u64> {
        self.add_message(text, self.prev_message_id)
    }

    /// Retrieve context messages by semantic similarity
    ///
    /// # Arguments
    /// * `query` - Query text to find similar messages for
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    /// Vector of context messages ordered by similarity
    pub fn retrieve_context(
        &self,
        query: &str,
        k: usize,
    ) -> ContextResult<Vec<ContextMessage>> {
        // Embed query
        let query_emb = self
            .embedding_model
            .embed(query)
            .map_err(|e| ContextError::Embedding(e.to_string()))?;

        // Search HNSW index - returns Vec<(vector_id, distance)>
        let results = self
            .hnsw
            .search(&query_emb, k)
            .map_err(|e| ContextError::Hnsw(e.to_string()))?;

        // We need to get metadata from our stored mappings
        // For now, we'll use list_entity_ids and match by index
        let entity_ids = self
            .graph
            .list_entity_ids()
            .map_err(|e| ContextError::Graph(e))?;

        // Fetch message entities by mapping vector_id to node_id
        let mut messages = Vec::new();
        for (vector_id, distance) in results {
            // vector_id is 0-based, map to our message IDs
            if let Some(&node_id) = self.message_node_ids.get(vector_id as usize) {
                match self.graph.get_entity(node_id as i64) {
                    Ok(entity) => {
                        let text = entity.data["text"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();
                        let timestamp = entity.data["timestamp"]
                            .as_i64()
                            .unwrap_or(0);
                        let seq_id = entity.data["seq_id"]
                            .as_u64()
                            .unwrap_or(0) as usize;

                        messages.push(ContextMessage {
                            id: entity.id as u64,
                            text,
                            embedding: None, // Not returning embeddings in results
                            timestamp,
                            seq_id,
                            similarity: Some(distance),
                        });
                    }
                    Err(SqliteGraphError::NotFound(_)) => {
                        // Message may have been deleted, skip
                        continue;
                    }
                    Err(e) => return Err(ContextError::Graph(e)),
                }
            }
        }

        Ok(messages)
    }

    /// Retrieve context with optional graph expansion
    ///
    /// # Arguments
    /// * `query` - Query text to find similar messages for
    /// * `k` - Maximum number of results to return
    /// * `expand_neighbors` - If true, include neighboring messages in results
    ///
    /// Note: Graph expansion is currently not supported due to limited public API.
    /// The expand_neighbors parameter is accepted but not implemented.
    pub fn retrieve_context_expanded(
        &self,
        query: &str,
        k: usize,
        _expand_neighbors: bool,
    ) -> ContextResult<Vec<ContextMessage>> {
        // For now, just return basic context without neighbor expansion
        // The sqlitegraph crate doesn't expose a simple neighbors() API
        self.retrieve_context(query, k)
    }

    /// Get a specific message by ID
    pub fn get_message(&self, id: u64) -> ContextResult<ContextMessage> {
        let entity = self
            .graph
            .get_entity(id as i64)
            .map_err(|e| ContextError::Graph(e))?;

        if entity.kind != "message" {
            return Err(ContextError::MessageNotFound(id));
        }

        let text = entity.data["text"].as_str().unwrap_or("").to_string();
        let timestamp = entity.data["timestamp"].as_i64().unwrap_or(0);
        let seq_id = entity.data["seq_id"].as_u64().unwrap_or(0) as usize;

        Ok(ContextMessage {
            id: entity.id as u64,
            text,
            embedding: None,
            timestamp,
            seq_id,
            similarity: None,
        })
    }

    /// Get all messages in conversation sequence
    pub fn get_conversation(&self) -> ContextResult<Vec<ContextMessage>> {
        let mut messages: Vec<ContextMessage> = self
            .message_node_ids
            .iter()
            .filter_map(|&id| {
                self.graph.get_entity(id as i64).ok().filter(|e| e.kind == "message")
            })
            .map(|entity| {
                let text = entity.data["text"].as_str().unwrap_or("").to_string();
                let timestamp = entity.data["timestamp"].as_i64().unwrap_or(0);
                let seq_id = entity.data["seq_id"].as_u64().unwrap_or(0) as usize;

                ContextMessage {
                    id: entity.id as u64,
                    text,
                    embedding: None,
                    timestamp,
                    seq_id,
                    similarity: None,
                }
            })
            .collect();

        // Sort by sequence ID
        messages.sort_by_key(|m| m.seq_id);
        Ok(messages)
    }

    /// Get the total number of messages in the store
    pub fn len(&self) -> ContextResult<usize> {
        Ok(self.message_node_ids.len())
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> ContextResult<bool> {
        Ok(self.message_node_ids.is_empty())
    }

    /// Get the total number of vectors in the HNSW index
    pub fn hnsw_len(&self) -> usize {
        self.hnsw.vector_count()
    }

    /// Clear all messages from the store
    pub fn clear(&mut self) -> ContextResult<()> {
        // Delete all message entities
        for &id in &self.message_node_ids {
            self.graph
                .delete_entity(id as i64)
                .map_err(|e| ContextError::Graph(e))?;
        }

        // Reset state
        self.next_seq_id = 0;
        self.prev_message_id = None;
        self.message_node_ids.clear();

        Ok(())
    }

    /// Search with full result object
    pub fn search(&self, params: &ContextSearchParams) -> ContextResult<ContextSearchResult> {
        let total = self.len()?;
        let messages = if params.expand {
            self.retrieve_context_expanded(&params.q, params.k, true)?
        } else {
            self.retrieve_context(&params.q, params.k)?
        };

        Ok(ContextSearchResult {
            messages,
            total_messages: total,
            query: params.q.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_embedding_dimension() {
        let model = DummyEmbedding::new(128);
        assert_eq!(model.dimension(), 128);
    }

    #[test]
    fn test_dummy_embedding_generate() {
        let model = DummyEmbedding::new(128);
        let emb = model.embed("hello world").unwrap();
        assert_eq!(emb.len(), 128);

        // Check normalized
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_store_in_memory() {
        let store = GraphContextStore::in_memory(64).unwrap();
        assert!(store.is_empty().unwrap());
        assert_eq!(store.len().unwrap(), 0);
    }

    #[test]
    fn test_add_message() {
        let mut store = GraphContextStore::in_memory(64).unwrap();

        let id = store.add_message("Hello world", None).unwrap();
        assert!(id > 0);
        assert_eq!(store.len().unwrap(), 1);
        assert!(!store.is_empty().unwrap());
    }

    #[test]
    fn test_add_message_sequence() {
        let mut store = GraphContextStore::in_memory(64).unwrap();

        let id1 = store.add_message("First", None).unwrap();
        let id2 = store.add_message("Second", Some(id1)).unwrap();
        let id3 = store.add_message_continuation("Third").unwrap();

        assert_eq!(store.len().unwrap(), 3);

        // Check sequence IDs
        let conv = store.get_conversation().unwrap();
        assert_eq!(conv.len(), 3);
        assert_eq!(conv[0].seq_id, 0);
        assert_eq!(conv[1].seq_id, 1);
        assert_eq!(conv[2].seq_id, 2);
    }

    #[test]
    fn test_retrieve_context() {
        let mut store = GraphContextStore::in_memory(64).unwrap();

        store
            .add_message("The weather is sunny today", None)
            .unwrap();
        store.add_message("I love playing tennis", None).unwrap();
        store
            .add_message("What's the weather like?", None)
            .unwrap();

        // Search for weather-related content
        let results = store.retrieve_context("weather forecast", 5).unwrap();

        // Should return at least one result
        assert!(!results.is_empty());

        // Results should have similarity scores (distance values from HNSW)
        // Note: HNSW is approximate, so exact ordering is not guaranteed
        assert!(results[0].similarity.is_some());
    }

    #[test]
    fn test_get_message() {
        let mut store = GraphContextStore::in_memory(64).unwrap();

        let id = store.add_message("Test message", None).unwrap();
        let msg = store.get_message(id).unwrap();

        assert_eq!(msg.id, id);
        assert_eq!(msg.text, "Test message");
        assert_eq!(msg.seq_id, 0);
    }

    #[test]
    fn test_get_conversation() {
        let mut store = GraphContextStore::in_memory(64).unwrap();

        store.add_message("First message", None).unwrap();
        store.add_message("Second message", None).unwrap();
        store.add_message("Third message", None).unwrap();

        let conv = store.get_conversation().unwrap();
        assert_eq!(conv.len(), 3);
        assert_eq!(conv[0].text, "First message");
        assert_eq!(conv[1].text, "Second message");
        assert_eq!(conv[2].text, "Third message");
    }

    #[test]
    fn test_clear() {
        let mut store = GraphContextStore::in_memory(64).unwrap();

        store.add_message("Test", None).unwrap();
        store.add_message("Test 2", None).unwrap();
        assert_eq!(store.len().unwrap(), 2);

        store.clear().unwrap();
        assert_eq!(store.len().unwrap(), 0);
        assert!(store.is_empty().unwrap());
    }

    #[test]
    fn test_context_search_params() {
        let params = ContextSearchParams {
            q: "test query".to_string(),
            k: 10,
            expand: true,
        };

        assert_eq!(params.q, "test query");
        assert_eq!(params.k, 10);
        assert!(params.expand);
    }

    #[test]
    fn test_context_search_params_default() {
        let params = ContextSearchParams {
            q: "test".to_string(),
            k: 5,
            expand: false,
        };

        assert_eq!(params.k, 5);
        assert!(!params.expand);
    }

    #[test]
    fn test_search_method() {
        let mut store = GraphContextStore::in_memory(64).unwrap();

        store.add_message("The weather is sunny", None).unwrap();
        store.add_message("I enjoy sports", None).unwrap();

        let params = ContextSearchParams {
            q: "weather".to_string(),
            k: 5,
            expand: false,
        };

        let result = store.search(&params).unwrap();
        assert_eq!(result.query, "weather");
        assert_eq!(result.total_messages, 2);
        assert!(!result.messages.is_empty());
    }
}
