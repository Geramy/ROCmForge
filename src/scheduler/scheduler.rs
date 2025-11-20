//! Continuous batching scheduler for efficient GPU utilization

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SchedulerError {
    #[error("Request not found: {0}")]
    RequestNotFound(u32),
    #[error("Batch size exceeded maximum: {max}, got {actual}")]
    BatchSizeExceeded { max: usize, actual: usize },
    #[error("Invalid request state transition")]
    InvalidStateTransition,
    #[error("Queue capacity exceeded")]
    QueueCapacityExceeded,
}

pub type SchedulerResult<T> = Result<T, SchedulerError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RequestState {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub request_id: u32,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub state: RequestState,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub generated_tokens: Vec<u32>,
    pub finish_reason: Option<String>,
}

impl GenerationRequest {
    pub fn new(
        request_id: u32,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Self {
        GenerationRequest {
            request_id,
            prompt_tokens,
            max_tokens,
            temperature,
            top_k,
            top_p,
            state: RequestState::Pending,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            generated_tokens: Vec::new(),
            finish_reason: None,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    pub fn is_complete(&self) -> bool {
        match self.state {
            RequestState::Completed | RequestState::Failed | RequestState::Cancelled => true,
            _ => self.generated_tokens.len() >= self.max_tokens,
        }
    }

    pub fn start_processing(&mut self) -> SchedulerResult<()> {
        if self.state != RequestState::Pending {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.state = RequestState::Processing;
        self.started_at = Some(Instant::now());
        Ok(())
    }

    pub fn complete(&mut self, reason: Option<String>) -> SchedulerResult<()> {
        if self.state != RequestState::Processing {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.state = RequestState::Completed;
        self.completed_at = Some(Instant::now());
        if let Some(reason) = reason {
            self.finish_reason = Some(reason);
        } else if self.finish_reason.is_none() {
            self.finish_reason = Some("completed".to_string());
        }
        Ok(())
    }

    pub fn fail(&mut self) -> SchedulerResult<()> {
        if self.state != RequestState::Processing {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.state = RequestState::Failed;
        self.completed_at = Some(Instant::now());
        self.finish_reason
            .get_or_insert_with(|| "failed".to_string());
        Ok(())
    }

    pub fn cancel(&mut self) -> SchedulerResult<()> {
        if matches!(
            self.state,
            RequestState::Completed | RequestState::Failed | RequestState::Cancelled
        ) {
            return Err(SchedulerError::InvalidStateTransition);
        }

        if self.started_at.is_none() {
            self.started_at = Some(Instant::now());
        }
        self.state = RequestState::Cancelled;
        self.completed_at = Some(Instant::now());
        self.finish_reason = Some("cancelled".to_string());
        Ok(())
    }

    pub fn add_generated_token(&mut self, token: u32) -> SchedulerResult<()> {
        if self.state != RequestState::Processing {
            return Err(SchedulerError::InvalidStateTransition);
        }

        self.generated_tokens.push(token);

        if self.generated_tokens.len() >= self.max_tokens {
            self.complete(Some("length".to_string()))?;
        } else if self.is_complete() {
            self.complete(None)?;
        }

        Ok(())
    }

    pub fn processing_time(&self) -> Option<Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            (Some(start), None) => Some(Instant::now().duration_since(start)),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Batch {
    pub batch_id: u32,
    pub requests: Vec<GenerationRequest>,
    pub created_at: Instant,
}

impl Batch {
    pub fn new(batch_id: u32) -> Self {
        Batch {
            batch_id,
            requests: Vec::new(),
            created_at: Instant::now(),
        }
    }

    pub fn add_request(&mut self, request: GenerationRequest) -> SchedulerResult<()> {
        self.requests.push(request);
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn total_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.total_tokens()).sum()
    }

    pub fn max_sequence_length(&self) -> usize {
        self.requests
            .iter()
            .map(|r| r.total_tokens())
            .max()
            .unwrap_or(0)
    }

    pub fn min_sequence_length(&self) -> usize {
        self.requests
            .iter()
            .map(|r| r.total_tokens())
            .min()
            .unwrap_or(0)
    }

    pub fn length_variance(&self) -> f32 {
        if self.requests.is_empty() {
            return 0.0;
        }

        let lengths: Vec<usize> = self.requests.iter().map(|r| r.total_tokens()).collect();

        let mean = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
        let variance = lengths
            .iter()
            .map(|&l| (l as f32 - mean).powi(2))
            .sum::<f32>()
            / lengths.len() as f32;

        variance.sqrt()
    }
}

#[derive(Debug)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_queue_size: usize,
    pub batch_timeout: Duration,
    pub max_sequence_length: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            max_batch_size: 32,
            max_queue_size: 1000,
            batch_timeout: Duration::from_millis(50),
            max_sequence_length: 4096,
        }
    }
}

#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    pending_queue: VecDeque<GenerationRequest>,
    processing_requests: HashMap<u32, GenerationRequest>,
    completed_requests: HashMap<u32, GenerationRequest>,
    next_batch_id: u32,
    next_request_id: u32,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Scheduler {
            config,
            pending_queue: VecDeque::new(),
            processing_requests: HashMap::new(),
            completed_requests: HashMap::new(),
            next_batch_id: 0,
            next_request_id: 0,
        }
    }

    pub fn submit_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> SchedulerResult<u32> {
        if self.pending_queue.len() >= self.config.max_queue_size {
            return Err(SchedulerError::QueueCapacityExceeded);
        }

        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let request = GenerationRequest::new(
            request_id,
            prompt_tokens,
            max_tokens,
            temperature,
            top_k,
            top_p,
        );

        self.pending_queue.push_back(request);
        Ok(request_id)
    }

    pub fn create_batch(&mut self) -> SchedulerResult<Batch> {
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        let mut batch = Batch::new(batch_id);

        // Simple FIFO batching with length-based grouping
        let mut request_ids: Vec<u32> = self.pending_queue.iter().map(|r| r.request_id).collect();

        // Sort by sequence length for better batching
        request_ids.sort_by_key(|&id| {
            self.pending_queue
                .iter()
                .find(|r| r.request_id == id)
                .map(|r| r.total_tokens())
                .unwrap_or(0)
        });

        // Group similar lengths together
        for &request_id in &request_ids {
            if batch.size() >= self.config.max_batch_size {
                break;
            }

            // Find and remove from queue
            let pos = self
                .pending_queue
                .iter()
                .position(|r| r.request_id == request_id);

            if let Some(pos) = pos {
                let mut request = self.pending_queue.remove(pos).unwrap();

                // Check if this request fits well with current batch
                if batch.is_empty()
                    || (request.total_tokens() as f32 - batch.max_sequence_length() as f32).abs()
                        < batch.max_sequence_length() as f32 * 0.3
                {
                    request.start_processing()?;
                    self.processing_requests
                        .insert(request.request_id, request.clone());
                    batch.add_request(request)?;
                } else {
                    // Put it back
                    self.pending_queue.push_front(request);
                }
            }
        }

        Ok(batch)
    }

    pub fn update_batch(&mut self, batch: Batch) -> SchedulerResult<Vec<GenerationRequest>> {
        let mut completed_requests = Vec::new();

        for request in batch.requests {
            if request.is_complete() || request.state == RequestState::Failed {
                let mut req = request;
                if req.state == RequestState::Processing {
                    req.complete(None)?;
                }

                self.processing_requests.remove(&req.request_id);
                self.completed_requests.insert(req.request_id, req.clone());
                completed_requests.push(req);
            } else {
                // Still processing, update in processing_requests
                self.processing_requests.insert(request.request_id, request);
            }
        }

        Ok(completed_requests)
    }

    pub fn get_request(&self, request_id: u32) -> SchedulerResult<&GenerationRequest> {
        self.processing_requests
            .get(&request_id)
            .or_else(|| self.completed_requests.get(&request_id))
            .or_else(|| {
                self.pending_queue
                    .iter()
                    .find(|r| r.request_id == request_id)
            })
            .ok_or(SchedulerError::RequestNotFound(request_id))
    }

    pub fn get_request_mut(&mut self, request_id: u32) -> SchedulerResult<&mut GenerationRequest> {
        if let Some(request) = self.processing_requests.get_mut(&request_id) {
            Ok(request)
        } else if let Some(request) = self.completed_requests.get_mut(&request_id) {
            Ok(request)
        } else {
            self.pending_queue
                .iter_mut()
                .find(|r| r.request_id == request_id)
                .ok_or(SchedulerError::RequestNotFound(request_id))
        }
    }

    pub fn cancel_request(&mut self, request_id: u32) -> SchedulerResult<GenerationRequest> {
        if let Some(mut request) = self.processing_requests.remove(&request_id) {
            request.cancel()?;
            self.completed_requests.insert(request_id, request.clone());
            return Ok(request);
        }

        if let Some(pos) = self
            .pending_queue
            .iter()
            .position(|r| r.request_id == request_id)
        {
            let mut request = self.pending_queue.remove(pos).unwrap();
            request.cancel()?;
            self.completed_requests.insert(request_id, request.clone());
            return Ok(request);
        }

        if let Some(request) = self.completed_requests.get(&request_id) {
            if request.state == RequestState::Cancelled {
                return Ok(request.clone());
            }
            return Err(SchedulerError::InvalidStateTransition);
        }

        Err(SchedulerError::RequestNotFound(request_id))
    }

    pub fn add_generated_token(&mut self, request_id: u32, token: u32) -> SchedulerResult<()> {
        let request = self.get_request_mut(request_id)?;
        request.add_generated_token(token)
    }

    pub fn get_queue_stats(&self) -> QueueStats {
        QueueStats {
            pending_requests: self.pending_queue.len(),
            processing_requests: self.processing_requests.len(),
            completed_requests: self.completed_requests.len(),
        }
    }

    pub fn has_pending_requests(&self) -> bool {
        !self.pending_queue.is_empty()
    }

    pub fn can_create_batch(&self) -> bool {
        !self.pending_queue.is_empty()
            && self.processing_requests.len() < self.config.max_batch_size
    }
}

#[derive(Debug, Clone)]
pub struct QueueStats {
    pub pending_requests: usize,
    pub processing_requests: usize,
    pub completed_requests: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_request_creation() {
        let request = GenerationRequest::new(1, vec![1, 2, 3], 10, 0.8, 50, 0.9);

        assert_eq!(request.request_id, 1);
        assert_eq!(request.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(request.max_tokens, 10);
        assert_eq!(request.state, RequestState::Pending);
        assert_eq!(request.total_tokens(), 3);
        assert!(!request.is_complete());
    }

    #[test]
    fn test_request_state_transitions() {
        let mut request = GenerationRequest::new(1, vec![1, 2, 3], 10, 0.8, 50, 0.9);

        // Start processing
        assert!(request.start_processing().is_ok());
        assert_eq!(request.state, RequestState::Processing);
        assert!(request.started_at.is_some());

        // Add tokens
        for i in 0..10 {
            assert!(request.add_generated_token(i).is_ok());
        }

        // Should be complete now
        assert_eq!(request.state, RequestState::Completed);
        assert!(request.completed_at.is_some());
        assert!(request.is_complete());
    }

    #[test]
    fn test_scheduler_request_submission() {
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config);

        let request_id = scheduler.submit_request(vec![1, 2, 3], 10, 0.8, 50, 0.9);

        assert!(request_id.is_ok());
        assert_eq!(request_id.unwrap(), 0);

        let stats = scheduler.get_queue_stats();
        assert_eq!(stats.pending_requests, 1);
        assert_eq!(stats.processing_requests, 0);
    }

    #[test]
    fn test_batch_creation() {
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config);

        // Submit multiple requests
        for i in 0..3 {
            scheduler
                .submit_request(vec![i, i + 1, i + 2], 10, 0.8, 50, 0.9)
                .unwrap();
        }

        let batch = scheduler.create_batch();
        assert!(batch.is_ok());

        let batch = batch.unwrap();
        assert_eq!(batch.size(), 3);
        assert_eq!(batch.batch_id, 0);

        let stats = scheduler.get_queue_stats();
        assert_eq!(stats.pending_requests, 0);
        assert_eq!(stats.processing_requests, 3);
    }

    #[test]
    fn test_batch_length_variance() {
        let mut batch = Batch::new(1);

        // Add requests with similar lengths
        batch
            .add_request(GenerationRequest::new(1, vec![1; 10], 10, 0.8, 50, 0.9))
            .unwrap();
        batch
            .add_request(GenerationRequest::new(2, vec![2; 12], 10, 0.8, 50, 0.9))
            .unwrap();
        batch
            .add_request(GenerationRequest::new(3, vec![3; 11], 10, 0.8, 50, 0.9))
            .unwrap();

        let variance = batch.length_variance();
        assert!(variance < 2.0); // Should be low variance
    }

    #[test]
    fn test_queue_capacity_limit() {
        let config = SchedulerConfig {
            max_queue_size: 2,
            ..Default::default()
        };
        let mut scheduler = Scheduler::new(config);

        // Submit up to capacity
        scheduler.submit_request(vec![1], 10, 0.8, 50, 0.9).unwrap();
        scheduler.submit_request(vec![2], 10, 0.8, 50, 0.9).unwrap();

        // Should fail when exceeding capacity
        let result = scheduler.submit_request(vec![3], 10, 0.8, 50, 0.9);
        assert!(result.is_err());
        assert!(matches!(result, Err(SchedulerError::QueueCapacityExceeded)));
    }

    #[test]
    fn test_batch_update() {
        let config = SchedulerConfig::default();
        let mut scheduler = Scheduler::new(config);

        scheduler
            .submit_request(vec![1, 2, 3], 2, 0.8, 50, 0.9)
            .unwrap();
        let batch = scheduler.create_batch().unwrap();

        // Simulate token generation
        let mut updated_batch = batch;
        for request in &mut updated_batch.requests {
            request.add_generated_token(42).unwrap();
            request.add_generated_token(43).unwrap(); // Should complete
        }

        let completed = scheduler.update_batch(updated_batch).unwrap();
        assert_eq!(completed.len(), 1);
        assert!(completed[0].is_complete());

        let stats = scheduler.get_queue_stats();
        assert_eq!(stats.processing_requests, 0);
        assert_eq!(stats.completed_requests, 1);
    }

    // Property tests
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_batching_properties(
            num_requests in 1..20usize,
            max_batch_size in 1..8usize
        ) {
            let config = SchedulerConfig {
                max_batch_size,
                ..Default::default()
            };
            let mut scheduler = Scheduler::new(config);

            // Submit requests
            for i in 0..num_requests {
                scheduler.submit_request(
                    vec![i as u32; (i % 10) + 1],
                    10,
                    0.8,
                    50,
                    0.9,
                ).unwrap();
            }

            // Create batches until no more pending requests
            let mut total_processed = 0;
            while scheduler.has_pending_requests() {
                let batch = scheduler.create_batch().unwrap();
                prop_assert!(batch.size() <= max_batch_size);
                total_processed += batch.size();

                // Complete all requests in batch
                let mut updated_batch = batch;
                for request in &mut updated_batch.requests {
                    for _ in 0..request.max_tokens {
                        let _ = request.add_generated_token(42);
                    }
                }

                scheduler.update_batch(updated_batch).unwrap();
            }

            prop_assert_eq!(total_processed, num_requests);
        }
    }
}
