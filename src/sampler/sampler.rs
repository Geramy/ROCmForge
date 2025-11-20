//! Logits sampling with top-k, top-p, and temperature support

use rand::{distributions::WeightedIndex, prelude::Distribution, thread_rng};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SamplerError {
    #[error("Empty logits vector")]
    EmptyLogits,
    #[error("Invalid temperature: {0}. Must be > 0")]
    InvalidTemperature(f32),
    #[error("Invalid top_k: {0}. Must be > 0")]
    InvalidTopK(usize),
    #[error("Invalid top_p: {0}. Must be in (0, 1]")]
    InvalidTopP(f32),
    #[error("All probabilities are zero")]
    ZeroProbabilities,
}

pub type SamplerResult<T> = Result<T, SamplerError>;

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        SamplingConfig {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingConfig {
    pub fn new(temperature: f32, top_k: usize, top_p: f32) -> SamplerResult<Self> {
        if temperature <= 0.0 {
            return Err(SamplerError::InvalidTemperature(temperature));
        }
        if top_k == 0 {
            return Err(SamplerError::InvalidTopK(top_k));
        }
        if top_p <= 0.0 || top_p > 1.0 {
            return Err(SamplerError::InvalidTopP(top_p));
        }

        Ok(SamplingConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty: 1.0,
        })
    }

    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }
}

#[derive(Debug, Clone)]
pub struct TokenScore {
    pub token_id: u32,
    pub score: f32,
}

#[derive(Debug)]
pub struct Sampler {
    config: SamplingConfig,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        Sampler { config }
    }

    pub fn sample(&mut self, logits: &[f32]) -> SamplerResult<u32> {
        if logits.is_empty() {
            return Err(SamplerError::EmptyLogits);
        }

        // Apply temperature
        let scaled_logits = self.apply_temperature(logits)?;

        // Apply top-k filtering
        let top_k_logits = self.apply_top_k(&scaled_logits)?;

        // Apply top-p (nucleus) filtering
        let filtered_logits = self.apply_top_p(&top_k_logits)?;

        // Convert to probabilities
        let scores: Vec<f32> = filtered_logits.iter().map(|ts| ts.score).collect();
        let probabilities = self.softmax(&scores)?;

        // Sample from distribution
        self.sample_from_distribution(&probabilities, &filtered_logits)
    }

    pub fn sample_with_history(
        &mut self,
        logits: &[f32],
        token_history: &[u32],
    ) -> SamplerResult<u32> {
        if logits.is_empty() {
            return Err(SamplerError::EmptyLogits);
        }

        // Apply repetition penalty
        let penalized_logits = self.apply_repetition_penalty(logits, token_history);

        // Apply temperature
        let scaled_logits = self.apply_temperature(&penalized_logits)?;

        // Apply top-k filtering
        let top_k_logits = self.apply_top_k(&scaled_logits)?;

        // Apply top-p (nucleus) filtering
        let filtered_logits = self.apply_top_p(&top_k_logits)?;

        // Convert to probabilities
        let scores: Vec<f32> = filtered_logits.iter().map(|ts| ts.score).collect();
        let probabilities = self.softmax(&scores)?;

        // Sample from distribution
        self.sample_from_distribution(&probabilities, &filtered_logits)
    }

    fn apply_temperature(&self, logits: &[f32]) -> SamplerResult<Vec<f32>> {
        Ok(logits
            .iter()
            .map(|&logit| logit / self.config.temperature)
            .collect())
    }

    fn apply_repetition_penalty(&self, logits: &[f32], token_history: &[u32]) -> Vec<f32> {
        if self.config.repetition_penalty == 1.0 || token_history.is_empty() {
            return logits.to_vec();
        }

        let mut penalized_logits = logits.to_vec();
        let penalty = self.config.repetition_penalty;

        for &token_id in token_history {
            if (token_id as usize) < penalized_logits.len() {
                let logit = penalized_logits[token_id as usize];
                if logit < 0.0 {
                    penalized_logits[token_id as usize] = logit * penalty;
                } else {
                    penalized_logits[token_id as usize] = logit / penalty;
                }
            }
        }

        penalized_logits
    }

    fn apply_top_k(&self, logits: &[f32]) -> SamplerResult<Vec<TokenScore>> {
        let mut token_scores: Vec<TokenScore> = logits
            .iter()
            .enumerate()
            .map(|(i, &score)| TokenScore {
                token_id: i as u32,
                score,
            })
            .collect();

        // Sort by score in descending order
        token_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Keep top-k tokens
        token_scores.truncate(self.config.top_k);

        Ok(token_scores)
    }

    fn apply_top_p(&self, token_scores: &[TokenScore]) -> SamplerResult<Vec<TokenScore>> {
        if token_scores.is_empty() {
            return Ok(Vec::new());
        }

        // Convert to probabilities
        let scores: Vec<f32> = token_scores.iter().map(|ts| ts.score).collect();
        let probabilities = self.softmax(&scores)?;

        // Sort by probability in descending order
        let mut indexed_probs: Vec<(usize, f32)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find smallest set where cumulative probability >= top_p
        let mut cumulative_prob = 0.0;
        let mut cutoff_index = indexed_probs.len();

        for (i, &(_, prob)) in indexed_probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= self.config.top_p {
                cutoff_index = i + 1;
                break;
            }
        }

        // Filter tokens
        let mut filtered_tokens = Vec::new();
        for (original_idx, _) in indexed_probs.iter().take(cutoff_index) {
            filtered_tokens.push(token_scores[*original_idx].clone());
        }

        Ok(filtered_tokens)
    }

    fn softmax(&self, scores: &[f32]) -> SamplerResult<Vec<f32>> {
        if scores.is_empty() {
            return Ok(Vec::new());
        }

        // Find maximum for numerical stability
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp and sum
        let exp_scores: Vec<f32> = scores
            .iter()
            .map(|&score| (score - max_score).exp())
            .collect();

        let sum_exp: f32 = exp_scores.iter().sum();

        if sum_exp == 0.0 {
            return Err(SamplerError::ZeroProbabilities);
        }

        // Normalize
        let probabilities: Vec<f32> = exp_scores
            .iter()
            .map(|&exp_score| exp_score / sum_exp)
            .collect();

        Ok(probabilities)
    }

    fn sample_from_distribution(
        &mut self,
        probabilities: &[f32],
        token_scores: &[TokenScore],
    ) -> SamplerResult<u32> {
        if probabilities.is_empty() {
            return Err(SamplerError::ZeroProbabilities);
        }

        // Create weighted distribution
        let weights: Vec<f32> = probabilities.iter().map(|&p| p * 1000.0).collect();

        match WeightedIndex::new(&weights) {
            Ok(dist) => {
                let index = dist.sample(&mut thread_rng());
                Ok(token_scores[index].token_id)
            }
            Err(_) => {
                // Fallback to argmax if weighted distribution fails
                let max_index = probabilities
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                Ok(token_scores[max_index].token_id)
            }
        }
    }

    pub fn greedy_sample(&mut self, logits: &[f32]) -> SamplerResult<u32> {
        if logits.is_empty() {
            return Err(SamplerError::EmptyLogits);
        }

        let max_index = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(max_index as u32)
    }

    pub fn update_config(&mut self, config: SamplingConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_config_creation() {
        let config = SamplingConfig::new(0.8, 50, 0.9);
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.temperature, 0.8);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.top_p, 0.9);
    }

    #[test]
    fn test_invalid_sampling_config() {
        let config = SamplingConfig::new(0.0, 50, 0.9);
        assert!(config.is_err());
        assert!(matches!(config, Err(SamplerError::InvalidTemperature(_))));

        let config = SamplingConfig::new(0.8, 0, 0.9);
        assert!(config.is_err());
        assert!(matches!(config, Err(SamplerError::InvalidTopK(_))));

        let config = SamplingConfig::new(0.8, 50, 0.0);
        assert!(config.is_err());
        assert!(matches!(config, Err(SamplerError::InvalidTopP(_))));
    }

    #[test]
    fn test_sampler_creation() {
        let config = SamplingConfig::new(0.8, 50, 0.9).unwrap();
        let sampler = Sampler::new(config);
        assert_eq!(sampler.config.temperature, 0.8);
        assert_eq!(sampler.config.top_k, 50);
        assert_eq!(sampler.config.top_p, 0.9);
    }

    #[test]
    fn test_greedy_sampling() {
        let config = SamplingConfig::default();
        let mut sampler = Sampler::new(config);

        let logits = vec![0.1, 0.2, 0.9, 0.3];
        let token_id = sampler.greedy_sample(&logits);

        assert!(token_id.is_ok());
        assert_eq!(token_id.unwrap(), 2); // Index of highest logit
    }

    #[test]
    fn test_temperature_application() {
        let config = SamplingConfig::new(0.5, 50, 0.9).unwrap();
        let sampler = Sampler::new(config);

        let logits = vec![1.0, 2.0, 3.0];
        let scaled = sampler.apply_temperature(&logits).unwrap();

        assert_eq!(scaled, vec![2.0, 4.0, 6.0]); // Divided by 0.5
    }

    #[test]
    fn test_top_k_filtering() {
        let config = SamplingConfig::new(1.0, 2, 0.9).unwrap();
        let sampler = Sampler::new(config);

        let logits = vec![0.1, 0.9, 0.8, 0.2, 0.7];
        let top_k = sampler.apply_top_k(&logits).unwrap();

        assert_eq!(top_k.len(), 2);
        assert_eq!(top_k[0].token_id, 1); // Highest score
        assert_eq!(top_k[1].token_id, 2); // Second highest
    }

    #[test]
    fn test_repetition_penalty() {
        let config = SamplingConfig::new(1.0, 50, 0.9)
            .unwrap()
            .with_repetition_penalty(2.0);
        let sampler = Sampler::new(config);

        let logits = vec![1.0, -1.0, 0.5];
        let history = vec![0, 2];
        let penalized = sampler.apply_repetition_penalty(&logits, &history);

        assert_eq!(penalized[0], 0.5); // 1.0 / 2.0
        assert_eq!(penalized[1], -1.0); // Unchanged
        assert_eq!(penalized[2], 0.25); // 0.5 / 2.0
    }

    #[test]
    fn test_softmax() {
        let config = SamplingConfig::default();
        let sampler = Sampler::new(config);

        let scores = vec![1.0, 2.0, 3.0];
        let probabilities = sampler.softmax(&scores).unwrap();

        assert_eq!(probabilities.len(), 3);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probabilities[2] > probabilities[1] && probabilities[1] > probabilities[0]);
    }

    #[test]
    fn test_full_sampling_pipeline() {
        let config = SamplingConfig::new(0.8, 3, 0.8).unwrap();
        let mut sampler = Sampler::new(config);

        let logits = vec![0.1, 0.2, 0.9, 0.3, 0.1, 0.4, 0.2, 0.8];
        let token_id = sampler.sample(&logits);

        assert!(token_id.is_ok());
        let token_id = token_id.unwrap();
        assert!(token_id < logits.len() as u32);
    }

    #[test]
    fn test_empty_logits() {
        let config = SamplingConfig::default();
        let mut sampler = Sampler::new(config);

        let result = sampler.sample(&[]);
        assert!(result.is_err());
        assert!(matches!(result, Err(SamplerError::EmptyLogits)));
    }

    #[test]
    fn test_config_update() {
        let config = SamplingConfig::default();
        let mut sampler = Sampler::new(config);

        let new_config = SamplingConfig::new(0.5, 20, 0.7).unwrap();
        sampler.update_config(new_config);

        assert_eq!(sampler.config.temperature, 0.5);
        assert_eq!(sampler.config.top_k, 20);
        assert_eq!(sampler.config.top_p, 0.7);
    }

    // Property tests
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_sampling_properties(
            logits in prop::collection::vec(-10.0f32..10.0f32, 1..100),
            temperature in 0.1f32..2.0f32,
            top_k in 1usize..20usize,
            top_p in 0.1f32..1.0f32
        ) {
            let config = SamplingConfig::new(temperature, top_k, top_p).unwrap();
            let mut sampler = Sampler::new(config);

            // Test that sampling always returns a valid token ID
            if !logits.is_empty() {
                let token_id = sampler.sample(&logits);
                prop_assert!(token_id.is_ok());
                let token_id = token_id.unwrap();
                prop_assert!(token_id < logits.len() as u32);
            }

            // Test greedy sampling
            if !logits.is_empty() {
                let greedy_id = sampler.greedy_sample(&logits).unwrap();
                let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let max_indices: Vec<usize> = logits.iter()
                    .enumerate()
                    .filter(|(_, &logit)| (logit - max_logit).abs() < 1e-6)
                    .map(|(i, _)| i)
                    .collect();
                prop_assert!(max_indices.contains(&(greedy_id as usize)));
            }
        }
    }
}
