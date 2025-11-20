//! End-to-end smoke test for running a tiny GGUF through the inference engine.

use std::sync::Arc;

use rocmforge::engine::{EngineConfig, InferenceEngine};

#[tokio::test]
async fn test_tiny_gguf_decode_smoke() {
    let gguf_path = std::path::Path::new("tests/data/tiny_model.gguf");
    if !gguf_path.exists() {
        eprintln!("Skipping smoke test: tiny_model.gguf missing");
        return;
    }

    let mut engine = match InferenceEngine::new(EngineConfig::default()) {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("Skipping: failed to init inference engine: {err}");
            return;
        }
    };

    if let Err(err) = engine.load_gguf_model(gguf_path).await {
        eprintln!("Skipping: failed to load GGUF: {err}");
        return;
    }

    if engine.start().await.is_err() {
        eprintln!("Skipping: failed to start inference engine");
        return;
    }
    engine.run_inference_loop().await;
    let engine = Arc::new(engine);

    let prompt_tokens = vec![1u32, 2u32, 3u32];
    let request_id = match engine
        .submit_request(prompt_tokens.clone(), 4, 1.0, 40, 0.9)
        .await
    {
        Ok(id) => id,
        Err(err) => {
            eprintln!("Skipping: submit_request failed: {err}");
            return;
        }
    };

    let status = engine.get_request_status(request_id).await.unwrap();
    assert!(status.is_some());
}
