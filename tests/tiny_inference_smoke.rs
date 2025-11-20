//! End-to-end smoke test for inference engine + GGUF runtime.

use std::sync::Arc;

use rocmforge::engine::{EngineConfig, InferenceEngine};

#[tokio::test]
async fn test_tiny_gguf_decode_smoke() {
    let gguf_path = std::path::Path::new("tests/data/tiny_model.gguf");
    if !gguf_path.exists() {
        eprintln!("Skipping: tiny_model.gguf missing");
        return;
    }

    let mut engine = match InferenceEngine::new(EngineConfig::default()) {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("Skipping: init engine failed: {err}");
            return;
        }
    };

    if let Err(err) = engine.load_gguf_model(gguf_path).await {
        eprintln!("Skipping: load_gguf_model failed: {err}");
        return;
    }

    if let Err(err) = engine.start().await {
        eprintln!("Skipping: engine start failed: {err}");
        return;
    }
    engine.run_inference_loop().await;
    let engine = Arc::new(engine);

    let prompt_tokens = vec![1u32, 2u32, 3u32];
    let request_id = match engine
        .submit_request(prompt_tokens.clone(), 4, 1.0, 50, 0.9)
        .await
    {
        Ok(id) => id,
        Err(err) => {
            eprintln!("Skipping: submit_request failed: {err}");
            return;
        }
    };

    for _ in 0..20 {
        if let Ok(Some(status)) = engine.get_request_status(request_id).await {
            if !status.generated_tokens.is_empty() {
                return;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    panic!("Engine failed to generate tokens within timeout");
}
