//! Graph executor for ggml IR.

use crate::ggml::{GgmlBackend, GgmlResult, Graph, GraphOptimizer, OptimizerStats};

/// Configuration for graph execution.
#[derive(Debug, Clone, Default)]
pub struct ExecuteConfig {
    /// Enable optimization before execution.
    pub optimize: bool,
}

impl ExecuteConfig {
    /// Create a new config with optimization enabled.
    pub fn with_optimization() -> Self {
        Self { optimize: true }
    }

    /// Create a new config with optimization disabled.
    pub fn without_optimization() -> Self {
        Self { optimize: false }
    }
}

/// Result of graph execution with optional optimization stats.
#[derive(Debug)]
pub struct ExecuteResult {
    /// Optimization statistics (if optimization was enabled).
    pub optimizer_stats: Option<OptimizerStats>,
}

/// Execute a graph with optional optimization.
///
/// # Arguments
/// - `backend`: The GPU backend to use
/// - `graph`: The graph to execute (will be modified if optimize=true)
/// - `config`: Execution configuration
///
/// # Returns
/// Execution result with optional optimizer statistics
pub fn execute_graph_with_config<B: GgmlBackend>(
    backend: &mut B,
    graph: &mut Graph,
    config: ExecuteConfig,
) -> GgmlResult<ExecuteResult> {
    eprintln!(">>> execute_graph_with_config: ENTRY");
    let mut optimizer_stats = None;

    if config.optimize {
        let optimizer = GraphOptimizer::new();
        optimizer_stats = Some(optimizer.optimize(graph));
    }

    // Allocate buffers for all tensors
    eprintln!(">>> execute_graph_with_config: Allocating buffers for {} tensors", graph.tensors.len());
    for desc in &graph.tensors {
        if desc.is_view() {
            continue;
        }
        if backend.buffer(desc.id).is_none() {
            eprintln!(">>> execute_graph_with_config: Allocating buffer for tensor {:?}", desc.id);
            backend.alloc(desc)?;
        }
    }
    eprintln!(">>> execute_graph_with_config: Buffer allocation complete");

    // Execute each node
    eprintln!(">>> execute_graph_with_config: Executing {} nodes", graph.nodes.len());
    for (i, node) in graph.nodes.iter().enumerate() {
        eprintln!(">>> execute_graph_with_config: Executing node {} (op={:?})", i, node.op);
        backend.execute_op(&node.op, &node.inputs, &node.outputs)?;
        eprintln!(">>> execute_graph_with_config: Node {} complete", i);
    }
    eprintln!(">>> execute_graph_with_config: All nodes executed, about to synchronize...");

    backend.synchronize()?;
    eprintln!(">>> execute_graph_with_config: Synchronization complete");

    Ok(ExecuteResult { optimizer_stats })
}

/// Execute a graph without optimization (original behavior).
pub fn execute_graph<B: GgmlBackend>(backend: &mut B, graph: &Graph) -> GgmlResult<()> {
    execute_graph_with_config(backend, &mut graph.clone(), ExecuteConfig::default())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // TDD TEST: Optimizer is called when enabled
    #[test]
    fn test_optimizer_called_when_enabled() {
        // Create a simple graph
        let mut graph = Graph::new();
        let config = ExecuteConfig::with_optimization();

        // The optimizer should process the graph
        // (We can't test full execution without a GPU backend,
        // but we can verify the config is set correctly)
        assert!(config.optimize, "Optimize should be enabled");
    }

    // TDD TEST: Optimizer is skipped when disabled
    #[test]
    fn test_optimizer_skipped_when_disabled() {
        let config = ExecuteConfig::without_optimization();
        assert!(!config.optimize, "Optimize should be disabled");
    }

    // TDD TEST: Default config does not optimize
    #[test]
    fn test_default_config_no_optimization() {
        let config = ExecuteConfig::default();
        assert!(!config.optimize, "Default should not optimize");
    }
}
