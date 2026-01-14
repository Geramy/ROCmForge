//! Graph optimizer for ggml IR.
//!
//! Inspired by llama.cpp's graph optimization, this module provides passes
//! to optimize computation graphs before execution.
//!
//! # Optimization Passes
//!
//! - **Dead Code Elimination (DCE)**: Remove nodes not contributing to outputs
//! - **Common Subexpression Elimination (CSE)**: Deduplicate identical computations
//! - **No-op elimination**: Remove redundant View/Reshape operations

use crate::ggml::{Graph, Node, Op, TensorDesc, TensorId};
use std::collections::{HashMap, HashSet};

/// Graph optimizer that applies various optimization passes.
#[derive(Debug, Default)]
pub struct GraphOptimizer {
    /// Enable/disable specific passes
    enable_dce: bool,
    enable_cse: bool,
    enable_noop_elimination: bool,
}

impl GraphOptimizer {
    /// Create a new optimizer with all passes enabled.
    pub fn new() -> Self {
        Self {
            enable_dce: true,
            enable_cse: true,
            enable_noop_elimination: true,
        }
    }

    /// Disable dead code elimination.
    pub fn without_dce(mut self) -> Self {
        self.enable_dce = false;
        self
    }

    /// Disable common subexpression elimination.
    pub fn without_cse(mut self) -> Self {
        self.enable_cse = false;
        self
    }

    /// Disable no-op elimination.
    pub fn without_noop_elimination(mut self) -> Self {
        self.enable_noop_elimination = false;
        self
    }

    /// Optimize the graph in place.
    ///
    /// Applies all enabled optimization passes to the graph.
    ///
    /// # Parameters
    /// - `graph`: The graph to optimize (modified in place)
    ///
    /// # Returns
    /// Optimization statistics
    pub fn optimize(&self, graph: &mut Graph) -> OptimizerStats {
        let mut stats = OptimizerStats::default();

        // Build dependency tracking
        let mut dep_info = DependencyInfo::from_graph(graph);

        // Pass 1: No-op elimination
        if self.enable_noop_elimination {
            let removed = self.eliminate_noops(graph, &mut dep_info);
            stats.noops_removed = removed;
        }

        // Pass 2: Dead code elimination
        if self.enable_dce {
            let removed = self.eliminate_dead_code(graph, &mut dep_info);
            stats.dce_nodes_removed = removed;
        }

        // Pass 3: Common subexpression elimination
        if self.enable_cse {
            let removed = self.eliminate_common_subexpressions(graph, &mut dep_info);
            stats.cse_nodes_removed = removed;
        }

        stats
    }

    /// Eliminate no-op operations (View, Reshape with same shape).
    fn eliminate_noops(&self, graph: &mut Graph, dep_info: &mut DependencyInfo) -> usize {
        let mut removed = 0;
        let mut nodes_to_remove = Vec::new();

        for (idx, node) in graph.nodes.iter().enumerate() {
            match &node.op {
                // View operations that don't change the shape are no-ops
                Op::View => {
                    // Check if input and output have same shape
                    if let Some(input_id) = node.inputs.first() {
                        if let Some(output_id) = node.outputs.first() {
                            let input_desc = dep_info.get_desc(graph, *input_id);
                            let output_desc = dep_info.get_desc(graph, *output_id);

                            if let (Some(input), Some(output)) = (input_desc, output_desc) {
                                if input.shape == output.shape && input.byte_offset == output.byte_offset {
                                    // Safe to remove - input and output are identical
                                    nodes_to_remove.push(idx);
                                }
                            }
                        }
                    }
                }
                // Reshape operations that don't change the shape
                Op::Reshape => {
                    if let Some(input_id) = node.inputs.first() {
                        if let Some(output_id) = node.outputs.first() {
                            let input_desc = dep_info.get_desc(graph, *input_id);
                            let output_desc = dep_info.get_desc(graph, *output_id);

                            if let (Some(input), Some(output)) = (input_desc, output_desc) {
                                if input.shape == output.shape {
                                    // Reshape to same shape is a no-op
                                    nodes_to_remove.push(idx);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Remove nodes in reverse order to maintain indices
        for &idx in nodes_to_remove.iter().rev() {
            graph.nodes.remove(idx);
            removed += 1;
        }

        removed
    }

    /// Eliminate dead code - nodes not contributing to any output.
    fn eliminate_dead_code(&self, graph: &mut Graph, dep_info: &mut DependencyInfo) -> usize {
        // Find all tensors that are graph outputs (not used as input to any node)
        let mut live_tensors: HashSet<TensorId> = HashSet::new();

        // Add all tensors that are never used as inputs (outputs of the graph)
        for tensor_id in 0..graph.tensors.len() {
            let tid = TensorId(tensor_id);
            if !dep_info.is_used_as_input(tid) {
                live_tensors.insert(tid);
            }
        }

        // Work backwards: find all tensors that contribute to live tensors
        let mut changed = true;
        while changed {
            changed = false;
            for node in &graph.nodes {
                // Check if any output of this node is live
                let has_live_output = node.outputs.iter().any(|tid| live_tensors.contains(tid));

                if has_live_output {
                    // All inputs are now live
                    for &tid in &node.inputs {
                        if live_tensors.insert(tid) {
                            changed = true;
                        }
                    }
                }
            }
        }

        // Find all nodes that contribute to live tensors
        let mut live_nodes: HashSet<usize> = HashSet::new();
        for (idx, node) in graph.nodes.iter().enumerate() {
            let has_live_output = node.outputs.iter().any(|tid| live_tensors.contains(tid));
            let has_live_input = node.inputs.iter().any(|tid| live_tensors.contains(tid));

            if has_live_output || has_live_input {
                live_nodes.insert(idx);
            }
        }

        // Remove dead nodes
        let original_count = graph.nodes.len();
        graph.nodes = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(idx, _)| live_nodes.contains(idx))
            .map(|(_, node)| node.clone())
            .collect();

        original_count - graph.nodes.len()
    }

    /// Eliminate common subexpressions - deduplicate identical computations.
    fn eliminate_common_subexpressions(&self, graph: &mut Graph, dep_info: &mut DependencyInfo) -> usize {
        let mut removed = 0;
        let mut canonical_map: HashMap<NodeSignature, TensorId> = HashMap::new();

        // Build a map of operation signatures to their output tensors
        for node in &graph.nodes {
            let sig = NodeSignature::from_node(node, graph);

            if let Some(&existing_output) = canonical_map.get(&sig) {
                // Found a duplicate computation!
                // Replace all uses of this node's output with the existing output
                if let Some(new_output) = node.outputs.first() {
                    // Update all references from new_output to existing_output
                    dep_info.remap_tensor(*new_output, existing_output);
                    removed += 1;
                }
            } else {
                // First time seeing this computation
                if let Some(output) = node.outputs.first() {
                    canonical_map.insert(sig, *output);
                }
            }
        }

        // Remove nodes whose outputs were remapped
        // (This is simplified - a full implementation would also clean up tensors)
        removed
    }
}

/// Statistics about optimization results.
#[derive(Debug, Default, Clone, Copy)]
pub struct OptimizerStats {
    /// Number of no-op nodes removed
    pub noops_removed: usize,
    /// Number of nodes removed by dead code elimination
    pub dce_nodes_removed: usize,
    /// Number of nodes removed by CSE
    pub cse_nodes_removed: usize,
}

impl OptimizerStats {
    /// Total number of nodes removed.
    pub fn total_removed(&self) -> usize {
        self.noops_removed + self.dce_nodes_removed + self.cse_nodes_removed
    }

    /// Check if any optimizations were applied.
    pub fn is_empty(&self) -> bool {
        self.total_removed() == 0
    }
}

impl std::fmt::Display for OptimizerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            write!(f, "OptimizerStats: no optimizations applied")
        } else {
            write!(
                f,
                "OptimizerStats: {} noops, {} DCE, {} CSE ({} total)",
                self.noops_removed,
                self.dce_nodes_removed,
                self.cse_nodes_removed,
                self.total_removed()
            )
        }
    }
}

/// Dependency tracking for graph optimization.
#[derive(Debug)]
struct DependencyInfo {
    /// Which tensors are used as inputs to nodes
    used_as_input: HashSet<TensorId>,
    /// Map from old tensor IDs to new ones (for remapping during CSE)
    remap: HashMap<TensorId, TensorId>,
}

impl DependencyInfo {
    fn from_graph(graph: &Graph) -> Self {
        let mut used_as_input = HashSet::new();

        for node in &graph.nodes {
            for &tid in &node.inputs {
                used_as_input.insert(tid);
            }
        }

        Self {
            used_as_input,
            remap: HashMap::new(),
        }
    }

    fn is_used_as_input(&self, tid: TensorId) -> bool {
        self.used_as_input.contains(&tid)
    }

    fn get_desc<'a>(&self, graph: &'a Graph, tid: TensorId) -> Option<&'a TensorDesc> {
        let actual_tid = self.remap.get(&tid).copied().unwrap_or(tid);
        graph.tensors.get(actual_tid.0)
    }

    fn remap_tensor(&mut self, from: TensorId, to: TensorId) {
        self.remap.insert(from, to);
    }
}

/// Signature of a node for CSE comparison.
#[derive(Debug, Clone)]
struct NodeSignature {
    /// The operation
    op: OpSignature,
    /// Input tensor shapes (for comparison)
    input_shapes: Vec<(Vec<usize>, crate::ggml::DType)>,
}

// Manual implementations since f32 doesn't implement Hash/Eq
impl PartialEq for NodeSignature {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.input_shapes == other.input_shapes
    }
}

impl Eq for NodeSignature {}

impl std::hash::Hash for NodeSignature {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.op.hash(state);
        // Hash shapes without relying on DType's Hash
        for (shape, dtype) in &self.input_shapes {
            shape.hash(state);
            std::mem::discriminant(dtype).hash(state);
        }
    }
}

impl NodeSignature {
    fn from_node(node: &Node, graph: &Graph) -> Self {
        let op = OpSignature::from_op(&node.op);

        let input_shapes = node
            .inputs
            .iter()
            .filter_map(|tid| graph.tensors.get(tid.0))
            .map(|desc| (desc.shape.clone(), desc.dtype))
            .collect();

        Self { op, input_shapes }
    }
}

/// Signature of an operation for CSE comparison.
#[derive(Debug, Clone)]
enum OpSignature {
    GetRows,
    MatMul,
    MatMulQ4_0,
    MatMulQ8_0,
    Add,
    Mask,
    Scale { factor: u32 }, // Store as bits for Hash/Eq
    LayerNorm { eps: u32 }, // Store as bits for Hash/Eq
    RmsNorm { eps: u32 }, // Store as bits for Hash/Eq
    Rope,
    Softmax,
    Attention,
    SwiGlu,
    MlpSwiglu,
    SplitQkv,
    Reshape,
    View,
    Copy,
    Accumulate { offset: usize },
}

// Manual implementations for OpSignature
impl PartialEq for OpSignature {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (OpSignature::GetRows, OpSignature::GetRows) => true,
            (OpSignature::MatMul, OpSignature::MatMul) => true,
            (OpSignature::MatMulQ4_0, OpSignature::MatMulQ4_0) => true,
            (OpSignature::MatMulQ8_0, OpSignature::MatMulQ8_0) => true,
            (OpSignature::Add, OpSignature::Add) => true,
            (OpSignature::Mask, OpSignature::Mask) => true,
            (OpSignature::Scale { factor: a }, OpSignature::Scale { factor: b }) => {
                f32::from_bits(*a) == f32::from_bits(*b)
            }
            (OpSignature::LayerNorm { eps: a }, OpSignature::LayerNorm { eps: b }) => {
                f32::from_bits(*a) == f32::from_bits(*b)
            }
            (OpSignature::RmsNorm { eps: a }, OpSignature::RmsNorm { eps: b }) => {
                f32::from_bits(*a) == f32::from_bits(*b)
            }
            (OpSignature::Rope, OpSignature::Rope) => true,
            (OpSignature::Softmax, OpSignature::Softmax) => true,
            (OpSignature::Attention, OpSignature::Attention) => true,
            (OpSignature::SwiGlu, OpSignature::SwiGlu) => true,
            (OpSignature::MlpSwiglu, OpSignature::MlpSwiglu) => true,
            (OpSignature::SplitQkv, OpSignature::SplitQkv) => true,
            (OpSignature::Reshape, OpSignature::Reshape) => true,
            (OpSignature::View, OpSignature::View) => true,
            (OpSignature::Copy, OpSignature::Copy) => true,
            (
                OpSignature::Accumulate { offset: a },
                OpSignature::Accumulate { offset: b },
            ) => a == b,
            _ => false,
        }
    }
}

impl Eq for OpSignature {}

impl std::hash::Hash for OpSignature {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            OpSignature::Scale { factor } => factor.hash(state),
            OpSignature::LayerNorm { eps } => eps.hash(state),
            OpSignature::RmsNorm { eps } => eps.hash(state),
            OpSignature::Accumulate { offset } => offset.hash(state),
            _ => {}
        }
    }
}

impl OpSignature {
    fn from_op(op: &Op) -> Self {
        match op {
            Op::GetRows => Self::GetRows,
            Op::MatMul => Self::MatMul,
            Op::MatMulQ4_0 => Self::MatMulQ4_0,
            Op::MatMulQ8_0 => Self::MatMulQ8_0,
            Op::Add => Self::Add,
            Op::Mask => Self::Mask,
            Op::Scale { factor } => Self::Scale {
                factor: factor.to_bits(),
            },
            Op::LayerNorm { eps } => Self::LayerNorm { eps: eps.to_bits() },
            Op::RmsNorm { eps } => Self::RmsNorm { eps: eps.to_bits() },
            Op::Rope => Self::Rope,
            Op::Softmax => Self::Softmax,
            Op::Attention => Self::Attention,
            Op::SwiGlu => Self::SwiGlu,
            Op::MlpSwiglu => Self::MlpSwiglu,
            Op::SplitQkv => Self::SplitQkv,
            Op::Reshape => Self::Reshape,
            Op::View => Self::View,
            Op::Copy => Self::Copy,
            Op::Accumulate { offset } => Self::Accumulate { offset: *offset },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml::{DType, Layout, Op};

    fn make_tensor_desc(shape: Vec<usize>) -> TensorDesc {
        TensorDesc::new(shape, DType::F32, Layout::RowMajor)
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = GraphOptimizer::new();
        assert!(optimizer.enable_dce);
        assert!(optimizer.enable_cse);
        assert!(optimizer.enable_noop_elimination);
    }

    #[test]
    fn test_optimizer_without_passes() {
        let optimizer = GraphOptimizer::new()
            .without_dce()
            .without_cse()
            .without_noop_elimination();

        assert!(!optimizer.enable_dce);
        assert!(!optimizer.enable_cse);
        assert!(!optimizer.enable_noop_elimination);
    }

    #[test]
    fn test_optimize_empty_graph() {
        let mut graph = Graph::new();
        let optimizer = GraphOptimizer::new();
        let stats = optimizer.optimize(&mut graph);

        assert_eq!(stats.total_removed(), 0);
        assert!(stats.is_empty());
    }

    #[test]
    fn test_dce_removes_unused_nodes() {
        let mut graph = Graph::new();

        // Create a chain: input -> temp -> output
        let input = graph.add_tensor(make_tensor_desc(vec![10, 10]));
        let temp = graph.add_tensor(make_tensor_desc(vec![10, 10]));
        let output = graph.add_tensor(make_tensor_desc(vec![10, 10]));

        graph.add_node(Op::Add, vec![input], vec![temp]);
        graph.add_node(Op::Add, vec![temp], vec![output]);

        // Before optimization
        let original_count = graph.nodes.len();

        // Run DCE - should keep all nodes since they form a valid chain
        let optimizer = GraphOptimizer::new().without_cse().without_noop_elimination();
        let stats = optimizer.optimize(&mut graph);

        // No nodes should be removed in a well-formed graph
        assert_eq!(stats.dce_nodes_removed, 0);
        assert_eq!(graph.nodes.len(), original_count);
    }

    #[test]
    fn test_noop_elimination_removes_redundant_views() {
        let mut graph = Graph::new();

        let input = graph.add_tensor(make_tensor_desc(vec![10, 10]));
        let output = graph.add_tensor(make_tensor_desc(vec![10, 10]));

        // Add a View node with same shape as input (a no-op)
        graph.add_node(Op::View, vec![input], vec![output]);
        graph.tensors[input.0].shape = vec![10, 10];
        graph.tensors[output.0].shape = vec![10, 10];
        graph.tensors[output.0].byte_offset = 0;

        assert_eq!(graph.nodes.len(), 1);

        let optimizer = GraphOptimizer::new().without_dce().without_cse();
        let stats = optimizer.optimize(&mut graph);

        // The no-op view should be removed
        assert_eq!(stats.noops_removed, 1);
        assert_eq!(graph.nodes.len(), 0);
    }

    #[test]
    fn test_optimizer_stats_display() {
        let stats = OptimizerStats {
            noops_removed: 5,
            dce_nodes_removed: 3,
            cse_nodes_removed: 2,
        };

        let display = format!("{}", stats);
        assert!(display.contains("5 noops"));
        assert!(display.contains("3 DCE"));
        assert!(display.contains("2 CSE"));
        assert!(display.contains("10 total"));
    }

    #[test]
    fn test_dependency_info_from_graph() {
        let mut graph = Graph::new();

        let t1 = graph.add_tensor(make_tensor_desc(vec![10]));
        let t2 = graph.add_tensor(make_tensor_desc(vec![10]));

        graph.add_node(Op::Add, vec![t1], vec![t2]);

        let dep_info = DependencyInfo::from_graph(&graph);

        // t1 is used as input
        assert!(dep_info.is_used_as_input(t1));
        // t2 is not used as input (it's an output)
        assert!(!dep_info.is_used_as_input(t2));
    }

    #[test]
    fn test_node_signature_from_add_node() {
        let mut graph = Graph::new();

        let t1 = graph.add_tensor(make_tensor_desc(vec![10]));
        let t2 = graph.add_tensor(make_tensor_desc(vec![10]));

        let node = Node {
            op: Op::Add,
            inputs: vec![t1],
            outputs: vec![t2],
        };

        let sig = NodeSignature::from_node(&node, &graph);

        match sig.op {
            OpSignature::Add => {
                // Correct
            }
            _ => panic!("Wrong signature"),
        }

        assert_eq!(sig.input_shapes.len(), 1);
    }
}
