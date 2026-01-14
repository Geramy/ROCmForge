//! Graph structures for ggml IR.

use crate::ggml::{Op, TensorDesc, TensorId};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

#[derive(Debug, Default, Clone)]
pub struct Graph {
    pub tensors: Vec<TensorDesc>,
    pub nodes: Vec<Node>,
    /// Explicitly marked output tensors (for DCE)
    outputs: HashSet<TensorId>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_tensor(&mut self, mut desc: TensorDesc) -> TensorId {
        let id = TensorId(self.tensors.len());
        desc.id = id;
        self.tensors.push(desc);
        id
    }

    pub fn add_node(&mut self, op: Op, inputs: Vec<TensorId>, outputs: Vec<TensorId>) {
        self.nodes.push(Node { op, inputs, outputs });
    }

    /// Mark a tensor as a graph output.
    ///
    /// Dead code elimination will preserve nodes that contribute
    /// to marked outputs. Unmarked tensors that aren't used as inputs
    /// will be considered dead code.
    pub fn mark_output(&mut self, tensor_id: TensorId) {
        self.outputs.insert(tensor_id);
    }

    /// Check if a tensor is marked as a graph output.
    pub fn is_output(&self, tensor_id: TensorId) -> bool {
        self.outputs.contains(&tensor_id)
    }

    /// Get all marked output tensors.
    pub fn get_outputs(&self) -> &HashSet<TensorId> {
        &self.outputs
    }

    /// Clear all marked outputs.
    pub fn clear_outputs(&mut self) {
        self.outputs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mark_and_check_output() {
        let mut graph = Graph::new();
        let tid = TensorId(0);

        assert!(!graph.is_output(tid), "Tensor should not be output initially");

        graph.mark_output(tid);
        assert!(graph.is_output(tid), "Tensor should be marked as output");
    }

    #[test]
    fn test_clear_outputs() {
        let mut graph = Graph::new();
        let tid = TensorId(0);

        graph.mark_output(tid);
        assert!(graph.is_output(tid));

        graph.clear_outputs();
        assert!(!graph.is_output(tid), "Output should be cleared");
    }
}
