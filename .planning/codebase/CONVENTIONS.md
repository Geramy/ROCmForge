# Coding Conventions

**Analysis Date:** 2026-01-18

## Naming Patterns

**Files:**
- snake_case.rs for all modules (e.g., `hip_backend.rs`, `matmul.rs`)
- {name}_tests.rs for test files (e.g., `hip_blas_matmul_tests.rs`)
- mod.rs for directory exports

**Functions:**
- camelCase for all functions (e.g., `validate_matmul_dims`, `load_gguf`)
- No special prefix for async functions
- Descriptive names preferred over abbreviations

**Variables:**
- camelCase for variables
- SCREAMING_SNAKE_CASE for constants
- No underscore prefix for private members (Rust's `pub` controls visibility)

**Types:**
- PascalCase for structs and enums (e.g., `DeviceTensor`, `HipError`)
- No `I` prefix for interfaces (Rust uses traits)
- PascalCase for type aliases

## Code Style

**Formatting:**
- Standard Rust 4-space indentation
- Double quotes for string literals
- Semicolons required where expected by Rust
- Same-line opening braces for structs/impls
- Next-line opening braces for control flow

**Linting:**
- Clippy with extensive allowances for GPU-specific patterns
- Allowances in `src/lib.rs`:
  - too_many_arguments (common in FFI)
  - manual_slice_size_calculation (GPU kernel pattern)
  - needless_range_loop (clearer for GPU ops)
  - And 12 other GPU-specific allowances
- No `.rustfmt.toml` or `.clippy.toml` found (using defaults)

## Import Organization

**Order:**
1. External crates (std, external dependencies)
2. Internal modules (crate::)
3. re-exports (pub use)

**Grouping:**
- Blank lines between groups
- Alphabetical within each group (typical Rust convention)

**Path Aliases:**
- No path aliases configured
- Uses `crate::` for internal imports

## Error Handling

**Patterns:**
- Custom error types via thiserror crate
- All error types derive `Debug` and `Error`
- Extensive use of `unwrap()` and `expect()` (technical debt)

**Error Types:**
- Backend-specific errors: `HipError`, `GgufError`
- Centralized error definitions with specific variants
- Location: `src/backend/hip_backend.rs:9-50`

## Logging

**Framework:**
- tracing crate for structured logging
- Levels: debug, info, warn, error

**Patterns:**
- Structured logging with context: `tracing::debug({...}, "message")`
- Log at service boundaries, not utilities
- Some temporary debug prints remain (e.g., in matmul operations)

## Comments

**When to Comment:**
- Explain why, not what
- Document GPU-specific optimizations
- Explain unsafe blocks (inconsistent)
- TODO/FIXME comments for phase-specific fixes

**JSDoc/TSDoc:**
- Rustdoc: `//!` for module-level documentation
- Rustdoc: `///` for item-level documentation
- 101 files with module-level docs
- 68 files with item-level docs

**TODO Comments:**
- Format: `// TODO: description` or `// FIX-10: description`
- 10 TODO/FIXME comments across codebase
- Phase-specific fix comments: `// PHASE 24 FIX:`

## Function Design

**Size:**
- No strict limit enforced
- Some files exceed 300 LOC significantly (e.g., `kv_cache.rs` at 1,439 lines)

**Parameters:**
- No explicit limit on parameter count
- Many functions with 8+ parameters (FFI-related)

**Return Values:**
- Result<T, E> for fallible operations
- Explicit return statements
- Early returns for guard clauses

## Module Design

**Exports:**
- Named exports preferred (pub fn, pub struct)
- Re-exports via `pub use` in mod.rs files
- Clear public API in `src/lib.rs`

**Barrel Files:**
- mod.rs files for directory exports
- Re-export public API
- Avoid circular dependencies

---

*Convention analysis: 2026-01-18*
*Update when patterns change*
