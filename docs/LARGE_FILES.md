# Architectural Core Files Registry

**Date**: 2026-01-11
**Policy**: Size Governance (not hard limits)

---

## Policy

### Default Rule
Target: **≤300 LOC per file**

Enforced by: CI / audit tools

### Exception Class: Architectural Core Files

Files **exceeding 300 LOC** are allowed only if **ALL** criteria are met:

| Criterion | Description |
|-----------|-------------|
| ✅ Single conceptual responsibility | One clear purpose, not a utility bag |
| ✅ No duplicated logic elsewhere | No code duplication in other files |
| ✅ Explicit module-level documentation | Clear `//!` docs explaining purpose |
| ✅ Listed in this registry | Tracked and visible |
| ✅ Periodically audited | Reviewed for relevance quarterly |

---

## Registered Architectural Core Files

### 1. `src/model/execution_plan.rs`

| Attribute | Value |
|-----------|-------|
| **LOC** | 2,429 (8.1x guideline) |
| **Responsibility** | Architecture detection, layer execution plans, weight loading/mapping |
| **Last Audited** | 2026-01-11 |
| **Status** | ✅ APPROVED - Core coordination center |

**Rationale**:
This file coordinates three tightly coupled concerns:
1. **Architecture Detection**: Auto-detects Qwen2/LLaMA/Mistral from tensor names
2. **Layer Execution Plans**: Pre-computes layer structure and weight locations
3. **Weight Loading**: Maps GGUF tensor names to layer structures

**Why not split?**
- Weight mapping logic is architecture-specific (Qwen2 ≠ LLaMA tensor names)
- Layer execution order matters (can't be hidden across modules)
- Cross-function invariant: tensor name patterns must match architecture detection

Fragmentation risk: Splitting would create hidden coupling between architecture detection and weight loading.

---

### 2. `src/backend/hip_backend.rs`

| Attribute | Value |
|-----------|-------|
| **LOC** | 2,392 (8.0x guideline) |
| **Responsibility** | All HIP FFI bindings, memory management, device operations |
| **Last Audited** | 2026-01-11 |
| **Status** | ✅ APPROVED - GPU backend coordination center |

**Rationale**:
This file contains all ROCm/HIP FFI bindings and GPU resource management:
1. **FFI Bindings**: `extern "C"` block with 30+ HIP functions
2. **Memory Management**: HipBuffer, HipStream, DeviceTensor
3. **Device Properties**: HipDeviceProp with 1472-byte buffer
4. **Error Handling**: HipError with all HIP error codes

**Why not split?**
- FFI bindings must stay together (ordering matters for C compatibility)
- Memory lifetime is cross-cutting (streams, buffers, device are coupled)
- Error handling is HIP-specific (can't be abstracted without losing safety)

Fragmentation risk: Splitting FFI from memory management would create "glue modules" with hidden invariants.

---

### 3. `src/loader/gguf.rs`

| Attribute | Value |
|-----------|-------|
| **LOC** | 2,117 (7.1x guideline) |
| **Responsibility** | GGUF format parsing, tensor loading, quantization |
| **Last Audited** | 2026-01-11 |
| **Status** | ✅ APPROVED - Model loading coordination center |

**Rationale**:
This file handles the complete GGUF model format:
1. **Format Parsing**: GGUF header, metadata, tensor registry
2. **Quantization**: Q8_0, Q4_0, FP16, FP32, MXFP4/MXFP6
3. **Memory Pooling**: Selective pooling strategy (Phase 10)
4. **Tensor Loading**: D2H copy with alignment handling

**Why not split?**
- Quantization formats are interdependent (MXFP uses E8M0 scales)
- Memory pooling strategy depends on tensor properties (size, usage pattern)
- GGUF format has cross-field invariants (tensor count matches tensor registry)

Fragmentation risk: Splitting quantization formats would scatter format-specific logic.

---

## Audit History

| Date | Action | By |
|------|--------|-----|
| 2026-01-11 | Initial registry created, 3 files evaluated | Code Review |
| TBD | Quarterly audit scheduled | - |

---

## Criteria for Removal

A file should be **removed from this registry** and **refactored/split** if:

1. ❌ Responsibility has drifted (multiple concerns now)
2. ❌ Code duplication found elsewhere
3. ❌ Documentation becomes stale/unclear
4. ❌ Better abstraction pattern emerges

## Criteria for Addition

A file may be **added to this registry** if:

1. ✅ Exceeds 300 LOC
2. ✅ Has single conceptual responsibility
3. ✅ Has clear module-level documentation
4. ✅ Passes review showing fragmentation would be harmful

---

**Next Audit**: 2026-04-11 (Quarterly)
