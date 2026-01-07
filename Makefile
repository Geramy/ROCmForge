# ROCmForge Makefile
#
# GPU tests require serial execution due to shared device state.

.PHONY: all build test test-lib test-docs clean help

# Default target
all: build

# Build the project
build:
	cargo build --features rocm --release

# Run tests (GPU tests require serial execution)
test:
	cargo test --features rocm --lib -- --test-threads=1

# Run tests with output
test-verbose:
	cargo test --features rocm --lib -- --test-threads=1 --nocapture

# Run only library unit tests
test-lib:
	cargo test --features rocm --lib -- --test-threads=1

# Run documentation tests
test-docs:
	cargo test --doc

# Check code without building
check:
	cargo check --features rocm

# Format code
fmt:
	cargo fmt

# Run linter
clippy:
	cargo clippy --features rocm -- -D warnings

# Clean build artifacts
clean:
	cargo clean

# Help target
help:
	@echo "ROCmForge Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make all         - Build the project (default)"
	@echo "  make build       - Build release version"
	@echo "  make test        - Run tests (serial execution for GPU)"
	@echo "  make test-verbose- Run tests with output"
	@echo "  make test-lib    - Run library unit tests"
	@echo "  make test-docs   - Run documentation tests"
	@echo "  make check       - Check code without building"
	@echo "  make fmt         - Format code"
	@echo "  make clippy      - Run linter"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make help        - Show this help"
	@echo ""
	@echo "Note: GPU tests require --test-threads=1 for reliable execution"
