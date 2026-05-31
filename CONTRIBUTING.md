# Contributing to PCCX — Bare-Metal Transformer Accelerator

Thank you for your interest in contributing to PCCX! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/pccx-FPGA-NPU-LLM-kv260.git
   cd pccx-FPGA-NPU-LLM-kv260
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- AMD/Xilinx Vivado (for FPGA synthesis)
- Python 3.8+ (for tooling and testing)
- Git

### Building

Refer to the [README](README.md) for build instructions specific to your target configuration.

## How to Contribute

### Types of Contributions

- **Bug fixes**: Fix issues in RTL, Python tooling, or documentation
- **Features**: Add new functionality to the accelerator or tooling
- **Documentation**: Improve README, add examples, or fix typos
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing implementations

### Before You Start

1. Check existing [issues](https://github.com/pccxai/pccx-FPGA-NPU-LLM-kv260/issues) for open tasks
2. Open an issue to discuss large changes before implementing
3. For small fixes (typos, minor bugs), feel free to submit a PR directly

## Pull Request Process

1. **Keep changes scoped** to one concern per PR
2. **Update documentation** when public behavior or setup changes
3. **Include tests** or validation notes when behavior changes
4. **Avoid unsupported claims** about performance, hardware, runtime, or releases
5. **Rebase** your branch on the latest `main` before submitting
6. **Write a clear PR description** explaining what and why

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Test addition/improvement

## Testing
Describe how you tested these changes.

## Checklist
- [ ] Code follows the project's coding standards
- [ ] Documentation has been updated
- [ ] Tests pass locally
```

## Coding Standards

### SystemVerilog (RTL)

- Use consistent indentation (2 spaces)
- Add comments for non-obvious logic
- Follow the existing naming conventions in `rtl/`
- Ensure timing constraints are met before submitting

### Python (Tooling)

- Follow PEP 8 style guidelines
- Add docstrings to public functions
- Include type hints where appropriate
- Keep functions focused and small

### Documentation

- Use clear, concise language
- Include code examples where helpful
- Update the README if adding new features

## Testing

### Running Tests

```bash
# Python tests
python -m pytest tests/

# RTL simulation (requires Vivado)
cd configs/<target>
make sim
```

### Writing Tests

- Add unit tests for new Python functions
- Add simulation testbenches for new RTL modules
- Ensure all tests pass before submitting a PR

## Reporting Issues

### Bug Reports

When reporting a bug, please include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Minimal steps to reproduce the behavior
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: OS, Vivado version, Python version, hardware setup

### Feature Requests

When requesting a feature, please include:

1. **Description**: Clear description of the feature
2. **Use case**: Why this feature would be useful
3. **Proposed solution**: If you have one

## Security Issues

Security issues should be reported privately as described in [SECURITY.md](SECURITY.md).

## Questions?

If you have questions about contributing, feel free to open an issue or reach out to the maintainers.

Thank you for contributing to PCCX! 🚀
