# .github Configuration

This directory contains GitHub-specific configuration files for the repository.

## Files

### workflows/
Contains GitHub Actions workflow definitions:
- **ci.yml**: Continuous Integration pipeline that runs on every push and pull request

### agents/
Contains instructions for AI/LLM agents working on this repository:
- **instructions.md**: Guidelines for code changes, style, and testing

### branch-protection.md
Documentation and configuration guide for setting up required status checks and branch protection rules.

**Important**: Repository administrators should review and apply the branch protection settings documented in `branch-protection.md` to enforce CI requirements on all pull requests.

### ACTIONS_CONFIGURATION.md
Documentation for configuring GitHub Actions to run automatically on all pull requests without requiring manual approval.

**Important**: Repository administrators should configure Actions settings to allow CI to run on all PRs as documented in `ACTIONS_CONFIGURATION.md`.

### verify-ci-setup.sh
Verification script to check that CI is properly configured. Run with:
```bash
./.github/verify-ci-setup.sh
```

## Quick Links

- [CI/CD Pipeline Documentation](../DEVELOPMENT.md#cicd-pipeline)
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Development Guide](../DEVELOPMENT.md)
