# Branch Protection Configuration

This document describes the required branch protection settings for the repository to ensure CI checks are mandatory for all pull requests.

## Required Branch Protection Rules

### For `main` branch:

1. **Require a pull request before merging**
   - ✅ Require approvals: 1
   - ✅ Dismiss stale pull request approvals when new commits are pushed

2. **Require status checks to pass before merging**
   - ✅ Require branches to be up to date before merging
   - **Required status checks:**
     - `Code Quality Checks`
     - `Tests (Python 3.11)`
     - `Tests (Python 3.12)`
     - `Tests (Python 3.13)`
     - `Test Coverage`

3. **Other settings**
   - ✅ Require conversation resolution before merging
   - ✅ Do not allow bypassing the above settings

### For `develop` branch (optional):

Same settings as `main` branch above.

## How to Apply These Settings

### Option 1: Via GitHub UI (Recommended)

1. Go to repository **Settings** → **Branches**
2. Click **Add branch protection rule**
3. Enter branch name pattern: `main`
4. Enable the settings listed above
5. Under "Require status checks to pass before merging":
   - Check "Require status checks to pass before merging"
   - Search and select the required status checks listed above
6. Click **Create** or **Save changes**
7. Repeat for `develop` branch if needed

### Option 2: Via GitHub CLI

```bash
# Protect main branch with required status checks
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["Code Quality Checks","Tests (Python 3.11)","Tests (Python 3.12)","Tests (Python 3.13)","Test Coverage"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
  --field restrictions=null \
  --field required_conversation_resolution=true
```

### Option 3: Via Terraform (for infrastructure as code)

```hcl
resource "github_branch_protection" "main" {
  repository_id = github_repository.repo.node_id
  pattern       = "main"

  required_status_checks {
    strict   = true
    contexts = [
      "Code Quality Checks",
      "Tests (Python 3.11)",
      "Tests (Python 3.12)",
      "Tests (Python 3.13)",
      "Test Coverage",
    ]
  }

  required_pull_request_reviews {
    required_approving_review_count = 1
    dismiss_stale_reviews          = true
  }

  enforce_admins              = true
  require_conversation_resolution = true
}
```

## Verification

After applying the branch protection rules:

1. Create a test pull request
2. Verify that the CI checks appear as required
3. Verify that you cannot merge until all checks pass
4. Verify that you need at least one approval before merging

## CI Workflow

The CI workflow is already configured in `.github/workflows/ci.yml` to run on pull requests to `main` and `develop` branches. The workflow includes:

- **Code Quality Checks**: Black formatting, Ruff linting, mypy type checking
- **Tests**: Run on Python 3.11, 3.12, and 3.13
- **Test Coverage**: Generate and upload coverage reports

## Notes

- The CI workflow runs automatically on all pull requests
- All status checks must pass before a PR can be merged (once branch protection is enabled)
- Repository administrators can configure these settings via the GitHub UI
- The exact job names in the workflow match the required status checks listed above
