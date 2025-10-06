# GitHub Actions Configuration

This document describes how to configure GitHub Actions to run automatically on all pull requests without requiring manual approval.

## Problem

By default, GitHub Actions workflows may require manual approval for:
- Pull requests from first-time contributors
- Pull requests from forked repositories
- Pull requests from users who are new to GitHub

This creates friction in the CI process and delays feedback for contributors.

## Solution

Configure the repository settings to allow workflows to run automatically on pull requests.

## Configuration Steps

### Via GitHub UI (Recommended)

1. Navigate to your repository on GitHub
2. Click **Settings** (requires admin access)
3. In the left sidebar, click **Actions** → **General**
4. Scroll down to the section **"Fork pull request workflows from outside collaborators"**
5. Choose one of the following options:

   **Option A: Balanced Security (Recommended)**
   - Select: **"Require approval for first-time contributors who are new to GitHub"**
   - This requires approval only for brand new GitHub accounts
   - Regular contributors and existing GitHub users can run workflows automatically
   
   **Option B: More Restrictive**
   - Select: **"Require approval for all outside collaborators"**
   - Requires approval for anyone not explicitly added as a collaborator
   - More secure but creates more friction
   
   **Option C: Maximum Automation (Use with Caution)**
   - Select: **"Run workflows from fork pull requests"** without restrictions
   - Workflows run automatically for all pull requests
   - ⚠️ **Security Risk**: Malicious actors could potentially run arbitrary code
   - Only recommended for public repositories with read-only CI workflows

6. Click **Save** at the bottom of the page

### Via GitHub CLI

```bash
# Enable workflows for first-time contributors who are new to GitHub (recommended)
gh api repos/:owner/:repo/actions/permissions/workflow \
  --method PUT \
  --field default_workflow_permissions=read \
  --field can_approve_pull_request_reviews=false

# Note: Specific fork approval settings are only available via UI
```

### Via Terraform

```hcl
resource "github_actions_repository_permissions" "repo" {
  repository = github_repository.repo.name
  
  allowed_actions = "all"
  enabled         = true
  
  # Note: Fork approval settings are not fully exposed in Terraform API
  # Configure these via GitHub UI as shown above
}
```

## Security Considerations

### Why This Is Safe

The CI workflow uses the `pull_request` event (not `pull_request_target`), which:
- ✅ Runs in the context of the fork (isolated environment)
- ✅ Has read-only access to the base repository
- ✅ Cannot access repository secrets
- ✅ Cannot push to the base repository
- ✅ Uses the code from the pull request branch (not the base branch)

### Best Practices

1. **Review Code Before Merging**: Always review the code changes before merging, even if CI passes
2. **Limit Secrets**: Don't use sensitive secrets in PR workflows
3. **Read-Only Operations**: Keep CI workflows to read-only operations (tests, linting, etc.)
4. **Monitor Workflow Usage**: Regularly check the Actions tab for unusual activity
5. **Use Branch Protection**: Require status checks and reviews before merging

### What to Avoid

❌ **Don't use `pull_request_target`** unless you absolutely need it and understand the security implications  
❌ **Don't expose secrets** to pull request workflows  
❌ **Don't allow automatic deployment** from pull request workflows  
❌ **Don't skip code review** just because CI passes  

## Verification

After configuring the settings:

1. Create a test pull request from a fork or new contributor
2. Check that the CI workflow starts automatically
3. Verify in the Actions tab that the workflow is running
4. Confirm no manual approval was required

## Current Workflow Configuration

The repository's CI workflow (`.github/workflows/ci.yml`) is configured with:

```yaml
on:
  pull_request:
    branches: [ main, develop ]
```

This is the **safe and recommended** configuration for running CI on pull requests.

## Troubleshooting

### Workflows Still Require Approval

- Check that you've saved the settings in **Settings** → **Actions** → **General**
- Verify you have admin access to the repository
- Try creating a PR from a different account to test
- Check the Actions tab for any error messages

### Workflows Not Running at All

- Verify the workflow file (`.github/workflows/ci.yml`) is present on the base branch
- Check that Actions are enabled for the repository
- Ensure the PR targets `main` or `develop` branch (as configured in the workflow)

## Additional Resources

- [GitHub Docs: Approving workflow runs from public forks](https://docs.github.com/en/actions/managing-workflow-runs/approving-workflow-runs-from-public-forks)
- [GitHub Docs: Events that trigger workflows](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#pull_request)
- [GitHub Security: Keeping GitHub Actions secure](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

## Quick Reference

**To run CI automatically on all PRs:**
1. Go to **Settings** → **Actions** → **General**
2. Find **"Fork pull request workflows from outside collaborators"**
3. Select **"Require approval for first-time contributors who are new to GitHub"**
4. Click **Save**

**To verify:**
```bash
# Check current Actions settings (requires gh CLI and admin access)
gh api repos/:owner/:repo/actions/permissions
```
