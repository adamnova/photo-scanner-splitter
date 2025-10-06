#!/bin/bash
# CI Setup Verification Script
# This script verifies that the CI workflow is properly configured

set -e

echo "🔍 Verifying CI Setup for photo-scanner-splitter"
echo "================================================="
echo ""

# Check if workflow file exists
echo "✓ Checking if CI workflow file exists..."
if [ -f ".github/workflows/ci.yml" ]; then
    echo "  ✅ Found .github/workflows/ci.yml"
else
    echo "  ❌ CI workflow file not found!"
    exit 1
fi

# Validate YAML syntax
echo ""
echo "✓ Validating CI workflow YAML syntax..."
if python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" 2>/dev/null; then
    echo "  ✅ YAML syntax is valid"
else
    echo "  ❌ YAML syntax is invalid!"
    exit 1
fi

# Check for pull_request trigger
echo ""
echo "✓ Checking if workflow runs on pull requests..."
if grep -q "pull_request:" ".github/workflows/ci.yml"; then
    echo "  ✅ Workflow is configured to run on pull requests"
else
    echo "  ❌ Workflow is not configured to run on pull requests!"
    exit 1
fi

# List configured jobs
echo ""
echo "✓ Configured CI jobs:"
grep -E "^  [a-z-]+:" ".github/workflows/ci.yml" | grep -v "push:\|permissions:\|contents:" | while read -r job; do
    job_id=$(echo "$job" | sed 's/:$//')
    job_name=$(grep -A 1 "$job" ".github/workflows/ci.yml" | grep "name:" | head -1 | sed 's/.*name: //' | sed 's/"//g')
    echo "  - $job_name (ID: $job_id)"
done

# Check for branch protection documentation
echo ""
echo "✓ Checking for branch protection documentation..."
if [ -f ".github/branch-protection.md" ]; then
    echo "  ✅ Found .github/branch-protection.md"
    echo "  📖 Review this file for instructions on enabling required status checks"
else
    echo "  ⚠️  Branch protection documentation not found"
fi

echo ""
echo "================================================="
echo "✅ CI Setup Verification Complete!"
echo ""
echo "Next steps for repository administrators:"
echo "1. Review .github/branch-protection.md"
echo "2. Enable branch protection rules via GitHub Settings"
echo "3. Make CI checks required for all pull requests"
echo ""
echo "For contributors:"
echo "- Run 'make ci' locally before pushing"
echo "- All CI checks must pass before merging"
echo ""
