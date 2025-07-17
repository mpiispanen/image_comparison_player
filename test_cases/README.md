# Test Cases

This directory contains test cases for the image comparison player visual diff testing.

## Structure

- `test_cases/` - Contains test case definitions and configurations
- `test_images/` - Generated during CI, contains:
  - `input/dir1/` - First set of input images for comparison
  - `input/dir2/` - Second set of input images for comparison  
  - `reference/` - Reference images showing expected output
  - `current/` - Current test run outputs (generated during testing)

## Test Generation vs Testing Separation

### Test Image Generation (NOT part of visual-diff job)
- Handled by the `generate-test-images.yml` workflow
- Runs the `test-image-generator` binary
- Creates input images and reference images
- Stores results as CI artifacts
- Runs separately from the main pipeline to avoid generating images during testing

### Visual Diff Testing (visual-diff job)
- Handled by the `visual-diff.yml` workflow  
- Downloads pre-generated test images from artifacts
- Runs the `visual-diff-test` binary
- Compares current outputs against reference images
- **Does NOT generate any images** - only tests against existing ones

## Adding New Test Cases

1. Add test case configuration to this directory
2. Update the `test-image-generator` to handle the new test case
3. Run the "Generate Test Images" workflow to create new reference images
4. The visual-diff tests will automatically pick up the new test cases

## Workflow Dependencies

```
generate-test-images.yml (creates artifacts)
           ↓
visual-diff.yml (downloads and tests against artifacts)
```

This separation ensures that:
- Visual diff testing is fast and doesn't regenerate images
- Test image generation happens independently 
- Reference images are versioned and stored as artifacts
- The main CI pipeline focuses on testing, not generation