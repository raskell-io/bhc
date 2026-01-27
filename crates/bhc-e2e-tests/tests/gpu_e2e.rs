//! GPU backend E2E tests.
//!
//! These tests verify GPU code generation. By default, tests run in mock mode
//! which validates PTX output without requiring CUDA hardware.
//!
//! To run with real CUDA hardware:
//! ```bash
//! cargo test -p bhc-e2e-tests --test gpu_e2e --features cuda -- --ignored
//! ```

use bhc_e2e_tests::{format_failure_report, Backend, E2ERunner, Profile};

/// Run a single fixture test for GPU backend in mock mode.
fn run_gpu_mock_test(fixture_name: &str) {
    let runner = E2ERunner::gpu_mock(Profile::Numeric).keep_artifacts();
    let result = runner
        .run_fixture(fixture_name)
        .expect("Failed to run test");

    if !result.is_pass() && !result.is_skipped() {
        let fixture_path = bhc_e2e_tests::fixtures_dir().join(fixture_name);
        let test_case = bhc_e2e_tests::E2ETestCase::from_fixture(&fixture_path).unwrap();
        let report =
            format_failure_report(&test_case, Backend::Gpu, Profile::Numeric, &result, None);
        panic!("{}", report);
    }

    assert!(
        result.is_pass() || result.is_skipped(),
        "Test failed: {:?}",
        result
    );
}

// =============================================================================
// Mock GPU Tests (PTX validation only)
// =============================================================================

#[test]
#[ignore = "GPU backend not yet implemented"]
fn test_gpu_arithmetic_mock() {
    run_gpu_mock_test("tier1_simple/arithmetic");
}

// =============================================================================
// Real CUDA Tests (requires hardware)
// =============================================================================

#[test]
#[ignore = "requires CUDA hardware"]
#[cfg(feature = "cuda")]
fn test_gpu_arithmetic_cuda() {
    let runner = bhc_e2e_tests::E2ERunner::gpu_cuda(Profile::Numeric).keep_artifacts();
    let result = runner
        .run_fixture("tier1_simple/arithmetic")
        .expect("Failed to run test");

    assert!(result.is_pass(), "Test failed: {:?}", result);
}

// =============================================================================
// CUDA Availability Check
// =============================================================================

#[test]
fn test_cuda_detection() {
    let available = bhc_e2e_tests::cuda_available();
    println!("CUDA available: {}", available);
    // This test always passes - it just reports CUDA availability
}
