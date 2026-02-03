//! WASM backend runner using wasmtime.

use crate::{E2EError, E2ETestCase, ExecutionOutput, Profile};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use wasmtime::*;
use wasmtime_wasi::pipe::MemoryOutputPipe;
use wasmtime_wasi::preview1;
use wasmtime_wasi::WasiCtxBuilder;

/// Compile a Haskell source file to WASM.
pub fn compile_wasm(
    test_case: &E2ETestCase,
    work_dir: &Path,
    profile: Profile,
) -> Result<PathBuf, E2EError> {
    let output_path = work_dir.join(format!("{}.wasm", test_case.name));

    // Build bhc command with WASM target
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--quiet", "-p", "bhc", "--"]);

    // Add all source paths
    for source in &test_case.source_paths {
        cmd.arg(source.to_str().unwrap());
    }

    cmd.args(["--target=wasm32-wasi", "-o", output_path.to_str().unwrap()]);

    // Add profile flag
    match profile {
        Profile::Default => {}
        Profile::Numeric => {
            cmd.arg("--profile=numeric");
        }
        Profile::Edge => {
            cmd.arg("--profile=edge");
        }
    }

    // Run compiler
    let result = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| E2EError::CompileError(format!("Failed to run compiler: {}", e)))?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        let stdout = String::from_utf8_lossy(&result.stdout);
        return Err(E2EError::CompileError(format!(
            "WASM compilation failed:\nstdout: {}\nstderr: {}",
            stdout, stderr
        )));
    }

    // Verify output exists
    if !output_path.exists() {
        return Err(E2EError::CompileError(format!(
            "Compiler succeeded but WASM output not found: {}",
            output_path.display()
        )));
    }

    Ok(output_path)
}

/// Run a WASM module using wasmtime with WASI support.
pub fn run_wasm(wasm_path: &Path, timeout: Duration) -> Result<ExecutionOutput, E2EError> {
    let start = Instant::now();

    // Create wasmtime engine with fuel-based timeout
    let mut config = Config::new();
    config.consume_fuel(true);

    let engine = Engine::new(&config)
        .map_err(|e| E2EError::ExecutionError(format!("Failed to create WASM engine: {}", e)))?;

    // Load the WASM module
    let module = Module::from_file(&engine, wasm_path)
        .map_err(|e| E2EError::ExecutionError(format!("Failed to load WASM module: {}", e)))?;

    // Create output pipes for capturing stdout/stderr
    let stdout_pipe = MemoryOutputPipe::new(4096);
    let stderr_pipe = MemoryOutputPipe::new(4096);

    // Create WASI context with captured stdout
    let wasi_ctx = WasiCtxBuilder::new()
        .stdout(stdout_pipe.clone())
        .stderr(stderr_pipe.clone())
        .build_p1();

    // Create store with WASI context
    let mut store = Store::new(&engine, wasi_ctx);

    // Add fuel for timeout (rough approximation: 1M instructions per second)
    let fuel_per_second = 1_000_000u64;
    let fuel = fuel_per_second * timeout.as_secs()
        + fuel_per_second * timeout.subsec_millis() as u64 / 1000;
    store
        .set_fuel(fuel)
        .map_err(|e| E2EError::ExecutionError(format!("Failed to set fuel: {}", e)))?;

    // Create linker and add WASI
    let mut linker = Linker::new(&engine);
    preview1::add_to_linker_sync(&mut linker, |ctx| ctx)
        .map_err(|e| E2EError::ExecutionError(format!("Failed to add WASI to linker: {}", e)))?;

    // Instantiate the module
    let instance = linker.instantiate(&mut store, &module).map_err(|e| {
        E2EError::ExecutionError(format!("Failed to instantiate WASM module: {}", e))
    })?;

    // Get the _start function (WASI entry point)
    let start_func = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .map_err(|e| E2EError::ExecutionError(format!("Failed to find _start function: {}", e)))?;

    // Run the program
    let exit_code = match start_func.call(&mut store, ()) {
        Ok(()) => 0,
        Err(e) => {
            // Check if it's an intentional exit
            if let Some(exit) = e.downcast_ref::<wasmtime_wasi::I32Exit>() {
                exit.0
            } else if e.to_string().contains("out of fuel") {
                return Err(E2EError::Timeout(timeout));
            } else {
                return Err(E2EError::ExecutionError(format!(
                    "WASM execution error: {}",
                    e
                )));
            }
        }
    };

    let duration = start.elapsed();

    // Extract captured output
    let stdout = String::from_utf8_lossy(&stdout_pipe.contents()).to_string();
    let stderr = String::from_utf8_lossy(&stderr_pipe.contents()).to_string();

    Ok(ExecutionOutput {
        stdout,
        stderr,
        exit_code,
        duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_output_pipe() {
        let pipe = MemoryOutputPipe::new(1024);
        // MemoryOutputPipe is the wasmtime-wasi type for capturing output
        assert!(pipe.contents().is_empty());
    }

    #[test]
    fn test_wasm_engine_creation() {
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config);
        assert!(engine.is_ok());
    }
}
