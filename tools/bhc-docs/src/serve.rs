//! Development server with live reload.
//!
//! This module provides a local development server for previewing
//! documentation with automatic rebuild and browser refresh.

use anyhow::Result;
use axum::{routing::get_service, Router};
use std::path::PathBuf;
use tower_http::compression::CompressionLayer;
use tower_http::services::ServeDir;

/// Server configuration.
pub struct ServeConfig {
    /// Documentation directory.
    pub dir: PathBuf,
    /// Port to listen on.
    pub port: u16,
    /// Watch for changes.
    pub watch: bool,
    /// Source directory to watch.
    pub source: Option<PathBuf>,
}

/// Run the development server.
pub fn run(config: ServeConfig) -> Result<()> {
    // Build the runtime
    let rt = tokio::runtime::Runtime::new()?;

    rt.block_on(async { run_async(config).await })
}

async fn run_async(config: ServeConfig) -> Result<()> {
    let dir = config.dir.clone();

    // Create the file service
    let service = get_service(ServeDir::new(&dir));

    // Build the router
    let app = Router::new()
        .fallback_service(service)
        .layer(CompressionLayer::new());

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], config.port));
    tracing::info!("Serving documentation at http://{}", addr);

    if config.watch {
        tracing::info!("Watching for changes (Ctrl+C to stop)");

        // Start file watcher in background
        let source = config.source.clone();
        let output = config.dir.clone();
        tokio::spawn(async move {
            if let Err(e) = watch_and_rebuild(source, output).await {
                tracing::error!("Watcher error: {}", e);
            }
        });
    }

    // Run the server
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn watch_and_rebuild(source: Option<PathBuf>, output: PathBuf) -> Result<()> {
    use notify::{recommended_watcher, RecursiveMode, Watcher};
    use std::sync::mpsc::channel;
    use std::time::Duration;

    let source = source.unwrap_or_else(|| PathBuf::from("."));

    let (tx, rx) = channel();

    let mut watcher = recommended_watcher(move |res| {
        if let Ok(event) = res {
            let _ = tx.send(event);
        }
    })?;

    watcher.watch(&source, RecursiveMode::Recursive)?;

    tracing::info!("Watching {:?} for changes", source);

    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(event) => {
                tracing::debug!("File change detected: {:?}", event);

                // Debounce - wait a bit for more changes
                std::thread::sleep(Duration::from_millis(500));

                // Drain any pending events
                while rx.try_recv().is_ok() {}

                // Rebuild
                tracing::info!("Rebuilding documentation...");
                if let Err(e) = rebuild(&source, &output) {
                    tracing::error!("Rebuild failed: {}", e);
                } else {
                    tracing::info!("Rebuild complete");
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    Ok(())
}

fn rebuild(source: &PathBuf, output: &PathBuf) -> Result<()> {
    crate::build::run(crate::build::BuildConfig {
        input: source.clone(),
        output: output.clone(),
        format: crate::build::Format::Html,
        playground: false,
        base_url: None,
        version: None,
        source_url: None,
    })
}
