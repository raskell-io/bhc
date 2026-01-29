//! LSP server implementation.
//!
//! This module contains the main server loop and message handling.

use crate::analysis::AnalysisEngine;
use crate::capabilities::server_capabilities;
use crate::config::Config;
use crate::document::DocumentManager;
use crate::handlers::RequestHandler;

use anyhow::Result;
use lsp_server::{Connection, Message, Notification, Request, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, DidSaveTextDocument,
    Initialized, Notification as _,
};
use lsp_types::request::{
    Completion, DocumentSymbolRequest, Formatting, GotoDefinition, HoverRequest, References,
    Request as _, WorkspaceSymbolRequest,
};
use lsp_types::{InitializeParams, InitializeResult, ServerInfo, Uri};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Run the LSP server.
///
/// This is the main entry point for the language server.
pub fn run() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    info!("Starting BHC Language Server");

    // Create the transport
    let (connection, io_threads) = Connection::stdio();

    // Run the server
    let server = Server::new(connection)?;
    server.run()?;

    // Wait for IO threads
    io_threads.join()?;

    info!("BHC Language Server stopped");
    Ok(())
}

/// The LSP server.
pub struct Server {
    /// Connection to the client.
    connection: Connection,
    /// Document manager.
    documents: Arc<DocumentManager>,
    /// Analysis engine.
    analysis: Arc<AnalysisEngine>,
    /// Server configuration.
    config: Arc<Config>,
    /// Request handler.
    handler: RequestHandler,
}

impl Server {
    /// Create a new server.
    pub fn new(connection: Connection) -> Result<Self> {
        let documents = Arc::new(DocumentManager::new());
        let analysis = Arc::new(AnalysisEngine::new());
        let config = Arc::new(Config::default());
        let handler = RequestHandler::new(
            Arc::clone(&documents),
            Arc::clone(&analysis),
            Arc::clone(&config),
        );

        Ok(Self {
            connection,
            documents,
            analysis,
            config,
            handler,
        })
    }

    /// Run the server main loop.
    pub fn run(mut self) -> Result<()> {
        // Initialize
        let (id, params) = self.connection.initialize_start()?;
        let params: InitializeParams = serde_json::from_value(params)?;

        self.on_initialize(&params);

        let result = InitializeResult {
            capabilities: server_capabilities(),
            server_info: Some(ServerInfo {
                name: "bhc-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        };

        self.connection
            .initialize_finish(id, serde_json::to_value(result)?)?;

        // Wait for initialized notification
        match self.connection.receiver.recv()? {
            Message::Notification(not) if not.method == Initialized::METHOD => {
                debug!("Received initialized notification");
            }
            msg => {
                warn!("Expected initialized notification, got: {:?}", msg);
            }
        }

        // Main message loop
        self.main_loop()?;

        Ok(())
    }

    /// Handle initialization.
    fn on_initialize(&mut self, params: &InitializeParams) {
        info!(
            "Initializing with client: {:?}",
            params.client_info.as_ref().map(|i| &i.name)
        );

        if let Some(ref root) = params.root_uri {
            info!("Root URI: {:?}", root);
        }

        // Update config from client settings
        if let Some(ref opts) = params.initialization_options {
            if let Err(e) = self.config.update_from_value(opts) {
                warn!("Failed to parse initialization options: {}", e);
            }
        }
    }

    /// Main message loop.
    fn main_loop(&mut self) -> Result<()> {
        loop {
            let msg = match self.connection.receiver.recv() {
                Ok(msg) => msg,
                Err(_) => {
                    debug!("Connection closed");
                    return Ok(());
                }
            };

            match msg {
                Message::Request(req) => {
                    if self.connection.handle_shutdown(&req)? {
                        return Ok(());
                    }
                    self.handle_request(req)?;
                }
                Message::Notification(not) => {
                    self.handle_notification(not)?;
                }
                Message::Response(resp) => {
                    self.handle_response(resp)?;
                }
            }
        }
    }

    /// Handle a request.
    fn handle_request(&mut self, req: Request) -> Result<()> {
        let id = req.id.clone();

        let result = match req.method.as_str() {
            GotoDefinition::METHOD => self.handler.handle_goto_definition(req),
            References::METHOD => self.handler.handle_references(req),
            HoverRequest::METHOD => self.handler.handle_hover(req),
            Completion::METHOD => self.handler.handle_completion(req),
            DocumentSymbolRequest::METHOD => self.handler.handle_document_symbols(req),
            WorkspaceSymbolRequest::METHOD => self.handler.handle_workspace_symbols(req),
            Formatting::METHOD => self.handler.handle_formatting(req),
            _ => {
                warn!("Unhandled request: {}", req.method);
                return Ok(());
            }
        };

        let response = match result {
            Ok(value) => Response::new_ok(id, value),
            Err(e) => {
                error!("Request failed: {}", e);
                Response::new_err(id, -32603, e.to_string())
            }
        };

        self.connection.sender.send(Message::Response(response))?;
        Ok(())
    }

    /// Handle a notification.
    fn handle_notification(&mut self, not: Notification) -> Result<()> {
        match not.method.as_str() {
            DidOpenTextDocument::METHOD => {
                let params = serde_json::from_value(not.params)?;
                self.on_did_open(params)?;
            }
            DidChangeTextDocument::METHOD => {
                let params = serde_json::from_value(not.params)?;
                self.on_did_change(params)?;
            }
            DidSaveTextDocument::METHOD => {
                let params = serde_json::from_value(not.params)?;
                self.on_did_save(params)?;
            }
            DidCloseTextDocument::METHOD => {
                let params = serde_json::from_value(not.params)?;
                self.on_did_close(params)?;
            }
            _ => {
                debug!("Unhandled notification: {}", not.method);
            }
        }
        Ok(())
    }

    /// Handle a response from the client.
    fn handle_response(&mut self, resp: Response) -> Result<()> {
        debug!("Received response: {:?}", resp.id);
        Ok(())
    }

    /// Handle textDocument/didOpen.
    fn on_did_open(&mut self, params: lsp_types::DidOpenTextDocumentParams) -> Result<()> {
        let uri = params.text_document.uri.clone();
        debug!("Document opened: {:?}", uri);

        self.documents.open(
            uri.clone(),
            params.text_document.text,
            params.text_document.version,
        );

        // Analyze and publish diagnostics
        self.analyze_and_publish_diagnostics(&uri)?;
        Ok(())
    }

    /// Handle textDocument/didChange.
    fn on_did_change(&mut self, params: lsp_types::DidChangeTextDocumentParams) -> Result<()> {
        let uri = params.text_document.uri.clone();
        debug!("Document changed: {:?}", uri);

        for change in params.content_changes {
            if let Some(range) = change.range {
                self.documents.apply_change(&uri, range, &change.text);
            } else {
                // Full document update
                self.documents
                    .update(&uri, change.text, params.text_document.version);
            }
        }

        // Analyze and publish diagnostics
        self.analyze_and_publish_diagnostics(&uri)?;
        Ok(())
    }

    /// Handle textDocument/didSave.
    fn on_did_save(&mut self, params: lsp_types::DidSaveTextDocumentParams) -> Result<()> {
        let uri = params.text_document.uri;
        debug!("Document saved: {:?}", uri);

        // Re-analyze on save
        self.analyze_and_publish_diagnostics(&uri)?;
        Ok(())
    }

    /// Handle textDocument/didClose.
    fn on_did_close(&mut self, params: lsp_types::DidCloseTextDocumentParams) -> Result<()> {
        let uri = params.text_document.uri;
        debug!("Document closed: {:?}", uri);

        self.documents.close(&uri);

        // Clear diagnostics
        self.publish_diagnostics(&uri, vec![])?;
        Ok(())
    }

    /// Analyze a document and publish diagnostics.
    fn analyze_and_publish_diagnostics(&self, uri: &Uri) -> Result<()> {
        if let Some(content) = self.documents.get_content(uri) {
            let diagnostics = self.analysis.analyze(&content, uri);
            self.publish_diagnostics(uri, diagnostics)?;
        }
        Ok(())
    }

    /// Publish diagnostics to the client.
    fn publish_diagnostics(
        &self,
        uri: &Uri,
        diagnostics: Vec<lsp_types::Diagnostic>,
    ) -> Result<()> {
        let params = lsp_types::PublishDiagnosticsParams {
            uri: uri.clone(),
            diagnostics,
            version: None,
        };

        let not = Notification::new(
            lsp_types::notification::PublishDiagnostics::METHOD.to_string(),
            params,
        );

        self.connection.sender.send(Message::Notification(not))?;
        Ok(())
    }
}
