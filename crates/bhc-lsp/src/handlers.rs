//! Request handlers for LSP.
//!
//! This module contains handlers for various LSP requests.

use crate::analysis::AnalysisEngine;
use crate::completion::complete;
use crate::config::Config;
use crate::document::DocumentManager;
use crate::hover::hover_info;
use crate::navigation::{find_definition, find_references};
use crate::symbols::{document_symbols, workspace_symbols};
use crate::LspResult;

use lsp_server::Request;
use lsp_types::{
    CompletionParams, CompletionResponse, DocumentFormattingParams, DocumentSymbolParams,
    DocumentSymbolResponse, GotoDefinitionParams, GotoDefinitionResponse, HoverParams,
    ReferenceParams, TextEdit, WorkspaceSymbolParams,
};
use std::sync::Arc;
use tracing::debug;

/// Handler for LSP requests.
pub struct RequestHandler {
    /// Document manager.
    documents: Arc<DocumentManager>,
    /// Analysis engine.
    analysis: Arc<AnalysisEngine>,
    /// Configuration.
    config: Arc<Config>,
}

impl RequestHandler {
    /// Create a new request handler.
    pub fn new(
        documents: Arc<DocumentManager>,
        analysis: Arc<AnalysisEngine>,
        config: Arc<Config>,
    ) -> Self {
        Self {
            documents,
            analysis,
            config,
        }
    }

    /// Handle textDocument/definition.
    pub fn handle_goto_definition(&self, req: Request) -> LspResult<serde_json::Value> {
        let params: GotoDefinitionParams = serde_json::from_value(req.params)?;
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        debug!("Go to definition: {:?} at {:?}", uri, position);

        if let Some(doc) = self.documents.get(uri) {
            if let Some(result) = self.analysis.get_cached(uri) {
                if let Some(location) = find_definition(&doc, &result, position) {
                    let response = GotoDefinitionResponse::Scalar(location);
                    return Ok(serde_json::to_value(response)?);
                }
            }
        }

        Ok(serde_json::Value::Null)
    }

    /// Handle textDocument/references.
    pub fn handle_references(&self, req: Request) -> LspResult<serde_json::Value> {
        let params: ReferenceParams = serde_json::from_value(req.params)?;
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        debug!("Find references: {:?} at {:?}", uri, position);

        if let Some(doc) = self.documents.get(uri) {
            if let Some(result) = self.analysis.get_cached(uri) {
                let locations = find_references(&doc, &result, position, uri);
                if !locations.is_empty() {
                    return Ok(serde_json::to_value(locations)?);
                }
            }
        }

        Ok(serde_json::Value::Null)
    }

    /// Handle textDocument/hover.
    pub fn handle_hover(&self, req: Request) -> LspResult<serde_json::Value> {
        let params: HoverParams = serde_json::from_value(req.params)?;
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        debug!("Hover: {:?} at {:?}", uri, position);

        if let Some(doc) = self.documents.get(uri) {
            if let Some(result) = self.analysis.get_cached(uri) {
                if let Some(hover) = hover_info(&doc, &result, position) {
                    return Ok(serde_json::to_value(hover)?);
                }
            }
        }

        Ok(serde_json::Value::Null)
    }

    /// Handle textDocument/completion.
    pub fn handle_completion(&self, req: Request) -> LspResult<serde_json::Value> {
        let params: CompletionParams = serde_json::from_value(req.params)?;
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        debug!("Completion: {:?} at {:?}", uri, position);

        if let Some(doc) = self.documents.get(uri) {
            let items = complete(&doc, &self.analysis, position, &self.config.completion);
            let response = CompletionResponse::Array(items);
            return Ok(serde_json::to_value(response)?);
        }

        Ok(serde_json::to_value(CompletionResponse::Array(Vec::new()))?)
    }

    /// Handle textDocument/documentSymbol.
    pub fn handle_document_symbols(&self, req: Request) -> LspResult<serde_json::Value> {
        let params: DocumentSymbolParams = serde_json::from_value(req.params)?;
        let uri = &params.text_document.uri;

        debug!("Document symbols: {:?}", uri);

        if let Some(result) = self.analysis.get_cached(uri) {
            let symbols = document_symbols(&result);
            let response = DocumentSymbolResponse::Nested(symbols);
            return Ok(serde_json::to_value(response)?);
        }

        Ok(serde_json::to_value(DocumentSymbolResponse::Nested(
            Vec::new(),
        ))?)
    }

    /// Handle workspace/symbol.
    pub fn handle_workspace_symbols(&self, req: Request) -> LspResult<serde_json::Value> {
        let params: WorkspaceSymbolParams = serde_json::from_value(req.params)?;
        let query = &params.query;

        debug!("Workspace symbols: query={}", query);

        let symbols = workspace_symbols(&self.documents, &self.analysis, query);

        Ok(serde_json::to_value(symbols)?)
    }

    /// Handle textDocument/formatting.
    pub fn handle_formatting(&self, req: Request) -> LspResult<serde_json::Value> {
        let params: DocumentFormattingParams = serde_json::from_value(req.params)?;
        let uri = &params.text_document.uri;

        debug!("Formatting: {:?}", uri);

        if !self.config.formatting.enabled {
            return Ok(serde_json::Value::Null);
        }

        if let Some(doc) = self.documents.get(uri) {
            let content = doc.text();
            if let Some(formatted) = format_haskell(&content, &self.config.formatting) {
                if formatted != content {
                    // Create a single edit replacing the entire document
                    let edit = TextEdit {
                        range: lsp_types::Range {
                            start: lsp_types::Position {
                                line: 0,
                                character: 0,
                            },
                            end: lsp_types::Position {
                                line: doc.line_count() as u32,
                                character: 0,
                            },
                        },
                        new_text: formatted,
                    };
                    return Ok(serde_json::to_value(vec![edit])?);
                }
            }
        }

        Ok(serde_json::to_value(Vec::<TextEdit>::new())?)
    }
}

/// Format Haskell code (placeholder implementation).
fn format_haskell(content: &str, _config: &crate::config::FormattingConfig) -> Option<String> {
    // In a real implementation, this would invoke a formatter like ormolu or fourmolu
    // For now, just return the content unchanged
    Some(content.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_creation() {
        let docs = Arc::new(DocumentManager::new());
        let analysis = Arc::new(AnalysisEngine::new());
        let config = Arc::new(Config::default());

        let handler = RequestHandler::new(docs, analysis, config);
        // Just verify it compiles
    }
}
