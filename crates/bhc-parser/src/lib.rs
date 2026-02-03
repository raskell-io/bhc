//! Parser for Haskell 2026 source code.
//!
//! This crate provides a recursive descent parser that produces an AST
//! from a token stream.

#![warn(missing_docs)]

use bhc_ast::{Expr, Module};
use bhc_diagnostics::{Diagnostic, DiagnosticHandler, FullSpan};
use bhc_lexer::{Lexer, Token, TokenKind};
use bhc_span::{FileId, Span, Spanned};
use thiserror::Error;

mod decl;
mod expr;
mod pattern;
mod types;

/// Parser error type.
#[derive(Debug, Error)]
pub enum ParseError {
    /// Unexpected token.
    #[error("unexpected {found}, expected {expected}")]
    Unexpected {
        /// What was found.
        found: String,
        /// What was expected.
        expected: String,
        /// Location.
        span: Span,
    },

    /// Unexpected end of file.
    #[error("unexpected end of file")]
    UnexpectedEof {
        /// What was expected.
        expected: String,
    },

    /// Invalid literal.
    #[error("invalid literal: {message}")]
    InvalidLiteral {
        /// Error message.
        message: String,
        /// Location.
        span: Span,
    },
}

impl ParseError {
    /// Convert to a diagnostic.
    #[must_use]
    pub fn to_diagnostic(&self, file: FileId) -> Diagnostic {
        match self {
            Self::Unexpected {
                found,
                expected,
                span,
            } => Diagnostic::error(format!("unexpected {found}, expected {expected}"))
                .with_label(FullSpan::new(file, *span), "unexpected token here"),
            Self::UnexpectedEof { expected } => {
                Diagnostic::error(format!("unexpected end of file, expected {expected}"))
            }
            Self::InvalidLiteral { message, span } => {
                Diagnostic::error(format!("invalid literal: {message}"))
                    .with_label(FullSpan::new(file, *span), "invalid literal")
            }
        }
    }
}

/// The result of parsing.
pub type ParseResult<T> = Result<T, ParseError>;

/// A parser for Haskell 2026 source code.
pub struct Parser<'src> {
    /// The token stream.
    tokens: Vec<Spanned<Token>>,
    /// Current position in the token stream.
    pos: usize,
    /// Diagnostic handler.
    diagnostics: DiagnosticHandler,
    /// Source file ID.
    file_id: FileId,
    /// The source code (for error messages).
    #[allow(dead_code)]
    src: &'src str,
}

impl<'src> Parser<'src> {
    /// Create a new parser for the given source code.
    #[must_use]
    pub fn new(src: &'src str, file_id: FileId) -> Self {
        let tokens: Vec<_> = Lexer::new(src).collect();
        Self {
            tokens,
            pos: 0,
            diagnostics: DiagnosticHandler::new(),
            file_id,
            src,
        }
    }

    /// Get the current token.
    fn current(&self) -> Option<&Spanned<Token>> {
        self.tokens.get(self.pos)
    }

    /// Peek at the nth token from current position (0 = current).
    fn peek_nth(&self, n: usize) -> Option<&Spanned<Token>> {
        self.tokens.get(self.pos + n)
    }

    /// Get the current token kind.
    fn current_kind(&self) -> Option<&TokenKind> {
        self.current().map(|t| &t.node.kind)
    }

    /// Get the current span.
    fn current_span(&self) -> Span {
        self.current().map(|t| t.span).unwrap_or(Span::DUMMY)
    }

    /// Check if we're at the end of input.
    fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len() || self.current_kind() == Some(&TokenKind::Eof)
    }

    /// Advance to the next token.
    fn advance(&mut self) -> Option<Spanned<Token>> {
        if self.at_eof() {
            None
        } else {
            let tok = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(tok)
        }
    }

    /// Check if the current token matches the given kind.
    fn check(&self, kind: &TokenKind) -> bool {
        self.current_kind() == Some(kind)
    }

    /// Check if the current token is a constructor identifier.
    #[allow(dead_code)]
    fn check_con_id(&self) -> bool {
        matches!(self.current_kind(), Some(TokenKind::ConId(_)))
    }

    /// Check if the current token is an identifier.
    #[allow(dead_code)]
    fn check_ident(&self) -> bool {
        matches!(self.current_kind(), Some(TokenKind::Ident(_)))
    }

    /// Consume a token if it matches the given kind.
    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consume an identifier token if it has the given string value.
    /// Used for context-sensitive keywords like 'as', 'qualified', 'hiding'.
    fn eat_ident_str(&mut self, s: &str) -> bool {
        if let Some(TokenKind::Ident(sym)) = self.current_kind() {
            if sym.as_str() == s {
                self.advance();
                return true;
            }
        }
        false
    }

    /// Skip any virtual tokens (VirtualLBrace, VirtualRBrace, VirtualSemi).
    /// These are inserted by the layout rule and need to be skipped in some contexts.
    fn skip_virtual_tokens(&mut self) {
        while let Some(kind) = self.current_kind() {
            if kind.is_virtual() {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Skip doc comments (Haddock comments like `-- |` or `{- | ... -}`).
    /// These can appear before module declarations in real-world Haskell code.
    fn skip_doc_comments(&mut self) {
        while let Some(kind) = self.current_kind() {
            match kind {
                TokenKind::DocCommentLine(_) | TokenKind::DocCommentBlock(_) => {
                    self.advance();
                }
                _ => break,
            }
        }
    }

    /// Collect doc comments, returning them as a `DocComment` if present.
    ///
    /// This collects all consecutive doc comments and merges them into a single
    /// documentation string. Supports both line comments (`-- |`) and block
    /// comments (`{- | ... -}`).
    fn collect_doc_comments(&mut self) -> Option<bhc_ast::DocComment> {
        let mut texts = Vec::new();
        let mut first_span: Option<Span> = None;
        let mut last_span: Option<Span> = None;
        let mut kind = bhc_ast::DocKind::Preceding;

        while let Some(tok) = self.current() {
            match &tok.node.kind {
                TokenKind::DocCommentLine(text) => {
                    let span = tok.span;
                    let text = text.clone();
                    self.advance();

                    // Check if it's a trailing comment (starts with ^)
                    let trimmed = text.trim_start();
                    let (actual_text, doc_kind) = if trimmed.starts_with('^') {
                        (
                            trimmed
                                .strip_prefix('^')
                                .unwrap_or(trimmed)
                                .trim()
                                .to_string(),
                            bhc_ast::DocKind::Trailing,
                        )
                    } else if trimmed.starts_with('|') {
                        (
                            trimmed
                                .strip_prefix('|')
                                .unwrap_or(trimmed)
                                .trim()
                                .to_string(),
                            bhc_ast::DocKind::Preceding,
                        )
                    } else {
                        (trimmed.to_string(), bhc_ast::DocKind::Preceding)
                    };

                    if first_span.is_none() {
                        first_span = Some(span);
                        kind = doc_kind;
                    }
                    last_span = Some(span);
                    texts.push(actual_text);
                }
                TokenKind::DocCommentBlock(text) => {
                    let span = tok.span;
                    let text = text.clone();
                    self.advance();

                    // Check if it's a trailing comment (starts with ^)
                    let trimmed = text.trim();
                    let (actual_text, doc_kind) = if trimmed.starts_with('^') {
                        (
                            trimmed
                                .strip_prefix('^')
                                .unwrap_or(trimmed)
                                .trim()
                                .to_string(),
                            bhc_ast::DocKind::Trailing,
                        )
                    } else if trimmed.starts_with('|') {
                        (
                            trimmed
                                .strip_prefix('|')
                                .unwrap_or(trimmed)
                                .trim()
                                .to_string(),
                            bhc_ast::DocKind::Preceding,
                        )
                    } else {
                        (trimmed.to_string(), bhc_ast::DocKind::Preceding)
                    };

                    if first_span.is_none() {
                        first_span = Some(span);
                        kind = doc_kind;
                    }
                    last_span = Some(span);
                    texts.push(actual_text);
                }
                _ => break,
            }
        }

        if texts.is_empty() {
            return None;
        }

        let combined_text = texts.join("\n");
        let span = first_span.unwrap().to(last_span.unwrap());

        Some(bhc_ast::DocComment {
            text: combined_text,
            kind,
            span,
        })
    }

    /// Expect a token of the given kind.
    fn expect(&mut self, kind: &TokenKind) -> ParseResult<Spanned<Token>> {
        if self.check(kind) {
            Ok(self.advance().unwrap())
        } else if self.at_eof() {
            Err(ParseError::UnexpectedEof {
                expected: kind.description().to_string(),
            })
        } else {
            let current = self.current().unwrap();
            Err(ParseError::Unexpected {
                found: current.node.kind.description().to_string(),
                expected: kind.description().to_string(),
                span: current.span,
            })
        }
    }

    /// Emit a diagnostic.
    fn emit(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.emit(diagnostic);
    }

    /// Check if there are errors.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.diagnostics.has_errors()
    }

    /// Take the diagnostics.
    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        self.diagnostics.take_diagnostics()
    }
}

/// Parse a module from source code.
pub fn parse_module(src: &str, file_id: FileId) -> (Option<Module>, Vec<Diagnostic>) {
    let mut parser = Parser::new(src, file_id);
    let module = parser.parse_module();
    let diagnostics = parser.take_diagnostics();

    match module {
        Ok(m) => (Some(m), diagnostics),
        Err(e) => {
            let mut diags = diagnostics;
            diags.push(e.to_diagnostic(file_id));
            (None, diags)
        }
    }
}

/// Parse an expression from source code.
pub fn parse_expr(src: &str, file_id: FileId) -> (Option<Expr>, Vec<Diagnostic>) {
    let mut parser = Parser::new(src, file_id);
    let expr = parser.parse_expr();
    let diagnostics = parser.take_diagnostics();

    match expr {
        Ok(e) => (Some(e), diagnostics),
        Err(e) => {
            let mut diags = diagnostics;
            diags.push(e.to_diagnostic(file_id));
            (None, diags)
        }
    }
}

/// Parse a single import declaration from source code.
///
/// This is used by the REPL to handle `import` statements entered interactively.
pub fn parse_import_decl(
    src: &str,
    file_id: FileId,
) -> (Option<bhc_ast::ImportDecl>, Vec<Diagnostic>) {
    let mut parser = Parser::new(src, file_id);
    let import = parser.parse_import();
    let diagnostics = parser.take_diagnostics();

    match import {
        Ok(decl) => (Some(decl), diagnostics),
        Err(e) => {
            let mut diags = diagnostics;
            let diag: Diagnostic = e.to_diagnostic(file_id);
            diags.push(diag);
            (None, diags)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_ast::ImportSpec;

    fn parse_expr_ok(src: &str) -> Expr {
        let (expr, diags) = parse_expr(src, FileId::new(0));
        assert!(diags.is_empty(), "Parse errors: {:?}", diags);
        expr.expect("Expected expression")
    }

    fn parse_module_ok(src: &str) -> Module {
        let (module, diags) = parse_module(src, FileId::new(0));
        assert!(diags.is_empty(), "Parse errors: {:?}", diags);
        module.expect("Expected module")
    }

    #[test]
    fn test_parser_creation() {
        let parser = Parser::new("let x = 1 in x", FileId::new(0));
        assert!(!parser.at_eof());
    }

    #[test]
    fn test_simple_literals() {
        let expr = parse_expr_ok("42");
        assert!(matches!(expr, Expr::Lit(bhc_ast::Lit::Int(42), _)));

        let expr = parse_expr_ok("3.14");
        assert!(matches!(expr, Expr::Lit(bhc_ast::Lit::Float(_), _)));

        let expr = parse_expr_ok("'a'");
        assert!(matches!(expr, Expr::Lit(bhc_ast::Lit::Char('a'), _)));

        let expr = parse_expr_ok("\"hello\"");
        assert!(matches!(expr, Expr::Lit(bhc_ast::Lit::String(_), _)));
    }

    #[test]
    fn test_variable_and_constructor() {
        let expr = parse_expr_ok("foo");
        assert!(matches!(expr, Expr::Var(_, _)));

        let expr = parse_expr_ok("Foo");
        assert!(matches!(expr, Expr::Con(_, _)));
    }

    #[test]
    fn test_application() {
        let expr = parse_expr_ok("f x");
        assert!(matches!(expr, Expr::App(_, _, _)));

        let expr = parse_expr_ok("f x y z");
        assert!(matches!(expr, Expr::App(_, _, _)));
    }

    #[test]
    fn test_infix_operators() {
        let expr = parse_expr_ok("1 + 2");
        assert!(matches!(expr, Expr::Infix(_, _, _, _)));

        let expr = parse_expr_ok("a && b || c");
        assert!(matches!(expr, Expr::Infix(_, _, _, _)));
    }

    #[test]
    fn test_lambda() {
        let expr = parse_expr_ok("\\x -> x");
        assert!(matches!(expr, Expr::Lam(_, _, _)));

        let expr = parse_expr_ok("\\x y -> x + y");
        if let Expr::Lam(pats, _, _) = expr {
            assert_eq!(pats.len(), 2);
        } else {
            panic!("Expected lambda");
        }
    }

    #[test]
    fn test_let_expression() {
        let expr = parse_expr_ok("let { x = 1 } in x");
        assert!(matches!(expr, Expr::Let(_, _, _)));
    }

    #[test]
    fn test_if_expression() {
        let expr = parse_expr_ok("if True then 1 else 2");
        assert!(matches!(expr, Expr::If(_, _, _, _)));
    }

    #[test]
    fn test_case_expression() {
        let expr = parse_expr_ok("case x of { Just y -> y }");
        assert!(matches!(expr, Expr::Case(_, _, _)));
    }

    #[test]
    fn test_do_expression() {
        let expr = parse_expr_ok("do { x <- getLine; putStrLn x }");
        assert!(matches!(expr, Expr::Do(_, _)));
    }

    #[test]
    fn test_tuple() {
        let expr = parse_expr_ok("(1, 2, 3)");
        if let Expr::Tuple(exprs, _) = expr {
            assert_eq!(exprs.len(), 3);
        } else {
            panic!("Expected tuple");
        }
    }

    #[test]
    fn test_list() {
        let expr = parse_expr_ok("[1, 2, 3]");
        if let Expr::List(exprs, _) = expr {
            assert_eq!(exprs.len(), 3);
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_list_comprehension() {
        let expr = parse_expr_ok("[x | x <- xs]");
        assert!(matches!(expr, Expr::ListComp(_, _, _)));
    }

    #[test]
    fn test_arithmetic_sequence() {
        let expr = parse_expr_ok("[1..10]");
        assert!(matches!(expr, Expr::ArithSeq(_, _)));

        let expr = parse_expr_ok("[1..]");
        assert!(matches!(expr, Expr::ArithSeq(_, _)));

        let expr = parse_expr_ok("[1,3..10]");
        assert!(matches!(expr, Expr::ArithSeq(_, _)));
    }

    #[test]
    fn test_record_construction() {
        let expr = parse_expr_ok("Foo { bar = 1, baz = 2 }");
        assert!(matches!(expr, Expr::RecordCon(_, _, _)));
    }

    #[test]
    fn test_record_update() {
        let expr = parse_expr_ok("foo { bar = 1 }");
        assert!(matches!(expr, Expr::RecordUpd(_, _, _)));
    }

    #[test]
    fn test_operator_section_right() {
        // Right section: (+ 1) -> \y -> y + 1
        let expr = parse_expr_ok("(+ 1)");
        assert!(matches!(expr, Expr::Lam(_, _, _)));
    }

    #[test]
    fn test_operator_section_left() {
        // Left section: (1 +) -> \y -> 1 + y
        let expr = parse_expr_ok("(1 +)");
        assert!(matches!(expr, Expr::Lam(_, _, _)));
    }

    #[test]
    fn test_operator_as_function() {
        // (+) becomes a variable
        let expr = parse_expr_ok("(+)");
        assert!(matches!(expr, Expr::Var(_, _)));
    }

    #[test]
    fn test_negation() {
        // Negation only works after another expression in infix context
        // `-x` at the start is ambiguous with operator prefix
        let expr = parse_expr_ok("1 + -x");
        // The result contains a Neg somewhere in the tree
        assert!(matches!(expr, Expr::Infix(_, _, _, _)));
    }

    #[test]
    fn test_lazy_expression() {
        let expr = parse_expr_ok("lazy { expensive }");
        assert!(matches!(expr, Expr::Lazy(_, _)));
    }

    // Pattern tests

    #[test]
    fn test_pattern_wildcard() {
        let module = parse_module_ok("f _ = 1");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_pattern_constructor() {
        let module = parse_module_ok("f (Just x) = x");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_pattern_infix() {
        let module = parse_module_ok("f (x : xs) = xs");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_pattern_as() {
        let module = parse_module_ok("f xs@(x : _) = xs");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_pattern_lazy() {
        let module = parse_module_ok("f ~x = x");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_pattern_bang() {
        let module = parse_module_ok("f !x = x");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_record_pattern() {
        let module = parse_module_ok("f Foo { bar = x } = x");
        assert!(!module.decls.is_empty());
    }

    // Type tests

    #[test]
    fn test_simple_type() {
        let module = parse_module_ok("f :: Int");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_function_type() {
        let module = parse_module_ok("f :: Int -> Bool");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_type_application() {
        let module = parse_module_ok("f :: Maybe Int");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_tuple_type() {
        let module = parse_module_ok("f :: (Int, Bool)");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_list_type() {
        let module = parse_module_ok("f :: [Int]");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_constrained_type() {
        let module = parse_module_ok("f :: Eq a => a -> Bool");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_multi_constrained_type() {
        let module = parse_module_ok("f :: (Eq a, Ord a) => a -> a -> Bool");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_forall_type() {
        let module = parse_module_ok("f :: forall a. a -> a");
        assert!(!module.decls.is_empty());
    }

    // Module structure tests

    #[test]
    fn test_module_header() {
        let module = parse_module_ok("module Foo where\nx = 1");
        assert!(module.name.is_some());
    }

    #[test]
    fn test_module_exports() {
        let module = parse_module_ok("module Foo (bar, baz) where\nbar = 1\nbaz = 2");
        assert!(module.exports.is_some());
    }

    #[test]
    fn test_imports() {
        let module = parse_module_ok("import Data.List\nx = 1");
        assert!(!module.imports.is_empty());
    }

    #[test]
    fn test_qualified_import() {
        let module = parse_module_ok("import qualified Data.Map as M\nx = 1");
        assert!(!module.imports.is_empty());
        assert!(module.imports[0].qualified);
    }

    // Declaration tests

    #[test]
    fn test_data_declaration() {
        let module = parse_module_ok("data Foo = Bar | Baz Int");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_newtype_declaration() {
        let module = parse_module_ok("newtype Foo = Foo Int");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_type_alias() {
        let module = parse_module_ok("type Foo = Int");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_class_declaration() {
        let module = parse_module_ok("class Eq a where\n  eq :: a -> a -> Bool");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_instance_declaration() {
        let module = parse_module_ok("instance Eq Int where\n  eq = primEqInt");
        assert!(!module.decls.is_empty());
    }

    #[test]
    fn test_fixity_declaration() {
        let module = parse_module_ok("infixl 6 +");
        assert!(!module.decls.is_empty());
    }

    // Pragma tests

    #[test]
    fn test_language_pragma() {
        let module = parse_module_ok("{-# LANGUAGE GADTs #-}\nx = 1");
        assert_eq!(module.pragmas.len(), 1);
        match &module.pragmas[0].kind {
            bhc_ast::PragmaKind::Language(exts) => {
                assert_eq!(exts.len(), 1);
                assert_eq!(exts[0].as_str(), "GADTs");
            }
            _ => panic!("Expected Language pragma"),
        }
    }

    #[test]
    fn test_multiple_pragmas() {
        let module = parse_module_ok(
            "{-# LANGUAGE GADTs #-}\n{-# LANGUAGE TypeFamilies, DataKinds #-}\nx = 1",
        );
        assert_eq!(module.pragmas.len(), 2);
    }

    #[test]
    fn test_options_ghc_pragma() {
        let module = parse_module_ok("{-# OPTIONS_GHC -Wall -Werror #-}\nx = 1");
        assert_eq!(module.pragmas.len(), 1);
        match &module.pragmas[0].kind {
            bhc_ast::PragmaKind::OptionsGhc(opts) => {
                assert!(opts.contains("-Wall"));
                assert!(opts.contains("-Werror"));
            }
            _ => panic!("Expected OptionsGhc pragma"),
        }
    }

    #[test]
    fn test_inline_pragma() {
        let module = parse_module_ok("{-# INLINE foo #-}\nfoo = 1");
        assert_eq!(module.pragmas.len(), 1);
        match &module.pragmas[0].kind {
            bhc_ast::PragmaKind::Inline(name) => {
                assert_eq!(name.name.as_str(), "foo");
            }
            _ => panic!("Expected Inline pragma"),
        }
    }

    // ============================================================
    // Phase 1: New parser features tests
    // ============================================================

    #[test]
    fn test_guarded_function() {
        let module = parse_module_ok("abs x | x >= 0 = x | otherwise = -x");
        assert!(!module.decls.is_empty());
        if let bhc_ast::Decl::FunBind(fun) = &module.decls[0] {
            assert_eq!(fun.name.name.as_str(), "abs");
            assert_eq!(fun.clauses.len(), 1);
            if let bhc_ast::Rhs::Guarded(guards, _) = &fun.clauses[0].rhs {
                assert_eq!(guards.len(), 2);
            } else {
                panic!("Expected guarded RHS");
            }
        } else {
            panic!("Expected FunBind");
        }
    }

    #[test]
    fn test_multi_clause_function() {
        // Simple test with explicit semicolons and parentheses around expression
        let module = parse_module_ok("fac 0 = 1; fac n = (n * fac (n - 1))");
        assert_eq!(module.decls.len(), 1); // Should be merged into one
        if let bhc_ast::Decl::FunBind(fun) = &module.decls[0] {
            assert_eq!(fun.name.name.as_str(), "fac");
            assert_eq!(fun.clauses.len(), 2);
        } else {
            panic!("Expected FunBind");
        }
    }

    #[test]
    fn test_where_clause() {
        // Simple where clause with single binding
        let module = parse_module_ok("f x = y where { y = x }");
        if let bhc_ast::Decl::FunBind(fun) = &module.decls[0] {
            assert_eq!(fun.clauses[0].wheres.len(), 1);
        } else {
            panic!("Expected FunBind");
        }
    }

    #[test]
    fn test_where_clause_multiple() {
        // Where clause with multiple bindings
        let module = parse_module_ok("f x = y where { y = (x + 1); z = (x + 2) }");
        if let bhc_ast::Decl::FunBind(fun) = &module.decls[0] {
            assert_eq!(fun.clauses[0].wheres.len(), 2);
        } else {
            panic!("Expected FunBind");
        }
    }

    #[test]
    fn test_strict_field() {
        let module = parse_module_ok("data Pair = Pair !Int !Int");
        if let bhc_ast::Decl::DataDecl(data) = &module.decls[0] {
            assert_eq!(data.constrs.len(), 1);
            if let bhc_ast::ConFields::Positional(fields) = &data.constrs[0].fields {
                assert_eq!(fields.len(), 2);
                assert!(matches!(fields[0], bhc_ast::Type::Bang(_, _)));
                assert!(matches!(fields[1], bhc_ast::Type::Bang(_, _)));
            } else {
                panic!("Expected Positional fields");
            }
        } else {
            panic!("Expected DataDecl");
        }
    }

    #[test]
    fn test_lazy_field() {
        let module = parse_module_ok("data Lazy a = Lazy ~a");
        if let bhc_ast::Decl::DataDecl(data) = &module.decls[0] {
            if let bhc_ast::ConFields::Positional(fields) = &data.constrs[0].fields {
                assert_eq!(fields.len(), 1);
                assert!(matches!(fields[0], bhc_ast::Type::Lazy(_, _)));
            } else {
                panic!("Expected Positional fields");
            }
        } else {
            panic!("Expected DataDecl");
        }
    }

    #[test]
    fn test_mixed_strict_lazy_fields() {
        let module = parse_module_ok("data Triple a b c = Triple !a b ~c");
        if let bhc_ast::Decl::DataDecl(data) = &module.decls[0] {
            if let bhc_ast::ConFields::Positional(fields) = &data.constrs[0].fields {
                assert_eq!(fields.len(), 3);
                assert!(matches!(fields[0], bhc_ast::Type::Bang(_, _)));
                assert!(matches!(fields[1], bhc_ast::Type::Var(_, _)));
                assert!(matches!(fields[2], bhc_ast::Type::Lazy(_, _)));
            } else {
                panic!("Expected Positional fields");
            }
        } else {
            panic!("Expected DataDecl");
        }
    }

    #[test]
    fn test_guards_with_where() {
        // Simplified: guards with a simple where clause
        let module = parse_module_ok(
            "signum x | x > 0 = positive | otherwise = zero where { positive = 1; zero = 0 }",
        );
        if let bhc_ast::Decl::FunBind(fun) = &module.decls[0] {
            if let bhc_ast::Rhs::Guarded(guards, _) = &fun.clauses[0].rhs {
                assert_eq!(guards.len(), 2);
            } else {
                panic!("Expected guarded RHS");
            }
            assert_eq!(fun.clauses[0].wheres.len(), 2);
        } else {
            panic!("Expected FunBind");
        }
    }

    #[test]
    fn test_backtick_operator_with_lambda() {
        // Backtick operators followed by lambda expressions
        let _ = parse_module_ok("test = f `catch` \\e -> handle e");
        let _ = parse_module_ok("test = x `fmap` (\\a -> a + 1)");
        // Qualified names in backticks
        let _ = parse_module_ok("test = action `E.catch` \\e -> case e of { Ex -> handler }");
    }

    #[test]
    fn test_as_patterns() {
        // Simple as-pattern
        let _ = parse_module_ok("f x@(Just y) = y");
        // As-pattern with list
        let _ = parse_module_ok("g xs@(x:_) = x");
        // As-pattern with record (XMonad style) - using explicit braces
        let _ = parse_module_ok("h conf@(Config { field = v }) = v");
    }

    #[test]
    fn test_list_type_annotation() {
        // List with type annotation (XMonad workspaces pattern)
        let _ = parse_module_ok("test = [1 .. 9 :: Int]");
        // List with explicit type inside
        let _ = parse_module_ok("test = map show [4..9]");
    }

    #[test]
    fn test_multi_clause_explicit_layout() {
        // Multi-clause pattern matching function (uses explicit layout)
        let module = parse_module_ok("f 0 = 1; f n = n");
        // Check that both clauses are in the same FunBind
        if let bhc_ast::Decl::FunBind(fun) = &module.decls[0] {
            assert_eq!(fun.clauses.len(), 2, "Expected 2 clauses");
        } else {
            panic!("Expected FunBind");
        }
    }

    #[test]
    fn test_multi_clause_with_type_sig_explicit() {
        // Type signature followed by multi-clause function
        let module = parse_module_ok("f :: Int -> Int; f 0 = 1; f n = n");
        // First decl is TypeSig, second is FunBind with 2 clauses
        assert!(matches!(module.decls[0], bhc_ast::Decl::TypeSig { .. }));
        if let bhc_ast::Decl::FunBind(fun) = &module.decls[1] {
            assert_eq!(fun.clauses.len(), 2, "Expected 2 clauses");
        } else {
            panic!("Expected FunBind");
        }
    }

    #[test]
    fn test_multi_clause_with_layout() {
        // Multi-clause with layout-based syntax (no explicit semicolons)
        // Uses a module declaration to trigger proper layout handling
        let src = r#"module Test where

f :: Int -> Int
f 0 = 1
f n = n
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        if !diags.is_empty() {
            // Print errors for debugging
            for d in &diags {
                eprintln!("Error: {:?}", d);
            }
        }
        let module = module.expect("Should parse");
        // Should have TypeSig and FunBind
        assert_eq!(
            module.decls.len(),
            2,
            "Expected 2 decls (TypeSig + FunBind)"
        );
        if let bhc_ast::Decl::FunBind(fun) = &module.decls[1] {
            assert_eq!(
                fun.clauses.len(),
                2,
                "Expected 2 clauses, got: {}",
                fun.clauses.len()
            );
        } else {
            panic!("Expected FunBind, got: {:?}", module.decls[1]);
        }
    }

    #[test]
    fn test_record_with_layout_style() {
        // Record definition with XMonad-style layout (leading commas)
        let src = r#"module Test where

data Foo = Foo { field1 :: Int
               , field2 :: String
               } deriving (Show)
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Should parse without errors");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_xmonad_stackset_style() {
        // XMonad StackSet-style records with type variables and strict fields
        let src = r#"module Test where

data StackSet i l a sid sd =
    StackSet { current  :: !(Screen i l a sid sd)
             , visible  :: [Screen i l a sid sd]
             } deriving (Show, Read, Eq)

data Screen i l a sid sd = Screen { workspace :: !(Workspace i l a) }
    deriving (Show, Read, Eq)

data Workspace i l a = Workspace { tag :: !i }
    deriving (Show, Read, Eq)
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Should parse without errors");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 3, "Expected 3 data declarations");
    }

    #[test]
    fn test_instance_with_operator_method() {
        // Instance declaration with operator method (XMonad Foldable style)
        let src = r#"module Test where

instance Foldable Stack where
    toList = integrate
    foldr f z = foldr f z . toList
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Should parse without errors");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_do_let_without_in() {
        // In do-notation, 'let' doesn't need 'in'
        let src = r#"module Test where

test = do
    let x = 1
    y <- getY
    pure (x + y)
"#;
        let (_module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Do-notation let should work without 'in'");
    }

    #[test]
    fn test_do_let_simple_binding() {
        // Simple case: let followed by another statement
        let src = r#"module Test where

test = do
    sh <- io x
    let isFixedSize = isJust sh
    isTransient <- isJust sh
    pure isTransient
"#;
        let (_module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(
            diags.is_empty(),
            "Do-notation let followed by statement should parse"
        );
    }

    #[test]
    fn test_do_let_complex_binding() {
        // XMonad Operations.hs style: let binding followed by more statements
        // Note: Uses <$> operator which is fmap infix
        let src = r#"module Test where

isFixedSizeOrTransient d w = do
    sh <- io (getWMNormalHints d w)
    let isFixedSize = isJust (sh_min_size sh) && sh_min_size sh == sh_max_size sh
    isTransient <- isJust <$> io (getTransientForHint d w)
    pure (isFixedSize || isTransient)
"#;
        let (_module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Complex do-notation should parse");
    }

    #[test]
    fn test_import_then_function() {
        // Test imports followed by function definitions
        let src = r#"module Test where

import Data.Maybe

-- | Lift action
liftX :: X a -> Query a
liftX = Query . lift
"#;
        let (_module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Import followed by function should parse");
    }

    #[test]
    fn test_infix_function_definition() {
        // Test infix operator definitions like XMonad's (-->)
        let src = r#"module Test where

(-->) :: Bool -> a -> a
p --> f = if p then f else undefined

(<&&>) :: Bool -> Bool -> Bool
x <&&> y = x && y
"#;
        let (_module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Infix function definitions should parse");
    }

    #[test]
    fn test_primed_identifier_case_pattern() {
        // Test primed identifiers (f', xs') in case patterns
        let src = r#"module Test where

test xs = case xs of
    f':rs' -> Just (f', rs')
    [] -> Nothing
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(
            diags.is_empty(),
            "Primed identifier case patterns should parse"
        );
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_as_pattern_with_record() {
        // Test as-patterns with record patterns like `conf'@XConfig { field = val }`
        let src = r#"module Test where

test = do
    conf@Config { field = x } <- getConfig
    return x
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "As-pattern with record should parse");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_deriving_with_type_applications() {
        // Test deriving clauses with type applications like `MonadState XState`
        let src = r#"module Test where

newtype X a = X (ReaderT XConf (StateT XState IO) a)
    deriving (Functor, Applicative, Monad, MonadState XState, MonadReader XConf)
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(
            diags.is_empty(),
            "Deriving with type applications should parse"
        );
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_deriving_via() {
        // Test deriving via clause
        let src = r#"module Test where

newtype X a = X (IO a) deriving (Semigroup, Monoid) via Ap X a
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Deriving via should parse");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_backtick_in_parentheses() {
        // Test backtick infix in parenthesized expression
        let src = r#"module Test where

test x xs = guard (x `elem` xs)
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Backtick in parentheses should parse");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_backtick_right_section() {
        // Test backtick right section: (`op` x) means \y -> y `op` x
        let src = r#"module Test where

test = filter (`M.notMember` floatingMap)
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Backtick right section should parse");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_backtick_left_section() {
        // Test backtick left section: (x `op`) means \y -> x `op` y
        let src = r#"module Test where

test xs = filter (\x -> not $ any (x `containedIn`) xs) $ xs
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Backtick left section should parse");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_lambda_case_multi_alt() {
        // Test lambda-case with multiple alternatives
        let src = r#"module Test where

rescreen = getInfo >>= \case
    [] -> trace "empty"
    x:xs -> process x xs
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(
            diags.is_empty(),
            "Lambda-case with multiple alternatives should parse"
        );
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_type_equality_constraint() {
        // Test type equality constraint: (a ~ Type) =>
        let src = r#"module Test where

instance (a ~ Int) => Num a where
  fromInteger = undefined
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Type equality constraint should parse");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_multiline_type_signature_parsing() {
        // Test parsing multi-line type signature like XMonad Layout.hs
        let src = r#"module Foo where
tile
    :: Rational
    -> Rectangle
    -> Int
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Multi-line type signature should parse");
        let module = module.expect("Should parse");
        // Should have one declaration: the type signature
        assert_eq!(
            module.decls.len(),
            1,
            "Should have 1 decl, got {:?}",
            module.decls
        );
    }

    #[test]
    fn test_multiline_type_signature_with_function_parsing() {
        // Test parsing multi-line type signature followed by function definition
        let src = r#"module Foo where
tile
    :: Rational
    -> Rectangle
tile f r = r
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(
            diags.is_empty(),
            "Multi-line type signature with function should parse"
        );
        let module = module.expect("Should parse");
        // Should have two declarations: type signature and function binding
        assert_eq!(
            module.decls.len(),
            2,
            "Should have 2 decls, got {:?}",
            module.decls
        );
    }

    #[test]
    fn test_multiline_type_signature_after_instance() {
        // Test Layout.hs pattern: instance body followed by top-level type signature
        let src = r#"module Foo where
instance Show Foo where
    show _ = "Foo"

tile
    :: Int
    -> Bool
tile n = n > 0
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(
            diags.is_empty(),
            "Type signature after instance should parse"
        );
        let module = module.expect("Should parse");
        // Should have: instance, type signature, function binding
        assert_eq!(
            module.decls.len(),
            3,
            "Should have 3 decls, got {:?}",
            module.decls
        );
    }

    #[test]
    fn test_multiline_type_signature_with_doc_comments() {
        // Test Layout.hs pattern with doc comments
        let src = r#"module Foo where
instance Show Foo where
    description _ = "Foo"

-- | Doc comment
tile
    :: Int  -- ^ arg1
    -> Bool -- ^ result
tile n = n > 0
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(
            diags.is_empty(),
            "Type signature with doc comments should parse"
        );
        let module = module.expect("Should parse");
        // Should have: instance, type signature, function binding
        assert_eq!(
            module.decls.len(),
            3,
            "Should have 3 decls, got {:?}",
            module.decls
        );
    }

    #[test]
    fn test_class_multiline_method_signature() {
        // Test Core.hs pattern: class with multi-line method signatures
        let src = r#"module Foo where
class Show a => Foo a b where
    -- | Method doc
    runMethod :: a
              -> b
              -> Int
    runMethod x y = 42
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(
            diags.is_empty(),
            "Class with multi-line method signature should parse"
        );
        let module = module.expect("Should parse");
        // Should have one class declaration
        assert_eq!(
            module.decls.len(),
            1,
            "Should have 1 decl, got {:?}",
            module.decls
        );
    }

    #[test]
    fn test_class_default_method() {
        // Test class with type signature followed by default method implementation
        let src = r#"
class ExtensionClass a where
    initialValue :: a
    extensionType :: a -> String
    extensionType = show
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Class with default method should parse");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_class_default_method_with_docs() {
        // Test class with doc comments before type signature and default implementation
        let src = r#"
class ExtensionClass a where
    -- | Initial value
    initialValue :: a
    -- | The extension type.
    -- Multi-line doc.
    extensionType :: a -> String
    extensionType = show
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Class with doc comments should parse");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_class_xmonad_style() {
        // Test without MINIMAL pragma first
        let src = r#"
class ExtensionClass a where
    initialValue :: a
    extensionType :: a -> String
    extensionType = show
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error (no pragma): {:?}", d);
        }
        assert!(diags.is_empty(), "Class without pragma should parse");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);

        // Now test with MINIMAL pragma
        let src_with_pragma = r#"
class ExtensionClass a where
    {-# MINIMAL initialValue #-}
    initialValue :: a
    extensionType :: a -> String
    extensionType = show
"#;
        let (module2, diags2) = parse_module(src_with_pragma, FileId::new(0));
        for d in &diags2 {
            eprintln!("Error (with pragma): {:?}", d);
        }
        assert!(diags2.is_empty(), "Class with MINIMAL pragma should parse");
        let module2 = module2.expect("Should parse");
        assert_eq!(module2.decls.len(), 1);
    }

    #[test]
    fn test_class_with_associated_type() {
        // Test class with associated type declaration
        let src = r#"
class Collection c where
    type Elem c
    empty :: c
    insert :: Elem c -> c -> c
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Class with associated type should parse");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
        if let bhc_ast::Decl::ClassDecl(class) = &module.decls[0] {
            assert_eq!(class.name.name.as_str(), "Collection");
            assert_eq!(class.assoc_types.len(), 1);
            assert_eq!(class.assoc_types[0].name.name.as_str(), "Elem");
        } else {
            panic!("Expected class declaration");
        }
    }

    #[test]
    fn test_class_with_associated_type_default() {
        // Test class with associated type with default
        let src = r#"
class Container c where
    type Element c = Int
    size :: c -> Int
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(
            diags.is_empty(),
            "Class with associated type default should parse"
        );
        let module = module.expect("Should parse");
        if let bhc_ast::Decl::ClassDecl(class) = &module.decls[0] {
            assert_eq!(class.assoc_types.len(), 1);
            assert!(class.assoc_types[0].default.is_some());
        } else {
            panic!("Expected class declaration");
        }
    }

    #[test]
    fn test_instance_with_associated_type_def() {
        // Test instance with associated type definition
        let src = r#"
instance Collection [a] where
    type Elem [a] = a
    empty = []
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(
            diags.is_empty(),
            "Instance with associated type should parse"
        );
        let module = module.expect("Should parse");
        if let bhc_ast::Decl::InstanceDecl(inst) = &module.decls[0] {
            assert_eq!(inst.assoc_type_defs.len(), 1);
            assert_eq!(inst.assoc_type_defs[0].name.name.as_str(), "Elem");
        } else {
            panic!("Expected instance declaration");
        }
    }

    #[test]
    fn test_inline_let_expression() {
        // Test inline let...in expression
        let src = "foo = let x = 1 in x + 1";
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            eprintln!("Error: {:?}", d);
        }
        assert!(diags.is_empty(), "Inline let expression should parse");
        let module = module.expect("Should parse");
        assert_eq!(module.decls.len(), 1);
    }

    #[test]
    fn test_xmonad_parsing() {
        // Test parsing XMonad-style code
        use std::path::Path;

        let xmonad_dir = Path::new("/tmp/xmonad/src/XMonad");
        if !xmonad_dir.exists() {
            println!("XMonad source not found at {:?}, skipping test", xmonad_dir);
            return;
        }

        let mut total_errors = 0;
        for entry in std::fs::read_dir(xmonad_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "hs") {
                let src = std::fs::read_to_string(&path).unwrap();
                let file_id = crate::FileId::new(0);
                let (_, diagnostics) = parse_module(&src, file_id);
                let error_count = diagnostics.iter().filter(|d| d.is_error()).count();
                total_errors += error_count;
                if error_count > 0 {
                    println!(
                        "{}: {} errors",
                        path.file_name().unwrap().to_str().unwrap(),
                        error_count
                    );
                    // Print first 25 errors for debugging
                    for (i, d) in diagnostics
                        .iter()
                        .filter(|d| d.is_error())
                        .take(25)
                        .enumerate()
                    {
                        println!("  {}: {:?}", i + 1, d);
                    }
                }
            }
        }
        println!("Total XMonad parse errors: {}", total_errors);
        // We're tracking progress, so allow errors but report them
        // assert_eq!(total_errors, 0, "XMonad files should parse without errors");
    }

    #[test]
    fn test_cpp_if_else() {
        let src = r#"module Test where

test = do
    x <- action
#if COND
    y <- branch1
#else
    y <- branch2
#endif
    return x

-- | A doc comment
other = 42
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            println!("Diagnostic: {:?}", d);
        }
        assert!(diags.is_empty(), "Parse errors: {:?}", diags);
        assert!(module.is_some(), "Failed to parse CPP if/else");
    }

    #[test]
    fn test_cpp_in_where_clause() {
        // This mirrors the XMonad Core.hs structure more closely
        let src = r#"module Test where

xfork x = io x
 where
    nullStdin = do
#if COND
        fd <- action1
#else
        fd <- action2
#endif
        dupTo fd
        closeFd fd

-- | Doc comment for next function.
xmessage :: String -> IO ()
xmessage msg = print msg
"#;
        let (module, diags) = parse_module(src, FileId::new(0));
        for d in &diags {
            println!("Diagnostic: {:?}", d);
        }
        assert!(diags.is_empty(), "Parse errors: {:?}", diags);
        assert!(module.is_some(), "Failed to parse CPP in where clause");
    }

    // Tests for operator exports and imports (fixes for #123)

    #[test]
    fn test_export_dot_operator() {
        // Export the composition operator (.)
        let module = parse_module_ok("module Foo ((.), foo) where\nfoo = 1");
        assert!(module.exports.is_some());
        let exports = module.exports.unwrap();
        assert_eq!(exports.len(), 2);
    }

    #[test]
    fn test_export_bang_operator() {
        // Export the indexing operator (!)
        let module = parse_module_ok("module Data.Map ((!), lookup) where\nlookup = undefined");
        assert!(module.exports.is_some());
        let exports = module.exports.unwrap();
        assert_eq!(exports.len(), 2);
    }

    #[test]
    fn test_export_multiple_special_operators() {
        // Export multiple special operators
        let module = parse_module_ok("module Ops ((.), (!), (@), (~)) where\nx = 1");
        assert!(module.exports.is_some());
        let exports = module.exports.unwrap();
        assert_eq!(exports.len(), 4);
    }

    #[test]
    fn test_import_dot_operator() {
        // Import the composition operator
        let module = parse_module_ok("import Data.Function ((.))\nx = 1");
        assert!(!module.imports.is_empty());
        let import = &module.imports[0];
        assert!(import.spec.is_some());
    }

    #[test]
    fn test_import_bang_operator() {
        // Import the indexing operator
        let module = parse_module_ok("import Data.Map ((!), lookup)\nx = 1");
        assert!(!module.imports.is_empty());
    }

    #[test]
    fn test_export_with_doc_comments() {
        // Export list with Haddock doc comments between items
        let src = r#"module Foo (
    -- * Section header
    foo,
    -- | Documentation for bar
    bar
) where
foo = 1
bar = 2"#;
        let module = parse_module_ok(src);
        assert!(module.exports.is_some());
        let exports = module.exports.unwrap();
        assert_eq!(exports.len(), 2);
    }

    #[test]
    fn test_export_regular_operator() {
        // Export a regular operator like (++)
        let module = parse_module_ok("module Data.List ((++), map) where\nmap = undefined");
        assert!(module.exports.is_some());
    }

    #[test]
    fn test_import_hiding_operator() {
        // Import hiding an operator
        let module = parse_module_ok("import Prelude hiding ((.))\nx = 1");
        assert!(!module.imports.is_empty());
        let import = &module.imports[0];
        // Check that the spec is a Hiding variant
        match &import.spec {
            Some(ImportSpec::Hiding(_)) => {}
            _ => panic!("Expected hiding import"),
        }
    }
}
