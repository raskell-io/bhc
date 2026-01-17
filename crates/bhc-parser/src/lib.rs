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

mod expr;
mod decl;
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

    /// Get the current token kind.
    fn current_kind(&self) -> Option<&TokenKind> {
        self.current().map(|t| &t.node.kind)
    }

    /// Get the current span.
    fn current_span(&self) -> Span {
        self.current()
            .map(|t| t.span)
            .unwrap_or(Span::DUMMY)
    }

    /// Check if we're at the end of input.
    fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len()
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

#[cfg(test)]
mod tests {
    use super::*;

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
    #[ignore] // Left sections need more work to parse correctly
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
    #[ignore] // Negation needs lexer changes to properly distinguish prefix -
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
    #[ignore] // The `:` is lexed as ConOperator, needs lexer fix
    fn test_pattern_infix() {
        let module = parse_module_ok("f (x : xs) = xs");
        assert!(!module.decls.is_empty());
    }

    #[test]
    #[ignore] // The `:` is lexed as ConOperator, needs lexer fix
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
    #[ignore] // Layout rule inserts tokens that interfere with record parsing
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
    #[ignore] // forall parsing needs work on `.` handling
    fn test_forall_type() {
        let module = parse_module_ok("f :: forall a. a -> a");
        assert!(!module.decls.is_empty());
    }

    // Module structure tests

    #[test]
    #[ignore] // Layout rule interference with module parsing
    fn test_module_header() {
        let module = parse_module_ok("module Foo where\nx = 1");
        assert!(module.name.is_some());
    }

    #[test]
    #[ignore] // Layout rule interference with module parsing
    fn test_module_exports() {
        let module = parse_module_ok("module Foo (bar, baz) where\nbar = 1\nbaz = 2");
        assert!(module.exports.is_some());
    }

    #[test]
    #[ignore] // Qualified module names need lexer/parser coordination
    fn test_imports() {
        let module = parse_module_ok("import Data.List\nx = 1");
        assert!(!module.imports.is_empty());
    }

    #[test]
    #[ignore] // Qualified module names need lexer/parser coordination
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
    #[ignore] // Layout rule interference with class parsing
    fn test_class_declaration() {
        let module = parse_module_ok("class Eq a where\n  eq :: a -> a -> Bool");
        assert!(!module.decls.is_empty());
    }

    #[test]
    #[ignore] // Layout rule interference with instance parsing
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
        let module = parse_module_ok("{-# LANGUAGE GADTs #-}\n{-# LANGUAGE TypeFamilies, DataKinds #-}\nx = 1");
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
}
