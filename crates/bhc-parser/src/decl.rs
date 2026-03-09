//! Declaration parsing.

use bhc_ast::*;
use bhc_intern::{Ident, Symbol};
use bhc_lexer::TokenKind;
use bhc_span::Span;

use crate::{ParseError, ParseResult, Parser};

impl<'src> Parser<'src> {
    /// Parse a complete module.
    pub fn parse_module(&mut self) -> ParseResult<Module> {
        let start = self.current_span();

        // Parse pragmas at the start of the module
        let pragmas = self.parse_pragmas();

        // Collect doc comments that may appear before the module declaration (Haddock)
        let doc = self.collect_doc_comments();

        // Skip virtual tokens (VirtualSemi) that the layout rule may insert
        // between a doc comment and the `module` keyword. The doc comment is
        // the first real token, so the layout rule's `first_token` guard is
        // already cleared and a VirtualSemi gets emitted before `module`.
        self.skip_virtual_tokens();

        // Optional module header
        let (name, exports) = if self.eat(&TokenKind::Module) {
            let name = self.parse_module_name()?;
            // Skip virtual tokens between module name and export list `(` —
            // when the `(` starts at column 1 the layout rule emits VirtualSemi.
            self.skip_virtual_tokens();
            let exports = if self.check(&TokenKind::LParen) {
                Some(self.parse_export_list()?)
            } else {
                None
            };
            // Skip virtual tokens that the layout rule may insert between
            // the export list `)` and `where` when they are on separate lines:
            //   module Foo ( bar )
            //   where          -- `where` on its own line
            self.skip_virtual_tokens();
            self.expect(&TokenKind::Where)?;
            (Some(name), exports)
        } else {
            (None, None)
        };

        // Skip virtual brace from layout rule after `where`
        self.skip_virtual_tokens();

        // Imports
        let mut imports = Vec::new();
        while self.check(&TokenKind::Import) {
            imports.push(self.parse_import()?);
            // Skip virtual semicolons between imports
            self.skip_virtual_tokens();
        }

        // Declarations
        let mut decls = Vec::new();
        while !self.at_eof() {
            // Skip any virtual tokens, pragmas, and explicit semicolons between declarations
            // Note: We do NOT skip doc comments here, since parse_top_decl collects them
            self.skip_virtual_tokens();
            self.skip_standalone_pragmas();
            while self.eat(&TokenKind::Semi) || self.eat(&TokenKind::VirtualSemi) {
                self.skip_virtual_tokens();
                self.skip_standalone_pragmas();
            }
            if self.at_eof() {
                break;
            }

            match self.parse_top_decl() {
                Ok(decl) => decls.push(decl),
                Err(e) => {
                    // Try to recover
                    self.emit(e.to_diagnostic(self.file_id));
                    self.recover_to_next_decl();
                    if self.at_eof() {
                        break;
                    }
                }
            }
        }

        let span = start.to(self.tokens.last().map(|t| t.span).unwrap_or(start));

        // Merge consecutive clauses of the same function
        let decls = self.merge_function_clauses(decls);

        Ok(Module {
            doc,
            pragmas,
            name,
            exports,
            imports,
            decls,
            span,
        })
    }

    /// Merge consecutive function bindings with the same name into multi-clause functions.
    ///
    /// In Haskell, functions can be defined with multiple clauses:
    /// ```haskell
    /// fac 0 = 1
    /// fac n = n * fac (n - 1)
    /// ```
    ///
    /// These are parsed as separate `FunBind` declarations initially,
    /// and this function merges them into a single `FunBind` with multiple clauses.
    fn merge_function_clauses(&self, decls: Vec<Decl>) -> Vec<Decl> {
        let mut result: Vec<Decl> = Vec::new();

        for decl in decls {
            match decl {
                Decl::FunBind(mut fun_bind) => {
                    // Don't merge $patbind declarations - each pattern binding is independent
                    // (they all have the same synthetic name but are not clauses of the same function)
                    if fun_bind.name.name.as_str() == "$patbind" {
                        result.push(Decl::FunBind(fun_bind));
                        continue;
                    }

                    // Check if the last declaration is a FunBind with the same name
                    if let Some(Decl::FunBind(ref mut last)) = result.last_mut() {
                        if last.name.name == fun_bind.name.name {
                            // Merge clauses
                            last.clauses.append(&mut fun_bind.clauses);
                            // Update span to include all clauses
                            last.span = last.span.to(fun_bind.span);
                            continue;
                        }
                    }
                    result.push(Decl::FunBind(fun_bind));
                }
                other => result.push(other),
            }
        }

        result
    }

    /// Parse pragmas at the start of a module.
    ///
    /// Consumes all consecutive Pragma tokens and parses their contents.
    /// Skips virtual tokens (VirtualSemi) between pragmas.
    fn parse_pragmas(&mut self) -> Vec<Pragma> {
        let mut pragmas = Vec::new();

        loop {
            // Skip virtual tokens between pragmas
            while let Some(kind) = self.current_kind() {
                if kind.is_virtual() {
                    self.advance();
                } else {
                    break;
                }
            }

            if let Some(tok) = self.current() {
                if let TokenKind::Pragma(content) = &tok.node.kind {
                    let span = tok.span;
                    let content = content.clone();
                    self.advance();

                    if let Some(pragma) = self.parse_pragma_content(&content, span) {
                        pragmas.push(pragma);
                    }
                    continue;
                }
            }
            break;
        }

        pragmas
    }

    /// Parse the content of a pragma.
    ///
    /// The content is the text between `{-#` and `#-}`, e.g., `"LANGUAGE GADTs, TypeFamilies"`.
    fn parse_pragma_content(&self, content: &str, span: Span) -> Option<Pragma> {
        let content = content.trim();
        if content.is_empty() {
            return None;
        }

        // Split into pragma name and arguments
        let mut parts = content.splitn(2, char::is_whitespace);
        let pragma_name = parts.next()?.to_uppercase();
        let args = parts.next().unwrap_or("").trim();

        let kind = match pragma_name.as_str() {
            "LANGUAGE" => {
                // Parse comma-separated extension names
                let extensions: Vec<Symbol> = args
                    .split(',')
                    .map(|s| Symbol::intern(s.trim()))
                    .filter(|s| !s.as_str().is_empty())
                    .collect();
                PragmaKind::Language(extensions)
            }
            "OPTIONS_GHC" | "OPTIONS" => PragmaKind::OptionsGhc(args.to_string()),
            "INLINE" => {
                if let Some(name) = args.split_whitespace().next() {
                    PragmaKind::Inline(Ident::new(Symbol::intern(name)))
                } else {
                    PragmaKind::Other(content.to_string())
                }
            }
            "NOINLINE" | "NOTINLINE" => {
                if let Some(name) = args.split_whitespace().next() {
                    PragmaKind::NoInline(Ident::new(Symbol::intern(name)))
                } else {
                    PragmaKind::Other(content.to_string())
                }
            }
            "INLINABLE" | "INLINEABLE" => {
                if let Some(name) = args.split_whitespace().next() {
                    PragmaKind::Inlinable(Ident::new(Symbol::intern(name)))
                } else {
                    PragmaKind::Other(content.to_string())
                }
            }
            "UNPACK" => PragmaKind::Unpack,
            "NOUNPACK" => PragmaKind::NoUnpack,
            "SOURCE" => PragmaKind::Source,
            "COMPLETE" => {
                let names: Vec<Ident> = args
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| Ident::new(Symbol::intern(s)))
                    .collect();
                PragmaKind::Complete(names)
            }
            "MINIMAL" => PragmaKind::Minimal(args.to_string()),
            "DEPRECATED" => {
                // Simple parsing: everything after DEPRECATED is the message
                // More sophisticated parsing would extract specific names
                PragmaKind::Deprecated(None, args.to_string())
            }
            "WARNING" => PragmaKind::Warning(None, args.to_string()),
            "SPECIALIZE" | "SPECIALISE" => {
                // For now, store as Other since we'd need to parse the type signature
                PragmaKind::Other(content.to_string())
            }
            _ => PragmaKind::Other(content.to_string()),
        };

        Some(Pragma { kind, span })
    }

    /// Skip standalone pragmas that can appear between declarations.
    /// This includes INLINE, NOINLINE, DEPRECATED, WARNING, SPECIALISE, etc.
    fn skip_standalone_pragmas(&mut self) {
        while let Some(TokenKind::Pragma(_)) = self.current_kind() {
            self.advance();
        }
    }

    /// Parse a module name.
    fn parse_module_name(&mut self) -> ParseResult<ModuleName> {
        let start = self.current_span();
        let mut parts = Vec::new();

        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "module name".to_string(),
        })?;

        match &tok.node.kind {
            // Simple module name: Foo
            TokenKind::ConId(sym) => {
                parts.push(*sym);
                self.advance();

                // Check for dot continuation (for manual A.B.C parsing)
                while let Some(next) = self.current() {
                    if matches!(&next.node.kind, TokenKind::Operator(s) if s.as_str() == ".") {
                        self.advance();
                        // Expect another ConId
                        if let Some(tok) = self.current() {
                            if let TokenKind::ConId(sym) = &tok.node.kind {
                                parts.push(*sym);
                                self.advance();
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
            // Qualified module name: Data.List (lexer produces this as single token)
            TokenKind::QualConId(qualifier, name) => {
                // Split qualifier on dots and add parts
                for part in qualifier.as_str().split('.') {
                    parts.push(Symbol::intern(part));
                }
                parts.push(*name);
                self.advance();
            }
            _ => {
                return Err(ParseError::Unexpected {
                    found: tok.node.kind.description().to_string(),
                    expected: "module name".to_string(),
                    span: tok.span,
                });
            }
        }

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(ModuleName { parts, span })
    }

    /// Parse an export list.
    fn parse_export_list(&mut self) -> ParseResult<Vec<Export>> {
        self.expect(&TokenKind::LParen)?;
        let mut exports = Vec::new();

        // Skip any doc comments at the start of the export list
        self.skip_doc_comments();

        if !self.check(&TokenKind::RParen) {
            exports.push(self.parse_export()?);
            while self.eat(&TokenKind::Comma) {
                // Skip doc comments between export items (Haddock section headers)
                self.skip_doc_comments();
                if self.check(&TokenKind::RParen) {
                    break;
                }
                exports.push(self.parse_export()?);
            }
        }

        self.expect(&TokenKind::RParen)?;
        Ok(exports)
    }

    /// Parse a single export item.
    fn parse_export(&mut self) -> ParseResult<Export> {
        let start = self.current_span();

        if self.eat(&TokenKind::Module) {
            let name = self.parse_module_name()?;
            let span = start.to(name.span);
            return Ok(Export::Module(name, span));
        }

        // Check for `pattern` prefix (context-sensitive keyword for pattern synonym exports)
        if self.check_ident_str("pattern") {
            let pat_start = self.current_span();
            self.advance(); // consume `pattern`
            let tok = self.current().ok_or(ParseError::UnexpectedEof {
                expected: "pattern synonym name".to_string(),
            })?;
            if let TokenKind::ConId(sym) = &tok.node.kind {
                let ident = Ident::new(*sym);
                let span = pat_start.to(tok.span);
                self.advance();
                return Ok(Export::Pattern(ident, span));
            }
            return Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "pattern synonym name (constructor)".to_string(),
                span: tok.span,
            });
        }

        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "export item".to_string(),
        })?;

        match &tok.node.kind.clone() {
            TokenKind::Ident(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();
                Ok(Export::Var(ident, span))
            }
            TokenKind::ConId(sym) => {
                let ident = Ident::new(*sym);
                self.advance();

                let constrs = if self.check(&TokenKind::LParen) {
                    Some(self.parse_subspec()?)
                } else {
                    None
                };

                let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
                Ok(Export::Type(ident, constrs, span))
            }
            // Handle operators in parentheses: module Foo ((+), (.), (!)) where
            TokenKind::LParen => {
                self.advance(); // consume (

                // Expect an operator
                let op_tok = self.current().ok_or(ParseError::UnexpectedEof {
                    expected: "operator".to_string(),
                })?;

                let ident = match &op_tok.node.kind {
                    TokenKind::Operator(sym) => Ident::new(*sym),
                    TokenKind::ConOperator(sym) => Ident::new(*sym),
                    // Special tokens that are valid operators when in parentheses
                    TokenKind::Dot => Ident::new(Symbol::intern(".")),
                    TokenKind::Bang => Ident::new(Symbol::intern("!")),
                    TokenKind::At => Ident::new(Symbol::intern("@")),
                    TokenKind::Tilde => Ident::new(Symbol::intern("~")),
                    _ => {
                        return Err(ParseError::Unexpected {
                            found: op_tok.node.kind.description().to_string(),
                            expected: "operator".to_string(),
                            span: op_tok.span,
                        });
                    }
                };
                self.advance();

                let end = self.expect(&TokenKind::RParen)?;
                let span = start.to(end.span);
                Ok(Export::Var(ident, span))
            }
            _ => Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "export item".to_string(),
                span: tok.span,
            }),
        }
    }

    /// Parse a subspecification like `(..)` or `(A, B)`.
    fn parse_subspec(&mut self) -> ParseResult<Vec<Ident>> {
        self.expect(&TokenKind::LParen)?;

        if self.eat(&TokenKind::DotDot) {
            self.expect(&TokenKind::RParen)?;
            return Ok(vec![]); // (..) means all
        }

        let mut names = Vec::new();
        if !self.check(&TokenKind::RParen) {
            names.push(self.parse_ident_or_con()?);
            while self.eat(&TokenKind::Comma) {
                if self.check(&TokenKind::RParen) {
                    break;
                }
                names.push(self.parse_ident_or_con()?);
            }
        }

        self.expect(&TokenKind::RParen)?;
        Ok(names)
    }

    /// Parse an identifier or constructor.
    /// Also handles operators in parentheses like `(:|)` or `(++)`.
    fn parse_ident_or_con(&mut self) -> ParseResult<Ident> {
        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "identifier".to_string(),
        })?;

        match &tok.node.kind {
            TokenKind::Ident(sym) | TokenKind::ConId(sym) => {
                let ident = Ident::new(*sym);
                self.advance();
                Ok(ident)
            }
            // Handle operators in parentheses: (:|), (++)
            TokenKind::LParen => {
                self.advance(); // consume (
                let op_tok = self.current().ok_or(ParseError::UnexpectedEof {
                    expected: "operator".to_string(),
                })?;

                let ident = match &op_tok.node.kind {
                    TokenKind::Operator(sym) | TokenKind::ConOperator(sym) => Ident::new(*sym),
                    _ => {
                        return Err(ParseError::Unexpected {
                            found: op_tok.node.kind.description().to_string(),
                            expected: "operator".to_string(),
                            span: op_tok.span,
                        });
                    }
                };
                self.advance();
                self.expect(&TokenKind::RParen)?;
                Ok(ident)
            }
            _ => Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "identifier".to_string(),
                span: tok.span,
            }),
        }
    }

    /// Parse an import declaration.
    pub fn parse_import(&mut self) -> ParseResult<ImportDecl> {
        let start = self.current_span();
        self.expect(&TokenKind::Import)?;

        let qualified = self.eat(&TokenKind::Qualified);
        let module = self.parse_module_name()?;

        // Check for "as Alias" - 'as' is a context-sensitive keyword
        let alias = if self.eat_ident_str("as") {
            Some(self.parse_module_name()?)
        } else {
            None
        };

        let spec = self.parse_import_spec()?;
        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);

        Ok(ImportDecl {
            module,
            qualified,
            alias,
            spec,
            span,
        })
    }

    /// Parse an import specification.
    fn parse_import_spec(&mut self) -> ParseResult<Option<ImportSpec>> {
        // Check for "hiding"
        if self.eat(&TokenKind::Hiding) {
            let imports = self.parse_import_list()?;
            return Ok(Some(ImportSpec::Hiding(imports)));
        }

        if self.check(&TokenKind::LParen) {
            let imports = self.parse_import_list()?;
            return Ok(Some(ImportSpec::Only(imports)));
        }

        Ok(None)
    }

    /// Parse an import list.
    fn parse_import_list(&mut self) -> ParseResult<Vec<Import>> {
        self.expect(&TokenKind::LParen)?;
        let mut imports = Vec::new();

        if !self.check(&TokenKind::RParen) {
            imports.push(self.parse_import_item()?);
            while self.eat(&TokenKind::Comma) {
                if self.check(&TokenKind::RParen) {
                    break;
                }
                imports.push(self.parse_import_item()?);
            }
        }

        self.expect(&TokenKind::RParen)?;
        Ok(imports)
    }

    /// Parse a single import item.
    fn parse_import_item(&mut self) -> ParseResult<Import> {
        // Check for `pattern` prefix (context-sensitive keyword for pattern synonym imports)
        if self.check_ident_str("pattern") {
            let start = self.current_span();
            self.advance(); // consume `pattern`
            let tok = self.current().ok_or(ParseError::UnexpectedEof {
                expected: "pattern synonym name".to_string(),
            })?;
            if let TokenKind::ConId(sym) = &tok.node.kind {
                let ident = Ident::new(*sym);
                let span = start.to(tok.span);
                self.advance();
                return Ok(Import::Pattern(ident, span));
            }
            return Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "pattern synonym name (constructor)".to_string(),
                span: tok.span,
            });
        }

        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "import item".to_string(),
        })?;

        match &tok.node.kind.clone() {
            TokenKind::Ident(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();
                Ok(Import::Var(ident, span))
            }
            TokenKind::ConId(sym) => {
                let ident = Ident::new(*sym);
                let start = tok.span;
                self.advance();

                let constrs = if self.check(&TokenKind::LParen) {
                    Some(self.parse_subspec()?)
                } else {
                    None
                };

                let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
                Ok(Import::Type(ident, constrs, span))
            }
            // Handle operators in parentheses: import Data.Bits ((.|.))
            TokenKind::LParen => {
                let start = tok.span;
                self.advance(); // consume (

                // Expect an operator
                let op_tok = self.current().ok_or(ParseError::UnexpectedEof {
                    expected: "operator".to_string(),
                })?;

                let ident = match &op_tok.node.kind {
                    TokenKind::Operator(sym) => Ident::new(*sym),
                    TokenKind::ConOperator(sym) => Ident::new(*sym),
                    // Special tokens that are valid operators when in parentheses
                    TokenKind::Dot => Ident::new(Symbol::intern(".")),
                    TokenKind::Bang => Ident::new(Symbol::intern("!")),
                    TokenKind::At => Ident::new(Symbol::intern("@")),
                    TokenKind::Tilde => Ident::new(Symbol::intern("~")),
                    _ => {
                        return Err(ParseError::Unexpected {
                            found: op_tok.node.kind.description().to_string(),
                            expected: "operator".to_string(),
                            span: op_tok.span,
                        });
                    }
                };
                self.advance();

                let end = self.expect(&TokenKind::RParen)?;
                let span = start.to(end.span);
                Ok(Import::Var(ident, span))
            }
            _ => Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "import item".to_string(),
                span: tok.span,
            }),
        }
    }

    /// Parse a top-level declaration.
    fn parse_top_decl(&mut self) -> ParseResult<Decl> {
        // Collect Haddock documentation comments before declarations
        let doc = self.collect_doc_comments();

        // After doc comments, there may be a VirtualSemi if the next token
        // is at the same indentation level. Skip it.
        self.eat(&TokenKind::VirtualSemi);

        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "declaration".to_string(),
        })?;

        match &tok.node.kind {
            TokenKind::Data => self.parse_data_decl_with_doc(doc),
            TokenKind::Type => self.parse_type_decl_with_doc(doc),
            TokenKind::Newtype => self.parse_newtype_decl_with_doc(doc),
            TokenKind::Class => self.parse_class_decl_with_doc(doc),
            TokenKind::Instance => self.parse_instance_decl_with_doc(doc),
            TokenKind::Foreign => self.parse_foreign_decl_with_doc(doc),
            TokenKind::Infix | TokenKind::Infixl | TokenKind::Infixr => self.parse_fixity_decl(),
            TokenKind::Deriving => self.parse_standalone_deriving(),
            TokenKind::Ident(sym) if sym.as_str() == "pattern" => {
                self.parse_pattern_synonym()
            }
            TokenKind::Ident(_) => {
                // Could be type signature or binding
                self.parse_value_decl_with_doc(doc)
            }
            TokenKind::LParen => {
                // Operator type signature or binding: (<+>) :: ... or (<+>) = ...
                self.parse_value_decl_with_doc(doc)
            }
            _ => Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "declaration".to_string(),
                span: tok.span,
            }),
        }
    }

    /// Parse local declarations (in let or where).
    pub fn parse_local_decls(&mut self) -> ParseResult<Vec<Decl>> {
        let mut decls = Vec::new();

        if self.eat(&TokenKind::LBrace) {
            // Explicit braces
            if !self.check(&TokenKind::RBrace) {
                decls.push(self.parse_value_decl()?);
                while self.eat(&TokenKind::Semi) {
                    if self.check(&TokenKind::RBrace) {
                        break;
                    }
                    decls.push(self.parse_value_decl()?);
                }
            }
            self.expect(&TokenKind::RBrace)?;
        } else if self.eat(&TokenKind::VirtualLBrace) {
            // Layout-based declarations (implicit braces)
            // Note: `in` can also end a let-block (e.g., `let x = 1 in ...`)
            if !self.check(&TokenKind::VirtualRBrace) && !self.check(&TokenKind::In) {
                decls.push(self.parse_value_decl()?);
                while self.eat(&TokenKind::VirtualSemi) {
                    if self.check(&TokenKind::VirtualRBrace) || self.check(&TokenKind::In) {
                        break;
                    }
                    decls.push(self.parse_value_decl()?);
                }
            }
            // Accept either VirtualRBrace or implicitly end on `in`
            if !self.eat(&TokenKind::VirtualRBrace) && !self.check(&TokenKind::In) {
                // If neither, report error expecting VirtualRBrace
                return Err(ParseError::Unexpected {
                    found: self
                        .current()
                        .map(|t| t.node.kind.description().to_string())
                        .unwrap_or("end of file".to_string()),
                    expected: "layout `}`".to_string(),
                    span: self.current_span(),
                });
            }
        } else {
            // Single declaration (no braces)
            decls.push(self.parse_value_decl()?);
        }

        // Merge multi-clause functions in local declarations too
        Ok(self.merge_function_clauses(decls))
    }

    /// Parse a value declaration (type signature or binding).
    fn parse_value_decl(&mut self) -> ParseResult<Decl> {
        // Collect Haddock documentation comments before value declarations
        let doc = self.collect_doc_comments();
        self.parse_value_decl_with_doc(doc)
    }

    /// Parse a value declaration with an optional doc comment.
    fn parse_value_decl_with_doc(&mut self, doc: Option<DocComment>) -> ParseResult<Decl> {
        // After doc comments, there may be a VirtualSemi if the next token
        // is at the same indentation level. Skip it.
        self.eat(&TokenKind::VirtualSemi);

        // Handle pragmas in class/instance bodies (like {-# MINIMAL initialValue #-})
        if let Some(TokenKind::Pragma(content)) = self.current_kind() {
            let content = content.clone();
            let span = self.current_span();
            self.advance();
            if let Some(pragma) = self.parse_pragma_content(&content, span) {
                return Ok(Decl::PragmaDecl(pragma));
            }
            // If we couldn't parse the pragma content, return an empty pragma
            return Ok(Decl::PragmaDecl(Pragma {
                kind: PragmaKind::Other(content),
                span,
            }));
        }

        let start = self.current_span();

        // Check for pattern binding: `(a, b) = expr` or `!pat = expr` or `~pat = expr`
        // These start with a pattern, not a function name.
        // A pattern binding is `(pat, ...) = expr` where the paren contains patterns, not an operator.
        // To distinguish from `(<+>) = ...`, check if the token after `(` is NOT an operator.
        if self.check(&TokenKind::LParen) {
            // Peek ahead: is this (operator) or (pattern)?
            if let Some(next) = self.peek_nth(1) {
                let is_pattern_start = matches!(
                    next.node.kind,
                    TokenKind::Ident(_)
                        | TokenKind::ConId(_)
                        | TokenKind::Underscore
                        | TokenKind::IntLit(_)
                        | TokenKind::CharLit(_)
                        | TokenKind::StringLit(_)
                        | TokenKind::LParen
                        | TokenKind::LBracket
                        | TokenKind::Bang
                        | TokenKind::Tilde
                );
                if is_pattern_start {
                    // This is a pattern binding like (a, b) = expr
                    return self.parse_pattern_binding(start);
                }
            }
        }
        // Also handle bang patterns and lazy patterns at top level: !pat = ... or ~pat = ...
        if self.check(&TokenKind::Bang) || self.check(&TokenKind::Tilde) {
            return self.parse_pattern_binding(start);
        }

        // Handle declarations starting with a constructor (ConId):
        // This could be an infix operator definition like `Box f <*> Box x = Box (f x)`
        // (i.e., `pat varop pat = rhs`) or a pattern binding like `Box x = expr`.
        if matches!(self.current_kind(), Some(TokenKind::ConId(_))) {
            let saved_pos = self.pos;
            // Try parsing a pattern (e.g., `Box f`)
            if let Ok(left_pat) = self.parse_pattern() {
                if self.is_infix_var_op_start() {
                    // This is an infix operator definition: pat varop pat = rhs
                    let op_name = self.parse_infix_op()?;
                    let right_pat = self.parse_pattern()?;

                    let mut extra_pats = Vec::new();
                    while self.is_apat_start() {
                        extra_pats.push(self.parse_atom_pattern()?);
                    }

                    let rhs = self.parse_binding_rhs()?;

                    let wheres = if self.eat(&TokenKind::Where) {
                        self.parse_local_decls()?
                    } else {
                        vec![]
                    };

                    let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);

                    let mut pats = vec![left_pat, right_pat];
                    pats.extend(extra_pats);

                    let clause = Clause {
                        pats,
                        rhs,
                        wheres,
                        span,
                    };

                    return Ok(Decl::FunBind(FunBind {
                        doc: doc.clone(),
                        name: op_name,
                        clauses: vec![clause],
                        span,
                    }));
                } else if self.check(&TokenKind::Eq) {
                    // Pattern binding: `ConPat = expr`
                    self.expect(&TokenKind::Eq)?;
                    let expr = self.parse_expr()?;

                    let wheres = if self.eat(&TokenKind::Where) {
                        self.parse_local_decls()?
                    } else {
                        vec![]
                    };

                    let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);

                    let clause = Clause {
                        pats: vec![left_pat],
                        rhs: Rhs::Simple(expr, span),
                        wheres,
                        span,
                    };

                    return Ok(Decl::FunBind(FunBind {
                        doc: doc.clone(),
                        name: Ident::from_str("$patbind"),
                        clauses: vec![clause],
                        span,
                    }));
                } else {
                    // Not an infix definition or pattern binding, backtrack
                    self.pos = saved_pos;
                }
            } else {
                // Pattern parse failed, backtrack
                self.pos = saved_pos;
            }
        }

        // Parse either a regular identifier or a parenthesized operator like (<+>)
        let name = self.parse_var_or_op()?;

        // Check for multi-name type signature: `a, b, c :: Type`
        if self.check(&TokenKind::Comma) || self.check(&TokenKind::DoubleColon) {
            // Collect all names for a potential type signature
            let mut names = vec![name.clone()];
            while self.eat(&TokenKind::Comma) {
                names.push(self.parse_var_or_op()?);
            }

            if self.eat(&TokenKind::DoubleColon) {
                // Type signature
                let ty = self.parse_type()?;
                let span = start.to(ty.span());
                return Ok(Decl::TypeSig(TypeSig {
                    doc,
                    names,
                    ty,
                    span,
                }));
            } else if names.len() > 1 {
                // We parsed multiple names but no ::, this is an error
                let tok = self.current().unwrap();
                return Err(ParseError::Unexpected {
                    found: tok.node.kind.description().to_string(),
                    expected: "`::`".to_string(),
                    span: tok.span,
                });
            }
            // Single name, no ::, fall through to function binding
        }

        // Function binding - if we get here, we have a single name (stored in `name`)
        {
            // Function binding - could be prefix or infix
            let mut pats = Vec::new();

            // Check for infix binding: `x `op` y = ...` or `x --> y = ...`
            // BUT: if the operator is a constructor (starts with :), this is a pattern binding
            // e.g., `cur :| visi = expr` is a pattern binding, not a function definition
            let (actual_name, _is_infix) = if self.is_infix_con_op_start() {
                // Constructor operator: this is actually a pattern binding like `x :| xs = expr`
                // Rewind and parse as pattern binding
                let pat_start = start;
                // Build the infix pattern: name op pat
                let left_pat = Pat::Var(name.clone(), start);
                let op = self.parse_infix_op()?;
                let right_pat = self.parse_pattern()?;
                let pat_span = pat_start.to(right_pat.span());
                let full_pat = Pat::Infix(Box::new(left_pat), op, Box::new(right_pat), pat_span);

                // Parse `= expr`
                self.expect(&TokenKind::Eq)?;
                let expr = self.parse_expr()?;

                // Parse optional where clause
                let wheres = if self.eat(&TokenKind::Where) {
                    self.parse_local_decls()?
                } else {
                    vec![]
                };

                let span = pat_start.to(self.tokens[self.pos.saturating_sub(1)].span);

                // Represent as pattern binding with $patbind
                let clause = Clause {
                    pats: vec![full_pat],
                    rhs: Rhs::Simple(expr, span),
                    wheres,
                    span,
                };

                return Ok(Decl::FunBind(FunBind {
                    doc: doc.clone(),
                    name: Ident::from_str("$patbind"),
                    clauses: vec![clause],
                    span,
                }));
            } else if self.is_infix_var_op_start() {
                // Variable operator: this is an infix function binding like `x --> y = ...`
                let first_pat = Pat::Var(name.clone(), start);
                let op_name = self.parse_infix_op()?;
                let second_pat = self.parse_pattern()?;
                pats.push(first_pat);
                pats.push(second_pat);
                (op_name, true)
            } else {
                // Prefix: parse atomic patterns for function arguments
                // (in Haskell, function LHS uses apat, not full patterns)
                while self.is_apat_start() {
                    pats.push(self.parse_atom_pattern()?);
                }
                (name, false)
            };

            // Parse RHS: either `= expr` or guarded: `| guard = expr`
            let rhs = self.parse_binding_rhs()?;

            // Parse optional where clause
            let wheres = if self.eat(&TokenKind::Where) {
                self.parse_local_decls()?
            } else {
                vec![]
            };

            let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);

            let clause = Clause {
                pats,
                rhs,
                wheres,
                span,
            };

            Ok(Decl::FunBind(FunBind {
                doc,
                name: actual_name,
                clauses: vec![clause],
                span,
            }))
        }
    }

    /// Parse a pattern binding: `(a, b) = expr` or `!pat = expr`
    /// These are bindings where the LHS is a pattern, not a function name.
    fn parse_pattern_binding(&mut self, start: Span) -> ParseResult<Decl> {
        // Parse the pattern
        let pat = self.parse_pattern()?;

        // Expect `=`
        self.expect(&TokenKind::Eq)?;

        // Parse the expression
        let expr = self.parse_expr()?;

        // Parse optional where clause
        let wheres = if self.eat(&TokenKind::Where) {
            self.parse_local_decls()?
        } else {
            vec![]
        };

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);

        // Represent pattern binding as FunBind with a generated name
        // and the pattern as the first clause's single pattern.
        // This is a common representation that allows uniform handling.
        let clause = Clause {
            pats: vec![pat],
            rhs: Rhs::Simple(expr, span),
            wheres,
            span,
        };

        // Use a special name to indicate this is a pattern binding
        // The generated name `$patbind` is not a valid Haskell identifier
        // so it won't conflict with user-defined names.
        Ok(Decl::FunBind(FunBind {
            doc: None, // Pattern bindings don't have documentation
            name: Ident::from_str("$patbind"),
            clauses: vec![clause],
            span,
        }))
    }

    /// Parse the right-hand side of a binding: either `= expr` or guarded `| guard = expr`.
    fn parse_binding_rhs(&mut self) -> ParseResult<Rhs> {
        let start = self.current_span();

        if self.check(&TokenKind::Pipe) {
            // Guarded RHS: `| guard = expr`
            let guards = self.parse_guarded_binding_rhss()?;
            let end_span = guards.last().map(|g| g.span).unwrap_or(start);
            Ok(Rhs::Guarded(guards, start.to(end_span)))
        } else {
            // Simple RHS: `= expr`
            self.expect(&TokenKind::Eq)?;
            let expr = self.parse_expr()?;
            let span = start.to(expr.span());
            Ok(Rhs::Simple(expr, span))
        }
    }

    /// Parse guarded right-hand sides for bindings: `| guard = expr | guard = expr ...`
    fn parse_guarded_binding_rhss(&mut self) -> ParseResult<Vec<GuardedRhs>> {
        let mut guarded_rhss = Vec::new();
        while self.eat(&TokenKind::Pipe) {
            let guard_start = self.current_span();
            let guards = self.parse_guards()?;
            self.expect(&TokenKind::Eq)?;
            let body = self.parse_expr()?;
            let span = guard_start.to(body.span());
            guarded_rhss.push(GuardedRhs { guards, body, span });
        }
        Ok(guarded_rhss)
    }

    /// Parse guards: `guard1, guard2, ...` where each guard is either:
    /// - A pattern guard: `pat <- expr`
    /// - A boolean guard: `expr`
    fn parse_guards(&mut self) -> ParseResult<Vec<Guard>> {
        let mut guards = Vec::new();
        guards.push(self.parse_single_guard()?);
        while self.eat(&TokenKind::Comma) {
            guards.push(self.parse_single_guard()?);
        }
        Ok(guards)
    }

    /// Parse a single guard.
    fn parse_single_guard(&mut self) -> ParseResult<Guard> {
        let start = self.current_span();

        // Try to parse as pattern guard first by looking ahead
        // Pattern guards have the form: pat <- expr
        // We need to check if there's a `<-` ahead

        // Save position for backtracking
        let saved_pos = self.pos;

        // Try parsing as pattern
        if let Ok(pat) = self.parse_pattern() {
            if self.eat(&TokenKind::LeftArrow) {
                // This is a pattern guard
                let expr = self.parse_expr()?;
                let span = start.to(expr.span());
                return Ok(Guard::Pattern(pat, expr, span));
            }
            // Not a pattern guard, backtrack
            self.pos = saved_pos;
        } else {
            // Couldn't parse as pattern, reset position
            self.pos = saved_pos;
        }

        // Parse as boolean guard
        let expr = self.parse_expr()?;
        let span = start.to(expr.span());
        Ok(Guard::Expr(expr, span))
    }

    /// Check if the current token starts an infix operator for bindings.
    /// This includes operators like `-->` and backtick-quoted identifiers like `` `elem` ``.
    #[allow(dead_code)]
    fn is_infix_op_start(&self) -> bool {
        match self.current_kind() {
            Some(TokenKind::Operator(_)) | Some(TokenKind::ConOperator(_)) => true,
            Some(TokenKind::Backtick) => true,
            _ => false,
        }
    }

    /// Check if the current token starts a constructor operator (starts with `:`)
    /// This indicates a pattern binding like `x :| xs = expr`
    fn is_infix_con_op_start(&self) -> bool {
        matches!(self.current_kind(), Some(TokenKind::ConOperator(_)))
    }

    /// Check if the current token starts a variable operator (doesn't start with `:`)
    /// This indicates an infix function binding like `x --> y = expr`
    fn is_infix_var_op_start(&self) -> bool {
        match self.current_kind() {
            Some(TokenKind::Operator(_)) => true,
            Some(TokenKind::Backtick) => true,
            _ => false,
        }
    }

    /// Parse an infix operator for bindings.
    /// Handles `-->` style operators and `` `foo` `` backtick-quoted identifiers.
    fn parse_infix_op(&mut self) -> ParseResult<Ident> {
        if self.eat(&TokenKind::Backtick) {
            // Backtick-quoted identifier: `foo`
            let ident = self.parse_ident()?;
            self.expect(&TokenKind::Backtick)?;
            Ok(ident)
        } else {
            // Regular operator
            let tok = self.current().ok_or(ParseError::UnexpectedEof {
                expected: "operator".to_string(),
            })?;

            match &tok.node.kind {
                TokenKind::Operator(sym) | TokenKind::ConOperator(sym) => {
                    let ident = Ident::new(*sym);
                    self.advance();
                    Ok(ident)
                }
                _ => Err(ParseError::Unexpected {
                    found: tok.node.kind.description().to_string(),
                    expected: "operator".to_string(),
                    span: tok.span,
                }),
            }
        }
    }

    /// Parse an identifier.
    fn parse_ident(&mut self) -> ParseResult<Ident> {
        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "identifier".to_string(),
        })?;

        match &tok.node.kind {
            TokenKind::Ident(sym) => {
                let ident = Ident::new(*sym);
                self.advance();
                Ok(ident)
            }
            _ => Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "identifier".to_string(),
                span: tok.span,
            }),
        }
    }

    /// Parse a variable name or parenthesized operator.
    /// Handles both `foo` and `(<+>)` style names.
    fn parse_var_or_op(&mut self) -> ParseResult<Ident> {
        if self.eat(&TokenKind::LParen) {
            // Parenthesized operator: (<+>)
            let tok = self.current().ok_or(ParseError::UnexpectedEof {
                expected: "operator".to_string(),
            })?;

            let ident = match &tok.node.kind {
                TokenKind::Operator(sym) | TokenKind::ConOperator(sym) => {
                    let ident = Ident::new(*sym);
                    self.advance();
                    ident
                }
                // Handle special punctuation used as operators
                TokenKind::Dot => {
                    let ident = Ident::new(Symbol::intern("."));
                    self.advance();
                    ident
                }
                TokenKind::Minus => {
                    let ident = Ident::new(Symbol::intern("-"));
                    self.advance();
                    ident
                }
                TokenKind::Backslash => {
                    // List difference operator \\
                    let ident = Ident::new(Symbol::intern("\\"));
                    self.advance();
                    ident
                }
                _ => {
                    return Err(ParseError::Unexpected {
                        found: tok.node.kind.description().to_string(),
                        expected: "operator".to_string(),
                        span: tok.span,
                    });
                }
            };

            self.expect(&TokenKind::RParen)?;
            Ok(ident)
        } else {
            // Regular identifier
            self.parse_ident()
        }
    }

    /// Parse a data declaration.
    #[allow(dead_code)]
    fn parse_data_decl(&mut self) -> ParseResult<Decl> {
        self.parse_data_decl_with_doc(None)
    }

    /// Parse a data declaration with optional documentation.
    fn parse_data_decl_with_doc(&mut self, doc: Option<DocComment>) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Data)?;

        // Check for data family / data instance
        if self.check_ident_str("family") {
            return self.parse_data_family_decl(start, doc);
        }
        if self.check(&TokenKind::Instance) {
            return self.parse_data_instance_decl(start, doc);
        }

        // Check for infix operator data declaration: `data a :+: b = ...`
        // Pattern: Ident ConOperator Ident (type_var operator type_var)
        let (name, params) = if let Some(TokenKind::Ident(_)) = self.current_kind() {
            // Save position to try infix pattern
            let saved_pos = self.pos;
            if let Some(TokenKind::Ident(lhs_sym)) = self.current_kind().cloned() {
                let lhs_span = self.current_span();
                self.advance();
                if let Some(TokenKind::ConOperator(op_sym)) = self.current_kind().cloned() {
                    self.advance();
                    if let Some(TokenKind::Ident(rhs_sym)) = self.current_kind().cloned() {
                        let rhs_span = self.current_span();
                        self.advance();
                        // Successfully parsed infix: `a :+: b`
                        let op_name = Ident::new(op_sym);
                        let lhs_var = TyVar { name: Ident::new(lhs_sym), span: lhs_span };
                        let rhs_var = TyVar { name: Ident::new(rhs_sym), span: rhs_span };
                        (op_name, vec![lhs_var, rhs_var])
                    } else {
                        // Not infix pattern, backtrack
                        self.pos = saved_pos;
                        let name = self.parse_conid()?;
                        let params = self.parse_ty_var_list()?;
                        (name, params)
                    }
                } else {
                    // Not infix pattern, backtrack
                    self.pos = saved_pos;
                    let name = self.parse_conid()?;
                    let params = self.parse_ty_var_list()?;
                    (name, params)
                }
            } else {
                self.pos = saved_pos;
                let name = self.parse_conid()?;
                let params = self.parse_ty_var_list()?;
                (name, params)
            }
        } else {
            let name = self.parse_conid_or_op()?;
            let params = self.parse_ty_var_list()?;
            (name, params)
        };

        // Three forms:
        // 1. H98: `data T a = Con1 a | Con2`
        // 2. GADT: `data T a where Con1 :: a -> T a; ...`
        // 3. EmptyDataDecls: `data T a`
        let (constrs, gadt_constrs, deriving) = if self.eat(&TokenKind::Eq) {
            let constrs = self.parse_constructors()?;
            let deriving = self.parse_deriving()?;
            (constrs, vec![], deriving)
        } else if self.check(&TokenKind::Where) {
            // GADT syntax
            let gadt_constrs = self.parse_gadt_constructors()?;
            let deriving = self.parse_deriving()?;
            (vec![], gadt_constrs, deriving)
        } else {
            let deriving = self.parse_deriving()?;
            (vec![], vec![], deriving)
        };

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);

        Ok(Decl::DataDecl(DataDecl {
            doc,
            name,
            params,
            constrs,
            gadt_constrs,
            deriving,
            span,
        }))
    }

    /// Parse GADT constructors in a `where` block.
    ///
    /// Each entry is `ConName :: Type` separated by layout or semicolons.
    fn parse_gadt_constructors(&mut self) -> ParseResult<Vec<GadtConDecl>> {
        self.expect(&TokenKind::Where)?;

        let mut constrs = Vec::new();

        if self.eat(&TokenKind::LBrace) {
            // Explicit braces
            if !self.check(&TokenKind::RBrace) {
                constrs.push(self.parse_gadt_con_decl()?);
                while self.eat(&TokenKind::Semi) {
                    if self.check(&TokenKind::RBrace) {
                        break;
                    }
                    constrs.push(self.parse_gadt_con_decl()?);
                }
            }
            self.expect(&TokenKind::RBrace)?;
        } else if self.eat(&TokenKind::VirtualLBrace) {
            // Layout-based declarations
            if !self.check(&TokenKind::VirtualRBrace) {
                constrs.push(self.parse_gadt_con_decl()?);
                while self.eat(&TokenKind::VirtualSemi) {
                    if self.check(&TokenKind::VirtualRBrace) {
                        break;
                    }
                    constrs.push(self.parse_gadt_con_decl()?);
                }
            }
            self.eat(&TokenKind::VirtualRBrace);
        }

        Ok(constrs)
    }

    /// Parse a single GADT constructor declaration: `ConName :: Type`.
    fn parse_gadt_con_decl(&mut self) -> ParseResult<GadtConDecl> {
        let start = self.current_span();
        let name = self.parse_conid()?;
        self.expect(&TokenKind::DoubleColon)?;
        let ty = self.parse_type()?;
        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(GadtConDecl {
            doc: None,
            name,
            ty,
            span,
        })
    }

    /// Parse a constructor identifier.
    fn parse_conid(&mut self) -> ParseResult<Ident> {
        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "constructor".to_string(),
        })?;

        match &tok.node.kind {
            TokenKind::ConId(sym) => {
                let ident = Ident::new(*sym);
                self.advance();
                Ok(ident)
            }
            _ => Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "constructor".to_string(),
                span: tok.span,
            }),
        }
    }

    /// Parse a constructor name: either a `ConId` or a parenthesized `ConOperator`.
    fn parse_conid_or_op(&mut self) -> ParseResult<Ident> {
        // Extract kind and span upfront to avoid borrow conflict
        let (kind, span) = match self.current() {
            Some(tok) => (tok.node.kind.clone(), tok.span),
            None => {
                return Err(ParseError::UnexpectedEof {
                    expected: "constructor".to_string(),
                });
            }
        };

        match &kind {
            TokenKind::ConId(sym) => {
                let ident = Ident::new(*sym);
                self.advance();
                Ok(ident)
            }
            TokenKind::LParen => {
                // Check for parenthesized constructor operator: (:+:)
                let saved = self.pos;
                self.advance(); // consume (
                if let Some(TokenKind::ConOperator(sym)) = self.current_kind().cloned() {
                    let ident = Ident::new(sym);
                    self.advance(); // consume operator
                    if self.eat(&TokenKind::RParen) {
                        return Ok(ident);
                    }
                }
                // Not a parenthesized operator, backtrack
                self.pos = saved;
                Err(ParseError::Unexpected {
                    found: kind.description().to_string(),
                    expected: "constructor".to_string(),
                    span,
                })
            }
            _ => Err(ParseError::Unexpected {
                found: kind.description().to_string(),
                expected: "constructor".to_string(),
                span,
            }),
        }
    }

    /// Parse type variable list.
    fn parse_ty_var_list(&mut self) -> ParseResult<Vec<TyVar>> {
        let mut vars = Vec::new();
        while let Some(tok) = self.current() {
            match &tok.node.kind {
                TokenKind::Ident(sym) => {
                    let span = tok.span;
                    let name = Ident::new(*sym);
                    self.advance();
                    vars.push(TyVar { name, span });
                }
                _ => break,
            }
        }
        Ok(vars)
    }

    /// Parse data constructors.
    fn parse_constructors(&mut self) -> ParseResult<Vec<ConDecl>> {
        let mut constrs = vec![self.parse_constructor()?];
        // Collect trailing doc comments after constructor (e.g., `-- ^ description`)
        if let Some(trailing_doc) = self.collect_doc_comments() {
            // Attach trailing doc to the last constructor
            if let Some(last) = constrs.last_mut() {
                if last.doc.is_none() {
                    last.doc = Some(trailing_doc);
                }
            }
        }
        while self.eat(&TokenKind::Pipe) {
            // Collect doc comments before next constructor
            let doc = self.collect_doc_comments();
            let mut constr = self.parse_constructor()?;
            // If there was a preceding doc, use it
            if constr.doc.is_none() && doc.is_some() {
                constr.doc = doc;
            }
            constrs.push(constr);
            // Collect trailing doc comments after constructor
            if let Some(trailing_doc) = self.collect_doc_comments() {
                if let Some(last) = constrs.last_mut() {
                    if last.doc.is_none() {
                        last.doc = Some(trailing_doc);
                    }
                }
            }
        }
        Ok(constrs)
    }

    /// Parse a single constructor.
    /// Supports existential quantification: `forall a. C a => Con a`
    fn parse_constructor(&mut self) -> ParseResult<ConDecl> {
        let start = self.current_span();

        // Parse existential quantification: forall a b. ...
        let mut existential_vars = Vec::new();
        if self.eat(&TokenKind::Forall) {
            while let Some(tok) = self.current() {
                match &tok.node.kind {
                    TokenKind::Ident(sym) => {
                        let name = Ident::new(*sym);
                        let span = tok.span;
                        self.advance();
                        existential_vars.push(TyVar { name, span });
                    }
                    TokenKind::LParen => {
                        // Kind annotation like (a :: Type) — parse the var name, skip the rest
                        self.advance(); // consume (
                        if let Some(TokenKind::Ident(sym)) = self.current_kind() {
                            let name = Ident::new(*sym);
                            let span = self.current_span();
                            existential_vars.push(TyVar { name, span });
                        }
                        // Skip until matching )
                        let mut depth = 1;
                        while depth > 0 && !self.at_eof() {
                            if self.check(&TokenKind::LParen) {
                                depth += 1;
                            } else if self.check(&TokenKind::RParen) {
                                depth -= 1;
                            }
                            self.advance();
                        }
                    }
                    TokenKind::Dot => {
                        self.advance();
                        break;
                    }
                    _ => break,
                }
            }
        }

        // Parse existential context: (C a, D b) => or C a =>
        let mut existential_context = Vec::new();
        if !existential_vars.is_empty() {
            let saved_pos = self.pos;
            if let Some(constraints) = self.try_parse_context()? {
                existential_context = constraints;
            } else {
                self.pos = saved_pos;
            }
        }

        let name = self.parse_conid_or_op()?;

        let fields = if self.check(&TokenKind::LBrace) {
            self.parse_record_fields()?
        } else {
            let types = self.parse_constr_types()?;
            ConFields::Positional(types)
        };

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(ConDecl {
            doc: None,
            name,
            fields,
            existential_vars,
            existential_context,
            span,
        })
    }

    /// Parse constructor argument types.
    fn parse_constr_types(&mut self) -> ParseResult<Vec<Type>> {
        let mut types = Vec::new();
        while self.is_atype_start() {
            types.push(self.parse_atype()?);
        }
        Ok(types)
    }

    /// Parse record fields.
    fn parse_record_fields(&mut self) -> ParseResult<ConFields> {
        self.expect(&TokenKind::LBrace)?;
        let mut fields = Vec::new();

        // Skip leading doc comments
        self.skip_doc_comments();

        if !self.check(&TokenKind::RBrace) {
            fields.push(self.parse_field_decl()?);
            // Skip trailing doc comments after field (e.g., `-- ^ description`)
            self.skip_doc_comments();
            while self.eat(&TokenKind::Comma) {
                // Skip doc comments before next field
                self.skip_doc_comments();
                if self.check(&TokenKind::RBrace) {
                    break;
                }
                fields.push(self.parse_field_decl()?);
                // Skip trailing doc comments after field
                self.skip_doc_comments();
            }
        }

        self.expect(&TokenKind::RBrace)?;
        Ok(ConFields::Record(fields))
    }

    /// Parse a field declaration.
    fn parse_field_decl(&mut self) -> ParseResult<FieldDecl> {
        let start = self.current_span();
        let name = self.parse_ident()?;
        self.expect(&TokenKind::DoubleColon)?;
        let ty = self.parse_type()?;
        let span = start.to(ty.span());
        Ok(FieldDecl {
            doc: None,
            name,
            ty,
            span,
        })
    }

    /// Parse all deriving clauses (there may be multiple in a row).
    fn parse_deriving(&mut self) -> ParseResult<Vec<DerivingClause>> {
        let mut all_clauses = Vec::new();

        // Parse all consecutive deriving clauses
        loop {
            // Skip any virtual layout tokens between deriving clauses
            self.skip_virtual_tokens();

            if self.check(&TokenKind::Deriving) {
                // Peek ahead: if next token after `deriving` is `instance`,
                // this is a standalone deriving declaration, not part of the
                // data type's deriving clause.
                if self.pos + 1 < self.tokens.len()
                    && self.tokens[self.pos + 1].node.kind == TokenKind::Instance
                {
                    break;
                }
                let clauses = self.parse_single_deriving()?;
                all_clauses.extend(clauses);
            } else {
                break;
            }
        }

        Ok(all_clauses)
    }

    /// Parse a single deriving clause.
    fn parse_single_deriving(&mut self) -> ParseResult<Vec<DerivingClause>> {
        if !self.eat(&TokenKind::Deriving) {
            return Ok(vec![]);
        }

        // Detect deriving strategy: stock, newtype, anyclass
        let strategy = if self.eat_ident_str("stock") {
            DerivingStrategy::Stock
        } else if self.eat(&TokenKind::Newtype) {
            DerivingStrategy::Newtype
        } else if self.eat_ident_str("anyclass") {
            DerivingStrategy::Anyclass
        } else {
            DerivingStrategy::Default
        };

        if self.eat(&TokenKind::LParen) {
            let mut classes = Vec::new();
            if !self.check(&TokenKind::RParen) {
                // Parse a type (which may be an application like `MonadState XState`)
                let ty = self.parse_type()?;
                // Extract the class name from the type
                classes.push(self.type_to_class_name(&ty));
                while self.eat(&TokenKind::Comma) {
                    if self.check(&TokenKind::RParen) {
                        break;
                    }
                    let ty = self.parse_type()?;
                    classes.push(self.type_to_class_name(&ty));
                }
            }
            self.expect(&TokenKind::RParen)?;

            // Handle `via` clause (DerivingVia extension)
            let strategy = if self.eat_ident_str("via") {
                let via_type = self.parse_type()?;
                DerivingStrategy::Via(via_type)
            } else {
                strategy
            };

            Ok(classes
                .into_iter()
                .map(|class| DerivingClause {
                    strategy: strategy.clone(),
                    class,
                })
                .collect())
        } else {
            // Single class without parens
            let ty = self.parse_type()?;
            let class = self.type_to_class_name(&ty);
            Ok(vec![DerivingClause { strategy, class }])
        }
    }

    /// Extract the class name from a type (e.g., `MonadState XState` -> `MonadState`)
    fn type_to_class_name(&self, ty: &Type) -> Ident {
        match ty {
            Type::Con(name, _) => name.clone(),
            Type::App(f, _, _) => self.type_to_class_name(f),
            Type::Paren(inner, _) => self.type_to_class_name(inner),
            _ => Ident::from_str("<unknown>"),
        }
    }

    /// Parse a standalone deriving declaration: `deriving instance Show Foo`
    fn parse_standalone_deriving(&mut self) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Deriving)?;
        self.expect(&TokenKind::Instance)?;

        // Parse class name
        let class = self.parse_conid()?;

        // Parse the type to derive for
        let ty = self.parse_type()?;

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);

        Ok(Decl::StandaloneDeriving(StandaloneDeriving {
            class,
            ty,
            span,
        }))
    }

    /// Parse a pattern synonym: `pattern Zero = Lit 0`
    fn parse_pattern_synonym(&mut self) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect_ident_str("pattern")?;

        // Parse the pattern synonym name (uppercase constructor-like)
        let name = self.parse_conid()?;

        // Parse zero or more variable arguments
        let mut args = Vec::new();
        while let Some(&TokenKind::Ident(sym)) = self.current_kind() {
            self.advance();
            args.push(Ident::new(sym));
        }

        // Check direction: `=` (bidirectional) or `<-` (unidirectional)
        let direction = if self.eat(&TokenKind::Eq) {
            PatSynDir::Bidirectional
        } else if self.eat(&TokenKind::LeftArrow) {
            PatSynDir::Unidirectional
        } else {
            return Err(ParseError::Unexpected {
                found: self
                    .current()
                    .map_or("end of input".to_string(), |t| {
                        t.node.kind.description().to_string()
                    }),
                expected: "'=' or '<-' in pattern synonym".to_string(),
                span: self.current_span(),
            });
        };

        // Parse the RHS pattern
        let pattern = self.parse_pattern()?;

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);

        Ok(Decl::PatternSynonym(PatternSynonymDecl {
            name,
            args,
            direction,
            pattern,
            span,
        }))
    }


    /// Parse a type alias.
    #[allow(dead_code)]
    fn parse_type_alias(&mut self) -> ParseResult<Decl> {
        self.parse_type_decl_with_doc(None)
    }

    /// Parse a type alias with optional documentation.
    /// Dispatch `type` declarations: type alias, type family, or type instance.
    fn parse_type_decl_with_doc(&mut self, doc: Option<DocComment>) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Type)?;

        if self.check_ident_str("family") {
            return self.parse_type_family_decl(start, doc);
        }
        if self.check(&TokenKind::Instance) {
            return self.parse_type_instance_decl(start, doc);
        }
        // Fall through to existing type alias parsing
        self.parse_type_alias_after_type(start, doc)
    }

    /// Parse a type alias after the `type` keyword has been consumed.
    fn parse_type_alias_after_type(
        &mut self,
        start: Span,
        doc: Option<DocComment>,
    ) -> ParseResult<Decl> {
        let name = self.parse_conid()?;
        let params = self.parse_ty_var_list()?;

        self.expect(&TokenKind::Eq)?;
        let ty = self.parse_type()?;

        let span = start.to(ty.span());
        Ok(Decl::TypeAlias(TypeAlias {
            doc,
            name,
            params,
            ty,
            span,
        }))
    }

    /// Parse a standalone type family declaration.
    ///
    /// Open:   `type family F a`
    /// Closed: `type family F a where { F Int = Bool; F a = () }`
    fn parse_type_family_decl(
        &mut self,
        start: Span,
        doc: Option<DocComment>,
    ) -> ParseResult<Decl> {
        self.expect_ident_str("family")?;

        let name = self.parse_conid()?;
        let params = self.parse_ty_var_list()?;

        // Optional kind signature: `:: * -> *`
        let kind = if self.eat(&TokenKind::DoubleColon) {
            Some(self.parse_kind()?)
        } else {
            None
        };

        // Check for `where` (closed family) vs end (open family)
        if self.eat(&TokenKind::Where) {
            let equations = self.parse_type_family_equations(&name)?;
            let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
            Ok(Decl::TypeFamilyDecl(TypeFamilyDecl {
                doc,
                name,
                params,
                kind,
                family_kind: TypeFamilyKind::Closed,
                equations,
                span,
            }))
        } else {
            let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
            Ok(Decl::TypeFamilyDecl(TypeFamilyDecl {
                doc,
                name,
                params,
                kind,
                family_kind: TypeFamilyKind::Open,
                equations: vec![],
                span,
            }))
        }
    }

    /// Parse equations within a closed type family `where` block.
    fn parse_type_family_equations(
        &mut self,
        _family_name: &Ident,
    ) -> ParseResult<Vec<TypeFamilyEqn>> {
        let mut equations = Vec::new();

        if self.eat(&TokenKind::LBrace) {
            // Explicit braces
            if !self.check(&TokenKind::RBrace) {
                equations.push(self.parse_type_family_equation()?);
                while self.eat(&TokenKind::Semi) {
                    if self.check(&TokenKind::RBrace) {
                        break;
                    }
                    equations.push(self.parse_type_family_equation()?);
                }
            }
            self.expect(&TokenKind::RBrace)?;
        } else if self.eat(&TokenKind::VirtualLBrace) {
            // Layout-based
            if !self.check(&TokenKind::VirtualRBrace) {
                equations.push(self.parse_type_family_equation()?);
                while self.eat(&TokenKind::VirtualSemi) {
                    if self.check(&TokenKind::VirtualRBrace) {
                        break;
                    }
                    equations.push(self.parse_type_family_equation()?);
                }
            }
            self.eat(&TokenKind::VirtualRBrace);
        } else {
            // Single equation on same line
            equations.push(self.parse_type_family_equation()?);
        }

        Ok(equations)
    }

    /// Parse a single type family equation: `F Int = Bool`
    fn parse_type_family_equation(&mut self) -> ParseResult<TypeFamilyEqn> {
        let start = self.current_span();

        // Parse family name (skip it, we already know it)
        let _name = self.parse_conid()?;

        // Parse type argument patterns
        let mut args = Vec::new();
        while !self.check(&TokenKind::Eq) && !self.at_eof() {
            args.push(self.parse_atype()?);
        }

        self.expect(&TokenKind::Eq)?;
        let rhs = self.parse_type()?;

        let span = start.to(rhs.span());
        Ok(TypeFamilyEqn { args, rhs, span })
    }

    /// Parse a standalone type instance: `type instance F Int = Bool`
    fn parse_type_instance_decl(
        &mut self,
        start: Span,
        doc: Option<DocComment>,
    ) -> ParseResult<Decl> {
        self.expect(&TokenKind::Instance)?;

        let name = self.parse_conid()?;

        // Parse type argument patterns
        let mut args = Vec::new();
        while !self.check(&TokenKind::Eq) && !self.at_eof() {
            args.push(self.parse_atype()?);
        }

        self.expect(&TokenKind::Eq)?;
        let rhs = self.parse_type()?;

        let span = start.to(rhs.span());
        Ok(Decl::TypeInstanceDecl(TypeInstanceDecl {
            doc,
            name,
            args,
            rhs,
            span,
        }))
    }

    /// Parse a standalone data family declaration: `data family F a`
    fn parse_data_family_decl(
        &mut self,
        start: Span,
        doc: Option<DocComment>,
    ) -> ParseResult<Decl> {
        self.expect_ident_str("family")?;

        let name = self.parse_conid()?;
        let params = self.parse_ty_var_list()?;

        // Optional kind signature: `:: * -> *`
        let kind = if self.eat(&TokenKind::DoubleColon) {
            Some(self.parse_kind()?)
        } else {
            None
        };

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Decl::DataFamilyDecl(DataFamilyDecl {
            doc,
            name,
            params,
            kind,
            span,
        }))
    }

    /// Parse a data family instance: `data instance F Int = Con1 Int | Con2`
    fn parse_data_instance_decl(
        &mut self,
        start: Span,
        doc: Option<DocComment>,
    ) -> ParseResult<Decl> {
        self.expect(&TokenKind::Instance)?;

        let family_name = self.parse_conid()?;

        // Parse type argument patterns (stop at = or where)
        let mut args = Vec::new();
        while !self.check(&TokenKind::Eq)
            && !self.check(&TokenKind::Where)
            && !self.at_eof()
        {
            args.push(self.parse_atype()?);
        }

        // Parse constructors (same as regular data decl)
        let (constrs, gadt_constrs, deriving) = if self.eat(&TokenKind::Eq) {
            let constrs = self.parse_constructors()?;
            let deriving = self.parse_deriving()?;
            (constrs, vec![], deriving)
        } else if self.check(&TokenKind::Where) {
            let gadt_constrs = self.parse_gadt_constructors()?;
            let deriving = self.parse_deriving()?;
            (vec![], gadt_constrs, deriving)
        } else {
            (vec![], vec![], vec![])
        };

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Decl::DataInstanceDecl(DataInstanceDecl {
            doc,
            family_name,
            args,
            constrs,
            gadt_constrs,
            deriving,
            span,
        }))
    }

    /// Parse a newtype declaration.
    #[allow(dead_code)]
    fn parse_newtype_decl(&mut self) -> ParseResult<Decl> {
        self.parse_newtype_decl_with_doc(None)
    }

    /// Parse a newtype declaration with optional documentation.
    fn parse_newtype_decl_with_doc(&mut self, doc: Option<DocComment>) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Newtype)?;

        let name = self.parse_conid()?;
        let params = self.parse_ty_var_list()?;

        self.expect(&TokenKind::Eq)?;
        let constr = self.parse_constructor()?;
        let deriving = self.parse_deriving()?;

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Decl::Newtype(NewtypeDecl {
            doc,
            name,
            params,
            constr,
            deriving,
            span,
        }))
    }

    /// Parse a class declaration.
    /// Handles multi-parameter type classes and functional dependencies.
    /// Examples:
    ///   - `class Eq a where ...`
    ///   - `class (Show a, Typeable a) => LayoutClass layout a where ...`
    ///   - `class MonadState s m | m -> s where ...`
    #[allow(dead_code)]
    fn parse_class_decl(&mut self) -> ParseResult<Decl> {
        self.parse_class_decl_with_doc(None)
    }

    /// Parse a class declaration with optional documentation.
    fn parse_class_decl_with_doc(&mut self, doc: Option<DocComment>) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Class)?;

        let context = self.parse_optional_context()?;
        let name = self.parse_conid()?;

        // Parse one or more type parameters
        let mut params = Vec::new();
        while self.check_ident() {
            let tok = self.current().unwrap();
            if let TokenKind::Ident(sym) = &tok.node.kind {
                let param_name = Ident::new(*sym);
                let span = tok.span;
                self.advance();
                params.push(TyVar {
                    name: param_name,
                    span,
                });
            }
        }

        // Parse optional functional dependencies: | a -> b, c -> d
        let fundeps = if self.eat(&TokenKind::Pipe) {
            self.parse_fundeps()?
        } else {
            vec![]
        };

        // The `where` clause is optional in Haskell
        // e.g., `class Foo a` with no methods
        let (methods, assoc_types) = if self.eat(&TokenKind::Where) {
            self.parse_class_body()?
        } else {
            (vec![], vec![])
        };

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Decl::ClassDecl(ClassDecl {
            doc,
            context,
            name,
            params,
            fundeps,
            methods,
            assoc_types,
            span,
        }))
    }

    /// Parse functional dependencies: `a -> b, c d -> e`
    fn parse_fundeps(&mut self) -> ParseResult<Vec<FunDep>> {
        let mut fundeps = Vec::new();

        loop {
            let start = self.current_span();

            // Parse 'from' variables
            let mut from = Vec::new();
            while self.check_ident() {
                if let Some(TokenKind::Ident(sym)) = self.current_kind() {
                    from.push(Ident::new(*sym));
                    self.advance();
                }
            }

            self.expect(&TokenKind::Arrow)?;

            // Parse 'to' variables
            let mut to = Vec::new();
            while self.check_ident() {
                if let Some(TokenKind::Ident(sym)) = self.current_kind() {
                    to.push(Ident::new(*sym));
                    self.advance();
                }
            }

            let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
            fundeps.push(FunDep { from, to, span });

            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }

        Ok(fundeps)
    }

    /// Parse optional class/instance context.
    /// Handles patterns like:
    ///   - `(Eq a, Show a) =>`
    ///   - `Eq a =>`
    ///   - `(a ~ Type) =>` (type equality constraint)
    fn parse_optional_context(&mut self) -> ParseResult<Vec<Constraint>> {
        // Save position in case we need to backtrack
        let saved_pos = self.pos;

        let start_span = self.current_span();

        let constraints = if self.check(&TokenKind::LParen) {
            // Could be tuple context `(C1 a, C2 a) =>` or type equality `(a ~ Type) =>`
            // or just a type in parens
            self.advance();

            // Try parsing as constraints
            let mut constraints = Vec::new();
            if !self.check(&TokenKind::RParen) {
                // Parse first constraint - could be:
                // - ConId Type... (normal constraint like Eq a)
                // - Type ~ Type (type equality)
                // - Type (just a type, might be part of equality)
                let constraint = self.parse_constraint_item(start_span)?;
                if let Some(c) = constraint {
                    constraints.push(c);
                }

                // Parse more constraints
                while self.eat(&TokenKind::Comma) {
                    let c_start = self.current_span();
                    let constraint = self.parse_constraint_item(c_start)?;
                    if let Some(c) = constraint {
                        constraints.push(c);
                    }
                }
            }

            if !self.eat(&TokenKind::RParen) {
                // Not a valid context tuple, backtrack
                self.pos = saved_pos;
                return Ok(vec![]);
            }
            constraints
        } else if let Some(TokenKind::ConId(_)) = self.current_kind() {
            // Could be single constraint `Eq a =>`
            let class = self.parse_conid()?;
            // Parse type arguments until we see `=>` or run out
            let mut args = Vec::new();
            while !self.check(&TokenKind::FatArrow) && !self.at_eof() {
                // Don't parse operators as part of the constraint args
                if let Some(TokenKind::Operator(_)) = self.current_kind() {
                    break;
                }
                if let Ok(arg) = self.parse_atype() {
                    args.push(arg);
                } else {
                    break;
                }
            }
            let span = start_span.to(self.tokens[self.pos.saturating_sub(1)].span);
            vec![Constraint { class, args, span }]
        } else {
            return Ok(vec![]);
        };

        // Check for `=>`
        if self.eat(&TokenKind::FatArrow) {
            Ok(constraints)
        } else {
            // No `=>` found, backtrack
            self.pos = saved_pos;
            Ok(vec![])
        }
    }

    /// Parse a single constraint item within a context.
    /// Handles normal constraints (C a) and type equality (a ~ b).
    fn parse_constraint_item(&mut self, start_span: Span) -> ParseResult<Option<Constraint>> {
        // Try to parse as a normal constraint first: ConId Type...
        if let Some(TokenKind::ConId(_)) = self.current_kind() {
            let class = self.parse_conid()?;
            // Parse type arguments
            let mut args = Vec::new();
            while !self.check(&TokenKind::Comma)
                && !self.check(&TokenKind::RParen)
                && !self.at_eof()
            {
                // Don't parse `~` as part of args (it's a type operator)
                if let Some(TokenKind::Operator(sym)) = self.current_kind() {
                    if sym.as_str() == "~" {
                        break;
                    }
                }
                if let Ok(arg) = self.parse_atype() {
                    args.push(arg);
                } else {
                    break;
                }
            }
            let span = start_span.to(self.tokens[self.pos.saturating_sub(1)].span);
            return Ok(Some(Constraint { class, args, span }));
        }

        // Try type equality constraint: a ~ Type
        // Skip the LHS type (use atype to not consume `~` as part of the type)
        if self.check_ident() || self.check(&TokenKind::LParen) {
            let saved = self.pos;
            // Try to parse just the atomic type on the left of ~
            if self.parse_atype().is_ok() {
                // Check for `~` token (type equality operator)
                if self.eat(&TokenKind::Tilde) {
                    // Parse RHS type (full type since nothing follows)
                    if self.parse_type().is_ok() {
                        // Successfully parsed type equality, but we'll represent it
                        // as an opaque constraint for now (class name (~))
                        let span = start_span.to(self.tokens[self.pos.saturating_sub(1)].span);
                        return Ok(Some(Constraint {
                            class: Ident::from_str("~"),
                            args: vec![], // We lose the actual types, but that's OK for now
                            span,
                        }));
                    }
                }
            }
            // Backtrack if it wasn't a type equality
            self.pos = saved;
        }

        Ok(None)
    }

    /// Parse class body (methods and associated type declarations).
    /// Returns (methods, assoc_types).
    fn parse_class_body(&mut self) -> ParseResult<(Vec<Decl>, Vec<AssocType>)> {
        let mut methods = Vec::new();
        let mut assoc_types = Vec::new();

        if self.eat(&TokenKind::LBrace) {
            // Explicit braces
            if !self.check(&TokenKind::RBrace) {
                self.parse_class_body_item(&mut methods, &mut assoc_types)?;
                while self.eat(&TokenKind::Semi) {
                    if self.check(&TokenKind::RBrace) {
                        break;
                    }
                    self.parse_class_body_item(&mut methods, &mut assoc_types)?;
                }
            }
            self.expect(&TokenKind::RBrace)?;
        } else if self.eat(&TokenKind::VirtualLBrace) {
            // Layout-based declarations
            if !self.check(&TokenKind::VirtualRBrace) {
                self.parse_class_body_item(&mut methods, &mut assoc_types)?;
                while self.eat(&TokenKind::VirtualSemi) {
                    if self.check(&TokenKind::VirtualRBrace) {
                        break;
                    }
                    self.parse_class_body_item(&mut methods, &mut assoc_types)?;
                }
            }
            self.eat(&TokenKind::VirtualRBrace);
        }

        // Merge multi-clause functions
        methods = self.merge_function_clauses(methods);

        Ok((methods, assoc_types))
    }

    /// Parse a single class body item (either a method or an associated type).
    fn parse_class_body_item(
        &mut self,
        methods: &mut Vec<Decl>,
        assoc_types: &mut Vec<AssocType>,
    ) -> ParseResult<()> {
        // Skip doc comments
        self.skip_doc_comments();
        self.eat(&TokenKind::VirtualSemi);

        // Check for associated type: `type Name params`
        if self.check(&TokenKind::Type) {
            let assoc_type = self.parse_assoc_type_decl()?;
            assoc_types.push(assoc_type);
        } else if self.check(&TokenKind::Default) {
            // DefaultSignatures: `default methodName :: ConstrainedType`
            // Parse and discard — BHC handles default method bodies via FunBind.
            // The `default` keyword before a type sig is purely informational.
            self.advance(); // eat 'default'
            let _decl = self.parse_value_decl()?;
            // Discard: it's just a more-constrained type sig for the default
        } else {
            let decl = self.parse_value_decl()?;
            methods.push(decl);
        }
        Ok(())
    }

    /// Parse an associated type declaration within a class.
    /// Example: `type Elem c` or `type Elem c :: * -> *` or `type Elem c = [c]`
    fn parse_assoc_type_decl(&mut self) -> ParseResult<AssocType> {
        let start = self.current_span();
        self.expect(&TokenKind::Type)?;

        let name = self.parse_conid()?;

        // Parse type parameters
        let mut params = Vec::new();
        while self.check_ident() {
            let tok = self.current().unwrap();
            if let TokenKind::Ident(sym) = &tok.node.kind {
                let param_name = Ident::new(*sym);
                let span = tok.span;
                self.advance();
                params.push(TyVar {
                    name: param_name,
                    span,
                });
            }
        }

        // Parse optional kind signature: `:: * -> *`
        let kind = if self.eat(&TokenKind::DoubleColon) {
            Some(self.parse_kind()?)
        } else {
            None
        };

        // Parse optional default type: `= Type`
        let default = if self.eat(&TokenKind::Eq) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(AssocType {
            name,
            params,
            kind,
            default,
            span,
        })
    }

    /// Parse a kind (for kind signatures).
    fn parse_kind(&mut self) -> ParseResult<Kind> {
        let kind = self.parse_kind_atom()?;

        // Check for arrow
        if self.eat(&TokenKind::Arrow) {
            let right = self.parse_kind()?;
            Ok(Kind::Arrow(Box::new(kind), Box::new(right)))
        } else {
            Ok(kind)
        }
    }

    /// Parse an atomic kind.
    fn parse_kind_atom(&mut self) -> ParseResult<Kind> {
        // Check for `*` (Star token) or `Type`
        if self.check(&TokenKind::Star) {
            self.advance();
            return Ok(Kind::Star);
        }
        if let Some(TokenKind::Operator(sym)) = self.current_kind() {
            if sym.as_str() == "*" {
                self.advance();
                return Ok(Kind::Star);
            }
        }

        if let Some(TokenKind::ConId(sym)) = self.current_kind() {
            if sym.as_str() == "Type" {
                self.advance();
                return Ok(Kind::Star);
            }
            // Named kind variable
            let name = Ident::new(*sym);
            self.advance();
            return Ok(Kind::Var(name));
        }

        // Parenthesized kind
        if self.eat(&TokenKind::LParen) {
            let kind = self.parse_kind()?;
            self.expect(&TokenKind::RParen)?;
            return Ok(kind);
        }

        Err(ParseError::Unexpected {
            found: self
                .current()
                .map(|t| t.node.kind.description().to_string())
                .unwrap_or("end of file".to_string()),
            expected: "kind (*, Type, or kind variable)".to_string(),
            span: self.current_span(),
        })
    }

    /// Parse class methods (legacy, for compatibility).
    #[allow(dead_code)]
    fn parse_class_methods(&mut self) -> ParseResult<Vec<Decl>> {
        // Simplified - just parse as local decls
        self.parse_local_decls()
    }

    /// Parse an instance declaration.
    #[allow(dead_code)]
    fn parse_instance_decl(&mut self) -> ParseResult<Decl> {
        self.parse_instance_decl_with_doc(None)
    }

    /// Parse an instance declaration with optional documentation.
    fn parse_instance_decl_with_doc(&mut self, doc: Option<DocComment>) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Instance)?;

        let context = self.parse_optional_context()?;
        let class = self.parse_conid()?;
        let ty = self.parse_type()?;

        // The `where` clause is optional in Haskell
        // e.g., `instance Message Resize` with no methods
        let (methods, assoc_type_defs) = if self.eat(&TokenKind::Where) {
            self.parse_instance_body()?
        } else {
            (vec![], vec![])
        };

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Decl::InstanceDecl(InstanceDecl {
            doc,
            context,
            class,
            ty,
            methods,
            assoc_type_defs,
            span,
        }))
    }

    /// Parse instance body (methods and associated type definitions).
    /// Returns (methods, assoc_type_defs).
    fn parse_instance_body(&mut self) -> ParseResult<(Vec<Decl>, Vec<AssocTypeDef>)> {
        let mut methods = Vec::new();
        let mut assoc_type_defs = Vec::new();

        if self.eat(&TokenKind::LBrace) {
            // Explicit braces
            if !self.check(&TokenKind::RBrace) {
                self.parse_instance_body_item(&mut methods, &mut assoc_type_defs)?;
                while self.eat(&TokenKind::Semi) {
                    if self.check(&TokenKind::RBrace) {
                        break;
                    }
                    self.parse_instance_body_item(&mut methods, &mut assoc_type_defs)?;
                }
            }
            self.expect(&TokenKind::RBrace)?;
        } else if self.eat(&TokenKind::VirtualLBrace) {
            // Layout-based declarations
            if !self.check(&TokenKind::VirtualRBrace) {
                self.parse_instance_body_item(&mut methods, &mut assoc_type_defs)?;
                while self.eat(&TokenKind::VirtualSemi) {
                    if self.check(&TokenKind::VirtualRBrace) {
                        break;
                    }
                    self.parse_instance_body_item(&mut methods, &mut assoc_type_defs)?;
                }
            }
            self.eat(&TokenKind::VirtualRBrace);
        }

        // Merge multi-clause functions
        methods = self.merge_function_clauses(methods);

        Ok((methods, assoc_type_defs))
    }

    /// Parse a single instance body item (either a method or an associated type definition).
    fn parse_instance_body_item(
        &mut self,
        methods: &mut Vec<Decl>,
        assoc_type_defs: &mut Vec<AssocTypeDef>,
    ) -> ParseResult<()> {
        // Skip doc comments
        self.skip_doc_comments();
        self.eat(&TokenKind::VirtualSemi);

        // Check for associated type definition: `type Name args = rhs`
        if self.check(&TokenKind::Type) {
            let assoc_type_def = self.parse_assoc_type_def()?;
            assoc_type_defs.push(assoc_type_def);
        } else {
            let decl = self.parse_value_decl()?;
            methods.push(decl);
        }
        Ok(())
    }

    /// Parse an associated type definition within an instance.
    /// Example: `type Elem [a] = a`
    fn parse_assoc_type_def(&mut self) -> ParseResult<AssocTypeDef> {
        let start = self.current_span();
        self.expect(&TokenKind::Type)?;

        let name = self.parse_conid()?;

        // Parse type arguments (the patterns for the associated type)
        let mut args = Vec::new();
        while !self.check(&TokenKind::Eq) && !self.at_eof() {
            // Stop if we see operators or other delimiters
            if let Some(
                TokenKind::Semi
                | TokenKind::VirtualSemi
                | TokenKind::RBrace
                | TokenKind::VirtualRBrace,
            ) = self.current_kind()
            {
                break;
            }
            let arg = self.parse_atype()?;
            args.push(arg);
        }

        self.expect(&TokenKind::Eq)?;

        let rhs = self.parse_type()?;

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(AssocTypeDef {
            name,
            args,
            rhs,
            span,
        })
    }

    /// Parse a foreign declaration.
    #[allow(dead_code)]
    fn parse_foreign_decl(&mut self) -> ParseResult<Decl> {
        self.parse_foreign_decl_with_doc(None)
    }

    /// Parse a foreign declaration with optional documentation.
    ///
    /// Grammar:
    /// ```text
    /// foreign_decl ::= 'foreign' 'import' calling_conv [safety] [c_name] hs_name '::' type
    ///                | 'foreign' 'export' calling_conv [c_name] hs_name '::' type
    /// calling_conv ::= 'ccall' | 'capi' | 'stdcall' | 'javascript'
    /// safety       ::= 'safe' | 'unsafe' | 'interruptible'
    /// ```
    fn parse_foreign_decl_with_doc(&mut self, doc: Option<DocComment>) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Foreign)?;

        // Parse 'import' or 'export'
        // 'import' is a keyword token, 'export' is an identifier
        let kind = if self.check(&TokenKind::Import) {
            self.advance();
            ForeignKind::Import
        } else if let Some(tok) = self.current() {
            if let TokenKind::Ident(sym) = &tok.node.kind {
                if sym.as_str() == "export" {
                    self.advance();
                    ForeignKind::Export
                } else {
                    return Err(ParseError::Unexpected {
                        found: tok.node.kind.description().to_string(),
                        expected: "`import` or `export`".to_string(),
                        span: tok.span,
                    });
                }
            } else {
                return Err(ParseError::Unexpected {
                    found: tok.node.kind.description().to_string(),
                    expected: "`import` or `export`".to_string(),
                    span: tok.span,
                });
            }
        } else {
            return Err(ParseError::UnexpectedEof {
                expected: "`import` or `export`".to_string(),
            });
        };

        // Parse calling convention: ccall, capi, stdcall, javascript
        let convention = if let Some(tok) = self.current() {
            if let TokenKind::Ident(sym) = &tok.node.kind {
                match sym.as_str() {
                    "ccall" | "capi" | "stdcall" | "javascript" => {
                        let conv = *sym;
                        self.advance();
                        conv
                    }
                    _ => {
                        // Default to ccall if no convention specified
                        Symbol::intern("ccall")
                    }
                }
            } else {
                Symbol::intern("ccall")
            }
        } else {
            return Err(ParseError::UnexpectedEof {
                expected: "calling convention".to_string(),
            });
        };

        // Parse optional safety: safe, unsafe, interruptible
        let mut safety = ForeignSafety::Safe;
        if let Some(tok) = self.current() {
            if let TokenKind::Ident(sym) = &tok.node.kind {
                match sym.as_str() {
                    "safe" => {
                        safety = ForeignSafety::Safe;
                        self.advance();
                    }
                    "unsafe" => {
                        safety = ForeignSafety::Unsafe;
                        self.advance();
                    }
                    "interruptible" => {
                        safety = ForeignSafety::Interruptible;
                        self.advance();
                    }
                    _ => {}
                }
            }
        }

        // Now we need to parse: [c_name_string] haskell_name :: type
        // If the next token is a string literal, it's the C name.
        // Otherwise, the Haskell name doubles as the C name.
        let mut external_name: Option<String> = None;
        if let Some(tok) = self.current() {
            if let TokenKind::StringLit(s) = &tok.node.kind {
                external_name = Some(s.clone());
                self.advance();
            }
        }

        // Parse the Haskell name
        let name = self.parse_ident()?;

        // Parse '::' and type
        self.expect(&TokenKind::DoubleColon)?;
        let ty = self.parse_type()?;

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Decl::Foreign(ForeignDecl {
            doc,
            kind,
            convention,
            safety,
            external_name,
            name,
            ty,
            span,
        }))
    }

    /// Parse a fixity declaration.
    fn parse_fixity_decl(&mut self) -> ParseResult<Decl> {
        let start = self.current_span();

        let fixity = match self.current().map(|t| &t.node.kind) {
            Some(TokenKind::Infix) => {
                self.advance();
                Fixity::None
            }
            Some(TokenKind::Infixl) => {
                self.advance();
                Fixity::Left
            }
            Some(TokenKind::Infixr) => {
                self.advance();
                Fixity::Right
            }
            _ => unreachable!(),
        };

        // Parse precedence
        let prec = if let Some(tok) = self.current() {
            if let TokenKind::IntLit(ref lit) = &tok.node.kind {
                let p: u8 = lit.parse().map(|v| v as u8).unwrap_or(9);
                self.advance();
                p.min(9)
            } else {
                9
            }
        } else {
            9
        };

        // Parse operators
        let mut ops = Vec::new();
        loop {
            let tok = self.current().ok_or(ParseError::UnexpectedEof {
                expected: "operator".to_string(),
            })?;

            match &tok.node.kind {
                TokenKind::Operator(sym) => {
                    ops.push(Ident::new(*sym));
                    self.advance();
                }
                TokenKind::Backtick => {
                    self.advance();
                    ops.push(self.parse_ident()?);
                    self.expect(&TokenKind::Backtick)?;
                }
                _ => {
                    if ops.is_empty() {
                        return Err(ParseError::Unexpected {
                            found: tok.node.kind.description().to_string(),
                            expected: "operator".to_string(),
                            span: tok.span,
                        });
                    }
                    break;
                }
            }

            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Decl::Fixity(FixityDecl {
            fixity,
            prec,
            ops,
            span,
        }))
    }

    /// Try to recover to the next declaration after an error.
    fn recover_to_next_decl(&mut self) {
        while !self.at_eof() {
            if let Some(tok) = self.current() {
                match &tok.node.kind {
                    TokenKind::Data
                    | TokenKind::Type
                    | TokenKind::Newtype
                    | TokenKind::Class
                    | TokenKind::Instance
                    | TokenKind::Foreign
                    | TokenKind::Infix
                    | TokenKind::Infixl
                    | TokenKind::Infixr => return,
                    TokenKind::Ident(_) => {
                        // Could be start of new binding
                        return;
                    }
                    _ => {
                        self.advance();
                    }
                }
            } else {
                break;
            }
        }
    }
}
