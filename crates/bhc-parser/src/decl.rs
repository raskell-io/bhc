//! Declaration parsing.

use bhc_ast::*;
use bhc_intern::{Ident, Symbol};
use bhc_lexer::TokenKind;
use bhc_span::Span;

use crate::{ParseResult, Parser, ParseError};

impl<'src> Parser<'src> {
    /// Parse a complete module.
    pub fn parse_module(&mut self) -> ParseResult<Module> {
        let start = self.current_span();

        // Parse pragmas at the start of the module
        let pragmas = self.parse_pragmas();

        // Skip doc comments that may appear before the module declaration (Haddock)
        self.skip_doc_comments();

        // Optional module header
        let (name, exports) = if self.eat(&TokenKind::Module) {
            let name = self.parse_module_name()?;
            let exports = if self.check(&TokenKind::LParen) {
                Some(self.parse_export_list()?)
            } else {
                None
            };
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
            // Skip any virtual tokens and explicit semicolons between declarations
            self.skip_virtual_tokens();
            while self.eat(&TokenKind::Semi) {
                self.skip_virtual_tokens();
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
    fn parse_pragmas(&mut self) -> Vec<Pragma> {
        let mut pragmas = Vec::new();

        while let Some(tok) = self.current() {
            if let TokenKind::Pragma(content) = &tok.node.kind {
                let span = tok.span;
                let content = content.clone();
                self.advance();

                if let Some(pragma) = self.parse_pragma_content(&content, span) {
                    pragmas.push(pragma);
                }
            } else {
                break;
            }
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
            "OPTIONS_GHC" | "OPTIONS" => {
                PragmaKind::OptionsGhc(args.to_string())
            }
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
            "WARNING" => {
                PragmaKind::Warning(None, args.to_string())
            }
            "SPECIALIZE" | "SPECIALISE" => {
                // For now, store as Other since we'd need to parse the type signature
                PragmaKind::Other(content.to_string())
            }
            _ => PragmaKind::Other(content.to_string()),
        };

        Some(Pragma { kind, span })
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

        if !self.check(&TokenKind::RParen) {
            exports.push(self.parse_export()?);
            while self.eat(&TokenKind::Comma) {
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
            // Handle operators in parentheses: module Foo ((+), (++)) where
            TokenKind::LParen => {
                self.advance(); // consume (

                // Expect an operator
                let op_tok = self.current().ok_or(ParseError::UnexpectedEof {
                    expected: "operator".to_string(),
                })?;

                let ident = match &op_tok.node.kind {
                    TokenKind::Operator(sym) => Ident::new(*sym),
                    TokenKind::ConOperator(sym) => Ident::new(*sym),
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
    fn parse_import(&mut self) -> ParseResult<ImportDecl> {
        let start = self.current_span();
        self.expect(&TokenKind::Import)?;

        let qualified = self.eat(&TokenKind::Qualified);
        let module = self.parse_module_name()?;

        // Check for "as Alias"
        let alias = if self.eat(&TokenKind::As) {
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
        // Skip Haddock documentation comments before declarations
        self.skip_doc_comments();

        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "declaration".to_string(),
        })?;

        match &tok.node.kind {
            TokenKind::Data => self.parse_data_decl(),
            TokenKind::Type => self.parse_type_alias(),
            TokenKind::Newtype => self.parse_newtype_decl(),
            TokenKind::Class => self.parse_class_decl(),
            TokenKind::Instance => self.parse_instance_decl(),
            TokenKind::Foreign => self.parse_foreign_decl(),
            TokenKind::Infix | TokenKind::Infixl | TokenKind::Infixr => self.parse_fixity_decl(),
            TokenKind::Ident(_) => {
                // Could be type signature or binding
                self.parse_value_decl()
            }
            TokenKind::LParen => {
                // Operator type signature or binding: (<+>) :: ... or (<+>) = ...
                self.parse_value_decl()
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
            if !self.check(&TokenKind::VirtualRBrace) {
                decls.push(self.parse_value_decl()?);
                while self.eat(&TokenKind::VirtualSemi) {
                    if self.check(&TokenKind::VirtualRBrace) {
                        break;
                    }
                    decls.push(self.parse_value_decl()?);
                }
            }
            self.expect(&TokenKind::VirtualRBrace)?;
        } else {
            // Single declaration (no braces)
            decls.push(self.parse_value_decl()?);
        }

        // Merge multi-clause functions in local declarations too
        Ok(self.merge_function_clauses(decls))
    }

    /// Parse a value declaration (type signature or binding).
    fn parse_value_decl(&mut self) -> ParseResult<Decl> {
        // Skip Haddock documentation comments before value declarations
        self.skip_doc_comments();

        let start = self.current_span();
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
                return Ok(Decl::TypeSig(TypeSig { names, ty, span }));
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
            let (actual_name, _is_infix) = if self.is_infix_op_start() {
                // This is infix: the name we parsed is actually the first pattern
                let first_pat = Pat::Var(name.clone(), start);
                let op_name = self.parse_infix_op()?;
                let second_pat = self.parse_pattern()?;
                pats.push(first_pat);
                pats.push(second_pat);
                (op_name, true)
            } else {
                // Prefix: parse patterns normally
                while self.is_pattern_start() {
                    pats.push(self.parse_pattern()?);
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
                name: actual_name,
                clauses: vec![clause],
                span,
            }))
        }
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
    fn is_infix_op_start(&self) -> bool {
        match self.current_kind() {
            Some(TokenKind::Operator(_)) | Some(TokenKind::ConOperator(_)) => true,
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
    fn parse_data_decl(&mut self) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Data)?;

        let name = self.parse_conid()?;
        let params = self.parse_ty_var_list()?;

        self.expect(&TokenKind::Eq)?;

        let constrs = self.parse_constructors()?;
        let deriving = self.parse_deriving()?;

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);

        Ok(Decl::DataDecl(DataDecl {
            name,
            params,
            constrs,
            deriving,
            span,
        }))
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
        // Skip doc comments after constructor (e.g., `-- ^ description`)
        self.skip_doc_comments();
        while self.eat(&TokenKind::Pipe) {
            // Skip doc comments before next constructor
            self.skip_doc_comments();
            constrs.push(self.parse_constructor()?);
            // Skip doc comments after constructor
            self.skip_doc_comments();
        }
        Ok(constrs)
    }

    /// Parse a single constructor.
    fn parse_constructor(&mut self) -> ParseResult<ConDecl> {
        let start = self.current_span();
        let name = self.parse_conid()?;

        let fields = if self.check(&TokenKind::LBrace) {
            self.parse_record_fields()?
        } else {
            let types = self.parse_constr_types()?;
            ConFields::Positional(types)
        };

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(ConDecl { name, fields, span })
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
        Ok(FieldDecl { name, ty, span })
    }

    /// Parse deriving clause.
    fn parse_deriving(&mut self) -> ParseResult<Vec<Ident>> {
        if !self.eat(&TokenKind::Deriving) {
            return Ok(vec![]);
        }

        if self.eat(&TokenKind::LParen) {
            let mut classes = Vec::new();
            if !self.check(&TokenKind::RParen) {
                classes.push(self.parse_conid()?);
                while self.eat(&TokenKind::Comma) {
                    if self.check(&TokenKind::RParen) {
                        break;
                    }
                    classes.push(self.parse_conid()?);
                }
            }
            self.expect(&TokenKind::RParen)?;
            Ok(classes)
        } else {
            Ok(vec![self.parse_conid()?])
        }
    }

    /// Parse a type alias.
    fn parse_type_alias(&mut self) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Type)?;

        let name = self.parse_conid()?;
        let params = self.parse_ty_var_list()?;

        self.expect(&TokenKind::Eq)?;
        let ty = self.parse_type()?;

        let span = start.to(ty.span());
        Ok(Decl::TypeAlias(TypeAlias {
            name,
            params,
            ty,
            span,
        }))
    }

    /// Parse a newtype declaration.
    fn parse_newtype_decl(&mut self) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Newtype)?;

        let name = self.parse_conid()?;
        let params = self.parse_ty_var_list()?;

        self.expect(&TokenKind::Eq)?;
        let constr = self.parse_constructor()?;
        let deriving = self.parse_deriving()?;

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Decl::Newtype(NewtypeDecl {
            name,
            params,
            constr,
            deriving,
            span,
        }))
    }

    /// Parse a class declaration.
    fn parse_class_decl(&mut self) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Class)?;

        let context = self.parse_optional_context()?;
        let name = self.parse_conid()?;
        let param = {
            let tok = self.current().ok_or(ParseError::UnexpectedEof {
                expected: "type variable".to_string(),
            })?;
            match &tok.node.kind {
                TokenKind::Ident(sym) => {
                    let name = Ident::new(*sym);
                    let span = tok.span;
                    self.advance();
                    TyVar { name, span }
                }
                _ => {
                    return Err(ParseError::Unexpected {
                        found: tok.node.kind.description().to_string(),
                        expected: "type variable".to_string(),
                        span: tok.span,
                    });
                }
            }
        };

        self.expect(&TokenKind::Where)?;
        let methods = self.parse_class_methods()?;

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Decl::ClassDecl(ClassDecl {
            context,
            name,
            param,
            methods,
            span,
        }))
    }

    /// Parse optional class/instance context.
    fn parse_optional_context(&mut self) -> ParseResult<Vec<Constraint>> {
        // Simplified: look for `... =>`
        // For now, just return empty
        Ok(vec![])
    }

    /// Parse class methods.
    fn parse_class_methods(&mut self) -> ParseResult<Vec<Decl>> {
        // Simplified
        self.parse_local_decls()
    }

    /// Parse an instance declaration.
    fn parse_instance_decl(&mut self) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Instance)?;

        let context = self.parse_optional_context()?;
        let class = self.parse_conid()?;
        let ty = self.parse_type()?;

        self.expect(&TokenKind::Where)?;
        let methods = self.parse_class_methods()?;

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Decl::InstanceDecl(InstanceDecl {
            context,
            class,
            ty,
            methods,
            span,
        }))
    }

    /// Parse a foreign declaration.
    fn parse_foreign_decl(&mut self) -> ParseResult<Decl> {
        let start = self.current_span();
        self.expect(&TokenKind::Foreign)?;

        // For now, just skip to end of declaration
        // This is a placeholder implementation
        while !self.at_eof() {
            if self.check(&TokenKind::Semi) || self.check(&TokenKind::RBrace) {
                break;
            }
            self.advance();
        }

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Err(ParseError::Unexpected {
            found: "foreign declaration".to_string(),
            expected: "foreign declarations not yet supported".to_string(),
            span,
        })
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
