//! Expression parsing.

use bhc_ast::{
    Alt, ArithSeq, Expr, FieldBind, Guard, GuardedRhs, Lit, ModuleName, Pat, Rhs, Stmt,
};
use bhc_intern::Ident;
use bhc_lexer::TokenKind;
use bhc_span::Span;

use crate::{ParseResult, Parser, ParseError};

impl<'src> Parser<'src> {
    /// Parse an expression.
    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        self.parse_infix_expr(0)
    }

    /// Parse an infix expression with precedence climbing.
    fn parse_infix_expr(&mut self, min_prec: u8) -> ParseResult<Expr> {
        let mut lhs = self.parse_prefix_expr()?;

        while let Some(tok) = self.current() {
            let (op, prec, assoc) = match &tok.node.kind {
                TokenKind::Operator(sym) => {
                    let (prec, assoc) = self.get_operator_info(sym.as_str());
                    if prec < min_prec {
                        break;
                    }
                    (Ident::new(*sym), prec, assoc)
                }
                // Handle * (Star token is lexed separately for kind syntax)
                TokenKind::Star => {
                    let (prec, assoc) = self.get_operator_info("*");
                    if prec < min_prec {
                        break;
                    }
                    (Ident::from_str("*"), prec, assoc)
                }
                // Handle - (Minus token is lexed separately for prefix negation)
                TokenKind::Minus => {
                    let (prec, assoc) = self.get_operator_info("-");
                    if prec < min_prec {
                        break;
                    }
                    (Ident::from_str("-"), prec, assoc)
                }
                TokenKind::Backtick => {
                    // Infix function application: `x `mod` y`
                    let _start = tok.span;
                    self.advance(); // `
                    let Some(func_tok) = self.current() else {
                        return Err(ParseError::UnexpectedEof {
                            expected: "identifier".to_string(),
                        });
                    };
                    let func = match &func_tok.node.kind {
                        TokenKind::Ident(sym) => Ident::new(*sym),
                        TokenKind::ConId(sym) => Ident::new(*sym),
                        _ => {
                            return Err(ParseError::Unexpected {
                                found: func_tok.node.kind.description().to_string(),
                                expected: "identifier".to_string(),
                                span: func_tok.span,
                            });
                        }
                    };
                    self.advance();
                    self.expect(&TokenKind::Backtick)?;
                    (func, 9, Assoc::Left) // Default infix precedence
                }
                // Handle . (Dot token for function composition)
                TokenKind::Dot => {
                    let (prec, assoc) = self.get_operator_info(".");
                    if prec < min_prec {
                        break;
                    }
                    (Ident::from_str("."), prec, assoc)
                }
                _ => break,
            };

            if prec < min_prec {
                break;
            }

            let _op_span = self.current_span();
            self.advance();

            let next_min_prec = match assoc {
                Assoc::Left => prec + 1,
                Assoc::Right => prec,
                Assoc::None => prec + 1,
            };

            let rhs = self.parse_infix_expr(next_min_prec)?;
            let span = lhs.span().merge(rhs.span());
            lhs = Expr::Infix(Box::new(lhs), op, Box::new(rhs), span);
        }

        Ok(lhs)
    }

    /// Parse a prefix expression (negation, etc.).
    fn parse_prefix_expr(&mut self) -> ParseResult<Expr> {
        if let Some(tok) = self.current() {
            // Handle prefix negation - lexer produces TokenKind::Minus for standalone `-`
            let is_minus = matches!(&tok.node.kind, TokenKind::Minus)
                || matches!(&tok.node.kind, TokenKind::Operator(s) if s.as_str() == "-");
            if is_minus {
                let start = tok.span;
                self.advance();
                let expr = self.parse_prefix_expr()?;
                let span = start.to(expr.span());
                return Ok(Expr::Neg(Box::new(expr), span));
            }
        }

        self.parse_app_expr()
    }

    /// Parse an application expression.
    fn parse_app_expr(&mut self) -> ParseResult<Expr> {
        let mut expr = self.parse_atom_expr()?;

        loop {
            // Check for record update: `expr { field = value }`
            if self.check(&TokenKind::LBrace) && !matches!(expr, Expr::Con(_, _)) {
                let start = expr.span();
                expr = self.parse_record_update(expr, start)?;
                continue;
            }

            // Check if this looks like an argument
            if let Some(tok) = self.current() {
                if self.is_atom_start(&tok.node.kind) {
                    let arg = self.parse_atom_expr()?;
                    let span = expr.span().to(arg.span());
                    expr = Expr::App(Box::new(expr), Box::new(arg), span);
                    continue;
                }
            }

            break;
        }

        Ok(expr)
    }

    /// Check if a token can start an atom expression.
    fn is_atom_start(&self, kind: &TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Ident(_)
                | TokenKind::QualIdent(_, _)
                | TokenKind::ConId(_)
                | TokenKind::QualConId(_, _)
                | TokenKind::IntLit(_)
                | TokenKind::FloatLit(_)
                | TokenKind::CharLit(_)
                | TokenKind::StringLit(_)
                | TokenKind::LParen
                | TokenKind::LBracket
                | TokenKind::Backslash
                | TokenKind::Let
                | TokenKind::If
                | TokenKind::Case
                | TokenKind::Do
                | TokenKind::Lazy
        )
    }

    /// Parse an atomic expression.
    fn parse_atom_expr(&mut self) -> ParseResult<Expr> {
        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "expression".to_string(),
        })?;

        match &tok.node.kind.clone() {
            TokenKind::Ident(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();
                Ok(Expr::Var(ident, span))
            }

            TokenKind::QualIdent(qualifier, name) => {
                let module_name = ModuleName {
                    parts: vec![*qualifier],
                    span: tok.span,
                };
                let ident = Ident::new(*name);
                let span = tok.span;
                self.advance();
                Ok(Expr::QualVar(module_name, ident, span))
            }

            TokenKind::ConId(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();

                // Check for record construction: Con { field = value }
                if self.check(&TokenKind::LBrace) {
                    return self.parse_record_con(ident, span);
                }

                Ok(Expr::Con(ident, span))
            }

            TokenKind::QualConId(qualifier, name) => {
                let module_name = ModuleName {
                    parts: vec![*qualifier],
                    span: tok.span,
                };
                let ident = Ident::new(*name);
                let span = tok.span;
                self.advance();

                // Check for record construction: M.Con { field = value }
                if self.check(&TokenKind::LBrace) {
                    // TODO: handle qualified record construction
                    // For now, just return as QualCon
                }

                Ok(Expr::QualCon(module_name, ident, span))
            }

            TokenKind::IntLit(ref lit) => {
                let span = tok.span;
                let value = self.parse_int_literal(&lit.text, span)?;
                self.advance();
                Ok(Expr::Lit(Lit::Int(value), span))
            }

            TokenKind::FloatLit(ref lit) => {
                let span = tok.span;
                let value = self.parse_float_literal(&lit.text, span)?;
                self.advance();
                Ok(Expr::Lit(Lit::Float(value), span))
            }

            TokenKind::CharLit(c) => {
                let span = tok.span;
                let c = *c;
                self.advance();
                Ok(Expr::Lit(Lit::Char(c), span))
            }

            TokenKind::StringLit(s) => {
                let span = tok.span;
                let s = s.clone();
                self.advance();
                Ok(Expr::Lit(Lit::String(s), span))
            }

            TokenKind::LParen => self.parse_paren_expr(),

            TokenKind::LBracket => self.parse_list_expr(),

            TokenKind::Backslash => self.parse_lambda(),

            TokenKind::Let => self.parse_let_expr(),

            TokenKind::If => self.parse_if_expr(),

            TokenKind::Case => self.parse_case_expr(),

            TokenKind::Do => self.parse_do_expr(),

            TokenKind::Lazy => self.parse_lazy_expr(),

            _ => Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "expression".to_string(),
                span: tok.span,
            }),
        }
    }

    /// Parse an integer literal.
    pub(crate) fn parse_int_literal(&self, s: &str, span: Span) -> ParseResult<i64> {
        let s = s.replace('_', "");
        let value = if s.starts_with("0x") || s.starts_with("0X") {
            i64::from_str_radix(&s[2..], 16)
        } else if s.starts_with("0o") || s.starts_with("0O") {
            i64::from_str_radix(&s[2..], 8)
        } else if s.starts_with("0b") || s.starts_with("0B") {
            i64::from_str_radix(&s[2..], 2)
        } else {
            s.parse()
        };

        value.map_err(|e| ParseError::InvalidLiteral {
            message: e.to_string(),
            span,
        })
    }

    /// Parse a float literal.
    pub(crate) fn parse_float_literal(&self, s: &str, span: Span) -> ParseResult<f64> {
        let s = s.replace('_', "");
        s.parse().map_err(|e: std::num::ParseFloatError| ParseError::InvalidLiteral {
            message: e.to_string(),
            span,
        })
    }

    /// Get operator precedence and associativity.
    fn get_operator_info(&self, op: &str) -> (u8, Assoc) {
        // Default precedences (can be overridden by fixity declarations)
        match op {
            "." => (9, Assoc::Right),
            "^" | "^^" | "**" => (8, Assoc::Right),
            "*" | "/" | "`div`" | "`mod`" => (7, Assoc::Left),
            "+" | "-" => (6, Assoc::Left),
            ":" | "++" => (5, Assoc::Right),
            "==" | "/=" | "<" | "<=" | ">" | ">=" => (4, Assoc::None),
            "&&" => (3, Assoc::Right),
            "||" => (2, Assoc::Right),
            ">>=" | ">>" => (1, Assoc::Left),
            "$" | "$!" => (0, Assoc::Right),
            _ => (9, Assoc::Left), // Default
        }
    }

    /// Parse a parenthesized expression, tuple, or operator section.
    fn parse_paren_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(&TokenKind::LParen)?;

        if self.eat(&TokenKind::RParen) {
            // Unit: ()
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Expr::Con(Ident::from_str("()"), span));
        }

        // Check for operator section starting with operator: `(+)` or `(+ x)` or `(.)`
        if matches!(self.current_kind(), Some(TokenKind::Operator(_)) | Some(TokenKind::Star) | Some(TokenKind::Minus) | Some(TokenKind::Dot)) {
            return self.parse_operator_section(start);
        }

        // For left sections like `(1 +)`, we need to parse without consuming operators
        // Parse application expression first (no infix operators)
        let first = self.parse_app_expr()?;

        // Check for left operator section: `(x +)` or infix expression in parens
        let op_ident = match self.current_kind().cloned() {
            Some(TokenKind::Operator(sym)) => Some(Ident::new(sym)),
            Some(TokenKind::Star) => Some(Ident::from_str("*")),
            Some(TokenKind::Minus) => Some(Ident::from_str("-")),
            Some(TokenKind::Dot) => Some(Ident::from_str(".")),
            _ => None,
        };

        if let Some(op) = op_ident {
            self.advance();

            if self.eat(&TokenKind::RParen) {
                // Left section: `(x +)` desugars to `(\y -> x + y)`
                let span = start.to(self.tokens[self.pos - 1].span);
                let y = Ident::from_str("$section_arg");
                let y_pat = Pat::Var(y.clone(), Span::DUMMY);
                let y_expr = Expr::Var(y, Span::DUMMY);
                let body = Expr::Infix(Box::new(first), op, Box::new(y_expr), span);
                return Ok(Expr::Lam(vec![y_pat], Box::new(body), span));
            }

            // Otherwise, this is just an infix expression in parens
            let rhs = self.parse_infix_expr(0)?;
            let infix_span = first.span().merge(rhs.span());
            let infix_expr = Expr::Infix(Box::new(first), op, Box::new(rhs), infix_span);
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            return Ok(Expr::Paren(Box::new(infix_expr), span));
        }

        if self.eat(&TokenKind::Comma) {
            // Tuple
            let mut exprs = vec![first];
            loop {
                exprs.push(self.parse_expr()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            Ok(Expr::Tuple(exprs, span))
        } else {
            // Parenthesized expression
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            Ok(Expr::Paren(Box::new(first), span))
        }
    }

    /// Parse a list expression.
    fn parse_list_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(&TokenKind::LBracket)?;

        if self.eat(&TokenKind::RBracket) {
            // Empty list
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Expr::List(vec![], span));
        }

        let first = self.parse_expr()?;

        // Check for list comprehension: [x | ...]
        if self.eat(&TokenKind::Pipe) {
            let stmts = self.parse_list_comp_stmts()?;
            let end = self.expect(&TokenKind::RBracket)?;
            let span = start.to(end.span);
            return Ok(Expr::ListComp(Box::new(first), stmts, span));
        }

        // Check for arithmetic sequence: [1..] or [1..10] or [1,2..]
        if self.eat(&TokenKind::DotDot) {
            if self.eat(&TokenKind::RBracket) {
                // [from..]
                let span = start.to(self.tokens[self.pos - 1].span);
                return Ok(Expr::ArithSeq(ArithSeq::From(Box::new(first)), span));
            }
            let to = self.parse_expr()?;
            let end = self.expect(&TokenKind::RBracket)?;
            let span = start.to(end.span);
            return Ok(Expr::ArithSeq(ArithSeq::FromTo(Box::new(first), Box::new(to)), span));
        }

        if self.eat(&TokenKind::Comma) {
            // Could be tuple-like list or [from, then..]
            let second = self.parse_expr()?;

            if self.eat(&TokenKind::DotDot) {
                if self.eat(&TokenKind::RBracket) {
                    // [from, then..]
                    let span = start.to(self.tokens[self.pos - 1].span);
                    return Ok(Expr::ArithSeq(
                        ArithSeq::FromThen(Box::new(first), Box::new(second)),
                        span,
                    ));
                }
                let to = self.parse_expr()?;
                let end = self.expect(&TokenKind::RBracket)?;
                let span = start.to(end.span);
                return Ok(Expr::ArithSeq(
                    ArithSeq::FromThenTo(Box::new(first), Box::new(second), Box::new(to)),
                    span,
                ));
            }

            // Regular list
            let mut exprs = vec![first, second];
            while self.eat(&TokenKind::Comma) {
                exprs.push(self.parse_expr()?);
            }
            let end = self.expect(&TokenKind::RBracket)?;
            let span = start.to(end.span);
            return Ok(Expr::List(exprs, span));
        }

        // Single-element list
        let end = self.expect(&TokenKind::RBracket)?;
        let span = start.to(end.span);
        Ok(Expr::List(vec![first], span))
    }

    /// Parse list comprehension statements.
    fn parse_list_comp_stmts(&mut self) -> ParseResult<Vec<Stmt>> {
        let mut stmts = vec![self.parse_stmt()?];
        while self.eat(&TokenKind::Comma) {
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    /// Parse a statement (for do blocks and list comprehensions).
    fn parse_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.current_span();

        // Try to parse as generator: pat <- expr
        // For simplicity, we first try parsing an expression
        // and check if it's followed by <-
        if self.check(&TokenKind::Let) {
            self.advance();
            let decls = self.parse_local_decls()?;
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Stmt::LetStmt(decls, span));
        }

        // Try to parse as pattern <- expr
        let pat_or_expr = self.parse_expr()?;

        if self.eat(&TokenKind::LeftArrow) {
            // Generator
            let pat = self.expr_to_pat(pat_or_expr)?;
            let expr = self.parse_expr()?;
            let span = start.to(expr.span());
            Ok(Stmt::Generator(pat, expr, span))
        } else {
            // Qualifier
            let span = pat_or_expr.span();
            Ok(Stmt::Qualifier(pat_or_expr, span))
        }
    }

    /// Convert an expression to a pattern (for generators).
    fn expr_to_pat(&self, expr: Expr) -> ParseResult<Pat> {
        match expr {
            Expr::Var(id, span) => Ok(Pat::Var(id, span)),
            Expr::Con(id, span) => Ok(Pat::Con(id, vec![], span)),
            Expr::Lit(lit, span) => Ok(Pat::Lit(lit, span)),
            Expr::Tuple(exprs, span) => {
                let pats: ParseResult<Vec<_>> = exprs.into_iter().map(|e| self.expr_to_pat(e)).collect();
                Ok(Pat::Tuple(pats?, span))
            }
            Expr::List(exprs, span) => {
                let pats: ParseResult<Vec<_>> = exprs.into_iter().map(|e| self.expr_to_pat(e)).collect();
                Ok(Pat::List(pats?, span))
            }
            Expr::Paren(e, span) => {
                let p = self.expr_to_pat(*e)?;
                Ok(Pat::Paren(Box::new(p), span))
            }
            Expr::App(f, x, span) => {
                // Could be constructor application
                if let Expr::Con(id, _) = *f {
                    let pat = self.expr_to_pat(*x)?;
                    Ok(Pat::Con(id, vec![pat], span))
                } else {
                    Err(ParseError::Unexpected {
                        found: "expression".to_string(),
                        expected: "pattern".to_string(),
                        span,
                    })
                }
            }
            _ => Err(ParseError::Unexpected {
                found: "expression".to_string(),
                expected: "pattern".to_string(),
                span: expr.span(),
            }),
        }
    }

    /// Parse a lambda expression.
    fn parse_lambda(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(&TokenKind::Backslash)?;

        let mut pats = vec![self.parse_pattern()?];
        while !self.check(&TokenKind::Arrow) && !self.at_eof() {
            if self.is_pattern_start() {
                pats.push(self.parse_pattern()?);
            } else {
                break;
            }
        }

        self.expect(&TokenKind::Arrow)?;
        let body = self.parse_expr()?;
        let span = start.to(body.span());

        Ok(Expr::Lam(pats, Box::new(body), span))
    }

    /// Parse a let expression.
    fn parse_let_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(&TokenKind::Let)?;

        let decls = self.parse_local_decls()?;

        self.expect(&TokenKind::In)?;
        let body = self.parse_expr()?;
        let span = start.to(body.span());

        Ok(Expr::Let(decls, Box::new(body), span))
    }

    /// Parse an if expression.
    fn parse_if_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(&TokenKind::If)?;

        let cond = self.parse_expr()?;
        self.expect(&TokenKind::Then)?;
        let then_branch = self.parse_expr()?;
        self.expect(&TokenKind::Else)?;
        let else_branch = self.parse_expr()?;

        let span = start.to(else_branch.span());
        Ok(Expr::If(
            Box::new(cond),
            Box::new(then_branch),
            Box::new(else_branch),
            span,
        ))
    }

    /// Parse a case expression.
    fn parse_case_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(&TokenKind::Case)?;

        let scrutinee = self.parse_expr()?;
        self.expect(&TokenKind::Of)?;

        let alts = self.parse_case_alts()?;
        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);

        Ok(Expr::Case(Box::new(scrutinee), alts, span))
    }

    /// Parse case alternatives.
    fn parse_case_alts(&mut self) -> ParseResult<Vec<Alt>> {
        // Simplified: expect braces or use layout
        let mut alts = Vec::new();

        if self.eat(&TokenKind::LBrace) {
            if !self.check(&TokenKind::RBrace) {
                alts.push(self.parse_alt()?);
                while self.eat(&TokenKind::Semi) {
                    if self.check(&TokenKind::RBrace) {
                        break;
                    }
                    alts.push(self.parse_alt()?);
                }
            }
            self.expect(&TokenKind::RBrace)?;
        } else {
            // Layout: simplified version
            alts.push(self.parse_alt()?);
        }

        Ok(alts)
    }

    /// Parse a case alternative.
    fn parse_alt(&mut self) -> ParseResult<Alt> {
        let start = self.current_span();
        let pat = self.parse_pattern()?;

        // Check for guards: `| guard -> expr`
        let rhs = if self.check(&TokenKind::Pipe) {
            let guards = self.parse_guarded_rhss()?;
            let end_span = guards.last().map(|g| g.span).unwrap_or(start);
            Rhs::Guarded(guards, start.to(end_span))
        } else {
            self.expect(&TokenKind::Arrow)?;
            let expr = self.parse_expr()?;
            let span = start.to(expr.span());
            Rhs::Simple(expr, span)
        };

        // Parse optional where clause
        let wheres = if self.eat(&TokenKind::Where) {
            self.parse_local_decls()?
        } else {
            vec![]
        };

        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);
        Ok(Alt {
            pat,
            rhs,
            wheres,
            span,
        })
    }

    /// Parse guarded right-hand sides: `| guard -> expr | guard -> expr ...`
    fn parse_guarded_rhss(&mut self) -> ParseResult<Vec<GuardedRhs>> {
        let mut guarded_rhss = Vec::new();
        while self.eat(&TokenKind::Pipe) {
            let start = self.current_span();
            let guard_expr = self.parse_expr()?;
            let guard_span = guard_expr.span();
            self.expect(&TokenKind::Arrow)?;
            let body = self.parse_expr()?;
            let span = start.to(body.span());
            guarded_rhss.push(GuardedRhs {
                guards: vec![Guard::Expr(guard_expr, guard_span)],
                body,
                span,
            });
        }
        Ok(guarded_rhss)
    }

    /// Parse a do expression.
    fn parse_do_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(&TokenKind::Do)?;

        let stmts = self.parse_do_stmts()?;
        let span = start.to(self.tokens[self.pos.saturating_sub(1)].span);

        Ok(Expr::Do(stmts, span))
    }

    /// Parse do statements.
    fn parse_do_stmts(&mut self) -> ParseResult<Vec<Stmt>> {
        let mut stmts = Vec::new();

        // Check for explicit or virtual opening brace
        let explicit_braces = self.eat(&TokenKind::LBrace);
        let virtual_braces = !explicit_braces && self.eat(&TokenKind::VirtualLBrace);
        let has_braces = explicit_braces || virtual_braces;

        if has_braces {
            // Parse statements separated by semicolons (real or virtual)
            let rbrace = if explicit_braces {
                TokenKind::RBrace
            } else {
                TokenKind::VirtualRBrace
            };

            if !self.check(&rbrace) {
                stmts.push(self.parse_stmt()?);
                // Accept either real or virtual semicolons
                while self.eat(&TokenKind::Semi) || self.eat(&TokenKind::VirtualSemi) {
                    if self.check(&rbrace) || self.check(&TokenKind::VirtualRBrace) {
                        break;
                    }
                    // Skip any extra semicolons
                    while self.eat(&TokenKind::Semi) || self.eat(&TokenKind::VirtualSemi) {}
                    if self.check(&rbrace) || self.check(&TokenKind::VirtualRBrace) {
                        break;
                    }
                    stmts.push(self.parse_stmt()?);
                }
            }

            if explicit_braces {
                self.expect(&TokenKind::RBrace)?;
            } else {
                // Virtual braces: consume if present, but don't require
                self.eat(&TokenKind::VirtualRBrace);
            }
        } else {
            // No braces at all - single statement
            stmts.push(self.parse_stmt()?);
        }

        Ok(stmts)
    }

    /// Parse a lazy expression (H26 extension).
    fn parse_lazy_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(&TokenKind::Lazy)?;
        self.expect(&TokenKind::LBrace)?;
        let expr = self.parse_expr()?;
        let end = self.expect(&TokenKind::RBrace)?;
        let span = start.to(end.span);

        Ok(Expr::Lazy(Box::new(expr), span))
    }

    /// Parse record construction: `Con { field = value, ... }`
    fn parse_record_con(&mut self, con: Ident, start: Span) -> ParseResult<Expr> {
        self.expect(&TokenKind::LBrace)?;

        let mut fields = Vec::new();
        if !self.check(&TokenKind::RBrace) {
            fields.push(self.parse_field_bind()?);
            while self.eat(&TokenKind::Comma) {
                if self.check(&TokenKind::RBrace) {
                    break;
                }
                fields.push(self.parse_field_bind()?);
            }
        }

        let end = self.expect(&TokenKind::RBrace)?;
        let span = start.to(end.span);
        Ok(Expr::RecordCon(con, fields, span))
    }

    /// Parse record update: `expr { field = value, ... }`
    fn parse_record_update(&mut self, expr: Expr, start: Span) -> ParseResult<Expr> {
        self.expect(&TokenKind::LBrace)?;

        let mut fields = Vec::new();
        if !self.check(&TokenKind::RBrace) {
            fields.push(self.parse_field_bind()?);
            while self.eat(&TokenKind::Comma) {
                if self.check(&TokenKind::RBrace) {
                    break;
                }
                fields.push(self.parse_field_bind()?);
            }
        }

        let end = self.expect(&TokenKind::RBrace)?;
        let span = start.to(end.span);
        Ok(Expr::RecordUpd(Box::new(expr), fields, span))
    }

    /// Parse a field binding: `field = expr`, `Mod.field = expr`, or `field` (punning)
    fn parse_field_bind(&mut self) -> ParseResult<FieldBind> {
        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "field name".to_string(),
        })?;

        let (qualifier, name, span) = match &tok.node.kind {
            TokenKind::Ident(sym) => (None, Ident::new(*sym), tok.span),
            TokenKind::QualIdent(qual, sym) => {
                let module_name = ModuleName {
                    parts: vec![*qual],
                    span: tok.span,
                };
                (Some(module_name), Ident::new(*sym), tok.span)
            }
            _ => {
                return Err(ParseError::Unexpected {
                    found: tok.node.kind.description().to_string(),
                    expected: "field name".to_string(),
                    span: tok.span,
                });
            }
        };
        self.advance();

        let value = if self.eat(&TokenKind::Eq) {
            Some(self.parse_expr()?)
        } else {
            None // Punning: `Foo { bar }` means `Foo { bar = bar }`
        };

        let end_span = value.as_ref().map(|e| e.span()).unwrap_or(span);
        let full_span = span.to(end_span);
        Ok(FieldBind {
            qualifier,
            name,
            value,
            span: full_span,
        })
    }

    /// Parse an operator section: `(+ 1)` or `(1 +)`
    fn parse_operator_section(&mut self, start: Span) -> ParseResult<Expr> {
        // Already consumed LParen, now at operator
        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "operator".to_string(),
        })?;

        let op = match &tok.node.kind {
            TokenKind::Operator(sym) => Ident::new(*sym),
            TokenKind::Star => Ident::from_str("*"),
            TokenKind::Minus => Ident::from_str("-"),
            TokenKind::Dot => Ident::from_str("."),
            _ => {
                return Err(ParseError::Unexpected {
                    found: tok.node.kind.description().to_string(),
                    expected: "operator".to_string(),
                    span: tok.span,
                });
            }
        };
        self.advance();

        if self.eat(&TokenKind::RParen) {
            // Just `(+)` - operator as a function
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Expr::Var(op, span));
        }

        // Right section: `(+ 1)`
        let arg = self.parse_expr()?;
        let end = self.expect(&TokenKind::RParen)?;
        let span = start.to(end.span);

        // Desugar (+ x) to (\y -> y + x)
        let y = Ident::from_str("$section_arg");
        let y_pat = Pat::Var(y.clone(), Span::DUMMY);
        let y_expr = Expr::Var(y, Span::DUMMY);
        let body = Expr::Infix(Box::new(y_expr), op, Box::new(arg), span);
        Ok(Expr::Lam(vec![y_pat], Box::new(body), span))
    }

    /// Parse type annotation: `expr :: Type`
    #[allow(dead_code)]
    fn parse_type_annotation(&mut self, expr: Expr) -> ParseResult<Expr> {
        let start = expr.span();
        self.expect(&TokenKind::DoubleColon)?;
        let ty = self.parse_type()?;
        let span = start.to(ty.span());
        Ok(Expr::Ann(Box::new(expr), ty, span))
    }
}

/// Operator associativity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Assoc {
    Left,
    Right,
    None,
}
