//! Pattern parsing.

use bhc_ast::{FieldPat, Lit, ModuleName, Pat};
use bhc_intern::Ident;
use bhc_lexer::TokenKind;
use bhc_span::Span;

use crate::{ParseResult, Parser, ParseError};

impl<'src> Parser<'src> {
    /// Check if the current token can start a pattern.
    pub fn is_pattern_start(&self) -> bool {
        match self.current_kind() {
            Some(kind) => matches!(
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
                    | TokenKind::Underscore
                    | TokenKind::Tilde
                    | TokenKind::Bang
            ),
            None => false,
        }
    }

    /// Parse a pattern.
    pub fn parse_pattern(&mut self) -> ParseResult<Pat> {
        self.parse_infix_pattern()
    }

    /// Parse an infix pattern like `x : xs` or `x :| xs`.
    fn parse_infix_pattern(&mut self) -> ParseResult<Pat> {
        let mut pat = self.parse_app_pattern()?;

        while let Some(tok) = self.current() {
            match &tok.node.kind {
                // Constructor operators like `:`, `:|` are valid in infix patterns
                TokenKind::ConOperator(sym) => {
                    let op = Ident::new(*sym);
                    self.advance();
                    let rhs = self.parse_infix_pattern()?;
                    let span = pat.span().to(rhs.span());
                    pat = Pat::Infix(Box::new(pat), op, Box::new(rhs), span);
                }
                _ => break,
            }
        }

        Ok(pat)
    }

    /// Parse an application pattern like `Just x`.
    fn parse_app_pattern(&mut self) -> ParseResult<Pat> {
        let first = self.parse_atom_pattern()?;

        // Check for constructor application
        if let Pat::Con(con, args, span) = first {
            if args.is_empty() {
                let mut new_args = Vec::new();
                while self.is_apat_start() {
                    new_args.push(self.parse_atom_pattern()?);
                }
                if new_args.is_empty() {
                    return Ok(Pat::Con(con, args, span));
                }
                let new_span = span.to(new_args.last().unwrap().span());
                return Ok(Pat::Con(con, new_args, new_span));
            }
            return Ok(Pat::Con(con, args, span));
        }

        Ok(first)
    }

    /// Check if current token can start an atomic pattern.
    fn is_apat_start(&self) -> bool {
        match self.current_kind() {
            Some(kind) => matches!(
                kind,
                TokenKind::Ident(_)
                    | TokenKind::IntLit(_)
                    | TokenKind::FloatLit(_)
                    | TokenKind::CharLit(_)
                    | TokenKind::StringLit(_)
                    | TokenKind::LParen
                    | TokenKind::LBracket
                    | TokenKind::Underscore
            ),
            None => false,
        }
    }

    /// Parse an atomic pattern.
    fn parse_atom_pattern(&mut self) -> ParseResult<Pat> {
        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "pattern".to_string(),
        })?;

        match &tok.node.kind.clone() {
            TokenKind::Underscore => {
                let span = tok.span;
                self.advance();
                Ok(Pat::Wildcard(span))
            }

            TokenKind::Ident(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();

                // Check for as-pattern: x@pat
                if self.eat(&TokenKind::At) {
                    let pat = self.parse_atom_pattern()?;
                    let new_span = span.to(pat.span());
                    Ok(Pat::As(ident, Box::new(pat), new_span))
                } else {
                    Ok(Pat::Var(ident, span))
                }
            }

            TokenKind::QualIdent(qual, name) => {
                // Qualified identifier like M.x - treat as variable
                let full_name = format!("{}.{}", qual.as_str(), name.as_str());
                let ident = Ident::from_str(&full_name);
                let span = tok.span;
                self.advance();
                Ok(Pat::Var(ident, span))
            }

            TokenKind::ConId(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();

                // Check for record pattern: Con { field = pat, ... }
                if self.check(&TokenKind::LBrace) {
                    return self.parse_record_pattern(ident, span);
                }

                Ok(Pat::Con(ident, vec![], span))
            }

            TokenKind::QualConId(qual, name) => {
                // Qualified constructor like W.RationalRect
                let full_name = format!("{}.{}", qual.as_str(), name.as_str());
                let ident = Ident::from_str(&full_name);
                let span = tok.span;
                self.advance();

                // Check for record pattern: Qual.Con { field = pat, ... }
                if self.check(&TokenKind::LBrace) {
                    return self.parse_record_pattern(ident, span);
                }

                Ok(Pat::Con(ident, vec![], span))
            }

            TokenKind::IntLit(ref lit) => {
                let span = tok.span;
                let value = self.parse_int_literal(&lit.text, span)?;
                self.advance();
                Ok(Pat::Lit(Lit::Int(value), span))
            }

            TokenKind::FloatLit(ref lit) => {
                let span = tok.span;
                let value = self.parse_float_literal(&lit.text, span)?;
                self.advance();
                Ok(Pat::Lit(Lit::Float(value), span))
            }

            TokenKind::CharLit(c) => {
                let span = tok.span;
                let c = *c;
                self.advance();
                Ok(Pat::Lit(Lit::Char(c), span))
            }

            TokenKind::StringLit(s) => {
                let span = tok.span;
                let s = s.clone();
                self.advance();
                Ok(Pat::Lit(Lit::String(s), span))
            }

            TokenKind::LParen => self.parse_paren_pattern(),

            TokenKind::LBracket => self.parse_list_pattern(),

            TokenKind::Tilde => {
                let start = tok.span;
                self.advance();
                let pat = self.parse_atom_pattern()?;
                let span = start.to(pat.span());
                Ok(Pat::Lazy(Box::new(pat), span))
            }

            TokenKind::Bang => {
                let start = tok.span;
                self.advance();
                let pat = self.parse_atom_pattern()?;
                let span = start.to(pat.span());
                Ok(Pat::Bang(Box::new(pat), span))
            }

            _ => Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "pattern".to_string(),
                span: tok.span,
            }),
        }
    }

    /// Parse a parenthesized pattern or tuple pattern.
    fn parse_paren_pattern(&mut self) -> ParseResult<Pat> {
        let start = self.current_span();
        self.expect(&TokenKind::LParen)?;

        if self.eat(&TokenKind::RParen) {
            // Unit pattern: ()
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Pat::Con(Ident::from_str("()"), vec![], span));
        }

        let first = self.parse_pattern()?;

        // Check for pattern type signature: (pat :: Type)
        if self.eat(&TokenKind::DoubleColon) {
            let ty = self.parse_type()?;
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            return Ok(Pat::Ann(Box::new(first), ty, span));
        }

        if self.eat(&TokenKind::Comma) {
            // Tuple pattern
            let mut pats = vec![first];
            loop {
                pats.push(self.parse_pattern()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            Ok(Pat::Tuple(pats, span))
        } else {
            // Parenthesized pattern
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            Ok(Pat::Paren(Box::new(first), span))
        }
    }

    /// Parse a list pattern.
    fn parse_list_pattern(&mut self) -> ParseResult<Pat> {
        let start = self.current_span();
        self.expect(&TokenKind::LBracket)?;

        if self.eat(&TokenKind::RBracket) {
            // Empty list: []
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Pat::List(vec![], span));
        }

        let mut pats = vec![self.parse_pattern()?];
        while self.eat(&TokenKind::Comma) {
            if self.check(&TokenKind::RBracket) {
                break;
            }
            pats.push(self.parse_pattern()?);
        }

        let end = self.expect(&TokenKind::RBracket)?;
        let span = start.to(end.span);
        Ok(Pat::List(pats, span))
    }

    /// Parse a record pattern: `Con { field = pat, ... }`
    fn parse_record_pattern(&mut self, con: Ident, start: Span) -> ParseResult<Pat> {
        self.expect(&TokenKind::LBrace)?;

        let mut fields = Vec::new();
        if !self.check(&TokenKind::RBrace) {
            fields.push(self.parse_field_pat()?);
            while self.eat(&TokenKind::Comma) {
                if self.check(&TokenKind::RBrace) {
                    break;
                }
                fields.push(self.parse_field_pat()?);
            }
        }

        let end = self.expect(&TokenKind::RBrace)?;
        let span = start.to(end.span);
        Ok(Pat::Record(con, fields, span))
    }

    /// Parse a field pattern: `field = pat`, `Mod.field = pat`, or `field` (punning)
    fn parse_field_pat(&mut self) -> ParseResult<FieldPat> {
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

        let pat = if self.eat(&TokenKind::Eq) {
            Some(self.parse_pattern()?)
        } else {
            None // Punning: `Foo { bar }` means `Foo { bar = bar }`
        };

        let end_span = pat.as_ref().map(|p| p.span()).unwrap_or(span);
        let full_span = span.to(end_span);
        Ok(FieldPat {
            qualifier,
            name,
            pat,
            span: full_span,
        })
    }
}
