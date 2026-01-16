//! Pattern parsing.

use bhc_ast::*;
use bhc_intern::Ident;
use bhc_lexer::TokenKind;

use crate::{ParseResult, Parser, ParseError};

impl<'src> Parser<'src> {
    /// Check if the current token can start a pattern.
    pub fn is_pattern_start(&self) -> bool {
        match self.current_kind() {
            Some(kind) => matches!(
                kind,
                TokenKind::Ident(_)
                    | TokenKind::ConId(_)
                    | TokenKind::IntLit(_)
                    | TokenKind::FloatLit(_)
                    | TokenKind::CharLit(_)
                    | TokenKind::StringLit(_)
                    | TokenKind::LParen
                    | TokenKind::LBracket
                    | TokenKind::Underscore
                    | TokenKind::Tilde
            ),
            None => false,
        }
    }

    /// Parse a pattern.
    pub fn parse_pattern(&mut self) -> ParseResult<Pat> {
        self.parse_infix_pattern()
    }

    /// Parse an infix pattern like `x : xs`.
    fn parse_infix_pattern(&mut self) -> ParseResult<Pat> {
        let mut pat = self.parse_app_pattern()?;

        while let Some(tok) = self.current() {
            match &tok.node.kind {
                TokenKind::Operator(sym) if sym.as_str() == ":" => {
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

            TokenKind::ConId(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();
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
}
