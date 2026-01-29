//! Type parsing.

use bhc_ast::{Constraint, ModuleName, TyVar, Type};
use bhc_intern::Ident;
use bhc_lexer::TokenKind;

use crate::{ParseError, ParseResult, Parser};

impl<'src> Parser<'src> {
    /// Parse a type.
    pub fn parse_type(&mut self) -> ParseResult<Type> {
        let start = self.current_span();

        // Check for forall
        if self.check(&TokenKind::Forall) {
            return self.parse_forall_type();
        }

        // Try to parse a constrained type: `Eq a => ...`
        // This is tricky because we need lookahead to distinguish
        // `Class a => ...` from `Type -> ...`
        if let Some(constraints) = self.try_parse_context()? {
            let ty = self.parse_fun_type()?;
            let span = start.to(ty.span());
            return Ok(Type::Constrained(constraints, Box::new(ty), span));
        }

        self.parse_fun_type()
    }

    /// Try to parse a context (type class constraints).
    /// Returns None if this doesn't look like a context.
    fn try_parse_context(&mut self) -> ParseResult<Option<Vec<Constraint>>> {
        // Save position for backtracking
        let saved_pos = self.pos;

        // A context looks like: `Class arg` or `(Class1 a, Class2 b)` followed by `=>`
        let constraints = if self.check(&TokenKind::LParen) {
            // Try to parse parenthesized context
            self.advance(); // consume (

            if self.check(&TokenKind::RParen) {
                // Empty context `() =>` - unlikely but valid
                self.advance();
                if self.eat(&TokenKind::FatArrow) {
                    return Ok(Some(vec![]));
                }
                // Not a context, backtrack
                self.pos = saved_pos;
                return Ok(None);
            }

            let mut constraints = vec![];
            match self.try_parse_constraint() {
                Ok(Some(c)) => constraints.push(c),
                _ => {
                    self.pos = saved_pos;
                    return Ok(None);
                }
            }

            while self.eat(&TokenKind::Comma) {
                match self.try_parse_constraint() {
                    Ok(Some(c)) => constraints.push(c),
                    _ => {
                        self.pos = saved_pos;
                        return Ok(None);
                    }
                }
            }

            if !self.eat(&TokenKind::RParen) {
                self.pos = saved_pos;
                return Ok(None);
            }

            constraints
        } else {
            // Try to parse a single constraint
            match self.try_parse_constraint() {
                Ok(Some(c)) => vec![c],
                _ => {
                    self.pos = saved_pos;
                    return Ok(None);
                }
            }
        };

        // Check for =>
        if self.eat(&TokenKind::FatArrow) {
            Ok(Some(constraints))
        } else {
            // Not a context, backtrack
            self.pos = saved_pos;
            Ok(None)
        }
    }

    /// Try to parse a single constraint like `Eq a` or `Functor f`.
    fn try_parse_constraint(&mut self) -> ParseResult<Option<Constraint>> {
        let start = self.current_span();

        // Constraint class must be a ConId
        let class = match self.current_kind() {
            Some(TokenKind::ConId(sym)) => {
                let ident = Ident::new(*sym);
                self.advance();
                ident
            }
            _ => return Ok(None),
        };

        // Parse type arguments
        let mut args = vec![];
        while self.is_atype_start() {
            args.push(self.parse_atype()?);
        }

        let end_span = args.last().map(|t| t.span()).unwrap_or(start);
        let span = start.to(end_span);

        Ok(Some(Constraint { class, args, span }))
    }

    /// Parse a function type: `a -> b`.
    fn parse_fun_type(&mut self) -> ParseResult<Type> {
        let lhs = self.parse_app_type()?;

        // Skip any doc comments before checking for ->
        // (Haddock argument documentation like `-- ^`)
        self.skip_doc_comments();

        if self.eat(&TokenKind::Arrow) {
            let rhs = self.parse_fun_type()?;
            let span = lhs.span().to(rhs.span());
            Ok(Type::Fun(Box::new(lhs), Box::new(rhs), span))
        } else {
            Ok(lhs)
        }
    }

    /// Parse a type application: `Maybe Int`.
    fn parse_app_type(&mut self) -> ParseResult<Type> {
        let mut ty = self.parse_atype()?;

        while self.is_atype_start() {
            let arg = self.parse_atype()?;
            let span = ty.span().to(arg.span());
            ty = Type::App(Box::new(ty), Box::new(arg), span);
        }

        Ok(ty)
    }

    /// Check if current token can start an atomic type.
    pub fn is_atype_start(&self) -> bool {
        match self.current_kind() {
            Some(kind) => matches!(
                kind,
                TokenKind::Ident(_)
                    | TokenKind::ConId(_)
                    | TokenKind::QualConId(_, _)
                    | TokenKind::LParen
                    | TokenKind::LBracket
                    // M9: Type-level naturals and promoted lists
                    | TokenKind::IntLit(_)
                    | TokenKind::TickLBracket
                    // Strictness/laziness annotations for constructor fields
                    | TokenKind::Bang
                    | TokenKind::Tilde
            ),
            None => false,
        }
    }

    /// Parse an atomic type.
    pub fn parse_atype(&mut self) -> ParseResult<Type> {
        let tok = self.current().ok_or(ParseError::UnexpectedEof {
            expected: "type".to_string(),
        })?;

        match &tok.node.kind.clone() {
            TokenKind::Ident(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();
                Ok(Type::Var(TyVar { name: ident, span }, span))
            }

            TokenKind::ConId(sym) => {
                let ident = Ident::new(*sym);
                let span = tok.span;
                self.advance();
                Ok(Type::Con(ident, span))
            }

            TokenKind::QualConId(qualifier, name) => {
                let module_name = ModuleName {
                    parts: vec![*qualifier],
                    span: tok.span,
                };
                let ident = Ident::new(*name);
                let span = tok.span;
                self.advance();
                Ok(Type::QualCon(module_name, ident, span))
            }

            TokenKind::LParen => self.parse_paren_type(),

            TokenKind::LBracket => self.parse_list_type(),

            // M9: Type-level natural literal
            TokenKind::IntLit(lit) => {
                let span = tok.span;
                let value = lit.parse().ok_or_else(|| ParseError::Unexpected {
                    found: "invalid integer".to_string(),
                    expected: "type-level natural".to_string(),
                    span,
                })?;
                // Type-level naturals must be non-negative
                if value < 0 {
                    return Err(ParseError::Unexpected {
                        found: "negative integer".to_string(),
                        expected: "type-level natural (non-negative)".to_string(),
                        span,
                    });
                }
                self.advance();
                Ok(Type::NatLit(value as u64, span))
            }

            // M9: Promoted list syntax '[a, b, c]
            TokenKind::TickLBracket => self.parse_promoted_list(),

            // Strict type annotation: !Type
            TokenKind::Bang => {
                let start = tok.span;
                self.advance();
                let inner = self.parse_atype()?;
                let span = start.to(inner.span());
                Ok(Type::Bang(Box::new(inner), span))
            }

            // Lazy type annotation: ~Type
            TokenKind::Tilde => {
                let start = tok.span;
                self.advance();
                let inner = self.parse_atype()?;
                let span = start.to(inner.span());
                Ok(Type::Lazy(Box::new(inner), span))
            }

            _ => Err(ParseError::Unexpected {
                found: tok.node.kind.description().to_string(),
                expected: "type".to_string(),
                span: tok.span,
            }),
        }
    }

    /// Parse a promoted list: `'[a, b, c]`.
    fn parse_promoted_list(&mut self) -> ParseResult<Type> {
        let start = self.current_span();
        self.expect(&TokenKind::TickLBracket)?;

        if self.eat(&TokenKind::RBracket) {
            // Empty promoted list: '[]
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Type::PromotedList(vec![], span));
        }

        let mut elems = vec![self.parse_type()?];
        while self.eat(&TokenKind::Comma) {
            elems.push(self.parse_type()?);
        }

        let end = self.expect(&TokenKind::RBracket)?;
        let span = start.to(end.span);

        Ok(Type::PromotedList(elems, span))
    }

    /// Parse a parenthesized type or tuple type.
    fn parse_paren_type(&mut self) -> ParseResult<Type> {
        let start = self.current_span();
        self.expect(&TokenKind::LParen)?;

        if self.eat(&TokenKind::RParen) {
            // Unit type: ()
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Type::Tuple(vec![], span));
        }

        // Check for function type in parens: (->)
        if self.eat(&TokenKind::Arrow) {
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            return Ok(Type::Con(Ident::from_str("->"), span));
        }

        let first = self.parse_type()?;

        if self.eat(&TokenKind::Comma) {
            // Tuple type
            let mut types = vec![first];
            loop {
                types.push(self.parse_type()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            Ok(Type::Tuple(types, span))
        } else {
            // Parenthesized type
            let end = self.expect(&TokenKind::RParen)?;
            let span = start.to(end.span);
            Ok(Type::Paren(Box::new(first), span))
        }
    }

    /// Parse a list type: `[a]`.
    fn parse_list_type(&mut self) -> ParseResult<Type> {
        let start = self.current_span();
        self.expect(&TokenKind::LBracket)?;

        if self.eat(&TokenKind::RBracket) {
            // List type constructor: []
            let span = start.to(self.tokens[self.pos - 1].span);
            return Ok(Type::Con(Ident::from_str("[]"), span));
        }

        let elem = self.parse_type()?;
        let end = self.expect(&TokenKind::RBracket)?;
        let span = start.to(end.span);

        Ok(Type::List(Box::new(elem), span))
    }

    /// Parse a forall type: `forall a b. Type`.
    fn parse_forall_type(&mut self) -> ParseResult<Type> {
        let start = self.current_span();
        self.expect(&TokenKind::Forall)?;

        let mut vars = Vec::new();
        while let Some(tok) = self.current() {
            match &tok.node.kind {
                TokenKind::Ident(sym) => {
                    let name = Ident::new(*sym);
                    let span = tok.span;
                    self.advance();
                    vars.push(TyVar { name, span });
                }
                // The `.` is lexed as TokenKind::Dot, not Operator(".")
                TokenKind::Dot => {
                    self.advance();
                    break;
                }
                _ => break,
            }
        }

        let ty = self.parse_type()?;
        let span = start.to(ty.span());

        Ok(Type::Forall(vars, Box::new(ty), span))
    }
}
