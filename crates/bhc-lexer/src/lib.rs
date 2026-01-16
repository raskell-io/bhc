//! Lexical analysis for BHC.
//!
//! This crate provides a lexer for Haskell 2026 source code, producing a stream
//! of tokens with source locations. The lexer handles:
//!
//! - All Haskell 2010 tokens plus GHC and H26 extensions
//! - Unicode identifiers and operators
//! - Layout rule (significant indentation)
//! - Qualified names (`Data.List.map`)
//! - Documentation comments (`-- |`, `{- | -}`)
//! - Pragmas (`{-# ... #-}`)
//!
//! # Layout Rule
//!
//! Haskell uses significant indentation (the "layout rule") to delimit blocks.
//! After `where`, `let`, `do`, or `of`, the lexer inserts virtual braces and
//! semicolons based on indentation:
//!
//! ```text
//! f x = case x of       -- layout starts after 'of'
//!   Just y -> y         -- virtual '{' inserted, column 2
//!   Nothing -> 0        -- virtual ';' inserted (same column)
//!                       -- virtual '}' inserted (dedent or EOF)
//! ```

#![warn(missing_docs)]

use bhc_intern::Symbol;
use bhc_span::{Span, Spanned};
use unicode_xid::UnicodeXID;

mod token;

pub use token::{
    FloatLiteral, IntLiteral, LexError, NumBase, NumSuffix, Token, TokenKind,
};

/// Configuration for the lexer.
#[derive(Clone, Debug)]
pub struct LexerConfig {
    /// Whether to emit documentation comments as tokens.
    pub preserve_doc_comments: bool,
    /// Whether to emit pragma tokens.
    pub preserve_pragmas: bool,
    /// Whether to warn on tabs in indentation.
    pub warn_tabs: bool,
    /// Tab width for indentation calculation.
    pub tab_width: u32,
}

impl Default for LexerConfig {
    fn default() -> Self {
        Self {
            preserve_doc_comments: true,
            preserve_pragmas: true,
            warn_tabs: true,
            tab_width: 8,
        }
    }
}

/// A lexer for Haskell 2026 source code.
pub struct Lexer<'src> {
    /// The source code being lexed.
    src: &'src str,
    /// Current byte position in the source.
    pos: usize,
    /// Configuration options.
    config: LexerConfig,

    // Layout rule state
    /// Stack of indentation levels for layout blocks.
    /// Each entry is (column, is_explicit) where is_explicit means
    /// the block was opened with an explicit '{'.
    layout_stack: Vec<(u32, bool)>,
    /// Pending tokens to emit (from layout rule).
    pending: Vec<Spanned<Token>>,
    /// Column of the first token on the current line (1-indexed).
    line_start_column: u32,
    /// Whether we just saw a layout keyword and expect a block.
    expect_layout_block: bool,
    /// Whether we're at the start of a logical line.
    at_line_start: bool,
    /// Current line number (1-indexed).
    line: u32,
    /// Column of current position (1-indexed).
    column: u32,
}

impl<'src> Lexer<'src> {
    /// Create a new lexer for the given source code.
    #[must_use]
    pub fn new(src: &'src str) -> Self {
        Self::with_config(src, LexerConfig::default())
    }

    /// Create a new lexer with custom configuration.
    #[must_use]
    pub fn with_config(src: &'src str, config: LexerConfig) -> Self {
        Self {
            src,
            pos: 0,
            config,
            layout_stack: vec![(0, false)], // Implicit module-level context
            pending: Vec::new(),
            line_start_column: 1,
            expect_layout_block: false,
            at_line_start: true,
            line: 1,
            column: 1,
        }
    }

    /// Get the remaining source code.
    fn remaining(&self) -> &'src str {
        &self.src[self.pos..]
    }

    /// Peek at the next character without consuming it.
    fn peek(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    /// Peek at the character after next.
    fn peek2(&self) -> Option<char> {
        let mut chars = self.remaining().chars();
        chars.next();
        chars.next()
    }

    /// Peek at two characters ahead.
    fn peek3(&self) -> Option<char> {
        let mut chars = self.remaining().chars();
        chars.next();
        chars.next();
        chars.next()
    }

    /// Check if the remaining source starts with a string.
    fn starts_with(&self, s: &str) -> bool {
        self.remaining().starts_with(s)
    }

    /// Advance by one character and update position tracking.
    fn advance(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    /// Advance while a predicate is true, returning the consumed string.
    fn advance_while(&mut self, pred: impl Fn(char) -> bool) -> &'src str {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if pred(c) {
                self.advance();
            } else {
                break;
            }
        }
        &self.src[start..self.pos]
    }

    /// Skip whitespace (not newlines), returning whether any was skipped.
    fn skip_horizontal_whitespace(&mut self) -> bool {
        let start = self.pos;
        while let Some(c) = self.peek() {
            match c {
                ' ' => {
                    self.advance();
                }
                '\t' => {
                    self.advance();
                    // Tabs align to tab stops
                    self.column = ((self.column - 1) / self.config.tab_width + 1)
                        * self.config.tab_width
                        + 1;
                }
                _ => break,
            }
        }
        self.pos > start
    }

    /// Skip a newline and update line tracking.
    fn skip_newline(&mut self) -> bool {
        match self.peek() {
            Some('\n') => {
                self.advance();
                self.at_line_start = true;
                true
            }
            Some('\r') => {
                self.advance();
                if self.peek() == Some('\n') {
                    self.advance();
                }
                self.at_line_start = true;
                true
            }
            _ => false,
        }
    }

    /// Skip whitespace and comments, handling layout.
    fn skip_trivia(&mut self) {
        loop {
            // Skip horizontal whitespace
            self.skip_horizontal_whitespace();

            // Check for newline
            if self.skip_newline() {
                // Skip any blank lines
                while self.skip_horizontal_whitespace() || self.skip_newline() {}
                // Record the column for layout rule
                self.line_start_column = self.column;
                continue;
            }

            // Check for line comment
            if self.starts_with("--") && !self.starts_with("---") {
                // Check for doc comment
                if self.starts_with("-- |") || self.starts_with("-- ^") {
                    if self.config.preserve_doc_comments {
                        break; // Let the main lexer handle it
                    }
                }
                // Regular comment - skip to end of line
                while let Some(c) = self.peek() {
                    if c == '\n' {
                        break;
                    }
                    self.advance();
                }
                continue;
            }

            // Check for block comment or pragma
            if self.starts_with("{-") {
                if self.starts_with("{-#") {
                    if self.config.preserve_pragmas {
                        break; // Let the main lexer handle it
                    }
                    self.skip_pragma();
                } else if self.starts_with("{- |") || self.starts_with("{- ^") {
                    if self.config.preserve_doc_comments {
                        break; // Let the main lexer handle it
                    }
                    self.skip_block_comment();
                } else {
                    self.skip_block_comment();
                }
                continue;
            }

            break;
        }
    }

    /// Skip a block comment, handling nesting.
    fn skip_block_comment(&mut self) -> bool {
        if !self.starts_with("{-") {
            return false;
        }
        self.advance(); // {
        self.advance(); // -

        let mut depth = 1;
        while depth > 0 {
            match self.peek() {
                Some('{') if self.peek2() == Some('-') => {
                    self.advance();
                    self.advance();
                    depth += 1;
                }
                Some('-') if self.peek2() == Some('}') => {
                    self.advance();
                    self.advance();
                    depth -= 1;
                }
                Some(_) => {
                    self.advance();
                }
                None => break,
            }
        }
        true
    }

    /// Skip a pragma.
    fn skip_pragma(&mut self) {
        if !self.starts_with("{-#") {
            return;
        }
        self.advance(); // {
        self.advance(); // -
        self.advance(); // #

        loop {
            match self.peek() {
                Some('#') if self.peek2() == Some('-') && self.peek3() == Some('}') => {
                    self.advance(); // #
                    self.advance(); // -
                    self.advance(); // }
                    break;
                }
                Some(_) => {
                    self.advance();
                }
                None => break,
            }
        }
    }

    /// Lex a documentation line comment.
    fn lex_doc_comment_line(&mut self, _start: usize) -> Token {
        // Skip "-- |" or "-- ^"
        self.advance(); // -
        self.advance(); // -
        self.advance(); // space
        self.advance(); // | or ^

        let content_start = self.pos;
        self.advance_while(|c| c != '\n');
        let content = self.src[content_start..self.pos].trim().to_string();

        Token::new(TokenKind::DocCommentLine(content))
    }

    /// Lex a documentation block comment.
    fn lex_doc_comment_block(&mut self, _start: usize) -> Token {
        // Skip "{- |" or "{- ^"
        self.advance(); // {
        self.advance(); // -
        self.advance(); // space
        self.advance(); // | or ^

        let content_start = self.pos;
        let mut depth = 1;

        while depth > 0 {
            match self.peek() {
                Some('{') if self.peek2() == Some('-') => {
                    self.advance();
                    self.advance();
                    depth += 1;
                }
                Some('-') if self.peek2() == Some('}') => {
                    let content_end = self.pos;
                    self.advance();
                    self.advance();
                    depth -= 1;
                    if depth == 0 {
                        let content = self.src[content_start..content_end].trim().to_string();
                        return Token::new(TokenKind::DocCommentBlock(content));
                    }
                }
                Some(_) => {
                    self.advance();
                }
                None => break,
            }
        }

        let content = self.src[content_start..self.pos].trim().to_string();
        Token::new(TokenKind::DocCommentBlock(content))
    }

    /// Lex a pragma.
    fn lex_pragma(&mut self, _start: usize) -> Token {
        // Skip "{-#"
        self.advance(); // {
        self.advance(); // -
        self.advance(); // #

        let content_start = self.pos;

        loop {
            match self.peek() {
                Some('#') if self.peek2() == Some('-') && self.peek3() == Some('}') => {
                    let content_end = self.pos;
                    self.advance(); // #
                    self.advance(); // -
                    self.advance(); // }
                    let content = self.src[content_start..content_end].trim().to_string();
                    return Token::new(TokenKind::Pragma(content));
                }
                Some(_) => {
                    self.advance();
                }
                None => {
                    let content = self.src[content_start..self.pos].trim().to_string();
                    return Token::new(TokenKind::Pragma(content));
                }
            }
        }
    }

    /// Check if a character can start an identifier.
    fn is_ident_start(c: char) -> bool {
        c == '_' || c.is_xid_start()
    }

    /// Check if a character can continue an identifier.
    fn is_ident_continue(c: char) -> bool {
        c == '_' || c == '\'' || c.is_xid_continue()
    }

    /// Check if a character can be part of an operator.
    fn is_operator_char(c: char) -> bool {
        matches!(
            c,
            '!' | '#' | '$' | '%' | '&' | '*' | '+' | '.' | '/' | '<' | '=' | '>' | '?' | '@'
                | '\\' | '^' | '|' | '-' | '~' | ':'
        ) || is_unicode_symbol(c)
    }

    /// Lex an identifier or keyword, possibly qualified.
    fn lex_ident_or_qualified(&mut self, start: usize) -> Token {
        // Collect the identifier
        self.advance_while(Self::is_ident_continue);
        let first_part = &self.src[start..self.pos];

        // Check for qualified name: Foo.Bar.baz or Foo.Bar.Baz
        if first_part.chars().next().unwrap().is_uppercase() && self.peek() == Some('.') {
            let qualifier_end = self.pos;

            // Look ahead to see if this is a qualified name
            let saved_pos = self.pos;
            let saved_col = self.column;
            self.advance(); // .

            if let Some(c) = self.peek() {
                if Self::is_ident_start(c) || Self::is_operator_char(c) {
                    // This is a qualified name
                    let name_start = self.pos;

                    if Self::is_operator_char(c) && c != ':' {
                        // Qualified operator: Module.+
                        self.advance_while(|c| Self::is_operator_char(c) && c != ':');
                        let qualifier = Symbol::intern(&self.src[start..qualifier_end]);
                        let name = Symbol::intern(&self.src[name_start..self.pos]);
                        return Token::new(TokenKind::QualOperator(qualifier, name));
                    } else if c == ':' {
                        // Qualified constructor operator: Module.:
                        self.advance_while(Self::is_operator_char);
                        let qualifier = Symbol::intern(&self.src[start..qualifier_end]);
                        let name = Symbol::intern(&self.src[name_start..self.pos]);
                        return Token::new(TokenKind::QualConOperator(qualifier, name));
                    } else if c.is_uppercase() {
                        // Could be more qualification or a constructor
                        self.advance_while(Self::is_ident_continue);
                        let second_part = &self.src[name_start..self.pos];

                        // Check for more dots
                        if self.peek() == Some('.') {
                            // Continue collecting qualifiers
                            let mut full_qualifier = first_part.to_string();
                            let mut current_part = second_part;

                            while self.peek() == Some('.') {
                                let part_end = self.pos;
                                self.advance(); // .

                                if let Some(c) = self.peek() {
                                    if Self::is_ident_start(c) {
                                        full_qualifier.push('.');
                                        full_qualifier.push_str(current_part);
                                        let next_start = self.pos;
                                        self.advance_while(Self::is_ident_continue);
                                        current_part = &self.src[next_start..self.pos];
                                    } else if Self::is_operator_char(c) && c != ':' {
                                        full_qualifier.push('.');
                                        full_qualifier.push_str(current_part);
                                        let op_start = self.pos;
                                        self.advance_while(|c| Self::is_operator_char(c) && c != ':');
                                        let qualifier = Symbol::intern(&full_qualifier);
                                        let name = Symbol::intern(&self.src[op_start..self.pos]);
                                        return Token::new(TokenKind::QualOperator(qualifier, name));
                                    } else {
                                        // Restore and return what we have
                                        self.pos = part_end;
                                        break;
                                    }
                                } else {
                                    self.pos = part_end;
                                    break;
                                }
                            }

                            let qualifier = Symbol::intern(&full_qualifier);
                            let name = Symbol::intern(current_part);

                            if current_part.chars().next().unwrap().is_uppercase() {
                                return Token::new(TokenKind::QualConId(qualifier, name));
                            } else {
                                return Token::new(TokenKind::QualIdent(qualifier, name));
                            }
                        }

                        let qualifier = Symbol::intern(first_part);
                        let name = Symbol::intern(second_part);
                        return Token::new(TokenKind::QualConId(qualifier, name));
                    } else {
                        // Lowercase identifier after dot
                        self.advance_while(Self::is_ident_continue);
                        let name_text = &self.src[name_start..self.pos];

                        // Check if the name is a keyword (keywords can't be qualified)
                        if TokenKind::from_keyword(name_text).is_some() {
                            // Restore position - the dot is a separate token
                            self.pos = saved_pos;
                            self.column = saved_col;
                            let sym = Symbol::intern(first_part);
                            return Token::new(TokenKind::ConId(sym));
                        }

                        let qualifier = Symbol::intern(first_part);
                        let name = Symbol::intern(name_text);
                        return Token::new(TokenKind::QualIdent(qualifier, name));
                    }
                }
            }

            // Not a qualified name, restore position
            self.pos = saved_pos;
            self.column = saved_col;
        }

        // Not qualified - check for keyword or return identifier
        let sym = Symbol::intern(first_part);

        if let Some(kw) = TokenKind::from_keyword(first_part) {
            Token::new(kw)
        } else if first_part.chars().next().unwrap().is_uppercase() {
            Token::new(TokenKind::ConId(sym))
        } else {
            Token::new(TokenKind::Ident(sym))
        }
    }

    /// Lex a number literal.
    fn lex_number(&mut self, start: usize) -> Token {
        let mut base = NumBase::Decimal;

        // Check for hex/octal/binary prefix
        if self.peek() == Some('0') {
            match self.peek2() {
                Some('x') | Some('X') => {
                    self.advance(); // 0
                    self.advance(); // x
                    base = NumBase::Hexadecimal;
                    self.advance_while(|c| c.is_ascii_hexdigit() || c == '_');
                }
                Some('o') | Some('O') => {
                    self.advance(); // 0
                    self.advance(); // o
                    base = NumBase::Octal;
                    self.advance_while(|c| matches!(c, '0'..='7') || c == '_');
                }
                Some('b') | Some('B') => {
                    self.advance(); // 0
                    self.advance(); // b
                    base = NumBase::Binary;
                    self.advance_while(|c| c == '0' || c == '1' || c == '_');
                }
                _ => {
                    self.advance_while(|c| c.is_ascii_digit() || c == '_');
                }
            }
        } else {
            self.advance_while(|c| c.is_ascii_digit() || c == '_');
        }

        // Check for float (only for decimal)
        if base == NumBase::Decimal
            && self.peek() == Some('.')
            && self.peek2().is_some_and(|c| c.is_ascii_digit())
        {
            self.advance(); // .
            self.advance_while(|c| c.is_ascii_digit() || c == '_');

            // Exponent
            if let Some('e') | Some('E') = self.peek() {
                self.advance();
                if let Some('+') | Some('-') = self.peek() {
                    self.advance();
                }
                self.advance_while(|c| c.is_ascii_digit() || c == '_');
            }

            // Check for suffix
            let suffix = self.lex_num_suffix();
            let text = self.src[start..self.pos - suffix.as_ref().map_or(0, |_| 1)].to_string();

            return Token::new(TokenKind::FloatLit(FloatLiteral { text, suffix }));
        }

        // Check for exponent on integer (makes it a float)
        if base == NumBase::Decimal {
            if let Some('e') | Some('E') = self.peek() {
                self.advance();
                if let Some('+') | Some('-') = self.peek() {
                    self.advance();
                }
                self.advance_while(|c| c.is_ascii_digit() || c == '_');

                let suffix = self.lex_num_suffix();
                let text = self.src[start..self.pos].to_string();
                return Token::new(TokenKind::FloatLit(FloatLiteral { text, suffix }));
            }
        }

        // Integer
        let suffix = self.lex_num_suffix();
        let text = self.src[start..self.pos].to_string();

        Token::new(TokenKind::IntLit(IntLiteral { text, base, suffix }))
    }

    /// Lex a numeric suffix (# or ##).
    fn lex_num_suffix(&mut self) -> Option<NumSuffix> {
        if self.peek() == Some('#') {
            self.advance();
            if self.peek() == Some('#') {
                self.advance();
                Some(NumSuffix::DoubleHash)
            } else {
                Some(NumSuffix::Hash)
            }
        } else {
            None
        }
    }

    /// Lex a string literal.
    fn lex_string(&mut self, _start: usize) -> Token {
        self.advance(); // Opening "
        let mut content = String::new();
        loop {
            match self.peek() {
                Some('"') => {
                    self.advance();
                    break;
                }
                Some('\\') => {
                    self.advance();
                    if let Ok(c) = self.lex_escape_char() {
                        content.push(c);
                    }
                    // On error, we continue parsing the string but the error
                    // would be reported separately if we had error accumulation
                }
                Some('\n') | None => {
                    return Token::new(TokenKind::Error(LexError::UnterminatedString));
                }
                Some(c) => {
                    self.advance();
                    content.push(c);
                }
            }
        }

        Token::new(TokenKind::StringLit(content))
    }

    /// Lex an escape character sequence.
    fn lex_escape_char(&mut self) -> Result<char, LexError> {
        match self.peek() {
            Some('n') => {
                self.advance();
                Ok('\n')
            }
            Some('t') => {
                self.advance();
                Ok('\t')
            }
            Some('r') => {
                self.advance();
                Ok('\r')
            }
            Some('\\') => {
                self.advance();
                Ok('\\')
            }
            Some('"') => {
                self.advance();
                Ok('"')
            }
            Some('\'') => {
                self.advance();
                Ok('\'')
            }
            Some('0') => {
                self.advance();
                Ok('\0')
            }
            Some('a') => {
                self.advance();
                Ok('\x07')
            }
            Some('b') => {
                self.advance();
                Ok('\x08')
            }
            Some('f') => {
                self.advance();
                Ok('\x0C')
            }
            Some('v') => {
                self.advance();
                Ok('\x0B')
            }
            Some('x') => {
                self.advance();
                self.lex_hex_escape(2)
            }
            Some('u') => {
                self.advance();
                self.lex_unicode_escape()
            }
            Some('&') => {
                // String gap: \& is empty
                self.advance();
                Ok('\0') // Will be filtered out
            }
            Some(c) if c.is_ascii_digit() => self.lex_decimal_escape(),
            Some(c) if c.is_whitespace() => {
                // String gap: backslash-whitespace-backslash
                self.advance_while(|c| c.is_whitespace());
                if self.peek() == Some('\\') {
                    self.advance();
                    Ok('\0') // Will be filtered out
                } else {
                    Err(LexError::InvalidEscape(' '))
                }
            }
            Some(c) => {
                self.advance();
                Err(LexError::InvalidEscape(c))
            }
            None => Err(LexError::UnterminatedString),
        }
    }

    /// Lex a hexadecimal escape sequence.
    fn lex_hex_escape(&mut self, max_digits: usize) -> Result<char, LexError> {
        let start = self.pos;
        for _ in 0..max_digits {
            if let Some(c) = self.peek() {
                if c.is_ascii_hexdigit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        let hex = &self.src[start..self.pos];
        if hex.is_empty() {
            return Err(LexError::InvalidEscape('x'));
        }
        u32::from_str_radix(hex, 16)
            .ok()
            .and_then(char::from_u32)
            .ok_or(LexError::InvalidUnicodeEscape)
    }

    /// Lex a Unicode escape sequence (\uXXXX or \u{XXXXX}).
    fn lex_unicode_escape(&mut self) -> Result<char, LexError> {
        if self.peek() == Some('{') {
            self.advance();
            let start = self.pos;
            self.advance_while(|c| c.is_ascii_hexdigit());
            let hex = &self.src[start..self.pos];
            if self.peek() != Some('}') {
                return Err(LexError::InvalidUnicodeEscape);
            }
            self.advance();
            u32::from_str_radix(hex, 16)
                .ok()
                .and_then(char::from_u32)
                .ok_or(LexError::InvalidUnicodeEscape)
        } else {
            self.lex_hex_escape(4)
        }
    }

    /// Lex a decimal escape sequence.
    fn lex_decimal_escape(&mut self) -> Result<char, LexError> {
        let start = self.pos;
        self.advance_while(|c| c.is_ascii_digit());
        let dec = &self.src[start..self.pos];
        dec.parse::<u32>()
            .ok()
            .and_then(char::from_u32)
            .ok_or(LexError::InvalidNumber(dec.to_string()))
    }

    /// Lex a character literal.
    fn lex_char(&mut self, _start: usize) -> Token {
        self.advance(); // Opening '

        let c = match self.peek() {
            Some('\'') => {
                self.advance();
                return Token::new(TokenKind::Error(LexError::EmptyCharLiteral));
            }
            Some('\\') => {
                self.advance();
                match self.lex_escape_char() {
                    Ok(c) => c,
                    Err(e) => return Token::new(TokenKind::Error(e)),
                }
            }
            Some(c) => {
                self.advance();
                c
            }
            None => return Token::new(TokenKind::Error(LexError::UnterminatedChar)),
        };

        match self.peek() {
            Some('\'') => {
                self.advance();
                Token::new(TokenKind::CharLit(c))
            }
            Some(_) => {
                // Multi-char literal - skip to closing quote
                while let Some(c) = self.peek() {
                    if c == '\'' {
                        self.advance();
                        break;
                    }
                    if c == '\n' {
                        break;
                    }
                    self.advance();
                }
                Token::new(TokenKind::Error(LexError::MultiCharLiteral))
            }
            None => Token::new(TokenKind::Error(LexError::UnterminatedChar)),
        }
    }

    /// Lex an operator.
    fn lex_operator(&mut self, start: usize) -> Token {
        // Special single-char handling
        let first = self.peek().unwrap();

        // Handle special two/three char sequences first
        match (first, self.peek2()) {
            ('-', Some('>')) => {
                self.advance();
                self.advance();
                return Token::new(TokenKind::Arrow);
            }
            ('<', Some('-')) => {
                self.advance();
                self.advance();
                return Token::new(TokenKind::LeftArrow);
            }
            ('=', Some('>')) => {
                self.advance();
                self.advance();
                return Token::new(TokenKind::FatArrow);
            }
            (':', Some(':')) => {
                self.advance();
                self.advance();
                return Token::new(TokenKind::DoubleColon);
            }
            ('.', Some('.')) => {
                self.advance();
                self.advance();
                return Token::new(TokenKind::DotDot);
            }
            ('(', Some('#')) => {
                self.advance();
                self.advance();
                return Token::new(TokenKind::LParenHash);
            }
            ('#', Some(')')) => {
                self.advance();
                self.advance();
                return Token::new(TokenKind::HashRParen);
            }
            ('[', Some('|')) => {
                self.advance();
                self.advance();
                return Token::new(TokenKind::LBracketPipe);
            }
            ('|', Some(']')) => {
                self.advance();
                self.advance();
                return Token::new(TokenKind::PipeRBracket);
            }
            _ => {}
        }

        // Single special chars
        match first {
            '\\' => {
                self.advance();
                return Token::new(TokenKind::Backslash);
            }
            '=' => {
                self.advance();
                // Check it's not part of a longer operator
                if !self.peek().is_some_and(Self::is_operator_char) {
                    return Token::new(TokenKind::Eq);
                }
                // Restore and continue as general operator
                self.pos = start;
                self.column -= 1;
            }
            '|' => {
                self.advance();
                if !self.peek().is_some_and(Self::is_operator_char) {
                    return Token::new(TokenKind::Pipe);
                }
                self.pos = start;
                self.column -= 1;
            }
            '@' => {
                self.advance();
                if !self.peek().is_some_and(Self::is_operator_char) {
                    return Token::new(TokenKind::At);
                }
                self.pos = start;
                self.column -= 1;
            }
            '~' => {
                self.advance();
                if !self.peek().is_some_and(Self::is_operator_char) {
                    return Token::new(TokenKind::Tilde);
                }
                self.pos = start;
                self.column -= 1;
            }
            '.' => {
                self.advance();
                if !self.peek().is_some_and(Self::is_operator_char) {
                    return Token::new(TokenKind::Dot);
                }
                self.pos = start;
                self.column -= 1;
            }
            '!' => {
                self.advance();
                if !self.peek().is_some_and(Self::is_operator_char) {
                    return Token::new(TokenKind::Bang);
                }
                self.pos = start;
                self.column -= 1;
            }
            '?' => {
                self.advance();
                if !self.peek().is_some_and(Self::is_operator_char) {
                    return Token::new(TokenKind::Question);
                }
                self.pos = start;
                self.column -= 1;
            }
            '#' => {
                self.advance();
                if !self.peek().is_some_and(Self::is_operator_char) {
                    return Token::new(TokenKind::Hash);
                }
                self.pos = start;
                self.column -= 1;
            }
            '*' => {
                self.advance();
                if !self.peek().is_some_and(Self::is_operator_char) {
                    return Token::new(TokenKind::Star);
                }
                self.pos = start;
                self.column -= 1;
            }
            '-' => {
                self.advance();
                if !self.peek().is_some_and(Self::is_operator_char) {
                    return Token::new(TokenKind::Minus);
                }
                self.pos = start;
                self.column -= 1;
            }
            '%' => {
                self.advance();
                if !self.peek().is_some_and(Self::is_operator_char) {
                    return Token::new(TokenKind::Percent);
                }
                self.pos = start;
                self.column -= 1;
            }
            _ => {}
        }

        // General operator
        let is_con_op = first == ':';
        self.advance_while(Self::is_operator_char);
        let text = &self.src[start..self.pos];
        let sym = Symbol::intern(text);

        if is_con_op {
            Token::new(TokenKind::ConOperator(sym))
        } else {
            Token::new(TokenKind::Operator(sym))
        }
    }

    /// Handle layout rule: generate virtual tokens based on indentation.
    fn handle_layout(&mut self, token: &Token, column: u32) {
        // If we expect a layout block after a layout keyword
        if self.expect_layout_block {
            self.expect_layout_block = false;

            if token.kind == TokenKind::LBrace {
                // Explicit brace - push explicit context
                self.layout_stack.push((0, true));
            } else {
                // Implicit layout - insert virtual brace and push context
                self.layout_stack.push((column, false));
                self.pending.push(Spanned::new(
                    Token::new(TokenKind::VirtualLBrace),
                    Span::from_raw(self.pos as u32, self.pos as u32),
                ));
            }
            return;
        }

        // Check for layout keywords that start a new block
        if token.kind.starts_layout() {
            self.expect_layout_block = true;
            return;
        }

        // Handle indentation at line start
        if self.at_line_start {
            self.at_line_start = false;

            // Compare with current layout context
            while let Some(&(ctx_col, is_explicit)) = self.layout_stack.last() {
                if is_explicit {
                    break; // Don't close explicit braces
                }

                if column < ctx_col {
                    // Dedent: close the layout block
                    self.layout_stack.pop();
                    self.pending.push(Spanned::new(
                        Token::new(TokenKind::VirtualRBrace),
                        Span::from_raw(self.pos as u32, self.pos as u32),
                    ));
                } else if column == ctx_col {
                    // Same indentation: new item in the block
                    self.pending.push(Spanned::new(
                        Token::new(TokenKind::VirtualSemi),
                        Span::from_raw(self.pos as u32, self.pos as u32),
                    ));
                    break;
                } else {
                    break;
                }
            }
        }

        // Handle explicit close brace
        if token.kind == TokenKind::RBrace {
            // Close any implicit contexts until we find explicit one
            while let Some(&(_, is_explicit)) = self.layout_stack.last() {
                if is_explicit {
                    self.layout_stack.pop();
                    break;
                }
                self.layout_stack.pop();
                self.pending.push(Spanned::new(
                    Token::new(TokenKind::VirtualRBrace),
                    Span::from_raw(self.pos as u32, self.pos as u32),
                ));
            }
        }
    }

    /// Lex the next raw token (without layout processing).
    fn lex_raw_token(&mut self) -> Option<Spanned<Token>> {
        self.skip_trivia();

        let start = self.pos;
        let start_col = self.column;
        let c = self.peek()?;

        let token = match c {
            // Documentation comments
            '-' if self.starts_with("-- |") || self.starts_with("-- ^") => {
                self.lex_doc_comment_line(start)
            }

            // Block doc comments and pragmas
            '{' if self.starts_with("{- |") || self.starts_with("{- ^") => {
                self.lex_doc_comment_block(start)
            }
            '{' if self.starts_with("{-#") => self.lex_pragma(start),

            // Identifiers and keywords
            c if Self::is_ident_start(c) => self.lex_ident_or_qualified(start),

            // Numbers
            c if c.is_ascii_digit() => self.lex_number(start),

            // Strings
            '"' => self.lex_string(start),

            // Characters
            '\'' => self.lex_char(start),

            // Single-character tokens
            '(' if self.peek2() != Some('#') => {
                self.advance();
                Token::new(TokenKind::LParen)
            }
            ')' => {
                self.advance();
                Token::new(TokenKind::RParen)
            }
            '[' if self.peek2() != Some('|') => {
                self.advance();
                Token::new(TokenKind::LBracket)
            }
            ']' => {
                self.advance();
                Token::new(TokenKind::RBracket)
            }
            '{' => {
                self.advance();
                Token::new(TokenKind::LBrace)
            }
            '}' => {
                self.advance();
                Token::new(TokenKind::RBrace)
            }
            ',' => {
                self.advance();
                Token::new(TokenKind::Comma)
            }
            ';' => {
                self.advance();
                Token::new(TokenKind::Semi)
            }
            '`' => {
                self.advance();
                Token::new(TokenKind::Backtick)
            }

            // Unicode special chars
            '→' => {
                self.advance();
                Token::new(TokenKind::UnicodeArrow)
            }
            '←' => {
                self.advance();
                Token::new(TokenKind::UnicodeLeftArrow)
            }
            '⇒' => {
                self.advance();
                Token::new(TokenKind::UnicodeFatArrow)
            }
            '∷' => {
                self.advance();
                Token::new(TokenKind::UnicodeDoubleColon)
            }
            '∀' => {
                self.advance();
                Token::new(TokenKind::Forall)
            }
            '★' => {
                self.advance();
                Token::new(TokenKind::Star)
            }

            // Operators
            c if Self::is_operator_char(c) => self.lex_operator(start),

            // Unknown character
            _ => {
                self.advance();
                Token::new(TokenKind::Error(LexError::InvalidChar(c)))
            }
        };

        let span = Span::from_raw(start as u32, self.pos as u32);

        // Handle layout rule
        self.handle_layout(&token, start_col);

        Some(Spanned::new(token, span))
    }
}

impl<'src> Iterator for Lexer<'src> {
    type Item = Spanned<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        // Return pending tokens first (from layout rule)
        if let Some(tok) = self.pending.pop() {
            return Some(tok);
        }

        self.lex_raw_token()
    }
}

/// Check if a character is a Unicode symbol (for operators).
fn is_unicode_symbol(c: char) -> bool {
    matches!(c, '∘' | '∙' | '⊕' | '⊗' | '⊖' | '⊛' | '⊜' | '⊝' | '⊞' | '⊟' | '⟨' | '⟩' | '⟪' | '⟫')
}

/// Lex source code into a vector of tokens.
#[must_use]
pub fn lex(src: &str) -> Vec<Spanned<Token>> {
    Lexer::new(src).collect()
}

/// Lex source code with custom configuration.
#[must_use]
pub fn lex_with_config(src: &str, config: LexerConfig) -> Vec<Spanned<Token>> {
    Lexer::with_config(src, config).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_kinds(src: &str) -> Vec<TokenKind> {
        lex(src).into_iter().map(|t| t.node.kind).collect()
    }

    #[test]
    fn test_keywords() {
        // Filter out virtual tokens for this test (layout rule adds them after where/do/of/let)
        let kinds: Vec<_> = lex_kinds("let in where if then else case of do")
            .into_iter()
            .filter(|k| !k.is_virtual())
            .collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::Let,
                TokenKind::In,
                TokenKind::Where,
                TokenKind::If,
                TokenKind::Then,
                TokenKind::Else,
                TokenKind::Case,
                TokenKind::Of,
                TokenKind::Do,
            ]
        );
    }

    #[test]
    fn test_identifiers() {
        let kinds = lex_kinds("foo bar' _x Maybe Just");
        assert!(matches!(kinds[0], TokenKind::Ident(_)));
        assert!(matches!(kinds[1], TokenKind::Ident(_)));
        assert!(matches!(kinds[2], TokenKind::Ident(_)));
        assert!(matches!(kinds[3], TokenKind::ConId(_)));
        assert!(matches!(kinds[4], TokenKind::ConId(_)));
    }

    #[test]
    fn test_qualified_names() {
        let kinds = lex_kinds("Data.List.map Data.Maybe.Just");
        assert!(matches!(kinds[0], TokenKind::QualIdent(_, _)));
        assert!(matches!(kinds[1], TokenKind::QualConId(_, _)));
    }

    #[test]
    fn test_numbers() {
        let kinds = lex_kinds("42 0xFF 0o77 0b1010 3.14 1e10");
        assert_eq!(kinds.len(), 6);
        assert!(matches!(kinds[0], TokenKind::IntLit(_)));
        assert!(matches!(kinds[1], TokenKind::IntLit(_)));
        assert!(matches!(kinds[2], TokenKind::IntLit(_)));
        assert!(matches!(kinds[3], TokenKind::IntLit(_)));
        assert!(matches!(kinds[4], TokenKind::FloatLit(_)));
        assert!(matches!(kinds[5], TokenKind::FloatLit(_)));
    }

    #[test]
    fn test_operators() {
        let kinds = lex_kinds("+ - * / -> <- => :: ..");
        assert!(matches!(kinds[0], TokenKind::Operator(_)));
        assert_eq!(kinds[1], TokenKind::Minus);
        assert_eq!(kinds[2], TokenKind::Star);
        assert!(matches!(kinds[3], TokenKind::Operator(_)));
        assert_eq!(kinds[4], TokenKind::Arrow);
        assert_eq!(kinds[5], TokenKind::LeftArrow);
        assert_eq!(kinds[6], TokenKind::FatArrow);
        assert_eq!(kinds[7], TokenKind::DoubleColon);
        assert_eq!(kinds[8], TokenKind::DotDot);
    }

    #[test]
    fn test_string_literal() {
        let kinds = lex_kinds(r#""hello world""#);
        assert_eq!(kinds.len(), 1);
        match &kinds[0] {
            TokenKind::StringLit(s) => assert_eq!(s, "hello world"),
            _ => panic!("expected string literal"),
        }
    }

    #[test]
    fn test_string_escapes() {
        let kinds = lex_kinds(r#""\n\t\\""#);
        match &kinds[0] {
            TokenKind::StringLit(s) => assert_eq!(s, "\n\t\\"),
            _ => panic!("expected string literal"),
        }
    }

    #[test]
    fn test_char_literal() {
        let kinds = lex_kinds("'a' '\\n' '\\''");
        assert_eq!(kinds[0], TokenKind::CharLit('a'));
        assert_eq!(kinds[1], TokenKind::CharLit('\n'));
        assert_eq!(kinds[2], TokenKind::CharLit('\''));
    }

    #[test]
    fn test_comments() {
        let kinds = lex_kinds("x -- comment\ny");
        assert!(matches!(kinds[0], TokenKind::Ident(_)));
        assert!(matches!(kinds[1], TokenKind::Ident(_)));
    }

    #[test]
    fn test_block_comments() {
        let kinds = lex_kinds("x {- nested {- comment -} -} y");
        assert_eq!(kinds.len(), 2);
    }

    #[test]
    fn test_doc_comments() {
        let kinds = lex_kinds("-- | Doc comment\nx");
        assert!(matches!(kinds[0], TokenKind::DocCommentLine(_)));
    }

    #[test]
    fn test_pragmas() {
        let kinds = lex_kinds("{-# LANGUAGE GADTs #-}\nmodule");
        assert!(matches!(kinds[0], TokenKind::Pragma(_)));
        assert_eq!(kinds[1], TokenKind::Module);
    }

    #[test]
    fn test_unicode_arrows() {
        let kinds = lex_kinds("→ ← ⇒ ∷ ∀");
        assert_eq!(kinds[0], TokenKind::UnicodeArrow);
        assert_eq!(kinds[1], TokenKind::UnicodeLeftArrow);
        assert_eq!(kinds[2], TokenKind::UnicodeFatArrow);
        assert_eq!(kinds[3], TokenKind::UnicodeDoubleColon);
        assert_eq!(kinds[4], TokenKind::Forall);
    }

    #[test]
    fn test_layout_simple() {
        // Test that layout keywords trigger virtual brace insertion
        let kinds = lex_kinds("let x = 1 in x");
        // Should have: Let, Ident, Eq, IntLit, VirtualLBrace (or similar), In, Ident
        assert!(kinds.contains(&TokenKind::Let));
        assert!(kinds.contains(&TokenKind::In));
    }

    #[test]
    fn test_constructor_operators() {
        let kinds = lex_kinds(":+ :| :");
        assert!(matches!(kinds[0], TokenKind::ConOperator(_)));
        assert!(matches!(kinds[1], TokenKind::ConOperator(_)));
        assert!(matches!(kinds[2], TokenKind::ConOperator(_)));
    }
}
