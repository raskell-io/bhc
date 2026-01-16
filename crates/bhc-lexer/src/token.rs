//! Token definitions for the BHC lexer.

use bhc_intern::Symbol;
use std::fmt;

/// A token produced by the lexer.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    /// The kind of token.
    pub kind: TokenKind,
}

impl Token {
    /// Create a new token with the given kind.
    #[must_use]
    pub const fn new(kind: TokenKind) -> Self {
        Self { kind }
    }
}

/// The kind of token.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // =========================================================================
    // Keywords (Haskell 2010 + GHC extensions + H26)
    // =========================================================================
    /// `case`
    Case,
    /// `class`
    Class,
    /// `data`
    Data,
    /// `default`
    Default,
    /// `deriving`
    Deriving,
    /// `do`
    Do,
    /// `else`
    Else,
    /// `forall` / `∀`
    Forall,
    /// `foreign`
    Foreign,
    /// `hiding`
    Hiding,
    /// `if`
    If,
    /// `import`
    Import,
    /// `in`
    In,
    /// `infix`
    Infix,
    /// `infixl`
    Infixl,
    /// `infixr`
    Infixr,
    /// `instance`
    Instance,
    /// `let`
    Let,
    /// `mdo` (recursive do)
    Mdo,
    /// `module`
    Module,
    /// `newtype`
    Newtype,
    /// `of`
    Of,
    /// `proc` (arrow notation)
    Proc,
    /// `qualified`
    Qualified,
    /// `rec` (recursive bindings)
    Rec,
    /// `then`
    Then,
    /// `type`
    Type,
    /// `where`
    Where,

    // GHC extension keywords
    /// `as` (import qualifier)
    As,
    /// `family` (type families)
    Family,
    /// `pattern` (pattern synonyms)
    Pattern,
    /// `role` (role annotations)
    Role,
    /// `stock` (deriving strategy)
    Stock,
    /// `anyclass` (deriving strategy)
    Anyclass,
    /// `newtype` as deriving strategy (deriving via)
    Via,

    // H26 extension keywords
    /// `lazy` (laziness annotation)
    Lazy,
    /// `strict` (strictness annotation)
    Strict,
    /// `linear` (linear types)
    Linear,
    /// `tensor` (tensor type annotation)
    Tensor,

    // =========================================================================
    // Identifiers
    // =========================================================================
    /// A lowercase identifier: `foo`, `bar'`, `_x`
    Ident(Symbol),
    /// An uppercase identifier (constructor/type name): `Just`, `Maybe`
    ConId(Symbol),
    /// A qualified lowercase identifier: `Data.List.map`
    QualIdent(Symbol, Symbol),
    /// A qualified uppercase identifier: `Data.Maybe.Just`
    QualConId(Symbol, Symbol),
    /// An operator symbol: `+`, `>>=`, `<$>`
    Operator(Symbol),
    /// A qualified operator: `Data.List.\\`
    QualOperator(Symbol, Symbol),
    /// A constructor operator: `:+`, `:|`
    ConOperator(Symbol),
    /// A qualified constructor operator: `Data.List.:`
    QualConOperator(Symbol, Symbol),

    // =========================================================================
    // Literals
    // =========================================================================
    /// An integer literal: `42`, `0xFF`, `0o77`, `0b1010`
    IntLit(IntLiteral),
    /// A floating-point literal: `3.14`, `1e10`, `2.5e-3`
    FloatLit(FloatLiteral),
    /// A character literal: `'a'`, `'\n'`
    CharLit(char),
    /// A string literal: `"hello"`
    StringLit(String),
    /// A raw/multi-line string literal (H26): `r#"..."#`
    RawStringLit(String),

    // =========================================================================
    // Punctuation
    // =========================================================================
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `(#` (unboxed tuple open)
    LParenHash,
    /// `#)` (unboxed tuple close)
    HashRParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `[|` (quasiquote open)
    LBracketPipe,
    /// `|]` (quasiquote close)
    PipeRBracket,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `,`
    Comma,
    /// `;`
    Semi,
    /// `` ` ``
    Backtick,
    /// `_`
    Underscore,

    // =========================================================================
    // Special Operators / Punctuation
    // =========================================================================
    /// `=`
    Eq,
    /// `|`
    Pipe,
    /// `\` (lambda)
    Backslash,
    /// `->`
    Arrow,
    /// `→` (Unicode arrow)
    UnicodeArrow,
    /// `<-`
    LeftArrow,
    /// `←` (Unicode left arrow)
    UnicodeLeftArrow,
    /// `=>`
    FatArrow,
    /// `⇒` (Unicode fat arrow)
    UnicodeFatArrow,
    /// `::`
    DoubleColon,
    /// `∷` (Unicode double colon)
    UnicodeDoubleColon,
    /// `..`
    DotDot,
    /// `.`
    Dot,
    /// `@`
    At,
    /// `~`
    Tilde,
    /// `!`
    Bang,
    /// `?` (implicit parameters)
    Question,
    /// `#` (MagicHash)
    Hash,
    /// `*` / `★` (kind star)
    Star,
    /// `-` (minus, special for sections)
    Minus,
    /// `%` (linear arrow multiplicity)
    Percent,

    // =========================================================================
    // Layout tokens (inserted by layout rule)
    // =========================================================================
    /// Virtual `{` from layout rule.
    VirtualLBrace,
    /// Virtual `}` from layout rule.
    VirtualRBrace,
    /// Virtual `;` from layout rule.
    VirtualSemi,

    // =========================================================================
    // Comments (preserved for documentation)
    // =========================================================================
    /// Line documentation comment: `-- | ...`
    DocCommentLine(String),
    /// Block documentation comment: `{- | ... -}`
    DocCommentBlock(String),
    /// Pragma: `{-# ... #-}`
    Pragma(String),

    // =========================================================================
    // Special
    // =========================================================================
    /// End of file.
    Eof,
    /// Lexer error with message.
    Error(LexError),
}

/// Integer literal with base information.
#[derive(Clone, Debug, PartialEq)]
pub struct IntLiteral {
    /// The raw text of the literal.
    pub text: String,
    /// The numeric base.
    pub base: NumBase,
    /// Optional type suffix (e.g., `Int#`).
    pub suffix: Option<NumSuffix>,
}

impl IntLiteral {
    /// Parse the integer value.
    pub fn parse(&self) -> Option<i128> {
        let clean: String = self.text.chars().filter(|c| *c != '_').collect();
        let radix = match self.base {
            NumBase::Decimal => 10,
            NumBase::Hexadecimal => 16,
            NumBase::Octal => 8,
            NumBase::Binary => 2,
        };
        // Skip the prefix (0x, 0o, 0b)
        let digits = match self.base {
            NumBase::Decimal => &clean[..],
            _ => &clean[2..],
        };
        i128::from_str_radix(digits, radix).ok()
    }
}

/// Floating-point literal.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatLiteral {
    /// The raw text of the literal.
    pub text: String,
    /// Optional type suffix.
    pub suffix: Option<NumSuffix>,
}

impl FloatLiteral {
    /// Parse the float value.
    pub fn parse(&self) -> Option<f64> {
        let clean: String = self.text.chars().filter(|c| *c != '_').collect();
        clean.parse().ok()
    }
}

/// Numeric base for integer literals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumBase {
    /// Base 10: `42`
    Decimal,
    /// Base 16: `0xFF`
    Hexadecimal,
    /// Base 8: `0o77`
    Octal,
    /// Base 2: `0b1010`
    Binary,
}

/// Type suffix for numeric literals (MagicHash).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumSuffix {
    /// `#` - unboxed
    Hash,
    /// `##` - unboxed double-width
    DoubleHash,
}

/// Lexer error kinds.
#[derive(Clone, Debug, PartialEq)]
pub enum LexError {
    /// Unterminated string literal.
    UnterminatedString,
    /// Unterminated character literal.
    UnterminatedChar,
    /// Unterminated block comment.
    UnterminatedBlockComment,
    /// Invalid character in source.
    InvalidChar(char),
    /// Invalid escape sequence.
    InvalidEscape(char),
    /// Invalid numeric literal.
    InvalidNumber(String),
    /// Empty character literal.
    EmptyCharLiteral,
    /// Multi-character character literal.
    MultiCharLiteral,
    /// Invalid Unicode escape.
    InvalidUnicodeEscape,
    /// Tab character in indentation (configurable warning).
    TabInIndentation,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnterminatedString => write!(f, "unterminated string literal"),
            Self::UnterminatedChar => write!(f, "unterminated character literal"),
            Self::UnterminatedBlockComment => write!(f, "unterminated block comment"),
            Self::InvalidChar(c) => write!(f, "invalid character: {:?}", c),
            Self::InvalidEscape(c) => write!(f, "invalid escape sequence: \\{}", c),
            Self::InvalidNumber(s) => write!(f, "invalid numeric literal: {}", s),
            Self::EmptyCharLiteral => write!(f, "empty character literal"),
            Self::MultiCharLiteral => write!(f, "character literal contains multiple characters"),
            Self::InvalidUnicodeEscape => write!(f, "invalid Unicode escape sequence"),
            Self::TabInIndentation => write!(f, "tab character in indentation"),
        }
    }
}

impl TokenKind {
    /// Check if this is a keyword.
    #[must_use]
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            Self::Case
                | Self::Class
                | Self::Data
                | Self::Default
                | Self::Deriving
                | Self::Do
                | Self::Else
                | Self::Forall
                | Self::Foreign
                | Self::Hiding
                | Self::If
                | Self::Import
                | Self::In
                | Self::Infix
                | Self::Infixl
                | Self::Infixr
                | Self::Instance
                | Self::Let
                | Self::Mdo
                | Self::Module
                | Self::Newtype
                | Self::Of
                | Self::Proc
                | Self::Qualified
                | Self::Rec
                | Self::Then
                | Self::Type
                | Self::Where
                | Self::As
                | Self::Family
                | Self::Pattern
                | Self::Role
                | Self::Stock
                | Self::Anyclass
                | Self::Via
                | Self::Lazy
                | Self::Strict
                | Self::Linear
                | Self::Tensor
        )
    }

    /// Check if this is a literal.
    #[must_use]
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            Self::IntLit(_)
                | Self::FloatLit(_)
                | Self::CharLit(_)
                | Self::StringLit(_)
                | Self::RawStringLit(_)
        )
    }

    /// Check if this token can start a layout block.
    #[must_use]
    pub fn starts_layout(&self) -> bool {
        matches!(self, Self::Where | Self::Let | Self::Do | Self::Of | Self::Mdo)
    }

    /// Check if this is a virtual (layout-inserted) token.
    #[must_use]
    pub fn is_virtual(&self) -> bool {
        matches!(
            self,
            Self::VirtualLBrace | Self::VirtualRBrace | Self::VirtualSemi
        )
    }

    /// Check if this is a documentation comment.
    #[must_use]
    pub fn is_doc_comment(&self) -> bool {
        matches!(self, Self::DocCommentLine(_) | Self::DocCommentBlock(_))
    }

    /// Check if this is an opening bracket of any kind.
    #[must_use]
    pub fn is_open_bracket(&self) -> bool {
        matches!(
            self,
            Self::LParen
                | Self::LParenHash
                | Self::LBracket
                | Self::LBracketPipe
                | Self::LBrace
                | Self::VirtualLBrace
        )
    }

    /// Check if this is a closing bracket of any kind.
    #[must_use]
    pub fn is_close_bracket(&self) -> bool {
        matches!(
            self,
            Self::RParen
                | Self::HashRParen
                | Self::RBracket
                | Self::PipeRBracket
                | Self::RBrace
                | Self::VirtualRBrace
        )
    }

    /// Get the name of the token for error messages.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            // Keywords
            Self::Case => "`case`",
            Self::Class => "`class`",
            Self::Data => "`data`",
            Self::Default => "`default`",
            Self::Deriving => "`deriving`",
            Self::Do => "`do`",
            Self::Else => "`else`",
            Self::Forall => "`forall`",
            Self::Foreign => "`foreign`",
            Self::Hiding => "`hiding`",
            Self::If => "`if`",
            Self::Import => "`import`",
            Self::In => "`in`",
            Self::Infix => "`infix`",
            Self::Infixl => "`infixl`",
            Self::Infixr => "`infixr`",
            Self::Instance => "`instance`",
            Self::Let => "`let`",
            Self::Mdo => "`mdo`",
            Self::Module => "`module`",
            Self::Newtype => "`newtype`",
            Self::Of => "`of`",
            Self::Proc => "`proc`",
            Self::Qualified => "`qualified`",
            Self::Rec => "`rec`",
            Self::Then => "`then`",
            Self::Type => "`type`",
            Self::Where => "`where`",
            Self::As => "`as`",
            Self::Family => "`family`",
            Self::Pattern => "`pattern`",
            Self::Role => "`role`",
            Self::Stock => "`stock`",
            Self::Anyclass => "`anyclass`",
            Self::Via => "`via`",
            Self::Lazy => "`lazy`",
            Self::Strict => "`strict`",
            Self::Linear => "`linear`",
            Self::Tensor => "`tensor`",

            // Identifiers
            Self::Ident(_) => "identifier",
            Self::ConId(_) => "constructor",
            Self::QualIdent(_, _) => "qualified identifier",
            Self::QualConId(_, _) => "qualified constructor",
            Self::Operator(_) => "operator",
            Self::QualOperator(_, _) => "qualified operator",
            Self::ConOperator(_) => "constructor operator",
            Self::QualConOperator(_, _) => "qualified constructor operator",

            // Literals
            Self::IntLit(_) => "integer literal",
            Self::FloatLit(_) => "float literal",
            Self::CharLit(_) => "character literal",
            Self::StringLit(_) => "string literal",
            Self::RawStringLit(_) => "raw string literal",

            // Punctuation
            Self::LParen => "`(`",
            Self::RParen => "`)`",
            Self::LParenHash => "`(#`",
            Self::HashRParen => "`#)`",
            Self::LBracket => "`[`",
            Self::RBracket => "`]`",
            Self::LBracketPipe => "`[|`",
            Self::PipeRBracket => "`|]`",
            Self::LBrace => "`{`",
            Self::RBrace => "`}`",
            Self::Comma => "`,`",
            Self::Semi => "`;`",
            Self::Backtick => "`` ` ``",
            Self::Underscore => "`_`",

            // Special operators
            Self::Eq => "`=`",
            Self::Pipe => "`|`",
            Self::Backslash => "`\\`",
            Self::Arrow | Self::UnicodeArrow => "`->`",
            Self::LeftArrow | Self::UnicodeLeftArrow => "`<-`",
            Self::FatArrow | Self::UnicodeFatArrow => "`=>`",
            Self::DoubleColon | Self::UnicodeDoubleColon => "`::`",
            Self::DotDot => "`..`",
            Self::Dot => "`.`",
            Self::At => "`@`",
            Self::Tilde => "`~`",
            Self::Bang => "`!`",
            Self::Question => "`?`",
            Self::Hash => "`#`",
            Self::Star => "`*`",
            Self::Minus => "`-`",
            Self::Percent => "`%`",

            // Layout
            Self::VirtualLBrace => "layout `{`",
            Self::VirtualRBrace => "layout `}`",
            Self::VirtualSemi => "layout `;`",

            // Comments
            Self::DocCommentLine(_) => "documentation comment",
            Self::DocCommentBlock(_) => "documentation block comment",
            Self::Pragma(_) => "pragma",

            // Special
            Self::Eof => "end of file",
            Self::Error(_) => "error",
        }
    }

    /// Convert from a keyword string to TokenKind, if it's a keyword.
    #[must_use]
    pub fn from_keyword(s: &str) -> Option<Self> {
        Some(match s {
            "case" => Self::Case,
            "class" => Self::Class,
            "data" => Self::Data,
            "default" => Self::Default,
            "deriving" => Self::Deriving,
            "do" => Self::Do,
            "else" => Self::Else,
            "forall" => Self::Forall,
            "foreign" => Self::Foreign,
            "hiding" => Self::Hiding,
            "if" => Self::If,
            "import" => Self::Import,
            "in" => Self::In,
            "infix" => Self::Infix,
            "infixl" => Self::Infixl,
            "infixr" => Self::Infixr,
            "instance" => Self::Instance,
            "let" => Self::Let,
            "mdo" => Self::Mdo,
            "module" => Self::Module,
            "newtype" => Self::Newtype,
            "of" => Self::Of,
            "proc" => Self::Proc,
            "qualified" => Self::Qualified,
            "rec" => Self::Rec,
            "then" => Self::Then,
            "type" => Self::Type,
            "where" => Self::Where,
            "as" => Self::As,
            "family" => Self::Family,
            "pattern" => Self::Pattern,
            "role" => Self::Role,
            "stock" => Self::Stock,
            "anyclass" => Self::Anyclass,
            "via" => Self::Via,
            "lazy" => Self::Lazy,
            "strict" => Self::Strict,
            "linear" => Self::Linear,
            "tensor" => Self::Tensor,
            "_" => Self::Underscore,
            _ => return None,
        })
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}
