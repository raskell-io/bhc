//! Built-in CPP preprocessor for Haskell source files.
//!
//! Implements "traditional CPP" mode (as used by GHC): no token pasting (`##`),
//! no stringification (`#`), no trigraphs. Supports `#ifdef`, `#if`, `#define`,
//! `#elif`, `#else`, `#endif`, `#undef`, `#error`, `#warning`, and basic
//! macro expansion.

use rustc_hash::FxHashMap;
use std::fmt;

/// A CPP macro definition.
#[derive(Clone, Debug)]
pub enum MacroDef {
    /// Flag macro: `#define FOO`
    Flag,
    /// Object-like macro: `#define FOO value`
    Object(String),
    /// Function-like macro: `#define FOO(a,b) a+b`
    Function(Vec<String>, String),
}

/// State of a conditional block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CondState {
    /// Currently emitting lines (condition was true).
    Active,
    /// Skipping lines (condition was false, haven't seen true branch yet).
    Inactive,
    /// Already saw true branch — skip remaining branches.
    Done,
}

/// Configuration for the CPP preprocessor.
#[derive(Clone, Debug)]
pub struct CppConfig {
    /// Macro definitions.
    pub defines: FxHashMap<String, MacroDef>,
}

/// Error from CPP preprocessing.
#[derive(Clone, Debug)]
pub struct CppError {
    /// Line number (1-based) where the error occurred.
    pub line: usize,
    /// Error message.
    pub message: String,
}

impl fmt::Display for CppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CPP error at line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for CppError {}

/// The CPP preprocessor.
pub struct CppPreprocessor {
    config: CppConfig,
    cond_stack: Vec<CondState>,
}

impl CppPreprocessor {
    /// Create a new preprocessor with the given configuration.
    pub fn new(config: CppConfig) -> Self {
        Self {
            config,
            cond_stack: Vec::new(),
        }
    }

    /// Check if we're currently in an active (emitting) state.
    fn is_active(&self) -> bool {
        self.cond_stack.iter().all(|s| *s == CondState::Active)
    }

    /// Preprocess source code, returning the processed source.
    ///
    /// Empty lines are emitted for skipped/directive lines to preserve
    /// line numbering.
    pub fn preprocess(&mut self, source: &str) -> Result<String, CppError> {
        let mut output = String::with_capacity(source.len());
        let lines: Vec<&str> = source.lines().collect();
        let had_trailing_newline = source.ends_with('\n');

        for (line_idx, line) in lines.iter().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = line.trim_start();

            if trimmed.starts_with('#') {
                // Parse as CPP directive
                self.process_directive(trimmed, line_num)?;
                // Emit empty line to preserve line count
                output.push('\n');
            } else if self.is_active() {
                // Active: expand macros and emit
                let expanded = self.expand_macros(line);
                output.push_str(&expanded);
                output.push('\n');
            } else {
                // Inactive: emit empty line
                output.push('\n');
            }
        }

        // Check for unterminated conditionals
        if !self.cond_stack.is_empty() {
            return Err(CppError {
                line: lines.len(),
                message: format!(
                    "unterminated #if/#ifdef (#endif expected, {} level(s) open)",
                    self.cond_stack.len()
                ),
            });
        }

        // Remove the trailing newline we added if the source didn't have one
        if !had_trailing_newline && output.ends_with('\n') {
            output.pop();
        }

        Ok(output)
    }

    /// Process a CPP directive line.
    fn process_directive(&mut self, line: &str, line_num: usize) -> Result<(), CppError> {
        // Strip leading '#' and whitespace
        let after_hash = line[1..].trim_start();

        // Extract directive name
        let (directive, rest) = match after_hash.find(|c: char| c.is_whitespace()) {
            Some(pos) => (&after_hash[..pos], after_hash[pos..].trim_start()),
            None => (after_hash, ""),
        };

        match directive {
            "ifdef" => self.handle_ifdef(rest, line_num),
            "ifndef" => self.handle_ifndef(rest, line_num),
            "if" => self.handle_if(rest, line_num),
            "elif" => self.handle_elif(rest, line_num),
            "else" => self.handle_else(line_num),
            "endif" => self.handle_endif(line_num),
            "define" => {
                if self.is_active() {
                    self.handle_define(rest, line_num)
                } else {
                    Ok(())
                }
            }
            "undef" => {
                if self.is_active() {
                    self.handle_undef(rest)
                } else {
                    Ok(())
                }
            }
            "error" => {
                if self.is_active() {
                    Err(CppError {
                        line: line_num,
                        message: format!("#error {}", rest),
                    })
                } else {
                    Ok(())
                }
            }
            "warning" => {
                if self.is_active() {
                    tracing::warn!("CPP warning at line {}: {}", line_num, rest);
                }
                Ok(())
            }
            "include" => {
                if self.is_active() {
                    tracing::warn!(
                        "CPP #include at line {} is not supported, skipping: {}",
                        line_num,
                        rest
                    );
                }
                Ok(())
            }
            "line" | "pragma" => Ok(()), // Ignored
            "" => Ok(()),                // Bare '#' line
            _ => {
                // Unknown directive — ignore silently (GHC behavior)
                Ok(())
            }
        }
    }

    fn handle_ifdef(&mut self, name: &str, line_num: usize) -> Result<(), CppError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(CppError {
                line: line_num,
                message: "#ifdef requires a macro name".to_string(),
            });
        }
        // Strip any trailing comment
        let name = strip_cpp_comment(name);

        if !self.is_active() {
            // Already in inactive section — push Done so nested blocks are skipped
            self.cond_stack.push(CondState::Done);
        } else if self.config.defines.contains_key(name) {
            self.cond_stack.push(CondState::Active);
        } else {
            self.cond_stack.push(CondState::Inactive);
        }
        Ok(())
    }

    fn handle_ifndef(&mut self, name: &str, line_num: usize) -> Result<(), CppError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(CppError {
                line: line_num,
                message: "#ifndef requires a macro name".to_string(),
            });
        }
        let name = strip_cpp_comment(name);

        if !self.is_active() {
            self.cond_stack.push(CondState::Done);
        } else if !self.config.defines.contains_key(name) {
            self.cond_stack.push(CondState::Active);
        } else {
            self.cond_stack.push(CondState::Inactive);
        }
        Ok(())
    }

    fn handle_if(&mut self, expr: &str, line_num: usize) -> Result<(), CppError> {
        if !self.is_active() {
            self.cond_stack.push(CondState::Done);
            return Ok(());
        }

        let value = self.eval_cpp_expr(expr, line_num)?;
        if value != 0 {
            self.cond_stack.push(CondState::Active);
        } else {
            self.cond_stack.push(CondState::Inactive);
        }
        Ok(())
    }

    fn handle_elif(&mut self, expr: &str, line_num: usize) -> Result<(), CppError> {
        let state = self.cond_stack.last().copied().ok_or_else(|| CppError {
            line: line_num,
            message: "#elif without matching #if".to_string(),
        })?;

        match state {
            CondState::Active => {
                // Previous branch was taken — skip rest
                *self.cond_stack.last_mut().unwrap() = CondState::Done;
            }
            CondState::Inactive => {
                // Previous branch not taken — evaluate this one
                let value = self.eval_cpp_expr(expr, line_num)?;
                if value != 0 {
                    *self.cond_stack.last_mut().unwrap() = CondState::Active;
                }
                // else stays Inactive
            }
            CondState::Done => {
                // Already took a branch — stay done
            }
        }
        Ok(())
    }

    fn handle_else(&mut self, line_num: usize) -> Result<(), CppError> {
        let state = self.cond_stack.last().copied().ok_or_else(|| CppError {
            line: line_num,
            message: "#else without matching #if".to_string(),
        })?;

        match state {
            CondState::Active => {
                *self.cond_stack.last_mut().unwrap() = CondState::Done;
            }
            CondState::Inactive => {
                // Check if outer context is active
                let outer_active = self.cond_stack.len() <= 1
                    || self.cond_stack[..self.cond_stack.len() - 1]
                        .iter()
                        .all(|s| *s == CondState::Active);
                if outer_active {
                    *self.cond_stack.last_mut().unwrap() = CondState::Active;
                }
            }
            CondState::Done => {
                // Stay done
            }
        }
        Ok(())
    }

    fn handle_endif(&mut self, line_num: usize) -> Result<(), CppError> {
        if self.cond_stack.pop().is_none() {
            return Err(CppError {
                line: line_num,
                message: "#endif without matching #if/#ifdef".to_string(),
            });
        }
        Ok(())
    }

    fn handle_define(&mut self, rest: &str, line_num: usize) -> Result<(), CppError> {
        let rest = rest.trim();
        if rest.is_empty() {
            return Err(CppError {
                line: line_num,
                message: "#define requires a macro name".to_string(),
            });
        }

        // Check for function-like macro: NAME(args)
        if let Some(paren_pos) = rest.find('(') {
            let name = &rest[..paren_pos];
            // Ensure name is a valid identifier
            if !is_cpp_ident(name) {
                return Err(CppError {
                    line: line_num,
                    message: format!("invalid macro name: {}", name),
                });
            }
            // Only treat as function-like if '(' immediately follows name (no space)
            let after_name = &rest[paren_pos..];
            if let Some(close_paren) = after_name.find(')') {
                let args_str = &after_name[1..close_paren];
                let args: Vec<String> = args_str
                    .split(',')
                    .map(|a| a.trim().to_string())
                    .filter(|a| !a.is_empty())
                    .collect();
                let body = after_name[close_paren + 1..].trim().to_string();
                self.config
                    .defines
                    .insert(name.to_string(), MacroDef::Function(args, body));
            } else {
                return Err(CppError {
                    line: line_num,
                    message: "unclosed parenthesis in function-like macro".to_string(),
                });
            }
        } else {
            // Object-like macro or flag
            let (name, value) = match rest.find(|c: char| c.is_whitespace()) {
                Some(pos) => (&rest[..pos], rest[pos..].trim()),
                None => (rest, ""),
            };

            if !is_cpp_ident(name) {
                return Err(CppError {
                    line: line_num,
                    message: format!("invalid macro name: {}", name),
                });
            }

            if value.is_empty() {
                self.config
                    .defines
                    .insert(name.to_string(), MacroDef::Flag);
            } else {
                self.config
                    .defines
                    .insert(name.to_string(), MacroDef::Object(value.to_string()));
            }
        }

        Ok(())
    }

    fn handle_undef(&mut self, name: &str) -> Result<(), CppError> {
        let name = name.trim();
        let name = strip_cpp_comment(name);
        self.config.defines.remove(name);
        Ok(())
    }

    /// Expand macros in a line of code.
    ///
    /// Performs whole-word replacement only, skipping inside string literals
    /// and Haskell comments.
    fn expand_macros(&self, line: &str) -> String {
        if self.config.defines.is_empty() {
            return line.to_string();
        }

        let mut result = String::with_capacity(line.len());
        let chars: Vec<char> = line.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            // Skip string literals
            if chars[i] == '"' {
                result.push('"');
                i += 1;
                while i < len && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < len {
                        result.push(chars[i]);
                        result.push(chars[i + 1]);
                        i += 2;
                    } else {
                        result.push(chars[i]);
                        i += 1;
                    }
                }
                if i < len {
                    result.push('"');
                    i += 1;
                }
                continue;
            }

            // Skip char literals
            if chars[i] == '\'' {
                result.push('\'');
                i += 1;
                while i < len && chars[i] != '\'' {
                    if chars[i] == '\\' && i + 1 < len {
                        result.push(chars[i]);
                        result.push(chars[i + 1]);
                        i += 2;
                    } else {
                        result.push(chars[i]);
                        i += 1;
                    }
                }
                if i < len {
                    result.push('\'');
                    i += 1;
                }
                continue;
            }

            // Skip line comments
            if chars[i] == '-' && i + 1 < len && chars[i + 1] == '-' {
                // Emit rest of line unchanged
                for c in &chars[i..] {
                    result.push(*c);
                }
                break;
            }

            // Skip block comments
            if chars[i] == '{' && i + 1 < len && chars[i + 1] == '-' {
                result.push('{');
                result.push('-');
                i += 2;
                let mut depth = 1;
                while i < len && depth > 0 {
                    if chars[i] == '{' && i + 1 < len && chars[i + 1] == '-' {
                        depth += 1;
                        result.push('{');
                        result.push('-');
                        i += 2;
                    } else if chars[i] == '-' && i + 1 < len && chars[i + 1] == '}' {
                        depth -= 1;
                        result.push('-');
                        result.push('}');
                        i += 2;
                    } else {
                        result.push(chars[i]);
                        i += 1;
                    }
                }
                continue;
            }

            // Try to match an identifier
            if is_ident_start(chars[i]) {
                let start = i;
                while i < len && is_ident_continue(chars[i]) {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();

                if let Some(def) = self.config.defines.get(&word) {
                    match def {
                        MacroDef::Flag => {
                            // Flag macros expand to 1 in code context
                            result.push('1');
                        }
                        MacroDef::Object(value) => {
                            result.push_str(value);
                        }
                        MacroDef::Function(params, body) => {
                            // Check if followed by '('
                            if i < len && chars[i] == '(' {
                                if let Some((args, end)) =
                                    parse_macro_args(&chars, i, len)
                                {
                                    let expanded =
                                        expand_function_macro(params, body, &args);
                                    result.push_str(&expanded);
                                    i = end;
                                } else {
                                    // No matching ')' — emit as-is
                                    result.push_str(&word);
                                }
                            } else {
                                // Function-like macro without args — emit as-is
                                result.push_str(&word);
                            }
                        }
                    }
                } else {
                    result.push_str(&word);
                }
                continue;
            }

            result.push(chars[i]);
            i += 1;
        }

        result
    }

    // ========================================================================
    // Expression evaluator for #if conditions
    // ========================================================================

    /// Evaluate a CPP `#if` expression. Returns an integer value.
    fn eval_cpp_expr(&self, expr: &str, line_num: usize) -> Result<i64, CppError> {
        // First expand macros in the expression
        let expanded = self.expand_expr_macros(expr);
        let tokens = tokenize_cpp_expr(&expanded);
        let mut parser = ExprParser::new(&tokens, &self.config.defines, line_num);
        let value = parser.parse_or()?;
        Ok(value)
    }

    /// Expand macros within a `#if` expression.
    ///
    /// Similar to `expand_macros` but skips the argument of `defined(...)` /
    /// `defined NAME` so that `defined(FOO)` works even when `FOO` is a
    /// flag macro that would normally expand to `1`.
    fn expand_expr_macros(&self, expr: &str) -> String {
        let mut result = String::with_capacity(expr.len());
        let chars: Vec<char> = expr.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            if is_ident_start(chars[i]) {
                let start = i;
                while i < len && is_ident_continue(chars[i]) {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();

                // Don't expand 'defined' keyword, and skip its argument
                if word == "defined" {
                    result.push_str("defined");
                    // Skip whitespace
                    while i < len && chars[i].is_whitespace() {
                        result.push(chars[i]);
                        i += 1;
                    }
                    // Skip parenthesized or bare identifier argument
                    if i < len && chars[i] == '(' {
                        result.push('(');
                        i += 1;
                        while i < len && chars[i] != ')' {
                            result.push(chars[i]);
                            i += 1;
                        }
                        if i < len {
                            result.push(')');
                            i += 1;
                        }
                    } else if i < len && is_ident_start(chars[i]) {
                        // Bare: defined NAME
                        while i < len && is_ident_continue(chars[i]) {
                            result.push(chars[i]);
                            i += 1;
                        }
                    }
                    continue;
                }

                if let Some(def) = self.config.defines.get(&word) {
                    match def {
                        MacroDef::Flag => result.push('1'),
                        MacroDef::Object(val) => result.push_str(val),
                        MacroDef::Function(params, body) => {
                            // Check for args
                            if i < len && chars[i] == '(' {
                                if let Some((args, end)) = parse_macro_args(&chars, i, len) {
                                    let expanded = expand_function_macro(params, body, &args);
                                    result.push_str(&expanded);
                                    i = end;
                                } else {
                                    result.push_str(&word);
                                }
                            } else {
                                result.push_str(&word);
                            }
                        }
                    }
                } else {
                    // In #if expressions, undefined names stay as-is
                    // (the expression parser will treat them as 0)
                    result.push_str(&word);
                }
                continue;
            }
            result.push(chars[i]);
            i += 1;
        }

        result
    }
}

// ============================================================================
// Expression parser for #if conditions
// ============================================================================

/// Token for CPP expression parsing.
#[derive(Clone, Debug, PartialEq)]
enum CppToken {
    Int(i64),
    Ident(String),
    LParen,
    RParen,
    Not,
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Comma,
    Defined,
}

/// Tokenize a CPP expression string.
fn tokenize_cpp_expr(expr: &str) -> Vec<CppToken> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = expr.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }

        // Integer literals
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < len && (chars[i].is_ascii_digit() || chars[i] == 'x' || chars[i] == 'X'
                || (chars[i].is_ascii_hexdigit() && start + 1 < i && (chars[start + 1] == 'x' || chars[start + 1] == 'X')))
            {
                i += 1;
            }
            // Skip trailing L/U/LL suffixes
            while i < len && (chars[i] == 'L' || chars[i] == 'U' || chars[i] == 'l' || chars[i] == 'u') {
                i += 1;
            }
            let num_str: String = chars[start..i].iter().collect();
            let num_str = num_str.trim_end_matches(|c: char| c == 'L' || c == 'U' || c == 'l' || c == 'u');
            let value = if num_str.starts_with("0x") || num_str.starts_with("0X") {
                i64::from_str_radix(&num_str[2..], 16).unwrap_or(0)
            } else {
                num_str.parse::<i64>().unwrap_or(0)
            };
            tokens.push(CppToken::Int(value));
            continue;
        }

        // Identifiers
        if is_ident_start(chars[i]) {
            let start = i;
            while i < len && is_ident_continue(chars[i]) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            if word == "defined" {
                tokens.push(CppToken::Defined);
            } else {
                tokens.push(CppToken::Ident(word));
            }
            continue;
        }

        // Two-char operators
        if i + 1 < len {
            let two: String = chars[i..i + 2].iter().collect();
            match two.as_str() {
                "&&" => {
                    tokens.push(CppToken::And);
                    i += 2;
                    continue;
                }
                "||" => {
                    tokens.push(CppToken::Or);
                    i += 2;
                    continue;
                }
                "==" => {
                    tokens.push(CppToken::Eq);
                    i += 2;
                    continue;
                }
                "!=" => {
                    tokens.push(CppToken::Ne);
                    i += 2;
                    continue;
                }
                "<=" => {
                    tokens.push(CppToken::Le);
                    i += 2;
                    continue;
                }
                ">=" => {
                    tokens.push(CppToken::Ge);
                    i += 2;
                    continue;
                }
                _ => {}
            }
        }

        // Single-char operators
        match chars[i] {
            '(' => tokens.push(CppToken::LParen),
            ')' => tokens.push(CppToken::RParen),
            '!' => tokens.push(CppToken::Not),
            '<' => tokens.push(CppToken::Lt),
            '>' => tokens.push(CppToken::Gt),
            '+' => tokens.push(CppToken::Plus),
            '-' => tokens.push(CppToken::Minus),
            '*' => tokens.push(CppToken::Star),
            '/' => tokens.push(CppToken::Slash),
            '%' => tokens.push(CppToken::Percent),
            ',' => tokens.push(CppToken::Comma),
            _ => {} // Skip unknown chars
        }
        i += 1;
    }

    tokens
}

/// Recursive descent parser for CPP expressions.
struct ExprParser<'a> {
    tokens: &'a [CppToken],
    pos: usize,
    defines: &'a FxHashMap<String, MacroDef>,
    line_num: usize,
}

impl<'a> ExprParser<'a> {
    fn new(
        tokens: &'a [CppToken],
        defines: &'a FxHashMap<String, MacroDef>,
        line_num: usize,
    ) -> Self {
        Self {
            tokens,
            pos: 0,
            defines,
            line_num,
        }
    }

    fn peek(&self) -> Option<&CppToken> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&CppToken> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: &CppToken) -> Result<(), CppError> {
        let line = self.line_num;
        let tok = self.advance().cloned();
        match tok {
            Some(ref tok) if tok == expected => Ok(()),
            Some(tok) => Err(CppError {
                line,
                message: format!("expected {:?}, got {:?}", expected, tok),
            }),
            None => Err(CppError {
                line,
                message: format!("expected {:?}, got end of expression", expected),
            }),
        }
    }

    /// Parse: or_expr = and_expr ('||' and_expr)*
    fn parse_or(&mut self) -> Result<i64, CppError> {
        let mut left = self.parse_and()?;
        while self.peek() == Some(&CppToken::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = if left != 0 || right != 0 { 1 } else { 0 };
        }
        Ok(left)
    }

    /// Parse: and_expr = eq_expr ('&&' eq_expr)*
    fn parse_and(&mut self) -> Result<i64, CppError> {
        let mut left = self.parse_equality()?;
        while self.peek() == Some(&CppToken::And) {
            self.advance();
            let right = self.parse_equality()?;
            left = if left != 0 && right != 0 { 1 } else { 0 };
        }
        Ok(left)
    }

    /// Parse: eq_expr = rel_expr (('==' | '!=') rel_expr)*
    fn parse_equality(&mut self) -> Result<i64, CppError> {
        let mut left = self.parse_relational()?;
        loop {
            match self.peek() {
                Some(&CppToken::Eq) => {
                    self.advance();
                    let right = self.parse_relational()?;
                    left = if left == right { 1 } else { 0 };
                }
                Some(&CppToken::Ne) => {
                    self.advance();
                    let right = self.parse_relational()?;
                    left = if left != right { 1 } else { 0 };
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// Parse: rel_expr = add_expr (('<' | '>' | '<=' | '>=') add_expr)*
    fn parse_relational(&mut self) -> Result<i64, CppError> {
        let mut left = self.parse_additive()?;
        loop {
            match self.peek() {
                Some(&CppToken::Lt) => {
                    self.advance();
                    let right = self.parse_additive()?;
                    left = if left < right { 1 } else { 0 };
                }
                Some(&CppToken::Gt) => {
                    self.advance();
                    let right = self.parse_additive()?;
                    left = if left > right { 1 } else { 0 };
                }
                Some(&CppToken::Le) => {
                    self.advance();
                    let right = self.parse_additive()?;
                    left = if left <= right { 1 } else { 0 };
                }
                Some(&CppToken::Ge) => {
                    self.advance();
                    let right = self.parse_additive()?;
                    left = if left >= right { 1 } else { 0 };
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// Parse: add_expr = mul_expr (('+' | '-') mul_expr)*
    fn parse_additive(&mut self) -> Result<i64, CppError> {
        let mut left = self.parse_multiplicative()?;
        loop {
            match self.peek() {
                Some(&CppToken::Plus) => {
                    self.advance();
                    let right = self.parse_multiplicative()?;
                    left = left.wrapping_add(right);
                }
                Some(&CppToken::Minus) => {
                    self.advance();
                    let right = self.parse_multiplicative()?;
                    left = left.wrapping_sub(right);
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// Parse: mul_expr = unary_expr (('*' | '/' | '%') unary_expr)*
    fn parse_multiplicative(&mut self) -> Result<i64, CppError> {
        let mut left = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(&CppToken::Star) => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = left.wrapping_mul(right);
                }
                Some(&CppToken::Slash) => {
                    self.advance();
                    let right = self.parse_unary()?;
                    if right == 0 {
                        return Err(CppError {
                            line: self.line_num,
                            message: "division by zero in #if expression".to_string(),
                        });
                    }
                    left = left.wrapping_div(right);
                }
                Some(&CppToken::Percent) => {
                    self.advance();
                    let right = self.parse_unary()?;
                    if right == 0 {
                        return Err(CppError {
                            line: self.line_num,
                            message: "modulo by zero in #if expression".to_string(),
                        });
                    }
                    left = left.wrapping_rem(right);
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// Parse: unary = '!' unary | '-' unary | primary
    fn parse_unary(&mut self) -> Result<i64, CppError> {
        match self.peek() {
            Some(&CppToken::Not) => {
                self.advance();
                let val = self.parse_unary()?;
                Ok(if val == 0 { 1 } else { 0 })
            }
            Some(&CppToken::Minus) => {
                self.advance();
                let val = self.parse_unary()?;
                Ok(-val)
            }
            _ => self.parse_primary(),
        }
    }

    /// Parse: primary = int | defined(NAME) | defined NAME | '(' expr ')' | ident
    fn parse_primary(&mut self) -> Result<i64, CppError> {
        match self.peek().cloned() {
            Some(CppToken::Int(n)) => {
                self.advance();
                Ok(n)
            }
            Some(CppToken::Defined) => {
                self.advance();
                // defined(NAME) or defined NAME
                let has_paren = self.peek() == Some(&CppToken::LParen);
                if has_paren {
                    self.advance(); // skip '('
                }
                let name = match self.advance().cloned() {
                    Some(CppToken::Ident(name)) => name,
                    other => {
                        return Err(CppError {
                            line: self.line_num,
                            message: format!(
                                "expected identifier after 'defined', got {:?}",
                                other
                            ),
                        });
                    }
                };
                if has_paren {
                    self.expect(&CppToken::RParen)?;
                }
                Ok(if self.defines.contains_key(&name) { 1 } else { 0 })
            }
            Some(CppToken::LParen) => {
                self.advance();
                let val = self.parse_or()?;
                self.expect(&CppToken::RParen)?;
                Ok(val)
            }
            Some(CppToken::Ident(_)) => {
                self.advance();
                // Unknown identifiers evaluate to 0 in CPP
                Ok(0)
            }
            other => Err(CppError {
                line: self.line_num,
                message: format!("unexpected token in expression: {:?}", other),
            }),
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Check if a character can start a C identifier.
fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Check if a character can continue a C identifier.
fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Check if a string is a valid C identifier.
fn is_cpp_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if is_ident_start(c) => chars.all(is_ident_continue),
        _ => false,
    }
}

/// Strip a trailing C-style comment (/* ... */ or //) from a token.
fn strip_cpp_comment(s: &str) -> &str {
    if let Some(pos) = s.find("/*") {
        s[..pos].trim_end()
    } else if let Some(pos) = s.find("//") {
        s[..pos].trim_end()
    } else {
        // Also strip trailing whitespace
        s.split_whitespace().next().unwrap_or(s)
    }
}

/// Parse function-like macro arguments from a character slice starting at '('.
/// Returns (Vec<arg_strings>, position_after_closing_paren).
fn parse_macro_args(chars: &[char], start: usize, len: usize) -> Option<(Vec<String>, usize)> {
    assert_eq!(chars[start], '(');
    let mut i = start + 1;
    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 1;

    while i < len && depth > 0 {
        match chars[i] {
            '(' => {
                depth += 1;
                current.push('(');
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    args.push(current.trim().to_string());
                } else {
                    current.push(')');
                }
            }
            ',' if depth == 1 => {
                args.push(current.trim().to_string());
                current = String::new();
            }
            c => current.push(c),
        }
        i += 1;
    }

    if depth != 0 {
        return None;
    }

    // Filter out empty args from no-arg invocations like FOO()
    if args.len() == 1 && args[0].is_empty() {
        args.clear();
    }

    Some((args, i))
}

/// Expand a function-like macro by substituting parameters.
fn expand_function_macro(params: &[String], body: &str, args: &[String]) -> String {
    let mut result = body.to_string();
    for (i, param) in params.iter().enumerate() {
        if let Some(arg) = args.get(i) {
            // Whole-word replacement of parameter
            result = replace_whole_word(&result, param, arg);
        }
    }
    result
}

/// Replace whole-word occurrences of `from` with `to` in `text`.
fn replace_whole_word(text: &str, from: &str, to: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let from_chars: Vec<char> = from.chars().collect();
    let from_len = from_chars.len();
    let mut i = 0;

    while i < len {
        if i + from_len <= len && chars[i..i + from_len] == from_chars[..] {
            // Check word boundaries
            let before_ok = i == 0 || !is_ident_continue(chars[i - 1]);
            let after_ok = i + from_len >= len || !is_ident_continue(chars[i + from_len]);
            if before_ok && after_ok {
                result.push_str(to);
                i += from_len;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }

    result
}

/// Build the default CPP configuration with predefined macros.
pub fn default_cpp_config(options: &crate::Options) -> CppConfig {
    let mut defines = FxHashMap::default();

    // GHC version compatibility (BHC reports as GHC 9.10)
    defines.insert(
        "__GLASGOW_HASKELL__".to_string(),
        MacroDef::Object("910".to_string()),
    );

    // Platform detection
    #[cfg(target_os = "macos")]
    {
        defines.insert("darwin_HOST_OS".to_string(), MacroDef::Object("1".to_string()));
        defines.insert("__APPLE__".to_string(), MacroDef::Object("1".to_string()));
    }

    #[cfg(target_os = "linux")]
    {
        defines.insert("linux_HOST_OS".to_string(), MacroDef::Object("1".to_string()));
        defines.insert("__linux__".to_string(), MacroDef::Object("1".to_string()));
    }

    #[cfg(target_os = "windows")]
    {
        defines.insert("mingw32_HOST_OS".to_string(), MacroDef::Object("1".to_string()));
        defines.insert("_WIN32".to_string(), MacroDef::Object("1".to_string()));
    }

    // Architecture detection
    #[cfg(target_arch = "x86_64")]
    {
        defines.insert("x86_64_HOST_ARCH".to_string(), MacroDef::Object("1".to_string()));
    }

    #[cfg(target_arch = "aarch64")]
    {
        defines.insert("aarch64_HOST_ARCH".to_string(), MacroDef::Object("1".to_string()));
    }

    // Pointer size
    #[cfg(target_pointer_width = "64")]
    {
        defines.insert("SIZEOF_HSINT".to_string(), MacroDef::Object("8".to_string()));
        defines.insert("SIZEOF_HSWORD".to_string(), MacroDef::Object("8".to_string()));
    }

    // MIN_VERSION macros for common packages.
    // These are function-like macros: MIN_VERSION_base(major1,major2,minor)
    // BHC ships with base-4.20, so MIN_VERSION_base(4,20,0) is true.
    defines.insert(
        "MIN_VERSION_base".to_string(),
        MacroDef::Function(
            vec!["major1".to_string(), "major2".to_string(), "minor".to_string()],
            "(major1 < 4 || (major1 == 4 && (major2 < 20 || (major2 == 20 && minor <= 0))))".to_string(),
        ),
    );

    defines.insert(
        "MIN_VERSION_text".to_string(),
        MacroDef::Function(
            vec!["major1".to_string(), "major2".to_string(), "minor".to_string()],
            "(major1 < 2 || (major1 == 2 && (major2 < 1 || (major2 == 1 && minor <= 0))))".to_string(),
        ),
    );

    defines.insert(
        "MIN_VERSION_bytestring".to_string(),
        MacroDef::Function(
            vec!["major1".to_string(), "major2".to_string(), "minor".to_string()],
            "(major1 < 0 || (major1 == 0 && (major2 < 12 || (major2 == 12 && minor <= 0))))".to_string(),
        ),
    );

    defines.insert(
        "MIN_VERSION_containers".to_string(),
        MacroDef::Function(
            vec!["major1".to_string(), "major2".to_string(), "minor".to_string()],
            "(major1 < 0 || (major1 == 0 && (major2 < 7 || (major2 == 7 && minor <= 0))))".to_string(),
        ),
    );

    // Process user-supplied defines from CLI (-D flags)
    for def in &options.cpp_defines {
        if let Some(eq_pos) = def.find('=') {
            let name = def[..eq_pos].to_string();
            let value = def[eq_pos + 1..].to_string();
            defines.insert(name, MacroDef::Object(value));
        } else {
            defines.insert(def.clone(), MacroDef::Flag);
        }
    }

    CppConfig { defines }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn preprocess(source: &str) -> String {
        let config = CppConfig {
            defines: FxHashMap::default(),
        };
        let mut cpp = CppPreprocessor::new(config);
        cpp.preprocess(source).unwrap()
    }

    fn preprocess_with(source: &str, defines: &[(&str, &str)]) -> String {
        let mut map = FxHashMap::default();
        for (k, v) in defines {
            if v.is_empty() {
                map.insert(k.to_string(), MacroDef::Flag);
            } else {
                map.insert(k.to_string(), MacroDef::Object(v.to_string()));
            }
        }
        let config = CppConfig { defines: map };
        let mut cpp = CppPreprocessor::new(config);
        cpp.preprocess(source).unwrap()
    }

    fn preprocess_err(source: &str) -> CppError {
        let config = CppConfig {
            defines: FxHashMap::default(),
        };
        let mut cpp = CppPreprocessor::new(config);
        cpp.preprocess(source).unwrap_err()
    }

    #[test]
    fn test_ifdef_defined() {
        let source = "#ifdef FOO\nyes\n#endif\n";
        let result = preprocess_with(source, &[("FOO", "")]);
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_ifdef_undefined() {
        let source = "#ifdef FOO\nyes\n#endif\n";
        let result = preprocess(source);
        assert!(!result.contains("yes"));
    }

    #[test]
    fn test_ifndef() {
        let source = "#ifndef FOO\nyes\n#endif\n";
        let result = preprocess(source);
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_ifdef_else() {
        let source = "#ifdef FOO\nyes\n#else\nno\n#endif\n";
        let result = preprocess(source);
        assert!(!result.contains("yes"));
        assert!(result.contains("no"));
    }

    #[test]
    fn test_if_true() {
        let source = "#if 1\nyes\n#endif\n";
        let result = preprocess(source);
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_if_false() {
        let source = "#if 0\nyes\n#endif\n";
        let result = preprocess(source);
        assert!(!result.contains("yes"));
    }

    #[test]
    fn test_if_comparison() {
        let source = "#if 910 >= 900\nyes\n#endif\n";
        let result = preprocess(source);
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_if_defined_operator() {
        let source = "#if defined(FOO)\nyes\n#endif\n";
        let result = preprocess_with(source, &[("FOO", "")]);
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_if_defined_and() {
        let source = "#if defined(A) && defined(B)\nyes\n#endif\n";
        let result = preprocess_with(source, &[("A", ""), ("B", "")]);
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_if_defined_and_missing() {
        let source = "#if defined(A) && defined(B)\nyes\n#endif\n";
        let result = preprocess_with(source, &[("A", "")]);
        assert!(!result.contains("yes"));
    }

    #[test]
    fn test_elif_chain() {
        let source = "#if 0\na\n#elif 0\nb\n#elif 1\nc\n#else\nd\n#endif\n";
        let result = preprocess(source);
        assert!(result.contains("c"));
        assert!(!result.contains("a"));
        assert!(!result.contains("b"));
        assert!(!result.contains("d"));
    }

    #[test]
    fn test_define_in_file() {
        let source = "#define GREETING hello\nGREETING world\n";
        let result = preprocess(source);
        assert!(result.contains("hello world"));
    }

    #[test]
    fn test_define_flag() {
        let source = "#define FOO\n#ifdef FOO\nyes\n#endif\n";
        let result = preprocess(source);
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_undef() {
        let source = "#define FOO\n#undef FOO\n#ifdef FOO\nyes\n#endif\n";
        let result = preprocess(source);
        assert!(!result.contains("yes"));
    }

    #[test]
    fn test_nested_ifdef() {
        let source = "#ifdef A\n#ifdef B\nyes\n#endif\n#endif\n";
        let result = preprocess_with(source, &[("A", ""), ("B", "")]);
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_nested_ifdef_outer_false() {
        let source = "#ifdef A\n#ifdef B\nyes\n#endif\n#endif\n";
        let result = preprocess_with(source, &[("B", "")]);
        assert!(!result.contains("yes"));
    }

    #[test]
    fn test_line_count_preserved() {
        let source = "line1\n#ifdef FOO\nskipped\n#endif\nline2\n";
        let result = preprocess(source);
        let input_lines = source.lines().count();
        let output_lines = result.lines().count();
        assert_eq!(input_lines, output_lines);
    }

    #[test]
    fn test_macro_expansion_in_string_skipped() {
        let source = "#define FOO bar\n\"FOO\"\n";
        let result = preprocess(source);
        assert!(result.contains("\"FOO\""));
        assert!(!result.contains("\"bar\""));
    }

    #[test]
    fn test_macro_expansion_whole_word() {
        let source = "#define A 1\nA AB BA A\n";
        let result = preprocess(source);
        // A should be replaced, but AB and BA should not
        assert!(result.contains("1 AB BA 1"));
    }

    #[test]
    fn test_unterminated_if_error() {
        let err = preprocess_err("#ifdef FOO\nyes\n");
        assert!(err.message.contains("unterminated"));
    }

    #[test]
    fn test_error_directive() {
        let source = "#error something went wrong\n";
        let config = CppConfig {
            defines: FxHashMap::default(),
        };
        let mut cpp = CppPreprocessor::new(config);
        let err = cpp.preprocess(source).unwrap_err();
        assert!(err.message.contains("something went wrong"));
    }

    #[test]
    fn test_if_macro_value() {
        let source = "#if __GLASGOW_HASKELL__ >= 900\nyes\n#endif\n";
        let result = preprocess_with(source, &[("__GLASGOW_HASKELL__", "910")]);
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_function_macro() {
        let source = "#define ADD(a,b) a + b\nresult = ADD(1,2)\n";
        let result = preprocess(source);
        assert!(result.contains("result = 1 + 2"));
    }

    #[test]
    fn test_min_version_in_if() {
        // Simulate MIN_VERSION_base(4,19,0) with base-4.20
        let mut defines = FxHashMap::default();
        defines.insert(
            "MIN_VERSION_base".to_string(),
            MacroDef::Function(
                vec!["major1".to_string(), "major2".to_string(), "minor".to_string()],
                "(major1 < 4 || (major1 == 4 && (major2 < 20 || (major2 == 20 && minor <= 0))))".to_string(),
            ),
        );
        let config = CppConfig { defines };
        let mut cpp = CppPreprocessor::new(config);
        let source = "#if MIN_VERSION_base(4,19,0)\nyes\n#endif\n";
        let result = cpp.preprocess(source).unwrap();
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_comment_in_code_not_expanded() {
        let source = "#define FOO bar\n-- FOO should not expand\n";
        let result = preprocess(source);
        assert!(result.contains("-- FOO should not expand"));
    }

    #[test]
    fn test_passthrough_no_directives() {
        let source = "module Main where\n\nmain :: IO ()\nmain = putStrLn \"Hello\"\n";
        let result = preprocess(source);
        assert_eq!(result, source);
    }

    #[test]
    fn test_if_not() {
        let source = "#if !defined(FOO)\nyes\n#endif\n";
        let result = preprocess(source);
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_if_arithmetic() {
        let source = "#if 2 + 3 == 5\nyes\n#endif\n";
        let result = preprocess(source);
        assert!(result.contains("yes"));
    }

    #[test]
    fn test_define_skipped_in_inactive() {
        let source = "#if 0\n#define FOO\n#endif\n#ifdef FOO\nyes\n#endif\n";
        let result = preprocess(source);
        assert!(!result.contains("yes"));
    }
}
