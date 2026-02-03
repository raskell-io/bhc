//! Basel Haskell Compiler Interactive (bhci) - REPL
//!
//! An interactive environment for exploring Haskell 2026 code.
//!
//! # Commands
//!
//! - `:help` - Show help
//! - `:quit` - Exit the REPL
//! - `:type <expr>` - Show the type of an expression
//! - `:kind <type>` - Show the kind of a type
//! - `:info <name>` - Show information about a name
//! - `:load <file>` - Load a Haskell module
//! - `:reload` - Reload the current module
//! - `:browse [module]` - Browse module exports
//! - `:set <option>` - Set a REPL option
//! - `:unset <option>` - Unset a REPL option
//! - `:module [+/-] <mod>` - Add/remove modules from scope
//! - `:cd <dir>` - Change working directory
//! - `:!<cmd>` - Run shell command

use anyhow::Result;
use bhc_ast::Expr as AstExpr;
use bhc_core::eval::{EvalMode, Evaluator, Value};
use bhc_diagnostics::{DiagnosticRenderer, SourceMap};
use bhc_intern::kw;
use bhc_parser::parse_expr;
use bhc_session::Profile;
use bhc_span::FileId;
use bhc_typeck::TyCtxt;
use bhc_types::Ty;
use clap::Parser;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Config, Editor, Helper};
use std::borrow::Cow;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Basel Haskell Compiler Interactive
#[derive(Parser, Debug)]
#[command(name = "bhci")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Files to load on startup
    #[arg(value_name = "FILE")]
    files: Vec<PathBuf>,

    /// Profile to use (default, server, numeric, edge, realtime, embedded)
    #[arg(long, default_value = "default")]
    profile: String,

    /// Don't load the Prelude
    #[arg(long)]
    no_prelude: bool,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// REPL errors
#[derive(Debug, Error)]
pub enum ReplError {
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Type error: {0}")]
    Type(String),
    #[error("Evaluation error: {0}")]
    Eval(String),
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),
    #[error("Unknown command: {0}")]
    UnknownCommand(String),
}

/// REPL state
struct ReplState {
    /// Source map for error reporting
    source_map: SourceMap,
    /// Type checking context
    type_ctx: TyCtxt,
    /// Evaluator
    evaluator: Evaluator,
    /// Currently loaded files
    loaded_files: Vec<PathBuf>,
    /// Imported modules in scope
    modules_in_scope: HashSet<String>,
    /// Current profile
    profile: Profile,
    /// REPL options
    options: ReplOptions,
    /// Binding counter for let expressions
    binding_counter: u32,
    /// User-defined bindings: (name, type, value, var_id)
    bindings: Vec<(String, Ty, Value, Option<bhc_core::VarId>)>,
    /// Last inferred type from the type checker (for display purposes).
    last_inferred_type: Option<Ty>,
    /// Types loaded from compiled modules: (name, type)
    loaded_types: Vec<(String, Ty)>,
    /// Names of bindings that came from loaded files (vs REPL input)
    loaded_binding_names: Vec<String>,
    /// Multi-line input buffer (active between `:{` and `:}`)
    multiline_buffer: Option<String>,
    /// Accumulated import declarations from REPL `import` statements
    accumulated_imports: Vec<bhc_ast::ImportDecl>,
    /// Shared identifier list for tab-completion
    completion_identifiers: Arc<Mutex<Vec<String>>>,
}

/// REPL configuration options
#[derive(Debug, Clone)]
struct ReplOptions {
    /// Show types after evaluation
    show_types: bool,
    /// Show timing information
    show_timing: bool,
    /// Enable multi-line input
    multiline: bool,
    /// Show warnings
    warnings: bool,
    /// Verbose mode
    verbose: bool,
}

impl Default for ReplOptions {
    fn default() -> Self {
        Self {
            show_types: true,
            show_timing: false,
            multiline: true,
            warnings: true,
            verbose: false,
        }
    }
}

impl ReplState {
    fn new(profile: Profile, verbose: bool) -> Self {
        let mut options = ReplOptions::default();
        options.verbose = verbose;

        let source_map = SourceMap::new();
        let file_id = FileId(0);
        let identifiers: Vec<String> = PRELUDE_NAMES.iter().map(|s| s.to_string()).collect();
        Self {
            source_map,
            type_ctx: TyCtxt::new(file_id),
            evaluator: Evaluator::new(EvalMode::from(profile)),
            loaded_files: Vec::new(),
            modules_in_scope: HashSet::from(["Prelude".to_string()]),
            profile,
            options,
            binding_counter: 0,
            bindings: Vec::new(),
            last_inferred_type: None,
            loaded_types: Vec::new(),
            loaded_binding_names: Vec::new(),
            multiline_buffer: None,
            accumulated_imports: Vec::new(),
            completion_identifiers: Arc::new(Mutex::new(identifiers)),
        }
    }

    fn load_prelude(&mut self) {
        // Add standard Prelude bindings to scope
        self.modules_in_scope.insert("Prelude".to_string());
    }

    fn next_binding_name(&mut self) -> String {
        self.binding_counter += 1;
        format!("it{}", self.binding_counter)
    }
}

/// Well-known Prelude names for tab-completion.
const PRELUDE_NAMES: &[&str] = &[
    "map", "filter", "foldr", "foldl", "head", "tail", "null", "length",
    "reverse", "concat", "sum", "product", "maximum", "minimum", "zip",
    "take", "drop", "takeWhile", "dropWhile", "elem", "notElem",
    "putStrLn", "print", "show", "readFile", "writeFile",
    "True", "False", "Nothing", "Just", "Left", "Right",
    "not", "and", "or", "any", "all", "id", "const", "flip", "even", "odd",
    "div", "mod", "abs", "negate", "fromIntegral",
    "error", "undefined", "otherwise",
    "if", "then", "else", "let", "in", "where", "case", "of", "do",
];

/// Rustyline helper for completion and hints
struct ReplHelper {
    commands: Vec<String>,
    modules: Vec<String>,
    identifiers: Arc<Mutex<Vec<String>>>,
}

impl ReplHelper {
    fn new(identifiers: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            commands: vec![
                ":help".to_string(),
                ":quit".to_string(),
                ":type".to_string(),
                ":kind".to_string(),
                ":info".to_string(),
                ":load".to_string(),
                ":reload".to_string(),
                ":browse".to_string(),
                ":set".to_string(),
                ":unset".to_string(),
                ":module".to_string(),
                ":cd".to_string(),
                ":show".to_string(),
            ],
            modules: vec![
                "Prelude".to_string(),
                "Data.List".to_string(),
                "Data.Maybe".to_string(),
                "Data.Either".to_string(),
                "Control.Monad".to_string(),
            ],
            identifiers,
        }
    }
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let mut completions = Vec::new();

        // Command completion
        if line.starts_with(':') {
            let prefix = &line[..pos];
            for cmd in &self.commands {
                if cmd.starts_with(prefix) {
                    completions.push(Pair {
                        display: cmd.clone(),
                        replacement: cmd.clone(),
                    });
                }
            }
            return Ok((0, completions));
        }

        // Module completion after :load, :browse, :module, import
        if line.starts_with(":load ")
            || line.starts_with(":browse ")
            || line.starts_with(":module ")
            || line.starts_with("import ")
            || line.starts_with("import qualified ")
        {
            let word_start = line.rfind(' ').map(|i| i + 1).unwrap_or(0);
            let prefix = &line[word_start..pos];
            for module in &self.modules {
                if module.starts_with(prefix) {
                    completions.push(Pair {
                        display: module.clone(),
                        replacement: module.clone(),
                    });
                }
            }
            return Ok((word_start, completions));
        }

        // Identifier completion for general input
        let word_start = line[..pos]
            .rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != '\'')
            .map(|i| i + 1)
            .unwrap_or(0);
        let prefix = &line[word_start..pos];

        if !prefix.is_empty() {
            if let Ok(ids) = self.identifiers.lock() {
                for id in ids.iter() {
                    if id.starts_with(prefix) && id != prefix {
                        completions.push(Pair {
                            display: id.clone(),
                            replacement: id.clone(),
                        });
                    }
                }
            }
        }

        Ok((word_start, completions))
    }
}

impl Hinter for ReplHelper {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        None
    }
}

impl Highlighter for ReplHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        // Basic syntax highlighting
        if line.starts_with(':') {
            Cow::Owned(format!("\x1b[36m{}\x1b[0m", line)) // Cyan for commands
        } else {
            Cow::Borrowed(line)
        }
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Owned(format!("\x1b[1;32m{}\x1b[0m", prompt)) // Bold green
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        false
    }
}

impl Validator for ReplHelper {}

impl Helper for ReplHelper {}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    if cli.verbose {
        tracing_subscriber::fmt::init();
    }

    // Initialize keyword interner
    kw::intern_all();

    // Parse profile
    let profile = match cli.profile.as_str() {
        "default" => Profile::Default,
        "server" => Profile::Server,
        "numeric" => Profile::Numeric,
        "edge" => Profile::Edge,
        "realtime" => Profile::Realtime,
        "embedded" => Profile::Embedded,
        _ => {
            eprintln!("Unknown profile: {}. Using default.", cli.profile);
            Profile::Default
        }
    };

    // Create REPL state
    let mut state = ReplState::new(profile, cli.verbose);

    // Load prelude
    if !cli.no_prelude {
        state.load_prelude();
    }

    // Load initial files
    for file in &cli.files {
        if let Err(e) = load_file(&mut state, file) {
            eprintln!("Error loading {}: {}", file.display(), e);
        }
    }

    // Print banner
    print_banner(&state);

    // Setup rustyline
    let config = Config::builder()
        .history_ignore_space(true)
        .auto_add_history(true)
        .build();

    let helper = ReplHelper::new(Arc::clone(&state.completion_identifiers));
    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(helper));

    // Load history
    let history_path = dirs::data_dir()
        .map(|p| p.join("bhc").join("history"))
        .unwrap_or_else(|| PathBuf::from(".bhci_history"));

    if let Some(parent) = history_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let _ = rl.load_history(&history_path);

    // Main REPL loop
    let mut line_num = 1;

    loop {
        let prompt = if state.multiline_buffer.is_some() {
            format!("bhci:{:03}| ", line_num)
        } else {
            format!("bhci:{:03}> ", line_num)
        };

        match rl.readline(&prompt) {
            Ok(line) => {
                let trimmed = line.trim();

                // Handle multi-line mode
                if let Some(ref mut buf) = state.multiline_buffer {
                    if trimmed == ":}" {
                        let complete_input = buf.clone();
                        state.multiline_buffer = None;
                        if !complete_input.trim().is_empty() {
                            eval_input(&mut state, complete_input.trim());
                        }
                        line_num += 1;
                        continue;
                    }
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(&line);
                    line_num += 1;
                    continue;
                }

                if trimmed.is_empty() {
                    continue;
                }

                // Start multi-line mode
                if trimmed == ":{" {
                    state.multiline_buffer = Some(String::new());
                    line_num += 1;
                    continue;
                }

                // Handle commands
                if trimmed.starts_with(':') {
                    match handle_command(&mut state, trimmed) {
                        CommandResult::Continue => {}
                        CommandResult::Quit => break,
                    }
                } else if trimmed.starts_with("import ") {
                    handle_import(&mut state, trimmed);
                } else {
                    // Evaluate expression
                    eval_input(&mut state, trimmed);
                }

                line_num += 1;
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    // Save history
    let _ = rl.save_history(&history_path);

    println!("\nGoodbye!");
    Ok(())
}

fn print_banner(state: &ReplState) {
    println!("Basel Haskell Compiler Interactive (bhci)");
    println!("Version {}", env!("CARGO_PKG_VERSION"));
    println!("Profile: {:?}", state.profile);
    println!("Type :help for help, :quit to exit");
    println!();
}

enum CommandResult {
    Continue,
    Quit,
}

fn handle_command(state: &mut ReplState, cmd: &str) -> CommandResult {
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
    let cmd_name = parts.first().copied().unwrap_or("");
    let args = parts.get(1).copied().unwrap_or("");

    match cmd_name {
        ":quit" | ":q" => CommandResult::Quit,

        ":help" | ":h" | ":?" => {
            print_help();
            CommandResult::Continue
        }

        ":type" | ":t" => {
            if !args.is_empty() {
                show_type(state, args);
            } else {
                println!("Usage: :type <expression>");
            }
            CommandResult::Continue
        }

        ":kind" | ":k" => {
            if !args.is_empty() {
                show_kind(state, args);
            } else {
                println!("Usage: :kind <type>");
            }
            CommandResult::Continue
        }

        ":info" | ":i" => {
            if !args.is_empty() {
                show_info(state, args);
            } else {
                println!("Usage: :info <name>");
            }
            CommandResult::Continue
        }

        ":load" | ":l" => {
            if !args.is_empty() {
                let path = PathBuf::from(args);
                if let Err(e) = load_file(state, &path) {
                    eprintln!("Error: {}", e);
                }
            } else {
                println!("Usage: :load <file>");
            }
            CommandResult::Continue
        }

        ":reload" | ":r" => {
            reload_modules(state);
            CommandResult::Continue
        }

        ":browse" | ":b" => {
            browse_module(state, if args.is_empty() { None } else { Some(args) });
            CommandResult::Continue
        }

        ":set" => {
            if !args.is_empty() {
                set_option(state, args);
            } else {
                show_options(state);
            }
            CommandResult::Continue
        }

        ":unset" => {
            if !args.is_empty() {
                unset_option(state, args);
            } else {
                println!("Usage: :unset <option>");
            }
            CommandResult::Continue
        }

        ":module" | ":m" => {
            if !args.is_empty() {
                set_modules(state, args);
            } else {
                show_modules(state);
            }
            CommandResult::Continue
        }

        ":cd" => {
            if !args.is_empty() {
                if let Err(e) = std::env::set_current_dir(args) {
                    eprintln!("Error: {}", e);
                } else {
                    println!("Changed to: {}", args);
                }
            } else {
                if let Ok(cwd) = std::env::current_dir() {
                    println!("{}", cwd.display());
                }
            }
            CommandResult::Continue
        }

        cmd if cmd.starts_with(":!") => {
            let shell_cmd = &cmd[2..].trim();
            let full_cmd = if args.is_empty() {
                shell_cmd.to_string()
            } else {
                format!("{} {}", shell_cmd, args)
            };
            run_shell_command(&full_cmd);
            CommandResult::Continue
        }

        ":show" => {
            handle_show(state, args);
            CommandResult::Continue
        }

        _ => {
            println!("Unknown command: {}", cmd_name);
            println!("Type :help for help");
            CommandResult::Continue
        }
    }
}

fn print_help() {
    println!(
        r#"Commands:
  :help, :h, :?         Show this help
  :quit, :q             Exit the REPL
  :type <expr>          Show the type of an expression
  :kind <type>          Show the kind of a type
  :info <name>          Show information about a name
  :load <file>          Load a Haskell module
  :reload               Reload the current modules
  :browse [module]      Browse module exports
  :show [target]        Show REPL state (bindings, imports, modules)
  :module [+/-] <mod>   Add/remove modules from scope
  :set [option]         Show or set REPL options
  :unset <option>       Unset a REPL option
  :cd <dir>             Change working directory
  :!<cmd>               Run shell command

Statements:
  import Data.List      Import a module
  import qualified M as Q  Qualified import
  let x = expr          Bind a variable
  expr                  Evaluate an expression

Options:
  :set +t               Show types after evaluation
  :set +s               Show timing information
  :set +m               Enable multiline mode
  :set -w               Disable warnings
  :set profile <name>   Change compilation profile

Keybindings:
  Ctrl-C                Cancel current input
  Ctrl-D                Exit (at empty prompt)
  Tab                   Autocomplete
  Up/Down               Navigate history
"#
    );
}

fn handle_import(state: &mut ReplState, input: &str) {
    let file_id = state
        .source_map
        .add_file("<repl>".to_string(), input.to_string());
    let (import, diagnostics) = bhc_parser::parse_import_decl(input, file_id);

    if !diagnostics.is_empty() {
        let renderer = DiagnosticRenderer::new(&state.source_map);
        renderer.render_all(&diagnostics);
        if diagnostics.iter().any(|d| d.is_error()) {
            return;
        }
    }

    if let Some(decl) = import {
        let module_name = decl
            .module
            .parts
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(".");
        let desc = format_import_description(&decl);
        state.accumulated_imports.push(decl);
        state.modules_in_scope.insert(module_name);
        println!("{}", desc);
    }
}

fn format_import_description(decl: &bhc_ast::ImportDecl) -> String {
    let module_name = decl
        .module
        .parts
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(".");

    let mut desc = String::from("import ");
    if decl.qualified {
        desc.push_str("qualified ");
    }
    desc.push_str(&module_name);
    if let Some(ref alias) = decl.alias {
        let alias_name = alias
            .parts
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(".");
        desc.push_str(" as ");
        desc.push_str(&alias_name);
    }
    desc
}

fn handle_show(state: &ReplState, args: &str) {
    match args.trim() {
        "bindings" | "b" => show_bindings(state),
        "imports" | "i" => show_repl_imports(state),
        "modules" | "m" => show_modules_in_scope(state),
        "" => {
            println!("Profile: {:?}", state.profile);
            println!();
            show_repl_imports(state);
            println!();
            show_bindings(state);
        }
        other => {
            println!("Unknown :show target: {}", other);
            println!("Usage: :show [bindings|imports|modules]");
        }
    }
}

fn show_bindings(state: &ReplState) {
    if state.bindings.is_empty() {
        println!("No bindings defined.");
        return;
    }
    for (name, _ty, value, _) in &state.bindings {
        println!("  {} :: {}", name, type_from_value(value));
    }
}

fn show_repl_imports(state: &ReplState) {
    if state.accumulated_imports.is_empty() {
        println!("No imports (implicit Prelude only).");
        return;
    }
    for decl in &state.accumulated_imports {
        println!("  {}", format_import_description(decl));
    }
}

fn show_modules_in_scope(state: &ReplState) {
    let mut mods: Vec<_> = state.modules_in_scope.iter().collect();
    mods.sort();
    for m in mods {
        println!("  {}", m);
    }
}

fn update_completion_identifiers(state: &ReplState) {
    if let Ok(mut ids) = state.completion_identifiers.lock() {
        ids.clear();
        for (name, _, _, _) in &state.bindings {
            ids.push(name.clone());
        }
        for (name, _) in &state.loaded_types {
            ids.push(name.clone());
        }
        ids.extend(PRELUDE_NAMES.iter().map(|s| s.to_string()));
    }
}

fn eval_input(state: &mut ReplState, input: &str) {
    use std::time::Instant;

    let start = Instant::now();

    // Handle let declarations: `let name = expr`
    let trimmed = input.trim();
    if let Some(rest) = trimmed.strip_prefix("let ") {
        if let Some(eq_pos) = rest.find('=') {
            let name = rest[..eq_pos].trim().to_string();
            let expr_str = rest[eq_pos + 1..].trim();
            if !name.is_empty() && !expr_str.is_empty() {
                eval_let_binding(state, &name, expr_str);
                return;
            }
        }
    }

    // Add to source map
    let file_id = state
        .source_map
        .add_file("<repl>".to_string(), input.to_string());

    // Parse
    let (expr, diagnostics) = parse_expr(input, file_id);

    if !diagnostics.is_empty() {
        let renderer = DiagnosticRenderer::new(&state.source_map);
        renderer.render_all(&diagnostics);

        if diagnostics.iter().any(|d| d.is_error()) {
            return;
        }
    }

    let Some(ast_expr) = expr else {
        return;
    };

    // Type check (placeholder)
    let ty = infer_type(state, &ast_expr).unwrap_or_else(|_| state.type_ctx.fresh_ty());

    // Evaluate
    match evaluate_expr(state, &ast_expr) {
        Ok(value) => {
            // Store as binding
            let name = state.next_binding_name();
            state
                .bindings
                .push((name.clone(), ty.clone(), value.clone(), None));

            // Print IO output if any, otherwise print the value
            let io_output = state.evaluator.take_io_output();
            if !io_output.is_empty() {
                print!("{io_output}");
            } else {
                print_value(&state.evaluator, &value);
            }

            // Print type if enabled
            if state.options.show_types {
                let ty_str = if let Some(ref ty) = state.last_inferred_type {
                    format!("{ty}")
                } else {
                    type_from_value(&value)
                };
                println!("  :: {}", ty_str);
                state.last_inferred_type = None;
            }

            // Print timing if enabled
            if state.options.show_timing {
                let elapsed = start.elapsed();
                println!("  ({:.3}ms)", elapsed.as_secs_f64() * 1000.0);
            }

            update_completion_identifiers(state);
        }
        Err(e) => {
            eprintln!("Evaluation error: {}", e);
        }
    }
}

/// Evaluate a `let name = expr` declaration and store the binding.
fn eval_let_binding(state: &mut ReplState, name: &str, expr_str: &str) {
    let file_id = state
        .source_map
        .add_file("<repl>".to_string(), expr_str.to_string());

    let (expr, diagnostics) = parse_expr(expr_str, file_id);

    if !diagnostics.is_empty() {
        let renderer = DiagnosticRenderer::new(&state.source_map);
        renderer.render_all(&diagnostics);

        if diagnostics.iter().any(|d| d.is_error()) {
            return;
        }
    }

    let Some(ast_expr) = expr else {
        return;
    };

    let ty = infer_type(state, &ast_expr).unwrap_or_else(|_| state.type_ctx.fresh_ty());

    match evaluate_expr(state, &ast_expr) {
        Ok(value) => {
            // Consume any IO output
            let io_output = state.evaluator.take_io_output();
            if !io_output.is_empty() {
                print!("{io_output}");
            }

            let type_str = type_from_value(&value);
            let display_val = value
                .clone()
                .deep_force(&state.evaluator)
                .unwrap_or_else(|_| value.clone());
            println!("{} :: {} = {}", name, type_str, display_val);

            // Store in the evaluator's named bindings for persistence
            let sym = bhc_intern::Symbol::intern(name);
            state.evaluator.set_named_binding(sym, value.clone());

            state
                .bindings
                .push((name.to_string(), ty, value, None));

            update_completion_identifiers(state);
        }
        Err(e) => {
            eprintln!("Evaluation error: {}", e);
        }
    }
}

fn infer_type(state: &mut ReplState, _expr: &AstExpr) -> Result<Ty, String> {
    // Use type context to infer type
    // For now, return a placeholder — actual types are inferred from values post-evaluation
    Ok(state.type_ctx.fresh_ty())
}

/// Infer a display type string from a runtime Value.
fn type_from_value(value: &Value) -> String {
    match value {
        Value::Int(_) => "Int".to_string(),
        Value::Integer(_) => "Integer".to_string(),
        Value::Float(_) => "Float".to_string(),
        Value::Double(_) => "Double".to_string(),
        Value::Char(_) => "Char".to_string(),
        Value::String(_) => "String".to_string(),
        Value::Data(d) => {
            let name = d.con.name.as_str();
            match name {
                "True" | "False" => "Bool".to_string(),
                "()" => "()".to_string(),
                _ => name.to_string(),
            }
        }
        Value::Closure(_) => "_ -> _".to_string(),
        Value::Handle(_) => "Handle".to_string(),
        Value::IORef(_) => "IORef _".to_string(),
        Value::Map(_) => "Map _ _".to_string(),
        Value::Set(_) => "Set _".to_string(),
        Value::IntMap(_) => "IntMap _".to_string(),
        Value::IntSet(_) => "IntSet".to_string(),
        Value::UArrayInt(_) => "UArray Int".to_string(),
        Value::UArrayDouble(_) => "UArray Double".to_string(),
        _ => "_".to_string(),
    }
}

fn evaluate_expr(state: &mut ReplState, expr: &AstExpr) -> Result<Value, String> {
    // 1. Wrap expression in a synthetic module: module REPL where { it = <expr> }
    let module = wrap_expr_as_module(expr, &state.accumulated_imports);

    // 2. Lower AST -> HIR
    let mut lower_ctx = bhc_lower::LowerContext::with_builtins();

    // Register previously defined bindings (from REPL and :load) so they resolve
    for (name, _, _, _) in &state.bindings {
        let sym = bhc_intern::Symbol::intern(name);
        let def_id = lower_ctx.fresh_def_id();
        lower_ctx.define(def_id, sym, bhc_lower::DefKind::Value, bhc_span::Span::default());
        lower_ctx.bind_value(sym, def_id);
    }

    let config = bhc_lower::LowerConfig {
        include_builtins: true,
        warn_unused: false,
        search_paths: vec![],
    };
    let hir = bhc_lower::lower_module(&mut lower_ctx, &module, &config)
        .map_err(|e| format!("Lowering error: {e}"))?;

    // 2b. Type check (best-effort; errors are non-fatal for evaluation)
    let file_id = FileId(0);
    let typed = bhc_typeck::type_check_module_with_defs(&hir, file_id, Some(&lower_ctx.defs));
    if let Ok(ref typed_module) = typed {
        // Try to find the "it" DefId and extract its type scheme
        for (def_id, scheme) in &typed_module.def_schemes {
            if let Some(info) = lower_ctx.defs.get(def_id) {
                if info.name.as_str() == "it" {
                    // Store the inferred type for display
                    state.last_inferred_type = Some(scheme.ty.clone());
                    break;
                }
            }
        }
    }

    // 2c. Build def map for Core lowering
    let def_map: bhc_hir_to_core::DefMap = lower_ctx
        .defs
        .iter()
        .map(|(def_id, def_info)| {
            (
                *def_id,
                bhc_hir_to_core::DefInfo {
                    id: *def_id,
                    name: def_info.name,
                },
            )
        })
        .collect();

    // 3. Lower HIR -> Core
    let core = bhc_hir_to_core::lower_module_with_defs(&hir, Some(&def_map), None)
        .map_err(|e| format!("Core lowering error: {e}"))?;

    // 4. Find the "it" binding
    let it_expr = find_it_binding(&core)
        .ok_or_else(|| "Failed to find 'it' binding in lowered Core".to_string())?;

    // 5. Evaluate
    let env = bhc_core::eval::Env::new();
    let result = state
        .evaluator
        .eval(it_expr, &env)
        .map_err(|e| format!("{e}"))?;

    Ok(result)
}

/// Wrap an AST expression in a synthetic module for lowering.
///
/// Creates: `module REPL where { it = <expr> }`
fn wrap_expr_as_module(
    expr: &AstExpr,
    imports: &[bhc_ast::ImportDecl],
) -> bhc_ast::Module {
    use bhc_ast::{Clause, Decl, FunBind, ModuleName, Rhs};
    use bhc_intern::Ident;
    use bhc_span::Span;

    let span = expr.span();
    let it_ident = Ident::new(bhc_intern::Symbol::intern("it"));
    let clause = Clause {
        pats: vec![],
        rhs: Rhs::Simple(expr.clone(), span),
        wheres: vec![],
        span,
    };
    let fun_bind = FunBind {
        doc: None,
        name: it_ident,
        clauses: vec![clause],
        span,
    };
    bhc_ast::Module {
        doc: None,
        pragmas: vec![],
        name: Some(ModuleName {
            parts: vec![bhc_intern::Symbol::intern("REPL")],
            span: Span::DUMMY,
        }),
        exports: None,
        imports: imports.to_vec(),
        decls: vec![Decl::FunBind(fun_bind)],
        span,
    }
}

/// Find the "it" binding in a Core module.
fn find_it_binding(core: &bhc_core::CoreModule) -> Option<&bhc_core::Expr> {
    for bind in &core.bindings {
        match bind {
            bhc_core::Bind::NonRec(var, expr) if var.name.as_str() == "it" => {
                return Some(expr);
            }
            bhc_core::Bind::Rec(bindings) => {
                for (var, expr) in bindings {
                    if var.name.as_str() == "it" {
                        return Some(expr);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn show_type(state: &mut ReplState, expr: &str) {
    let trimmed = expr.trim();

    // Check REPL bindings first
    for (name, _ty, _val, _) in &state.bindings {
        if name == trimmed {
            let ty_str = type_from_value(_val);
            println!("{} :: {}", name, ty_str);
            return;
        }
    }

    // Check loaded module types
    for (name, ty) in &state.loaded_types {
        if name == trimmed {
            println!("{} :: {}", name, ty);
            return;
        }
    }

    let file_id = state
        .source_map
        .add_file("<repl>".to_string(), expr.to_string());

    let (parsed, diagnostics) = parse_expr(expr, file_id);

    if !diagnostics.is_empty() {
        let renderer = DiagnosticRenderer::new(&state.source_map);
        renderer.render_all(&diagnostics);
        return;
    }

    let Some(ast_expr) = parsed else {
        return;
    };

    // Build a synthetic module for type checking
    let module = wrap_expr_as_module(&ast_expr, &state.accumulated_imports);
    let mut lower_ctx = bhc_lower::LowerContext::with_builtins();
    let config = bhc_lower::LowerConfig {
        include_builtins: true,
        warn_unused: false,
        search_paths: vec![],
    };

    let hir = match bhc_lower::lower_module(&mut lower_ctx, &module, &config) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Lowering error: {e}");
            return;
        }
    };

    match bhc_typeck::type_check_module_with_defs(&hir, file_id, Some(&lower_ctx.defs)) {
        Ok(typed_module) => {
            // Find the "it" binding's type
            for (def_id, scheme) in &typed_module.def_schemes {
                if let Some(info) = lower_ctx.defs.get(def_id) {
                    if info.name.as_str() == "it" {
                        println!("{} :: {}", expr, scheme.ty);
                        return;
                    }
                }
            }
            // Fallback if "it" not found in typed module
            match infer_type(state, &ast_expr) {
                Ok(ty) => println!("{} :: {}", expr, format_type(&ty)),
                Err(e) => eprintln!("Type error: {}", e),
            }
        }
        Err(diagnostics) => {
            let renderer = DiagnosticRenderer::new(&state.source_map);
            renderer.render_all(&diagnostics);
        }
    }
}

fn format_kind(kind: &bhc_types::Kind) -> String {
    match kind {
        bhc_types::Kind::Star => "*".to_string(),
        bhc_types::Kind::Constraint => "Constraint".to_string(),
        bhc_types::Kind::Var(n) => format!("k{n}"),
        bhc_types::Kind::Nat => "Nat".to_string(),
        bhc_types::Kind::List(inner) => format!("[{}]", format_kind(inner)),
        bhc_types::Kind::Arrow(a, b) => {
            let left = match a.as_ref() {
                bhc_types::Kind::Arrow(_, _) => format!("({})", format_kind(a)),
                _ => format_kind(a),
            };
            format!("{} -> {}", left, format_kind(b))
        }
    }
}

fn show_kind(_state: &mut ReplState, type_str: &str) {
    let builtins = bhc_typeck::builtins::Builtins::new();
    let name = type_str.trim();

    // Check all known type constructors
    let all_cons: Vec<&bhc_types::TyCon> = vec![
        &builtins.int_con,
        &builtins.float_con,
        &builtins.char_con,
        &builtins.bool_con,
        &builtins.string_con,
        &builtins.list_con,
        &builtins.maybe_con,
        &builtins.either_con,
        &builtins.io_con,
        &builtins.tensor_con,
        &builtins.dyn_tensor_con,
        &builtins.shape_witness_con,
    ];

    // Also accept common aliases
    let lookup_name = match name {
        "List" | "[]" => "[]",
        other => other,
    };

    for con in all_cons {
        if con.name.as_str() == lookup_name {
            println!("{} :: {}", name, format_kind(&con.kind));
            return;
        }
    }

    // Check additional well-known types
    match name {
        "Integer" | "Double" | "Word" | "Ordering" => {
            println!("{} :: *", name);
        }
        "(,)" | "Tuple2" => {
            println!("{} :: * -> * -> *", name);
        }
        "(,,)" | "Tuple3" => {
            println!("{} :: * -> * -> * -> *", name);
        }
        _ => {
            println!("'{}' is not in scope as a type constructor", name);
        }
    }
}

fn show_info(state: &mut ReplState, name: &str) {
    // 1. Check REPL bindings
    for (n, ty, _, _) in &state.bindings {
        if n == name {
            println!("{} :: {}", name, format_type(ty));
            println!("  -- Defined at <repl>");
            return;
        }
    }

    // 2. Check loaded module types
    for (n, ty) in &state.loaded_types {
        if n == name {
            println!("{} :: {}", name, format_type(ty));
            println!("  -- Defined in loaded module");
            return;
        }
    }

    // 3. Try type-checking the name as an expression
    let file_id = state
        .source_map
        .add_file("<repl>".to_string(), name.to_string());

    let (parsed, _) = parse_expr(name, file_id);
    if let Some(ast_expr) = parsed {
        let module = wrap_expr_as_module(&ast_expr, &state.accumulated_imports);
        let mut lower_ctx = bhc_lower::LowerContext::with_builtins();
        let config = bhc_lower::LowerConfig {
            include_builtins: true,
            warn_unused: false,
            search_paths: vec![],
        };

        if let Ok(hir) = bhc_lower::lower_module(&mut lower_ctx, &module, &config) {
            if let Ok(typed_module) =
                bhc_typeck::type_check_module_with_defs(&hir, file_id, Some(&lower_ctx.defs))
            {
                for (def_id, scheme) in &typed_module.def_schemes {
                    if let Some(info) = lower_ctx.defs.get(def_id) {
                        if info.name.as_str() == "it" {
                            println!("{} :: {}", name, scheme.ty);
                            println!("  -- Defined in 'Prelude'");
                            return;
                        }
                    }
                }
            }
        }
    }

    // 4. Check type constructors for :info on types
    let builtins = bhc_typeck::builtins::Builtins::new();
    let type_cons: Vec<(&str, &bhc_types::TyCon)> = vec![
        ("Int", &builtins.int_con),
        ("Float", &builtins.float_con),
        ("Char", &builtins.char_con),
        ("Bool", &builtins.bool_con),
        ("String", &builtins.string_con),
        ("Maybe", &builtins.maybe_con),
        ("Either", &builtins.either_con),
        ("IO", &builtins.io_con),
        ("Tensor", &builtins.tensor_con),
    ];

    for (type_name, con) in type_cons {
        if type_name == name {
            println!("type {} :: {}", name, format_kind(&con.kind));
            println!("  -- Defined in 'Prelude'");
            return;
        }
    }

    println!("'{}' is not in scope", name);
}

fn load_file(state: &mut ReplState, path: &PathBuf) -> Result<(), ReplError> {
    if !path.exists() {
        return Err(ReplError::FileNotFound(path.clone()));
    }

    println!("[1 of 1] Compiling {}...", path.display());

    // Read and parse file
    let content =
        std::fs::read_to_string(path).map_err(|_| ReplError::FileNotFound(path.clone()))?;

    let file_id = state
        .source_map
        .add_file(path.display().to_string(), content.clone());

    // Parse module
    let (module, diagnostics) = bhc_parser::parse_module(&content, file_id);

    if !diagnostics.is_empty() {
        let renderer = DiagnosticRenderer::new(&state.source_map);
        renderer.render_all(&diagnostics);

        if diagnostics.iter().any(|d| d.is_error()) {
            return Err(ReplError::Parse("Parse errors in module".into()));
        }
    }

    let Some(module) = module else {
        return Err(ReplError::Parse("Failed to parse module".into()));
    };

    let module_name = module
        .name
        .as_ref()
        .map(|n| n.parts.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("."))
        .unwrap_or_else(|| "Main".to_string());

    // 1. Lower AST → HIR
    let mut lower_ctx = bhc_lower::LowerContext::with_builtins();
    let config = bhc_lower::LowerConfig {
        include_builtins: true,
        warn_unused: false,
        search_paths: vec![],
    };
    let hir = bhc_lower::lower_module(&mut lower_ctx, &module, &config)
        .map_err(|e| ReplError::Parse(format!("Lowering error: {e}")))?;

    // 2. Type check (best-effort)
    let typed = bhc_typeck::type_check_module_with_defs(&hir, file_id, Some(&lower_ctx.defs));

    // 3. Build def map for Core lowering
    let def_map: bhc_hir_to_core::DefMap = lower_ctx
        .defs
        .iter()
        .map(|(def_id, def_info)| {
            (
                *def_id,
                bhc_hir_to_core::DefInfo {
                    id: *def_id,
                    name: def_info.name,
                },
            )
        })
        .collect();

    // 4. Lower HIR → Core
    let core = bhc_hir_to_core::lower_module_with_defs(&hir, Some(&def_map), None)
        .map_err(|e| ReplError::Parse(format!("Core lowering error: {e}")))?;

    // 5. Extract type information from typed module
    if let Ok(ref typed_module) = typed {
        for (def_id, scheme) in &typed_module.def_schemes {
            if let Some(info) = lower_ctx.defs.get(def_id) {
                let name = info.name.as_str().to_string();
                state.loaded_types.push((name, scheme.ty.clone()));
            }
        }
    }

    // 6. Evaluate each binding and store in the evaluator
    let env = bhc_core::eval::Env::new();
    let mut def_count = 0;

    for bind in &core.bindings {
        match bind {
            bhc_core::Bind::NonRec(var, expr) => {
                let name = var.name.as_str().to_string();
                match state.evaluator.eval(expr, &env) {
                    Ok(value) => {
                        // Consume any IO output produced during evaluation
                        let io_output = state.evaluator.take_io_output();
                        if !io_output.is_empty() {
                            print!("{io_output}");
                        }

                        let sym = bhc_intern::Symbol::intern(&name);
                        state.evaluator.set_named_binding(sym, value.clone());

                        let ty = state
                            .loaded_types
                            .iter()
                            .find(|(n, _)| n == &name)
                            .map(|(_, t)| t.clone())
                            .unwrap_or_else(|| Ty::Error);

                        state
                            .bindings
                            .push((name.clone(), ty, value, None));
                        state.loaded_binding_names.push(name);
                        def_count += 1;
                    }
                    Err(e) => {
                        eprintln!("  Warning: could not evaluate '{}': {}", name, e);
                    }
                }
            }
            bhc_core::Bind::Rec(bindings) => {
                for (var, expr) in bindings {
                    let name = var.name.as_str().to_string();
                    match state.evaluator.eval(expr, &env) {
                        Ok(value) => {
                            let io_output = state.evaluator.take_io_output();
                            if !io_output.is_empty() {
                                print!("{io_output}");
                            }

                            let sym = bhc_intern::Symbol::intern(&name);
                            state.evaluator.set_named_binding(sym, value.clone());

                            let ty = state
                                .loaded_types
                                .iter()
                                .find(|(n, _)| n == &name)
                                .map(|(_, t)| t.clone())
                                .unwrap_or_else(|| Ty::Error);

                            state
                                .bindings
                                .push((name.clone(), ty, value, None));
                            state.loaded_binding_names.push(name);
                            def_count += 1;
                        }
                        Err(e) => {
                            eprintln!("  Warning: could not evaluate '{}': {}", name, e);
                        }
                    }
                }
            }
        }
    }

    state.loaded_files.push(path.clone());
    state.modules_in_scope.insert(module_name.clone());
    println!("Ok, {} definitions loaded from module {}.", def_count, module_name);

    Ok(())
}

fn reload_modules(state: &mut ReplState) {
    if state.loaded_files.is_empty() {
        println!("No modules loaded.");
        return;
    }

    println!("Reloading...");
    let files: Vec<PathBuf> = state.loaded_files.clone();

    // Clear evaluator named bindings for loaded file definitions
    let symbols: Vec<_> = state.loaded_binding_names.iter()
        .map(|n| bhc_intern::Symbol::intern(n))
        .collect();
    state.evaluator.remove_named_bindings(&symbols);

    // Clear bindings that came from loaded files
    let loaded_names: HashSet<String> = state.loaded_binding_names.drain(..).collect();
    state.bindings.retain(|(name, _, _, _)| !loaded_names.contains(name));
    state.loaded_types.clear();
    state.loaded_files.clear();
    state.accumulated_imports.clear();

    for file in files {
        if let Err(e) = load_file(state, &file) {
            eprintln!("Error reloading {}: {}", file.display(), e);
        }
    }
}

fn browse_module(state: &ReplState, module: Option<&str>) {
    let module_name = module.unwrap_or("Prelude");

    // Show REPL-defined and loaded bindings when no module specified or browsing a loaded module
    let has_real_data = !state.bindings.is_empty() || !state.loaded_types.is_empty();

    if module.is_none() && has_real_data {
        // Show all known bindings
        if !state.bindings.is_empty() {
            println!("-- REPL bindings -------");
            for (name, ty, _, _) in &state.bindings {
                let ty_str = format_type(ty);
                println!("{} :: {}", name, ty_str);
            }
        }

        // Show loaded types not already shown as bindings
        let binding_names: HashSet<&str> = state.bindings.iter().map(|(n, _, _, _)| n.as_str()).collect();
        let extra_types: Vec<_> = state
            .loaded_types
            .iter()
            .filter(|(n, _)| !binding_names.contains(n.as_str()))
            .collect();
        if !extra_types.is_empty() {
            println!("-- Loaded definitions -------");
            for (name, ty) in extra_types {
                println!("{} :: {}", name, format_type(ty));
            }
        }
        println!();
    }

    if !state.modules_in_scope.contains(module_name) && module.is_some() {
        println!("Module '{}' is not in scope", module_name);
        return;
    }

    // Fallback Prelude listing for reference
    if module_name == "Prelude" {
        println!("-- {} -------", module_name);
        println!("(++) :: [a] -> [a] -> [a]");
        println!("map :: (a -> b) -> [a] -> [b]");
        println!("filter :: (a -> Bool) -> [a] -> [a]");
        println!("foldr :: (a -> b -> b) -> b -> [a] -> b");
        println!("foldl :: (b -> a -> b) -> b -> [a] -> b");
        println!("head :: [a] -> a");
        println!("tail :: [a] -> [a]");
        println!("null :: [a] -> Bool");
        println!("length :: [a] -> Int");
        println!("sum :: Num a => [a] -> a");
        println!("product :: Num a => [a] -> a");
        println!("print :: Show a => a -> IO ()");
        println!("putStrLn :: String -> IO ()");
    } else if module_name == "Data.List" {
        println!("-- {} -------", module_name);
        println!("sort :: Ord a => [a] -> [a]");
        println!("nub :: Eq a => [a] -> [a]");
        println!("group :: Eq a => [a] -> [[a]]");
        println!("intersperse :: a -> [a] -> [a]");
        println!("intercalate :: [a] -> [[a]] -> [a]");
    }
}

fn set_option(state: &mut ReplState, opt: &str) {
    match opt.trim() {
        "+t" => {
            state.options.show_types = true;
            println!("Type display on");
        }
        "-t" => {
            state.options.show_types = false;
            println!("Type display off");
        }
        "+s" => {
            state.options.show_timing = true;
            println!("Timing on");
        }
        "-s" => {
            state.options.show_timing = false;
            println!("Timing off");
        }
        "+m" => {
            state.options.multiline = true;
            println!("Multiline mode on");
        }
        "-m" => {
            state.options.multiline = false;
            println!("Multiline mode off");
        }
        "+w" => {
            state.options.warnings = true;
            println!("Warnings on");
        }
        "-w" => {
            state.options.warnings = false;
            println!("Warnings off");
        }
        opt if opt.starts_with("profile ") => {
            let profile_name = opt.strip_prefix("profile ").unwrap().trim();
            let new_profile = match profile_name {
                "default" => Profile::Default,
                "server" => Profile::Server,
                "numeric" => Profile::Numeric,
                "edge" => Profile::Edge,
                "realtime" => Profile::Realtime,
                "embedded" => Profile::Embedded,
                _ => {
                    println!("Unknown profile: {}", profile_name);
                    println!("Valid profiles: default, server, numeric, edge, realtime, embedded");
                    return;
                }
            };

            state.profile = new_profile;

            // Recreate evaluator with new profile mode
            state.evaluator = Evaluator::with_profile(new_profile);

            // Re-register all named bindings in the new evaluator
            for (name, _ty, value, _var_id) in &state.bindings {
                let sym = bhc_intern::Symbol::intern(name);
                state.evaluator.set_named_binding(sym, value.clone());
            }

            let mode = if matches!(new_profile, Profile::Numeric | Profile::Embedded) {
                "strict"
            } else {
                "lazy"
            };
            println!("Profile set to: {:?} (evaluation: {})", new_profile, mode);
        }
        _ => {
            println!("Unknown option: {}", opt);
            println!("Use :set +t/-t, +s/-s, +m/-m, +w/-w, or 'profile <name>'");
        }
    }
}

fn unset_option(state: &mut ReplState, opt: &str) {
    match opt.trim() {
        "t" => {
            state.options.show_types = false;
            println!("Type display off");
        }
        "s" => {
            state.options.show_timing = false;
            println!("Timing off");
        }
        "m" => {
            state.options.multiline = false;
            println!("Multiline mode off");
        }
        "w" => {
            state.options.warnings = false;
            println!("Warnings off");
        }
        _ => {
            println!("Unknown option: {}", opt);
        }
    }
}

fn show_options(state: &ReplState) {
    println!("Current options:");
    println!(
        "  show types:   {} (:set {}t)",
        state.options.show_types,
        if state.options.show_types { "-" } else { "+" }
    );
    println!(
        "  show timing:  {} (:set {}s)",
        state.options.show_timing,
        if state.options.show_timing { "-" } else { "+" }
    );
    println!(
        "  multiline:    {} (:set {}m)",
        state.options.multiline,
        if state.options.multiline { "-" } else { "+" }
    );
    println!(
        "  warnings:     {} (:set {}w)",
        state.options.warnings,
        if state.options.warnings { "-" } else { "+" }
    );
    println!("  profile:      {:?}", state.profile);
}

fn set_modules(state: &mut ReplState, args: &str) {
    let parts: Vec<&str> = args.split_whitespace().collect();

    for part in parts {
        if let Some(module) = part.strip_prefix('+') {
            state.modules_in_scope.insert(module.to_string());
            println!("Added {} to scope", module);
        } else if let Some(module) = part.strip_prefix('-') {
            state.modules_in_scope.remove(module);
            println!("Removed {} from scope", module);
        } else {
            // Replace all modules
            state.modules_in_scope.clear();
            state.modules_in_scope.insert(part.to_string());
            println!("Scope set to: {}", part);
        }
    }
}

fn show_modules(state: &ReplState) {
    println!("Modules in scope:");
    for module in &state.modules_in_scope {
        println!("  {}", module);
    }
}

fn run_shell_command(cmd: &str) {
    use std::process::Command;

    let output = if cfg!(target_os = "windows") {
        Command::new("cmd").args(["/C", cmd]).output()
    } else {
        Command::new("sh").args(["-c", cmd]).output()
    };

    match output {
        Ok(output) => {
            if !output.stdout.is_empty() {
                print!("{}", String::from_utf8_lossy(&output.stdout));
            }
            if !output.stderr.is_empty() {
                eprint!("{}", String::from_utf8_lossy(&output.stderr));
            }
        }
        Err(e) => {
            eprintln!("Failed to run command: {}", e);
        }
    }
}

fn format_type(ty: &Ty) -> String {
    format!("{ty}")
}

fn print_value(evaluator: &Evaluator, value: &Value) {
    match value.clone().deep_force(evaluator) {
        Ok(forced) => println!("{forced}"),
        Err(_) => println!("{value}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_parsing() {
        let mut state = ReplState::new(Profile::Default, false);

        // Test help command
        assert!(matches!(
            handle_command(&mut state, ":help"),
            CommandResult::Continue
        ));

        // Test quit command
        assert!(matches!(
            handle_command(&mut state, ":quit"),
            CommandResult::Quit
        ));
    }

    #[test]
    fn test_options() {
        let mut state = ReplState::new(Profile::Default, false);

        assert!(state.options.show_types);
        set_option(&mut state, "-t");
        assert!(!state.options.show_types);
        set_option(&mut state, "+t");
        assert!(state.options.show_types);
    }

    #[test]
    fn test_modules() {
        let mut state = ReplState::new(Profile::Default, false);
        state.load_prelude();

        assert!(state.modules_in_scope.contains("Prelude"));

        set_modules(&mut state, "+Data.List");
        assert!(state.modules_in_scope.contains("Data.List"));

        set_modules(&mut state, "-Prelude");
        assert!(!state.modules_in_scope.contains("Prelude"));
    }
}
