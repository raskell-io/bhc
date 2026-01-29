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
    /// User-defined bindings
    bindings: Vec<(String, Ty, Value)>,
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

/// Rustyline helper for completion and hints
struct ReplHelper {
    commands: Vec<String>,
    modules: Vec<String>,
}

impl ReplHelper {
    fn new() -> Self {
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
            ],
            modules: vec![
                "Prelude".to_string(),
                "Data.List".to_string(),
                "Data.Maybe".to_string(),
                "Data.Either".to_string(),
                "Control.Monad".to_string(),
            ],
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

        // Module completion after :load, :browse, :module
        if line.starts_with(":load ")
            || line.starts_with(":browse ")
            || line.starts_with(":module ")
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

        Ok((pos, completions))
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

    let helper = ReplHelper::new();
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
        let prompt = format!("bhci:{:03}> ", line_num);

        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();

                if line.is_empty() {
                    continue;
                }

                // Handle commands
                if line.starts_with(':') {
                    match handle_command(&mut state, line) {
                        CommandResult::Continue => {}
                        CommandResult::Quit => break,
                    }
                } else {
                    // Evaluate expression
                    eval_input(&mut state, line);
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
  :module [+/-] <mod>   Add/remove modules from scope
  :set [option]         Show or set REPL options
  :unset <option>       Unset a REPL option
  :cd <dir>             Change working directory
  :!<cmd>               Run shell command

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

fn eval_input(state: &mut ReplState, input: &str) {
    use std::time::Instant;

    let start = Instant::now();

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

    // Type check
    match infer_type(state, &ast_expr) {
        Ok(ty) => {
            // Evaluate
            match evaluate_expr(state, &ast_expr) {
                Ok(value) => {
                    // Store as binding
                    let name = state.next_binding_name();
                    state
                        .bindings
                        .push((name.clone(), ty.clone(), value.clone()));

                    // Print result
                    print_value(&value);

                    // Print type if enabled
                    if state.options.show_types {
                        println!("  :: {}", format_type(&ty));
                    }

                    // Print timing if enabled
                    if state.options.show_timing {
                        let elapsed = start.elapsed();
                        println!("  ({:.3}ms)", elapsed.as_secs_f64() * 1000.0);
                    }
                }
                Err(e) => {
                    eprintln!("Evaluation error: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Type error: {}", e);
        }
    }
}

fn infer_type(state: &mut ReplState, _expr: &AstExpr) -> Result<Ty, String> {
    // Use type context to infer type
    // For now, return a placeholder
    Ok(state.type_ctx.fresh_ty())
}

fn evaluate_expr(state: &mut ReplState, _expr: &AstExpr) -> Result<Value, String> {
    // Lower AST -> HIR -> Core -> Evaluate
    // For now, return a placeholder value
    Ok(Value::Int(42))
}

fn show_type(state: &mut ReplState, expr: &str) {
    let file_id = state
        .source_map
        .add_file("<repl>".to_string(), expr.to_string());

    let (parsed, diagnostics) = parse_expr(expr, file_id);

    if !diagnostics.is_empty() {
        let renderer = DiagnosticRenderer::new(&state.source_map);
        renderer.render_all(&diagnostics);
        return;
    }

    if let Some(ast_expr) = parsed {
        match infer_type(state, &ast_expr) {
            Ok(ty) => {
                println!("{} :: {}", expr, format_type(&ty));
            }
            Err(e) => {
                eprintln!("Type error: {}", e);
            }
        }
    }
}

fn show_kind(state: &mut ReplState, type_str: &str) {
    // Parse and show kind of type
    println!("{} :: *", type_str);
    let _ = state; // Use state for actual kind inference
}

fn show_info(state: &mut ReplState, name: &str) {
    // Look up name in context and show info
    println!("-- Defined at <unknown>");

    // Check bindings
    for (n, ty, _) in &state.bindings {
        if n == name {
            println!("{} :: {}", name, format_type(ty));
            return;
        }
    }

    // Check standard library
    match name {
        "map" => {
            println!("map :: (a -> b) -> [a] -> [b]");
            println!("  -- Defined in 'Prelude'");
        }
        "filter" => {
            println!("filter :: (a -> Bool) -> [a] -> [a]");
            println!("  -- Defined in 'Prelude'");
        }
        "foldr" => {
            println!("foldr :: (a -> b -> b) -> b -> [a] -> b");
            println!("  -- Defined in 'Prelude'");
        }
        "foldl" => {
            println!("foldl :: (b -> a -> b) -> b -> [a] -> b");
            println!("  -- Defined in 'Prelude'");
        }
        "foldl'" => {
            println!("foldl' :: (b -> a -> b) -> b -> [a] -> b");
            println!("  -- Defined in 'Data.List'");
            println!("  -- Strict left fold");
        }
        _ => {
            println!("'{}' is not in scope", name);
        }
    }
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

    if module.is_some() {
        state.loaded_files.push(path.clone());
        println!("Ok, module loaded.");
    }

    Ok(())
}

fn reload_modules(state: &mut ReplState) {
    if state.loaded_files.is_empty() {
        println!("No modules loaded.");
        return;
    }

    println!("Reloading...");
    let files: Vec<PathBuf> = state.loaded_files.clone();
    state.loaded_files.clear();

    for file in files {
        if let Err(e) = load_file(state, &file) {
            eprintln!("Error reloading {}: {}", file.display(), e);
        }
    }
}

fn browse_module(state: &ReplState, module: Option<&str>) {
    let module_name = module.unwrap_or("Prelude");

    if !state.modules_in_scope.contains(module_name) {
        println!("Module '{}' is not in scope", module_name);
        return;
    }

    println!("-- {} -------", module_name);

    match module_name {
        "Prelude" => {
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
        }
        "Data.List" => {
            println!("sort :: Ord a => [a] -> [a]");
            println!("nub :: Eq a => [a] -> [a]");
            println!("group :: Eq a => [a] -> [[a]]");
            println!("intersperse :: a -> [a] -> [a]");
            println!("intercalate :: [a] -> [[a]] -> [a]");
        }
        _ => {
            println!("(no exports)");
        }
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
            match profile_name {
                "default" => state.profile = Profile::Default,
                "server" => state.profile = Profile::Server,
                "numeric" => state.profile = Profile::Numeric,
                "edge" => state.profile = Profile::Edge,
                "realtime" => state.profile = Profile::Realtime,
                "embedded" => state.profile = Profile::Embedded,
                _ => {
                    println!("Unknown profile: {}", profile_name);
                    return;
                }
            }
            println!("Profile set to: {:?}", state.profile);
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
    format!("{:?}", ty)
}

fn print_value(value: &Value) {
    match value {
        Value::Int(n) => println!("{}", n),
        Value::Float(f) => println!("{}", f),
        Value::Double(d) => println!("{}", d),
        Value::Char(c) => println!("'{}'", c),
        Value::String(s) => println!("\"{}\"", s),
        Value::Data(d) => {
            let name = d.con.name.as_str();
            match (name, d.args.as_slice()) {
                ("True", []) => println!("True"),
                ("False", []) => println!("False"),
                ("()", []) => println!("()"),
                _ => println!("{:?}", value),
            }
        }
        _ => println!("{:?}", value),
    }
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Double(d) => d.to_string(),
        Value::Char(c) => format!("'{}'", c),
        Value::String(s) => format!("\"{}\"", s),
        _ => format!("{:?}", value),
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
