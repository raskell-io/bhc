# bhci

Interactive REPL for the Basel Haskell Compiler.

## Overview

bhci (Basel Haskell Compiler Interactive) provides a Read-Eval-Print-Loop environment for interactive Haskell development. It supports expression evaluation, type queries, and incremental compilation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         bhci                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐                                                │
│  │  Input  │  ←── User input (expressions, commands)        │
│  └────┬────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Parse  │→ │  Lower  │→ │ TypeCk  │→ │  Eval   │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│       │                         │            │              │
│       ▼                         ▼            ▼              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Environment State                    │   │
│  │  - Bindings (let x = ...)                           │   │
│  │  - Type definitions (data Foo = ...)                │   │
│  │  - Loaded modules                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────┐                                                │
│  │ Output  │  ──▶ Result, type, errors                      │
│  └─────────┘                                                │
└─────────────────────────────────────────────────────────────┘
```

## Command System

### Built-in Commands

```rust
enum Command {
    /// :quit, :q - Exit REPL
    Quit,

    /// :help, :h, :? - Show help
    Help,

    /// :type <expr>, :t - Show type
    Type(String),

    /// :kind <type>, :k - Show kind
    Kind(String),

    /// :info <name>, :i - Show info about name
    Info(String),

    /// :load <file>, :l - Load file
    Load(PathBuf),

    /// :reload, :r - Reload current file
    Reload,

    /// :browse, :b - Browse loaded modules
    Browse(Option<String>),

    /// :set <option> - Set REPL option
    Set(String),

    /// :unset <option> - Unset REPL option
    Unset(String),

    /// :! <shell command> - Run shell command
    Shell(String),

    /// :dump-ir <stage> - Dump IR for last expression
    DumpIr(IrStage),
}
```

### Command Parsing

```rust
fn parse_command(input: &str) -> Result<Command, CommandError> {
    let input = input.trim();

    if !input.starts_with(':') {
        return Err(CommandError::NotACommand);
    }

    let (cmd, args) = input[1..].split_once(' ')
        .map(|(c, a)| (c, a.trim()))
        .unwrap_or((&input[1..], ""));

    match cmd {
        "quit" | "q" => Ok(Command::Quit),
        "help" | "h" | "?" => Ok(Command::Help),
        "type" | "t" => Ok(Command::Type(args.to_string())),
        "kind" | "k" => Ok(Command::Kind(args.to_string())),
        "info" | "i" => Ok(Command::Info(args.to_string())),
        "load" | "l" => Ok(Command::Load(PathBuf::from(args))),
        "reload" | "r" => Ok(Command::Reload),
        "browse" | "b" => Ok(Command::Browse(
            if args.is_empty() { None } else { Some(args.to_string()) }
        )),
        "set" => Ok(Command::Set(args.to_string())),
        "unset" => Ok(Command::Unset(args.to_string())),
        _ => Err(CommandError::Unknown(cmd.to_string())),
    }
}
```

## REPL State

```rust
pub struct ReplState {
    /// Accumulated bindings
    bindings: HashMap<Symbol, Value>,

    /// Type environment
    type_env: TypeEnv,

    /// Loaded modules
    loaded_modules: Vec<ModuleId>,

    /// Current file (for :reload)
    current_file: Option<PathBuf>,

    /// REPL options
    options: ReplOptions,

    /// History
    history: Vec<String>,

    /// Line number counter
    line_number: u32,
}

pub struct ReplOptions {
    /// Show type after evaluation (+t)
    show_type: bool,

    /// Show timing information (+s)
    show_timing: bool,

    /// Show IR after evaluation
    show_ir: Option<IrStage>,

    /// Warning flags
    warnings: HashSet<Warning>,

    /// Current profile
    profile: Profile,
}
```

## Evaluation Pipeline

```rust
impl ReplState {
    pub fn eval(&mut self, input: &str) -> Result<EvalResult, ReplError> {
        let start = Instant::now();

        // 1. Parse input
        let parsed = self.parse_input(input)?;

        // 2. Handle declarations vs expressions
        match parsed {
            Input::Decl(decl) => {
                // Add to environment
                self.add_decl(decl)?;
                Ok(EvalResult::DeclAdded)
            }
            Input::Expr(expr) => {
                // 3. Lower to HIR
                let hir = self.lower_expr(expr)?;

                // 4. Type check
                let (typed_hir, ty) = self.type_check(hir)?;

                // 5. Lower to Core
                let core = self.lower_to_core(typed_hir)?;

                // 6. Evaluate
                let value = self.evaluate(core)?;

                let elapsed = start.elapsed();

                Ok(EvalResult::Value {
                    value,
                    ty,
                    time: if self.options.show_timing { Some(elapsed) } else { None },
                })
            }
        }
    }
}
```

## Multiline Input

```rust
/// Handle multiline input with :{ ... :}
fn read_multiline(&mut self) -> String {
    let mut buffer = String::new();
    let mut depth = 1; // Already seen :{

    loop {
        let prompt = format!("bhci:{:03}| ", self.line_number);
        let line = self.read_line(&prompt);

        if line.trim() == ":}" {
            depth -= 1;
            if depth == 0 {
                break;
            }
        } else if line.trim() == ":{" {
            depth += 1;
        }

        buffer.push_str(&line);
        buffer.push('\n');
        self.line_number += 1;
    }

    buffer
}
```

## Tab Completion

```rust
impl Completer for ReplCompleter {
    fn complete(&self, line: &str, pos: usize) -> Vec<Completion> {
        let prefix = &line[..pos];

        if prefix.starts_with(':') {
            // Complete commands
            self.complete_command(&prefix[1..])
        } else {
            // Complete identifiers
            let word_start = prefix.rfind(|c: char| !c.is_alphanumeric() && c != '_')
                .map(|i| i + 1)
                .unwrap_or(0);
            let word = &prefix[word_start..];

            self.complete_identifier(word)
        }
    }

    fn complete_identifier(&self, prefix: &str) -> Vec<Completion> {
        let mut completions = Vec::new();

        // Local bindings
        for name in self.state.bindings.keys() {
            if name.as_str().starts_with(prefix) {
                completions.push(Completion::new(name.as_str()));
            }
        }

        // Prelude functions
        for name in PRELUDE_NAMES {
            if name.starts_with(prefix) {
                completions.push(Completion::new(name));
            }
        }

        completions
    }
}
```

## Error Display

```rust
fn display_error(error: &ReplError) {
    match error {
        ReplError::Parse(e) => {
            eprintln!("Parse error:");
            display_diagnostic(e);
        }
        ReplError::Type(e) => {
            eprintln!("Type error:");
            display_diagnostic(e);
        }
        ReplError::Eval(e) => {
            eprintln!("Evaluation error: {}", e);
        }
        ReplError::Command(e) => {
            eprintln!("Command error: {}", e);
        }
    }
}

fn display_diagnostic(diag: &Diagnostic) {
    // Show source context
    if let Some(span) = diag.span {
        println!("  --> {}:{}:{}", span.file, span.line, span.col);
        println!("   |");
        println!("{:3}| {}", span.line, get_source_line(span));
        println!("   | {}", "^".repeat(span.len));
    }
    println!("{}", diag.message);

    // Show suggestions
    for suggestion in &diag.suggestions {
        println!("   = help: {}", suggestion);
    }
}
```

## Profile Support

```rust
fn set_profile(&mut self, profile_name: &str) -> Result<(), ReplError> {
    let profile = match profile_name {
        "default" => Profile::Default,
        "server" => Profile::Server,
        "numeric" => Profile::Numeric,
        "edge" => Profile::Edge,
        _ => return Err(ReplError::UnknownProfile(profile_name.to_string())),
    };

    self.options.profile = profile;

    // Reinitialize with new profile
    println!("Switching to {} profile", profile_name);

    if profile == Profile::Numeric {
        println!("Note: Numeric profile uses strict evaluation");
    }

    Ok(())
}
```

## See Also

- `bhc` - Compiler CLI
- `bhi` - IR inspector
- `bhc-core` - Core IR evaluator
