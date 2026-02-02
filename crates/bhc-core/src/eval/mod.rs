//! Tree-walking interpreter for Core IR.
//!
//! This module implements a simple interpreter for Core IR expressions.
//! It supports both lazy (Default Profile) and strict (Numeric Profile)
//! evaluation modes.
//!
//! # Example
//!
//! ```ignore
//! use bhc_core::eval::{Evaluator, EvalMode};
//! use bhc_session::Profile;
//!
//! // Create evaluator from profile
//! let evaluator = Evaluator::with_profile(Profile::Numeric);
//! let result = evaluator.eval(&expr)?;
//!
//! // Or create with explicit mode
//! let evaluator = Evaluator::new(EvalMode::Lazy);
//! let result = evaluator.eval(&expr)?;
//! ```

mod env;
mod value;

pub use env::Env;
pub use value::{Closure, DataValue, HandleKind, HandleValue, OrdValue, PrimOp, Thunk, Value};

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use bhc_index::Idx;
use bhc_intern::Symbol;
use bhc_session::Profile;
use thiserror::Error;

use crate::{AltCon, Bind, Expr, Literal, Var, VarId};

/// Errors that can occur during evaluation.
#[derive(Debug, Error)]
pub enum EvalError {
    /// An unbound variable was referenced.
    #[error("unbound variable: {0}")]
    UnboundVariable(String),

    /// Type mismatch during evaluation.
    #[error("type error: expected {expected}, got {got}")]
    TypeError {
        /// The expected type.
        expected: String,
        /// The actual type found.
        got: String,
    },

    /// A pattern match failed (non-exhaustive).
    #[error("pattern match failure")]
    PatternMatchFailure,

    /// Division by zero.
    #[error("division by zero")]
    DivisionByZero,

    /// An explicit error was raised.
    #[error("error: {0}")]
    UserError(String),

    /// Recursion depth exceeded.
    #[error("stack overflow: maximum recursion depth exceeded")]
    StackOverflow,

    /// A thunk was forced during its own evaluation (black hole).
    #[error("infinite loop detected (black hole)")]
    BlackHole,
}

/// Evaluation mode controlling strictness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// Lazy evaluation (Default Profile).
    /// Arguments are wrapped in thunks and forced on demand.
    Lazy,

    /// Strict evaluation (Numeric Profile).
    /// Arguments are evaluated before function application.
    Strict,
}

impl From<Profile> for EvalMode {
    fn from(profile: Profile) -> Self {
        if profile.is_strict_by_default() {
            Self::Strict
        } else {
            Self::Lazy
        }
    }
}

/// The Core IR evaluator.
pub struct Evaluator {
    /// Evaluation mode (lazy or strict).
    mode: EvalMode,
    /// Primitive operations environment.
    primitives: HashMap<Symbol, Value>,
    /// Maximum recursion depth.
    max_depth: usize,
    /// Current recursion depth.
    depth: RefCell<usize>,
    /// Module-level environment for top-level bindings.
    /// This is always consulted for variable lookups, allowing recursive
    /// functions to find themselves without capturing a circular environment.
    module_env: RefCell<Option<Env>>,
    /// Stack of recursive environments for local let bindings.
    /// When evaluating the body of a recursive let, we push the environment
    /// containing all the recursive bindings here, so that closures can find
    /// their recursive references.
    rec_env_stack: RefCell<Vec<Env>>,
    /// Captured IO output from print/putStrLn/putStr operations.
    io_output: RefCell<String>,
    /// Name-based binding map for REPL persistence.
    /// When the REPL evaluates `let x = 5`, it stores `("x", Value::Int(5))`.
    /// Subsequent evaluations can reference `x` even though they have different VarIds.
    named_bindings: RefCell<HashMap<Symbol, Value>>,
}

impl Evaluator {
    /// Creates a new evaluator with the given mode.
    #[must_use]
    pub fn new(mode: EvalMode) -> Self {
        let mut primitives = HashMap::new();
        Self::register_primitives(&mut primitives);

        Self {
            mode,
            primitives,
            max_depth: 10000,
            depth: RefCell::new(0),
            module_env: RefCell::new(None),
            rec_env_stack: RefCell::new(Vec::new()),
            io_output: RefCell::new(String::new()),
            named_bindings: RefCell::new(HashMap::new()),
        }
    }

    /// Creates a new evaluator from a compilation profile.
    ///
    /// This is the preferred way to create an evaluator when working with
    /// the BHC compilation pipeline. The profile determines the evaluation
    /// semantics:
    ///
    /// - `Profile::Default` / `Profile::Server` → Lazy evaluation
    /// - `Profile::Numeric` / `Profile::Edge` → Strict evaluation
    ///
    /// # Example
    ///
    /// ```ignore
    /// use bhc_core::eval::Evaluator;
    /// use bhc_session::Profile;
    ///
    /// let eval = Evaluator::with_profile(Profile::Numeric);
    /// // Expressions are now evaluated strictly
    /// ```
    #[must_use]
    pub fn with_profile(profile: Profile) -> Self {
        Self::new(EvalMode::from(profile))
    }

    /// Returns the evaluation mode of this evaluator.
    #[must_use]
    pub fn mode(&self) -> EvalMode {
        self.mode
    }

    /// Returns true if this evaluator uses strict evaluation.
    #[must_use]
    pub fn is_strict(&self) -> bool {
        self.mode == EvalMode::Strict
    }

    /// Sets the maximum recursion depth.
    pub fn set_max_depth(&mut self, depth: usize) {
        self.max_depth = depth;
    }

    /// Sets the module-level environment.
    ///
    /// This environment is always consulted when looking up variables,
    /// allowing recursive functions to find themselves without needing
    /// to capture a circular environment in their closures.
    pub fn set_module_env(&self, env: Env) {
        *self.module_env.borrow_mut() = Some(env);
    }

    /// Clears the module-level environment.
    pub fn clear_module_env(&self) {
        *self.module_env.borrow_mut() = None;
    }

    /// Stores a named binding for REPL persistence.
    ///
    /// Named bindings are looked up by `Symbol` (name) rather than `VarId`,
    /// allowing values from previous evaluations (with different VarIds)
    /// to be referenced by name in subsequent expressions.
    pub fn set_named_binding(&self, name: Symbol, value: Value) {
        self.named_bindings.borrow_mut().insert(name, value);
    }

    /// Looks up a named binding by symbol.
    pub fn get_named_binding(&self, name: Symbol) -> Option<Value> {
        self.named_bindings.borrow().get(&name).cloned()
    }

    /// Returns the captured IO output from print/putStrLn/putStr operations.
    #[must_use]
    pub fn take_io_output(&self) -> String {
        self.io_output.borrow_mut().split_off(0)
    }

    /// Registers primitive operations in the environment.
    fn register_primitives(prims: &mut HashMap<Symbol, Value>) {
        let ops = [
            ("+", PrimOp::AddInt),
            ("-", PrimOp::SubInt),
            ("*", PrimOp::MulInt),
            ("div", PrimOp::DivInt),
            ("mod", PrimOp::ModInt),
            ("negate", PrimOp::NegInt),
            ("+.", PrimOp::AddDouble),
            ("-.", PrimOp::SubDouble),
            ("*.", PrimOp::MulDouble),
            ("/.", PrimOp::DivDouble),
            ("==", PrimOp::EqInt),
            ("/=", PrimOp::EqInt), // We'll negate the result
            ("<", PrimOp::LtInt),
            ("<=", PrimOp::LeInt),
            (">", PrimOp::GtInt),
            (">=", PrimOp::GeInt),
            ("&&", PrimOp::AndBool),
            ("||", PrimOp::OrBool),
            ("not", PrimOp::NotBool),
            ("seq", PrimOp::Seq),
            ("error", PrimOp::Error),
            // UArray operations
            ("fromList", PrimOp::UArrayFromList),
            ("toList", PrimOp::UArrayToList),
            ("uarrayMap", PrimOp::UArrayMap),
            ("map", PrimOp::UArrayMap), // Standard list map
            ("uarrayZipWith", PrimOp::UArrayZipWith),
            ("uarrayFold", PrimOp::UArrayFold),
            ("sum", PrimOp::UArraySum),
            ("length", PrimOp::UArrayLength),
            ("range", PrimOp::UArrayRange),
            // List operations
            ("++", PrimOp::Concat),
            ("concat", PrimOp::Concat),
            ("concatMap", PrimOp::ConcatMap),
            ("append", PrimOp::Append),
            // Monad operations (list monad)
            (">>=", PrimOp::MonadBind),
            (">>", PrimOp::MonadThen),
            ("return", PrimOp::ListReturn),
            // Additional list operations
            ("foldr", PrimOp::Foldr),
            ("foldl", PrimOp::Foldl),
            ("foldl'", PrimOp::FoldlStrict),
            ("filter", PrimOp::Filter),
            ("zip", PrimOp::Zip),
            ("zipWith", PrimOp::ZipWith),
            ("take", PrimOp::Take),
            ("drop", PrimOp::Drop),
            ("head", PrimOp::Head),
            ("tail", PrimOp::Tail),
            ("last", PrimOp::Last),
            ("init", PrimOp::Init),
            ("reverse", PrimOp::Reverse),
            ("null", PrimOp::Null),
            ("!!", PrimOp::Index),
            ("replicate", PrimOp::Replicate),
            ("enumFromTo", PrimOp::EnumFromTo),
            // Char operations
            ("ord", PrimOp::CharToInt),
            ("chr", PrimOp::IntToChar),
            // Additional list/prelude operations
            ("even", PrimOp::Even),
            ("odd", PrimOp::Odd),
            ("elem", PrimOp::Elem),
            ("notElem", PrimOp::NotElem),
            ("takeWhile", PrimOp::TakeWhile),
            ("dropWhile", PrimOp::DropWhile),
            ("span", PrimOp::Span),
            ("break", PrimOp::Break),
            ("splitAt", PrimOp::SplitAt),
            ("iterate", PrimOp::Iterate),
            ("repeat", PrimOp::Repeat),
            ("cycle", PrimOp::Cycle),
            ("lookup", PrimOp::Lookup),
            ("unzip", PrimOp::Unzip),
            ("product", PrimOp::Product),
            ("flip", PrimOp::Flip),
            ("min", PrimOp::Min),
            ("max", PrimOp::Max),
            ("fromIntegral", PrimOp::FromIntegral),
            ("toInteger", PrimOp::FromIntegral),
            ("maybe", PrimOp::MaybeElim),
            ("fromMaybe", PrimOp::FromMaybe),
            ("either", PrimOp::EitherElim),
            ("isJust", PrimOp::IsJust),
            ("isNothing", PrimOp::IsNothing),
            ("abs", PrimOp::Abs),
            ("signum", PrimOp::Signum),
            ("curry", PrimOp::Curry),
            ("uncurry", PrimOp::Uncurry),
            ("swap", PrimOp::Swap),
            ("any", PrimOp::Any),
            ("all", PrimOp::All),
            ("and", PrimOp::And),
            ("or", PrimOp::Or),
            ("lines", PrimOp::Lines),
            ("unlines", PrimOp::Unlines),
            ("words", PrimOp::Words),
            ("unwords", PrimOp::Unwords),
            ("show", PrimOp::Show),
            ("id", PrimOp::Id),
            ("const", PrimOp::Const),
            // IO operations
            ("putStrLn", PrimOp::PutStrLn),
            ("putStr", PrimOp::PutStr),
            ("print", PrimOp::Print),
            ("getLine", PrimOp::GetLine),
            // Enum operations
            ("succ", PrimOp::Succ),
            ("pred", PrimOp::Pred),
            ("toEnum", PrimOp::ToEnum),
            ("fromEnum", PrimOp::FromEnum),
            // Integral operations
            ("gcd", PrimOp::Gcd),
            ("lcm", PrimOp::Lcm),
            ("quot", PrimOp::Quot),
            ("rem", PrimOp::Rem),
            ("quotRem", PrimOp::QuotRem),
            ("divMod", PrimOp::DivMod),
            ("subtract", PrimOp::Subtract),
            // Scan operations
            ("scanl", PrimOp::Scanl),
            ("scanl'", PrimOp::Scanl),
            ("scanr", PrimOp::Scanr),
            ("scanl1", PrimOp::Scanl1),
            ("scanr1", PrimOp::Scanr1),
            // More list operations
            ("maximum", PrimOp::Maximum),
            ("minimum", PrimOp::Minimum),
            ("zip3", PrimOp::Zip3),
            ("zipWith3", PrimOp::ZipWith3),
            ("unzip3", PrimOp::Unzip3),
            // Show helpers
            ("showString", PrimOp::ShowString),
            ("showChar", PrimOp::ShowChar),
            ("showParen", PrimOp::ShowParen),
            // IO operations (additional)
            ("getChar", PrimOp::GetChar),
            ("getContents", PrimOp::GetContents),
            ("readFile", PrimOp::ReadFile),
            ("writeFile", PrimOp::WriteFile),
            ("appendFile", PrimOp::AppendFile),
            ("interact", PrimOp::Interact),
            // Misc Prelude
            ("otherwise", PrimOp::Otherwise),
            ("until", PrimOp::Until),
            ("asTypeOf", PrimOp::AsTypeOf),
            ("realToFrac", PrimOp::RealToFrac),
            // Data.List
            ("sort", PrimOp::Sort),
            ("sortBy", PrimOp::SortBy),
            ("sortOn", PrimOp::SortOn),
            ("nub", PrimOp::Nub),
            ("nubBy", PrimOp::NubBy),
            ("group", PrimOp::Group),
            ("groupBy", PrimOp::GroupBy),
            ("intersperse", PrimOp::Intersperse),
            ("intercalate", PrimOp::Intercalate),
            ("transpose", PrimOp::Transpose),
            ("subsequences", PrimOp::Subsequences),
            ("permutations", PrimOp::Permutations),
            ("partition", PrimOp::Partition),
            ("find", PrimOp::Find),
            ("stripPrefix", PrimOp::StripPrefix),
            ("isPrefixOf", PrimOp::IsPrefixOf),
            ("isSuffixOf", PrimOp::IsSuffixOf),
            ("isInfixOf", PrimOp::IsInfixOf),
            ("delete", PrimOp::Delete),
            ("deleteBy", PrimOp::DeleteBy),
            ("union", PrimOp::Union),
            ("unionBy", PrimOp::UnionBy),
            ("intersect", PrimOp::Intersect),
            ("intersectBy", PrimOp::IntersectBy),
            ("\\\\", PrimOp::ListDiff),
            ("tails", PrimOp::Tails),
            ("inits", PrimOp::Inits),
            ("mapAccumL", PrimOp::MapAccumL),
            ("mapAccumR", PrimOp::MapAccumR),
            ("unfoldr", PrimOp::Unfoldr),
            ("genericLength", PrimOp::GenericLength),
            ("genericTake", PrimOp::GenericTake),
            ("genericDrop", PrimOp::GenericDrop),
            // Data.Char
            ("isAlpha", PrimOp::IsAlpha),
            ("isAlphaNum", PrimOp::IsAlphaNum),
            ("isAscii", PrimOp::IsAscii),
            ("isControl", PrimOp::IsControl),
            ("isDigit", PrimOp::IsDigit),
            ("isHexDigit", PrimOp::IsHexDigit),
            ("isLetter", PrimOp::IsLetter),
            ("isLower", PrimOp::IsLower),
            ("isNumber", PrimOp::IsNumber),
            ("isPrint", PrimOp::IsPrint),
            ("isPunctuation", PrimOp::IsPunctuation),
            ("isSpace", PrimOp::IsSpace),
            ("isSymbol", PrimOp::IsSymbol),
            ("isUpper", PrimOp::IsUpper),
            ("toLower", PrimOp::ToLower),
            ("toUpper", PrimOp::ToUpper),
            ("toTitle", PrimOp::ToTitle),
            ("digitToInt", PrimOp::DigitToInt),
            ("intToDigit", PrimOp::IntToDigit),
            ("isLatin1", PrimOp::IsLatin1),
            ("isAsciiLower", PrimOp::IsAsciiLower),
            ("isAsciiUpper", PrimOp::IsAsciiUpper),
            // Data.Function
            ("on", PrimOp::On),
            ("fix", PrimOp::Fix),
            ("&", PrimOp::Amp),
            // Data.Maybe additional
            ("listToMaybe", PrimOp::ListToMaybe),
            ("maybeToList", PrimOp::MaybeToList),
            ("catMaybes", PrimOp::CatMaybes),
            ("mapMaybe", PrimOp::MapMaybe),
            // Data.Either additional
            ("isLeft", PrimOp::IsLeft),
            ("isRight", PrimOp::IsRight),
            ("lefts", PrimOp::Lefts),
            ("rights", PrimOp::Rights),
            ("partitionEithers", PrimOp::PartitionEithers),
            // Math functions
            ("sqrt", PrimOp::Sqrt),
            ("exp", PrimOp::Exp),
            ("log", PrimOp::Log),
            ("sin", PrimOp::Sin),
            ("cos", PrimOp::Cos),
            ("tan", PrimOp::Tan),
            ("^", PrimOp::Power),
            ("truncate", PrimOp::Truncate),
            ("round", PrimOp::Round),
            ("ceiling", PrimOp::Ceiling),
            ("floor", PrimOp::Floor),
            // Tuple
            ("fst", PrimOp::Fst),
            ("snd", PrimOp::Snd),
        ];

        for (name, op) in ops {
            prims.insert(Symbol::intern(name), Value::PrimOp(op));
        }

        // Register container PrimOps with qualified names
        let container_ops: &[(&str, PrimOp)] = &[
            // Data.Map
            ("Data.Map.empty", PrimOp::MapEmpty),
            ("Data.Map.singleton", PrimOp::MapSingleton),
            ("Data.Map.null", PrimOp::MapNull),
            ("Data.Map.size", PrimOp::MapSize),
            ("Data.Map.member", PrimOp::MapMember),
            ("Data.Map.notMember", PrimOp::MapNotMember),
            ("Data.Map.lookup", PrimOp::MapLookup),
            ("Data.Map.findWithDefault", PrimOp::MapFindWithDefault),
            ("Data.Map.!", PrimOp::MapIndex),
            ("Data.Map.insert", PrimOp::MapInsert),
            ("Data.Map.insertWith", PrimOp::MapInsertWith),
            ("Data.Map.delete", PrimOp::MapDelete),
            ("Data.Map.adjust", PrimOp::MapAdjust),
            ("Data.Map.update", PrimOp::MapUpdate),
            ("Data.Map.alter", PrimOp::MapAlter),
            ("Data.Map.union", PrimOp::MapUnion),
            ("Data.Map.unionWith", PrimOp::MapUnionWith),
            ("Data.Map.unionWithKey", PrimOp::MapUnionWithKey),
            ("Data.Map.unions", PrimOp::MapUnions),
            ("Data.Map.intersection", PrimOp::MapIntersection),
            ("Data.Map.intersectionWith", PrimOp::MapIntersectionWith),
            ("Data.Map.difference", PrimOp::MapDifference),
            ("Data.Map.differenceWith", PrimOp::MapDifferenceWith),
            ("Data.Map.map", PrimOp::MapMap),
            ("Data.Map.mapWithKey", PrimOp::MapMapWithKey),
            ("Data.Map.mapKeys", PrimOp::MapMapKeys),
            ("Data.Map.filter", PrimOp::MapFilter),
            ("Data.Map.filterWithKey", PrimOp::MapFilterWithKey),
            ("Data.Map.foldr", PrimOp::MapFoldr),
            ("Data.Map.foldl", PrimOp::MapFoldl),
            ("Data.Map.foldrWithKey", PrimOp::MapFoldrWithKey),
            ("Data.Map.foldlWithKey", PrimOp::MapFoldlWithKey),
            ("Data.Map.keys", PrimOp::MapKeys),
            ("Data.Map.elems", PrimOp::MapElems),
            ("Data.Map.assocs", PrimOp::MapAssocs),
            ("Data.Map.toList", PrimOp::MapToList),
            ("Data.Map.fromList", PrimOp::MapFromList),
            ("Data.Map.fromListWith", PrimOp::MapFromListWith),
            ("Data.Map.toAscList", PrimOp::MapToAscList),
            ("Data.Map.toDescList", PrimOp::MapToDescList),
            ("Data.Map.isSubmapOf", PrimOp::MapIsSubmapOf),
            // Data.Map.Strict aliases
            ("Data.Map.Strict.empty", PrimOp::MapEmpty),
            ("Data.Map.Strict.singleton", PrimOp::MapSingleton),
            ("Data.Map.Strict.null", PrimOp::MapNull),
            ("Data.Map.Strict.size", PrimOp::MapSize),
            ("Data.Map.Strict.member", PrimOp::MapMember),
            ("Data.Map.Strict.insert", PrimOp::MapInsert),
            ("Data.Map.Strict.delete", PrimOp::MapDelete),
            ("Data.Map.Strict.lookup", PrimOp::MapLookup),
            ("Data.Map.Strict.union", PrimOp::MapUnion),
            ("Data.Map.Strict.map", PrimOp::MapMap),
            ("Data.Map.Strict.filter", PrimOp::MapFilter),
            ("Data.Map.Strict.fromList", PrimOp::MapFromList),
            // Data.Set
            ("Data.Set.empty", PrimOp::SetEmpty),
            ("Data.Set.singleton", PrimOp::SetSingleton),
            ("Data.Set.null", PrimOp::SetNull),
            ("Data.Set.size", PrimOp::SetSize),
            ("Data.Set.member", PrimOp::SetMember),
            ("Data.Set.notMember", PrimOp::SetNotMember),
            ("Data.Set.insert", PrimOp::SetInsert),
            ("Data.Set.delete", PrimOp::SetDelete),
            ("Data.Set.union", PrimOp::SetUnion),
            ("Data.Set.unions", PrimOp::SetUnions),
            ("Data.Set.intersection", PrimOp::SetIntersection),
            ("Data.Set.difference", PrimOp::SetDifference),
            ("Data.Set.isSubsetOf", PrimOp::SetIsSubsetOf),
            ("Data.Set.isProperSubsetOf", PrimOp::SetIsProperSubsetOf),
            ("Data.Set.map", PrimOp::SetMap),
            ("Data.Set.filter", PrimOp::SetFilter),
            ("Data.Set.partition", PrimOp::SetPartition),
            ("Data.Set.foldr", PrimOp::SetFoldr),
            ("Data.Set.foldl", PrimOp::SetFoldl),
            ("Data.Set.toList", PrimOp::SetToList),
            ("Data.Set.fromList", PrimOp::SetFromList),
            ("Data.Set.toAscList", PrimOp::SetToAscList),
            ("Data.Set.toDescList", PrimOp::SetToDescList),
            ("Data.Set.findMin", PrimOp::SetFindMin),
            ("Data.Set.findMax", PrimOp::SetFindMax),
            ("Data.Set.deleteMin", PrimOp::SetDeleteMin),
            ("Data.Set.deleteMax", PrimOp::SetDeleteMax),
            ("Data.Set.elems", PrimOp::SetElems),
            ("Data.Set.lookupMin", PrimOp::SetLookupMin),
            ("Data.Set.lookupMax", PrimOp::SetLookupMax),
            // Data.IntMap
            ("Data.IntMap.empty", PrimOp::IntMapEmpty),
            ("Data.IntMap.singleton", PrimOp::IntMapSingleton),
            ("Data.IntMap.null", PrimOp::IntMapNull),
            ("Data.IntMap.size", PrimOp::IntMapSize),
            ("Data.IntMap.member", PrimOp::IntMapMember),
            ("Data.IntMap.lookup", PrimOp::IntMapLookup),
            ("Data.IntMap.findWithDefault", PrimOp::IntMapFindWithDefault),
            ("Data.IntMap.insert", PrimOp::IntMapInsert),
            ("Data.IntMap.insertWith", PrimOp::IntMapInsertWith),
            ("Data.IntMap.delete", PrimOp::IntMapDelete),
            ("Data.IntMap.adjust", PrimOp::IntMapAdjust),
            ("Data.IntMap.union", PrimOp::IntMapUnion),
            ("Data.IntMap.unionWith", PrimOp::IntMapUnionWith),
            ("Data.IntMap.intersection", PrimOp::IntMapIntersection),
            ("Data.IntMap.difference", PrimOp::IntMapDifference),
            ("Data.IntMap.map", PrimOp::IntMapMap),
            ("Data.IntMap.mapWithKey", PrimOp::IntMapMapWithKey),
            ("Data.IntMap.filter", PrimOp::IntMapFilter),
            ("Data.IntMap.foldr", PrimOp::IntMapFoldr),
            ("Data.IntMap.foldlWithKey", PrimOp::IntMapFoldlWithKey),
            ("Data.IntMap.keys", PrimOp::IntMapKeys),
            ("Data.IntMap.elems", PrimOp::IntMapElems),
            ("Data.IntMap.toList", PrimOp::IntMapToList),
            ("Data.IntMap.fromList", PrimOp::IntMapFromList),
            ("Data.IntMap.toAscList", PrimOp::IntMapToAscList),
            // Data.IntSet
            ("Data.IntSet.empty", PrimOp::IntSetEmpty),
            ("Data.IntSet.singleton", PrimOp::IntSetSingleton),
            ("Data.IntSet.null", PrimOp::IntSetNull),
            ("Data.IntSet.size", PrimOp::IntSetSize),
            ("Data.IntSet.member", PrimOp::IntSetMember),
            ("Data.IntSet.insert", PrimOp::IntSetInsert),
            ("Data.IntSet.delete", PrimOp::IntSetDelete),
            ("Data.IntSet.union", PrimOp::IntSetUnion),
            ("Data.IntSet.intersection", PrimOp::IntSetIntersection),
            ("Data.IntSet.difference", PrimOp::IntSetDifference),
            ("Data.IntSet.isSubsetOf", PrimOp::IntSetIsSubsetOf),
            ("Data.IntSet.filter", PrimOp::IntSetFilter),
            ("Data.IntSet.foldr", PrimOp::IntSetFoldr),
            ("Data.IntSet.toList", PrimOp::IntSetToList),
            ("Data.IntSet.fromList", PrimOp::IntSetFromList),
        ];

        for (name, op) in container_ops {
            prims.insert(Symbol::intern(name), Value::PrimOp(*op));
        }

        // Register IO PrimOps (System.IO, Data.IORef, System.Exit, System.Environment, System.Directory)
        let io_ops: &[(&str, PrimOp)] = &[
            // System.IO handles
            ("stdin", PrimOp::Stdin),
            ("stdout", PrimOp::Stdout),
            ("stderr", PrimOp::Stderr),
            ("openFile", PrimOp::OpenFile),
            ("System.IO.openFile", PrimOp::OpenFile),
            ("hClose", PrimOp::HClose),
            ("System.IO.hClose", PrimOp::HClose),
            ("hGetChar", PrimOp::HGetChar),
            ("System.IO.hGetChar", PrimOp::HGetChar),
            ("hGetLine", PrimOp::HGetLine),
            ("System.IO.hGetLine", PrimOp::HGetLine),
            ("hGetContents", PrimOp::HGetContents),
            ("System.IO.hGetContents", PrimOp::HGetContents),
            ("hPutChar", PrimOp::HPutChar),
            ("System.IO.hPutChar", PrimOp::HPutChar),
            ("hPutStr", PrimOp::HPutStr),
            ("System.IO.hPutStr", PrimOp::HPutStr),
            ("hPutStrLn", PrimOp::HPutStrLn),
            ("System.IO.hPutStrLn", PrimOp::HPutStrLn),
            ("hPrint", PrimOp::HPrint),
            ("System.IO.hPrint", PrimOp::HPrint),
            ("hFlush", PrimOp::HFlush),
            ("System.IO.hFlush", PrimOp::HFlush),
            ("hIsEOF", PrimOp::HIsEOF),
            ("System.IO.hIsEOF", PrimOp::HIsEOF),
            ("hSetBuffering", PrimOp::HSetBuffering),
            ("System.IO.hSetBuffering", PrimOp::HSetBuffering),
            ("hGetBuffering", PrimOp::HGetBuffering),
            ("System.IO.hGetBuffering", PrimOp::HGetBuffering),
            ("hSeek", PrimOp::HSeek),
            ("System.IO.hSeek", PrimOp::HSeek),
            ("hTell", PrimOp::HTell),
            ("System.IO.hTell", PrimOp::HTell),
            ("hFileSize", PrimOp::HFileSize),
            ("System.IO.hFileSize", PrimOp::HFileSize),
            ("withFile", PrimOp::WithFile),
            ("System.IO.withFile", PrimOp::WithFile),
            // Data.IORef
            ("newIORef", PrimOp::NewIORef),
            ("Data.IORef.newIORef", PrimOp::NewIORef),
            ("readIORef", PrimOp::ReadIORef),
            ("Data.IORef.readIORef", PrimOp::ReadIORef),
            ("writeIORef", PrimOp::WriteIORef),
            ("Data.IORef.writeIORef", PrimOp::WriteIORef),
            ("modifyIORef", PrimOp::ModifyIORef),
            ("Data.IORef.modifyIORef", PrimOp::ModifyIORef),
            ("modifyIORef'", PrimOp::ModifyIORefStrict),
            ("Data.IORef.modifyIORef'", PrimOp::ModifyIORefStrict),
            ("atomicModifyIORef", PrimOp::AtomicModifyIORef),
            ("Data.IORef.atomicModifyIORef", PrimOp::AtomicModifyIORef),
            ("atomicModifyIORef'", PrimOp::AtomicModifyIORefStrict),
            ("Data.IORef.atomicModifyIORef'", PrimOp::AtomicModifyIORefStrict),
            // System.Exit
            ("exitSuccess", PrimOp::ExitSuccess),
            ("System.Exit.exitSuccess", PrimOp::ExitSuccess),
            ("exitFailure", PrimOp::ExitFailure),
            ("System.Exit.exitFailure", PrimOp::ExitFailure),
            ("exitWith", PrimOp::ExitWith),
            ("System.Exit.exitWith", PrimOp::ExitWith),
            // System.Environment
            ("getArgs", PrimOp::GetArgs),
            ("System.Environment.getArgs", PrimOp::GetArgs),
            ("getProgName", PrimOp::GetProgName),
            ("System.Environment.getProgName", PrimOp::GetProgName),
            ("getEnv", PrimOp::GetEnv),
            ("System.Environment.getEnv", PrimOp::GetEnv),
            ("lookupEnv", PrimOp::LookupEnv),
            ("System.Environment.lookupEnv", PrimOp::LookupEnv),
            ("setEnv", PrimOp::SetEnv),
            ("System.Environment.setEnv", PrimOp::SetEnv),
            // System.Directory
            ("doesFileExist", PrimOp::DoesFileExist),
            ("System.Directory.doesFileExist", PrimOp::DoesFileExist),
            ("doesDirectoryExist", PrimOp::DoesDirectoryExist),
            ("System.Directory.doesDirectoryExist", PrimOp::DoesDirectoryExist),
            ("createDirectory", PrimOp::CreateDirectory),
            ("System.Directory.createDirectory", PrimOp::CreateDirectory),
            ("createDirectoryIfMissing", PrimOp::CreateDirectoryIfMissing),
            ("System.Directory.createDirectoryIfMissing", PrimOp::CreateDirectoryIfMissing),
            ("removeFile", PrimOp::RemoveFile),
            ("System.Directory.removeFile", PrimOp::RemoveFile),
            ("removeDirectory", PrimOp::RemoveDirectory),
            ("System.Directory.removeDirectory", PrimOp::RemoveDirectory),
            ("getCurrentDirectory", PrimOp::GetCurrentDirectory),
            ("System.Directory.getCurrentDirectory", PrimOp::GetCurrentDirectory),
            ("setCurrentDirectory", PrimOp::SetCurrentDirectory),
            ("System.Directory.setCurrentDirectory", PrimOp::SetCurrentDirectory),
        ];
        for (name, op) in io_ops {
            prims.insert(Symbol::intern(name), Value::PrimOp(*op));
        }

        // Control.* operations
        let control_ops: &[(&str, PrimOp)] = &[
            // Control.Monad
            ("when", PrimOp::MonadWhen),
            ("Control.Monad.when", PrimOp::MonadWhen),
            ("unless", PrimOp::MonadUnless),
            ("Control.Monad.unless", PrimOp::MonadUnless),
            ("guard", PrimOp::MonadGuard),
            ("Control.Monad.guard", PrimOp::MonadGuard),
            ("void", PrimOp::MonadVoid),
            ("Control.Monad.void", PrimOp::MonadVoid),
            ("Data.Functor.void", PrimOp::MonadVoid),
            ("join", PrimOp::MonadJoin),
            ("Control.Monad.join", PrimOp::MonadJoin),
            ("ap", PrimOp::MonadAp),
            ("Control.Monad.ap", PrimOp::MonadAp),
            ("liftM", PrimOp::LiftM),
            ("Control.Monad.liftM", PrimOp::LiftM),
            ("liftM2", PrimOp::LiftM2),
            ("Control.Monad.liftM2", PrimOp::LiftM2),
            ("liftM3", PrimOp::LiftM3),
            ("Control.Monad.liftM3", PrimOp::LiftM3),
            ("liftM4", PrimOp::LiftM4),
            ("Control.Monad.liftM4", PrimOp::LiftM4),
            ("liftM5", PrimOp::LiftM5),
            ("Control.Monad.liftM5", PrimOp::LiftM5),
            ("filterM", PrimOp::FilterM),
            ("Control.Monad.filterM", PrimOp::FilterM),
            ("mapAndUnzipM", PrimOp::MapAndUnzipM),
            ("Control.Monad.mapAndUnzipM", PrimOp::MapAndUnzipM),
            ("zipWithM", PrimOp::ZipWithM),
            ("Control.Monad.zipWithM", PrimOp::ZipWithM),
            ("zipWithM_", PrimOp::ZipWithM_),
            ("Control.Monad.zipWithM_", PrimOp::ZipWithM_),
            ("foldM", PrimOp::FoldM),
            ("Control.Monad.foldM", PrimOp::FoldM),
            ("foldM_", PrimOp::FoldM_),
            ("Control.Monad.foldM_", PrimOp::FoldM_),
            ("replicateM", PrimOp::ReplicateM),
            ("Control.Monad.replicateM", PrimOp::ReplicateM),
            ("replicateM_", PrimOp::ReplicateM_),
            ("Control.Monad.replicateM_", PrimOp::ReplicateM_),
            ("forever", PrimOp::Forever),
            ("Control.Monad.forever", PrimOp::Forever),
            ("mzero", PrimOp::Mzero),
            ("Control.Monad.mzero", PrimOp::Mzero),
            ("mplus", PrimOp::Mplus),
            ("Control.Monad.mplus", PrimOp::Mplus),
            ("msum", PrimOp::Msum),
            ("Control.Monad.msum", PrimOp::Msum),
            ("mfilter", PrimOp::Mfilter),
            ("Control.Monad.mfilter", PrimOp::Mfilter),
            (">=>", PrimOp::KleisliCompose),
            ("Control.Monad.>=>", PrimOp::KleisliCompose),
            ("<=<", PrimOp::KleisliComposeFlip),
            ("Control.Monad.<=<", PrimOp::KleisliComposeFlip),
            // Control.Applicative
            ("liftA", PrimOp::LiftA),
            ("Control.Applicative.liftA", PrimOp::LiftA),
            ("liftA2", PrimOp::LiftA2),
            ("Control.Applicative.liftA2", PrimOp::LiftA2),
            ("liftA3", PrimOp::LiftA3),
            ("Control.Applicative.liftA3", PrimOp::LiftA3),
            ("optional", PrimOp::Optional),
            ("Control.Applicative.optional", PrimOp::Optional),
            // Control.Exception
            ("catch", PrimOp::ExnCatch),
            ("Control.Exception.catch", PrimOp::ExnCatch),
            ("try", PrimOp::ExnTry),
            ("Control.Exception.try", PrimOp::ExnTry),
            ("throw", PrimOp::ExnThrow),
            ("Control.Exception.throw", PrimOp::ExnThrow),
            ("throwIO", PrimOp::ExnThrowIO),
            ("Control.Exception.throwIO", PrimOp::ExnThrowIO),
            ("bracket", PrimOp::ExnBracket),
            ("Control.Exception.bracket", PrimOp::ExnBracket),
            ("bracket_", PrimOp::ExnBracket_),
            ("Control.Exception.bracket_", PrimOp::ExnBracket_),
            ("bracketOnError", PrimOp::ExnBracketOnError),
            ("Control.Exception.bracketOnError", PrimOp::ExnBracketOnError),
            ("finally", PrimOp::ExnFinally),
            ("Control.Exception.finally", PrimOp::ExnFinally),
            ("onException", PrimOp::ExnOnException),
            ("Control.Exception.onException", PrimOp::ExnOnException),
            ("handle", PrimOp::ExnHandle),
            ("Control.Exception.handle", PrimOp::ExnHandle),
            ("handleJust", PrimOp::ExnHandleJust),
            ("Control.Exception.handleJust", PrimOp::ExnHandleJust),
            ("evaluate", PrimOp::ExnEvaluate),
            ("Control.Exception.evaluate", PrimOp::ExnEvaluate),
            ("mask", PrimOp::ExnMask),
            ("Control.Exception.mask", PrimOp::ExnMask),
            ("mask_", PrimOp::ExnMask_),
            ("Control.Exception.mask_", PrimOp::ExnMask_),
            ("uninterruptibleMask", PrimOp::ExnUninterruptibleMask),
            ("Control.Exception.uninterruptibleMask", PrimOp::ExnUninterruptibleMask),
            ("uninterruptibleMask_", PrimOp::ExnUninterruptibleMask_),
            ("Control.Exception.uninterruptibleMask_", PrimOp::ExnUninterruptibleMask_),
            // Control.Concurrent
            ("forkIO", PrimOp::ForkIO),
            ("Control.Concurrent.forkIO", PrimOp::ForkIO),
            ("threadDelay", PrimOp::ThreadDelay),
            ("Control.Concurrent.threadDelay", PrimOp::ThreadDelay),
            ("myThreadId", PrimOp::MyThreadId),
            ("Control.Concurrent.myThreadId", PrimOp::MyThreadId),
            ("newMVar", PrimOp::NewMVar),
            ("Control.Concurrent.MVar.newMVar", PrimOp::NewMVar),
            ("newEmptyMVar", PrimOp::NewEmptyMVar),
            ("Control.Concurrent.MVar.newEmptyMVar", PrimOp::NewEmptyMVar),
            ("takeMVar", PrimOp::TakeMVar),
            ("Control.Concurrent.MVar.takeMVar", PrimOp::TakeMVar),
            ("putMVar", PrimOp::PutMVar),
            ("Control.Concurrent.MVar.putMVar", PrimOp::PutMVar),
            ("readMVar", PrimOp::ReadMVar),
            ("Control.Concurrent.MVar.readMVar", PrimOp::ReadMVar),
            ("throwTo", PrimOp::ThrowTo),
            ("Control.Concurrent.throwTo", PrimOp::ThrowTo),
            ("Control.Exception.throwTo", PrimOp::ThrowTo),
            ("killThread", PrimOp::KillThread),
            ("Control.Concurrent.killThread", PrimOp::KillThread),
        ];
        for (name, op) in control_ops {
            prims.insert(Symbol::intern(name), Value::PrimOp(*op));
        }

        // Data.* operations
        let data_ops: &[(&str, PrimOp)] = &[
            // Data.Ord
            ("comparing", PrimOp::Comparing),
            ("Data.Ord.comparing", PrimOp::Comparing),
            ("clamp", PrimOp::Clamp),
            ("Data.Ord.clamp", PrimOp::Clamp),
            // Data.Foldable
            ("fold", PrimOp::Fold),
            ("Data.Foldable.fold", PrimOp::Fold),
            ("foldMap", PrimOp::FoldMap),
            ("Data.Foldable.foldMap", PrimOp::FoldMap),
            ("foldr'", PrimOp::FoldrStrict),
            ("Data.Foldable.foldr'", PrimOp::FoldrStrict),
            ("foldl1", PrimOp::Foldl1),
            ("Data.Foldable.foldl1", PrimOp::Foldl1),
            ("foldr1", PrimOp::Foldr1),
            ("Data.Foldable.foldr1", PrimOp::Foldr1),
            ("maximumBy", PrimOp::MaximumBy),
            ("Data.Foldable.maximumBy", PrimOp::MaximumBy),
            ("Data.List.maximumBy", PrimOp::MaximumBy),
            ("minimumBy", PrimOp::MinimumBy),
            ("Data.Foldable.minimumBy", PrimOp::MinimumBy),
            ("Data.List.minimumBy", PrimOp::MinimumBy),
            ("Data.Foldable.notElem", PrimOp::NotElem),
            ("asum", PrimOp::Asum),
            ("Data.Foldable.asum", PrimOp::Asum),
            ("traverse_", PrimOp::Traverse_),
            ("Data.Foldable.traverse_", PrimOp::Traverse_),
            ("for_", PrimOp::For_),
            ("Data.Foldable.for_", PrimOp::For_),
            ("sequenceA_", PrimOp::SequenceA_),
            ("Data.Foldable.sequenceA_", PrimOp::SequenceA_),
            // Data.Traversable
            ("traverse", PrimOp::Traverse),
            ("Data.Traversable.traverse", PrimOp::Traverse),
            ("sequenceA", PrimOp::SequenceA),
            ("Data.Traversable.sequenceA", PrimOp::SequenceA),
            ("Data.Traversable.mapAccumL", PrimOp::MapAccumL),
            ("Data.List.mapAccumL", PrimOp::MapAccumL),
            ("Data.Traversable.mapAccumR", PrimOp::MapAccumR),
            ("Data.List.mapAccumR", PrimOp::MapAccumR),
            // Data.Monoid
            ("mempty", PrimOp::Mempty),
            ("Data.Monoid.mempty", PrimOp::Mempty),
            ("mappend", PrimOp::Mappend),
            ("Data.Monoid.mappend", PrimOp::Mappend),
            ("mconcat", PrimOp::Mconcat),
            ("Data.Monoid.mconcat", PrimOp::Mconcat),
            // Data.String
            ("fromString", PrimOp::FromString),
            ("Data.String.fromString", PrimOp::FromString),
            // Data.Bits
            (".&.", PrimOp::BitAnd),
            ("Data.Bits..&.", PrimOp::BitAnd),
            (".|.", PrimOp::BitOr),
            ("Data.Bits..|.", PrimOp::BitOr),
            ("xor", PrimOp::BitXor),
            ("Data.Bits.xor", PrimOp::BitXor),
            ("complement", PrimOp::BitComplement),
            ("Data.Bits.complement", PrimOp::BitComplement),
            ("shift", PrimOp::BitShift),
            ("Data.Bits.shift", PrimOp::BitShift),
            ("shiftL", PrimOp::BitShiftL),
            ("Data.Bits.shiftL", PrimOp::BitShiftL),
            ("shiftR", PrimOp::BitShiftR),
            ("Data.Bits.shiftR", PrimOp::BitShiftR),
            ("rotate", PrimOp::BitRotate),
            ("Data.Bits.rotate", PrimOp::BitRotate),
            ("rotateL", PrimOp::BitRotateL),
            ("Data.Bits.rotateL", PrimOp::BitRotateL),
            ("rotateR", PrimOp::BitRotateR),
            ("Data.Bits.rotateR", PrimOp::BitRotateR),
            ("bit", PrimOp::BitBit),
            ("Data.Bits.bit", PrimOp::BitBit),
            ("setBit", PrimOp::BitSetBit),
            ("Data.Bits.setBit", PrimOp::BitSetBit),
            ("clearBit", PrimOp::BitClearBit),
            ("Data.Bits.clearBit", PrimOp::BitClearBit),
            ("complementBit", PrimOp::BitComplementBit),
            ("Data.Bits.complementBit", PrimOp::BitComplementBit),
            ("testBit", PrimOp::BitTestBit),
            ("Data.Bits.testBit", PrimOp::BitTestBit),
            ("popCount", PrimOp::BitPopCount),
            ("Data.Bits.popCount", PrimOp::BitPopCount),
            ("zeroBits", PrimOp::BitZeroBits),
            ("Data.Bits.zeroBits", PrimOp::BitZeroBits),
            ("countLeadingZeros", PrimOp::BitCountLeadingZeros),
            ("Data.Bits.countLeadingZeros", PrimOp::BitCountLeadingZeros),
            ("countTrailingZeros", PrimOp::BitCountTrailingZeros),
            ("Data.Bits.countTrailingZeros", PrimOp::BitCountTrailingZeros),
            // Data.Proxy
            ("asProxyTypeOf", PrimOp::AsProxyTypeOf),
            ("Data.Proxy.asProxyTypeOf", PrimOp::AsProxyTypeOf),
            // Data.Void
            ("absurd", PrimOp::Absurd),
            ("Data.Void.absurd", PrimOp::Absurd),
            ("vacuous", PrimOp::Vacuous),
            ("Data.Void.vacuous", PrimOp::Vacuous),
        ];
        for (name, op) in data_ops {
            prims.insert(Symbol::intern(name), Value::PrimOp(*op));
        }

        // Register list constructors
        prims.insert(Symbol::intern("[]"), Value::nil());
        prims.insert(
            Symbol::intern(":"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern(":"),
                    ty_con: bhc_types::TyCon::new(
                        Symbol::intern("[]"),
                        bhc_types::Kind::star_to_star(),
                    ),
                    tag: 1,
                    arity: 2,
                },
                args: vec![],
            }),
        );

        // Register tuple constructors
        // Unit tuple ()
        prims.insert(
            Symbol::intern("()"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern("()"),
                    ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star),
                    tag: 0,
                    arity: 0,
                },
                args: vec![],
            }),
        );

        // Pair (,)
        prims.insert(
            Symbol::intern("(,)"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern("(,)"),
                    ty_con: bhc_types::TyCon::new(Symbol::intern("(,)"), bhc_types::Kind::Star),
                    tag: 0,
                    arity: 2,
                },
                args: vec![],
            }),
        );

        // Triple (,,)
        prims.insert(
            Symbol::intern("(,,)"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern("(,,)"),
                    ty_con: bhc_types::TyCon::new(Symbol::intern("(,,)"), bhc_types::Kind::Star),
                    tag: 0,
                    arity: 3,
                },
                args: vec![],
            }),
        );

        // Quadruple (,,,)
        prims.insert(
            Symbol::intern("(,,,)"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern("(,,,)"),
                    ty_con: bhc_types::TyCon::new(Symbol::intern("(,,,)"), bhc_types::Kind::Star),
                    tag: 0,
                    arity: 4,
                },
                args: vec![],
            }),
        );

        // Quintuple (,,,,)
        prims.insert(
            Symbol::intern("(,,,,)"),
            Value::Data(DataValue {
                con: crate::DataCon {
                    name: Symbol::intern("(,,,,)"),
                    ty_con: bhc_types::TyCon::new(Symbol::intern("(,,,,)"), bhc_types::Kind::Star),
                    tag: 0,
                    arity: 5,
                },
                args: vec![],
            }),
        );

        // Boolean constructors and otherwise
        prims.insert(Symbol::intern("True"), Value::bool(true));
        prims.insert(Symbol::intern("False"), Value::bool(false));
        prims.insert(Symbol::intern("otherwise"), Value::bool(true));
    }

    /// Evaluates an expression to a value.
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation fails (unbound variable, type error, etc.)
    pub fn eval(&self, expr: &Expr, env: &Env) -> Result<Value, EvalError> {
        // Check recursion depth
        {
            let mut depth = self.depth.borrow_mut();
            if *depth >= self.max_depth {
                return Err(EvalError::StackOverflow);
            }
            *depth += 1;
        }

        let result = self.eval_inner(expr, env);

        // Decrement depth
        *self.depth.borrow_mut() -= 1;

        result
    }

    fn eval_inner(&self, expr: &Expr, env: &Env) -> Result<Value, EvalError> {
        match expr {
            Expr::Var(var, _) => self.eval_var(var, env),

            Expr::Lit(lit, _, _) => Ok(self.eval_lit(lit)),

            Expr::App(fun, arg, _) => self.eval_app(fun, arg, env),

            Expr::TyApp(fun, _, _) => {
                // Type applications are erased at runtime
                self.eval(fun, env)
            }

            Expr::Lam(var, body, _) => Ok(Value::Closure(Closure {
                var: var.clone(),
                body: body.clone(),
                env: env.clone(),
            })),

            Expr::TyLam(_, body, _) => {
                // Type lambdas are erased at runtime
                self.eval(body, env)
            }

            Expr::Let(bind, body, _) => self.eval_let(bind, body, env),

            Expr::Case(scrut, alts, _, _) => self.eval_case(scrut, alts, env),

            Expr::Lazy(inner, _) => {
                // The lazy escape hatch: always create a thunk, even in strict mode.
                // This allows code that genuinely needs lazy evaluation to work
                // correctly in Numeric Profile.
                Ok(Value::Thunk(Thunk {
                    expr: inner.clone(),
                    env: env.clone(),
                }))
            }

            Expr::Cast(e, _, _) => {
                // Coercions are erased at runtime
                self.eval(e, env)
            }

            Expr::Tick(_, e, _) => {
                // Ticks are ignored during evaluation
                self.eval(e, env)
            }

            Expr::Type(_, _) | Expr::Coercion(_, _) => {
                // Types and coercions shouldn't be evaluated
                Ok(Value::unit())
            }
        }
    }

    fn eval_var(&self, var: &Var, env: &Env) -> Result<Value, EvalError> {
        // First check the local environment
        if let Some(value) = env.lookup(var.id) {
            return self.force(value.clone());
        }

        // Then check the recursive environment stack (for local recursive lets)
        // This allows closures created in recursive bindings to find their
        // siblings and themselves
        for rec_env in self.rec_env_stack.borrow().iter().rev() {
            if let Some(value) = rec_env.lookup(var.id) {
                return self.force(value.clone());
            }
        }

        // Then check the module-level environment
        // This is crucial for recursive functions: when a closure is applied,
        // it can find other module-level bindings through this env
        if let Some(ref module_env) = *self.module_env.borrow() {
            if let Some(value) = module_env.lookup(var.id) {
                return self.force(value.clone());
            }
        }

        // Then check primitives
        if let Some(value) = self.primitives.get(&var.name) {
            // Arity-0 primops (like getLine) execute immediately
            if let Value::PrimOp(op) = value {
                if op.arity() == 0 {
                    return self.apply_primop(*op, vec![]);
                }
            }
            return Ok(value.clone());
        }

        // Check if it's a primitive by its raw name
        if let Some(op) = PrimOp::from_name(var.name.as_str()) {
            // Arity-0 primops (like getLine) execute immediately
            if op.arity() == 0 {
                return self.apply_primop(op, vec![]);
            }
            return Ok(Value::PrimOp(op));
        }

        // Check if it's a dictionary field selector ($sel_N)
        // These are generated by the HIR-to-Core lowering for type class
        // dictionary method extraction.
        if let Some(index) = Self::parse_selector_name(var.name.as_str()) {
            return Ok(Value::PrimOp(PrimOp::DictSelect(index)));
        }

        // Check if it looks like a data constructor (starts with uppercase)
        // User-defined constructors won't be in primitives or the env
        let name_str = var.name.as_str();
        if !name_str.is_empty() {
            let first_char = name_str.chars().next().unwrap();
            if first_char.is_uppercase() {
                // This is a constructor - create an unsaturated DataValue
                // We don't know the exact arity here, so use a large number
                // to allow arguments to be added. Pattern matching uses tags.
                let tycon = bhc_types::TyCon::new(var.name, bhc_types::Kind::Star);
                return Ok(Value::Data(DataValue {
                    con: crate::DataCon {
                        name: var.name,
                        ty_con: tycon,
                        tag: var.id.index() as u32,
                        arity: 100, // Large arity to accept arguments
                    },
                    args: vec![],
                }));
            }
        }

        // Check named bindings (REPL persistence)
        if let Some(value) = self.named_bindings.borrow().get(&var.name) {
            return Ok(value.clone());
        }

        Err(EvalError::UnboundVariable(var.name.to_string()))
    }

    fn eval_lit(&self, lit: &Literal) -> Value {
        match lit {
            Literal::Int(n) => Value::Int(*n),
            Literal::Integer(n) => Value::Integer(*n),
            Literal::Float(n) => Value::Float(*n),
            Literal::Double(n) => Value::Double(*n),
            Literal::Char(c) => Value::Char(*c),
            Literal::String(s) => Value::String(s.as_str().into()),
        }
    }

    fn eval_app(&self, fun: &Expr, arg: &Expr, env: &Env) -> Result<Value, EvalError> {
        let fun_val = self.eval(fun, env)?;

        // Evaluate or thunk the argument based on mode
        let arg_val = match self.mode {
            EvalMode::Strict => self.eval(arg, env)?,
            EvalMode::Lazy => Value::Thunk(Thunk {
                expr: Box::new(arg.clone()),
                env: env.clone(),
            }),
        };

        self.apply(fun_val, arg_val)
    }

    fn apply(&self, fun: Value, arg: Value) -> Result<Value, EvalError> {
        match fun {
            Value::Closure(closure) => {
                let new_env = closure.env.extend(closure.var.id, arg);
                self.eval(&closure.body, &new_env)
            }

            Value::PrimOp(op) => self.apply_primop(op, vec![arg]),

            Value::PartialPrimOp(op, mut args) => {
                // Add argument to partial application
                args.push(arg);
                self.apply_primop(op, args)
            }

            Value::Data(mut data) if !data.is_saturated() => {
                // Partial application of data constructor
                data.args.push(arg);
                Ok(Value::Data(data))
            }

            Value::Thunk(thunk) => {
                // Force the thunk and retry
                let forced = self.force_thunk(&thunk)?;
                self.apply(forced, arg)
            }

            other => Err(EvalError::TypeError {
                expected: "function".to_string(),
                got: format!("{other:?}"),
            }),
        }
    }

    fn apply_primop(&self, op: PrimOp, args: Vec<Value>) -> Result<Value, EvalError> {
        // Check if we have enough arguments
        if args.len() < op.arity() {
            // Return a partially applied primop as a closure
            // For simplicity, we'll create a wrapper
            return self.partial_primop(op, args);
        }

        // Force all arguments
        let forced: Result<Vec<_>, _> = args.into_iter().map(|a| self.force(a)).collect();
        let args = forced?;

        match op {
            PrimOp::AddInt => {
                let a = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let b = args[1].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[1]),
                })?;
                Ok(Value::Int(a.wrapping_add(b)))
            }

            PrimOp::SubInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::Int(a.wrapping_sub(b)))
            }

            PrimOp::MulInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::Int(a.wrapping_mul(b)))
            }

            PrimOp::DivInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                if b == 0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Int(a / b))
            }

            PrimOp::ModInt => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                if b == 0 {
                    return Err(EvalError::DivisionByZero);
                }
                Ok(Value::Int(a % b))
            }

            PrimOp::NegInt => {
                let a = args[0].as_int().unwrap_or(0);
                Ok(Value::Int(-a))
            }

            PrimOp::AddDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                Ok(Value::Double(a + b))
            }

            PrimOp::SubDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                Ok(Value::Double(a - b))
            }

            PrimOp::MulDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                Ok(Value::Double(a * b))
            }

            PrimOp::DivDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                Ok(Value::Double(a / b))
            }

            PrimOp::NegDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                Ok(Value::Double(-a))
            }

            PrimOp::EqInt => {
                // Polymorphic equality: dispatch based on argument types.
                // This handles derived Eq for types with Int, Bool, Char,
                // Float, Double fields, and enum/ADT types.
                match (&args[0], &args[1]) {
                    (Value::Int(a), Value::Int(b)) => Ok(Value::bool(a == b)),
                    (Value::Double(a), Value::Double(b)) =>
                    {
                        #[allow(clippy::float_cmp)]
                        Ok(Value::bool(a == b))
                    }
                    (Value::Float(a), Value::Float(b)) =>
                    {
                        #[allow(clippy::float_cmp)]
                        Ok(Value::bool(a == b))
                    }
                    (Value::Char(a), Value::Char(b)) => Ok(Value::bool(a == b)),
                    (Value::String(a), Value::String(b)) => Ok(Value::bool(a == b)),
                    (Value::Data(a), Value::Data(b)) => {
                        // For nullary constructors (enums), compare tags
                        if a.args.is_empty() && b.args.is_empty() {
                            Ok(Value::bool(a.con.tag == b.con.tag))
                        } else if a.con.tag != b.con.tag {
                            // Different constructors are never equal
                            Ok(Value::bool(false))
                        } else {
                            // Same constructor: recursively compare all fields
                            if a.args.len() != b.args.len() {
                                Ok(Value::bool(false))
                            } else {
                                for (fa, fb) in a.args.iter().zip(b.args.iter()) {
                                    let fa = self.force(fa.clone())?;
                                    let fb = self.force(fb.clone())?;
                                    let eq = self.apply_primop(PrimOp::EqInt, vec![fa, fb])?;
                                    if eq.as_bool() == Some(false) {
                                        return Ok(Value::bool(false));
                                    }
                                }
                                Ok(Value::bool(true))
                            }
                        }
                    }
                    _ => {
                        // Fallback to int comparison for backward compatibility
                        let a = args[0].as_int().unwrap_or(0);
                        let b = args[1].as_int().unwrap_or(0);
                        Ok(Value::bool(a == b))
                    }
                }
            }

            PrimOp::LtInt => {
                // Polymorphic less-than for derived Ord support
                match (&args[0], &args[1]) {
                    (Value::Int(a), Value::Int(b)) => Ok(Value::bool(a < b)),
                    (Value::Double(a), Value::Double(b)) => Ok(Value::bool(a < b)),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::bool(a < b)),
                    (Value::Char(a), Value::Char(b)) => Ok(Value::bool(a < b)),
                    (Value::Data(a), Value::Data(b)) => Ok(Value::bool(a.con.tag < b.con.tag)),
                    _ => {
                        let a = args[0].as_int().unwrap_or(0);
                        let b = args[1].as_int().unwrap_or(0);
                        Ok(Value::bool(a < b))
                    }
                }
            }

            PrimOp::LeInt => match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::bool(a <= b)),
                (Value::Double(a), Value::Double(b)) => Ok(Value::bool(a <= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::bool(a <= b)),
                (Value::Char(a), Value::Char(b)) => Ok(Value::bool(a <= b)),
                (Value::Data(a), Value::Data(b)) => Ok(Value::bool(a.con.tag <= b.con.tag)),
                _ => {
                    let a = args[0].as_int().unwrap_or(0);
                    let b = args[1].as_int().unwrap_or(0);
                    Ok(Value::bool(a <= b))
                }
            },

            PrimOp::GtInt => match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::bool(a > b)),
                (Value::Double(a), Value::Double(b)) => Ok(Value::bool(a > b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::bool(a > b)),
                (Value::Char(a), Value::Char(b)) => Ok(Value::bool(a > b)),
                (Value::Data(a), Value::Data(b)) => Ok(Value::bool(a.con.tag > b.con.tag)),
                _ => {
                    let a = args[0].as_int().unwrap_or(0);
                    let b = args[1].as_int().unwrap_or(0);
                    Ok(Value::bool(a > b))
                }
            },

            PrimOp::GeInt => match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::bool(a >= b)),
                (Value::Double(a), Value::Double(b)) => Ok(Value::bool(a >= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::bool(a >= b)),
                (Value::Char(a), Value::Char(b)) => Ok(Value::bool(a >= b)),
                (Value::Data(a), Value::Data(b)) => Ok(Value::bool(a.con.tag >= b.con.tag)),
                _ => {
                    let a = args[0].as_int().unwrap_or(0);
                    let b = args[1].as_int().unwrap_or(0);
                    Ok(Value::bool(a >= b))
                }
            },

            PrimOp::EqDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                #[allow(clippy::float_cmp)]
                Ok(Value::bool(a == b))
            }

            PrimOp::LtDouble => {
                let a = args[0].as_double().unwrap_or(0.0);
                let b = args[1].as_double().unwrap_or(0.0);
                Ok(Value::bool(a < b))
            }

            PrimOp::AndBool => {
                let a = args[0].as_bool().unwrap_or(false);
                let b = args[1].as_bool().unwrap_or(false);
                Ok(Value::bool(a && b))
            }

            PrimOp::OrBool => {
                let a = args[0].as_bool().unwrap_or(false);
                let b = args[1].as_bool().unwrap_or(false);
                Ok(Value::bool(a || b))
            }

            PrimOp::NotBool => {
                let a = args[0].as_bool().unwrap_or(false);
                Ok(Value::bool(!a))
            }

            PrimOp::IntToDouble => {
                let a = args[0].as_int().unwrap_or(0);
                Ok(Value::Double(a as f64))
            }

            PrimOp::DoubleToInt => {
                let a = args[0].as_double().unwrap_or(0.0);
                #[allow(clippy::cast_possible_truncation)]
                Ok(Value::Int(a as i64))
            }

            PrimOp::EqChar => {
                let a = match &args[0] {
                    Value::Char(c) => *c,
                    _ => '\0',
                };
                let b = match &args[1] {
                    Value::Char(c) => *c,
                    _ => '\0',
                };
                Ok(Value::bool(a == b))
            }

            PrimOp::CharToInt => {
                let c = match &args[0] {
                    Value::Char(c) => *c,
                    _ => '\0',
                };
                Ok(Value::Int(i64::from(u32::from(c))))
            }

            PrimOp::IntToChar => {
                let n = args[0].as_int().unwrap_or(0);
                #[allow(clippy::cast_sign_loss)]
                let c = char::from_u32(n as u32).unwrap_or('\0');
                Ok(Value::Char(c))
            }

            PrimOp::Seq => {
                // seq forces first argument, returns second
                let _ = self.force(args[0].clone())?;
                Ok(args[1].clone())
            }

            PrimOp::Error => {
                let msg = match &args[0] {
                    Value::String(s) => s.to_string(),
                    other => format!("{other:?}"),
                };
                Err(EvalError::UserError(msg))
            }

            // UArray operations
            PrimOp::UArrayFromList => {
                // Convert a list to a UArray
                let list = &args[0];
                if let Some(arr) = Value::uarray_int_from_list(list) {
                    Ok(arr)
                } else if let Some(arr) = Value::uarray_double_from_list(list) {
                    Ok(arr)
                } else {
                    Err(EvalError::TypeError {
                        expected: "list of Int or Double".into(),
                        got: format!("{list:?}"),
                    })
                }
            }

            PrimOp::UArrayToList => {
                // Convert a UArray back to a list
                args[0]
                    .uarray_to_list()
                    .ok_or_else(|| EvalError::TypeError {
                        expected: "UArray".into(),
                        got: format!("{:?}", args[0]),
                    })
            }

            PrimOp::UArrayMap => {
                // map f arr
                let f = &args[0];
                let arr = &args[1];

                match arr {
                    Value::UArrayInt(uarr) => {
                        let mapped: Result<Vec<i64>, _> = uarr
                            .as_slice()
                            .iter()
                            .map(|x| {
                                let result = self.apply(f.clone(), Value::Int(*x))?;
                                let forced = self.force(result)?;
                                forced.as_int().ok_or_else(|| EvalError::TypeError {
                                    expected: "Int".into(),
                                    got: format!("{forced:?}"),
                                })
                            })
                            .collect();
                        Ok(Value::UArrayInt(crate::uarray::UArray::from_vec(mapped?)))
                    }
                    Value::UArrayDouble(uarr) => {
                        let mapped: Result<Vec<f64>, _> = uarr
                            .as_slice()
                            .iter()
                            .map(|x| {
                                let result = self.apply(f.clone(), Value::Double(*x))?;
                                let forced = self.force(result)?;
                                forced.as_double().ok_or_else(|| EvalError::TypeError {
                                    expected: "Double".into(),
                                    got: format!("{forced:?}"),
                                })
                            })
                            .collect();
                        Ok(Value::UArrayDouble(crate::uarray::UArray::from_vec(
                            mapped?,
                        )))
                    }
                    // Support mapping over lists - need to force thunks while traversing
                    _ => {
                        // Try to traverse list, forcing thunks along the way
                        let list_result = self.force_list(arr.clone())?;
                        let mapped: Result<Vec<Value>, _> = list_result
                            .iter()
                            .map(|x| {
                                let result = self.apply(f.clone(), x.clone())?;
                                self.force(result)
                            })
                            .collect();
                        Ok(Value::from_list(mapped?))
                    }
                }
            }

            PrimOp::UArrayZipWith => {
                // zipWith f arr1 arr2
                let f = &args[0];
                let arr1 = &args[1];
                let arr2 = &args[2];

                match (arr1, arr2) {
                    (Value::UArrayInt(a), Value::UArrayInt(b)) => {
                        let zipped: Result<Vec<i64>, _> = a
                            .as_slice()
                            .iter()
                            .zip(b.as_slice().iter())
                            .map(|(x, y)| {
                                let result = self.apply(
                                    self.apply(f.clone(), Value::Int(*x))?,
                                    Value::Int(*y),
                                )?;
                                let forced = self.force(result)?;
                                forced.as_int().ok_or_else(|| EvalError::TypeError {
                                    expected: "Int".into(),
                                    got: format!("{forced:?}"),
                                })
                            })
                            .collect();
                        Ok(Value::UArrayInt(crate::uarray::UArray::from_vec(zipped?)))
                    }
                    (Value::UArrayDouble(a), Value::UArrayDouble(b)) => {
                        let zipped: Result<Vec<f64>, _> = a
                            .as_slice()
                            .iter()
                            .zip(b.as_slice().iter())
                            .map(|(x, y)| {
                                let result = self.apply(
                                    self.apply(f.clone(), Value::Double(*x))?,
                                    Value::Double(*y),
                                )?;
                                let forced = self.force(result)?;
                                forced.as_double().ok_or_else(|| EvalError::TypeError {
                                    expected: "Double".into(),
                                    got: format!("{forced:?}"),
                                })
                            })
                            .collect();
                        Ok(Value::UArrayDouble(crate::uarray::UArray::from_vec(
                            zipped?,
                        )))
                    }
                    // Support lists - force thunks while traversing
                    _ => {
                        let list1 = self.force_list(arr1.clone())?;
                        let list2 = self.force_list(arr2.clone())?;
                        let zipped: Result<Vec<Value>, _> = list1
                            .iter()
                            .zip(list2.iter())
                            .map(|(x, y)| {
                                let result =
                                    self.apply(self.apply(f.clone(), x.clone())?, y.clone())?;
                                self.force(result)
                            })
                            .collect();
                        Ok(Value::from_list(zipped?))
                    }
                }
            }

            PrimOp::UArrayFold => {
                // fold f init arr
                let f = &args[0];
                let init = args[1].clone();
                let arr = &args[2];

                match arr {
                    Value::UArrayInt(uarr) => {
                        let mut acc = init;
                        for x in uarr.as_slice() {
                            let result = self.apply(self.apply(f.clone(), acc)?, Value::Int(*x))?;
                            acc = self.force(result)?;
                        }
                        Ok(acc)
                    }
                    Value::UArrayDouble(uarr) => {
                        let mut acc = init;
                        for x in uarr.as_slice() {
                            let result =
                                self.apply(self.apply(f.clone(), acc)?, Value::Double(*x))?;
                            acc = self.force(result)?;
                        }
                        Ok(acc)
                    }
                    _ => {
                        // Force the list and fold over it
                        let list = self.force_list(arr.clone())?;
                        let mut acc = init;
                        for x in list {
                            let result = self.apply(self.apply(f.clone(), acc)?, x)?;
                            acc = self.force(result)?;
                        }
                        Ok(acc)
                    }
                }
            }

            PrimOp::UArraySum => {
                // sum arr - works on both UArrays and lists
                match &args[0] {
                    Value::UArrayInt(arr) => Ok(Value::Int(arr.sum())),
                    Value::UArrayDouble(arr) => Ok(Value::Double(arr.sum())),
                    _ => {
                        // Force the list and sum it
                        let list = self.force_list(args[0].clone())?;
                        // Try to sum as integers first
                        let ints: Option<i64> = list
                            .iter()
                            .map(Value::as_int)
                            .try_fold(0i64, |acc, x| x.map(|n| acc.wrapping_add(n)));
                        if let Some(sum) = ints {
                            return Ok(Value::Int(sum));
                        }
                        // Try as doubles
                        let doubles: Option<f64> = list
                            .iter()
                            .map(Value::as_double)
                            .try_fold(0.0f64, |acc, x| x.map(|n| acc + n));
                        if let Some(sum) = doubles {
                            return Ok(Value::Double(sum));
                        }
                        Err(EvalError::TypeError {
                            expected: "list of numbers".into(),
                            got: format!("{:?}", args[0]),
                        })
                    }
                }
            }

            PrimOp::UArrayLength => {
                match &args[0] {
                    Value::UArrayInt(arr) => Ok(Value::Int(arr.len() as i64)),
                    Value::UArrayDouble(arr) => Ok(Value::Int(arr.len() as i64)),
                    _ => {
                        // Force the list and get its length
                        let list = self.force_list(args[0].clone())?;
                        Ok(Value::Int(list.len() as i64))
                    }
                }
            }

            PrimOp::UArrayRange => {
                // range start end
                let start = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let end = args[1].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[1]),
                })?;
                Ok(Value::UArrayInt(crate::uarray::UArray::range(start, end)))
            }

            // List operations
            PrimOp::Concat => {
                // xs ++ ys - concatenate two lists
                let xs = self.force_list(args[0].clone())?;
                let ys = self.force_list(args[1].clone())?;
                let mut result = xs;
                result.extend(ys);
                Ok(Value::from_list(result))
            }

            PrimOp::ConcatMap => {
                // concatMap f xs - map f over xs and concatenate results
                let f = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut result = Vec::new();
                for x in xs {
                    let mapped = self.apply(f.clone(), x)?;
                    let forced = self.force(mapped)?;
                    let list = self.force_list(forced)?;
                    result.extend(list);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Append => {
                // append x xs - add x to the end of xs
                let x = args[0].clone();
                let xs = self.force_list(args[1].clone())?;
                let mut result = xs;
                result.push(x);
                Ok(Value::from_list(result))
            }

            // Monad operations for list
            PrimOp::ListBind => {
                // xs >>= f = concatMap f xs
                // Argument order: xs, f
                let xs = self.force_list(args[0].clone())?;
                let f = &args[1];
                let mut result = Vec::new();
                for x in xs {
                    let mapped = self.apply(f.clone(), x)?;
                    let forced = self.force(mapped)?;
                    let list = self.force_list(forced)?;
                    result.extend(list);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::ListThen => {
                // xs >> ys = xs >>= \_ -> ys
                // Argument order: xs, ys
                let xs = self.force_list(args[0].clone())?;
                let ys = args[1].clone();
                let ys_list = self.force_list(ys)?;
                let mut result = Vec::new();
                for _ in xs {
                    result.extend(ys_list.clone());
                }
                Ok(Value::from_list(result))
            }

            PrimOp::ListReturn => {
                // return x = [x]
                let x = args[0].clone();
                Ok(Value::from_list(vec![x]))
            }

            // Additional list operations
            PrimOp::Foldr => {
                // foldr f z xs
                // foldr f z [] = z
                // foldr f z (x:xs) = f x (foldr f z xs)
                let f = &args[0];
                let z = args[1].clone();
                let xs = self.force_list(args[2].clone())?;

                // Process from right to left
                let mut acc = z;
                for x in xs.into_iter().rev() {
                    acc = self.apply(self.apply(f.clone(), x)?, acc)?;
                }
                Ok(acc)
            }

            PrimOp::Foldl => {
                // foldl f z xs
                // foldl f z [] = z
                // foldl f z (x:xs) = foldl f (f z x) xs
                let f = &args[0];
                let z = args[1].clone();
                let xs = self.force_list(args[2].clone())?;

                let mut acc = z;
                for x in xs {
                    acc = self.apply(self.apply(f.clone(), acc)?, x)?;
                }
                Ok(acc)
            }

            PrimOp::FoldlStrict => {
                // foldl' f z xs - strict left fold
                let f = &args[0];
                let z = args[1].clone();
                let xs = self.force_list(args[2].clone())?;

                let mut acc = self.force(z)?;
                for x in xs {
                    acc = self.apply(self.apply(f.clone(), acc)?, x)?;
                    acc = self.force(acc)?;
                }
                Ok(acc)
            }

            PrimOp::Filter => {
                // filter p xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;

                let mut result = Vec::new();
                for x in xs {
                    let pred_result = self.apply(p.clone(), x.clone())?;
                    let pred_forced = self.force(pred_result)?;
                    if pred_forced.as_bool().unwrap_or(false) {
                        result.push(x);
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Zip => {
                // zip xs ys = zipWith (,) xs ys
                let xs = self.force_list(args[0].clone())?;
                let ys = self.force_list(args[1].clone())?;

                let pairs: Vec<Value> = xs
                    .into_iter()
                    .zip(ys.into_iter())
                    .map(|(x, y)| {
                        use bhc_types::{Kind, TyCon};
                        Value::Data(DataValue {
                            con: crate::DataCon {
                                name: Symbol::intern("(,)"),
                                ty_con: TyCon::new(Symbol::intern("(,)"), Kind::Star),
                                tag: 0,
                                arity: 2,
                            },
                            args: vec![x, y],
                        })
                    })
                    .collect();
                Ok(Value::from_list(pairs))
            }

            PrimOp::ZipWith => {
                // zipWith f xs ys
                let f = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let ys = self.force_list(args[2].clone())?;

                let mut result = Vec::new();
                for (x, y) in xs.into_iter().zip(ys.into_iter()) {
                    let r = self.apply(self.apply(f.clone(), x)?, y)?;
                    result.push(self.force(r)?);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Take => {
                // take n xs
                let n = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let xs = self.force_list(args[1].clone())?;

                let n = n.max(0) as usize;
                let result: Vec<Value> = xs.into_iter().take(n).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Drop => {
                // drop n xs
                let n = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let xs = self.force_list(args[1].clone())?;

                let n = n.max(0) as usize;
                let result: Vec<Value> = xs.into_iter().skip(n).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Head => {
                // head xs - partial, errors on empty list
                let xs = self.force_list(args[0].clone())?;
                xs.into_iter()
                    .next()
                    .ok_or(EvalError::UserError("head: empty list".to_string()))
            }

            PrimOp::Tail => {
                // tail xs - partial, errors on empty list
                let xs = self.force_list(args[0].clone())?;
                if xs.is_empty() {
                    return Err(EvalError::UserError("tail: empty list".to_string()));
                }
                let result: Vec<Value> = xs.into_iter().skip(1).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Last => {
                // last xs - partial, errors on empty list
                let xs = self.force_list(args[0].clone())?;
                xs.into_iter()
                    .last()
                    .ok_or(EvalError::UserError("last: empty list".to_string()))
            }

            PrimOp::Init => {
                // init xs - partial, errors on empty list
                let xs = self.force_list(args[0].clone())?;
                if xs.is_empty() {
                    return Err(EvalError::UserError("init: empty list".to_string()));
                }
                let len = xs.len();
                let result: Vec<Value> = xs.into_iter().take(len - 1).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Reverse => {
                // reverse xs
                let xs = self.force_list(args[0].clone())?;
                let result: Vec<Value> = xs.into_iter().rev().collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Null => {
                // null xs - returns True if list is empty
                let xs = self.force_list(args[0].clone())?;
                Ok(Value::bool(xs.is_empty()))
            }

            PrimOp::Index => {
                // xs !! n - index into list
                let xs = self.force_list(args[0].clone())?;
                let n = args[1].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[1]),
                })?;

                if n < 0 {
                    return Err(EvalError::UserError("(!!): negative index".to_string()));
                }
                let n = n as usize;
                xs.into_iter()
                    .nth(n)
                    .ok_or_else(|| EvalError::UserError(format!("(!!): index {} too large", n)))
            }

            PrimOp::Replicate => {
                // replicate n x
                let n = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let x = args[1].clone();

                let n = n.max(0) as usize;
                let result: Vec<Value> = std::iter::repeat(x).take(n).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::EnumFromTo => {
                // enumFromTo start end = [start..end]
                let start = args[0].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[0]),
                })?;
                let end = args[1].as_int().ok_or_else(|| EvalError::TypeError {
                    expected: "Int".into(),
                    got: format!("{:?}", args[1]),
                })?;

                let result: Vec<Value> = (start..=end).map(Value::Int).collect();
                Ok(Value::from_list(result))
            }

            // Additional list/prelude operations
            PrimOp::Even => {
                let n = args[0].as_int().unwrap_or(0);
                Ok(Value::bool(n % 2 == 0))
            }

            PrimOp::Odd => {
                let n = args[0].as_int().unwrap_or(0);
                Ok(Value::bool(n % 2 != 0))
            }

            PrimOp::Elem => {
                // elem x xs
                let x = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let found = xs.iter().any(|y| self.values_equal(x, y));
                Ok(Value::bool(found))
            }

            PrimOp::NotElem => {
                // notElem x xs
                let x = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let found = xs.iter().any(|y| self.values_equal(x, y));
                Ok(Value::bool(!found))
            }

            PrimOp::TakeWhile => {
                // takeWhile p xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut result = Vec::new();
                for x in xs {
                    let pred_result = self.apply(p.clone(), x.clone())?;
                    let pred_forced = self.force(pred_result)?;
                    if pred_forced.as_bool().unwrap_or(false) {
                        result.push(x);
                    } else {
                        break;
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::DropWhile => {
                // dropWhile p xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut dropping = true;
                let mut result = Vec::new();
                for x in xs {
                    if dropping {
                        let pred_result = self.apply(p.clone(), x.clone())?;
                        let pred_forced = self.force(pred_result)?;
                        if pred_forced.as_bool().unwrap_or(false) {
                            continue;
                        }
                        dropping = false;
                    }
                    result.push(x);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Span => {
                // span p xs = (takeWhile p xs, dropWhile p xs)
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut prefix = Vec::new();
                let mut rest_start = 0;
                for (i, x) in xs.iter().enumerate() {
                    let pred_result = self.apply(p.clone(), x.clone())?;
                    let pred_forced = self.force(pred_result)?;
                    if pred_forced.as_bool().unwrap_or(false) {
                        prefix.push(x.clone());
                    } else {
                        rest_start = i;
                        break;
                    }
                    rest_start = i + 1;
                }
                let rest: Vec<Value> = xs.into_iter().skip(rest_start).collect();
                Ok(self.make_pair(Value::from_list(prefix), Value::from_list(rest)))
            }

            PrimOp::Break => {
                // break p xs = span (not . p) xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut prefix = Vec::new();
                let mut rest_start = 0;
                for (i, x) in xs.iter().enumerate() {
                    let pred_result = self.apply(p.clone(), x.clone())?;
                    let pred_forced = self.force(pred_result)?;
                    if !pred_forced.as_bool().unwrap_or(false) {
                        prefix.push(x.clone());
                    } else {
                        rest_start = i;
                        break;
                    }
                    rest_start = i + 1;
                }
                let rest: Vec<Value> = xs.into_iter().skip(rest_start).collect();
                Ok(self.make_pair(Value::from_list(prefix), Value::from_list(rest)))
            }

            PrimOp::SplitAt => {
                // splitAt n xs
                let n = args[0].as_int().unwrap_or(0).max(0) as usize;
                let xs = self.force_list(args[1].clone())?;
                let (prefix, rest) = if n >= xs.len() {
                    (xs, Vec::new())
                } else {
                    let mut v = xs;
                    let rest = v.split_off(n);
                    (v, rest)
                };
                Ok(self.make_pair(Value::from_list(prefix), Value::from_list(rest)))
            }

            PrimOp::Iterate => {
                // iterate f x - produces [x, f x, f (f x), ...]
                // Finite approximation: produce up to 1000 elements
                let f = &args[0];
                let mut current = args[1].clone();
                let limit = 1000;
                let mut result = Vec::with_capacity(limit);
                for _ in 0..limit {
                    result.push(current.clone());
                    current = self.apply(f.clone(), current)?;
                    current = self.force(current)?;
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Repeat => {
                // repeat x - infinite list of x (truncated to 1000)
                let x = args[0].clone();
                let result: Vec<Value> = std::iter::repeat(x).take(1000).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Cycle => {
                // cycle xs - infinite repetition (truncated to 1000)
                let xs = self.force_list(args[0].clone())?;
                if xs.is_empty() {
                    return Err(EvalError::UserError("cycle: empty list".to_string()));
                }
                let limit = 1000;
                let mut result = Vec::with_capacity(limit);
                let mut i = 0;
                while result.len() < limit {
                    result.push(xs[i % xs.len()].clone());
                    i += 1;
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Lookup => {
                // lookup k xs where xs is [(k, v)]
                let key = &args[0];
                let xs = self.force_list(args[1].clone())?;
                for pair in xs {
                    let pair = self.force(pair)?;
                    if let Value::Data(ref d) = pair {
                        if d.con.name.as_str() == "(,)" && d.args.len() == 2 {
                            let k = self.force(d.args[0].clone())?;
                            if self.values_equal(key, &k) {
                                let v = d.args[1].clone();
                                return Ok(self.make_just(v));
                            }
                        }
                    }
                }
                Ok(self.make_nothing())
            }

            PrimOp::Unzip => {
                // unzip :: [(a, b)] -> ([a], [b])
                let xs = self.force_list(args[0].clone())?;
                let mut as_list = Vec::new();
                let mut bs_list = Vec::new();
                for pair in xs {
                    let pair = self.force(pair)?;
                    if let Value::Data(ref d) = pair {
                        if d.con.name.as_str() == "(,)" && d.args.len() == 2 {
                            as_list.push(d.args[0].clone());
                            bs_list.push(d.args[1].clone());
                        }
                    }
                }
                Ok(self.make_pair(Value::from_list(as_list), Value::from_list(bs_list)))
            }

            PrimOp::Product => {
                // product xs
                match &args[0] {
                    Value::UArrayInt(arr) => {
                        let p: i64 = arr.as_slice().iter().product();
                        Ok(Value::Int(p))
                    }
                    _ => {
                        let list = self.force_list(args[0].clone())?;
                        let mut acc: i64 = 1;
                        for x in list {
                            if let Some(n) = x.as_int() {
                                acc = acc.wrapping_mul(n);
                            }
                        }
                        Ok(Value::Int(acc))
                    }
                }
            }

            PrimOp::Flip => {
                // flip f x y = f y x
                let f = &args[0];
                let x = args[1].clone();
                let y = args[2].clone();
                let partial = self.apply(f.clone(), y)?;
                self.apply(partial, x)
            }

            PrimOp::Min => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::Int(a.min(b)))
            }

            PrimOp::Max => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                Ok(Value::Int(a.max(b)))
            }

            PrimOp::FromIntegral => {
                // Identity for now (Int -> Int)
                Ok(args[0].clone())
            }

            PrimOp::MaybeElim => {
                // maybe def f m
                let def = args[0].clone();
                let f = &args[1];
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Data(d) if d.con.name.as_str() == "Nothing" => Ok(def),
                    Value::Data(d) if d.con.name.as_str() == "Just" && d.args.len() == 1 => {
                        self.apply(f.clone(), d.args[0].clone())
                    }
                    _ => Ok(def),
                }
            }

            PrimOp::FromMaybe => {
                // fromMaybe def m
                let def = args[0].clone();
                let m = self.force(args[1].clone())?;
                match &m {
                    Value::Data(d) if d.con.name.as_str() == "Nothing" => Ok(def),
                    Value::Data(d) if d.con.name.as_str() == "Just" && d.args.len() == 1 => {
                        Ok(d.args[0].clone())
                    }
                    _ => Ok(def),
                }
            }

            PrimOp::EitherElim => {
                // either f g e
                let f = &args[0];
                let g = &args[1];
                let e = self.force(args[2].clone())?;
                match &e {
                    Value::Data(d) if d.con.name.as_str() == "Left" && d.args.len() == 1 => {
                        self.apply(f.clone(), d.args[0].clone())
                    }
                    Value::Data(d) if d.con.name.as_str() == "Right" && d.args.len() == 1 => {
                        self.apply(g.clone(), d.args[0].clone())
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "Either".into(),
                        got: format!("{e:?}"),
                    }),
                }
            }

            PrimOp::IsJust => {
                let m = self.force(args[0].clone())?;
                match &m {
                    Value::Data(d) if d.con.name.as_str() == "Just" => Ok(Value::bool(true)),
                    Value::Data(d) if d.con.name.as_str() == "Nothing" => Ok(Value::bool(false)),
                    _ => Ok(Value::bool(false)),
                }
            }

            PrimOp::IsNothing => {
                let m = self.force(args[0].clone())?;
                match &m {
                    Value::Data(d) if d.con.name.as_str() == "Nothing" => Ok(Value::bool(true)),
                    Value::Data(d) if d.con.name.as_str() == "Just" => Ok(Value::bool(false)),
                    _ => Ok(Value::bool(true)),
                }
            }

            PrimOp::Abs => {
                let n = args[0].as_int().unwrap_or(0);
                Ok(Value::Int(n.abs()))
            }

            PrimOp::Signum => {
                let n = args[0].as_int().unwrap_or(0);
                Ok(Value::Int(n.signum()))
            }

            PrimOp::Curry => {
                // curry f x y = f (x, y)
                let f = &args[0];
                let x = args[1].clone();
                let y = args[2].clone();
                let pair = self.make_pair(x, y);
                self.apply(f.clone(), pair)
            }

            PrimOp::Uncurry => {
                // uncurry f (x, y) = f x y
                let f = &args[0];
                let pair = self.force(args[1].clone())?;
                if let Value::Data(ref d) = pair {
                    if d.con.name.as_str() == "(,)" && d.args.len() == 2 {
                        let partial = self.apply(f.clone(), d.args[0].clone())?;
                        return self.apply(partial, d.args[1].clone());
                    }
                }
                Err(EvalError::TypeError {
                    expected: "pair (a, b)".into(),
                    got: format!("{pair:?}"),
                })
            }

            PrimOp::Swap => {
                // swap (a, b) = (b, a)
                let pair = self.force(args[0].clone())?;
                if let Value::Data(ref d) = pair {
                    if d.con.name.as_str() == "(,)" && d.args.len() == 2 {
                        return Ok(self.make_pair(d.args[1].clone(), d.args[0].clone()));
                    }
                }
                Err(EvalError::TypeError {
                    expected: "pair (a, b)".into(),
                    got: format!("{pair:?}"),
                })
            }

            PrimOp::Any => {
                // any p xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                for x in xs {
                    let pred_result = self.apply(p.clone(), x)?;
                    let pred_forced = self.force(pred_result)?;
                    if pred_forced.as_bool().unwrap_or(false) {
                        return Ok(Value::bool(true));
                    }
                }
                Ok(Value::bool(false))
            }

            PrimOp::All => {
                // all p xs
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                for x in xs {
                    let pred_result = self.apply(p.clone(), x)?;
                    let pred_forced = self.force(pred_result)?;
                    if !pred_forced.as_bool().unwrap_or(true) {
                        return Ok(Value::bool(false));
                    }
                }
                Ok(Value::bool(true))
            }

            PrimOp::And => {
                // and xs - all True
                let xs = self.force_list(args[0].clone())?;
                for x in xs {
                    if !x.as_bool().unwrap_or(true) {
                        return Ok(Value::bool(false));
                    }
                }
                Ok(Value::bool(true))
            }

            PrimOp::Or => {
                // or xs - any True
                let xs = self.force_list(args[0].clone())?;
                for x in xs {
                    if x.as_bool().unwrap_or(false) {
                        return Ok(Value::bool(true));
                    }
                }
                Ok(Value::bool(false))
            }

            PrimOp::Lines => {
                // lines :: String -> [String]
                let s = self.value_to_string(&args[0])?;
                let result: Vec<Value> = s.lines().map(|l| Value::String(l.into())).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Unlines => {
                // unlines :: [String] -> String
                let xs = self.force_list(args[0].clone())?;
                let mut result = String::new();
                for x in xs {
                    let s = self.value_to_string(&x)?;
                    result.push_str(&s);
                    result.push('\n');
                }
                Ok(Value::String(result.into()))
            }

            PrimOp::Words => {
                // words :: String -> [String]
                let s = self.value_to_string(&args[0])?;
                let result: Vec<Value> = s
                    .split_whitespace()
                    .map(|w| Value::String(w.into()))
                    .collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Unwords => {
                // unwords :: [String] -> String
                let xs = self.force_list(args[0].clone())?;
                let mut parts = Vec::new();
                for x in xs {
                    let s = self.value_to_string(&x)?;
                    parts.push(s);
                }
                Ok(Value::String(parts.join(" ").into()))
            }

            PrimOp::Show => {
                // show :: a -> String
                let displayed = self.display_value(&args[0])?;
                Ok(Value::String(displayed.into()))
            }

            PrimOp::Id => {
                // id :: a -> a
                Ok(args[0].clone())
            }

            PrimOp::Const => {
                // const :: a -> b -> a
                Ok(args[0].clone())
            }

            // IO operations
            PrimOp::PutStrLn => {
                // putStrLn :: String -> IO ()
                let s = self.value_to_string(&args[0])?;
                println!("{s}");
                {
                    let mut buf = self.io_output.borrow_mut();
                    buf.push_str(&s);
                    buf.push('\n');
                }
                Ok(Value::unit())
            }

            PrimOp::PutStr => {
                // putStr :: String -> IO ()
                use std::io::Write;
                let s = self.value_to_string(&args[0])?;
                print!("{s}");
                std::io::stdout().flush().ok();
                self.io_output.borrow_mut().push_str(&s);
                Ok(Value::unit())
            }

            PrimOp::Print => {
                // print :: Show a => a -> IO ()
                let displayed = self.display_value(&args[0])?;
                println!("{displayed}");
                {
                    let mut buf = self.io_output.borrow_mut();
                    buf.push_str(&displayed);
                    buf.push('\n');
                }
                Ok(Value::unit())
            }

            PrimOp::GetLine => {
                // getLine :: IO String
                use std::io::BufRead;
                let stdin = std::io::stdin();
                let mut line = String::new();
                stdin
                    .lock()
                    .read_line(&mut line)
                    .map_err(|e| EvalError::UserError(format!("getLine failed: {e}")))?;
                // Remove trailing newline
                if line.ends_with('\n') {
                    line.pop();
                    if line.ends_with('\r') {
                        line.pop();
                    }
                }
                Ok(Value::String(line.into()))
            }

            PrimOp::IoBind => {
                // (>>=) :: IO a -> (a -> IO b) -> IO b
                // In our interpreter, IO actions execute immediately,
                // so we just run the first action and pass result to continuation
                let io_a = args[0].clone();
                let f = &args[1];
                // "Execute" the IO action (it's already a value in our interpreter)
                // and apply the continuation
                self.apply(f.clone(), io_a)
            }

            PrimOp::IoThen => {
                // (>>) :: IO a -> IO b -> IO b
                // Execute first action (already done), return second
                Ok(args[1].clone())
            }

            PrimOp::IoReturn => {
                // return :: a -> IO a
                // In our interpreter, just return the value as-is
                Ok(args[0].clone())
            }

            PrimOp::DictSelect(index) => {
                // Extract field at `index` from a dictionary (tuple-like DataValue).
                // Dictionaries are represented as nested tuple applications:
                // e.g., (,,) method0 method1 method2
                let dict = self.force(args[0].clone())?;
                match dict {
                    Value::Data(ref d) => {
                        if index < d.args.len() {
                            self.force(d.args[index].clone())
                        } else {
                            Err(EvalError::UserError(format!(
                                "dictionary field selector $sel_{index} out of range \
                                 (dictionary has {} fields, constructor {})",
                                d.args.len(),
                                d.con.name
                            )))
                        }
                    }
                    // If the dictionary is a single-field value (not wrapped in a tuple),
                    // and we're selecting field 0, just return it directly.
                    _ if index == 0 => Ok(dict),
                    _ => Err(EvalError::TypeError {
                        expected: "dictionary (tuple)".into(),
                        got: format!("{dict:?}"),
                    }),
                }
            }

            // === Enum operations ===
            PrimOp::Succ => {
                match &args[0] {
                    Value::Int(n) => Ok(Value::Int(n + 1)),
                    Value::Char(c) => {
                        let n = u32::from(*c) + 1;
                        Ok(Value::Char(char::from_u32(n).unwrap_or(*c)))
                    }
                    _ => Ok(Value::Int(args[0].as_int().unwrap_or(0) + 1)),
                }
            }

            PrimOp::Pred => {
                match &args[0] {
                    Value::Int(n) => Ok(Value::Int(n - 1)),
                    Value::Char(c) => {
                        let n = u32::from(*c);
                        if n == 0 {
                            Err(EvalError::UserError("pred: zero character".into()))
                        } else {
                            Ok(Value::Char(char::from_u32(n - 1).unwrap_or(*c)))
                        }
                    }
                    _ => Ok(Value::Int(args[0].as_int().unwrap_or(0) - 1)),
                }
            }

            PrimOp::ToEnum => {
                // toEnum for Int is identity; for Char it converts
                let n = args[0].as_int().unwrap_or(0);
                #[allow(clippy::cast_sign_loss)]
                Ok(Value::Int(n))
            }

            PrimOp::FromEnum => {
                match &args[0] {
                    Value::Int(n) => Ok(Value::Int(*n)),
                    Value::Char(c) => Ok(Value::Int(i64::from(u32::from(*c)))),
                    Value::Data(d) if d.args.is_empty() => Ok(Value::Int(i64::from(d.con.tag))),
                    _ => Ok(Value::Int(args[0].as_int().unwrap_or(0))),
                }
            }

            // === Integral operations ===
            PrimOp::Gcd => {
                let a = args[0].as_int().unwrap_or(0).abs();
                let b = args[1].as_int().unwrap_or(0).abs();
                fn gcd(mut a: i64, mut b: i64) -> i64 {
                    while b != 0 {
                        let t = b;
                        b = a % b;
                        a = t;
                    }
                    a
                }
                Ok(Value::Int(gcd(a, b)))
            }

            PrimOp::Lcm => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                if a == 0 || b == 0 {
                    Ok(Value::Int(0))
                } else {
                    fn gcd(mut a: i64, mut b: i64) -> i64 {
                        while b != 0 { let t = b; b = a % b; a = t; } a
                    }
                    Ok(Value::Int((a / gcd(a.abs(), b.abs())).abs() * b.abs()))
                }
            }

            PrimOp::Quot => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                if b == 0 { return Err(EvalError::DivisionByZero); }
                // quot truncates towards zero
                Ok(Value::Int(a / b))
            }

            PrimOp::Rem => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                if b == 0 { return Err(EvalError::DivisionByZero); }
                Ok(Value::Int(a % b))
            }

            PrimOp::QuotRem => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                if b == 0 { return Err(EvalError::DivisionByZero); }
                Ok(self.make_pair(Value::Int(a / b), Value::Int(a % b)))
            }

            PrimOp::DivMod => {
                let a = args[0].as_int().unwrap_or(0);
                let b = args[1].as_int().unwrap_or(0);
                if b == 0 { return Err(EvalError::DivisionByZero); }
                // Haskell div/mod truncates towards negative infinity
                let d = a.div_euclid(b);
                let m = a.rem_euclid(b);
                // Adjust: Haskell divMod may differ from Euclidean for negative divisors
                let (d, m) = if b < 0 { (-d - if m != 0 { 1 } else { 0 }, if m != 0 { m + b } else { 0 }) } else { (d, m) };
                Ok(self.make_pair(Value::Int(d), Value::Int(m)))
            }

            PrimOp::Subtract => {
                // subtract :: Num a => a -> a -> a
                // subtract x y = y - x (note: flipped!)
                let x = args[0].as_int().unwrap_or(0);
                let y = args[1].as_int().unwrap_or(0);
                Ok(Value::Int(y.wrapping_sub(x)))
            }

            // === Scan operations ===
            PrimOp::Scanl => {
                // scanl f z xs = [z, f z x0, f (f z x0) x1, ...]
                let f = &args[0];
                let z = args[1].clone();
                let xs = self.force_list(args[2].clone())?;
                let mut result = Vec::with_capacity(xs.len() + 1);
                let mut acc = z;
                result.push(acc.clone());
                for x in xs {
                    acc = self.apply(self.apply(f.clone(), acc)?, x)?;
                    acc = self.force(acc)?;
                    result.push(acc.clone());
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Scanr => {
                // scanr f z xs
                let f = &args[0];
                let z = args[1].clone();
                let xs = self.force_list(args[2].clone())?;
                let mut result = vec![z.clone()];
                let mut acc = z;
                for x in xs.into_iter().rev() {
                    acc = self.apply(self.apply(f.clone(), x)?, acc)?;
                    acc = self.force(acc)?;
                    result.push(acc.clone());
                }
                result.reverse();
                Ok(Value::from_list(result))
            }

            PrimOp::Scanl1 => {
                // scanl1 f xs = scanl f (head xs) (tail xs)
                let f = &args[0];
                let xs = self.force_list(args[1].clone())?;
                if xs.is_empty() {
                    return Ok(Value::from_list(vec![]));
                }
                let mut acc = xs[0].clone();
                let mut result = vec![acc.clone()];
                for x in xs.into_iter().skip(1) {
                    acc = self.apply(self.apply(f.clone(), acc)?, x)?;
                    acc = self.force(acc)?;
                    result.push(acc.clone());
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Scanr1 => {
                // scanr1 f xs
                let f = &args[0];
                let xs = self.force_list(args[1].clone())?;
                if xs.is_empty() {
                    return Ok(Value::from_list(vec![]));
                }
                let mut result = Vec::with_capacity(xs.len());
                let mut acc = xs.last().unwrap().clone();
                result.push(acc.clone());
                for x in xs.into_iter().rev().skip(1) {
                    acc = self.apply(self.apply(f.clone(), x)?, acc)?;
                    acc = self.force(acc)?;
                    result.push(acc.clone());
                }
                result.reverse();
                Ok(Value::from_list(result))
            }

            // === Maximum/Minimum ===
            PrimOp::Maximum => {
                let xs = self.force_list(args[0].clone())?;
                if xs.is_empty() {
                    return Err(EvalError::UserError("maximum: empty list".into()));
                }
                let mut best = xs[0].clone();
                for x in xs.into_iter().skip(1) {
                    if self.value_compare(&x, &best) == std::cmp::Ordering::Greater {
                        best = x;
                    }
                }
                Ok(best)
            }

            PrimOp::Minimum => {
                let xs = self.force_list(args[0].clone())?;
                if xs.is_empty() {
                    return Err(EvalError::UserError("minimum: empty list".into()));
                }
                let mut best = xs[0].clone();
                for x in xs.into_iter().skip(1) {
                    if self.value_compare(&x, &best) == std::cmp::Ordering::Less {
                        best = x;
                    }
                }
                Ok(best)
            }

            // === Zip3 / ZipWith3 / Unzip3 ===
            PrimOp::Zip3 => {
                let xs = self.force_list(args[0].clone())?;
                let ys = self.force_list(args[1].clone())?;
                let zs = self.force_list(args[2].clone())?;
                let result: Vec<Value> = xs.into_iter().zip(ys).zip(zs)
                    .map(|((x, y), z)| self.make_triple(x, y, z))
                    .collect();
                Ok(Value::from_list(result))
            }

            PrimOp::ZipWith3 => {
                let f = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let ys = self.force_list(args[2].clone())?;
                let zs = self.force_list(args[3].clone())?;
                let mut result = Vec::new();
                for ((x, y), z) in xs.into_iter().zip(ys).zip(zs) {
                    let r = self.apply(self.apply(self.apply(f.clone(), x)?, y)?, z)?;
                    result.push(self.force(r)?);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Unzip3 => {
                let xs = self.force_list(args[0].clone())?;
                let mut as_ = Vec::new();
                let mut bs = Vec::new();
                let mut cs = Vec::new();
                for triple in xs {
                    let triple = self.force(triple)?;
                    if let Value::Data(ref d) = triple {
                        if d.args.len() == 3 {
                            as_.push(d.args[0].clone());
                            bs.push(d.args[1].clone());
                            cs.push(d.args[2].clone());
                        }
                    }
                }
                Ok(self.make_triple(
                    Value::from_list(as_),
                    Value::from_list(bs),
                    Value::from_list(cs),
                ))
            }

            // === Show helpers ===
            PrimOp::ShowString => {
                // showString s = (s ++)
                // For simplicity, return the string as-is (it's used as ShowS = String -> String)
                Ok(args[0].clone())
            }

            PrimOp::ShowChar => {
                // showChar c = (c :)
                // Return the char as a single-char string
                match &args[0] {
                    Value::Char(c) => Ok(Value::String(c.to_string().into())),
                    _ => Ok(args[0].clone()),
                }
            }

            PrimOp::ShowParen => {
                // showParen b p = if b then showChar '(' . p . showChar ')' else p
                // Simplified: for now return the second arg as-is
                let b = args[0].as_bool().unwrap_or(false);
                if b {
                    // Wrap in parens
                    let inner = self.value_to_string(&args[1])?;
                    Ok(Value::String(format!("({})", inner).into()))
                } else {
                    Ok(args[1].clone())
                }
            }

            // === IO operations ===
            PrimOp::GetChar => {
                use std::io::Read;
                let mut buf = [0u8; 1];
                std::io::stdin().read_exact(&mut buf)
                    .map_err(|e| EvalError::UserError(format!("getChar failed: {e}")))?;
                Ok(Value::Char(buf[0] as char))
            }

            PrimOp::GetContents => {
                use std::io::Read;
                let mut contents = String::new();
                std::io::stdin().read_to_string(&mut contents)
                    .map_err(|e| EvalError::UserError(format!("getContents failed: {e}")))?;
                Ok(Value::String(contents.into()))
            }

            PrimOp::ReadFile => {
                let path = self.value_to_string(&args[0])?;
                let contents = std::fs::read_to_string(&path)
                    .map_err(|e| EvalError::UserError(format!("{}: {e}", path)))?;
                Ok(Value::String(contents.into()))
            }

            PrimOp::WriteFile => {
                let path = self.value_to_string(&args[0])?;
                let contents = self.value_to_string(&args[1])?;
                std::fs::write(&path, &contents)
                    .map_err(|e| EvalError::UserError(format!("{}: {e}", path)))?;
                Ok(Value::unit())
            }

            PrimOp::AppendFile => {
                use std::io::Write;
                let path = self.value_to_string(&args[0])?;
                let contents = self.value_to_string(&args[1])?;
                let mut file = std::fs::OpenOptions::new().append(true).create(true).open(&path)
                    .map_err(|e| EvalError::UserError(format!("{}: {e}", path)))?;
                file.write_all(contents.as_bytes())
                    .map_err(|e| EvalError::UserError(format!("{}: {e}", path)))?;
                Ok(Value::unit())
            }

            PrimOp::Interact => {
                use std::io::Read;
                let f = &args[0];
                let mut input = String::new();
                std::io::stdin().read_to_string(&mut input)
                    .map_err(|e| EvalError::UserError(format!("interact failed: {e}")))?;
                let result = self.apply(f.clone(), Value::String(input.into()))?;
                let output = self.value_to_string(&self.force(result)?)?;
                println!("{output}");
                self.io_output.borrow_mut().push_str(&output);
                Ok(Value::unit())
            }

            // === Misc Prelude ===
            PrimOp::Otherwise => Ok(Value::bool(true)),

            PrimOp::Until => {
                // until p f x: apply f repeatedly until p holds
                let p = &args[0];
                let f = &args[1];
                let mut x = args[2].clone();
                for _ in 0..10000 {
                    let test = self.apply(p.clone(), x.clone())?;
                    let test = self.force(test)?;
                    if test.as_bool().unwrap_or(false) {
                        return Ok(x);
                    }
                    x = self.apply(f.clone(), x)?;
                    x = self.force(x)?;
                }
                Err(EvalError::UserError("until: iteration limit exceeded".into()))
            }

            PrimOp::AsTypeOf => {
                // asTypeOf x _ = x (type-level only, identity at runtime)
                Ok(args[0].clone())
            }

            PrimOp::RealToFrac => {
                // Identity conversion for now
                match &args[0] {
                    Value::Int(n) => Ok(Value::Double(*n as f64)),
                    Value::Double(_) => Ok(args[0].clone()),
                    Value::Float(f) => Ok(Value::Double(f64::from(*f))),
                    _ => Ok(args[0].clone()),
                }
            }

            // === Data.List operations ===
            PrimOp::Sort => {
                let mut xs = self.force_list(args[0].clone())?;
                xs.sort_by(|a, b| self.value_compare(a, b));
                Ok(Value::from_list(xs))
            }

            PrimOp::SortBy => {
                let cmp = &args[0];
                let mut xs = self.force_list(args[1].clone())?;
                // Use a stable sort with the comparison function
                // cmp returns Ordering (LT/EQ/GT data constructors)
                let cmp_clone = cmp.clone();
                let mut err: Option<EvalError> = None;
                xs.sort_by(|a, b| {
                    if err.is_some() { return std::cmp::Ordering::Equal; }
                    match self.apply(cmp_clone.clone(), a.clone())
                        .and_then(|f| self.apply(f, b.clone()))
                        .and_then(|v| self.force(v))
                    {
                        Ok(v) => self.ordering_value_to_cmp(&v),
                        Err(e) => { err = Some(e); std::cmp::Ordering::Equal }
                    }
                });
                if let Some(e) = err { return Err(e); }
                Ok(Value::from_list(xs))
            }

            PrimOp::SortOn => {
                let f = &args[0];
                let xs = self.force_list(args[1].clone())?;
                // Schwartzian transform: compute key once
                let mut keyed: Vec<(Value, Value)> = Vec::with_capacity(xs.len());
                for x in xs {
                    let key = self.apply(f.clone(), x.clone())?;
                    let key = self.force(key)?;
                    keyed.push((key, x));
                }
                keyed.sort_by(|(ka, _), (kb, _)| self.value_compare(ka, kb));
                let result: Vec<Value> = keyed.into_iter().map(|(_, v)| v).collect();
                Ok(Value::from_list(result))
            }

            PrimOp::Nub => {
                let xs = self.force_list(args[0].clone())?;
                let mut seen = Vec::new();
                let mut result = Vec::new();
                for x in xs {
                    if !seen.iter().any(|s| self.values_equal(s, &x)) {
                        seen.push(x.clone());
                        result.push(x);
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::NubBy => {
                let eq = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut seen: Vec<Value> = Vec::new();
                let mut result = Vec::new();
                for x in xs {
                    let mut found = false;
                    for s in &seen {
                        let r = self.apply(self.apply(eq.clone(), s.clone())?, x.clone())?;
                        let r = self.force(r)?;
                        if r.as_bool().unwrap_or(false) { found = true; break; }
                    }
                    if !found {
                        seen.push(x.clone());
                        result.push(x);
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Group => {
                let xs = self.force_list(args[0].clone())?;
                let mut result: Vec<Value> = Vec::new();
                let mut i = 0;
                while i < xs.len() {
                    let mut group = vec![xs[i].clone()];
                    let mut j = i + 1;
                    while j < xs.len() && self.values_equal(&xs[i], &xs[j]) {
                        group.push(xs[j].clone());
                        j += 1;
                    }
                    result.push(Value::from_list(group));
                    i = j;
                }
                Ok(Value::from_list(result))
            }

            PrimOp::GroupBy => {
                let eq = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut result: Vec<Value> = Vec::new();
                let mut i = 0;
                while i < xs.len() {
                    let mut group = vec![xs[i].clone()];
                    let mut j = i + 1;
                    while j < xs.len() {
                        let r = self.apply(self.apply(eq.clone(), xs[i].clone())?, xs[j].clone())?;
                        let r = self.force(r)?;
                        if r.as_bool().unwrap_or(false) {
                            group.push(xs[j].clone());
                            j += 1;
                        } else { break; }
                    }
                    result.push(Value::from_list(group));
                    i = j;
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Intersperse => {
                let sep = args[0].clone();
                let xs = self.force_list(args[1].clone())?;
                if xs.len() <= 1 { return Ok(Value::from_list(xs)); }
                let mut result = Vec::with_capacity(xs.len() * 2 - 1);
                for (i, x) in xs.into_iter().enumerate() {
                    if i > 0 { result.push(sep.clone()); }
                    result.push(x);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Intercalate => {
                let sep = self.force_list(args[0].clone())?;
                let xss = self.force_list(args[1].clone())?;
                let mut result = Vec::new();
                for (i, xs_val) in xss.into_iter().enumerate() {
                    if i > 0 { result.extend(sep.clone()); }
                    let xs = self.force_list(xs_val)?;
                    result.extend(xs);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Transpose => {
                let xss = self.force_list(args[0].clone())?;
                let mut rows: Vec<Vec<Value>> = Vec::new();
                for xs_val in xss {
                    let xs = self.force_list(xs_val)?;
                    rows.push(xs);
                }
                if rows.is_empty() { return Ok(Value::from_list(vec![])); }
                let max_len = rows.iter().map(Vec::len).max().unwrap_or(0);
                let mut result = Vec::new();
                for col in 0..max_len {
                    let mut column = Vec::new();
                    for row in &rows {
                        if col < row.len() { column.push(row[col].clone()); }
                    }
                    result.push(Value::from_list(column));
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Subsequences => {
                let xs = self.force_list(args[0].clone())?;
                let n = xs.len();
                let mut result = Vec::with_capacity(1 << n.min(20));
                // Limit to prevent memory explosion
                let limit = n.min(20);
                for mask in 0..(1u64 << limit) {
                    let mut sub = Vec::new();
                    for (i, x) in xs.iter().enumerate().take(limit) {
                        if mask & (1 << i) != 0 { sub.push(x.clone()); }
                    }
                    result.push(Value::from_list(sub));
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Permutations => {
                let xs = self.force_list(args[0].clone())?;
                if xs.len() > 8 {
                    return Err(EvalError::UserError("permutations: list too long (max 8)".into()));
                }
                let mut perms = Vec::new();
                let mut indices: Vec<usize> = (0..xs.len()).collect();
                loop {
                    let perm: Vec<Value> = indices.iter().map(|&i| xs[i].clone()).collect();
                    perms.push(Value::from_list(perm));
                    // Next permutation
                    if !next_permutation(&mut indices) { break; }
                }
                Ok(Value::from_list(perms))
            }

            PrimOp::Partition => {
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut yes = Vec::new();
                let mut no = Vec::new();
                for x in xs {
                    let r = self.apply(p.clone(), x.clone())?;
                    let r = self.force(r)?;
                    if r.as_bool().unwrap_or(false) { yes.push(x); } else { no.push(x); }
                }
                Ok(self.make_pair(Value::from_list(yes), Value::from_list(no)))
            }

            PrimOp::Find => {
                let p = &args[0];
                let xs = self.force_list(args[1].clone())?;
                for x in xs {
                    let r = self.apply(p.clone(), x.clone())?;
                    let r = self.force(r)?;
                    if r.as_bool().unwrap_or(false) {
                        return Ok(self.make_just(x));
                    }
                }
                Ok(self.make_nothing())
            }

            PrimOp::StripPrefix => {
                let prefix = self.force_list(args[0].clone())?;
                let xs = self.force_list(args[1].clone())?;
                if xs.len() < prefix.len() { return Ok(self.make_nothing()); }
                for (p, x) in prefix.iter().zip(xs.iter()) {
                    if !self.values_equal(p, x) { return Ok(self.make_nothing()); }
                }
                let rest: Vec<Value> = xs.into_iter().skip(prefix.len()).collect();
                Ok(self.make_just(Value::from_list(rest)))
            }

            PrimOp::IsPrefixOf => {
                let prefix = self.force_list(args[0].clone())?;
                let xs = self.force_list(args[1].clone())?;
                if prefix.len() > xs.len() { return Ok(Value::bool(false)); }
                for (p, x) in prefix.iter().zip(xs.iter()) {
                    if !self.values_equal(p, x) { return Ok(Value::bool(false)); }
                }
                Ok(Value::bool(true))
            }

            PrimOp::IsSuffixOf => {
                let suffix = self.force_list(args[0].clone())?;
                let xs = self.force_list(args[1].clone())?;
                if suffix.len() > xs.len() { return Ok(Value::bool(false)); }
                let offset = xs.len() - suffix.len();
                for (i, s) in suffix.iter().enumerate() {
                    if !self.values_equal(s, &xs[offset + i]) { return Ok(Value::bool(false)); }
                }
                Ok(Value::bool(true))
            }

            PrimOp::IsInfixOf => {
                let needle = self.force_list(args[0].clone())?;
                let haystack = self.force_list(args[1].clone())?;
                if needle.is_empty() { return Ok(Value::bool(true)); }
                if needle.len() > haystack.len() { return Ok(Value::bool(false)); }
                for i in 0..=(haystack.len() - needle.len()) {
                    let mut found = true;
                    for (j, n) in needle.iter().enumerate() {
                        if !self.values_equal(n, &haystack[i + j]) { found = false; break; }
                    }
                    if found { return Ok(Value::bool(true)); }
                }
                Ok(Value::bool(false))
            }

            PrimOp::Delete => {
                let x = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut result = Vec::new();
                let mut deleted = false;
                for item in xs {
                    if !deleted && self.values_equal(x, &item) {
                        deleted = true;
                    } else {
                        result.push(item);
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::DeleteBy => {
                let eq = &args[0];
                let x = &args[1];
                let xs = self.force_list(args[2].clone())?;
                let mut result = Vec::new();
                let mut deleted = false;
                for item in xs {
                    if !deleted {
                        let r = self.apply(self.apply(eq.clone(), x.clone())?, item.clone())?;
                        let r = self.force(r)?;
                        if r.as_bool().unwrap_or(false) { deleted = true; continue; }
                    }
                    result.push(item);
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Union => {
                let xs = self.force_list(args[0].clone())?;
                let ys = self.force_list(args[1].clone())?;
                let mut result = xs.clone();
                for y in ys {
                    if !result.iter().any(|x| self.values_equal(x, &y)) {
                        result.push(y);
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::UnionBy => {
                let eq = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let ys = self.force_list(args[2].clone())?;
                let mut result = xs.clone();
                for y in ys {
                    let mut found = false;
                    for x in &result {
                        let r = self.apply(self.apply(eq.clone(), x.clone())?, y.clone())?;
                        let r = self.force(r)?;
                        if r.as_bool().unwrap_or(false) { found = true; break; }
                    }
                    if !found { result.push(y); }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Intersect => {
                let xs = self.force_list(args[0].clone())?;
                let ys = self.force_list(args[1].clone())?;
                let result: Vec<Value> = xs.into_iter()
                    .filter(|x| ys.iter().any(|y| self.values_equal(x, y)))
                    .collect();
                Ok(Value::from_list(result))
            }

            PrimOp::IntersectBy => {
                let eq = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let ys = self.force_list(args[2].clone())?;
                let mut result = Vec::new();
                for x in xs {
                    let mut found = false;
                    for y in &ys {
                        let r = self.apply(self.apply(eq.clone(), x.clone())?, y.clone())?;
                        let r = self.force(r)?;
                        if r.as_bool().unwrap_or(false) { found = true; break; }
                    }
                    if found { result.push(x); }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::ListDiff => {
                let xs = self.force_list(args[0].clone())?;
                let ys = self.force_list(args[1].clone())?;
                let mut result = xs;
                for y in &ys {
                    if let Some(pos) = result.iter().position(|x| self.values_equal(x, y)) {
                        result.remove(pos);
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Tails => {
                let xs = self.force_list(args[0].clone())?;
                let mut result = Vec::with_capacity(xs.len() + 1);
                for i in 0..=xs.len() {
                    result.push(Value::from_list(xs[i..].to_vec()));
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Inits => {
                let xs = self.force_list(args[0].clone())?;
                let mut result = Vec::with_capacity(xs.len() + 1);
                for i in 0..=xs.len() {
                    result.push(Value::from_list(xs[..i].to_vec()));
                }
                Ok(Value::from_list(result))
            }

            PrimOp::MapAccumL => {
                // mapAccumL f acc xs = (acc', ys)
                let f = &args[0];
                let mut acc = args[1].clone();
                let xs = self.force_list(args[2].clone())?;
                let mut ys = Vec::with_capacity(xs.len());
                for x in xs {
                    let r = self.apply(self.apply(f.clone(), acc)?, x)?;
                    let r = self.force(r)?;
                    if let Value::Data(ref d) = r {
                        if d.con.name.as_str() == "(,)" && d.args.len() == 2 {
                            acc = d.args[0].clone();
                            ys.push(d.args[1].clone());
                            continue;
                        }
                    }
                    return Err(EvalError::TypeError {
                        expected: "pair (acc, y)".into(),
                        got: format!("{r:?}"),
                    });
                }
                Ok(self.make_pair(acc, Value::from_list(ys)))
            }

            PrimOp::MapAccumR => {
                let f = &args[0];
                let mut acc = args[1].clone();
                let xs = self.force_list(args[2].clone())?;
                let mut ys = Vec::with_capacity(xs.len());
                for x in xs.into_iter().rev() {
                    let r = self.apply(self.apply(f.clone(), acc)?, x)?;
                    let r = self.force(r)?;
                    if let Value::Data(ref d) = r {
                        if d.con.name.as_str() == "(,)" && d.args.len() == 2 {
                            acc = d.args[0].clone();
                            ys.push(d.args[1].clone());
                            continue;
                        }
                    }
                    return Err(EvalError::TypeError {
                        expected: "pair (acc, y)".into(),
                        got: format!("{r:?}"),
                    });
                }
                ys.reverse();
                Ok(self.make_pair(acc, Value::from_list(ys)))
            }

            PrimOp::Unfoldr => {
                let f = &args[0];
                let mut b = args[1].clone();
                let mut result = Vec::new();
                let limit = 10000;
                for _ in 0..limit {
                    let r = self.apply(f.clone(), b.clone())?;
                    let r = self.force(r)?;
                    match &r {
                        Value::Data(d) if d.con.name.as_str() == "Nothing" => break,
                        Value::Data(d) if d.con.name.as_str() == "Just" && d.args.len() == 1 => {
                            let pair = self.force(d.args[0].clone())?;
                            if let Value::Data(ref pd) = pair {
                                if pd.con.name.as_str() == "(,)" && pd.args.len() == 2 {
                                    result.push(pd.args[0].clone());
                                    b = pd.args[1].clone();
                                    continue;
                                }
                            }
                            break;
                        }
                        _ => break,
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::GenericLength => {
                let xs = self.force_list(args[0].clone())?;
                Ok(Value::Int(xs.len() as i64))
            }

            PrimOp::GenericTake => {
                let n = args[0].as_int().unwrap_or(0).max(0) as usize;
                let xs = self.force_list(args[1].clone())?;
                Ok(Value::from_list(xs.into_iter().take(n).collect()))
            }

            PrimOp::GenericDrop => {
                let n = args[0].as_int().unwrap_or(0).max(0) as usize;
                let xs = self.force_list(args[1].clone())?;
                Ok(Value::from_list(xs.into_iter().skip(n).collect()))
            }

            // === Data.Char operations ===
            PrimOp::IsAlpha => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_alphabetic())) }
            PrimOp::IsAlphaNum => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_alphanumeric())) }
            PrimOp::IsAscii => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_ascii())) }
            PrimOp::IsControl => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_control())) }
            PrimOp::IsDigit => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_ascii_digit())) }
            PrimOp::IsHexDigit => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_ascii_hexdigit())) }
            PrimOp::IsLetter => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_alphabetic())) }
            PrimOp::IsLower => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_lowercase())) }
            PrimOp::IsNumber => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_numeric())) }
            PrimOp::IsPrint => { let c = self.as_char(&args[0])?; Ok(Value::bool(!c.is_control())) }
            PrimOp::IsPunctuation => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_ascii_punctuation())) }
            PrimOp::IsSpace => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_whitespace())) }
            PrimOp::IsSymbol => {
                let c = self.as_char(&args[0])?;
                Ok(Value::bool(matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '`' | '|' | '~' | '¬' | '±' | '×' | '÷')))
            }
            PrimOp::IsUpper => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_uppercase())) }
            PrimOp::ToLower => { let c = self.as_char(&args[0])?; Ok(Value::Char(c.to_lowercase().next().unwrap_or(c))) }
            PrimOp::ToUpper => { let c = self.as_char(&args[0])?; Ok(Value::Char(c.to_uppercase().next().unwrap_or(c))) }
            PrimOp::ToTitle => { let c = self.as_char(&args[0])?; Ok(Value::Char(c.to_uppercase().next().unwrap_or(c))) }
            PrimOp::DigitToInt => {
                let c = self.as_char(&args[0])?;
                let n = match c {
                    '0'..='9' => (c as i64) - ('0' as i64),
                    'a'..='f' => (c as i64) - ('a' as i64) + 10,
                    'A'..='F' => (c as i64) - ('A' as i64) + 10,
                    _ => return Err(EvalError::UserError(format!("digitToInt: not a digit: {:?}", c))),
                };
                Ok(Value::Int(n))
            }
            PrimOp::IntToDigit => {
                let n = args[0].as_int().unwrap_or(0);
                let c = match n {
                    0..=9 => char::from(b'0' + n as u8),
                    10..=15 => char::from(b'a' + (n - 10) as u8),
                    _ => return Err(EvalError::UserError(format!("intToDigit: out of range: {n}"))),
                };
                Ok(Value::Char(c))
            }
            PrimOp::IsLatin1 => { let c = self.as_char(&args[0])?; Ok(Value::bool((c as u32) <= 255)) }
            PrimOp::IsAsciiLower => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_ascii_lowercase())) }
            PrimOp::IsAsciiUpper => { let c = self.as_char(&args[0])?; Ok(Value::bool(c.is_ascii_uppercase())) }

            // === Data.Function operations ===
            PrimOp::On => {
                // on f g x y = f (g x) (g y)
                let f = &args[0];
                let g = &args[1];
                let x = args[2].clone();
                let y = args[3].clone();
                let gx = self.apply(g.clone(), x)?;
                let gy = self.apply(g.clone(), y)?;
                self.apply(self.apply(f.clone(), gx)?, gy)
            }

            PrimOp::Fix => {
                // fix f = let x = f x in x
                // We implement by repeatedly applying f, with a depth limit
                let f = &args[0];
                self.apply(f.clone(), Value::PrimOp(PrimOp::Fix))
            }

            PrimOp::Amp => {
                // (&) x f = f x (flip ($))
                let x = args[0].clone();
                let f = &args[1];
                self.apply(f.clone(), x)
            }

            // === Data.Maybe additional ===
            PrimOp::ListToMaybe => {
                let xs = self.force_list(args[0].clone())?;
                if xs.is_empty() {
                    Ok(self.make_nothing())
                } else {
                    Ok(self.make_just(xs.into_iter().next().unwrap()))
                }
            }

            PrimOp::MaybeToList => {
                let m = self.force(args[0].clone())?;
                match &m {
                    Value::Data(d) if d.con.name.as_str() == "Nothing" => Ok(Value::from_list(vec![])),
                    Value::Data(d) if d.con.name.as_str() == "Just" && d.args.len() == 1 => {
                        Ok(Value::from_list(vec![d.args[0].clone()]))
                    }
                    _ => Ok(Value::from_list(vec![])),
                }
            }

            PrimOp::CatMaybes => {
                let xs = self.force_list(args[0].clone())?;
                let mut result = Vec::new();
                for x in xs {
                    let x = self.force(x)?;
                    if let Value::Data(ref d) = x {
                        if d.con.name.as_str() == "Just" && d.args.len() == 1 {
                            result.push(d.args[0].clone());
                        }
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::MapMaybe => {
                let f = &args[0];
                let xs = self.force_list(args[1].clone())?;
                let mut result = Vec::new();
                for x in xs {
                    let r = self.apply(f.clone(), x)?;
                    let r = self.force(r)?;
                    if let Value::Data(ref d) = r {
                        if d.con.name.as_str() == "Just" && d.args.len() == 1 {
                            result.push(d.args[0].clone());
                        }
                    }
                }
                Ok(Value::from_list(result))
            }

            // === Data.Either additional ===
            PrimOp::IsLeft => {
                let e = self.force(args[0].clone())?;
                match &e {
                    Value::Data(d) if d.con.name.as_str() == "Left" => Ok(Value::bool(true)),
                    _ => Ok(Value::bool(false)),
                }
            }

            PrimOp::IsRight => {
                let e = self.force(args[0].clone())?;
                match &e {
                    Value::Data(d) if d.con.name.as_str() == "Right" => Ok(Value::bool(true)),
                    _ => Ok(Value::bool(false)),
                }
            }

            PrimOp::Lefts => {
                let xs = self.force_list(args[0].clone())?;
                let mut result = Vec::new();
                for x in xs {
                    let x = self.force(x)?;
                    if let Value::Data(ref d) = x {
                        if d.con.name.as_str() == "Left" && d.args.len() == 1 {
                            result.push(d.args[0].clone());
                        }
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::Rights => {
                let xs = self.force_list(args[0].clone())?;
                let mut result = Vec::new();
                for x in xs {
                    let x = self.force(x)?;
                    if let Value::Data(ref d) = x {
                        if d.con.name.as_str() == "Right" && d.args.len() == 1 {
                            result.push(d.args[0].clone());
                        }
                    }
                }
                Ok(Value::from_list(result))
            }

            PrimOp::PartitionEithers => {
                let xs = self.force_list(args[0].clone())?;
                let mut lefts = Vec::new();
                let mut rights = Vec::new();
                for x in xs {
                    let x = self.force(x)?;
                    if let Value::Data(ref d) = x {
                        if d.con.name.as_str() == "Left" && d.args.len() == 1 {
                            lefts.push(d.args[0].clone());
                        } else if d.con.name.as_str() == "Right" && d.args.len() == 1 {
                            rights.push(d.args[0].clone());
                        }
                    }
                }
                Ok(self.make_pair(Value::from_list(lefts), Value::from_list(rights)))
            }

            // === Math functions ===
            PrimOp::Sqrt => { let a = args[0].as_double().unwrap_or(0.0); Ok(Value::Double(a.sqrt())) }
            PrimOp::Exp => { let a = args[0].as_double().unwrap_or(0.0); Ok(Value::Double(a.exp())) }
            PrimOp::Log => { let a = args[0].as_double().unwrap_or(0.0); Ok(Value::Double(a.ln())) }
            PrimOp::Sin => { let a = args[0].as_double().unwrap_or(0.0); Ok(Value::Double(a.sin())) }
            PrimOp::Cos => { let a = args[0].as_double().unwrap_or(0.0); Ok(Value::Double(a.cos())) }
            PrimOp::Tan => { let a = args[0].as_double().unwrap_or(0.0); Ok(Value::Double(a.tan())) }
            PrimOp::Power => {
                let base = args[0].as_int().unwrap_or(0);
                let exp = args[1].as_int().unwrap_or(0);
                if exp < 0 { return Err(EvalError::UserError("(^): negative exponent".into())); }
                Ok(Value::Int(base.wrapping_pow(exp as u32)))
            }
            PrimOp::Truncate => {
                let a = args[0].as_double().unwrap_or(0.0);
                Ok(Value::Int(a.trunc() as i64))
            }
            PrimOp::Round => {
                let a = args[0].as_double().unwrap_or(0.0);
                Ok(Value::Int(a.round() as i64))
            }
            PrimOp::Ceiling => {
                let a = args[0].as_double().unwrap_or(0.0);
                Ok(Value::Int(a.ceil() as i64))
            }
            PrimOp::Floor => {
                let a = args[0].as_double().unwrap_or(0.0);
                Ok(Value::Int(a.floor() as i64))
            }

            // === Tuple ===
            PrimOp::Fst => {
                let pair = self.force(args[0].clone())?;
                if let Value::Data(ref d) = pair {
                    if d.args.len() >= 2 {
                        return Ok(d.args[0].clone());
                    }
                }
                Err(EvalError::TypeError { expected: "pair".into(), got: format!("{pair:?}") })
            }

            PrimOp::Snd => {
                let pair = self.force(args[0].clone())?;
                if let Value::Data(ref d) = pair {
                    if d.args.len() >= 2 {
                        return Ok(d.args[1].clone());
                    }
                }
                Err(EvalError::TypeError { expected: "pair".into(), got: format!("{pair:?}") })
            }

            PrimOp::MonadBind => {
                // Polymorphic (>>=): dispatch based on first argument type
                // If it's a list, use list bind (concatMap)
                // Otherwise, use IO bind (apply continuation to result)
                let first_arg = self.force(args[0].clone())?;

                if self.is_list_value(&first_arg) {
                    // List monad: xs >>= f = concatMap f xs
                    self.apply_primop(PrimOp::ListBind, args)
                } else {
                    // IO monad (or other): just apply f to the value
                    self.apply_primop(PrimOp::IoBind, args)
                }
            }

            PrimOp::MonadThen => {
                // Polymorphic (>>): dispatch based on first argument type
                let first_arg = self.force(args[0].clone())?;

                if self.is_list_value(&first_arg) {
                    // List monad: xs >> ys = xs >>= \_ -> ys
                    self.apply_primop(PrimOp::ListThen, args)
                } else {
                    // IO monad: just return second arg (first already executed)
                    self.apply_primop(PrimOp::IoThen, args)
                }
            }

            // ========================================================
            // Data.Map PrimOps
            // ========================================================
            PrimOp::MapEmpty => {
                Ok(Value::Map(Arc::new(BTreeMap::new())))
            }
            PrimOp::MapSingleton => {
                let k = self.force(args[0].clone())?;
                let v = self.force(args[1].clone())?;
                let mut m = BTreeMap::new();
                m.insert(OrdValue(k), v);
                Ok(Value::Map(Arc::new(m)))
            }
            PrimOp::MapNull => {
                let m = self.force(args[0].clone())?;
                match &m {
                    Value::Map(map) => Ok(Value::bool(map.is_empty())),
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapSize => {
                let m = self.force(args[0].clone())?;
                match &m {
                    Value::Map(map) => Ok(Value::Int(map.len() as i64)),
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapMember => {
                let k = self.force(args[0].clone())?;
                let m = self.force(args[1].clone())?;
                match &m {
                    Value::Map(map) => Ok(Value::bool(map.contains_key(&OrdValue(k)))),
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapNotMember => {
                let k = self.force(args[0].clone())?;
                let m = self.force(args[1].clone())?;
                match &m {
                    Value::Map(map) => Ok(Value::bool(!map.contains_key(&OrdValue(k)))),
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapLookup => {
                let k = self.force(args[0].clone())?;
                let m = self.force(args[1].clone())?;
                match &m {
                    Value::Map(map) => match map.get(&OrdValue(k)) {
                        Some(v) => Ok(self.make_just(v.clone())),
                        None => Ok(self.make_nothing()),
                    },
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapFindWithDefault => {
                let def = self.force(args[0].clone())?;
                let k = self.force(args[1].clone())?;
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Map(map) => Ok(map.get(&OrdValue(k)).cloned().unwrap_or(def)),
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapIndex => {
                let m = self.force(args[0].clone())?;
                let k = self.force(args[1].clone())?;
                match &m {
                    Value::Map(map) => match map.get(&OrdValue(k)) {
                        Some(v) => Ok(v.clone()),
                        None => Err(EvalError::UserError("Map.!: key not found".into())),
                    },
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapInsert => {
                let k = self.force(args[0].clone())?;
                let v = self.force(args[1].clone())?;
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut new_map = (**map).clone();
                        new_map.insert(OrdValue(k), v);
                        Ok(Value::Map(Arc::new(new_map)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapInsertWith => {
                let f = self.force(args[0].clone())?;
                let k = self.force(args[1].clone())?;
                let v = self.force(args[2].clone())?;
                let m = self.force(args[3].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut new_map = (**map).clone();
                        let ok = OrdValue(k);
                        if let Some(old_v) = new_map.get(&ok) {
                            let tmp = self.apply(f.clone(), v)?;
                            let new_v = self.apply(tmp, old_v.clone())?;
                            new_map.insert(ok, new_v);
                        } else {
                            new_map.insert(ok, v);
                        }
                        Ok(Value::Map(Arc::new(new_map)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapDelete => {
                let k = self.force(args[0].clone())?;
                let m = self.force(args[1].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut new_map = (**map).clone();
                        new_map.remove(&OrdValue(k));
                        Ok(Value::Map(Arc::new(new_map)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapAdjust => {
                let f = self.force(args[0].clone())?;
                let k = self.force(args[1].clone())?;
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut new_map = (**map).clone();
                        let ok = OrdValue(k);
                        if let Some(v) = new_map.get(&ok).cloned() {
                            let new_v = self.apply(f, v)?;
                            new_map.insert(ok, new_v);
                        }
                        Ok(Value::Map(Arc::new(new_map)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapUpdate => {
                let f = self.force(args[0].clone())?;
                let k = self.force(args[1].clone())?;
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut new_map = (**map).clone();
                        let ok = OrdValue(k);
                        if let Some(v) = new_map.get(&ok).cloned() {
                            let result = self.apply(f, v)?;
                            let result = self.force(result)?;
                            if let Value::Data(ref d) = result {
                                if d.con.name.as_str() == "Just" && !d.args.is_empty() {
                                    new_map.insert(ok, d.args[0].clone());
                                } else if d.con.name.as_str() == "Nothing" {
                                    new_map.remove(&ok);
                                }
                            }
                        }
                        Ok(Value::Map(Arc::new(new_map)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapAlter => {
                let f = self.force(args[0].clone())?;
                let k = self.force(args[1].clone())?;
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut new_map = (**map).clone();
                        let ok = OrdValue(k);
                        let input = match map.get(&ok) {
                            Some(v) => self.make_just(v.clone()),
                            None => self.make_nothing(),
                        };
                        let result = self.apply(f, input)?;
                        let result = self.force(result)?;
                        if let Value::Data(ref d) = result {
                            if d.con.name.as_str() == "Just" && !d.args.is_empty() {
                                new_map.insert(ok, d.args[0].clone());
                            } else if d.con.name.as_str() == "Nothing" {
                                new_map.remove(&ok);
                            }
                        }
                        Ok(Value::Map(Arc::new(new_map)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapUnion => {
                let m1 = self.force(args[0].clone())?;
                let m2 = self.force(args[1].clone())?;
                match (&m1, &m2) {
                    (Value::Map(a), Value::Map(b)) => {
                        let mut result = (**a).clone();
                        for (k, v) in b.iter() {
                            result.entry(k.clone()).or_insert_with(|| v.clone());
                        }
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m1:?}") }),
                }
            }
            PrimOp::MapUnionWith => {
                let f = self.force(args[0].clone())?;
                let m1 = self.force(args[1].clone())?;
                let m2 = self.force(args[2].clone())?;
                match (&m1, &m2) {
                    (Value::Map(a), Value::Map(b)) => {
                        let mut result = (**a).clone();
                        for (k, v) in b.iter() {
                            if let Some(old) = result.get(k).cloned() {
                                let tmp = self.apply(f.clone(), old)?;
                                let new_v = self.apply(tmp, v.clone())?;
                                result.insert(k.clone(), new_v);
                            } else {
                                result.insert(k.clone(), v.clone());
                            }
                        }
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m1:?}") }),
                }
            }
            PrimOp::MapUnionWithKey => {
                let f = self.force(args[0].clone())?;
                let m1 = self.force(args[1].clone())?;
                let m2 = self.force(args[2].clone())?;
                match (&m1, &m2) {
                    (Value::Map(a), Value::Map(b)) => {
                        let mut result = (**a).clone();
                        for (k, v) in b.iter() {
                            if let Some(old) = result.get(k).cloned() {
                                let tmp1 = self.apply(f.clone(), k.inner().clone())?;
                                let tmp2 = self.apply(tmp1, old)?;
                                let new_v = self.apply(tmp2, v.clone())?;
                                result.insert(k.clone(), new_v);
                            } else {
                                result.insert(k.clone(), v.clone());
                            }
                        }
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m1:?}") }),
                }
            }
            PrimOp::MapUnions => {
                let list = self.force(args[0].clone())?;
                let maps = self.force_list(list)?;
                let mut result = BTreeMap::new();
                for m in maps {
                    let m = self.force(m)?;
                    if let Value::Map(map) = m {
                        for (k, v) in map.iter() {
                            result.entry(k.clone()).or_insert_with(|| v.clone());
                        }
                    }
                }
                Ok(Value::Map(Arc::new(result)))
            }
            PrimOp::MapIntersection => {
                let m1 = self.force(args[0].clone())?;
                let m2 = self.force(args[1].clone())?;
                match (&m1, &m2) {
                    (Value::Map(a), Value::Map(b)) => {
                        let result: BTreeMap<OrdValue, Value> = a.iter()
                            .filter(|(k, _)| b.contains_key(k))
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m1:?}") }),
                }
            }
            PrimOp::MapIntersectionWith => {
                let f = self.force(args[0].clone())?;
                let m1 = self.force(args[1].clone())?;
                let m2 = self.force(args[2].clone())?;
                match (&m1, &m2) {
                    (Value::Map(a), Value::Map(b)) => {
                        let mut result = BTreeMap::new();
                        for (k, va) in a.iter() {
                            if let Some(vb) = b.get(k) {
                                let tmp = self.apply(f.clone(), va.clone())?;
                                let v = self.apply(tmp, vb.clone())?;
                                result.insert(k.clone(), v);
                            }
                        }
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m1:?}") }),
                }
            }
            PrimOp::MapDifference => {
                let m1 = self.force(args[0].clone())?;
                let m2 = self.force(args[1].clone())?;
                match (&m1, &m2) {
                    (Value::Map(a), Value::Map(b)) => {
                        let result: BTreeMap<OrdValue, Value> = a.iter()
                            .filter(|(k, _)| !b.contains_key(k))
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m1:?}") }),
                }
            }
            PrimOp::MapDifferenceWith => {
                let f = self.force(args[0].clone())?;
                let m1 = self.force(args[1].clone())?;
                let m2 = self.force(args[2].clone())?;
                match (&m1, &m2) {
                    (Value::Map(a), Value::Map(b)) => {
                        let mut result = BTreeMap::new();
                        for (k, va) in a.iter() {
                            if let Some(vb) = b.get(k) {
                                let tmp = self.apply(f.clone(), va.clone())?;
                                let r = self.apply(tmp, vb.clone())?;
                                let r = self.force(r)?;
                                if let Value::Data(ref d) = r {
                                    if d.con.name.as_str() == "Just" && !d.args.is_empty() {
                                        result.insert(k.clone(), d.args[0].clone());
                                    }
                                }
                            } else {
                                result.insert(k.clone(), va.clone());
                            }
                        }
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m1:?}") }),
                }
            }
            PrimOp::MapMap => {
                let f = self.force(args[0].clone())?;
                let m = self.force(args[1].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut result = BTreeMap::new();
                        for (k, v) in map.iter() {
                            let new_v = self.apply(f.clone(), v.clone())?;
                            result.insert(k.clone(), new_v);
                        }
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapMapWithKey => {
                let f = self.force(args[0].clone())?;
                let _k = args[1].clone();
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut result = BTreeMap::new();
                        for (k, v) in map.iter() {
                            let tmp = self.apply(f.clone(), k.inner().clone())?;
                            let new_v = self.apply(tmp, v.clone())?;
                            result.insert(k.clone(), new_v);
                        }
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapMapKeys => {
                let f = self.force(args[0].clone())?;
                let m = self.force(args[1].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut result = BTreeMap::new();
                        for (k, v) in map.iter() {
                            let new_k = self.apply(f.clone(), k.inner().clone())?;
                            result.insert(OrdValue(new_k), v.clone());
                        }
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapFilter => {
                let f = self.force(args[0].clone())?;
                let m = self.force(args[1].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut result = BTreeMap::new();
                        for (k, v) in map.iter() {
                            let keep = self.apply(f.clone(), v.clone())?;
                            let keep = self.force(keep)?;
                            if keep.as_bool() == Some(true) {
                                result.insert(k.clone(), v.clone());
                            }
                        }
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapFilterWithKey => {
                let f = self.force(args[0].clone())?;
                let _k = args[1].clone();
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut result = BTreeMap::new();
                        for (k, v) in map.iter() {
                            let tmp = self.apply(f.clone(), k.inner().clone())?;
                            let keep = self.apply(tmp, v.clone())?;
                            let keep = self.force(keep)?;
                            if keep.as_bool() == Some(true) {
                                result.insert(k.clone(), v.clone());
                            }
                        }
                        Ok(Value::Map(Arc::new(result)))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapFoldr => {
                let f = self.force(args[0].clone())?;
                let z = self.force(args[1].clone())?;
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut acc = z;
                        for (_, v) in map.iter().rev() {
                            let tmp = self.apply(f.clone(), v.clone())?;
                            acc = self.apply(tmp, acc)?;
                        }
                        Ok(acc)
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapFoldl => {
                let f = self.force(args[0].clone())?;
                let z = self.force(args[1].clone())?;
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut acc = z;
                        for (_, v) in map.iter() {
                            let tmp = self.apply(f.clone(), acc)?;
                            acc = self.apply(tmp, v.clone())?;
                        }
                        Ok(acc)
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapFoldrWithKey => {
                let f = self.force(args[0].clone())?;
                let z = self.force(args[1].clone())?;
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut acc = z;
                        for (k, v) in map.iter().rev() {
                            let tmp1 = self.apply(f.clone(), k.inner().clone())?;
                            let tmp2 = self.apply(tmp1, v.clone())?;
                            acc = self.apply(tmp2, acc)?;
                        }
                        Ok(acc)
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapFoldlWithKey => {
                let f = self.force(args[0].clone())?;
                let z = self.force(args[1].clone())?;
                let m = self.force(args[2].clone())?;
                match &m {
                    Value::Map(map) => {
                        let mut acc = z;
                        for (k, v) in map.iter() {
                            let tmp1 = self.apply(f.clone(), acc)?;
                            let tmp2 = self.apply(tmp1, k.inner().clone())?;
                            acc = self.apply(tmp2, v.clone())?;
                        }
                        Ok(acc)
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapKeys => {
                let m = self.force(args[0].clone())?;
                match &m {
                    Value::Map(map) => {
                        let keys: Vec<Value> = map.keys().map(|k| k.inner().clone()).collect();
                        Ok(Value::from_list(keys))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapElems => {
                let m = self.force(args[0].clone())?;
                match &m {
                    Value::Map(map) => {
                        let vals: Vec<Value> = map.values().cloned().collect();
                        Ok(Value::from_list(vals))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapAssocs | PrimOp::MapToList | PrimOp::MapToAscList => {
                let m = self.force(args[0].clone())?;
                match &m {
                    Value::Map(map) => {
                        let pairs: Vec<Value> = map.iter()
                            .map(|(k, v)| self.make_pair(k.inner().clone(), v.clone()))
                            .collect();
                        Ok(Value::from_list(pairs))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapToDescList => {
                let m = self.force(args[0].clone())?;
                match &m {
                    Value::Map(map) => {
                        let pairs: Vec<Value> = map.iter().rev()
                            .map(|(k, v)| self.make_pair(k.inner().clone(), v.clone()))
                            .collect();
                        Ok(Value::from_list(pairs))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m:?}") }),
                }
            }
            PrimOp::MapFromList => {
                let list = self.force(args[0].clone())?;
                let pairs = self.force_list(list)?;
                let mut m = BTreeMap::new();
                for pair in pairs {
                    let pair = self.force(pair)?;
                    if let Value::Data(ref d) = pair {
                        if d.args.len() >= 2 {
                            let k = self.force(d.args[0].clone())?;
                            let v = self.force(d.args[1].clone())?;
                            m.insert(OrdValue(k), v);
                        }
                    }
                }
                Ok(Value::Map(Arc::new(m)))
            }
            PrimOp::MapFromListWith => {
                let f = self.force(args[0].clone())?;
                let list = self.force(args[1].clone())?;
                let pairs = self.force_list(list)?;
                let mut m = BTreeMap::new();
                for pair in pairs {
                    let pair = self.force(pair)?;
                    if let Value::Data(ref d) = pair {
                        if d.args.len() >= 2 {
                            let k = self.force(d.args[0].clone())?;
                            let v = self.force(d.args[1].clone())?;
                            let ok = OrdValue(k);
                            if let Some(old) = m.get(&ok).cloned() {
                                let tmp = self.apply(f.clone(), v)?;
                                let new_v = self.apply(tmp, old)?;
                                m.insert(ok, new_v);
                            } else {
                                m.insert(ok, v);
                            }
                        }
                    }
                }
                Ok(Value::Map(Arc::new(m)))
            }
            PrimOp::MapIsSubmapOf => {
                let m1 = self.force(args[0].clone())?;
                let m2 = self.force(args[1].clone())?;
                match (&m1, &m2) {
                    (Value::Map(a), Value::Map(b)) => {
                        let is_sub = a.iter().all(|(k, v)| b.get(k).map_or(false, |bv| self.values_equal(v, bv)));
                        Ok(Value::bool(is_sub))
                    }
                    _ => Err(EvalError::TypeError { expected: "Map".into(), got: format!("{m1:?}") }),
                }
            }

            // ========================================================
            // Data.Set PrimOps
            // ========================================================
            PrimOp::SetEmpty => Ok(Value::Set(Arc::new(BTreeSet::new()))),
            PrimOp::SetSingleton => {
                let v = self.force(args[0].clone())?;
                let mut s = BTreeSet::new();
                s.insert(OrdValue(v));
                Ok(Value::Set(Arc::new(s)))
            }
            PrimOp::SetNull => {
                let s = self.force(args[0].clone())?;
                match &s { Value::Set(set) => Ok(Value::bool(set.is_empty())), _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetSize => {
                let s = self.force(args[0].clone())?;
                match &s { Value::Set(set) => Ok(Value::Int(set.len() as i64)), _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetMember => {
                let v = self.force(args[0].clone())?;
                let s = self.force(args[1].clone())?;
                match &s { Value::Set(set) => Ok(Value::bool(set.contains(&OrdValue(v)))), _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetNotMember => {
                let v = self.force(args[0].clone())?;
                let s = self.force(args[1].clone())?;
                match &s { Value::Set(set) => Ok(Value::bool(!set.contains(&OrdValue(v)))), _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetInsert => {
                let v = self.force(args[0].clone())?;
                let s = self.force(args[1].clone())?;
                match &s { Value::Set(set) => { let mut ns = (**set).clone(); ns.insert(OrdValue(v)); Ok(Value::Set(Arc::new(ns))) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetDelete => {
                let v = self.force(args[0].clone())?;
                let s = self.force(args[1].clone())?;
                match &s { Value::Set(set) => { let mut ns = (**set).clone(); ns.remove(&OrdValue(v)); Ok(Value::Set(Arc::new(ns))) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetUnion => {
                let s1 = self.force(args[0].clone())?;
                let s2 = self.force(args[1].clone())?;
                match (&s1, &s2) { (Value::Set(a), Value::Set(b)) => { let r: BTreeSet<OrdValue> = a.union(b).cloned().collect(); Ok(Value::Set(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s1:?}") }) }
            }
            PrimOp::SetUnions => {
                let list = self.force(args[0].clone())?;
                let sets = self.force_list(list)?;
                let mut result = BTreeSet::new();
                for s in sets { let s = self.force(s)?; if let Value::Set(set) = s { for v in set.iter() { result.insert(v.clone()); } } }
                Ok(Value::Set(Arc::new(result)))
            }
            PrimOp::SetIntersection => {
                let s1 = self.force(args[0].clone())?;
                let s2 = self.force(args[1].clone())?;
                match (&s1, &s2) { (Value::Set(a), Value::Set(b)) => { let r: BTreeSet<OrdValue> = a.intersection(b).cloned().collect(); Ok(Value::Set(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s1:?}") }) }
            }
            PrimOp::SetDifference => {
                let s1 = self.force(args[0].clone())?;
                let s2 = self.force(args[1].clone())?;
                match (&s1, &s2) { (Value::Set(a), Value::Set(b)) => { let r: BTreeSet<OrdValue> = a.difference(b).cloned().collect(); Ok(Value::Set(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s1:?}") }) }
            }
            PrimOp::SetIsSubsetOf => {
                let s1 = self.force(args[0].clone())?; let s2 = self.force(args[1].clone())?;
                match (&s1, &s2) { (Value::Set(a), Value::Set(b)) => Ok(Value::bool(a.is_subset(b))), _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s1:?}") }) }
            }
            PrimOp::SetIsProperSubsetOf => {
                let s1 = self.force(args[0].clone())?; let s2 = self.force(args[1].clone())?;
                match (&s1, &s2) { (Value::Set(a), Value::Set(b)) => Ok(Value::bool(a.is_subset(b) && a.len() < b.len())), _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s1:?}") }) }
            }
            PrimOp::SetMap => {
                let f = self.force(args[0].clone())?;
                let s = self.force(args[1].clone())?;
                match &s { Value::Set(set) => { let mut r = BTreeSet::new(); for v in set.iter() { let nv = self.apply(f.clone(), v.inner().clone())?; r.insert(OrdValue(nv)); } Ok(Value::Set(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetFilter => {
                let f = self.force(args[0].clone())?;
                let s = self.force(args[1].clone())?;
                match &s { Value::Set(set) => { let mut r = BTreeSet::new(); for v in set.iter() { let keep = self.apply(f.clone(), v.inner().clone())?; let keep = self.force(keep)?; if keep.as_bool() == Some(true) { r.insert(v.clone()); } } Ok(Value::Set(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetPartition => {
                let f = self.force(args[0].clone())?;
                let s = self.force(args[1].clone())?;
                match &s { Value::Set(set) => { let mut yes = BTreeSet::new(); let mut no = BTreeSet::new(); for v in set.iter() { let keep = self.apply(f.clone(), v.inner().clone())?; let keep = self.force(keep)?; if keep.as_bool() == Some(true) { yes.insert(v.clone()); } else { no.insert(v.clone()); } } Ok(self.make_pair(Value::Set(Arc::new(yes)), Value::Set(Arc::new(no)))) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetFoldr => {
                let f = self.force(args[0].clone())?; let z = self.force(args[1].clone())?; let s = self.force(args[2].clone())?;
                match &s { Value::Set(set) => { let mut acc = z; for v in set.iter().rev() { let tmp = self.apply(f.clone(), v.inner().clone())?; acc = self.apply(tmp, acc)?; } Ok(acc) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetFoldl => {
                let f = self.force(args[0].clone())?; let z = self.force(args[1].clone())?; let s = self.force(args[2].clone())?;
                match &s { Value::Set(set) => { let mut acc = z; for v in set.iter() { let tmp = self.apply(f.clone(), acc)?; acc = self.apply(tmp, v.inner().clone())?; } Ok(acc) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetToList | PrimOp::SetToAscList | PrimOp::SetElems => {
                let s = self.force(args[0].clone())?;
                match &s { Value::Set(set) => { let vals: Vec<Value> = set.iter().map(|v| v.inner().clone()).collect(); Ok(Value::from_list(vals)) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetToDescList => {
                let s = self.force(args[0].clone())?;
                match &s { Value::Set(set) => { let vals: Vec<Value> = set.iter().rev().map(|v| v.inner().clone()).collect(); Ok(Value::from_list(vals)) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetFromList => {
                let list = self.force(args[0].clone())?;
                let items = self.force_list(list)?;
                let set: BTreeSet<OrdValue> = items.into_iter().map(OrdValue).collect();
                Ok(Value::Set(Arc::new(set)))
            }
            PrimOp::SetFindMin => {
                let s = self.force(args[0].clone())?;
                match &s { Value::Set(set) => match set.iter().next() { Some(v) => Ok(v.inner().clone()), None => Err(EvalError::UserError("Set.findMin: empty set".into())) }, _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetFindMax => {
                let s = self.force(args[0].clone())?;
                match &s { Value::Set(set) => match set.iter().next_back() { Some(v) => Ok(v.inner().clone()), None => Err(EvalError::UserError("Set.findMax: empty set".into())) }, _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetDeleteMin => {
                let s = self.force(args[0].clone())?;
                match &s { Value::Set(set) => { let mut ns = (**set).clone(); if let Some(min) = set.iter().next().cloned() { ns.remove(&min); } Ok(Value::Set(Arc::new(ns))) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetDeleteMax => {
                let s = self.force(args[0].clone())?;
                match &s { Value::Set(set) => { let mut ns = (**set).clone(); if let Some(max) = set.iter().next_back().cloned() { ns.remove(&max); } Ok(Value::Set(Arc::new(ns))) } _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetLookupMin => {
                let s = self.force(args[0].clone())?;
                match &s { Value::Set(set) => match set.iter().next() { Some(v) => Ok(self.make_just(v.inner().clone())), None => Ok(self.make_nothing()) }, _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }
            PrimOp::SetLookupMax => {
                let s = self.force(args[0].clone())?;
                match &s { Value::Set(set) => match set.iter().next_back() { Some(v) => Ok(self.make_just(v.inner().clone())), None => Ok(self.make_nothing()) }, _ => Err(EvalError::TypeError { expected: "Set".into(), got: format!("{s:?}") }) }
            }

            // ========================================================
            // Data.IntMap PrimOps
            // ========================================================
            PrimOp::IntMapEmpty => Ok(Value::IntMap(Arc::new(BTreeMap::new()))),
            PrimOp::IntMapSingleton => { let k = self.force(args[0].clone())?.as_int().unwrap_or(0); let v = self.force(args[1].clone())?; let mut m = BTreeMap::new(); m.insert(k, v); Ok(Value::IntMap(Arc::new(m))) }
            PrimOp::IntMapNull => { let m = self.force(args[0].clone())?; match &m { Value::IntMap(map) => Ok(Value::bool(map.is_empty())), _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapSize => { let m = self.force(args[0].clone())?; match &m { Value::IntMap(map) => Ok(Value::Int(map.len() as i64)), _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapMember => { let k = self.force(args[0].clone())?.as_int().unwrap_or(0); let m = self.force(args[1].clone())?; match &m { Value::IntMap(map) => Ok(Value::bool(map.contains_key(&k))), _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapLookup => { let k = self.force(args[0].clone())?.as_int().unwrap_or(0); let m = self.force(args[1].clone())?; match &m { Value::IntMap(map) => match map.get(&k) { Some(v) => Ok(self.make_just(v.clone())), None => Ok(self.make_nothing()) }, _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapFindWithDefault => { let def = self.force(args[0].clone())?; let k = self.force(args[1].clone())?.as_int().unwrap_or(0); let m = self.force(args[2].clone())?; match &m { Value::IntMap(map) => Ok(map.get(&k).cloned().unwrap_or(def)), _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapInsert => { let k = self.force(args[0].clone())?.as_int().unwrap_or(0); let v = self.force(args[1].clone())?; let m = self.force(args[2].clone())?; match &m { Value::IntMap(map) => { let mut nm = (**map).clone(); nm.insert(k, v); Ok(Value::IntMap(Arc::new(nm))) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapInsertWith => { let f = self.force(args[0].clone())?; let k = self.force(args[1].clone())?.as_int().unwrap_or(0); let v = self.force(args[2].clone())?; let m = self.force(args[3].clone())?; match &m { Value::IntMap(map) => { let mut nm = (**map).clone(); if let Some(old) = nm.get(&k).cloned() { let tmp = self.apply(f.clone(), v)?; let nv = self.apply(tmp, old)?; nm.insert(k, nv); } else { nm.insert(k, v); } Ok(Value::IntMap(Arc::new(nm))) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapDelete => { let k = self.force(args[0].clone())?.as_int().unwrap_or(0); let m = self.force(args[1].clone())?; match &m { Value::IntMap(map) => { let mut nm = (**map).clone(); nm.remove(&k); Ok(Value::IntMap(Arc::new(nm))) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapAdjust => { let f = self.force(args[0].clone())?; let k = self.force(args[1].clone())?.as_int().unwrap_or(0); let m = self.force(args[2].clone())?; match &m { Value::IntMap(map) => { let mut nm = (**map).clone(); if let Some(v) = nm.get(&k).cloned() { let nv = self.apply(f, v)?; nm.insert(k, nv); } Ok(Value::IntMap(Arc::new(nm))) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapUnion => { let m1 = self.force(args[0].clone())?; let m2 = self.force(args[1].clone())?; match (&m1, &m2) { (Value::IntMap(a), Value::IntMap(b)) => { let mut r = (**a).clone(); for (k, v) in b.iter() { r.entry(*k).or_insert_with(|| v.clone()); } Ok(Value::IntMap(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m1:?}") }) } }
            PrimOp::IntMapUnionWith => { let f = self.force(args[0].clone())?; let m1 = self.force(args[1].clone())?; let m2 = self.force(args[2].clone())?; match (&m1, &m2) { (Value::IntMap(a), Value::IntMap(b)) => { let mut r = (**a).clone(); for (k, v) in b.iter() { if let Some(old) = r.get(k).cloned() { let tmp = self.apply(f.clone(), old)?; let nv = self.apply(tmp, v.clone())?; r.insert(*k, nv); } else { r.insert(*k, v.clone()); } } Ok(Value::IntMap(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m1:?}") }) } }
            PrimOp::IntMapIntersection => { let m1 = self.force(args[0].clone())?; let m2 = self.force(args[1].clone())?; match (&m1, &m2) { (Value::IntMap(a), Value::IntMap(b)) => { let r: BTreeMap<i64, Value> = a.iter().filter(|(k, _)| b.contains_key(k)).map(|(k, v)| (*k, v.clone())).collect(); Ok(Value::IntMap(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m1:?}") }) } }
            PrimOp::IntMapDifference => { let m1 = self.force(args[0].clone())?; let m2 = self.force(args[1].clone())?; match (&m1, &m2) { (Value::IntMap(a), Value::IntMap(b)) => { let r: BTreeMap<i64, Value> = a.iter().filter(|(k, _)| !b.contains_key(k)).map(|(k, v)| (*k, v.clone())).collect(); Ok(Value::IntMap(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m1:?}") }) } }
            PrimOp::IntMapMap => { let f = self.force(args[0].clone())?; let m = self.force(args[1].clone())?; match &m { Value::IntMap(map) => { let mut r = BTreeMap::new(); for (k, v) in map.iter() { let nv = self.apply(f.clone(), v.clone())?; r.insert(*k, nv); } Ok(Value::IntMap(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapMapWithKey => { let f = self.force(args[0].clone())?; let m = self.force(args[1].clone())?; match &m { Value::IntMap(map) => { let mut r = BTreeMap::new(); for (k, v) in map.iter() { let tmp = self.apply(f.clone(), Value::Int(*k))?; let nv = self.apply(tmp, v.clone())?; r.insert(*k, nv); } Ok(Value::IntMap(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapFilter => { let f = self.force(args[0].clone())?; let m = self.force(args[1].clone())?; match &m { Value::IntMap(map) => { let mut r = BTreeMap::new(); for (k, v) in map.iter() { let keep = self.apply(f.clone(), v.clone())?; let keep = self.force(keep)?; if keep.as_bool() == Some(true) { r.insert(*k, v.clone()); } } Ok(Value::IntMap(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapFoldr => { let f = self.force(args[0].clone())?; let z = self.force(args[1].clone())?; let m = self.force(args[2].clone())?; match &m { Value::IntMap(map) => { let mut acc = z; for (_, v) in map.iter().rev() { let tmp = self.apply(f.clone(), v.clone())?; acc = self.apply(tmp, acc)?; } Ok(acc) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapFoldlWithKey => { let f = self.force(args[0].clone())?; let z = self.force(args[1].clone())?; let m = self.force(args[2].clone())?; match &m { Value::IntMap(map) => { let mut acc = z; for (k, v) in map.iter() { let tmp1 = self.apply(f.clone(), acc)?; let tmp2 = self.apply(tmp1, Value::Int(*k))?; acc = self.apply(tmp2, v.clone())?; } Ok(acc) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapKeys => { let m = self.force(args[0].clone())?; match &m { Value::IntMap(map) => { let keys: Vec<Value> = map.keys().map(|k| Value::Int(*k)).collect(); Ok(Value::from_list(keys)) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapElems => { let m = self.force(args[0].clone())?; match &m { Value::IntMap(map) => { let vals: Vec<Value> = map.values().cloned().collect(); Ok(Value::from_list(vals)) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapToList | PrimOp::IntMapToAscList => { let m = self.force(args[0].clone())?; match &m { Value::IntMap(map) => { let pairs: Vec<Value> = map.iter().map(|(k, v)| self.make_pair(Value::Int(*k), v.clone())).collect(); Ok(Value::from_list(pairs)) } _ => Err(EvalError::TypeError { expected: "IntMap".into(), got: format!("{m:?}") }) } }
            PrimOp::IntMapFromList => { let list = self.force(args[0].clone())?; let pairs = self.force_list(list)?; let mut m = BTreeMap::new(); for pair in pairs { let pair = self.force(pair)?; if let Value::Data(ref d) = pair { if d.args.len() >= 2 { let k = self.force(d.args[0].clone())?.as_int().unwrap_or(0); let v = self.force(d.args[1].clone())?; m.insert(k, v); } } } Ok(Value::IntMap(Arc::new(m))) }

            // ========================================================
            // Data.IntSet PrimOps
            // ========================================================
            PrimOp::IntSetEmpty => Ok(Value::IntSet(Arc::new(BTreeSet::new()))),
            PrimOp::IntSetSingleton => { let v = self.force(args[0].clone())?.as_int().unwrap_or(0); let mut s = BTreeSet::new(); s.insert(v); Ok(Value::IntSet(Arc::new(s))) }
            PrimOp::IntSetNull => { let s = self.force(args[0].clone())?; match &s { Value::IntSet(set) => Ok(Value::bool(set.is_empty())), _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s:?}") }) } }
            PrimOp::IntSetSize => { let s = self.force(args[0].clone())?; match &s { Value::IntSet(set) => Ok(Value::Int(set.len() as i64)), _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s:?}") }) } }
            PrimOp::IntSetMember => { let v = self.force(args[0].clone())?.as_int().unwrap_or(0); let s = self.force(args[1].clone())?; match &s { Value::IntSet(set) => Ok(Value::bool(set.contains(&v))), _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s:?}") }) } }
            PrimOp::IntSetInsert => { let v = self.force(args[0].clone())?.as_int().unwrap_or(0); let s = self.force(args[1].clone())?; match &s { Value::IntSet(set) => { let mut ns = (**set).clone(); ns.insert(v); Ok(Value::IntSet(Arc::new(ns))) } _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s:?}") }) } }
            PrimOp::IntSetDelete => { let v = self.force(args[0].clone())?.as_int().unwrap_or(0); let s = self.force(args[1].clone())?; match &s { Value::IntSet(set) => { let mut ns = (**set).clone(); ns.remove(&v); Ok(Value::IntSet(Arc::new(ns))) } _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s:?}") }) } }
            PrimOp::IntSetUnion => { let s1 = self.force(args[0].clone())?; let s2 = self.force(args[1].clone())?; match (&s1, &s2) { (Value::IntSet(a), Value::IntSet(b)) => { let r: BTreeSet<i64> = a.union(b).copied().collect(); Ok(Value::IntSet(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s1:?}") }) } }
            PrimOp::IntSetIntersection => { let s1 = self.force(args[0].clone())?; let s2 = self.force(args[1].clone())?; match (&s1, &s2) { (Value::IntSet(a), Value::IntSet(b)) => { let r: BTreeSet<i64> = a.intersection(b).copied().collect(); Ok(Value::IntSet(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s1:?}") }) } }
            PrimOp::IntSetDifference => { let s1 = self.force(args[0].clone())?; let s2 = self.force(args[1].clone())?; match (&s1, &s2) { (Value::IntSet(a), Value::IntSet(b)) => { let r: BTreeSet<i64> = a.difference(b).copied().collect(); Ok(Value::IntSet(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s1:?}") }) } }
            PrimOp::IntSetIsSubsetOf => { let s1 = self.force(args[0].clone())?; let s2 = self.force(args[1].clone())?; match (&s1, &s2) { (Value::IntSet(a), Value::IntSet(b)) => Ok(Value::bool(a.is_subset(b))), _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s1:?}") }) } }
            PrimOp::IntSetFilter => { let f = self.force(args[0].clone())?; let s = self.force(args[1].clone())?; match &s { Value::IntSet(set) => { let mut r = BTreeSet::new(); for v in set.iter() { let keep = self.apply(f.clone(), Value::Int(*v))?; let keep = self.force(keep)?; if keep.as_bool() == Some(true) { r.insert(*v); } } Ok(Value::IntSet(Arc::new(r))) } _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s:?}") }) } }
            PrimOp::IntSetFoldr => { let f = self.force(args[0].clone())?; let z = self.force(args[1].clone())?; let s = self.force(args[2].clone())?; match &s { Value::IntSet(set) => { let mut acc = z; for v in set.iter().rev() { let tmp = self.apply(f.clone(), Value::Int(*v))?; acc = self.apply(tmp, acc)?; } Ok(acc) } _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s:?}") }) } }
            PrimOp::IntSetToList => { let s = self.force(args[0].clone())?; match &s { Value::IntSet(set) => { let vals: Vec<Value> = set.iter().map(|v| Value::Int(*v)).collect(); Ok(Value::from_list(vals)) } _ => Err(EvalError::TypeError { expected: "IntSet".into(), got: format!("{s:?}") }) } }
            PrimOp::IntSetFromList => { let list = self.force(args[0].clone())?; let items = self.force_list(list)?; let set: BTreeSet<i64> = items.into_iter().filter_map(|v| v.as_int()).collect(); Ok(Value::IntSet(Arc::new(set))) }

            // === System.IO PrimOps ===
            PrimOp::Stdin => {
                Ok(Value::Handle(Arc::new(HandleValue::stdin())))
            }
            PrimOp::Stdout => {
                Ok(Value::Handle(Arc::new(HandleValue::stdout())))
            }
            PrimOp::Stderr => {
                Ok(Value::Handle(Arc::new(HandleValue::stderr())))
            }
            PrimOp::OpenFile => {
                let path_val = self.force(args[0].clone())?;
                let mode_val = self.force(args[1].clone())?;
                let path = match &path_val {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{path_val:?}") }),
                };
                // Parse IOMode from Data constructor
                let (readable, writable, append) = match &mode_val {
                    Value::Data(d) => match d.con.name.as_str() {
                        "ReadMode" => (true, false, false),
                        "WriteMode" => (false, true, false),
                        "AppendMode" => (false, true, true),
                        "ReadWriteMode" => (true, true, false),
                        _ => (true, false, false),
                    },
                    _ => (true, false, false),
                };
                let file = if append {
                    std::fs::OpenOptions::new().append(true).create(true).open(&path)
                } else if writable {
                    std::fs::File::create(&path)
                } else {
                    std::fs::File::open(&path)
                };
                match file {
                    Ok(f) => Ok(Value::Handle(Arc::new(HandleValue::from_file(f, readable, writable)))),
                    Err(e) => Err(EvalError::UserError(format!("openFile: {e}"))),
                }
            }
            PrimOp::HClose => {
                let h = self.force(args[0].clone())?;
                match &h {
                    Value::Handle(handle) => {
                        let mut guard = handle.file.lock().unwrap();
                        *guard = None; // Drop the file to close it
                        Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HGetChar => {
                use std::io::Read;
                let h = self.force(args[0].clone())?;
                match &h {
                    Value::Handle(handle) => {
                        if handle.kind == HandleKind::Stdin {
                            let mut buf = [0u8; 1];
                            std::io::stdin().read_exact(&mut buf).map_err(|e| EvalError::UserError(format!("hGetChar: {e}")))?;
                            Ok(Value::Char(buf[0] as char))
                        } else {
                            let mut guard = handle.file.lock().unwrap();
                            if let Some(ref mut f) = *guard {
                                let mut buf = [0u8; 1];
                                f.read_exact(&mut buf).map_err(|e| EvalError::UserError(format!("hGetChar: {e}")))?;
                                Ok(Value::Char(buf[0] as char))
                            } else {
                                Err(EvalError::UserError("hGetChar: handle closed".into()))
                            }
                        }
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HGetLine => {
                use std::io::BufRead;
                let h = self.force(args[0].clone())?;
                match &h {
                    Value::Handle(handle) => {
                        if handle.kind == HandleKind::Stdin {
                            let mut line = String::new();
                            std::io::stdin().lock().read_line(&mut line).map_err(|e| EvalError::UserError(format!("hGetLine: {e}")))?;
                            if line.ends_with('\n') { line.pop(); }
                            if line.ends_with('\r') { line.pop(); }
                            Ok(Value::String(Arc::from(line.as_str())))
                        } else {
                            let mut guard = handle.file.lock().unwrap();
                            if let Some(ref mut f) = *guard {
                                let mut line = String::new();
                                std::io::BufReader::new(f).read_line(&mut line).map_err(|e| EvalError::UserError(format!("hGetLine: {e}")))?;
                                if line.ends_with('\n') { line.pop(); }
                                if line.ends_with('\r') { line.pop(); }
                                Ok(Value::String(Arc::from(line.as_str())))
                            } else {
                                Err(EvalError::UserError("hGetLine: handle closed".into()))
                            }
                        }
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HGetContents => {
                use std::io::Read;
                let h = self.force(args[0].clone())?;
                match &h {
                    Value::Handle(handle) => {
                        if handle.kind == HandleKind::Stdin {
                            let mut contents = String::new();
                            std::io::stdin().lock().read_to_string(&mut contents).map_err(|e| EvalError::UserError(format!("hGetContents: {e}")))?;
                            Ok(Value::String(Arc::from(contents.as_str())))
                        } else {
                            let mut guard = handle.file.lock().unwrap();
                            if let Some(ref mut f) = *guard {
                                let mut contents = String::new();
                                f.read_to_string(&mut contents).map_err(|e| EvalError::UserError(format!("hGetContents: {e}")))?;
                                Ok(Value::String(Arc::from(contents.as_str())))
                            } else {
                                Err(EvalError::UserError("hGetContents: handle closed".into()))
                            }
                        }
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HPutChar => {
                use std::io::Write;
                let h = self.force(args[0].clone())?;
                let c = self.force(args[1].clone())?;
                let ch = match &c {
                    Value::Char(c) => *c,
                    _ => return Err(EvalError::TypeError { expected: "Char".into(), got: format!("{c:?}") }),
                };
                match &h {
                    Value::Handle(handle) => {
                        if handle.kind == HandleKind::Stdout {
                            print!("{ch}");
                        } else if handle.kind == HandleKind::Stderr {
                            eprint!("{ch}");
                        } else {
                            let mut guard = handle.file.lock().unwrap();
                            if let Some(ref mut f) = *guard {
                                write!(f, "{ch}").map_err(|e| EvalError::UserError(format!("hPutChar: {e}")))?;
                            } else {
                                return Err(EvalError::UserError("hPutChar: handle closed".into()));
                            }
                        }
                        Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HPutStr => {
                use std::io::Write;
                let h = self.force(args[0].clone())?;
                let s = self.force(args[1].clone())?;
                let text = match &s {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{s:?}") }),
                };
                match &h {
                    Value::Handle(handle) => {
                        if handle.kind == HandleKind::Stdout {
                            print!("{text}");
                        } else if handle.kind == HandleKind::Stderr {
                            eprint!("{text}");
                        } else {
                            let mut guard = handle.file.lock().unwrap();
                            if let Some(ref mut f) = *guard {
                                write!(f, "{text}").map_err(|e| EvalError::UserError(format!("hPutStr: {e}")))?;
                            } else {
                                return Err(EvalError::UserError("hPutStr: handle closed".into()));
                            }
                        }
                        Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HPutStrLn => {
                use std::io::Write;
                let h = self.force(args[0].clone())?;
                let s = self.force(args[1].clone())?;
                let text = match &s {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{s:?}") }),
                };
                match &h {
                    Value::Handle(handle) => {
                        if handle.kind == HandleKind::Stdout {
                            println!("{text}");
                        } else if handle.kind == HandleKind::Stderr {
                            eprintln!("{text}");
                        } else {
                            let mut guard = handle.file.lock().unwrap();
                            if let Some(ref mut f) = *guard {
                                writeln!(f, "{text}").map_err(|e| EvalError::UserError(format!("hPutStrLn: {e}")))?;
                            } else {
                                return Err(EvalError::UserError("hPutStrLn: handle closed".into()));
                            }
                        }
                        Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HPrint => {
                use std::io::Write;
                let h = self.force(args[0].clone())?;
                let v = self.force(args[1].clone())?;
                let text = self.display_value(&v)?;
                match &h {
                    Value::Handle(handle) => {
                        if handle.kind == HandleKind::Stdout {
                            println!("{text}");
                        } else if handle.kind == HandleKind::Stderr {
                            eprintln!("{text}");
                        } else {
                            let mut guard = handle.file.lock().unwrap();
                            if let Some(ref mut f) = *guard {
                                writeln!(f, "{text}").map_err(|e| EvalError::UserError(format!("hPrint: {e}")))?;
                            } else {
                                return Err(EvalError::UserError("hPrint: handle closed".into()));
                            }
                        }
                        Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HFlush => {
                use std::io::Write;
                let h = self.force(args[0].clone())?;
                match &h {
                    Value::Handle(handle) => {
                        if handle.kind == HandleKind::Stdout {
                            std::io::stdout().flush().map_err(|e| EvalError::UserError(format!("hFlush: {e}")))?;
                        } else if handle.kind == HandleKind::Stderr {
                            std::io::stderr().flush().map_err(|e| EvalError::UserError(format!("hFlush: {e}")))?;
                        } else {
                            let mut guard = handle.file.lock().unwrap();
                            if let Some(ref mut f) = *guard {
                                f.flush().map_err(|e| EvalError::UserError(format!("hFlush: {e}")))?;
                            }
                        }
                        Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HIsEOF => {
                // Simplified: returns False for stdin, checks file position for file handles
                let h = self.force(args[0].clone())?;
                match &h {
                    Value::Handle(handle) => {
                        if handle.kind == HandleKind::Stdin {
                            Ok(Value::bool(false))
                        } else {
                            use std::io::Seek;
                            let mut guard = handle.file.lock().unwrap();
                            if let Some(ref mut f) = *guard {
                                let pos = f.stream_position().unwrap_or(0);
                                let len = f.seek(std::io::SeekFrom::End(0)).unwrap_or(0);
                                let _ = f.seek(std::io::SeekFrom::Start(pos));
                                Ok(Value::bool(pos >= len))
                            } else {
                                Ok(Value::bool(true))
                            }
                        }
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HSetBuffering | PrimOp::HGetBuffering => {
                // Simplified: buffering is a no-op in the interpreter
                let h = self.force(args[0].clone())?;
                match &h {
                    Value::Handle(_) => {
                        if matches!(op, PrimOp::HGetBuffering) {
                            // Return LineBuffering as default
                            Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("LineBuffering"), tag: 1, ty_con: bhc_types::TyCon::new(Symbol::intern("BufferMode"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
                        } else {
                            Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
                        }
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HSeek => {
                use std::io::Seek;
                let h = self.force(args[0].clone())?;
                let _mode = self.force(args[1].clone())?;
                let pos = self.force(args[2].clone())?;
                let offset = pos.as_int().unwrap_or(0);
                match &h {
                    Value::Handle(handle) => {
                        let mut guard = handle.file.lock().unwrap();
                        if let Some(ref mut f) = *guard {
                            f.seek(std::io::SeekFrom::Start(offset as u64)).map_err(|e| EvalError::UserError(format!("hSeek: {e}")))?;
                        }
                        Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HTell => {
                use std::io::Seek;
                let h = self.force(args[0].clone())?;
                match &h {
                    Value::Handle(handle) => {
                        let mut guard = handle.file.lock().unwrap();
                        if let Some(ref mut f) = *guard {
                            let pos = f.stream_position().map_err(|e| EvalError::UserError(format!("hTell: {e}")))?;
                            Ok(Value::Integer(pos as i128))
                        } else {
                            Err(EvalError::UserError("hTell: handle closed".into()))
                        }
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::HFileSize => {
                let h = self.force(args[0].clone())?;
                match &h {
                    Value::Handle(handle) => {
                        let guard = handle.file.lock().unwrap();
                        if let Some(ref f) = *guard {
                            let metadata = f.metadata().map_err(|e| EvalError::UserError(format!("hFileSize: {e}")))?;
                            Ok(Value::Integer(metadata.len() as i128))
                        } else {
                            Err(EvalError::UserError("hFileSize: handle closed".into()))
                        }
                    }
                    _ => Err(EvalError::TypeError { expected: "Handle".into(), got: format!("{h:?}") }),
                }
            }
            PrimOp::WithFile => {
                let path_val = self.force(args[0].clone())?;
                let mode_val = self.force(args[1].clone())?;
                let action = self.force(args[2].clone())?;
                // Open the file
                let handle_result = self.apply_primop(PrimOp::OpenFile, vec![path_val, mode_val])?;
                // Apply action to handle
                let result = self.apply(action, handle_result.clone())?;
                let result = self.force(result)?;
                // Close the handle
                if let Value::Handle(_) = &handle_result {
                    let _ = self.apply_primop(PrimOp::HClose, vec![handle_result]);
                }
                Ok(result)
            }

            // === Data.IORef PrimOps ===
            PrimOp::NewIORef => {
                let v = self.force(args[0].clone())?;
                Ok(Value::IORef(Arc::new(Mutex::new(v))))
            }
            PrimOp::ReadIORef => {
                let r = self.force(args[0].clone())?;
                match &r {
                    Value::IORef(ref_cell) => {
                        let guard = ref_cell.lock().unwrap();
                        Ok(guard.clone())
                    }
                    _ => Err(EvalError::TypeError { expected: "IORef".into(), got: format!("{r:?}") }),
                }
            }
            PrimOp::WriteIORef => {
                let r = self.force(args[0].clone())?;
                let v = self.force(args[1].clone())?;
                match &r {
                    Value::IORef(ref_cell) => {
                        let mut guard = ref_cell.lock().unwrap();
                        *guard = v;
                        Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
                    }
                    _ => Err(EvalError::TypeError { expected: "IORef".into(), got: format!("{r:?}") }),
                }
            }
            PrimOp::ModifyIORef | PrimOp::ModifyIORefStrict => {
                let r = self.force(args[0].clone())?;
                let f = self.force(args[1].clone())?;
                match &r {
                    Value::IORef(ref_cell) => {
                        let old = {
                            let guard = ref_cell.lock().unwrap();
                            guard.clone()
                        };
                        let new_val = self.apply(f, old)?;
                        let new_val = if matches!(op, PrimOp::ModifyIORefStrict) {
                            self.force(new_val)?
                        } else {
                            new_val
                        };
                        let mut guard = ref_cell.lock().unwrap();
                        *guard = new_val;
                        Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
                    }
                    _ => Err(EvalError::TypeError { expected: "IORef".into(), got: format!("{r:?}") }),
                }
            }
            PrimOp::AtomicModifyIORef | PrimOp::AtomicModifyIORefStrict => {
                let r = self.force(args[0].clone())?;
                let f = self.force(args[1].clone())?;
                match &r {
                    Value::IORef(ref_cell) => {
                        let old = {
                            let guard = ref_cell.lock().unwrap();
                            guard.clone()
                        };
                        let result = self.apply(f, old)?;
                        let result = self.force(result)?;
                        // Result should be a pair (new_value, return_value)
                        match &result {
                            Value::Data(d) if d.args.len() >= 2 => {
                                let new_val = if matches!(op, PrimOp::AtomicModifyIORefStrict) {
                                    self.force(d.args[0].clone())?
                                } else {
                                    d.args[0].clone()
                                };
                                let ret_val = d.args[1].clone();
                                let mut guard = ref_cell.lock().unwrap();
                                *guard = new_val;
                                Ok(ret_val)
                            }
                            _ => Err(EvalError::UserError("atomicModifyIORef: function must return a pair".into())),
                        }
                    }
                    _ => Err(EvalError::TypeError { expected: "IORef".into(), got: format!("{r:?}") }),
                }
            }

            // === System.Exit PrimOps ===
            PrimOp::ExitSuccess => {
                std::process::exit(0);
            }
            PrimOp::ExitFailure => {
                std::process::exit(1);
            }
            PrimOp::ExitWith => {
                let code = self.force(args[0].clone())?;
                match &code {
                    Value::Data(d) => match d.con.name.as_str() {
                        "ExitSuccess" => std::process::exit(0),
                        "ExitFailure" => {
                            let n = if !d.args.is_empty() {
                                self.force(d.args[0].clone())?.as_int().unwrap_or(1) as i32
                            } else {
                                1
                            };
                            std::process::exit(n);
                        }
                        _ => std::process::exit(1),
                    },
                    Value::Int(n) => std::process::exit(*n as i32),
                    _ => std::process::exit(1),
                }
            }

            // === System.Environment PrimOps ===
            PrimOp::GetArgs => {
                let args_list: Vec<Value> = std::env::args().skip(1).map(|s| Value::String(Arc::from(s.as_str()))).collect();
                Ok(Value::from_list(args_list))
            }
            PrimOp::GetProgName => {
                let name = std::env::args().next().unwrap_or_default();
                Ok(Value::String(Arc::from(name.as_str())))
            }
            PrimOp::GetEnv => {
                let key = self.force(args[0].clone())?;
                let key_str = match &key {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{key:?}") }),
                };
                match std::env::var(&key_str) {
                    Ok(val) => Ok(Value::String(Arc::from(val.as_str()))),
                    Err(_) => Err(EvalError::UserError(format!("getEnv: {key_str}: does not exist"))),
                }
            }
            PrimOp::LookupEnv => {
                let key = self.force(args[0].clone())?;
                let key_str = match &key {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{key:?}") }),
                };
                match std::env::var(&key_str) {
                    Ok(val) => Ok(self.make_just(Value::String(Arc::from(val.as_str())))),
                    Err(_) => Ok(self.make_nothing()),
                }
            }
            PrimOp::SetEnv => {
                let key = self.force(args[0].clone())?;
                let val = self.force(args[1].clone())?;
                let key_str = match &key {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{key:?}") }),
                };
                let val_str = match &val {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{val:?}") }),
                };
                // SAFETY: This is unsafe in Rust but we're in a single-threaded interpreter
                unsafe { std::env::set_var(&key_str, &val_str); }
                Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
            }

            // === System.Directory PrimOps ===
            PrimOp::DoesFileExist => {
                let path = self.force(args[0].clone())?;
                let path_str = match &path {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{path:?}") }),
                };
                Ok(Value::bool(std::path::Path::new(&path_str).is_file()))
            }
            PrimOp::DoesDirectoryExist => {
                let path = self.force(args[0].clone())?;
                let path_str = match &path {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{path:?}") }),
                };
                Ok(Value::bool(std::path::Path::new(&path_str).is_dir()))
            }
            PrimOp::CreateDirectory => {
                let path = self.force(args[0].clone())?;
                let path_str = match &path {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{path:?}") }),
                };
                std::fs::create_dir(&path_str).map_err(|e| EvalError::UserError(format!("createDirectory: {e}")))?;
                Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
            }
            PrimOp::CreateDirectoryIfMissing => {
                let parents = self.force(args[0].clone())?;
                let path = self.force(args[1].clone())?;
                let path_str = match &path {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{path:?}") }),
                };
                let create_parents = parents.as_bool().unwrap_or(false);
                if create_parents {
                    std::fs::create_dir_all(&path_str).map_err(|e| EvalError::UserError(format!("createDirectoryIfMissing: {e}")))?;
                } else if !std::path::Path::new(&path_str).exists() {
                    std::fs::create_dir(&path_str).map_err(|e| EvalError::UserError(format!("createDirectoryIfMissing: {e}")))?;
                }
                Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
            }
            PrimOp::RemoveFile => {
                let path = self.force(args[0].clone())?;
                let path_str = match &path {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{path:?}") }),
                };
                std::fs::remove_file(&path_str).map_err(|e| EvalError::UserError(format!("removeFile: {e}")))?;
                Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
            }
            PrimOp::RemoveDirectory => {
                let path = self.force(args[0].clone())?;
                let path_str = match &path {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{path:?}") }),
                };
                std::fs::remove_dir(&path_str).map_err(|e| EvalError::UserError(format!("removeDirectory: {e}")))?;
                Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
            }
            PrimOp::GetCurrentDirectory => {
                let cwd = std::env::current_dir().map_err(|e| EvalError::UserError(format!("getCurrentDirectory: {e}")))?;
                Ok(Value::String(Arc::from(cwd.to_string_lossy().as_ref())))
            }
            PrimOp::SetCurrentDirectory => {
                let path = self.force(args[0].clone())?;
                let path_str = match &path {
                    Value::String(s) => s.to_string(),
                    _ => return Err(EvalError::TypeError { expected: "String".into(), got: format!("{path:?}") }),
                };
                std::env::set_current_dir(&path_str).map_err(|e| EvalError::UserError(format!("setCurrentDirectory: {e}")))?;
                Ok(Value::Data(DataValue { con: crate::DataCon { name: Symbol::intern("()"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("()"), bhc_types::Kind::Star), arity: 0 }, args: vec![] }))
            }

            // ---- Control.Monad ----
            PrimOp::MonadWhen => {
                // when :: Bool -> IO () -> IO ()
                let cond = self.force(args[0].clone())?;
                match cond.as_bool() {
                    Some(true) => self.force(args[1].clone()),
                    _ => Ok(Value::unit()),
                }
            }
            PrimOp::MonadUnless => {
                // unless :: Bool -> IO () -> IO ()
                let cond = self.force(args[0].clone())?;
                match cond.as_bool() {
                    Some(false) => self.force(args[1].clone()),
                    _ => Ok(Value::unit()),
                }
            }
            PrimOp::MonadGuard => {
                // guard :: Bool -> [()]  (list monad interpretation)
                let cond = self.force(args[0].clone())?;
                match cond.as_bool() {
                    Some(true) => {
                        Ok(Value::from_list(vec![Value::unit()]))
                    }
                    _ => Ok(Value::nil()),
                }
            }
            PrimOp::MonadVoid => {
                // void :: f a -> f ()
                // Evaluate the action, discard result, return unit
                let _ = self.force(args[0].clone())?;
                Ok(Value::unit())
            }
            PrimOp::MonadJoin => {
                // join :: m (m a) -> m a
                // For lists: concat; for IO: just force
                let v = self.force(args[0].clone())?;
                if self.is_list_value(&v) {
                    let outer = self.force_list(v)?;
                    let mut result = Vec::new();
                    for inner in outer {
                        let inner_list = self.force_list(inner)?;
                        result.extend(inner_list);
                    }
                    Ok(Value::from_list(result))
                } else {
                    Ok(v)
                }
            }
            PrimOp::MonadAp => {
                // ap :: m (a -> b) -> m a -> m b
                let mf = self.force(args[0].clone())?;
                let ma = self.force(args[1].clone())?;
                if self.is_list_value(&mf) {
                    let fs = self.force_list(mf)?;
                    let xs = self.force_list(ma)?;
                    let mut result = Vec::new();
                    for f in &fs {
                        for x in &xs {
                            result.push(self.apply(f.clone(), x.clone())?);
                        }
                    }
                    Ok(Value::from_list(result))
                } else {
                    // IO monad: mf is the function, ma is the value
                    self.apply(mf, ma)
                }
            }
            PrimOp::LiftM => {
                // liftM :: (a -> b) -> m a -> m b (= fmap)
                let f = args[0].clone();
                let ma = self.force(args[1].clone())?;
                if self.is_list_value(&ma) {
                    let xs = self.force_list(ma)?;
                    let mut result = Vec::new();
                    for x in xs {
                        result.push(self.apply(f.clone(), x)?);
                    }
                    Ok(Value::from_list(result))
                } else {
                    self.apply(f, ma)
                }
            }
            PrimOp::LiftM2 => {
                // liftM2 :: (a -> b -> c) -> m a -> m b -> m c
                let f = args[0].clone();
                let ma = self.force(args[1].clone())?;
                let mb = self.force(args[2].clone())?;
                if self.is_list_value(&ma) {
                    let xs = self.force_list(ma)?;
                    let ys = self.force_list(mb)?;
                    let mut result = Vec::new();
                    for x in &xs {
                        for y in &ys {
                            let tmp = self.apply(f.clone(), x.clone())?;
                            result.push(self.apply(tmp, y.clone())?);
                        }
                    }
                    Ok(Value::from_list(result))
                } else {
                    let tmp = self.apply(f, ma)?;
                    self.apply(tmp, mb)
                }
            }
            PrimOp::LiftM3 => {
                // liftM3 :: (a -> b -> c -> d) -> m a -> m b -> m c -> m d
                let f = args[0].clone();
                let a = self.force(args[1].clone())?;
                let b = self.force(args[2].clone())?;
                let c = self.force(args[3].clone())?;
                let t1 = self.apply(f, a)?;
                let t2 = self.apply(t1, b)?;
                self.apply(t2, c)
            }
            PrimOp::LiftM4 => {
                // liftM4
                let f = args[0].clone();
                let a = self.force(args[1].clone())?;
                let b = self.force(args[2].clone())?;
                let c = self.force(args[3].clone())?;
                let d = self.force(args[4].clone())?;
                let t1 = self.apply(f, a)?;
                let t2 = self.apply(t1, b)?;
                let t3 = self.apply(t2, c)?;
                self.apply(t3, d)
            }
            PrimOp::LiftM5 => {
                // liftM5
                let f = args[0].clone();
                let a = self.force(args[1].clone())?;
                let b = self.force(args[2].clone())?;
                let c = self.force(args[3].clone())?;
                let d = self.force(args[4].clone())?;
                let e = self.force(args[5].clone())?;
                let t1 = self.apply(f, a)?;
                let t2 = self.apply(t1, b)?;
                let t3 = self.apply(t2, c)?;
                let t4 = self.apply(t3, d)?;
                self.apply(t4, e)
            }
            PrimOp::FilterM => {
                // filterM :: (a -> m Bool) -> [a] -> m [a]
                let pred_fn = args[0].clone();
                let list = self.force_list(args[1].clone())?;
                let mut result = Vec::new();
                for x in list {
                    let b = self.apply(pred_fn.clone(), x.clone())?;
                    let b = self.force(b)?;
                    if b.as_bool().unwrap_or(false) {
                        result.push(x);
                    }
                }
                Ok(Value::from_list(result))
            }
            PrimOp::MapAndUnzipM => {
                // mapAndUnzipM :: (a -> m (b, c)) -> [a] -> m ([b], [c])
                let f = args[0].clone();
                let list = self.force_list(args[1].clone())?;
                let mut bs = Vec::new();
                let mut cs = Vec::new();
                for x in list {
                    let pair = self.apply(f.clone(), x)?;
                    let pair = self.force(pair)?;
                    match &pair {
                        Value::Data(dv) if dv.args.len() == 2 => {
                            bs.push(dv.args[0].clone());
                            cs.push(dv.args[1].clone());
                        }
                        _ => return Err(EvalError::TypeError { expected: "pair".into(), got: format!("{pair:?}") }),
                    }
                }
                Ok(self.make_pair(Value::from_list(bs), Value::from_list(cs)))
            }
            PrimOp::ZipWithM => {
                // zipWithM :: (a -> b -> m c) -> [a] -> [b] -> m [c]
                let f = args[0].clone();
                let xs = self.force_list(args[1].clone())?;
                let ys = self.force_list(args[2].clone())?;
                let mut result = Vec::new();
                for (x, y) in xs.into_iter().zip(ys.into_iter()) {
                    let tmp = self.apply(f.clone(), x)?;
                    let val = self.apply(tmp, y)?;
                    result.push(self.force(val)?);
                }
                Ok(Value::from_list(result))
            }
            PrimOp::ZipWithM_ => {
                // zipWithM_ :: (a -> b -> m c) -> [a] -> [b] -> m ()
                let f = args[0].clone();
                let xs = self.force_list(args[1].clone())?;
                let ys = self.force_list(args[2].clone())?;
                for (x, y) in xs.into_iter().zip(ys.into_iter()) {
                    let tmp = self.apply(f.clone(), x)?;
                    let _ = self.apply(tmp, y)?;
                }
                Ok(Value::unit())
            }
            PrimOp::FoldM => {
                // foldM :: (b -> a -> m b) -> b -> [a] -> m b
                let f = args[0].clone();
                let mut acc = self.force(args[1].clone())?;
                let list = self.force_list(args[2].clone())?;
                for x in list {
                    let tmp = self.apply(f.clone(), acc)?;
                    acc = self.apply(tmp, x)?;
                    acc = self.force(acc)?;
                }
                Ok(acc)
            }
            PrimOp::FoldM_ => {
                // foldM_ :: (b -> a -> m b) -> b -> [a] -> m ()
                let f = args[0].clone();
                let mut acc = self.force(args[1].clone())?;
                let list = self.force_list(args[2].clone())?;
                for x in list {
                    let tmp = self.apply(f.clone(), acc)?;
                    acc = self.apply(tmp, x)?;
                    acc = self.force(acc)?;
                }
                Ok(Value::unit())
            }
            PrimOp::ReplicateM => {
                // replicateM :: Int -> m a -> m [a]
                let n = match self.force(args[0].clone())? {
                    Value::Int(n) => n as usize,
                    other => return Err(EvalError::TypeError { expected: "Int".into(), got: format!("{other:?}") }),
                };
                let action = args[1].clone();
                let mut result = Vec::new();
                for _ in 0..n {
                    result.push(self.force(action.clone())?);
                }
                Ok(Value::from_list(result))
            }
            PrimOp::ReplicateM_ => {
                // replicateM_ :: Int -> m a -> m ()
                let n = match self.force(args[0].clone())? {
                    Value::Int(n) => n as usize,
                    other => return Err(EvalError::TypeError { expected: "Int".into(), got: format!("{other:?}") }),
                };
                let action = args[1].clone();
                for _ in 0..n {
                    let _ = self.force(action.clone())?;
                }
                Ok(Value::unit())
            }
            PrimOp::Forever => {
                // forever :: m a -> m b
                // In the interpreter, this would loop infinitely.
                Err(EvalError::UserError("forever: infinite loop (not supported in interpreter)".into()))
            }
            PrimOp::Mzero => {
                // mzero :: MonadPlus m => m a  (empty list for list monad)
                Ok(Value::nil())
            }
            PrimOp::Mplus => {
                // mplus :: MonadPlus m => m a -> m a -> m a (list concat for list monad)
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                if self.is_list_value(&a) {
                    let mut xs = self.force_list(a)?;
                    let ys = self.force_list(b)?;
                    xs.extend(ys);
                    Ok(Value::from_list(xs))
                } else {
                    // For IO/Maybe: return first non-error
                    Ok(a)
                }
            }
            PrimOp::Msum => {
                // msum :: [m a] -> m a  (= mconcat for MonadPlus)
                let actions = self.force_list(args[0].clone())?;
                let mut result = Vec::new();
                for action in actions {
                    let v = self.force(action)?;
                    if self.is_list_value(&v) {
                        let xs = self.force_list(v)?;
                        result.extend(xs);
                    } else {
                        return Ok(v);
                    }
                }
                Ok(Value::from_list(result))
            }
            PrimOp::Mfilter => {
                // mfilter :: MonadPlus m => (a -> Bool) -> m a -> m a
                let pred_fn = args[0].clone();
                let ma = self.force(args[1].clone())?;
                if self.is_list_value(&ma) {
                    let xs = self.force_list(ma)?;
                    let mut result = Vec::new();
                    for x in xs {
                        let b = self.apply(pred_fn.clone(), x.clone())?;
                        let b = self.force(b)?;
                        if b.as_bool().unwrap_or(false) {
                            result.push(x);
                        }
                    }
                    Ok(Value::from_list(result))
                } else {
                    Ok(ma)
                }
            }
            PrimOp::KleisliCompose => {
                // (>=>) :: (a -> m b) -> (b -> m c) -> a -> m c
                let f = args[0].clone();
                let g = args[1].clone();
                let a = args[2].clone();
                let mb = self.apply(f, a)?;
                let b = self.force(mb)?;
                self.apply(g, b)
            }
            PrimOp::KleisliComposeFlip => {
                // (<=<) :: (b -> m c) -> (a -> m b) -> a -> m c
                let g = args[0].clone();
                let f = args[1].clone();
                let a = args[2].clone();
                let mb = self.apply(f, a)?;
                let b = self.force(mb)?;
                self.apply(g, b)
            }

            // ---- Control.Applicative ----
            PrimOp::LiftA => {
                // liftA :: (a -> b) -> f a -> f b (= fmap)
                let f = args[0].clone();
                let fa = self.force(args[1].clone())?;
                if self.is_list_value(&fa) {
                    let xs = self.force_list(fa)?;
                    let mut result = Vec::new();
                    for x in xs {
                        result.push(self.apply(f.clone(), x)?);
                    }
                    Ok(Value::from_list(result))
                } else {
                    self.apply(f, fa)
                }
            }
            PrimOp::LiftA2 => {
                // liftA2 :: (a -> b -> c) -> f a -> f b -> f c
                let f = args[0].clone();
                let fa = self.force(args[1].clone())?;
                let fb = self.force(args[2].clone())?;
                if self.is_list_value(&fa) {
                    let xs = self.force_list(fa)?;
                    let ys = self.force_list(fb)?;
                    let mut result = Vec::new();
                    for x in &xs {
                        for y in &ys {
                            let tmp = self.apply(f.clone(), x.clone())?;
                            result.push(self.apply(tmp, y.clone())?);
                        }
                    }
                    Ok(Value::from_list(result))
                } else {
                    let tmp = self.apply(f, fa)?;
                    self.apply(tmp, fb)
                }
            }
            PrimOp::LiftA3 => {
                // liftA3 :: (a -> b -> c -> d) -> f a -> f b -> f c -> f d
                let f = args[0].clone();
                let a = self.force(args[1].clone())?;
                let b = self.force(args[2].clone())?;
                let c = self.force(args[3].clone())?;
                let t1 = self.apply(f, a)?;
                let t2 = self.apply(t1, b)?;
                self.apply(t2, c)
            }
            PrimOp::Optional => {
                // optional :: Alternative f => f a -> f (Maybe a)
                // For lists: if empty return [Nothing], else map Just
                let fa = self.force(args[0].clone())?;
                if self.is_list_value(&fa) {
                    let xs = self.force_list(fa)?;
                    if xs.is_empty() {
                        Ok(Value::from_list(vec![self.make_nothing()]))
                    } else {
                        let mut result = Vec::new();
                        for x in xs {
                            result.push(self.make_just(x));
                        }
                        Ok(Value::from_list(result))
                    }
                } else {
                    Ok(self.make_just(fa))
                }
            }

            // ---- Control.Exception ----
            PrimOp::ExnCatch => {
                // catch :: IO a -> (SomeException -> IO a) -> IO a
                let action = args[0].clone();
                let handler = args[1].clone();
                match self.force(action) {
                    Ok(v) => Ok(v),
                    Err(e) => {
                        let err_str = Value::String(Arc::from(format!("{e}")));
                        self.apply(handler, err_str)
                    }
                }
            }
            PrimOp::ExnTry => {
                // try :: IO a -> IO (Either SomeException a)
                let action = args[0].clone();
                match self.force(action) {
                    Ok(v) => {
                        // Right v
                        Ok(Value::Data(DataValue {
                            con: crate::DataCon { name: Symbol::intern("Right"), tag: 1, ty_con: bhc_types::TyCon::new(Symbol::intern("Either"), bhc_types::Kind::Star), arity: 1 },
                            args: vec![v],
                        }))
                    }
                    Err(e) => {
                        // Left (show e)
                        let err_str = Value::String(Arc::from(format!("{e}")));
                        Ok(Value::Data(DataValue {
                            con: crate::DataCon { name: Symbol::intern("Left"), tag: 0, ty_con: bhc_types::TyCon::new(Symbol::intern("Either"), bhc_types::Kind::Star), arity: 1 },
                            args: vec![err_str],
                        }))
                    }
                }
            }
            PrimOp::ExnThrow => {
                // throw :: SomeException -> a
                let v = self.force(args[0].clone())?;
                let msg = match &v {
                    Value::String(s) => s.to_string(),
                    _ => format!("{v:?}"),
                };
                Err(EvalError::UserError(msg))
            }
            PrimOp::ExnThrowIO => {
                // throwIO :: SomeException -> IO a
                let v = self.force(args[0].clone())?;
                let msg = match &v {
                    Value::String(s) => s.to_string(),
                    _ => format!("{v:?}"),
                };
                Err(EvalError::UserError(msg))
            }
            PrimOp::ExnBracket => {
                // bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
                let acquire = args[0].clone();
                let release = args[1].clone();
                let body = args[2].clone();
                let resource = self.force(acquire)?;
                let result = self.apply(body, resource.clone());
                // Always run release
                let _ = self.apply(release, resource);
                result
            }
            PrimOp::ExnBracket_ => {
                // bracket_ :: IO a -> IO b -> IO c -> IO c
                let before = args[0].clone();
                let after = args[1].clone();
                let body = args[2].clone();
                let _ = self.force(before)?;
                let result = self.force(body);
                let _ = self.force(after);
                result
            }
            PrimOp::ExnBracketOnError => {
                // bracketOnError :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
                let acquire = args[0].clone();
                let release = args[1].clone();
                let body = args[2].clone();
                let resource = self.force(acquire)?;
                let result = self.apply(body, resource.clone());
                if result.is_err() {
                    let _ = self.apply(release, resource);
                }
                result
            }
            PrimOp::ExnFinally => {
                // finally :: IO a -> IO b -> IO a
                let action = args[0].clone();
                let cleanup = args[1].clone();
                let result = self.force(action);
                let _ = self.force(cleanup);
                result
            }
            PrimOp::ExnOnException => {
                // onException :: IO a -> IO b -> IO a
                let action = args[0].clone();
                let cleanup = args[1].clone();
                let result = self.force(action);
                if result.is_err() {
                    let _ = self.force(cleanup);
                }
                result
            }
            PrimOp::ExnHandle => {
                // handle :: (SomeException -> IO a) -> IO a -> IO a (= flip catch)
                let handler = args[0].clone();
                let action = args[1].clone();
                match self.force(action) {
                    Ok(v) => Ok(v),
                    Err(e) => {
                        let err_str = Value::String(Arc::from(format!("{e}")));
                        self.apply(handler, err_str)
                    }
                }
            }
            PrimOp::ExnHandleJust => {
                // handleJust :: (SomeException -> Maybe b) -> (b -> IO a) -> IO a -> IO a
                let pred_fn = args[0].clone();
                let handler = args[1].clone();
                let action = args[2].clone();
                match self.force(action) {
                    Ok(v) => Ok(v),
                    Err(e) => {
                        let err_str = Value::String(Arc::from(format!("{e}")));
                        let maybe_b = self.apply(pred_fn, err_str)?;
                        let maybe_b = self.force(maybe_b)?;
                        match &maybe_b {
                            Value::Data(dv) if dv.con.name.as_str() == "Just" && !dv.args.is_empty() => {
                                self.apply(handler, dv.args[0].clone())
                            }
                            _ => Err(e),
                        }
                    }
                }
            }
            PrimOp::ExnEvaluate => {
                // evaluate :: a -> IO a (force to WHNF)
                self.force(args[0].clone())
            }
            PrimOp::ExnMask => {
                // mask :: ((IO a -> IO a) -> IO b) -> IO b
                // In the interpreter: no masking needed, pass identity as the restore function
                let f = args[0].clone();
                let restore = args[1].clone();
                let tmp = self.apply(f, restore)?;
                self.force(tmp)
            }
            PrimOp::ExnMask_ => {
                // mask_ :: IO a -> IO a (just run the action)
                self.force(args[0].clone())
            }
            PrimOp::ExnUninterruptibleMask => {
                // uninterruptibleMask :: ((IO a -> IO a) -> IO b) -> IO b
                // Same as mask in the interpreter
                let f = args[0].clone();
                let restore = args[1].clone();
                let tmp = self.apply(f, restore)?;
                self.force(tmp)
            }
            PrimOp::ExnUninterruptibleMask_ => {
                // uninterruptibleMask_ :: IO a -> IO a
                self.force(args[0].clone())
            }

            // ---- Control.Concurrent ----
            PrimOp::ForkIO => {
                // forkIO :: IO () -> IO ThreadId
                // In interpreter: just run the action synchronously and return a fake thread id
                let _ = self.force(args[0].clone())?;
                Ok(Value::Int(0)) // Fake ThreadId
            }
            PrimOp::ThreadDelay => {
                // threadDelay :: Int -> IO ()
                let micros = match self.force(args[0].clone())? {
                    Value::Int(n) => n,
                    other => return Err(EvalError::TypeError { expected: "Int".into(), got: format!("{other:?}") }),
                };
                std::thread::sleep(std::time::Duration::from_micros(micros as u64));
                Ok(Value::unit())
            }
            PrimOp::MyThreadId => {
                // myThreadId :: IO ThreadId
                Ok(Value::Int(0)) // Fake ThreadId
            }
            PrimOp::NewMVar => {
                // newMVar :: a -> IO (MVar a)
                let v = self.force(args[0].clone())?;
                Ok(Value::IORef(Arc::new(Mutex::new(v))))
            }
            PrimOp::NewEmptyMVar => {
                // newEmptyMVar :: IO (MVar a)
                Ok(Value::IORef(Arc::new(Mutex::new(Value::unit()))))
            }
            PrimOp::TakeMVar => {
                // takeMVar :: MVar a -> IO a
                let mvar = self.force(args[0].clone())?;
                match &mvar {
                    Value::IORef(r) => {
                        let val = r.lock().unwrap().clone();
                        Ok(val)
                    }
                    _ => Err(EvalError::TypeError { expected: "MVar".into(), got: format!("{mvar:?}") }),
                }
            }
            PrimOp::PutMVar => {
                // putMVar :: MVar a -> a -> IO ()
                let mvar = self.force(args[0].clone())?;
                let val = self.force(args[1].clone())?;
                match &mvar {
                    Value::IORef(r) => {
                        *r.lock().unwrap() = val;
                        Ok(Value::unit())
                    }
                    _ => Err(EvalError::TypeError { expected: "MVar".into(), got: format!("{mvar:?}") }),
                }
            }
            PrimOp::ReadMVar => {
                // readMVar :: MVar a -> IO a (non-destructive read)
                let mvar = self.force(args[0].clone())?;
                match &mvar {
                    Value::IORef(r) => {
                        let val = r.lock().unwrap().clone();
                        Ok(val)
                    }
                    _ => Err(EvalError::TypeError { expected: "MVar".into(), got: format!("{mvar:?}") }),
                }
            }
            PrimOp::ThrowTo => {
                // throwTo :: ThreadId -> SomeException -> IO ()
                let _tid = self.force(args[0].clone())?;
                let exc = self.force(args[1].clone())?;
                let msg = match &exc {
                    Value::String(s) => s.to_string(),
                    _ => format!("{exc:?}"),
                };
                Err(EvalError::UserError(format!("throwTo: {msg}")))
            }
            PrimOp::KillThread => {
                // killThread :: ThreadId -> IO ()
                let _ = self.force(args[0].clone())?;
                Ok(Value::unit())
            }

            // ---- Data.Ord ----
            PrimOp::Comparing => {
                // comparing :: (a -> b) -> a -> a -> Ordering
                let f = args[0].clone();
                let x = args[1].clone();
                let y = args[2].clone();
                let fx = self.apply(f.clone(), x)?;
                let fy = self.apply(f, y)?;
                let fx = self.force(fx)?;
                let fy = self.force(fy)?;
                // Use OrdValue comparison to produce Ordering
                let ord = OrdValue(fx).cmp(&OrdValue(fy));
                Ok(self.make_ordering(ord))
            }
            PrimOp::Clamp => {
                // clamp :: (a, a) -> a -> a
                let bounds = self.force(args[0].clone())?;
                let val = self.force(args[1].clone())?;
                match &bounds {
                    Value::Data(dv) if dv.con.name.as_str() == "(,)" && dv.args.len() == 2 => {
                        let lo = self.force(dv.args[0].clone())?;
                        let hi = self.force(dv.args[1].clone())?;
                        let cmp_lo = OrdValue(val.clone()).cmp(&OrdValue(lo.clone()));
                        if cmp_lo == std::cmp::Ordering::Less {
                            return Ok(lo);
                        }
                        let cmp_hi = OrdValue(val.clone()).cmp(&OrdValue(hi.clone()));
                        if cmp_hi == std::cmp::Ordering::Greater {
                            return Ok(hi);
                        }
                        Ok(val)
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "pair".into(),
                        got: format!("{bounds:?}"),
                    }),
                }
            }

            // ---- Data.Foldable ----
            PrimOp::Fold => {
                // fold :: Monoid m => [m] -> m
                let list = self.force_list(args[0].clone())?;
                if list.is_empty() {
                    return Ok(Value::String("".into()));
                }
                let first = self.force(list[0].clone())?;
                match &first {
                    Value::String(_) => {
                        let mut s = String::new();
                        for item in list {
                            let item = self.force(item)?;
                            if let Value::String(ref rs) = item {
                                s.push_str(rs);
                            }
                        }
                        Ok(Value::String(s.into()))
                    }
                    _ if self.is_list_value(&first) => {
                        let mut all = Vec::new();
                        for item in list {
                            let item = self.force(item)?;
                            if self.is_list_value(&item) {
                                all.extend(self.force_list(item)?);
                            } else {
                                all.push(item);
                            }
                        }
                        Ok(Value::from_list(all))
                    }
                    _ => Ok(first),
                }
            }
            PrimOp::FoldMap => {
                // foldMap :: (a -> m) -> [a] -> m
                let f = args[0].clone();
                let list = self.force_list(args[1].clone())?;
                let mut results = Vec::new();
                for item in list {
                    let mapped = self.apply(f.clone(), item)?;
                    results.push(self.force(mapped)?);
                }
                if results.is_empty() {
                    return Ok(Value::String("".into()));
                }
                match &results[0] {
                    Value::String(_) => {
                        let mut s = String::new();
                        for r in results {
                            if let Value::String(ref rs) = r {
                                s.push_str(rs);
                            }
                        }
                        Ok(Value::String(s.into()))
                    }
                    _ => {
                        let mut all = Vec::new();
                        for r in results {
                            if self.is_list_value(&r) {
                                all.extend(self.force_list(r)?);
                            } else {
                                all.push(r);
                            }
                        }
                        Ok(Value::from_list(all))
                    }
                }
            }
            PrimOp::FoldrStrict => {
                // foldr' :: (a -> b -> b) -> b -> [a] -> b
                let f = args[0].clone();
                let z = self.force(args[1].clone())?;
                let list = self.force_list(args[2].clone())?;
                let mut acc = z;
                for item in list.into_iter().rev() {
                    let tmp = self.apply(f.clone(), item)?;
                    acc = self.apply(tmp, acc)?;
                    acc = self.force(acc)?;
                }
                Ok(acc)
            }
            PrimOp::Foldl1 => {
                // foldl1 :: (a -> a -> a) -> [a] -> a
                let f = args[0].clone();
                let list = self.force_list(args[1].clone())?;
                if list.is_empty() {
                    return Err(EvalError::UserError("foldl1: empty list".into()));
                }
                let mut acc = self.force(list[0].clone())?;
                for item in list.into_iter().skip(1) {
                    let tmp = self.apply(f.clone(), acc)?;
                    acc = self.apply(tmp, item)?;
                    acc = self.force(acc)?;
                }
                Ok(acc)
            }
            PrimOp::Foldr1 => {
                // foldr1 :: (a -> a -> a) -> [a] -> a
                let f = args[0].clone();
                let list = self.force_list(args[1].clone())?;
                if list.is_empty() {
                    return Err(EvalError::UserError("foldr1: empty list".into()));
                }
                let mut acc = self.force(list.last().unwrap().clone())?;
                for item in list.into_iter().rev().skip(1) {
                    let item = self.force(item)?;
                    let tmp = self.apply(f.clone(), item)?;
                    acc = self.apply(tmp, acc)?;
                    acc = self.force(acc)?;
                }
                Ok(acc)
            }
            PrimOp::MaximumBy => {
                // maximumBy :: (a -> a -> Ordering) -> [a] -> a
                let cmp = args[0].clone();
                let list = self.force_list(args[1].clone())?;
                if list.is_empty() {
                    return Err(EvalError::UserError("maximumBy: empty list".into()));
                }
                let mut best = self.force(list[0].clone())?;
                for item in list.into_iter().skip(1) {
                    let item = self.force(item)?;
                    let tmp = self.apply(cmp.clone(), item.clone())?;
                    let ord = self.apply(tmp, best.clone())?;
                    let ord = self.force(ord)?;
                    if let Value::Data(ref dv) = ord {
                        if dv.con.name.as_str() == "GT" {
                            best = item;
                        }
                    }
                }
                Ok(best)
            }
            PrimOp::MinimumBy => {
                // minimumBy :: (a -> a -> Ordering) -> [a] -> a
                let cmp = args[0].clone();
                let list = self.force_list(args[1].clone())?;
                if list.is_empty() {
                    return Err(EvalError::UserError("minimumBy: empty list".into()));
                }
                let mut best = self.force(list[0].clone())?;
                for item in list.into_iter().skip(1) {
                    let item = self.force(item)?;
                    let tmp = self.apply(cmp.clone(), item.clone())?;
                    let ord = self.apply(tmp, best.clone())?;
                    let ord = self.force(ord)?;
                    if let Value::Data(ref dv) = ord {
                        if dv.con.name.as_str() == "LT" {
                            best = item;
                        }
                    }
                }
                Ok(best)
            }
            PrimOp::Asum => {
                // asum :: [Maybe a] -> Maybe a (first Just)
                let list = self.force_list(args[0].clone())?;
                for item in list {
                    let item = self.force(item)?;
                    if let Value::Data(ref dv) = item {
                        if dv.con.name.as_str() == "Just" {
                            return Ok(item);
                        }
                    }
                }
                Ok(self.make_nothing())
            }
            PrimOp::Traverse_ => {
                // traverse_ :: (a -> f b) -> [a] -> f ()
                let f = args[0].clone();
                let list = self.force_list(args[1].clone())?;
                for item in list {
                    let _ = self.apply(f.clone(), item)?;
                }
                Ok(Value::unit())
            }
            PrimOp::For_ => {
                // for_ :: [a] -> (a -> f b) -> f ()
                let list = self.force_list(args[0].clone())?;
                let f = args[1].clone();
                for item in list {
                    let _ = self.apply(f.clone(), item)?;
                }
                Ok(Value::unit())
            }
            PrimOp::SequenceA_ => {
                // sequenceA_ :: [f a] -> f ()
                let list = self.force_list(args[0].clone())?;
                for item in list {
                    let _ = self.force(item)?;
                }
                Ok(Value::unit())
            }

            // ---- Data.Traversable ----
            PrimOp::Traverse => {
                // traverse :: (a -> f b) -> [a] -> f [b]
                let f = args[0].clone();
                let list = self.force_list(args[1].clone())?;
                let mut results = Vec::new();
                for item in list {
                    let r = self.apply(f.clone(), item)?;
                    results.push(self.force(r)?);
                }
                Ok(Value::from_list(results))
            }
            PrimOp::SequenceA => {
                // sequenceA :: [f a] -> f [a]
                let list = self.force_list(args[0].clone())?;
                let mut results = Vec::new();
                for item in list {
                    results.push(self.force(item)?);
                }
                Ok(Value::from_list(results))
            }

            // ---- Data.Monoid ----
            PrimOp::Mempty => {
                // mempty :: Monoid a => a (default: empty list)
                Ok(Value::nil())
            }
            PrimOp::Mappend => {
                // mappend :: a -> a -> a
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::String(s1), Value::String(s2)) => {
                        let mut s = s1.to_string();
                        s.push_str(s2);
                        Ok(Value::String(s.into()))
                    }
                    _ if self.is_list_value(&a) => {
                        let mut xs = self.force_list(a)?;
                        let ys = self.force_list(b)?;
                        xs.extend(ys);
                        Ok(Value::from_list(xs))
                    }
                    _ => Ok(a),
                }
            }
            PrimOp::Mconcat => {
                // mconcat :: [a] -> a
                let list = self.force_list(args[0].clone())?;
                if list.is_empty() {
                    return Ok(Value::nil());
                }
                let first = self.force(list[0].clone())?;
                match &first {
                    Value::String(_) => {
                        let mut s = String::new();
                        for item in list {
                            let item = self.force(item)?;
                            if let Value::String(ref rs) = item {
                                s.push_str(rs);
                            }
                        }
                        Ok(Value::String(s.into()))
                    }
                    _ => {
                        let mut all = Vec::new();
                        for item in list {
                            let item = self.force(item)?;
                            if self.is_list_value(&item) {
                                all.extend(self.force_list(item)?);
                            } else {
                                all.push(item);
                            }
                        }
                        Ok(Value::from_list(all))
                    }
                }
            }

            // ---- Data.String ----
            PrimOp::FromString => {
                // fromString :: String -> a (identity for String)
                self.force(args[0].clone())
            }

            // ---- Data.Bits ----
            PrimOp::BitAnd => {
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x & y)),
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitOr => {
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x | y)),
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitXor => {
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x ^ y)),
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitComplement => {
                let a = self.force(args[0].clone())?;
                match &a {
                    Value::Int(x) => Ok(Value::Int(!x)),
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitShift | PrimOp::BitShiftL => {
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::Int(x), Value::Int(n)) => {
                        if *n >= 0 {
                            Ok(Value::Int(x.wrapping_shl(*n as u32)))
                        } else {
                            Ok(Value::Int(x.wrapping_shr((-*n) as u32)))
                        }
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitShiftR => {
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::Int(x), Value::Int(n)) => Ok(Value::Int(x.wrapping_shr(*n as u32))),
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitRotate | PrimOp::BitRotateL => {
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::Int(x), Value::Int(n)) => {
                        Ok(Value::Int(x.rotate_left(*n as u32)))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitRotateR => {
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::Int(x), Value::Int(n)) => {
                        Ok(Value::Int(x.rotate_right(*n as u32)))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitBit => {
                let a = self.force(args[0].clone())?;
                match &a {
                    Value::Int(n) => Ok(Value::Int(1i64.wrapping_shl(*n as u32))),
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitSetBit => {
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::Int(x), Value::Int(n)) => {
                        Ok(Value::Int(x | (1i64.wrapping_shl(*n as u32))))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitClearBit => {
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::Int(x), Value::Int(n)) => {
                        Ok(Value::Int(x & !(1i64.wrapping_shl(*n as u32))))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitComplementBit => {
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::Int(x), Value::Int(n)) => {
                        Ok(Value::Int(x ^ (1i64.wrapping_shl(*n as u32))))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitTestBit => {
                let a = self.force(args[0].clone())?;
                let b = self.force(args[1].clone())?;
                match (&a, &b) {
                    (Value::Int(x), Value::Int(n)) => {
                        Ok(Value::bool((x.wrapping_shr(*n as u32)) & 1 == 1))
                    }
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitPopCount => {
                let a = self.force(args[0].clone())?;
                match &a {
                    Value::Int(x) => Ok(Value::Int(x.count_ones() as i64)),
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitZeroBits => Ok(Value::Int(0)),
            PrimOp::BitCountLeadingZeros => {
                let a = self.force(args[0].clone())?;
                match &a {
                    Value::Int(x) => Ok(Value::Int(x.leading_zeros() as i64)),
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }
            PrimOp::BitCountTrailingZeros => {
                let a = self.force(args[0].clone())?;
                match &a {
                    Value::Int(x) => Ok(Value::Int(x.trailing_zeros() as i64)),
                    _ => Err(EvalError::TypeError {
                        expected: "Int".into(),
                        got: format!("{a:?}"),
                    }),
                }
            }

            // ---- Data.Proxy ----
            PrimOp::AsProxyTypeOf => {
                // asProxyTypeOf :: a -> proxy a -> a (returns first arg)
                self.force(args[0].clone())
            }

            // ---- Data.Void ----
            PrimOp::Absurd => {
                Err(EvalError::UserError("absurd: Void value".into()))
            }
            PrimOp::Vacuous => {
                // vacuous :: f Void -> f a (just return the structure)
                self.force(args[0].clone())
            }
        }
    }

    fn partial_primop(&self, op: PrimOp, args: Vec<Value>) -> Result<Value, EvalError> {
        // Return a partially applied primop that stores accumulated arguments
        Ok(Value::PartialPrimOp(op, args))
    }

    fn eval_let(&self, bind: &Bind, body: &Expr, env: &Env) -> Result<Value, EvalError> {
        match bind {
            Bind::NonRec(var, rhs) => {
                let rhs_val = match self.mode {
                    EvalMode::Strict => self.eval(rhs, env)?,
                    EvalMode::Lazy => Value::Thunk(Thunk {
                        expr: rhs.clone(),
                        env: env.clone(),
                    }),
                };
                let new_env = env.extend(var.id, rhs_val);
                self.eval(body, &new_env)
            }

            Bind::Rec(bindings) => {
                // For recursive bindings, we need closures to be able to find
                // their recursive references. Since closures capture their env
                // at creation time and our envs are immutable, we use a
                // "recursive env stack" that is checked during variable lookup.
                //
                // Strategy:
                // 1. Evaluate all RHS expressions (typically lambdas) with current env
                //    This creates Closures that capture env (without recursive bindings)
                // 2. Build final_env with all the evaluated values
                // 3. Push final_env onto rec_env_stack before evaluating body
                // 4. Variable lookup checks rec_env_stack, so recursive calls work
                //
                // This approach works because:
                // - When a Closure is applied, its body is evaluated
                // - Variable lookup in the body checks rec_env_stack
                // - The rec_env_stack contains the binding for the recursive function

                let mut final_env = env.clone();

                // Evaluate all bindings and add to final_env
                // For lambdas, this just creates Closures (very cheap)
                for (var, rhs) in bindings {
                    let value = self.eval(rhs, env)?;
                    final_env = final_env.extend(var.id, value);
                }

                // Push the recursive environment onto the stack
                // This makes recursive bindings visible during body evaluation
                self.rec_env_stack.borrow_mut().push(final_env.clone());

                // Evaluate the body with final_env
                // The rec_env_stack ensures recursive calls can find their targets
                let result = self.eval(body, &final_env);

                // Pop the recursive environment
                self.rec_env_stack.borrow_mut().pop();

                result
            }
        }
    }

    fn eval_case(&self, scrut: &Expr, alts: &[crate::Alt], env: &Env) -> Result<Value, EvalError> {
        // Evaluate scrutinee to WHNF
        let scrut_val = self.eval(scrut, env)?;
        let scrut_val = self.force(scrut_val)?;

        // Find matching alternative
        for alt in alts {
            if let Some(bindings) = self.match_pattern(&alt.con, &alt.binders, &scrut_val)? {
                let new_env = env.extend_many(bindings);
                return self.eval(&alt.rhs, &new_env);
            }
        }

        Err(EvalError::PatternMatchFailure)
    }

    fn match_pattern(
        &self,
        pattern: &AltCon,
        binders: &[Var],
        value: &Value,
    ) -> Result<Option<Vec<(VarId, Value)>>, EvalError> {
        match pattern {
            AltCon::Default => {
                // Default matches anything
                // If there's a binder, bind it to the scrutinee value
                if binders.is_empty() {
                    Ok(Some(Vec::new()))
                } else {
                    // For variable patterns like `case e of x -> ...`,
                    // bind x to the scrutinee value
                    let bindings: Vec<_> =
                        binders.iter().map(|var| (var.id, value.clone())).collect();
                    Ok(Some(bindings))
                }
            }

            AltCon::Lit(lit) => {
                // Match literal
                let matches = match (lit, value) {
                    (Literal::Int(a), Value::Int(b)) => *a == *b,
                    (Literal::Integer(a), Value::Integer(b)) => *a == *b,
                    (Literal::Float(a), Value::Float(b)) => (*a - *b).abs() < f32::EPSILON,
                    (Literal::Double(a), Value::Double(b)) => (*a - *b).abs() < f64::EPSILON,
                    (Literal::Char(a), Value::Char(b)) => *a == *b,
                    (Literal::String(a), Value::String(b)) => a.as_str() == b.as_ref(),
                    _ => false,
                };
                Ok(if matches { Some(Vec::new()) } else { None })
            }

            AltCon::DataCon(con) => {
                // Match data constructor by name (tags may differ due to ID allocation)
                if let Value::Data(data) = value {
                    if data.con.name == con.name {
                        // Bind constructor arguments to pattern variables
                        let bindings: Vec<_> = binders
                            .iter()
                            .zip(data.args.iter())
                            .map(|(var, val)| (var.id, val.clone()))
                            .collect();
                        return Ok(Some(bindings));
                    }
                }
                Ok(None)
            }
        }
    }

    /// Forces a value to WHNF (evaluates thunks).
    pub fn force(&self, value: Value) -> Result<Value, EvalError> {
        match value {
            Value::Thunk(thunk) => self.force_thunk(&thunk),
            other => Ok(other),
        }
    }

    /// Checks if a value is a list (either empty list or cons cell).
    fn is_list_value(&self, value: &Value) -> bool {
        match value {
            Value::Data(d) => {
                let name = d.con.name.as_str();
                name == "[]" || name == ":"
            }
            Value::String(_) => true, // String is [Char]
            _ => false,
        }
    }

    /// Structural equality for values (used by elem, lookup, etc.)
    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => x == y,
            (Value::Double(x), Value::Double(y)) => x == y,
            (Value::Char(x), Value::Char(y)) => x == y,
            (Value::String(x), Value::String(y)) => x == y,
            (Value::Data(x), Value::Data(y)) => {
                x.con.name == y.con.name
                    && x.args.len() == y.args.len()
                    && x.args
                        .iter()
                        .zip(y.args.iter())
                        .all(|(a, b)| self.values_equal(a, b))
            }
            _ => false,
        }
    }

    /// Create a pair value (a, b).
    fn make_pair(&self, a: Value, b: Value) -> Value {
        use bhc_types::{Kind, TyCon};
        Value::Data(DataValue {
            con: crate::DataCon {
                name: Symbol::intern("(,)"),
                ty_con: TyCon::new(Symbol::intern("(,)"), Kind::Star),
                tag: 0,
                arity: 2,
            },
            args: vec![a, b],
        })
    }

    /// Create a Just value.
    fn make_just(&self, a: Value) -> Value {
        use bhc_types::{Kind, TyCon};
        Value::Data(DataValue {
            con: crate::DataCon {
                name: Symbol::intern("Just"),
                ty_con: TyCon::new(Symbol::intern("Maybe"), Kind::star_to_star()),
                tag: 1,
                arity: 1,
            },
            args: vec![a],
        })
    }

    /// Create a Nothing value.
    fn make_nothing(&self) -> Value {
        use bhc_types::{Kind, TyCon};
        Value::Data(DataValue {
            con: crate::DataCon {
                name: Symbol::intern("Nothing"),
                ty_con: TyCon::new(Symbol::intern("Maybe"), Kind::star_to_star()),
                tag: 0,
                arity: 0,
            },
            args: Vec::new(),
        })
    }

    /// Converts a Value to a String, for use in IO operations.
    /// Handles both String values and [Char] lists.
    fn value_to_string(&self, value: &Value) -> Result<String, EvalError> {
        let forced = self.force(value.clone())?;
        match &forced {
            Value::String(s) => Ok(s.to_string()),
            Value::Data(d) if d.con.name.as_str() == "[]" || d.con.name.as_str() == ":" => {
                // It's a list - try to interpret as [Char]
                let list = self.force_list(forced)?;
                let chars: Result<String, _> = list
                    .into_iter()
                    .map(|v| match v {
                        Value::Char(c) => Ok(c),
                        other => Err(EvalError::TypeError {
                            expected: "Char".into(),
                            got: format!("{other:?}"),
                        }),
                    })
                    .collect();
                chars
            }
            other => Err(EvalError::TypeError {
                expected: "String".into(),
                got: format!("{other:?}"),
            }),
        }
    }

    fn force_thunk(&self, thunk: &Thunk) -> Result<Value, EvalError> {
        self.eval(&thunk.expr, &thunk.env)
    }

    /// Forces a list structure, converting it to a Vec<Value>.
    /// This traverses the list spine, forcing thunks along the way.
    fn force_list(&self, value: Value) -> Result<Vec<Value>, EvalError> {
        let mut result = Vec::new();
        let mut current = self.force(value)?;

        loop {
            match &current {
                // String is [Char] - convert to list of characters
                Value::String(s) => {
                    for c in s.chars() {
                        result.push(Value::Char(c));
                    }
                    return Ok(result);
                }
                Value::Data(d) if d.con.name.as_str() == "[]" => {
                    return Ok(result);
                }
                Value::Data(d) if d.con.name.as_str() == ":" && d.args.len() == 2 => {
                    // Force the head element
                    let head = self.force(d.args[0].clone())?;
                    result.push(head);
                    // Force the tail and continue traversing
                    current = self.force(d.args[1].clone())?;
                }
                other => {
                    return Err(EvalError::TypeError {
                        expected: "List".into(),
                        got: format!("{other:?}"),
                    });
                }
            }
        }
    }

    /// Extract a char from a Value.
    fn as_char(&self, value: &Value) -> Result<char, EvalError> {
        let forced = self.force(value.clone())?;
        match &forced {
            Value::Char(c) => Ok(*c),
            other => Err(EvalError::TypeError {
                expected: "Char".into(),
                got: format!("{other:?}"),
            }),
        }
    }

    /// Compare two values for ordering.
    fn value_compare(&self, a: &Value, b: &Value) -> std::cmp::Ordering {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => x.cmp(y),
            (Value::Integer(x), Value::Integer(y)) => x.cmp(y),
            (Value::Double(x), Value::Double(y)) => {
                x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
            }
            (Value::Float(x), Value::Float(y)) => {
                x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
            }
            (Value::Char(x), Value::Char(y)) => x.cmp(y),
            (Value::String(x), Value::String(y)) => x.cmp(y),
            (Value::Data(x), Value::Data(y)) => {
                match x.con.tag.cmp(&y.con.tag) {
                    std::cmp::Ordering::Equal => {
                        for (ax, ay) in x.args.iter().zip(y.args.iter()) {
                            let c = self.value_compare(ax, ay);
                            if c != std::cmp::Ordering::Equal {
                                return c;
                            }
                        }
                        std::cmp::Ordering::Equal
                    }
                    ord => ord,
                }
            }
            _ => std::cmp::Ordering::Equal,
        }
    }

    /// Convert an Ordering data constructor value to std::cmp::Ordering.
    fn ordering_value_to_cmp(&self, v: &Value) -> std::cmp::Ordering {
        match v {
            Value::Data(d) => match d.con.name.as_str() {
                "LT" => std::cmp::Ordering::Less,
                "EQ" => std::cmp::Ordering::Equal,
                "GT" => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            },
            _ => std::cmp::Ordering::Equal,
        }
    }

    /// Create a triple value (a, b, c).
    fn make_triple(&self, a: Value, b: Value, c: Value) -> Value {
        use bhc_types::{Kind, TyCon};
        Value::Data(DataValue {
            con: crate::DataCon {
                name: Symbol::intern("(,,)"),
                ty_con: TyCon::new(Symbol::intern("(,,)"), Kind::Star),
                tag: 0,
                arity: 3,
            },
            args: vec![a, b, c],
        })
    }

    /// Create a Left value.
    fn make_left(&self, a: Value) -> Value {
        use bhc_types::{Kind, TyCon};
        Value::Data(DataValue {
            con: crate::DataCon {
                name: Symbol::intern("Left"),
                ty_con: TyCon::new(Symbol::intern("Either"), Kind::star_to_star()),
                tag: 0,
                arity: 1,
            },
            args: vec![a],
        })
    }

    /// Create a Right value.
    fn make_right(&self, a: Value) -> Value {
        use bhc_types::{Kind, TyCon};
        Value::Data(DataValue {
            con: crate::DataCon {
                name: Symbol::intern("Right"),
                ty_con: TyCon::new(Symbol::intern("Either"), Kind::star_to_star()),
                tag: 1,
                arity: 1,
            },
            args: vec![a],
        })
    }

    /// Create an Ordering value (LT, EQ, or GT) from std::cmp::Ordering.
    fn make_ordering(&self, ord: std::cmp::Ordering) -> Value {
        use bhc_types::{Kind, TyCon};
        let (name, tag) = match ord {
            std::cmp::Ordering::Less => ("LT", 0),
            std::cmp::Ordering::Equal => ("EQ", 1),
            std::cmp::Ordering::Greater => ("GT", 2),
        };
        Value::Data(DataValue {
            con: crate::DataCon {
                name: Symbol::intern(name),
                ty_con: TyCon::new(Symbol::intern("Ordering"), Kind::Star),
                tag,
                arity: 0,
            },
            args: Vec::new(),
        })
    }

    /// Display a value, deeply forcing all thunks.
    ///
    /// This produces a readable string representation of a value,
    /// recursively forcing any thunks encountered.
    pub fn display_value(&self, value: &Value) -> Result<String, EvalError> {
        self.display_value_impl(value, 0)
    }

    fn display_value_impl(&self, value: &Value, depth: usize) -> Result<String, EvalError> {
        // Prevent infinite recursion
        if depth > 1000 {
            return Ok("<deep>".to_string());
        }

        // Force thunks first
        let forced = match value {
            Value::Thunk(thunk) => self.force_thunk(thunk)?,
            v => v.clone(),
        };

        match &forced {
            Value::Int(n) => Ok(n.to_string()),
            Value::Integer(n) => Ok(n.to_string()),
            Value::Float(n) => Ok(format!("{n}")),
            Value::Double(n) => Ok(format!("{n}")),
            Value::Char(c) => Ok(format!("{c:?}")),
            Value::String(s) => Ok(format!("{s:?}")),
            Value::Closure(c) => Ok(format!("<function {}>", c.var.name)),
            Value::PrimOp(op) => Ok(format!("<primop {op:?}>")),
            Value::PartialPrimOp(op, args) => {
                Ok(format!("<partial {op:?} applied to {} args>", args.len()))
            }
            Value::UArrayInt(arr) => {
                let elements: Vec<String> = arr.as_slice().iter().map(|n| n.to_string()).collect();
                Ok(format!("[{}]", elements.join(", ")))
            }
            Value::UArrayDouble(arr) => {
                let elements: Vec<String> = arr.as_slice().iter().map(|n| format!("{n}")).collect();
                Ok(format!("[{}]", elements.join(", ")))
            }
            Value::Data(d) => {
                let name = d.con.name.as_str();
                // Special case for lists
                if name == "[]" {
                    return Ok("[]".to_string());
                }
                if name == ":" {
                    // It's a list - collect all elements
                    let list = self.force_list(forced.clone())?;
                    let elements: Result<Vec<String>, _> = list
                        .iter()
                        .map(|v| self.display_value_impl(v, depth + 1))
                        .collect();
                    return Ok(format!("[{}]", elements?.join(", ")));
                }
                // Special case for tuples
                if name.starts_with('(') && name.ends_with(')') && name.contains(',') {
                    let elements: Result<Vec<String>, _> = d
                        .args
                        .iter()
                        .map(|v| self.display_value_impl(v, depth + 1))
                        .collect();
                    return Ok(format!("({})", elements?.join(", ")));
                }
                // General data constructor
                if d.args.is_empty() {
                    Ok(name.to_string())
                } else {
                    let args: Result<Vec<String>, _> = d
                        .args
                        .iter()
                        .map(|v| self.display_value_impl(v, depth + 1))
                        .collect();
                    Ok(format!("{} {}", name, args?.join(" ")))
                }
            }
            Value::Map(m) => {
                let entries: Result<Vec<String>, _> = m
                    .iter()
                    .map(|(k, v)| {
                        let ks = self.display_value_impl(&k.0, depth + 1)?;
                        let vs = self.display_value_impl(v, depth + 1)?;
                        Ok(format!("({}, {})", ks, vs))
                    })
                    .collect();
                Ok(format!("fromList [{}]", entries?.join(", ")))
            }
            Value::Set(s) => {
                let entries: Result<Vec<String>, _> = s
                    .iter()
                    .map(|v| self.display_value_impl(&v.0, depth + 1))
                    .collect();
                Ok(format!("fromList [{}]", entries?.join(", ")))
            }
            Value::IntMap(m) => {
                let entries: Vec<String> = m
                    .iter()
                    .map(|(k, v)| {
                        let vs = self.display_value_impl(v, depth + 1).unwrap_or_else(|_| "<error>".to_string());
                        format!("({}, {})", k, vs)
                    })
                    .collect();
                Ok(format!("fromList [{}]", entries.join(", ")))
            }
            Value::IntSet(s) => {
                let entries: Vec<String> = s.iter().map(|v| v.to_string()).collect();
                Ok(format!("fromList [{}]", entries.join(", ")))
            }
            Value::Handle(h) => Ok(format!("<handle {:?}>", h.kind)),
            Value::IORef(_) => Ok("<IORef>".to_string()),
            Value::Thunk(_) => {
                // Should have been forced above, but just in case
                Ok("<thunk>".to_string())
            }
        }
    }

    /// Parses a dictionary selector name like "$sel_0", "$sel_1", etc.
    /// Returns the field index if the name matches the pattern.
    fn parse_selector_name(name: &str) -> Option<usize> {
        name.strip_prefix("$sel_")
            .and_then(|n| n.parse::<usize>().ok())
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new(EvalMode::Lazy)
    }
}

/// Generate the next permutation of indices in lexicographic order.
/// Returns false when all permutations have been exhausted.
fn next_permutation(indices: &mut Vec<usize>) -> bool {
    let n = indices.len();
    if n <= 1 {
        return false;
    }
    // Find largest i such that indices[i] < indices[i+1]
    let mut i = n - 2;
    loop {
        if indices[i] < indices[i + 1] {
            break;
        }
        if i == 0 {
            return false;
        }
        i -= 1;
    }
    // Find largest j such that indices[i] < indices[j]
    let mut j = n - 1;
    while indices[j] <= indices[i] {
        j -= 1;
    }
    indices.swap(i, j);
    indices[i + 1..].reverse();
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_span::Span;
    use bhc_types::Ty;

    fn make_var(name: &str, id: usize) -> Var {
        Var::new(Symbol::intern(name), VarId::new(id), Ty::Error)
    }

    fn make_int(n: i64) -> Expr {
        Expr::Lit(Literal::Int(n), Ty::Error, Span::default())
    }

    #[test]
    fn test_eval_literal() {
        let eval = Evaluator::new(EvalMode::Strict);
        let expr = make_int(42);
        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_eval_lambda_app() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (\x -> x) 42
        let x = make_var("x", 0);
        let lam = Expr::Lam(
            x.clone(),
            Box::new(Expr::Var(x, Span::default())),
            Span::default(),
        );
        let app = Expr::App(Box::new(lam), Box::new(make_int(42)), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_eval_let() {
        let eval = Evaluator::new(EvalMode::Strict);

        // let x = 42 in x
        let x = make_var("x", 0);
        let bind = Bind::NonRec(x.clone(), Box::new(make_int(42)));
        let body = Expr::Var(x, Span::default());
        let expr = Expr::Let(Box::new(bind), Box::new(body), Span::default());

        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_eval_primop_add() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (+) 1 2
        let add = Expr::Var(make_var("+", 100), Span::default());
        let app1 = Expr::App(Box::new(add), Box::new(make_int(1)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(2)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(3)));
    }

    #[test]
    fn test_eval_primop_mul() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (*) 6 7
        let mul = Expr::Var(make_var("*", 100), Span::default());
        let app1 = Expr::App(Box::new(mul), Box::new(make_int(6)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(7)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_eval_nested_let() {
        let eval = Evaluator::new(EvalMode::Strict);

        // let x = 1 in let y = 2 in x + y
        let x = make_var("x", 0);
        let y = make_var("y", 1);
        let add = Expr::Var(make_var("+", 100), Span::default());

        let add_xy = Expr::App(
            Box::new(Expr::App(
                Box::new(add),
                Box::new(Expr::Var(x.clone(), Span::default())),
                Span::default(),
            )),
            Box::new(Expr::Var(y.clone(), Span::default())),
            Span::default(),
        );

        let inner_let = Expr::Let(
            Box::new(Bind::NonRec(y, Box::new(make_int(2)))),
            Box::new(add_xy),
            Span::default(),
        );

        let outer_let = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(make_int(1)))),
            Box::new(inner_let),
            Span::default(),
        );

        let result = eval.eval(&outer_let, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(3)));
    }

    #[test]
    fn test_eval_comparison() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (<) 1 2
        let lt = Expr::Var(make_var("<", 100), Span::default());
        let app1 = Expr::App(Box::new(lt), Box::new(make_int(1)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(2)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_lazy_evaluation() {
        let eval = Evaluator::new(EvalMode::Lazy);

        // let x = error "boom" in 42
        // In lazy mode, x is never forced so no error
        let x = make_var("x", 0);
        let error_call = Expr::App(
            Box::new(Expr::Var(make_var("error", 100), Span::default())),
            Box::new(Expr::Lit(
                Literal::String(Symbol::intern("boom")),
                Ty::Error,
                Span::default(),
            )),
            Span::default(),
        );

        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(error_call))),
            Box::new(make_int(42)),
            Span::default(),
        );

        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_list_operations() {
        // Test creating and inspecting lists
        let list = Value::from_list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);

        let items = list.as_list().unwrap();
        assert_eq!(items.len(), 3);
        assert!(matches!(items[0], Value::Int(1)));
        assert!(matches!(items[1], Value::Int(2)));
        assert!(matches!(items[2], Value::Int(3)));
    }

    // =========================================================================
    // UArray Tests - M0 Exit Criteria
    // =========================================================================

    #[test]
    fn test_uarray_from_list() {
        // Create a list [1, 2, 3, 4, 5]
        let list = Value::from_list(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4),
            Value::Int(5),
        ]);

        // Convert to UArray
        let result = Value::uarray_int_from_list(&list).unwrap();
        match result {
            Value::UArrayInt(arr) => {
                assert_eq!(arr.len(), 5);
                assert_eq!(arr.to_vec(), vec![1, 2, 3, 4, 5]);
            }
            _ => panic!("Expected UArrayInt"),
        }
    }

    #[test]
    fn test_uarray_sum() {
        let eval = Evaluator::new(EvalMode::Strict);
        let sum_var = Expr::Var(make_var("sum", 100), Span::default());

        // Build: sum [1, 2, 3, 4, 5]
        // First, create a let binding for the list
        let list_var = make_var("xs", 0);
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);
        let sum_app = Expr::App(
            Box::new(sum_var),
            Box::new(Expr::Var(list_var.clone(), Span::default())),
            Span::default(),
        );
        let expr = Expr::Let(
            Box::new(Bind::NonRec(list_var, Box::new(list_expr))),
            Box::new(sum_app),
            Span::default(),
        );

        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(15)));
    }

    #[test]
    fn test_uarray_map() {
        let eval = Evaluator::new(EvalMode::Strict);

        // map (+1) [1, 2, 3, 4, 5]
        let add_one = {
            let x = make_var("x", 0);
            let add = Expr::Var(make_var("+", 100), Span::default());
            Expr::Lam(
                x.clone(),
                Box::new(Expr::App(
                    Box::new(Expr::App(
                        Box::new(add),
                        Box::new(Expr::Var(x, Span::default())),
                        Span::default(),
                    )),
                    Box::new(make_int(1)),
                    Span::default(),
                )),
                Span::default(),
            )
        };

        let map_var = Expr::Var(make_var("uarrayMap", 101), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);

        // Build: uarrayMap (+1) [1, 2, 3, 4, 5]
        let app1 = Expr::App(Box::new(map_var), Box::new(add_one), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(list_expr), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();

        // Should be a list [2, 3, 4, 5, 6]
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 5);
        assert_eq!(list[0].as_int(), Some(2));
        assert_eq!(list[1].as_int(), Some(3));
        assert_eq!(list[2].as_int(), Some(4));
        assert_eq!(list[3].as_int(), Some(5));
        assert_eq!(list[4].as_int(), Some(6));
    }

    #[test]
    fn test_m0_exit_criteria_sum_map() {
        // M0 Exit Criteria: sum (map (+1) [1,2,3,4,5]) == 20
        let eval = Evaluator::new(EvalMode::Strict);

        // Build: sum (map (+1) [1, 2, 3, 4, 5])
        let add_one = {
            let x = make_var("x", 0);
            let add = Expr::Var(make_var("+", 100), Span::default());
            Expr::Lam(
                x.clone(),
                Box::new(Expr::App(
                    Box::new(Expr::App(
                        Box::new(add),
                        Box::new(Expr::Var(x, Span::default())),
                        Span::default(),
                    )),
                    Box::new(make_int(1)),
                    Span::default(),
                )),
                Span::default(),
            )
        };

        let map_var = Expr::Var(make_var("uarrayMap", 101), Span::default());
        let sum_var = Expr::Var(make_var("sum", 102), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);

        // map (+1) [1, 2, 3, 4, 5]
        let map_app = Expr::App(
            Box::new(Expr::App(
                Box::new(map_var),
                Box::new(add_one),
                Span::default(),
            )),
            Box::new(list_expr),
            Span::default(),
        );

        // sum (map (+1) [1, 2, 3, 4, 5])
        let sum_app = Expr::App(Box::new(sum_var), Box::new(map_app), Span::default());

        let result = eval.eval(&sum_app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(20)));
    }

    #[test]
    fn test_uarray_zipwith() {
        let eval = Evaluator::new(EvalMode::Strict);

        // zipWith (+) [1, 2, 3] [4, 5, 6]
        let add = Expr::Var(make_var("+", 100), Span::default());
        let zip_var = Expr::Var(make_var("uarrayZipWith", 101), Span::default());
        let list1 = build_list_expr(vec![1, 2, 3]);
        let list2 = build_list_expr(vec![4, 5, 6]);

        // zipWith (+) [1, 2, 3] [4, 5, 6]
        let app1 = Expr::App(Box::new(zip_var), Box::new(add), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(list1), Span::default());
        let app3 = Expr::App(Box::new(app2), Box::new(list2), Span::default());

        let result = eval.eval(&app3, &Env::new()).unwrap();

        // Should be [5, 7, 9]
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].as_int(), Some(5));
        assert_eq!(list[1].as_int(), Some(7));
        assert_eq!(list[2].as_int(), Some(9));
    }

    #[test]
    fn test_uarray_dot_product() {
        // Test dot product: sum (zipWith (*) [1, 2, 3] [4, 5, 6])
        // = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let eval = Evaluator::new(EvalMode::Strict);

        let mul = Expr::Var(make_var("*", 100), Span::default());
        let zip_var = Expr::Var(make_var("uarrayZipWith", 101), Span::default());
        let sum_var = Expr::Var(make_var("sum", 102), Span::default());
        let list1 = build_list_expr(vec![1, 2, 3]);
        let list2 = build_list_expr(vec![4, 5, 6]);

        // zipWith (*) [1, 2, 3] [4, 5, 6]
        let zip_app = Expr::App(
            Box::new(Expr::App(
                Box::new(Expr::App(Box::new(zip_var), Box::new(mul), Span::default())),
                Box::new(list1),
                Span::default(),
            )),
            Box::new(list2),
            Span::default(),
        );

        // sum (zipWith (*) [1, 2, 3] [4, 5, 6])
        let sum_app = Expr::App(Box::new(sum_var), Box::new(zip_app), Span::default());

        let result = eval.eval(&sum_app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(32)));
    }

    // Helper function to build a list expression
    fn build_list_expr(values: Vec<i64>) -> Expr {
        // Start with nil
        let mut expr = Expr::Var(
            Var::new(Symbol::intern("[]"), VarId::new(9999), Ty::Error),
            Span::default(),
        );

        // Build from the end: (:) x ((:) y nil)
        for val in values.into_iter().rev() {
            let cons_var = Expr::Var(
                Var::new(Symbol::intern(":"), VarId::new(9998), Ty::Error),
                Span::default(),
            );
            let val_expr = make_int(val);
            expr = Expr::App(
                Box::new(Expr::App(
                    Box::new(cons_var),
                    Box::new(val_expr),
                    Span::default(),
                )),
                Box::new(expr),
                Span::default(),
            );
        }

        expr
    }

    // =========================================================================
    // M1 Tests - Numeric Profile Strict-by-Default
    // =========================================================================

    #[test]
    fn test_eval_mode_from_profile() {
        // Default and Server profiles use lazy evaluation
        assert_eq!(EvalMode::from(Profile::Default), EvalMode::Lazy);
        assert_eq!(EvalMode::from(Profile::Server), EvalMode::Lazy);

        // Numeric and Edge profiles use strict evaluation
        assert_eq!(EvalMode::from(Profile::Numeric), EvalMode::Strict);
        assert_eq!(EvalMode::from(Profile::Edge), EvalMode::Strict);
    }

    #[test]
    fn test_evaluator_with_profile() {
        let lazy_eval = Evaluator::with_profile(Profile::Default);
        assert!(!lazy_eval.is_strict());
        assert_eq!(lazy_eval.mode(), EvalMode::Lazy);

        let strict_eval = Evaluator::with_profile(Profile::Numeric);
        assert!(strict_eval.is_strict());
        assert_eq!(strict_eval.mode(), EvalMode::Strict);
    }

    #[test]
    fn test_strict_evaluation_forces_let_bindings() {
        // In strict mode, let bindings are eagerly evaluated
        // let x = 1 + 2 in x * 3
        // This is a key M1 exit criterion: strict evaluation of let-bindings

        let eval = Evaluator::with_profile(Profile::Numeric);

        let x = make_var("x", 0);
        let add = Expr::Var(make_var("+", 100), Span::default());
        let mul = Expr::Var(make_var("*", 101), Span::default());

        // 1 + 2
        let add_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(add),
                Box::new(make_int(1)),
                Span::default(),
            )),
            Box::new(make_int(2)),
            Span::default(),
        );

        // x * 3
        let mul_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(mul),
                Box::new(Expr::Var(x.clone(), Span::default())),
                Span::default(),
            )),
            Box::new(make_int(3)),
            Span::default(),
        );

        // let x = 1 + 2 in x * 3
        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(add_expr))),
            Box::new(mul_expr),
            Span::default(),
        );

        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(9))); // (1 + 2) * 3 = 9
    }

    #[test]
    fn test_strict_mode_evaluates_unused_bindings() {
        // In strict mode, error in binding is raised even if binding is unused
        // let x = error "boom" in 42 -> error in strict mode
        let eval = Evaluator::with_profile(Profile::Numeric);

        let x = make_var("x", 0);
        let error_call = Expr::App(
            Box::new(Expr::Var(make_var("error", 100), Span::default())),
            Box::new(Expr::Lit(
                Literal::String(Symbol::intern("boom")),
                Ty::Error,
                Span::default(),
            )),
            Span::default(),
        );

        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(error_call))),
            Box::new(make_int(42)),
            Span::default(),
        );

        // In strict mode, the error binding is evaluated even though unused
        let result = eval.eval(&expr, &Env::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_escape_hatch_in_strict_mode() {
        // The lazy { } escape hatch should make evaluation lazy even in strict mode
        // let x = lazy { error "boom" } in 42 -> 42 (thunk not forced)
        let eval = Evaluator::with_profile(Profile::Numeric);
        assert!(eval.is_strict());

        let x = make_var("x", 0);
        let error_call = Expr::App(
            Box::new(Expr::Var(make_var("error", 100), Span::default())),
            Box::new(Expr::Lit(
                Literal::String(Symbol::intern("boom")),
                Ty::Error,
                Span::default(),
            )),
            Span::default(),
        );

        // Wrap the error in lazy { }
        let lazy_error = Expr::Lazy(Box::new(error_call), Span::default());

        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(lazy_error))),
            Box::new(make_int(42)),
            Span::default(),
        );

        // With lazy escape hatch, the error is not evaluated
        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_lazy_escape_hatch_creates_thunk() {
        // lazy { expr } should create a thunk
        let eval = Evaluator::with_profile(Profile::Numeric);

        let lazy_expr = Expr::Lazy(Box::new(make_int(42)), Span::default());
        let result = eval.eval(&lazy_expr, &Env::new()).unwrap();

        // The result should be a thunk
        assert!(result.is_thunk());

        // When forced, it should give 42
        let forced = eval.force(result).unwrap();
        assert!(matches!(forced, Value::Int(42)));
    }

    #[test]
    fn test_m1_exit_criteria_strict_let_binding() {
        // M1 Exit Criterion: strict evaluation of `let x = 1 + 2 in x * 3`
        // evaluates to 9 with no thunks in Numeric Profile
        let eval = Evaluator::with_profile(Profile::Numeric);

        let x = make_var("x", 0);
        let add = Expr::Var(make_var("+", 100), Span::default());
        let mul = Expr::Var(make_var("*", 101), Span::default());

        // 1 + 2
        let add_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(add),
                Box::new(make_int(1)),
                Span::default(),
            )),
            Box::new(make_int(2)),
            Span::default(),
        );

        // x * 3
        let mul_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(mul),
                Box::new(Expr::Var(x.clone(), Span::default())),
                Span::default(),
            )),
            Box::new(make_int(3)),
            Span::default(),
        );

        // let x = 1 + 2 in x * 3
        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(add_expr))),
            Box::new(mul_expr),
            Span::default(),
        );

        let result = eval.eval(&expr, &Env::new()).unwrap();

        // Result should be 9
        assert!(matches!(result, Value::Int(9)));

        // Result should not be a thunk (no thunks in strict mode result)
        assert!(!result.is_thunk());
    }

    // =========================================================================
    // Additional Primop Tests
    // =========================================================================

    #[test]
    fn test_eval_primop_sub() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (-) 10 3
        let sub = Expr::Var(make_var("-", 100), Span::default());
        let app1 = Expr::App(Box::new(sub), Box::new(make_int(10)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(3)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(7)));
    }

    #[test]
    fn test_eval_primop_div() {
        let eval = Evaluator::new(EvalMode::Strict);

        // div 10 3
        let div = Expr::Var(make_var("div", 100), Span::default());
        let app1 = Expr::App(Box::new(div), Box::new(make_int(10)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(3)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(3)));
    }

    #[test]
    fn test_eval_primop_mod() {
        let eval = Evaluator::new(EvalMode::Strict);

        // mod 10 3
        let modop = Expr::Var(make_var("mod", 100), Span::default());
        let app1 = Expr::App(Box::new(modop), Box::new(make_int(10)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(3)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(1)));
    }

    #[test]
    fn test_eval_primop_negate() {
        let eval = Evaluator::new(EvalMode::Strict);

        // negate 42
        let neg = Expr::Var(make_var("negate", 100), Span::default());
        let app = Expr::App(Box::new(neg), Box::new(make_int(42)), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(-42)));
    }

    #[test]
    fn test_eval_div_by_zero() {
        let eval = Evaluator::new(EvalMode::Strict);

        // div 10 0
        let div = Expr::Var(make_var("div", 100), Span::default());
        let app1 = Expr::App(Box::new(div), Box::new(make_int(10)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(0)), Span::default());

        let result = eval.eval(&app2, &Env::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_eval_comparison_eq() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (==) 42 42
        let eq = Expr::Var(make_var("==", 100), Span::default());
        let app1 = Expr::App(Box::new(eq), Box::new(make_int(42)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(42)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_comparison_neq() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (==) 42 43
        let eq = Expr::Var(make_var("==", 100), Span::default());
        let app1 = Expr::App(Box::new(eq), Box::new(make_int(42)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(43)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(false));
    }

    #[test]
    fn test_eval_comparison_le() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (<=) 3 5
        let le = Expr::Var(make_var("<=", 100), Span::default());
        let app1 = Expr::App(Box::new(le), Box::new(make_int(3)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(5)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));

        // (<=) 5 5 (equal case)
        let le2 = Expr::Var(make_var("<=", 101), Span::default());
        let app3 = Expr::App(Box::new(le2), Box::new(make_int(5)), Span::default());
        let app4 = Expr::App(Box::new(app3), Box::new(make_int(5)), Span::default());

        let result2 = eval.eval(&app4, &Env::new()).unwrap();
        assert_eq!(result2.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_comparison_gt() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (>) 5 3
        let gt = Expr::Var(make_var(">", 100), Span::default());
        let app1 = Expr::App(Box::new(gt), Box::new(make_int(5)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(3)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_comparison_ge() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (>=) 5 5
        let ge = Expr::Var(make_var(">=", 100), Span::default());
        let app1 = Expr::App(Box::new(ge), Box::new(make_int(5)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(5)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_boolean_and() {
        let eval = Evaluator::new(EvalMode::Strict);

        // Create boolean values via comparison: (1 < 2) && (3 < 4) == true
        let lt1 = Expr::Var(make_var("<", 100), Span::default());
        let lt2 = Expr::Var(make_var("<", 101), Span::default());
        let and_op = Expr::Var(make_var("&&", 102), Span::default());

        // (1 < 2) = true
        let cmp1 = Expr::App(
            Box::new(Expr::App(
                Box::new(lt1),
                Box::new(make_int(1)),
                Span::default(),
            )),
            Box::new(make_int(2)),
            Span::default(),
        );

        // (3 < 4) = true
        let cmp2 = Expr::App(
            Box::new(Expr::App(
                Box::new(lt2),
                Box::new(make_int(3)),
                Span::default(),
            )),
            Box::new(make_int(4)),
            Span::default(),
        );

        // true && true
        let app1 = Expr::App(Box::new(and_op), Box::new(cmp1), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(cmp2), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_boolean_or() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (1 > 2) || (3 < 4) = false || true = true
        let gt = Expr::Var(make_var(">", 100), Span::default());
        let lt = Expr::Var(make_var("<", 101), Span::default());
        let or_op = Expr::Var(make_var("||", 102), Span::default());

        // (1 > 2) = false
        let cmp1 = Expr::App(
            Box::new(Expr::App(
                Box::new(gt),
                Box::new(make_int(1)),
                Span::default(),
            )),
            Box::new(make_int(2)),
            Span::default(),
        );

        // (3 < 4) = true
        let cmp2 = Expr::App(
            Box::new(Expr::App(
                Box::new(lt),
                Box::new(make_int(3)),
                Span::default(),
            )),
            Box::new(make_int(4)),
            Span::default(),
        );

        // false || true
        let app1 = Expr::App(Box::new(or_op), Box::new(cmp1), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(cmp2), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_boolean_not() {
        let eval = Evaluator::new(EvalMode::Strict);

        // not (1 > 2) = not false = true
        let gt = Expr::Var(make_var(">", 100), Span::default());
        let not_op = Expr::Var(make_var("not", 101), Span::default());

        // (1 > 2) = false
        let cmp = Expr::App(
            Box::new(Expr::App(
                Box::new(gt),
                Box::new(make_int(1)),
                Span::default(),
            )),
            Box::new(make_int(2)),
            Span::default(),
        );

        // not false
        let app = Expr::App(Box::new(not_op), Box::new(cmp), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));
    }

    #[test]
    fn test_eval_seq() {
        let eval = Evaluator::new(EvalMode::Lazy);

        // seq 1 42 = 42 (forces first arg, returns second)
        let seq_op = Expr::Var(make_var("seq", 100), Span::default());
        let app1 = Expr::App(Box::new(seq_op), Box::new(make_int(1)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(42)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn test_eval_nested_arithmetic() {
        let eval = Evaluator::new(EvalMode::Strict);

        // ((1 + 2) * 3) - 4 = 9 - 4 = 5
        let add = Expr::Var(make_var("+", 100), Span::default());
        let mul = Expr::Var(make_var("*", 101), Span::default());
        let sub = Expr::Var(make_var("-", 102), Span::default());

        // 1 + 2
        let add_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(add),
                Box::new(make_int(1)),
                Span::default(),
            )),
            Box::new(make_int(2)),
            Span::default(),
        );

        // (1 + 2) * 3
        let mul_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(mul),
                Box::new(add_expr),
                Span::default(),
            )),
            Box::new(make_int(3)),
            Span::default(),
        );

        // ((1 + 2) * 3) - 4
        let sub_expr = Expr::App(
            Box::new(Expr::App(
                Box::new(sub),
                Box::new(mul_expr),
                Span::default(),
            )),
            Box::new(make_int(4)),
            Span::default(),
        );

        let result = eval.eval(&sub_expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(5)));
    }

    #[test]
    fn test_eval_list_head() {
        let eval = Evaluator::new(EvalMode::Strict);

        // head [1, 2, 3] = 1
        let head_var = Expr::Var(make_var("head", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3]);
        let app = Expr::App(Box::new(head_var), Box::new(list_expr), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(1)));
    }

    #[test]
    fn test_eval_list_tail() {
        let eval = Evaluator::new(EvalMode::Strict);

        // tail [1, 2, 3] = [2, 3]
        let tail_var = Expr::Var(make_var("tail", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3]);
        let app = Expr::App(Box::new(tail_var), Box::new(list_expr), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].as_int(), Some(2));
        assert_eq!(list[1].as_int(), Some(3));
    }

    #[test]
    fn test_eval_list_length() {
        let eval = Evaluator::new(EvalMode::Strict);

        // length [1, 2, 3, 4, 5] = 5
        let length_var = Expr::Var(make_var("length", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);
        let app = Expr::App(Box::new(length_var), Box::new(list_expr), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(5)));
    }

    #[test]
    fn test_eval_list_null() {
        let eval = Evaluator::new(EvalMode::Strict);

        // null [] = true
        let null_var = Expr::Var(make_var("null", 100), Span::default());
        let nil = Expr::Var(
            Var::new(Symbol::intern("[]"), VarId::new(9999), Ty::Error),
            Span::default(),
        );
        let app = Expr::App(Box::new(null_var), Box::new(nil), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        assert_eq!(result.as_bool(), Some(true));

        // null [1] = false
        let null_var2 = Expr::Var(make_var("null", 101), Span::default());
        let list_expr = build_list_expr(vec![1]);
        let app2 = Expr::App(Box::new(null_var2), Box::new(list_expr), Span::default());

        let result2 = eval.eval(&app2, &Env::new()).unwrap();
        assert_eq!(result2.as_bool(), Some(false));
    }

    #[test]
    fn test_eval_list_reverse() {
        let eval = Evaluator::new(EvalMode::Strict);

        // reverse [1, 2, 3] = [3, 2, 1]
        let reverse_var = Expr::Var(make_var("reverse", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3]);
        let app = Expr::App(Box::new(reverse_var), Box::new(list_expr), Span::default());

        let result = eval.eval(&app, &Env::new()).unwrap();
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].as_int(), Some(3));
        assert_eq!(list[1].as_int(), Some(2));
        assert_eq!(list[2].as_int(), Some(1));
    }

    #[test]
    fn test_eval_list_take() {
        let eval = Evaluator::new(EvalMode::Strict);

        // take 2 [1, 2, 3, 4, 5] = [1, 2]
        let take_var = Expr::Var(make_var("take", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);
        let app1 = Expr::App(Box::new(take_var), Box::new(make_int(2)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(list_expr), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].as_int(), Some(1));
        assert_eq!(list[1].as_int(), Some(2));
    }

    #[test]
    fn test_eval_list_drop() {
        let eval = Evaluator::new(EvalMode::Strict);

        // drop 2 [1, 2, 3, 4, 5] = [3, 4, 5]
        let drop_var = Expr::Var(make_var("drop", 100), Span::default());
        let list_expr = build_list_expr(vec![1, 2, 3, 4, 5]);
        let app1 = Expr::App(Box::new(drop_var), Box::new(make_int(2)), Span::default());
        let app2 = Expr::App(Box::new(app1), Box::new(list_expr), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        let list = result.as_list().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].as_int(), Some(3));
        assert_eq!(list[1].as_int(), Some(4));
        assert_eq!(list[2].as_int(), Some(5));
    }

    #[test]
    fn test_eval_multiple_lambdas() {
        let eval = Evaluator::new(EvalMode::Strict);

        // (\x -> \y -> x + y) 3 4 = 7
        let x = make_var("x", 0);
        let y = make_var("y", 1);
        let add = Expr::Var(make_var("+", 100), Span::default());

        // x + y
        let add_xy = Expr::App(
            Box::new(Expr::App(
                Box::new(add),
                Box::new(Expr::Var(x.clone(), Span::default())),
                Span::default(),
            )),
            Box::new(Expr::Var(y.clone(), Span::default())),
            Span::default(),
        );

        // \y -> x + y
        let inner_lam = Expr::Lam(y, Box::new(add_xy), Span::default());

        // \x -> \y -> x + y
        let outer_lam = Expr::Lam(x, Box::new(inner_lam), Span::default());

        // (\x -> \y -> x + y) 3
        let app1 = Expr::App(Box::new(outer_lam), Box::new(make_int(3)), Span::default());

        // (\x -> \y -> x + y) 3 4
        let app2 = Expr::App(Box::new(app1), Box::new(make_int(4)), Span::default());

        let result = eval.eval(&app2, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(7)));
    }

    #[test]
    fn test_eval_closure_capture() {
        let eval = Evaluator::new(EvalMode::Strict);

        // let x = 10 in (\y -> x + y) 5 = 15
        let x = make_var("x", 0);
        let y = make_var("y", 1);
        let add = Expr::Var(make_var("+", 100), Span::default());

        // x + y
        let add_xy = Expr::App(
            Box::new(Expr::App(
                Box::new(add),
                Box::new(Expr::Var(x.clone(), Span::default())),
                Span::default(),
            )),
            Box::new(Expr::Var(y.clone(), Span::default())),
            Span::default(),
        );

        // \y -> x + y
        let lam = Expr::Lam(y, Box::new(add_xy), Span::default());

        // (\y -> x + y) 5
        let app = Expr::App(Box::new(lam), Box::new(make_int(5)), Span::default());

        // let x = 10 in (\y -> x + y) 5
        let expr = Expr::Let(
            Box::new(Bind::NonRec(x, Box::new(make_int(10)))),
            Box::new(app),
            Span::default(),
        );

        let result = eval.eval(&expr, &Env::new()).unwrap();
        assert!(matches!(result, Value::Int(15)));
    }
}
