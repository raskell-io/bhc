//! Constraint solver for type-level naturals.
//!
//! This module implements a solver for constraints on type-level natural numbers,
//! enabling compile-time verification of tensor dimension relationships.
//!
//! ## Overview
//!
//! During type checking of shape-indexed tensors, we collect constraints like:
//!
//! ```text
//! k1 = k2           -- Inner dimensions must match in matmul
//! m = 1024          -- Concrete dimension assignment
//! n + 1 = m         -- Arithmetic relationship
//! ```
//!
//! The solver attempts to find a substitution that satisfies all constraints,
//! or reports which constraints are unsatisfiable.
//!
//! ## Algorithm
//!
//! The solver uses a union-find based approach with arithmetic simplification:
//!
//! 1. **Normalization**: Convert constraints to canonical form
//! 2. **Propagation**: Substitute known values throughout
//! 3. **Simplification**: Evaluate arithmetic where possible
//! 4. **Unification**: Merge equivalent variables
//! 5. **Consistency check**: Detect contradictions
//!
//! ## Limitations
//!
//! The solver handles:
//! - Variable-to-variable equality (`m = n`)
//! - Variable-to-literal equality (`m = 1024`)
//! - Simple arithmetic with literals (`m + 1 = 5` → `m = 4`)
//!
//! It does NOT solve:
//! - Non-linear constraints (`m * n = 6`)
//! - Inequalities (`m < n`)
//! - Division or modulo

use bhc_types::{nat::TyNat, TyVar};
use rustc_hash::FxHashMap;

/// A constraint on type-level naturals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NatConstraint {
    /// Equality constraint: `left = right`.
    Equal(TyNat, TyNat),
}

impl NatConstraint {
    /// Creates an equality constraint.
    #[must_use]
    pub fn equal(left: TyNat, right: TyNat) -> Self {
        Self::Equal(left, right)
    }

    /// Applies a substitution to this constraint.
    #[must_use]
    pub fn apply_subst(&self, subst: &NatSubst) -> Self {
        match self {
            Self::Equal(left, right) => Self::Equal(subst.apply(left), subst.apply(right)),
        }
    }
}

impl std::fmt::Display for NatConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Equal(left, right) => write!(f, "{} = {}", left, right),
        }
    }
}

/// A substitution mapping type variables to type-level naturals.
#[derive(Clone, Debug, Default)]
pub struct NatSubst {
    /// The mapping from variable IDs to their substituted values.
    mapping: FxHashMap<u32, TyNat>,
}

impl NatSubst {
    /// Creates an empty substitution.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds a variable to a value.
    ///
    /// Returns `false` if the variable is already bound to a different value.
    pub fn bind(&mut self, var: &TyVar, value: TyNat) -> bool {
        if let Some(existing) = self.mapping.get(&var.id) {
            // Already bound - check if consistent
            existing == &value
        } else {
            self.mapping.insert(var.id, value);
            true
        }
    }

    /// Looks up the value bound to a variable.
    #[must_use]
    pub fn lookup(&self, var: &TyVar) -> Option<&TyNat> {
        self.mapping.get(&var.id)
    }

    /// Applies this substitution to a type-level natural.
    #[must_use]
    pub fn apply(&self, nat: &TyNat) -> TyNat {
        match nat {
            TyNat::Lit(_) => nat.clone(),
            TyNat::Var(v) => {
                if let Some(value) = self.mapping.get(&v.id) {
                    // Recursively apply in case of transitive bindings
                    self.apply(value)
                } else {
                    nat.clone()
                }
            }
            TyNat::Add(left, right) => {
                let left = self.apply(left);
                let right = self.apply(right);
                TyNat::add(left, right) // Uses smart constructor
            }
            TyNat::Mul(left, right) => {
                let left = self.apply(left);
                let right = self.apply(right);
                TyNat::mul(left, right) // Uses smart constructor
            }
        }
    }

    /// Returns the number of bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    /// Returns true if there are no bindings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

    /// Composes this substitution with another.
    ///
    /// The result applies `self` first, then `other`.
    #[must_use]
    pub fn compose(&self, other: &NatSubst) -> NatSubst {
        let mut result = NatSubst::new();

        // Apply other to all values in self
        for (&var_id, value) in &self.mapping {
            result.mapping.insert(var_id, other.apply(value));
        }

        // Add bindings from other that aren't in self
        for (&var_id, value) in &other.mapping {
            result
                .mapping
                .entry(var_id)
                .or_insert_with(|| value.clone());
        }

        result
    }

    /// Returns an iterator over the bindings.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &TyNat)> {
        self.mapping.iter().map(|(&k, v)| (k, v))
    }
}

/// Errors that can occur during constraint solving.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SolverError {
    /// Two literals don't match.
    LiteralMismatch {
        /// The expected value.
        expected: u64,
        /// The actual value.
        found: u64,
    },
    /// An occurs check failed (variable appears in its own definition).
    OccursCheck {
        /// The variable ID.
        var_id: u32,
        /// The term containing the variable.
        term: TyNat,
    },
    /// Constraints are inconsistent.
    Inconsistent {
        /// Description of the inconsistency.
        message: String,
    },
    /// Cannot solve constraint (e.g., non-linear).
    CannotSolve {
        /// The constraint that couldn't be solved.
        constraint: NatConstraint,
    },
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LiteralMismatch { expected, found } => {
                write!(
                    f,
                    "dimension mismatch: expected {}, found {}",
                    expected, found
                )
            }
            Self::OccursCheck { var_id, term } => {
                write!(f, "infinite type: n{} occurs in {}", var_id, term)
            }
            Self::Inconsistent { message } => {
                write!(f, "inconsistent constraints: {}", message)
            }
            Self::CannotSolve { constraint } => {
                write!(f, "cannot solve constraint: {}", constraint)
            }
        }
    }
}

impl std::error::Error for SolverError {}

/// Result of constraint solving.
pub type SolverResult<T> = Result<T, SolverError>;

/// A constraint solver for type-level naturals.
///
/// # Example
///
/// ```ignore
/// use bhc_typeck::nat_solver::{NatSolver, NatConstraint};
/// use bhc_types::{nat::TyNat, TyVar, Kind};
///
/// let mut solver = NatSolver::new();
///
/// let m = TyVar::new(1, Kind::Nat);
/// let k = TyVar::new(2, Kind::Nat);
///
/// // Add constraint: m = 1024
/// solver.add_constraint(NatConstraint::equal(
///     TyNat::Var(m.clone()),
///     TyNat::lit(1024),
/// ));
///
/// // Add constraint: k = m
/// solver.add_constraint(NatConstraint::equal(
///     TyNat::Var(k.clone()),
///     TyNat::Var(m.clone()),
/// ));
///
/// // Solve
/// let subst = solver.solve()?;
/// assert_eq!(subst.apply(&TyNat::Var(k)), TyNat::lit(1024));
/// ```
#[derive(Clone, Debug, Default)]
pub struct NatSolver {
    /// Collected constraints.
    constraints: Vec<NatConstraint>,
    /// Current substitution (built incrementally).
    subst: NatSubst,
}

impl NatSolver {
    /// Creates a new solver.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a solver with an initial substitution.
    #[must_use]
    pub fn with_subst(subst: NatSubst) -> Self {
        Self {
            constraints: Vec::new(),
            subst,
        }
    }

    /// Adds a constraint to the solver.
    pub fn add_constraint(&mut self, constraint: NatConstraint) {
        self.constraints.push(constraint);
    }

    /// Adds an equality constraint.
    pub fn add_equal(&mut self, left: TyNat, right: TyNat) {
        self.add_constraint(NatConstraint::equal(left, right));
    }

    /// Returns the current constraints.
    #[must_use]
    pub fn constraints(&self) -> &[NatConstraint] {
        &self.constraints
    }

    /// Returns the current substitution.
    #[must_use]
    pub fn substitution(&self) -> &NatSubst {
        &self.subst
    }

    /// Solves all constraints and returns the resulting substitution.
    ///
    /// # Errors
    ///
    /// Returns an error if the constraints are unsatisfiable.
    pub fn solve(mut self) -> SolverResult<NatSubst> {
        // Process constraints until fixed point
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            // Take constraints to process
            let constraints = std::mem::take(&mut self.constraints);

            for constraint in constraints {
                // Apply current substitution
                let constraint = constraint.apply_subst(&self.subst);

                // Try to solve this constraint
                match self.solve_one(&constraint)? {
                    SolveResult::Solved => {
                        changed = true;
                    }
                    SolveResult::Deferred(c) => {
                        self.constraints.push(c);
                    }
                    SolveResult::Trivial => {
                        // Constraint is satisfied, nothing to do
                    }
                }
            }
        }

        // Check remaining constraints
        for constraint in &self.constraints {
            let constraint = constraint.apply_subst(&self.subst);
            if !self.is_trivially_satisfied(&constraint) {
                return Err(SolverError::CannotSolve { constraint });
            }
        }

        Ok(self.subst)
    }

    /// Attempts to solve a single constraint.
    fn solve_one(&mut self, constraint: &NatConstraint) -> SolverResult<SolveResult> {
        match constraint {
            NatConstraint::Equal(left, right) => self.unify(left, right),
        }
    }

    /// Unifies two type-level naturals.
    fn unify(&mut self, left: &TyNat, right: &TyNat) -> SolverResult<SolveResult> {
        // Normalize both sides
        let left = self.normalize(left);
        let right = self.normalize(right);

        match (&left, &right) {
            // Same literal - trivially satisfied
            (TyNat::Lit(n1), TyNat::Lit(n2)) if n1 == n2 => Ok(SolveResult::Trivial),

            // Different literals - contradiction
            (TyNat::Lit(n1), TyNat::Lit(n2)) => Err(SolverError::LiteralMismatch {
                expected: *n1,
                found: *n2,
            }),

            // Same variable - trivially satisfied
            (TyNat::Var(v1), TyNat::Var(v2)) if v1.id == v2.id => Ok(SolveResult::Trivial),

            // Variable on left - bind it
            (TyNat::Var(v), _) => {
                // Occurs check
                if self.occurs_in(v, &right) {
                    return Err(SolverError::OccursCheck {
                        var_id: v.id,
                        term: right.clone(),
                    });
                }
                self.subst.bind(v, right.clone());
                Ok(SolveResult::Solved)
            }

            // Variable on right - bind it
            (_, TyNat::Var(v)) => {
                // Occurs check
                if self.occurs_in(v, &left) {
                    return Err(SolverError::OccursCheck {
                        var_id: v.id,
                        term: left.clone(),
                    });
                }
                self.subst.bind(v, left.clone());
                Ok(SolveResult::Solved)
            }

            // Both are Add - try to decompose
            (TyNat::Add(l1, l2), TyNat::Add(r1, r2)) => {
                // Add constraints for subterms
                self.add_equal((**l1).clone(), (**r1).clone());
                self.add_equal((**l2).clone(), (**r2).clone());
                Ok(SolveResult::Solved)
            }

            // Both are Mul - try to decompose
            (TyNat::Mul(l1, l2), TyNat::Mul(r1, r2)) => {
                self.add_equal((**l1).clone(), (**r1).clone());
                self.add_equal((**l2).clone(), (**r2).clone());
                Ok(SolveResult::Solved)
            }

            // Add on one side, literal on other - try to solve
            (TyNat::Add(a, b), TyNat::Lit(n)) | (TyNat::Lit(n), TyNat::Add(a, b)) => {
                self.solve_add_equals_lit(a, b, *n)
            }

            // Mul on one side, literal on other - try to solve
            (TyNat::Mul(a, b), TyNat::Lit(n)) | (TyNat::Lit(n), TyNat::Mul(a, b)) => {
                self.solve_mul_equals_lit(a, b, *n)
            }

            // Can't solve - defer
            _ => Ok(SolveResult::Deferred(NatConstraint::equal(left, right))),
        }
    }

    /// Tries to solve `a + b = n` where n is a literal.
    fn solve_add_equals_lit(&mut self, a: &TyNat, b: &TyNat, n: u64) -> SolverResult<SolveResult> {
        let a = self.normalize(a);
        let b = self.normalize(b);

        match (&a, &b) {
            // Both literals - check equality
            (TyNat::Lit(x), TyNat::Lit(y)) => {
                if x + y == n {
                    Ok(SolveResult::Trivial)
                } else {
                    Err(SolverError::LiteralMismatch {
                        expected: n,
                        found: x + y,
                    })
                }
            }

            // One is a literal - solve for the other
            (TyNat::Var(v), TyNat::Lit(k)) | (TyNat::Lit(k), TyNat::Var(v)) => {
                if n >= *k {
                    self.subst.bind(v, TyNat::lit(n - k));
                    Ok(SolveResult::Solved)
                } else {
                    Err(SolverError::Inconsistent {
                        message: format!("cannot solve {} + {} = {}", a, b, n),
                    })
                }
            }

            // Can't solve yet
            _ => Ok(SolveResult::Deferred(NatConstraint::equal(
                TyNat::add(a, b),
                TyNat::lit(n),
            ))),
        }
    }

    /// Tries to solve `a * b = n` where n is a literal.
    fn solve_mul_equals_lit(&mut self, a: &TyNat, b: &TyNat, n: u64) -> SolverResult<SolveResult> {
        let a = self.normalize(a);
        let b = self.normalize(b);

        match (&a, &b) {
            // Both literals - check equality
            (TyNat::Lit(x), TyNat::Lit(y)) => {
                if x * y == n {
                    Ok(SolveResult::Trivial)
                } else {
                    Err(SolverError::LiteralMismatch {
                        expected: n,
                        found: x * y,
                    })
                }
            }

            // One is a literal - solve for the other if divisible
            (TyNat::Var(v), TyNat::Lit(k)) | (TyNat::Lit(k), TyNat::Var(v)) if *k != 0 => {
                if n % k == 0 {
                    self.subst.bind(v, TyNat::lit(n / k));
                    Ok(SolveResult::Solved)
                } else {
                    Err(SolverError::Inconsistent {
                        message: format!("{} is not divisible by {}", n, k),
                    })
                }
            }

            // Can't solve yet
            _ => Ok(SolveResult::Deferred(NatConstraint::equal(
                TyNat::mul(a, b),
                TyNat::lit(n),
            ))),
        }
    }

    /// Normalizes a type-level natural by applying the current substitution
    /// and simplifying arithmetic.
    fn normalize(&self, nat: &TyNat) -> TyNat {
        self.subst.apply(nat)
    }

    /// Checks if a variable occurs in a term.
    fn occurs_in(&self, var: &TyVar, term: &TyNat) -> bool {
        match term {
            TyNat::Lit(_) => false,
            TyNat::Var(v) => v.id == var.id,
            TyNat::Add(a, b) | TyNat::Mul(a, b) => self.occurs_in(var, a) || self.occurs_in(var, b),
        }
    }

    /// Checks if a constraint is trivially satisfied.
    fn is_trivially_satisfied(&self, constraint: &NatConstraint) -> bool {
        match constraint {
            NatConstraint::Equal(left, right) => {
                let left = self.normalize(left);
                let right = self.normalize(right);
                left == right
            }
        }
    }
}

/// Result of attempting to solve a single constraint.
enum SolveResult {
    /// The constraint was solved and added bindings to the substitution.
    Solved,
    /// The constraint is trivially satisfied (e.g., `5 = 5`).
    Trivial,
    /// The constraint couldn't be solved yet and should be retried.
    Deferred(NatConstraint),
}

/// Convenience function to solve a set of equality constraints.
///
/// # Example
///
/// ```ignore
/// let m = TyVar::new(1, Kind::Nat);
/// let constraints = vec![
///     (TyNat::Var(m.clone()), TyNat::lit(1024)),
/// ];
/// let subst = solve_nat_constraints(constraints)?;
/// ```
pub fn solve_nat_constraints(
    constraints: impl IntoIterator<Item = (TyNat, TyNat)>,
) -> SolverResult<NatSubst> {
    let mut solver = NatSolver::new();
    for (left, right) in constraints {
        solver.add_equal(left, right);
    }
    solver.solve()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_types::Kind;

    fn nat_var(id: u32) -> TyVar {
        TyVar::new(id, Kind::Nat)
    }

    #[test]
    fn test_literal_equality() {
        let mut solver = NatSolver::new();
        solver.add_equal(TyNat::lit(42), TyNat::lit(42));

        let subst = solver.solve().unwrap();
        assert!(subst.is_empty());
    }

    #[test]
    fn test_literal_mismatch() {
        let mut solver = NatSolver::new();
        solver.add_equal(TyNat::lit(42), TyNat::lit(43));

        let result = solver.solve();
        assert!(matches!(result, Err(SolverError::LiteralMismatch { .. })));
    }

    #[test]
    fn test_variable_binding() {
        let m = nat_var(1);
        let mut solver = NatSolver::new();
        solver.add_equal(TyNat::Var(m.clone()), TyNat::lit(1024));

        let subst = solver.solve().unwrap();
        assert_eq!(subst.apply(&TyNat::Var(m)), TyNat::lit(1024));
    }

    #[test]
    fn test_variable_to_variable() {
        let m = nat_var(1);
        let n = nat_var(2);
        let mut solver = NatSolver::new();
        solver.add_equal(TyNat::Var(m.clone()), TyNat::Var(n.clone()));
        solver.add_equal(TyNat::Var(n.clone()), TyNat::lit(512));

        let subst = solver.solve().unwrap();
        assert_eq!(subst.apply(&TyNat::Var(m)), TyNat::lit(512));
        assert_eq!(subst.apply(&TyNat::Var(n)), TyNat::lit(512));
    }

    #[test]
    fn test_transitive_equality() {
        let a = nat_var(1);
        let b = nat_var(2);
        let c = nat_var(3);
        let mut solver = NatSolver::new();
        solver.add_equal(TyNat::Var(a.clone()), TyNat::Var(b.clone()));
        solver.add_equal(TyNat::Var(b.clone()), TyNat::Var(c.clone()));
        solver.add_equal(TyNat::Var(c.clone()), TyNat::lit(100));

        let subst = solver.solve().unwrap();
        assert_eq!(subst.apply(&TyNat::Var(a)), TyNat::lit(100));
        assert_eq!(subst.apply(&TyNat::Var(b)), TyNat::lit(100));
        assert_eq!(subst.apply(&TyNat::Var(c)), TyNat::lit(100));
    }

    #[test]
    fn test_add_constraint_with_literal() {
        // m + 5 = 10 → m = 5
        let m = nat_var(1);
        let mut solver = NatSolver::new();
        solver.add_equal(
            TyNat::add(TyNat::Var(m.clone()), TyNat::lit(5)),
            TyNat::lit(10),
        );

        let subst = solver.solve().unwrap();
        assert_eq!(subst.apply(&TyNat::Var(m)), TyNat::lit(5));
    }

    #[test]
    fn test_add_constraint_impossible() {
        // m + 10 = 5 → impossible (m would be negative)
        let m = nat_var(1);
        let mut solver = NatSolver::new();
        solver.add_equal(
            TyNat::add(TyNat::Var(m.clone()), TyNat::lit(10)),
            TyNat::lit(5),
        );

        let result = solver.solve();
        assert!(matches!(result, Err(SolverError::Inconsistent { .. })));
    }

    #[test]
    fn test_mul_constraint_with_literal() {
        // m * 4 = 12 → m = 3
        let m = nat_var(1);
        let mut solver = NatSolver::new();
        solver.add_equal(
            TyNat::mul(TyNat::Var(m.clone()), TyNat::lit(4)),
            TyNat::lit(12),
        );

        let subst = solver.solve().unwrap();
        assert_eq!(subst.apply(&TyNat::Var(m)), TyNat::lit(3));
    }

    #[test]
    fn test_mul_constraint_not_divisible() {
        // m * 4 = 10 → impossible (10 not divisible by 4)
        let m = nat_var(1);
        let mut solver = NatSolver::new();
        solver.add_equal(
            TyNat::mul(TyNat::Var(m.clone()), TyNat::lit(4)),
            TyNat::lit(10),
        );

        let result = solver.solve();
        assert!(matches!(result, Err(SolverError::Inconsistent { .. })));
    }

    #[test]
    fn test_matmul_dimension_check() {
        // Simulates checking matmul: [m, k1] x [k2, n] where k1 = k2
        let m = nat_var(1);
        let k = nat_var(2);
        let n = nat_var(3);

        let mut solver = NatSolver::new();
        // k1 = k2 (inner dimensions match)
        solver.add_equal(TyNat::Var(k.clone()), TyNat::Var(k.clone()));
        // m = 1024, k = 768, n = 512
        solver.add_equal(TyNat::Var(m.clone()), TyNat::lit(1024));
        solver.add_equal(TyNat::Var(k.clone()), TyNat::lit(768));
        solver.add_equal(TyNat::Var(n.clone()), TyNat::lit(512));

        let subst = solver.solve().unwrap();
        assert_eq!(subst.apply(&TyNat::Var(m)), TyNat::lit(1024));
        assert_eq!(subst.apply(&TyNat::Var(k)), TyNat::lit(768));
        assert_eq!(subst.apply(&TyNat::Var(n)), TyNat::lit(512));
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        // [m, 768] x [512, n] - inner dimensions don't match
        let mut solver = NatSolver::new();
        solver.add_equal(TyNat::lit(768), TyNat::lit(512));

        let result = solver.solve();
        assert!(matches!(
            result,
            Err(SolverError::LiteralMismatch {
                expected: 768,
                found: 512
            })
        ));
    }

    #[test]
    fn test_occurs_check() {
        // m = m + 1 → infinite type
        let m = nat_var(1);
        let mut solver = NatSolver::new();
        solver.add_equal(
            TyNat::Var(m.clone()),
            TyNat::Add(Box::new(TyNat::Var(m.clone())), Box::new(TyNat::lit(1))),
        );

        let result = solver.solve();
        assert!(matches!(result, Err(SolverError::OccursCheck { .. })));
    }

    #[test]
    fn test_subst_compose() {
        let m = nat_var(1);
        let n = nat_var(2);

        let mut s1 = NatSubst::new();
        s1.bind(&m, TyNat::Var(n.clone()));

        let mut s2 = NatSubst::new();
        s2.bind(&n, TyNat::lit(42));

        let composed = s1.compose(&s2);
        assert_eq!(composed.apply(&TyNat::Var(m)), TyNat::lit(42));
    }

    #[test]
    fn test_convenience_function() {
        let m = nat_var(1);
        let constraints = vec![(TyNat::Var(m.clone()), TyNat::lit(256))];

        let subst = solve_nat_constraints(constraints).unwrap();
        assert_eq!(subst.apply(&TyNat::Var(m)), TyNat::lit(256));
    }

    #[test]
    fn test_arithmetic_simplification() {
        // When we have m = 5 and query m + 3, it should simplify to 8
        let m = nat_var(1);
        let mut solver = NatSolver::new();
        solver.add_equal(TyNat::Var(m.clone()), TyNat::lit(5));

        let subst = solver.solve().unwrap();
        let result = subst.apply(&TyNat::add(TyNat::Var(m), TyNat::lit(3)));
        assert_eq!(result, TyNat::lit(8));
    }

    #[test]
    fn test_multiple_constraints() {
        let a = nat_var(1);
        let b = nat_var(2);
        let c = nat_var(3);

        let mut solver = NatSolver::new();
        // a + b = 10
        solver.add_equal(
            TyNat::add(TyNat::Var(a.clone()), TyNat::Var(b.clone())),
            TyNat::lit(10),
        );
        // a = 3
        solver.add_equal(TyNat::Var(a.clone()), TyNat::lit(3));
        // c = b
        solver.add_equal(TyNat::Var(c.clone()), TyNat::Var(b.clone()));

        let subst = solver.solve().unwrap();
        assert_eq!(subst.apply(&TyNat::Var(a)), TyNat::lit(3));
        assert_eq!(subst.apply(&TyNat::Var(b)), TyNat::lit(7));
        assert_eq!(subst.apply(&TyNat::Var(c)), TyNat::lit(7));
    }
}
