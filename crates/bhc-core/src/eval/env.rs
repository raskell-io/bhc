//! Environment for variable bindings during evaluation.
//!
//! The environment maps variable IDs to their values. It uses a persistent
//! data structure (im::HashMap) for efficient cloning when creating closures.

use im::HashMap;

use crate::VarId;

use super::value::Value;

/// An environment mapping variables to values.
///
/// The environment is immutable and uses structural sharing for efficiency.
/// When a closure captures an environment, only a reference count is incremented.
#[derive(Clone, Debug, Default)]
pub struct Env {
    bindings: HashMap<VarId, Value>,
}

impl Env {
    /// Creates an empty environment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Looks up a variable in the environment.
    #[must_use]
    pub fn lookup(&self, id: VarId) -> Option<&Value> {
        self.bindings.get(&id)
    }

    /// Extends the environment with a new binding.
    ///
    /// Returns a new environment with the binding added.
    /// The original environment is unchanged.
    #[must_use]
    pub fn extend(&self, id: VarId, value: Value) -> Self {
        Self {
            bindings: self.bindings.update(id, value),
        }
    }

    /// Extends the environment with multiple bindings.
    #[must_use]
    pub fn extend_many(&self, bindings: impl IntoIterator<Item = (VarId, Value)>) -> Self {
        let mut new_bindings = self.bindings.clone();
        for (id, value) in bindings {
            new_bindings.insert(id, value);
        }
        Self {
            bindings: new_bindings,
        }
    }

    /// Returns the number of bindings in the environment.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns true if the environment is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Returns all VarIds in the environment (for debugging).
    pub fn keys(&self) -> impl Iterator<Item = &VarId> {
        self.bindings.keys()
    }
}

#[cfg(test)]
mod tests {
    use bhc_index::Idx;

    use super::*;

    #[test]
    fn test_env_lookup() {
        let env = Env::new();
        let id = VarId::new(0);
        let val = Value::Int(42);

        let env2 = env.extend(id, val.clone());
        assert!(env.lookup(id).is_none());
        assert!(matches!(env2.lookup(id), Some(Value::Int(42))));
    }

    #[test]
    fn test_env_extend_many() {
        let env = Env::new();
        let bindings = vec![
            (VarId::new(0), Value::Int(1)),
            (VarId::new(1), Value::Int(2)),
            (VarId::new(2), Value::Int(3)),
        ];

        let env2 = env.extend_many(bindings);
        assert_eq!(env2.len(), 3);
        assert!(matches!(env2.lookup(VarId::new(1)), Some(Value::Int(2))));
    }

    #[test]
    fn test_env_shadowing() {
        let env = Env::new();
        let id = VarId::new(0);

        let env2 = env.extend(id, Value::Int(1));
        let env3 = env2.extend(id, Value::Int(2));

        assert!(matches!(env2.lookup(id), Some(Value::Int(1))));
        assert!(matches!(env3.lookup(id), Some(Value::Int(2))));
    }
}
