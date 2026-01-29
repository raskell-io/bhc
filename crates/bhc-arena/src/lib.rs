//! Arena allocators for efficient compiler data structure allocation.
//!
//! This crate provides arena allocators that enable fast allocation
//! of compiler data structures with automatic bulk deallocation.

#![warn(missing_docs)]

use std::cell::Cell;

pub use bumpalo::Bump;
pub use typed_arena::Arena as TypedArena;

/// A thread-local arena for fast, scoped allocations.
///
/// All allocations from this arena are freed when the arena is dropped.
/// This is ideal for per-function or per-module compiler passes.
#[derive(Debug)]
pub struct Arena {
    bump: Bump,
    bytes_allocated: Cell<usize>,
}

impl Arena {
    /// Create a new empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bump: Bump::new(),
            bytes_allocated: Cell::new(0),
        }
    }

    /// Create a new arena with the specified capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bump: Bump::with_capacity(capacity),
            bytes_allocated: Cell::new(0),
        }
    }

    /// Allocate a value in the arena.
    pub fn alloc<T>(&self, val: T) -> &mut T {
        self.bytes_allocated
            .set(self.bytes_allocated.get() + std::mem::size_of::<T>());
        self.bump.alloc(val)
    }

    /// Allocate a slice in the arena.
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &mut [T] {
        self.bytes_allocated
            .set(self.bytes_allocated.get() + std::mem::size_of_val(slice));
        self.bump.alloc_slice_copy(slice)
    }

    /// Allocate a string in the arena.
    pub fn alloc_str(&self, s: &str) -> &str {
        self.bytes_allocated
            .set(self.bytes_allocated.get() + s.len());
        self.bump.alloc_str(s)
    }

    /// Allocate an iterator's items in the arena.
    pub fn alloc_from_iter<T, I>(&self, iter: I) -> &mut [T]
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let result = self.bump.alloc_slice_fill_iter(iter);
        self.bytes_allocated
            .set(self.bytes_allocated.get() + std::mem::size_of_val(result));
        result
    }

    /// Get the total bytes allocated in this arena.
    #[must_use]
    pub fn bytes_allocated(&self) -> usize {
        self.bytes_allocated.get()
    }

    /// Get the allocated bytes including overhead.
    #[must_use]
    pub fn allocated_bytes_including_metadata(&self) -> usize {
        self.bump.allocated_bytes()
    }

    /// Reset the arena, deallocating all values.
    ///
    /// # Safety
    ///
    /// All references to arena-allocated values become invalid after this call.
    pub unsafe fn reset(&mut self) {
        self.bump.reset();
        self.bytes_allocated.set(0);
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

/// A dropless arena for types that don't need drop.
///
/// This is more efficient than `Arena` for types that implement `Copy`
/// or otherwise don't need destructors to run.
#[derive(Debug)]
pub struct DroplessArena {
    bump: Bump,
}

impl DroplessArena {
    /// Create a new dropless arena.
    #[must_use]
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }

    /// Allocate a value in the arena.
    ///
    /// The value's destructor will never be called.
    pub fn alloc<T>(&self, val: T) -> &mut T {
        self.bump.alloc(val)
    }

    /// Allocate a slice from an iterator.
    pub fn alloc_from_iter<T, I>(&self, iter: I) -> &mut [T]
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        self.bump.alloc_slice_fill_iter(iter)
    }
}

impl Default for DroplessArena {
    fn default() -> Self {
        Self::new()
    }
}

/// A sync arena that can be shared across threads.
#[derive(Debug)]
pub struct SyncArena {
    bump: parking_lot::Mutex<Bump>,
}

impl SyncArena {
    /// Create a new sync arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bump: parking_lot::Mutex::new(Bump::new()),
        }
    }

    /// Allocate a value in the arena.
    pub fn alloc<T: Send>(&self, val: T) -> &T {
        // Safety: The arena lives longer than any references we hand out,
        // and we're using a mutex for synchronization.
        let bump = self.bump.lock();
        let ptr = bump.alloc(val) as *const T;
        unsafe { &*ptr }
    }
}

impl Default for SyncArena {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: SyncArena uses internal synchronization
unsafe impl Send for SyncArena {}
unsafe impl Sync for SyncArena {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation() {
        let arena = Arena::new();
        let x = arena.alloc(42);
        let y = arena.alloc("hello");

        assert_eq!(*x, 42);
        assert_eq!(*y, "hello");
    }

    #[test]
    fn test_arena_slice() {
        let arena = Arena::new();
        let slice = arena.alloc_slice(&[1, 2, 3, 4, 5]);

        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_arena_from_iter() {
        let arena = Arena::new();
        let slice = arena.alloc_from_iter(0..5);

        assert_eq!(slice, &[0, 1, 2, 3, 4]);
    }
}
