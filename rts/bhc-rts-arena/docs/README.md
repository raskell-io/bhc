# bhc-rts-arena

Hot Arena Allocator for the Basel Haskell Compiler.

## Overview

This crate implements the Hot Arena memory region as specified in H26-SPEC Section 9.1. The hot arena provides ultra-fast O(1) allocation via pointer bumping with scope-based lifetime management. It's the primary allocation mechanism for the Numeric Profile.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Thread-Local Arena                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  base ──▶ ┌──────────────────────────────────────────────┐  │
│           │████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░│  │
│           └──────────────────────────────────────────────┘  │
│                             ▲                          ▲    │
│                             │                          │    │
│                          cursor                       end   │
│                                                              │
│  Used: ██████████████████   Free: ░░░░░░░░░░░░░░░░░░░░░░   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Allocation Algorithm

```rust
impl HotArena {
    /// O(1) bump allocation
    #[inline]
    pub fn alloc(&self, size: usize, align: usize) -> Option<*mut u8> {
        // Align cursor
        let aligned = (self.cursor.get() + align - 1) & !(align - 1);

        // Check if we have space
        let new_cursor = aligned + size;
        if new_cursor > self.end {
            return None; // Arena exhausted
        }

        // Bump cursor and return pointer
        self.cursor.set(new_cursor);
        Some(aligned as *mut u8)
    }

    /// O(1) reset - free all allocations
    #[inline]
    pub fn reset(&self) {
        self.cursor.set(self.base);
    }
}
```

## Scoped Allocation

```rust
/// Execute action with arena, free all on scope exit
pub fn with_arena<T, F>(size: usize, f: F) -> T
where
    F: FnOnce(&HotArena) -> T,
{
    let arena = HotArena::new(size).expect("arena allocation failed");
    let result = f(&arena);
    // arena automatically freed when dropped
    result
}

/// Nested arenas
pub fn with_nested_arena<T, F>(parent: &HotArena, size: usize, f: F) -> T
where
    F: FnOnce(&HotArena) -> T,
{
    let child = HotArena::from_parent(parent, size);
    let result = f(&child);
    // child freed, parent unchanged
    result
}
```

## Arena State

```rust
pub struct HotArena {
    /// Base address of arena memory
    base: usize,

    /// Current allocation cursor
    cursor: Cell<usize>,

    /// End of arena memory
    end: usize,

    /// Parent arena (for nested arenas)
    parent: Option<*const HotArena>,

    /// Alignment guarantee
    alignment: usize,
}

impl HotArena {
    /// Create new arena with given size
    pub fn new(size: usize) -> Result<Self, ArenaError>;

    /// Create arena from raw memory
    pub unsafe fn from_raw(ptr: *mut u8, size: usize) -> Self;

    /// Get current usage
    pub fn used(&self) -> usize {
        self.cursor.get() - self.base
    }

    /// Get remaining capacity
    pub fn remaining(&self) -> usize {
        self.end - self.cursor.get()
    }
}
```

## Typed Allocation

```rust
impl HotArena {
    /// Allocate slice of T
    pub fn alloc_slice<T>(&self, len: usize) -> Option<&mut [T]> {
        let size = len * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc(size, align)?;
        Some(unsafe {
            std::slice::from_raw_parts_mut(ptr as *mut T, len)
        })
    }

    /// Allocate and initialize single value
    pub fn alloc_value<T>(&self, value: T) -> Option<&mut T> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc(size, align)?;
        unsafe {
            std::ptr::write(ptr as *mut T, value);
            Some(&mut *(ptr as *mut T))
        }
    }

    /// Allocate with SIMD alignment
    pub fn alloc_simd<T>(&self, len: usize, simd_width: usize) -> Option<&mut [T]> {
        let size = len * std::mem::size_of::<T>();
        let align = simd_width.max(std::mem::align_of::<T>());
        let ptr = self.alloc(size, align)?;
        Some(unsafe {
            std::slice::from_raw_parts_mut(ptr as *mut T, len)
        })
    }
}
```

## Thread-Local Arenas

```rust
thread_local! {
    static THREAD_ARENA: RefCell<Option<HotArena>> = RefCell::new(None);
}

/// Initialize thread-local arena
pub fn init_thread_arena(size: usize) {
    THREAD_ARENA.with(|arena| {
        *arena.borrow_mut() = Some(HotArena::new(size).expect("arena init failed"));
    });
}

/// Get thread-local arena
pub fn thread_arena() -> &'static HotArena {
    THREAD_ARENA.with(|arena| {
        arena.borrow().as_ref().expect("thread arena not initialized")
    })
}
```

## Integration with Numeric Kernels

```rust
/// Matrix multiply with arena temporaries
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    with_arena(temp_size(a, b), |arena| {
        // Allocate temporary buffer in arena
        let tmp = arena.alloc_slice::<f32>(a.rows * b.cols)
            .expect("arena exhausted");

        // Compute with temporary
        compute_matmul_kernel(a, b, tmp);

        // Copy result out of arena
        Matrix::from_slice(a.rows, b.cols, tmp)
        // tmp freed here when arena scope ends
    })
}
```

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Allocate | O(1) | Single pointer bump |
| Reset | O(1) | Single pointer reset |
| Free individual | N/A | Not supported |
| Nested arena | O(1) | Carve from parent |

## Memory Layout Guarantees

- **Contiguous**: All allocations are contiguous in memory
- **Aligned**: Each allocation respects requested alignment
- **No fragmentation**: No holes within active allocations
- **Cache-friendly**: Sequential allocation pattern

## Alignment Support

| Alignment | Size | Use Case |
|-----------|------|----------|
| Default | 8 bytes | General purpose |
| SIMD128 | 16 bytes | SSE, NEON |
| SIMD256 | 32 bytes | AVX |
| SIMD512 | 64 bytes | AVX-512 |
| CacheLine | 64 bytes | Cache efficiency |
| Page | 4096 bytes | Page-aligned buffers |

## Error Handling

```rust
pub enum ArenaError {
    /// Failed to allocate arena memory
    AllocationFailed { requested: usize },

    /// Arena exhausted
    Exhausted { requested: usize, remaining: usize },

    /// Invalid alignment
    InvalidAlignment { alignment: usize },
}
```

## See Also

- `bhc-rts` - Core runtime
- `bhc-rts-alloc` - Allocation primitives
- `bhc-rts-gc` - Garbage collector (complementary)
