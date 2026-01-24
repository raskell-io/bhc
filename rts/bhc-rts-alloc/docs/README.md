# bhc-rts-alloc

Memory Allocation Primitives for the Basel Haskell Compiler.

## Overview

This crate provides foundational memory allocation primitives used by the BHC runtime. It defines core abstractions for memory regions, alignment requirements, and allocation interfaces as specified in H26-SPEC Section 9.

## Memory Regions

BHC defines three allocation regions with distinct characteristics:

| Region | Allocation | Deallocation | GC | Use Case |
|--------|------------|--------------|-----|----------|
| Hot Arena | Bump pointer | Scope-based | None | Kernel temporaries |
| Pinned Heap | malloc-style | Explicit | Never moved | FFI, device IO |
| General Heap | GC-managed | Automatic | May move | Boxed values |

## Alignment

```rust
/// Alignment requirements for various use cases
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Alignment {
    /// Default alignment (8 bytes on 64-bit)
    Default = 8,

    /// SSE/NEON alignment (16 bytes)
    Simd128 = 16,

    /// AVX alignment (32 bytes)
    Simd256 = 32,

    /// AVX-512 alignment (64 bytes)
    Simd512 = 64,

    /// Cache line alignment (typically 64 bytes)
    CacheLine = 64,

    /// Page alignment (typically 4096 bytes)
    Page = 4096,
}

impl Alignment {
    /// Get alignment as usize
    pub const fn as_usize(self) -> usize {
        self as usize
    }

    /// Check if an alignment is valid (power of 2)
    pub const fn is_valid(align: usize) -> bool {
        align > 0 && (align & (align - 1)) == 0
    }
}
```

## Alignment Utilities

```rust
/// Align value up to given alignment
#[inline]
pub const fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(Alignment::is_valid(align));
    (value + align - 1) & !(align - 1)
}

/// Align value down to given alignment
#[inline]
pub const fn align_down(value: usize, align: usize) -> usize {
    debug_assert!(Alignment::is_valid(align));
    value & !(align - 1)
}

/// Check if value is aligned
#[inline]
pub const fn is_aligned(value: usize, align: usize) -> bool {
    debug_assert!(Alignment::is_valid(align));
    value & (align - 1) == 0
}

/// Check if pointer is aligned
#[inline]
pub fn is_ptr_aligned<T>(ptr: *const T, align: usize) -> bool {
    is_aligned(ptr as usize, align)
}
```

## Memory Region Trait

```rust
/// Trait for memory regions
pub trait MemoryRegion {
    /// Allocate memory with given layout
    fn alloc(&self, layout: Layout) -> AllocResult<NonNull<u8>>;

    /// Deallocate memory
    ///
    /// # Safety
    /// - `ptr` must have been allocated by this region
    /// - `layout` must match the allocation layout
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout);

    /// Reallocate memory
    ///
    /// # Safety
    /// - `ptr` must have been allocated by this region
    /// - `old_layout` must match the original allocation
    unsafe fn realloc(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_size: usize,
    ) -> AllocResult<NonNull<u8>> {
        // Default implementation: alloc + copy + dealloc
        let new_layout = Layout::from_size_align(new_size, old_layout.align())?;
        let new_ptr = self.alloc(new_layout)?;
        std::ptr::copy_nonoverlapping(
            ptr.as_ptr(),
            new_ptr.as_ptr(),
            old_layout.size().min(new_size),
        );
        self.dealloc(ptr, old_layout);
        Ok(new_ptr)
    }

    /// Get allocation statistics
    fn stats(&self) -> AllocStats;
}
```

## Allocation Statistics

```rust
#[derive(Clone, Debug, Default)]
pub struct AllocStats {
    /// Total bytes allocated over lifetime
    pub total_allocated: usize,

    /// Total bytes deallocated over lifetime
    pub total_deallocated: usize,

    /// Current live bytes
    pub current_live: usize,

    /// Number of allocations
    pub allocation_count: usize,

    /// Number of deallocations
    pub deallocation_count: usize,

    /// Peak memory usage
    pub peak_usage: usize,

    /// Number of failed allocations
    pub failed_allocations: usize,
}

impl AllocStats {
    /// Update stats after allocation
    pub fn record_alloc(&mut self, size: usize) {
        self.total_allocated += size;
        self.current_live += size;
        self.allocation_count += 1;
        self.peak_usage = self.peak_usage.max(self.current_live);
    }

    /// Update stats after deallocation
    pub fn record_dealloc(&mut self, size: usize) {
        self.total_deallocated += size;
        self.current_live -= size;
        self.deallocation_count += 1;
    }
}
```

## Error Types

```rust
/// Allocation errors
#[derive(Clone, Debug)]
pub enum AllocError {
    /// Out of memory
    OutOfMemory {
        requested: usize,
        available: usize,
    },

    /// Invalid layout (size or alignment)
    InvalidLayout(LayoutError),

    /// Region exhausted
    RegionExhausted {
        region: &'static str,
    },

    /// Alignment not supported
    UnsupportedAlignment(usize),

    /// Allocation would overflow
    Overflow,
}

pub type AllocResult<T> = Result<T, AllocError>;
```

## Aligned Allocation

```rust
/// Allocate memory with specific alignment
pub fn alloc_aligned<T>(count: usize, align: Alignment) -> AllocResult<NonNull<T>> {
    let size = count * std::mem::size_of::<T>();
    let layout = Layout::from_size_align(size, align.as_usize())
        .map_err(AllocError::InvalidLayout)?;

    let ptr = unsafe { std::alloc::alloc(layout) };
    NonNull::new(ptr as *mut T)
        .ok_or(AllocError::OutOfMemory { requested: size, available: 0 })
}

/// Free aligned allocation
///
/// # Safety
/// - `ptr` must have been allocated with `alloc_aligned`
/// - `count` and `align` must match the original allocation
pub unsafe fn free_aligned<T>(ptr: NonNull<T>, count: usize, align: Alignment) {
    let size = count * std::mem::size_of::<T>();
    let layout = Layout::from_size_align_unchecked(size, align.as_usize());
    std::alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
}
```

## SIMD-Aligned Buffer

```rust
/// A buffer guaranteed to be SIMD-aligned
pub struct SimdBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    alignment: Alignment,
}

impl<T> SimdBuffer<T> {
    /// Create new SIMD-aligned buffer
    pub fn new(len: usize, alignment: Alignment) -> AllocResult<Self> {
        let ptr = alloc_aligned::<T>(len, alignment)?;
        Ok(Self { ptr, len, alignment })
    }

    /// Get slice
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for SimdBuffer<T> {
    fn drop(&mut self) {
        unsafe { free_aligned(self.ptr, self.len, self.alignment) }
    }
}
```

## Platform Detection

```rust
/// Detect best SIMD alignment for current platform
pub fn detect_simd_alignment() -> Alignment {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return Alignment::Simd512;
        }
        if is_x86_feature_detected!("avx") {
            return Alignment::Simd256;
        }
        return Alignment::Simd128;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on AArch64
        return Alignment::Simd128;
    }

    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64"
    )))]
    {
        return Alignment::Default;
    }
}
```

## See Also

- `bhc-rts` - Core runtime
- `bhc-rts-arena` - Hot arena (uses these primitives)
- `bhc-rts-gc` - General heap GC
