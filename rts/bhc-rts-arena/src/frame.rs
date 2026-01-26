//! Frame-based arena allocator for realtime applications.
//!
//! This module provides a frame-oriented allocation pattern commonly used in
//! games, audio processing, and other realtime applications where:
//!
//! - Allocations happen during a frame/tick
//! - All allocations are freed at the end of each frame
//! - Frames have bounded and predictable memory usage
//!
//! # Usage
//!
//! ```ignore
//! use bhc_rts_arena::frame::{FrameArena, FrameAllocator};
//!
//! let mut frame_arena = FrameArena::new(1024 * 1024);
//!
//! // Game loop
//! loop {
//!     frame_arena.begin_frame();
//!
//!     // Allocate during frame - O(1) bump allocation
//!     let entities = frame_arena.alloc_slice(&[Entity::default(); 100]);
//!     let particles = frame_arena.alloc_slice_zeroed::<Particle>(1000);
//!
//!     update_game(entities, particles);
//!     render();
//!
//!     frame_arena.end_frame();
//!     // All allocations freed
//! }
//! ```
//!
//! # Double Buffering
//!
//! For scenarios where data needs to persist for one extra frame (e.g., for
//! interpolation), use `DoubleBufferedFrameArena`:
//!
//! ```ignore
//! let mut arena = DoubleBufferedFrameArena::new(1024 * 1024);
//!
//! // Frame N: allocate current frame data
//! arena.begin_frame();
//! let current_positions = arena.current().alloc_slice(&positions);
//!
//! // Previous frame's data is still accessible
//! let prev_positions = arena.previous().get_slice::<Vec3>(prev_handle);
//!
//! interpolate(prev_positions, current_positions, alpha);
//!
//! arena.end_frame(); // Previous frame freed, current becomes previous
//! ```

use crate::HotArena;
use bhc_rts_alloc::{Alignment, AllocResult, AllocStats};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Statistics for frame arena usage.
#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    /// Number of frames completed.
    pub frames_completed: u64,
    /// Total bytes allocated across all frames.
    pub total_bytes_allocated: u64,
    /// Maximum bytes used in a single frame.
    pub max_frame_bytes: usize,
    /// Minimum bytes used in a single frame.
    pub min_frame_bytes: Option<usize>,
    /// Average bytes per frame.
    pub avg_frame_bytes: usize,
    /// Number of allocation failures.
    pub allocation_failures: u64,
    /// Maximum allocations in a single frame.
    pub max_frame_allocations: usize,
    /// Total time spent in frame transitions.
    pub frame_transition_time: Duration,
}

/// Frame lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameState {
    /// Not currently in a frame.
    Idle,
    /// Currently processing a frame.
    InFrame,
}

/// A frame-based arena allocator.
///
/// Provides begin_frame/end_frame lifecycle for realtime allocation patterns.
#[derive(Debug)]
pub struct FrameArena {
    /// Underlying hot arena.
    arena: HotArena,
    /// Current frame state.
    state: FrameState,
    /// Current frame number.
    frame_number: u64,
    /// Bytes allocated in current frame.
    current_frame_bytes: usize,
    /// Allocations in current frame.
    current_frame_allocs: usize,
    /// Statistics.
    stats: FrameStats,
    /// When the current frame started.
    frame_start: Option<Instant>,
}

impl FrameArena {
    /// Create a new frame arena with the specified capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            arena: HotArena::new(capacity),
            state: FrameState::Idle,
            frame_number: 0,
            current_frame_bytes: 0,
            current_frame_allocs: 0,
            stats: FrameStats::default(),
            frame_start: None,
        }
    }

    /// Create a new frame arena with cache-line aligned memory.
    #[must_use]
    pub fn with_alignment(capacity: usize, alignment: Alignment) -> Self {
        Self {
            arena: HotArena::with_alignment(capacity, alignment),
            state: FrameState::Idle,
            frame_number: 0,
            current_frame_bytes: 0,
            current_frame_allocs: 0,
            stats: FrameStats::default(),
            frame_start: None,
        }
    }

    /// Begin a new frame.
    ///
    /// Must be called before any allocations for this frame.
    /// Panics if already in a frame.
    pub fn begin_frame(&mut self) {
        assert_eq!(
            self.state,
            FrameState::Idle,
            "begin_frame called while already in a frame"
        );

        self.state = FrameState::InFrame;
        self.frame_start = Some(Instant::now());
        self.current_frame_bytes = 0;
        self.current_frame_allocs = 0;
    }

    /// End the current frame and free all allocations.
    ///
    /// Must be called after a matching begin_frame.
    /// Panics if not in a frame.
    pub fn end_frame(&mut self) {
        assert_eq!(
            self.state,
            FrameState::InFrame,
            "end_frame called without begin_frame"
        );

        // Update statistics
        self.stats.frames_completed += 1;
        self.stats.total_bytes_allocated += self.current_frame_bytes as u64;
        self.stats.max_frame_bytes = self.stats.max_frame_bytes.max(self.current_frame_bytes);
        self.stats.min_frame_bytes = Some(
            self.stats
                .min_frame_bytes
                .map_or(self.current_frame_bytes, |min| {
                    min.min(self.current_frame_bytes)
                }),
        );
        self.stats.avg_frame_bytes = if self.stats.frames_completed > 0 {
            (self.stats.total_bytes_allocated / self.stats.frames_completed) as usize
        } else {
            0
        };
        self.stats.max_frame_allocations = self
            .stats
            .max_frame_allocations
            .max(self.current_frame_allocs);

        if let Some(start) = self.frame_start {
            self.stats.frame_transition_time += start.elapsed();
        }

        // Reset the arena
        // Safety: All frame allocations are being invalidated
        unsafe {
            self.arena.reset();
        }

        self.state = FrameState::Idle;
        self.frame_number += 1;
        self.frame_start = None;
    }

    /// Allocate a value in the current frame.
    ///
    /// Panics if not in a frame.
    pub fn alloc<T>(&mut self, value: T) -> AllocResult<&mut T> {
        assert_eq!(
            self.state,
            FrameState::InFrame,
            "allocation outside of frame"
        );

        match self.arena.alloc(value) {
            Ok(ptr) => {
                self.current_frame_bytes += std::mem::size_of::<T>();
                self.current_frame_allocs += 1;
                Ok(ptr)
            }
            Err(e) => {
                self.stats.allocation_failures += 1;
                Err(e)
            }
        }
    }

    /// Allocate a slice in the current frame.
    ///
    /// Panics if not in a frame.
    pub fn alloc_slice<T: Copy>(&mut self, values: &[T]) -> AllocResult<&mut [T]> {
        assert_eq!(
            self.state,
            FrameState::InFrame,
            "allocation outside of frame"
        );

        match self.arena.alloc_slice(values) {
            Ok(slice) => {
                self.current_frame_bytes += std::mem::size_of::<T>() * values.len();
                self.current_frame_allocs += 1;
                Ok(slice)
            }
            Err(e) => {
                self.stats.allocation_failures += 1;
                Err(e)
            }
        }
    }

    /// Allocate a zeroed slice in the current frame.
    ///
    /// Panics if not in a frame.
    pub fn alloc_slice_zeroed<T: Copy>(&mut self, len: usize) -> AllocResult<&mut [T]> {
        assert_eq!(
            self.state,
            FrameState::InFrame,
            "allocation outside of frame"
        );

        match self.arena.alloc_slice_zeroed(len) {
            Ok(slice) => {
                self.current_frame_bytes += std::mem::size_of::<T>() * len;
                self.current_frame_allocs += 1;
                Ok(slice)
            }
            Err(e) => {
                self.stats.allocation_failures += 1;
                Err(e)
            }
        }
    }

    /// Allocate raw memory in the current frame.
    ///
    /// Panics if not in a frame.
    pub fn alloc_raw(&mut self, layout: Layout) -> AllocResult<NonNull<u8>> {
        assert_eq!(
            self.state,
            FrameState::InFrame,
            "allocation outside of frame"
        );

        match self.arena.alloc_raw(layout) {
            Ok(ptr) => {
                self.current_frame_bytes += layout.size();
                self.current_frame_allocs += 1;
                Ok(ptr)
            }
            Err(e) => {
                self.stats.allocation_failures += 1;
                Err(e)
            }
        }
    }

    /// Get the current frame number.
    #[must_use]
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Get the current frame state.
    #[must_use]
    pub fn state(&self) -> FrameState {
        self.state
    }

    /// Check if currently in a frame.
    #[must_use]
    pub fn in_frame(&self) -> bool {
        self.state == FrameState::InFrame
    }

    /// Get bytes used in the current frame.
    #[must_use]
    pub fn current_frame_bytes(&self) -> usize {
        self.current_frame_bytes
    }

    /// Get allocation count for the current frame.
    #[must_use]
    pub fn current_frame_allocations(&self) -> usize {
        self.current_frame_allocs
    }

    /// Get the total arena capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.arena.capacity()
    }

    /// Get the remaining capacity.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.arena.remaining()
    }

    /// Get frame statistics.
    #[must_use]
    pub fn stats(&self) -> &FrameStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = FrameStats::default();
    }
}

/// A double-buffered frame arena for data that persists across frames.
///
/// Maintains two arenas: current and previous. At frame end, the previous
/// arena is freed and the current becomes the new previous.
#[derive(Debug)]
pub struct DoubleBufferedFrameArena {
    /// The two arenas (ping-pong buffers).
    arenas: [HotArena; 2],
    /// Index of the current arena (0 or 1).
    current_index: usize,
    /// Current frame state.
    state: FrameState,
    /// Current frame number.
    frame_number: u64,
}

impl DoubleBufferedFrameArena {
    /// Create a new double-buffered frame arena.
    ///
    /// Each buffer has the specified capacity.
    #[must_use]
    pub fn new(capacity_per_buffer: usize) -> Self {
        Self {
            arenas: [
                HotArena::new(capacity_per_buffer),
                HotArena::new(capacity_per_buffer),
            ],
            current_index: 0,
            state: FrameState::Idle,
            frame_number: 0,
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        assert_eq!(
            self.state,
            FrameState::Idle,
            "begin_frame called while already in a frame"
        );

        self.state = FrameState::InFrame;
    }

    /// End the current frame.
    ///
    /// The previous frame's arena is freed, and the current arena
    /// becomes the new previous.
    pub fn end_frame(&mut self) {
        assert_eq!(
            self.state,
            FrameState::InFrame,
            "end_frame called without begin_frame"
        );

        // Swap buffers
        self.current_index = 1 - self.current_index;

        // Reset the new current arena (which was the previous)
        // Safety: The previous frame's data is being invalidated
        unsafe {
            self.arenas[self.current_index].reset();
        }

        self.state = FrameState::Idle;
        self.frame_number += 1;
    }

    /// Get the current arena for allocations.
    #[must_use]
    pub fn current(&self) -> &HotArena {
        assert_eq!(
            self.state,
            FrameState::InFrame,
            "current() called outside of frame"
        );
        &self.arenas[self.current_index]
    }

    /// Get the previous frame's arena (read-only).
    ///
    /// Returns None on the first frame.
    #[must_use]
    pub fn previous(&self) -> Option<&HotArena> {
        if self.frame_number == 0 {
            return None;
        }
        Some(&self.arenas[1 - self.current_index])
    }

    /// Get the current frame number.
    #[must_use]
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Check if in a frame.
    #[must_use]
    pub fn in_frame(&self) -> bool {
        self.state == FrameState::InFrame
    }
}

/// RAII guard for frame scope.
///
/// Automatically calls end_frame when dropped.
pub struct FrameGuard<'a> {
    arena: &'a mut FrameArena,
}

impl<'a> FrameGuard<'a> {
    /// Create a new frame guard.
    ///
    /// Calls begin_frame on the arena.
    pub fn new(arena: &'a mut FrameArena) -> Self {
        arena.begin_frame();
        Self { arena }
    }

    /// Allocate a value in this frame.
    pub fn alloc<T>(&mut self, value: T) -> AllocResult<&mut T> {
        self.arena.alloc(value)
    }

    /// Allocate a slice in this frame.
    pub fn alloc_slice<T: Copy>(&mut self, values: &[T]) -> AllocResult<&mut [T]> {
        self.arena.alloc_slice(values)
    }

    /// Get bytes used in this frame.
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        self.arena.current_frame_bytes()
    }
}

impl Drop for FrameGuard<'_> {
    fn drop(&mut self) {
        self.arena.end_frame();
    }
}

/// Execute a function within a frame scope.
pub fn with_frame<F, R>(arena: &mut FrameArena, f: F) -> R
where
    F: FnOnce(&mut FrameGuard<'_>) -> R,
{
    let mut guard = FrameGuard::new(arena);
    f(&mut guard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_arena_basic() {
        let mut arena = FrameArena::new(4096);

        assert_eq!(arena.state(), FrameState::Idle);
        assert_eq!(arena.frame_number(), 0);

        arena.begin_frame();
        assert_eq!(arena.state(), FrameState::InFrame);

        let x = arena.alloc(42i32).unwrap();
        assert_eq!(*x, 42);

        let slice = arena.alloc_slice(&[1, 2, 3]).unwrap();
        assert_eq!(slice, &[1, 2, 3]);

        assert!(arena.current_frame_bytes() > 0);
        assert_eq!(arena.current_frame_allocations(), 2);

        arena.end_frame();
        assert_eq!(arena.state(), FrameState::Idle);
        assert_eq!(arena.frame_number(), 1);
    }

    #[test]
    fn test_frame_arena_stats() {
        let mut arena = FrameArena::new(4096);

        // Frame 1
        arena.begin_frame();
        let _ = arena.alloc_slice(&[0u8; 100]).unwrap();
        arena.end_frame();

        // Frame 2
        arena.begin_frame();
        let _ = arena.alloc_slice(&[0u8; 200]).unwrap();
        arena.end_frame();

        let stats = arena.stats();
        assert_eq!(stats.frames_completed, 2);
        assert!(stats.max_frame_bytes >= 200);
        assert!(stats.min_frame_bytes.unwrap() >= 100);
    }

    #[test]
    #[should_panic(expected = "begin_frame called while already in a frame")]
    fn test_frame_arena_double_begin() {
        let mut arena = FrameArena::new(4096);
        arena.begin_frame();
        arena.begin_frame(); // Should panic
    }

    #[test]
    #[should_panic(expected = "end_frame called without begin_frame")]
    fn test_frame_arena_end_without_begin() {
        let mut arena = FrameArena::new(4096);
        arena.end_frame(); // Should panic
    }

    #[test]
    #[should_panic(expected = "allocation outside of frame")]
    fn test_frame_arena_alloc_outside_frame() {
        let mut arena = FrameArena::new(4096);
        let _ = arena.alloc(42i32); // Should panic
    }

    #[test]
    fn test_frame_guard() {
        let mut arena = FrameArena::new(4096);

        {
            let mut guard = FrameGuard::new(&mut arena);
            let _ = guard.alloc(42).unwrap();
            assert!(guard.bytes_used() > 0);
        }

        // Frame automatically ended
        assert_eq!(arena.state(), FrameState::Idle);
        assert_eq!(arena.frame_number(), 1);
    }

    #[test]
    fn test_with_frame() {
        let mut arena = FrameArena::new(4096);

        let result = with_frame(&mut arena, |frame| {
            let x = frame.alloc(42).unwrap();
            *x * 2
        });

        assert_eq!(result, 84);
        assert_eq!(arena.frame_number(), 1);
    }

    #[test]
    fn test_double_buffered_arena() {
        let mut arena = DoubleBufferedFrameArena::new(4096);

        // Frame 0
        arena.begin_frame();
        assert!(arena.previous().is_none()); // No previous on first frame

        let _ = arena.current().alloc(42i32).unwrap();
        arena.end_frame();

        // Frame 1
        arena.begin_frame();
        assert!(arena.previous().is_some()); // Now we have previous

        // Previous frame's data is still accessible
        let prev = arena.previous().unwrap();
        assert!(prev.used() > 0);

        let _ = arena.current().alloc(100i32).unwrap();
        arena.end_frame();

        // Frame 2
        arena.begin_frame();
        // Previous is now frame 1's data
        arena.end_frame();

        assert_eq!(arena.frame_number(), 3);
    }

    #[test]
    fn test_many_frames() {
        let mut arena = FrameArena::new(4096);

        for i in 0..1000 {
            arena.begin_frame();

            // Allocate some data each frame
            let _ = arena.alloc_slice(&[i as u8; 10]).unwrap();

            arena.end_frame();
        }

        assert_eq!(arena.frame_number(), 1000);
        assert_eq!(arena.stats().frames_completed, 1000);
    }
}
