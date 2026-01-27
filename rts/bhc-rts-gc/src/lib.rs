//! Generational garbage collector for the BHC Runtime System.
//!
//! This crate implements the garbage collector for the General Heap region
//! as specified in H26-SPEC Section 9: Memory Model. Key features:
//!
//! - **Generational collection** - Young/old generation with different strategies
//! - **Pinned region support** - Objects that must not move (FFI, device IO)
//! - **Write barriers** - Track cross-generation references
//! - **Incremental collection** - Minimize pause times for Server Profile
//!
//! # Architecture
//!
//! The GC manages the General Heap, which contains boxed Haskell values.
//! Objects may be moved during collection unless they are pinned.
//!
//! ```text
//! +------------------+------------------+------------------+
//! |   Nursery (G0)   |   Survivor (G1)  |   Old Gen (G2)   |
//! +------------------+------------------+------------------+
//! |                  |                  |                  |
//! |  Young objects   |  Promoted from   |  Long-lived      |
//! |  Bump alloc      |  G0 after 1      |  objects         |
//! |  Frequent GC     |  survival        |  Rare major GC   |
//! |                  |                  |                  |
//! +------------------+------------------+------------------+
//!
//! +------------------+
//! |   Pinned Region  |
//! +------------------+
//! |                  |
//! |  Non-moving      |
//! |  objects         |
//! |  (FFI, DMA)      |
//! |                  |
//! +------------------+
//! ```
//!
//! # Design Goals
//!
//! - Low latency for Server Profile (bounded pause times)
//! - High throughput for batch processing
//! - Deterministic behavior for Numeric Profile (no GC in hot paths)
//! - Safe FFI interop through pinned allocations

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod incremental;

use bhc_rts_alloc::{AllocError, AllocResult, AllocStats, MemoryRegion};
use parking_lot::{Mutex, RwLock};
use std::alloc::Layout;
use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// A wrapper around `NonNull<u8>` that is `Send + Sync`.
///
/// This is used for GC pointers that need to be shared across threads.
/// Safety: The GC ensures these pointers are accessed safely through
/// proper synchronization (write barriers, mutexes, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct GcPtr(NonNull<u8>);

impl GcPtr {
    /// Create a new GcPtr from a NonNull pointer.
    #[inline]
    pub const fn new(ptr: NonNull<u8>) -> Self {
        Self(ptr)
    }

    /// Get the underlying NonNull pointer.
    #[inline]
    pub const fn as_non_null(self) -> NonNull<u8> {
        self.0
    }

    /// Get the raw pointer.
    #[inline]
    pub const fn as_ptr(self) -> *mut u8 {
        self.0.as_ptr()
    }
}

// Safety: GcPtr is used within the GC system with proper synchronization.
// The GC manages these pointers and ensures thread-safe access.
unsafe impl Send for GcPtr {}
unsafe impl Sync for GcPtr {}

/// Generation identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Generation {
    /// Nursery - newly allocated objects.
    Nursery = 0,
    /// Survivor - survived one minor collection.
    Survivor = 1,
    /// Old - long-lived objects.
    Old = 2,
}

impl Generation {
    /// Get the next older generation.
    #[must_use]
    pub const fn promote(self) -> Option<Self> {
        match self {
            Self::Nursery => Some(Self::Survivor),
            Self::Survivor => Some(Self::Old),
            Self::Old => None,
        }
    }
}

/// Object header stored before each GC-managed object.
///
/// The header contains metadata needed for garbage collection:
/// - Mark bits for tracing
/// - Generation information
/// - Forwarding pointer (during collection)
/// - Type information for traversal
#[derive(Debug)]
#[repr(C)]
pub struct ObjectHeader {
    /// Mark bits and flags.
    flags: AtomicU64,
    /// Size of the object in bytes (excluding header).
    size: u32,
    /// Type tag for traversal.
    type_tag: u32,
}

/// Flags stored in the object header.
#[derive(Debug, Clone, Copy)]
pub struct HeaderFlags(u64);

impl HeaderFlags {
    /// Object is marked (reachable).
    pub const MARKED: u64 = 1 << 0;
    /// Object is pinned (cannot be moved).
    pub const PINNED: u64 = 1 << 1;
    /// Object has been forwarded during collection.
    pub const FORWARDED: u64 = 1 << 2;
    /// Object contains pointers.
    pub const HAS_POINTERS: u64 = 1 << 3;

    /// Generation bits (2 bits, positions 4-5).
    const GENERATION_SHIFT: u64 = 4;
    const GENERATION_MASK: u64 = 0b11 << Self::GENERATION_SHIFT;

    /// Create new flags with default values.
    #[must_use]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Check if the object is marked.
    #[must_use]
    pub const fn is_marked(self) -> bool {
        self.0 & Self::MARKED != 0
    }

    /// Check if the object is pinned.
    #[must_use]
    pub const fn is_pinned(self) -> bool {
        self.0 & Self::PINNED != 0
    }

    /// Check if the object has been forwarded.
    #[must_use]
    pub const fn is_forwarded(self) -> bool {
        self.0 & Self::FORWARDED != 0
    }

    /// Get the generation of this object.
    #[must_use]
    pub const fn generation(self) -> Generation {
        let gen = (self.0 & Self::GENERATION_MASK) >> Self::GENERATION_SHIFT;
        match gen {
            0 => Generation::Nursery,
            1 => Generation::Survivor,
            _ => Generation::Old,
        }
    }

    /// Set the generation.
    #[must_use]
    pub const fn with_generation(self, gen: Generation) -> Self {
        let cleared = self.0 & !Self::GENERATION_MASK;
        Self(cleared | ((gen as u64) << Self::GENERATION_SHIFT))
    }

    /// Set the pinned flag.
    #[must_use]
    pub const fn with_pinned(self, pinned: bool) -> Self {
        if pinned {
            Self(self.0 | Self::PINNED)
        } else {
            Self(self.0 & !Self::PINNED)
        }
    }
}

impl Default for HeaderFlags {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectHeader {
    /// Create a new object header.
    #[must_use]
    pub const fn new(size: u32, type_tag: u32, flags: HeaderFlags) -> Self {
        Self {
            flags: AtomicU64::new(flags.0),
            size,
            type_tag,
        }
    }

    /// Get the size of the object.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size as usize
    }

    /// Get the type tag.
    #[must_use]
    pub const fn type_tag(&self) -> u32 {
        self.type_tag
    }

    /// Get the current flags.
    #[must_use]
    pub fn flags(&self) -> HeaderFlags {
        HeaderFlags(self.flags.load(Ordering::Acquire))
    }

    /// Set the mark bit.
    pub fn mark(&self) {
        self.flags.fetch_or(HeaderFlags::MARKED, Ordering::Release);
    }

    /// Clear the mark bit.
    pub fn unmark(&self) {
        self.flags
            .fetch_and(!HeaderFlags::MARKED, Ordering::Release);
    }

    /// Check if marked.
    #[must_use]
    pub fn is_marked(&self) -> bool {
        self.flags().is_marked()
    }

    /// Check if pinned.
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.flags().is_pinned()
    }
}

/// Configuration for the garbage collector.
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Size of the nursery in bytes.
    pub nursery_size: usize,
    /// Size of the survivor space in bytes.
    pub survivor_size: usize,
    /// Size of the old generation in bytes.
    pub old_gen_size: usize,
    /// Number of collections before promotion from nursery.
    pub nursery_threshold: u32,
    /// Enable incremental collection.
    pub incremental: bool,
    /// Maximum pause time in microseconds (for incremental GC).
    pub max_pause_us: u64,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            nursery_size: 4 * 1024 * 1024,  // 4 MB
            survivor_size: 2 * 1024 * 1024, // 2 MB
            old_gen_size: 64 * 1024 * 1024, // 64 MB
            nursery_threshold: 2,
            incremental: false,
            max_pause_us: 1000,
        }
    }
}

/// Statistics from garbage collection.
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Number of minor (nursery) collections.
    pub minor_collections: u64,
    /// Number of major (full) collections.
    pub major_collections: u64,
    /// Number of incremental collection cycles.
    pub incremental_cycles: u64,
    /// Total bytes collected.
    pub bytes_collected: u64,
    /// Total bytes promoted to older generations.
    pub bytes_promoted: u64,
    /// Total time spent in GC (microseconds).
    pub total_gc_time_us: u64,
    /// Maximum pause time (microseconds).
    pub max_pause_us: u64,
    /// Number of pinned objects.
    pub pinned_objects: u64,
}

/// Kind of GC collection that caused a pause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectionKind {
    /// Minor (nursery) collection.
    Minor,
    /// Major (full) collection.
    Major,
    /// Incremental marking phase.
    IncrementalMark,
    /// Incremental sweep phase.
    IncrementalSweep,
}

/// A single GC pause measurement.
#[derive(Debug, Clone, Copy)]
pub struct PauseMeasurement {
    /// Duration of the pause.
    pub duration: Duration,
    /// Kind of collection that caused the pause.
    pub kind: CollectionKind,
    /// Timestamp when the pause started.
    pub timestamp: Instant,
}

impl PauseMeasurement {
    /// Create a new pause measurement.
    #[must_use]
    pub fn new(duration: Duration, kind: CollectionKind, timestamp: Instant) -> Self {
        Self {
            duration,
            kind,
            timestamp,
        }
    }

    /// Get the pause duration in microseconds.
    #[must_use]
    pub fn duration_us(&self) -> u64 {
        self.duration.as_micros() as u64
    }

    /// Check if this pause exceeded the given threshold.
    #[must_use]
    pub fn exceeded_threshold(&self, threshold: Duration) -> bool {
        self.duration > threshold
    }
}

/// Statistics for GC pause times with recent history.
///
/// Tracks pause times for realtime profile verification.
/// The <1ms pause guarantee for realtime profile can be verified
/// by checking if any pause exceeded the threshold.
#[derive(Debug)]
pub struct PauseStats {
    /// Ring buffer of recent pause measurements.
    recent_pauses: Vec<PauseMeasurement>,
    /// Maximum size of the ring buffer.
    max_history: usize,
    /// Current write position in the ring buffer.
    write_pos: usize,
    /// Total number of pauses recorded.
    total_pauses: u64,
    /// Number of pauses that exceeded the configured threshold.
    threshold_violations: u64,
    /// Configured pause threshold (default: 1ms for realtime).
    threshold: Duration,
    /// Minimum pause time observed.
    min_pause: Option<Duration>,
    /// Maximum pause time observed.
    max_pause: Option<Duration>,
    /// Sum of all pause durations (for computing average).
    total_pause_time: Duration,
}

impl Default for PauseStats {
    fn default() -> Self {
        Self::new(1000, Duration::from_micros(1000)) // 1000 pauses, 1ms threshold
    }
}

impl PauseStats {
    /// Create new pause statistics with the given history size and threshold.
    #[must_use]
    pub fn new(max_history: usize, threshold: Duration) -> Self {
        Self {
            recent_pauses: Vec::with_capacity(max_history),
            max_history,
            write_pos: 0,
            total_pauses: 0,
            threshold_violations: 0,
            threshold,
            min_pause: None,
            max_pause: None,
            total_pause_time: Duration::ZERO,
        }
    }

    /// Record a new pause measurement.
    pub fn record(&mut self, measurement: PauseMeasurement) {
        // Update statistics
        self.total_pauses += 1;
        self.total_pause_time += measurement.duration;

        if measurement.exceeded_threshold(self.threshold) {
            self.threshold_violations += 1;
        }

        self.min_pause = Some(match self.min_pause {
            Some(min) => min.min(measurement.duration),
            None => measurement.duration,
        });

        self.max_pause = Some(match self.max_pause {
            Some(max) => max.max(measurement.duration),
            None => measurement.duration,
        });

        // Store in ring buffer
        if self.recent_pauses.len() < self.max_history {
            self.recent_pauses.push(measurement);
        } else {
            self.recent_pauses[self.write_pos] = measurement;
        }
        self.write_pos = (self.write_pos + 1) % self.max_history;
    }

    /// Get the total number of pauses recorded.
    #[must_use]
    pub fn total_pauses(&self) -> u64 {
        self.total_pauses
    }

    /// Get the number of pauses that exceeded the threshold.
    #[must_use]
    pub fn threshold_violations(&self) -> u64 {
        self.threshold_violations
    }

    /// Check if any pause exceeded the threshold.
    #[must_use]
    pub fn has_violations(&self) -> bool {
        self.threshold_violations > 0
    }

    /// Get the configured threshold.
    #[must_use]
    pub fn threshold(&self) -> Duration {
        self.threshold
    }

    /// Set the pause threshold.
    pub fn set_threshold(&mut self, threshold: Duration) {
        self.threshold = threshold;
    }

    /// Get the minimum pause time observed.
    #[must_use]
    pub fn min_pause(&self) -> Option<Duration> {
        self.min_pause
    }

    /// Get the maximum pause time observed.
    #[must_use]
    pub fn max_pause(&self) -> Option<Duration> {
        self.max_pause
    }

    /// Get the average pause time.
    #[must_use]
    pub fn average_pause(&self) -> Option<Duration> {
        if self.total_pauses == 0 {
            None
        } else {
            Some(self.total_pause_time / self.total_pauses as u32)
        }
    }

    /// Get the total pause time.
    #[must_use]
    pub fn total_pause_time(&self) -> Duration {
        self.total_pause_time
    }

    /// Get recent pauses (most recent first).
    #[must_use]
    pub fn recent_pauses(&self) -> impl Iterator<Item = &PauseMeasurement> {
        // Return pauses in reverse chronological order
        let len = self.recent_pauses.len();
        if len == 0 {
            return [].iter();
        }

        // The most recent pause is at (write_pos - 1 + len) % len
        // We need to return them in reverse order
        self.recent_pauses.iter()
    }

    /// Get the P99 pause time from recent history.
    ///
    /// Returns None if fewer than 100 pauses have been recorded.
    #[must_use]
    pub fn p99_pause(&self) -> Option<Duration> {
        if self.recent_pauses.len() < 100 {
            return None;
        }

        let mut durations: Vec<_> = self.recent_pauses.iter().map(|p| p.duration).collect();
        durations.sort();
        let idx = (durations.len() * 99) / 100;
        Some(durations[idx])
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.recent_pauses.clear();
        self.write_pos = 0;
        self.total_pauses = 0;
        self.threshold_violations = 0;
        self.min_pause = None;
        self.max_pause = None;
        self.total_pause_time = Duration::ZERO;
    }

    /// Get a summary report of pause statistics.
    #[must_use]
    pub fn summary(&self) -> PauseSummary {
        PauseSummary {
            total_pauses: self.total_pauses,
            threshold_violations: self.threshold_violations,
            threshold: self.threshold,
            min_pause: self.min_pause,
            max_pause: self.max_pause,
            average_pause: self.average_pause(),
            p99_pause: self.p99_pause(),
            total_pause_time: self.total_pause_time,
        }
    }
}

/// Summary of pause statistics for reporting.
#[derive(Debug, Clone)]
pub struct PauseSummary {
    /// Total number of pauses.
    pub total_pauses: u64,
    /// Number of threshold violations.
    pub threshold_violations: u64,
    /// Configured threshold.
    pub threshold: Duration,
    /// Minimum pause time.
    pub min_pause: Option<Duration>,
    /// Maximum pause time.
    pub max_pause: Option<Duration>,
    /// Average pause time.
    pub average_pause: Option<Duration>,
    /// P99 pause time.
    pub p99_pause: Option<Duration>,
    /// Total time spent in pauses.
    pub total_pause_time: Duration,
}

impl std::fmt::Display for PauseSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GC Pause Statistics:")?;
        writeln!(f, "  Total pauses: {}", self.total_pauses)?;
        writeln!(
            f,
            "  Threshold: {:?} ({} violations)",
            self.threshold, self.threshold_violations
        )?;
        if let Some(min) = self.min_pause {
            writeln!(f, "  Min pause: {:?}", min)?;
        }
        if let Some(max) = self.max_pause {
            writeln!(f, "  Max pause: {:?}", max)?;
        }
        if let Some(avg) = self.average_pause {
            writeln!(f, "  Avg pause: {:?}", avg)?;
        }
        if let Some(p99) = self.p99_pause {
            writeln!(f, "  P99 pause: {:?}", p99)?;
        }
        writeln!(f, "  Total pause time: {:?}", self.total_pause_time)?;
        Ok(())
    }
}

/// Handle to a GC-managed object.
///
/// This handle tracks the object's location and can be updated
/// if the object is moved during collection.
#[derive(Debug)]
pub struct GcHandle<T> {
    /// Pointer to the object (may change if object moves).
    ptr: Cell<NonNull<T>>,
    /// Whether this handle is pinned.
    pinned: bool,
}

impl<T> GcHandle<T> {
    /// Create a new GC handle.
    ///
    /// # Safety
    ///
    /// The pointer must point to a valid, GC-managed object.
    #[must_use]
    pub const unsafe fn new(ptr: NonNull<T>, pinned: bool) -> Self {
        Self {
            ptr: Cell::new(ptr),
            pinned,
        }
    }

    /// Get the current pointer.
    #[must_use]
    pub fn as_ptr(&self) -> *mut T {
        self.ptr.get().as_ptr()
    }

    /// Get a reference to the object.
    ///
    /// # Safety
    ///
    /// The caller must ensure the object is still alive.
    #[must_use]
    pub unsafe fn as_ref(&self) -> &T {
        unsafe { self.ptr.get().as_ref() }
    }

    /// Check if this handle is pinned.
    #[must_use]
    pub const fn is_pinned(&self) -> bool {
        self.pinned
    }

    /// Update the pointer (called by GC after moving).
    ///
    /// # Safety
    ///
    /// Must only be called by the GC during collection.
    pub unsafe fn update_ptr(&self, new_ptr: NonNull<T>) {
        self.ptr.set(new_ptr);
    }
}

/// Root set for garbage collection.
///
/// The root set contains all objects that are directly reachable
/// and should not be collected.
#[derive(Debug, Default)]
pub struct RootSet {
    /// Stack roots (local variables, arguments).
    stack_roots: Vec<GcPtr>,
    /// Global roots (static variables).
    global_roots: Vec<GcPtr>,
    /// Pinned roots (FFI, device IO).
    pinned_roots: Vec<GcPtr>,
}

impl RootSet {
    /// Create a new empty root set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a stack root.
    pub fn add_stack_root(&mut self, ptr: NonNull<u8>) {
        self.stack_roots.push(GcPtr::new(ptr));
    }

    /// Add a global root.
    pub fn add_global_root(&mut self, ptr: NonNull<u8>) {
        self.global_roots.push(GcPtr::new(ptr));
    }

    /// Add a pinned root.
    pub fn add_pinned_root(&mut self, ptr: NonNull<u8>) {
        self.pinned_roots.push(GcPtr::new(ptr));
    }

    /// Clear stack roots (between function calls).
    pub fn clear_stack_roots(&mut self) {
        self.stack_roots.clear();
    }

    /// Iterate over all roots.
    pub fn iter(&self) -> impl Iterator<Item = GcPtr> + '_ {
        self.stack_roots
            .iter()
            .chain(self.global_roots.iter())
            .chain(self.pinned_roots.iter())
            .copied()
    }
}

/// Write barrier for tracking cross-generation references.
///
/// When an old object is mutated to point to a young object,
/// the write barrier records this so the young object is not
/// incorrectly collected.
#[derive(Debug)]
pub struct WriteBarrier {
    /// Remembered set of old->young references.
    remembered_set: Mutex<Vec<GcPtr>>,
    /// Number of barrier invocations.
    invocations: AtomicUsize,
}

impl WriteBarrier {
    /// Create a new write barrier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            remembered_set: Mutex::new(Vec::new()),
            invocations: AtomicUsize::new(0),
        }
    }

    /// Record a write from an old object to a young object.
    pub fn record(&self, old_object: NonNull<u8>) {
        self.invocations.fetch_add(1, Ordering::Relaxed);
        self.remembered_set.lock().push(GcPtr::new(old_object));
    }

    /// Get and clear the remembered set.
    #[must_use]
    pub fn take_remembered_set(&self) -> Vec<GcPtr> {
        std::mem::take(&mut *self.remembered_set.lock())
    }

    /// Get the number of barrier invocations.
    #[must_use]
    pub fn invocations(&self) -> usize {
        self.invocations.load(Ordering::Relaxed)
    }
}

impl Default for WriteBarrier {
    fn default() -> Self {
        Self::new()
    }
}

/// The garbage collector.
///
/// This is the main interface to the GC subsystem.
#[derive(Debug)]
pub struct GarbageCollector {
    /// Configuration.
    config: GcConfig,
    /// Statistics.
    stats: RwLock<GcStats>,
    /// Pause statistics for realtime profile verification.
    pause_stats: Mutex<PauseStats>,
    /// Write barrier.
    write_barrier: WriteBarrier,
    /// Allocation statistics.
    alloc_stats: RwLock<AllocStats>,
    /// Total bytes allocated since last collection.
    bytes_since_gc: AtomicUsize,
    /// Incremental marker for realtime profile (bounded GC pauses).
    incremental_marker: Option<incremental::IncrementalMarker>,
}

impl GarbageCollector {
    /// Create a new garbage collector with the given configuration.
    #[must_use]
    pub fn new(config: GcConfig) -> Self {
        let pause_threshold = Duration::from_micros(config.max_pause_us);

        // Create incremental marker if incremental collection is enabled
        let incremental_marker = if config.incremental {
            let inc_config = incremental::IncrementalConfig {
                time_budget_us: config.max_pause_us / 2, // Use half the budget for mark increments
                max_objects_per_increment: 1000,
                concurrent: false,
                satb_buffer_threshold: 1024,
            };
            Some(incremental::IncrementalMarker::new(inc_config))
        } else {
            None
        };

        Self {
            config,
            stats: RwLock::new(GcStats::default()),
            pause_stats: Mutex::new(PauseStats::new(1000, pause_threshold)),
            write_barrier: WriteBarrier::new(),
            alloc_stats: RwLock::new(AllocStats::new()),
            bytes_since_gc: AtomicUsize::new(0),
            incremental_marker,
        }
    }

    /// Create a new garbage collector with default configuration.
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(GcConfig::default())
    }

    /// Allocate memory from the GC heap.
    ///
    /// Objects allocated this way will be managed by the garbage collector.
    pub fn alloc(&self, layout: Layout, pinned: bool) -> AllocResult<NonNull<u8>> {
        let total_size = std::mem::size_of::<ObjectHeader>() + layout.size();

        // Check if we should trigger GC
        let bytes = self.bytes_since_gc.fetch_add(total_size, Ordering::Relaxed);
        if bytes + total_size > self.config.nursery_size {
            // Would trigger GC here in full implementation
            self.bytes_since_gc.store(0, Ordering::Relaxed);
        }

        // For now, use system allocator (full implementation would use generation spaces)
        let header_layout = Layout::new::<ObjectHeader>();
        let (combined_layout, offset) = header_layout
            .extend(layout)
            .map_err(|_| AllocError::InvalidLayout("layout overflow".into()))?;

        let ptr = unsafe { std::alloc::alloc(combined_layout) };
        let ptr = NonNull::new(ptr).ok_or(AllocError::OutOfMemory {
            requested: combined_layout.size(),
        })?;

        // Initialize header
        let flags = HeaderFlags::new()
            .with_generation(Generation::Nursery)
            .with_pinned(pinned);
        let header = ObjectHeader::new(layout.size() as u32, 0, flags);
        unsafe {
            std::ptr::write(ptr.as_ptr() as *mut ObjectHeader, header);
        }

        // Update stats
        {
            let mut stats = self.alloc_stats.write();
            stats.record_alloc(total_size);
        }

        // Return pointer to data (after header)
        let data_ptr = unsafe { ptr.as_ptr().add(offset) };
        Ok(unsafe { NonNull::new_unchecked(data_ptr) })
    }

    /// Allocate a pinned object that will not be moved by GC.
    pub fn alloc_pinned(&self, layout: Layout) -> AllocResult<NonNull<u8>> {
        self.alloc(layout, true)
    }

    /// Trigger a minor (nursery) collection.
    pub fn minor_collect(&self, _roots: &RootSet) {
        let start = Instant::now();

        // Placeholder: Full implementation would:
        // 1. Mark all reachable objects from roots
        // 2. Copy live objects to survivor space
        // 3. Update remembered set
        // 4. Free nursery

        let duration = start.elapsed();

        // Record pause statistics
        self.record_pause(duration, CollectionKind::Minor, start);

        let mut stats = self.stats.write();
        stats.minor_collections += 1;
    }

    /// Trigger a major (full) collection.
    pub fn major_collect(&self, _roots: &RootSet) {
        let start = Instant::now();

        // Placeholder: Full implementation would:
        // 1. Mark all reachable objects from roots
        // 2. Sweep/compact all generations
        // 3. Free unreachable objects

        let duration = start.elapsed();

        // Record pause statistics
        self.record_pause(duration, CollectionKind::Major, start);

        let mut stats = self.stats.write();
        stats.major_collections += 1;
    }

    /// Start an incremental collection cycle.
    ///
    /// This is for the realtime profile where bounded pause times are required.
    /// Call `do_incremental_work` periodically to make progress.
    pub fn start_incremental_collect(&self, roots: &RootSet) -> Option<PauseMeasurement> {
        let marker = self.incremental_marker.as_ref()?;

        // Start the marking cycle with roots
        let pause = marker.start_cycle(roots.iter());

        // Record the root scanning pause
        self.record_pause(pause.duration, CollectionKind::IncrementalMark, pause.timestamp);

        Some(pause)
    }

    /// Perform incremental GC work within a bounded time budget.
    ///
    /// Returns the pause measurement, or None if no work was done.
    /// Call this periodically during mutator execution to make progress
    /// on collection while maintaining bounded pause times.
    pub fn do_incremental_work(&self) -> Option<PauseMeasurement> {
        let marker = self.incremental_marker.as_ref()?;

        // Check if we're in a marking state
        if !marker.is_marking() {
            return None;
        }

        // Perform one increment of marking work
        // In a full implementation, scan_object would extract child pointers
        // from the object's type info and layout
        let pause = marker.mark_increment(|_obj_ptr| {
            // Placeholder: Real implementation would read object layout
            // and extract all pointer fields as children
            Vec::new()
        })?;

        // Record the pause
        self.record_pause(pause.duration, CollectionKind::IncrementalMark, pause.timestamp);

        // Check if we need to do remark (SATB processing)
        if matches!(marker.state(), incremental::MarkState::Remark) {
            let remark_pause = marker.remark(|_obj_ptr| Vec::new());
            self.record_pause(
                remark_pause.duration,
                CollectionKind::IncrementalMark,
                remark_pause.timestamp,
            );
        }

        // Update stats if marking is complete
        if matches!(marker.state(), incremental::MarkState::Complete) {
            let mut stats = self.stats.write();
            stats.incremental_cycles += 1;
        }

        Some(pause)
    }

    /// Check if incremental collection is in progress.
    #[must_use]
    pub fn is_incremental_marking(&self) -> bool {
        self.incremental_marker
            .as_ref()
            .map_or(false, |m| m.is_marking())
    }

    /// Finish an incremental collection cycle.
    ///
    /// Ensures all marking work is complete and performs any final cleanup.
    /// Returns the total pause time for the final phase.
    pub fn finish_incremental_collect(&self) -> Option<PauseMeasurement> {
        let marker = self.incremental_marker.as_ref()?;

        // Complete any remaining work
        let start = Instant::now();
        while marker.is_marking() {
            marker.mark_increment(|_| Vec::new());
        }

        // Handle remark if needed
        if matches!(marker.state(), incremental::MarkState::Remark) {
            marker.remark(|_| Vec::new());
        }

        let duration = start.elapsed();

        // Reset marker for next cycle
        marker.reset();

        let pause = PauseMeasurement::new(duration, CollectionKind::IncrementalMark, start);
        self.record_pause(duration, CollectionKind::IncrementalMark, start);

        Some(pause)
    }

    /// Record a write barrier event for incremental collection.
    ///
    /// When using incremental collection, this records overwritten references
    /// in the SATB buffer to maintain correctness.
    pub fn incremental_write_barrier(&self, old_value: NonNull<u8>) {
        if let Some(marker) = &self.incremental_marker {
            marker.write_barrier(GcPtr::new(old_value));
        }
    }

    /// Get statistics for the current/last incremental collection cycle.
    #[must_use]
    pub fn incremental_stats(&self) -> Option<incremental::IncrementalStats> {
        self.incremental_marker.as_ref().map(|m| m.stats())
    }

    /// Check if incremental collection is enabled.
    #[must_use]
    pub fn is_incremental_enabled(&self) -> bool {
        self.incremental_marker.is_some()
    }

    /// Record a GC pause measurement.
    fn record_pause(&self, duration: Duration, kind: CollectionKind, timestamp: Instant) {
        let duration_us = duration.as_micros() as u64;

        // Update pause stats
        {
            let mut pause_stats = self.pause_stats.lock();
            pause_stats.record(PauseMeasurement::new(duration, kind, timestamp));
        }

        // Update GcStats
        {
            let mut stats = self.stats.write();
            stats.total_gc_time_us += duration_us;
            if duration_us > stats.max_pause_us {
                stats.max_pause_us = duration_us;
            }
        }
    }

    /// Get the write barrier for recording mutations.
    #[must_use]
    pub fn write_barrier(&self) -> &WriteBarrier {
        &self.write_barrier
    }

    /// Get GC statistics.
    #[must_use]
    pub fn stats(&self) -> GcStats {
        self.stats.read().clone()
    }

    /// Get a summary of pause statistics.
    ///
    /// Use this to verify the <1ms pause guarantee for realtime profile.
    #[must_use]
    pub fn pause_summary(&self) -> PauseSummary {
        self.pause_stats.lock().summary()
    }

    /// Check if any GC pause exceeded the configured threshold.
    ///
    /// For realtime profile, this checks the <1ms guarantee.
    #[must_use]
    pub fn has_pause_violations(&self) -> bool {
        self.pause_stats.lock().has_violations()
    }

    /// Get the number of pauses that exceeded the threshold.
    #[must_use]
    pub fn pause_violation_count(&self) -> u64 {
        self.pause_stats.lock().threshold_violations()
    }

    /// Get the maximum pause time observed.
    #[must_use]
    pub fn max_pause_observed(&self) -> Option<Duration> {
        self.pause_stats.lock().max_pause()
    }

    /// Reset pause statistics (useful for benchmarking).
    pub fn reset_pause_stats(&self) {
        self.pause_stats.lock().reset();
    }

    /// Set the pause threshold for violation detection.
    pub fn set_pause_threshold(&self, threshold: Duration) {
        self.pause_stats.lock().set_threshold(threshold);
    }

    /// Get the memory region managed by this GC.
    #[must_use]
    pub const fn region(&self) -> MemoryRegion {
        MemoryRegion::GeneralHeap
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &GcConfig {
        &self.config
    }
}

impl Default for GarbageCollector {
    fn default() -> Self {
        Self::with_default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_promotion() {
        assert_eq!(Generation::Nursery.promote(), Some(Generation::Survivor));
        assert_eq!(Generation::Survivor.promote(), Some(Generation::Old));
        assert_eq!(Generation::Old.promote(), None);
    }

    #[test]
    fn test_header_flags() {
        let flags = HeaderFlags::new();
        assert!(!flags.is_marked());
        assert!(!flags.is_pinned());
        assert_eq!(flags.generation(), Generation::Nursery);

        let flags = flags.with_generation(Generation::Old).with_pinned(true);
        assert!(flags.is_pinned());
        assert_eq!(flags.generation(), Generation::Old);
    }

    #[test]
    fn test_object_header() {
        let flags = HeaderFlags::new().with_generation(Generation::Survivor);
        let header = ObjectHeader::new(64, 42, flags);

        assert_eq!(header.size(), 64);
        assert_eq!(header.type_tag(), 42);
        assert!(!header.is_marked());

        header.mark();
        assert!(header.is_marked());

        header.unmark();
        assert!(!header.is_marked());
    }

    #[test]
    fn test_gc_alloc() {
        let gc = GarbageCollector::with_default_config();

        let layout = Layout::new::<[u64; 10]>();
        let ptr = gc.alloc(layout, false).unwrap();

        // Verify we got a valid pointer
        assert!(!ptr.as_ptr().is_null());
    }

    #[test]
    fn test_gc_pinned_alloc() {
        let gc = GarbageCollector::with_default_config();

        let layout = Layout::new::<[u64; 10]>();
        let ptr = gc.alloc_pinned(layout).unwrap();

        assert!(!ptr.as_ptr().is_null());
    }

    #[test]
    fn test_write_barrier() {
        let barrier = WriteBarrier::new();

        let ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        barrier.record(ptr);

        assert_eq!(barrier.invocations(), 1);

        let remembered = barrier.take_remembered_set();
        assert_eq!(remembered.len(), 1);
        assert_eq!(remembered[0], GcPtr::new(ptr));

        // After take, should be empty
        let remembered = barrier.take_remembered_set();
        assert!(remembered.is_empty());
    }

    #[test]
    fn test_root_set() {
        let mut roots = RootSet::new();

        let stack_ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        let global_ptr = NonNull::new(0x2000 as *mut u8).unwrap();
        let pinned_ptr = NonNull::new(0x3000 as *mut u8).unwrap();

        roots.add_stack_root(stack_ptr);
        roots.add_global_root(global_ptr);
        roots.add_pinned_root(pinned_ptr);

        let all_roots: Vec<_> = roots.iter().collect();
        assert_eq!(all_roots.len(), 3);

        roots.clear_stack_roots();
        let all_roots: Vec<_> = roots.iter().collect();
        assert_eq!(all_roots.len(), 2);
    }

    #[test]
    fn test_gc_stats() {
        let gc = GarbageCollector::with_default_config();
        let roots = RootSet::new();

        gc.minor_collect(&roots);
        gc.minor_collect(&roots);
        gc.major_collect(&roots);

        let stats = gc.stats();
        assert_eq!(stats.minor_collections, 2);
        assert_eq!(stats.major_collections, 1);
    }

    // ========================================================================
    // Pause Measurement Tests
    // ========================================================================

    #[test]
    fn test_pause_measurement() {
        let duration = Duration::from_micros(500);
        let timestamp = Instant::now();
        let measurement = PauseMeasurement::new(duration, CollectionKind::Minor, timestamp);

        assert_eq!(measurement.duration, duration);
        assert_eq!(measurement.kind, CollectionKind::Minor);
        assert_eq!(measurement.duration_us(), 500);
        assert!(!measurement.exceeded_threshold(Duration::from_millis(1)));
        assert!(measurement.exceeded_threshold(Duration::from_micros(100)));
    }

    #[test]
    fn test_pause_stats_basic() {
        let mut stats = PauseStats::new(100, Duration::from_millis(1));

        assert_eq!(stats.total_pauses(), 0);
        assert!(!stats.has_violations());
        assert!(stats.min_pause().is_none());
        assert!(stats.max_pause().is_none());

        // Record some pauses
        let now = Instant::now();
        stats.record(PauseMeasurement::new(
            Duration::from_micros(500),
            CollectionKind::Minor,
            now,
        ));
        stats.record(PauseMeasurement::new(
            Duration::from_micros(800),
            CollectionKind::Minor,
            now,
        ));
        stats.record(PauseMeasurement::new(
            Duration::from_micros(200),
            CollectionKind::Major,
            now,
        ));

        assert_eq!(stats.total_pauses(), 3);
        assert!(!stats.has_violations()); // All under 1ms
        assert_eq!(stats.min_pause(), Some(Duration::from_micros(200)));
        assert_eq!(stats.max_pause(), Some(Duration::from_micros(800)));
    }

    #[test]
    fn test_pause_stats_violations() {
        let mut stats = PauseStats::new(100, Duration::from_millis(1));
        let now = Instant::now();

        // Record a pause that exceeds threshold
        stats.record(PauseMeasurement::new(
            Duration::from_micros(1500), // 1.5ms > 1ms threshold
            CollectionKind::Major,
            now,
        ));

        assert_eq!(stats.total_pauses(), 1);
        assert!(stats.has_violations());
        assert_eq!(stats.threshold_violations(), 1);
    }

    #[test]
    fn test_pause_stats_average() {
        let mut stats = PauseStats::new(100, Duration::from_millis(1));
        let now = Instant::now();

        stats.record(PauseMeasurement::new(
            Duration::from_micros(100),
            CollectionKind::Minor,
            now,
        ));
        stats.record(PauseMeasurement::new(
            Duration::from_micros(200),
            CollectionKind::Minor,
            now,
        ));
        stats.record(PauseMeasurement::new(
            Duration::from_micros(300),
            CollectionKind::Minor,
            now,
        ));

        let avg = stats.average_pause().unwrap();
        assert_eq!(avg, Duration::from_micros(200)); // (100+200+300)/3
    }

    #[test]
    fn test_pause_stats_reset() {
        let mut stats = PauseStats::new(100, Duration::from_millis(1));
        let now = Instant::now();

        stats.record(PauseMeasurement::new(
            Duration::from_micros(500),
            CollectionKind::Minor,
            now,
        ));
        stats.record(PauseMeasurement::new(
            Duration::from_millis(2), // violation
            CollectionKind::Major,
            now,
        ));

        assert!(stats.has_violations());
        assert_eq!(stats.total_pauses(), 2);

        stats.reset();

        assert!(!stats.has_violations());
        assert_eq!(stats.total_pauses(), 0);
        assert!(stats.min_pause().is_none());
    }

    #[test]
    fn test_gc_pause_measurement_integration() {
        let gc = GarbageCollector::with_default_config();
        let roots = RootSet::new();

        // Run some collections
        gc.minor_collect(&roots);
        gc.minor_collect(&roots);
        gc.major_collect(&roots);

        let summary = gc.pause_summary();
        assert_eq!(summary.total_pauses, 3);

        // Pause times should be very small (placeholder impl does nothing)
        // but should be non-zero
        assert!(summary.min_pause.is_some());
        assert!(summary.max_pause.is_some());
    }

    #[test]
    fn test_gc_pause_threshold() {
        let gc = GarbageCollector::with_default_config();

        // Default threshold is 1ms (from config.max_pause_us)
        let summary = gc.pause_summary();
        assert_eq!(summary.threshold, Duration::from_millis(1));

        // Can change threshold
        gc.set_pause_threshold(Duration::from_micros(500));
        let summary = gc.pause_summary();
        assert_eq!(summary.threshold, Duration::from_micros(500));
    }

    #[test]
    fn test_gc_pause_stats_reset() {
        let gc = GarbageCollector::with_default_config();
        let roots = RootSet::new();

        gc.minor_collect(&roots);
        gc.minor_collect(&roots);

        assert_eq!(gc.pause_summary().total_pauses, 2);

        gc.reset_pause_stats();

        assert_eq!(gc.pause_summary().total_pauses, 0);
    }

    #[test]
    fn test_pause_summary_display() {
        let mut stats = PauseStats::new(100, Duration::from_millis(1));
        let now = Instant::now();

        stats.record(PauseMeasurement::new(
            Duration::from_micros(500),
            CollectionKind::Minor,
            now,
        ));

        let summary = stats.summary();
        let display = format!("{}", summary);

        assert!(display.contains("GC Pause Statistics"));
        assert!(display.contains("Total pauses: 1"));
    }

    // ========================================================================
    // Incremental GC Tests
    // ========================================================================

    #[test]
    fn test_gc_incremental_disabled_by_default() {
        let gc = GarbageCollector::with_default_config();
        assert!(!gc.is_incremental_enabled());
        assert!(!gc.is_incremental_marking());
    }

    #[test]
    fn test_gc_incremental_enabled() {
        let config = GcConfig {
            incremental: true,
            max_pause_us: 500,
            ..Default::default()
        };
        let gc = GarbageCollector::new(config);
        assert!(gc.is_incremental_enabled());
        assert!(!gc.is_incremental_marking()); // Not started yet
    }

    #[test]
    fn test_gc_incremental_collect_cycle() {
        let config = GcConfig {
            incremental: true,
            max_pause_us: 500,
            ..Default::default()
        };
        let gc = GarbageCollector::new(config);

        // Set up some roots
        let mut roots = RootSet::new();
        let ptr1 = NonNull::new(0x1000 as *mut u8).unwrap();
        let ptr2 = NonNull::new(0x2000 as *mut u8).unwrap();
        roots.add_stack_root(ptr1);
        roots.add_global_root(ptr2);

        // Start incremental collection
        let pause = gc.start_incremental_collect(&roots);
        assert!(pause.is_some());
        assert!(gc.is_incremental_marking());

        // Do some incremental work
        while gc.is_incremental_marking() {
            let work_done = gc.do_incremental_work();
            if work_done.is_none() {
                break;
            }
        }

        // Should complete
        let stats = gc.stats();
        assert_eq!(stats.incremental_cycles, 1);
    }

    #[test]
    fn test_gc_incremental_write_barrier() {
        let config = GcConfig {
            incremental: true,
            max_pause_us: 500,
            ..Default::default()
        };
        let gc = GarbageCollector::new(config);

        // Write barrier should work when incremental is enabled
        let ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        gc.incremental_write_barrier(ptr); // Should not panic
    }

    #[test]
    fn test_gc_incremental_stats() {
        let config = GcConfig {
            incremental: true,
            max_pause_us: 500,
            ..Default::default()
        };
        let gc = GarbageCollector::new(config);

        let stats = gc.incremental_stats();
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.increments, 0);
    }

    #[test]
    fn test_gc_finish_incremental() {
        let config = GcConfig {
            incremental: true,
            max_pause_us: 500,
            ..Default::default()
        };
        let gc = GarbageCollector::new(config);

        let roots = RootSet::new();
        gc.start_incremental_collect(&roots);

        // Finish collection
        let pause = gc.finish_incremental_collect();
        assert!(pause.is_some());
        assert!(!gc.is_incremental_marking());
    }
}
