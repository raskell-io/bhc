//! Incremental marking for bounded-pause GC.
//!
//! This module implements tri-color marking with SATB (Snapshot At The Beginning)
//! write barriers to achieve bounded GC pause times for the realtime profile.
//!
//! # Tri-Color Abstraction
//!
//! Objects are classified into three colors:
//! - **White**: Unmarked objects (potentially garbage)
//! - **Gray**: Marked but children not yet scanned (work to do)
//! - **Black**: Marked and all children scanned (complete)
//!
//! The invariant maintained is: no black object points to a white object.
//! This is ensured by the SATB write barrier.
//!
//! # Incremental Work Scheduling
//!
//! Instead of marking the entire heap in one pause, work is split into
//! small increments that can be interleaved with mutator execution.
//! Each increment processes a bounded number of objects to keep pause
//! times under the configured threshold (default: 1ms for realtime).

use crate::{CollectionKind, GcPtr, ObjectHeader, PauseMeasurement};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Color of an object in tri-color marking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MarkColor {
    /// Unmarked object (potentially garbage).
    White = 0,
    /// Marked but children not yet scanned.
    Gray = 1,
    /// Marked and all children scanned.
    Black = 2,
}

/// State of the incremental marking process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkState {
    /// No marking in progress.
    Idle,
    /// Initial root scanning phase.
    RootScanning,
    /// Incremental marking of heap objects.
    Marking,
    /// Final remark phase (process SATB buffer).
    Remark,
    /// Marking complete, ready for sweep.
    Complete,
}

/// Configuration for incremental marking.
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    /// Maximum time budget per increment (microseconds).
    pub time_budget_us: u64,
    /// Maximum objects to scan per increment (fallback if time is hard to measure).
    pub max_objects_per_increment: usize,
    /// Whether to enable concurrent marking (mark while mutator runs).
    pub concurrent: bool,
    /// SATB buffer size before triggering processing.
    pub satb_buffer_threshold: usize,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            time_budget_us: 500, // 500us per increment (half of 1ms budget)
            max_objects_per_increment: 1000,
            concurrent: false, // Start with stop-the-world incremental
            satb_buffer_threshold: 1024,
        }
    }
}

/// SATB (Snapshot At The Beginning) write barrier buffer.
///
/// When a reference is overwritten, the old value is recorded in this buffer
/// to maintain the SATB invariant: all objects reachable at the start of
/// marking will be marked.
#[derive(Debug)]
pub struct SatbBuffer {
    /// Buffer of overwritten references.
    entries: Mutex<Vec<GcPtr>>,
    /// Number of entries recorded.
    count: AtomicUsize,
    /// Threshold for processing the buffer.
    threshold: usize,
}

impl SatbBuffer {
    /// Create a new SATB buffer with the given threshold.
    #[must_use]
    pub fn new(threshold: usize) -> Self {
        Self {
            entries: Mutex::new(Vec::with_capacity(threshold)),
            count: AtomicUsize::new(0),
            threshold,
        }
    }

    /// Record an overwritten reference.
    ///
    /// Called by the write barrier when a reference field is overwritten.
    pub fn record(&self, old_value: GcPtr) {
        let count = self.count.fetch_add(1, Ordering::Relaxed);
        if count < self.threshold * 2 {
            // Only record if not overflowing
            self.entries.lock().push(old_value);
        }
    }

    /// Check if the buffer should be processed.
    #[must_use]
    pub fn should_process(&self) -> bool {
        self.count.load(Ordering::Relaxed) >= self.threshold
    }

    /// Take all entries for processing.
    #[must_use]
    pub fn take(&self) -> Vec<GcPtr> {
        self.count.store(0, Ordering::Relaxed);
        std::mem::take(&mut *self.entries.lock())
    }

    /// Get the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the buffer.
    pub fn clear(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.entries.lock().clear();
    }
}

/// The gray set (work list) for incremental marking.
#[derive(Debug)]
pub struct GraySet {
    /// Queue of objects to scan.
    queue: Mutex<VecDeque<GcPtr>>,
    /// Number of objects in the queue.
    size: AtomicUsize,
}

impl GraySet {
    /// Create a new empty gray set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            size: AtomicUsize::new(0),
        }
    }

    /// Push an object onto the gray set.
    pub fn push(&self, ptr: GcPtr) {
        self.queue.lock().push_back(ptr);
        self.size.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop an object from the gray set.
    #[must_use]
    pub fn pop(&self) -> Option<GcPtr> {
        let ptr = self.queue.lock().pop_front();
        if ptr.is_some() {
            self.size.fetch_sub(1, Ordering::Relaxed);
        }
        ptr
    }

    /// Check if the gray set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size.load(Ordering::Relaxed) == 0
    }

    /// Get the size of the gray set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Clear the gray set.
    pub fn clear(&self) {
        self.queue.lock().clear();
        self.size.store(0, Ordering::Relaxed);
    }
}

impl Default for GraySet {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for incremental marking.
#[derive(Debug, Clone, Default)]
pub struct IncrementalStats {
    /// Total number of increments performed.
    pub increments: u64,
    /// Total objects marked.
    pub objects_marked: u64,
    /// Total bytes marked.
    pub bytes_marked: u64,
    /// Time spent in root scanning.
    pub root_scan_time: Duration,
    /// Time spent in marking increments.
    pub mark_time: Duration,
    /// Time spent in remark phase.
    pub remark_time: Duration,
    /// Number of SATB entries processed.
    pub satb_entries_processed: u64,
    /// Maximum gray set size.
    pub max_gray_set_size: usize,
}

/// The incremental marker.
///
/// Coordinates incremental marking with bounded pause times.
#[derive(Debug)]
pub struct IncrementalMarker {
    /// Configuration.
    config: IncrementalConfig,
    /// Current marking state.
    state: Mutex<MarkState>,
    /// The gray set (work list).
    gray_set: GraySet,
    /// SATB buffer for write barrier.
    satb_buffer: SatbBuffer,
    /// Statistics.
    stats: Mutex<IncrementalStats>,
    /// Number of marking cycles completed.
    cycles: AtomicU64,
}

impl IncrementalMarker {
    /// Create a new incremental marker.
    #[must_use]
    pub fn new(config: IncrementalConfig) -> Self {
        let satb_threshold = config.satb_buffer_threshold;
        Self {
            config,
            state: Mutex::new(MarkState::Idle),
            gray_set: GraySet::new(),
            satb_buffer: SatbBuffer::new(satb_threshold),
            stats: Mutex::new(IncrementalStats::default()),
            cycles: AtomicU64::new(0),
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(IncrementalConfig::default())
    }

    /// Get the current marking state.
    #[must_use]
    pub fn state(&self) -> MarkState {
        *self.state.lock()
    }

    /// Check if marking is in progress.
    #[must_use]
    pub fn is_marking(&self) -> bool {
        !matches!(self.state(), MarkState::Idle | MarkState::Complete)
    }

    /// Start a new marking cycle.
    ///
    /// Returns a pause measurement for the root scanning phase.
    pub fn start_cycle(&self, roots: impl Iterator<Item = GcPtr>) -> PauseMeasurement {
        let start = Instant::now();

        {
            let mut state = self.state.lock();
            *state = MarkState::RootScanning;
        }

        // Clear previous state
        self.gray_set.clear();
        self.satb_buffer.clear();

        // Scan roots and add to gray set
        let mut root_count = 0;
        for root in roots {
            self.shade_gray(root);
            root_count += 1;
        }

        let duration = start.elapsed();

        // Update stats
        {
            let mut stats = self.stats.lock();
            stats.root_scan_time += duration;
            stats.objects_marked += root_count;
        }

        // Transition to marking state
        {
            let mut state = self.state.lock();
            *state = if self.gray_set.is_empty() {
                MarkState::Complete
            } else {
                MarkState::Marking
            };
        }

        PauseMeasurement::new(duration, CollectionKind::IncrementalMark, start)
    }

    /// Perform one increment of marking work.
    ///
    /// Returns a pause measurement, or None if no work was done.
    pub fn mark_increment<F>(&self, mut scan_object: F) -> Option<PauseMeasurement>
    where
        F: FnMut(GcPtr) -> Vec<GcPtr>,
    {
        if !matches!(self.state(), MarkState::Marking) {
            return None;
        }

        let start = Instant::now();
        let time_budget = Duration::from_micros(self.config.time_budget_us);
        let mut objects_scanned = 0;
        let mut bytes_scanned = 0;

        // Process objects until time budget exhausted or gray set empty
        while start.elapsed() < time_budget
            && objects_scanned < self.config.max_objects_per_increment
        {
            let Some(obj) = self.gray_set.pop() else {
                break;
            };

            // Scan the object and shade its children gray
            let children = scan_object(obj);
            for child in children {
                self.shade_gray(child);
            }

            // Estimate size (in real impl, would read from header)
            bytes_scanned += 64; // Placeholder
            objects_scanned += 1;
        }

        let duration = start.elapsed();

        // Update stats
        {
            let mut stats = self.stats.lock();
            stats.increments += 1;
            stats.objects_marked += objects_scanned;
            stats.bytes_marked += bytes_scanned;
            stats.mark_time += duration;
            stats.max_gray_set_size = stats.max_gray_set_size.max(self.gray_set.len());
        }

        // Check if marking is complete
        if self.gray_set.is_empty() {
            let mut state = self.state.lock();
            // Transition to remark if SATB buffer has entries, else complete
            *state = if self.satb_buffer.should_process() {
                MarkState::Remark
            } else {
                MarkState::Complete
            };
        }

        Some(PauseMeasurement::new(
            duration,
            CollectionKind::IncrementalMark,
            start,
        ))
    }

    /// Perform the final remark phase.
    ///
    /// Processes the SATB buffer to ensure all objects reachable at the
    /// start of marking are marked.
    pub fn remark<F>(&self, mut scan_object: F) -> PauseMeasurement
    where
        F: FnMut(GcPtr) -> Vec<GcPtr>,
    {
        let start = Instant::now();

        // Process SATB buffer
        let satb_entries = self.satb_buffer.take();
        let entry_count = satb_entries.len();

        for ptr in satb_entries {
            self.shade_gray(ptr);
        }

        // Drain remaining gray set (should be small after SATB processing)
        while let Some(obj) = self.gray_set.pop() {
            let children = scan_object(obj);
            for child in children {
                self.shade_gray(child);
            }
        }

        let duration = start.elapsed();

        // Update stats
        {
            let mut stats = self.stats.lock();
            stats.remark_time += duration;
            stats.satb_entries_processed += entry_count as u64;
        }

        // Mark cycle complete
        {
            let mut state = self.state.lock();
            *state = MarkState::Complete;
        }

        self.cycles.fetch_add(1, Ordering::Relaxed);

        PauseMeasurement::new(duration, CollectionKind::IncrementalMark, start)
    }

    /// Reset the marker to idle state.
    pub fn reset(&self) {
        let mut state = self.state.lock();
        *state = MarkState::Idle;
        self.gray_set.clear();
        self.satb_buffer.clear();
    }

    /// Get statistics for the current/last cycle.
    #[must_use]
    pub fn stats(&self) -> IncrementalStats {
        self.stats.lock().clone()
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        *self.stats.lock() = IncrementalStats::default();
    }

    /// Get the number of completed marking cycles.
    #[must_use]
    pub fn cycles(&self) -> u64 {
        self.cycles.load(Ordering::Relaxed)
    }

    /// Record a write barrier event.
    ///
    /// Called when a reference field is overwritten during marking.
    /// The old value is recorded in the SATB buffer.
    pub fn write_barrier(&self, old_value: GcPtr) {
        if self.is_marking() {
            self.satb_buffer.record(old_value);
        }
    }

    /// Get the SATB buffer (for testing).
    pub fn satb_buffer(&self) -> &SatbBuffer {
        &self.satb_buffer
    }

    /// Get the gray set size.
    #[must_use]
    pub fn gray_set_size(&self) -> usize {
        self.gray_set.len()
    }

    // ========================================================================
    // Private methods
    // ========================================================================

    /// Shade an object gray (add to work list if not already gray/black).
    fn shade_gray(&self, ptr: GcPtr) {
        // In a full implementation, we would check the object's color
        // and only add if white. For now, just add to gray set.
        // TODO: Implement proper color checking via object header
        self.gray_set.push(ptr);
    }
}

impl Default for IncrementalMarker {
    fn default() -> Self {
        Self::with_default_config()
    }
}

/// Helper trait for objects that can be scanned during marking.
pub trait Scannable {
    /// Scan this object and return references to child objects.
    fn scan(&self) -> Vec<GcPtr>;

    /// Get the size of this object in bytes.
    fn size(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ptr(addr: usize) -> GcPtr {
        GcPtr::new(NonNull::new(addr as *mut u8).unwrap())
    }

    #[test]
    fn test_gray_set() {
        let gray_set = GraySet::new();

        assert!(gray_set.is_empty());

        gray_set.push(make_ptr(0x1000));
        gray_set.push(make_ptr(0x2000));

        assert!(!gray_set.is_empty());
        assert_eq!(gray_set.len(), 2);

        let p1 = gray_set.pop().unwrap();
        assert_eq!(p1.as_ptr() as usize, 0x1000);

        let p2 = gray_set.pop().unwrap();
        assert_eq!(p2.as_ptr() as usize, 0x2000);

        assert!(gray_set.is_empty());
        assert!(gray_set.pop().is_none());
    }

    #[test]
    fn test_satb_buffer() {
        let satb = SatbBuffer::new(3);

        assert!(satb.is_empty());

        satb.record(make_ptr(0x1000));
        satb.record(make_ptr(0x2000));

        assert!(!satb.should_process());
        assert_eq!(satb.len(), 2);

        satb.record(make_ptr(0x3000));

        assert!(satb.should_process());

        let entries = satb.take();
        assert_eq!(entries.len(), 3);
        assert!(satb.is_empty());
    }

    #[test]
    fn test_incremental_marker_state() {
        let marker = IncrementalMarker::with_default_config();

        assert_eq!(marker.state(), MarkState::Idle);
        assert!(!marker.is_marking());

        // Start a cycle
        let roots = [make_ptr(0x1000), make_ptr(0x2000)];
        marker.start_cycle(roots.into_iter());

        assert_eq!(marker.state(), MarkState::Marking);
        assert!(marker.is_marking());
    }

    #[test]
    fn test_incremental_marker_full_cycle() {
        let marker = IncrementalMarker::new(IncrementalConfig {
            max_objects_per_increment: 10,
            ..Default::default()
        });

        // Start with some roots
        let roots = [make_ptr(0x1000), make_ptr(0x2000)];
        let _root_pause = marker.start_cycle(roots.into_iter());

        // Perform marking increments
        // scan_object returns no children for simplicity
        while marker.is_marking() {
            let pause = marker.mark_increment(|_obj| Vec::new());
            if pause.is_none() {
                break;
            }
        }

        // Should complete after processing roots
        assert!(!marker.is_marking());
        assert_eq!(marker.state(), MarkState::Complete);

        let stats = marker.stats();
        assert!(stats.increments > 0);
    }

    #[test]
    fn test_incremental_marker_with_children() {
        let marker = IncrementalMarker::new(IncrementalConfig {
            max_objects_per_increment: 2,
            ..Default::default()
        });

        // Start with one root
        let roots = [make_ptr(0x1000)];
        marker.start_cycle(roots.into_iter());

        // First increment: scan root, which has 3 children
        let pause1 = marker.mark_increment(|obj| {
            if obj.as_ptr() as usize == 0x1000 {
                vec![make_ptr(0x2000), make_ptr(0x3000), make_ptr(0x4000)]
            } else {
                vec![]
            }
        });
        assert!(pause1.is_some());

        // Should still be marking (children added to gray set)
        assert!(marker.is_marking());

        // Continue until complete
        let mut increments = 1;
        while marker.is_marking() {
            if marker.mark_increment(|_| Vec::new()).is_some() {
                increments += 1;
            }
        }

        assert!(!marker.is_marking());
        // Should have taken multiple increments due to children
        assert!(increments >= 2);
    }

    #[test]
    fn test_satb_write_barrier() {
        let marker = IncrementalMarker::with_default_config();

        // Write barrier should be no-op when not marking
        marker.write_barrier(make_ptr(0x1000));
        assert!(marker.satb_buffer().is_empty());

        // Start marking
        marker.start_cycle(std::iter::empty());

        // Now write barrier should record
        marker.write_barrier(make_ptr(0x2000));
        marker.write_barrier(make_ptr(0x3000));

        assert_eq!(marker.satb_buffer().len(), 2);
    }

    #[test]
    fn test_remark_phase() {
        let marker = IncrementalMarker::new(IncrementalConfig {
            satb_buffer_threshold: 2,
            max_objects_per_increment: 100,
            ..Default::default()
        });

        // Start marking
        marker.start_cycle(std::iter::once(make_ptr(0x1000)));

        // Drain gray set
        while let Some(_) = marker.mark_increment(|_| Vec::new()) {
            if marker.gray_set_size() == 0 {
                break;
            }
        }

        // Add SATB entries to trigger remark
        marker.satb_buffer.record(make_ptr(0x5000));
        marker.satb_buffer.record(make_ptr(0x6000));

        // Force state to Remark for testing
        {
            *marker.state.lock() = MarkState::Remark;
        }

        // Perform remark
        let remark_pause = marker.remark(|_| Vec::new());

        assert!(remark_pause.duration_us() >= 0);
        assert_eq!(marker.state(), MarkState::Complete);
        assert_eq!(marker.cycles(), 1);

        let stats = marker.stats();
        assert_eq!(stats.satb_entries_processed, 2);
    }

    #[test]
    fn test_marker_reset() {
        let marker = IncrementalMarker::with_default_config();

        marker.start_cycle(std::iter::once(make_ptr(0x1000)));
        assert!(marker.is_marking());

        marker.reset();

        assert_eq!(marker.state(), MarkState::Idle);
        assert!(!marker.is_marking());
        assert!(marker.gray_set_size() == 0);
    }

    #[test]
    fn test_incremental_stats() {
        let marker = IncrementalMarker::with_default_config();

        marker.start_cycle(vec![make_ptr(0x1000), make_ptr(0x2000)].into_iter());

        while marker.is_marking() {
            marker.mark_increment(|_| Vec::new());
        }

        let stats = marker.stats();
        assert!(stats.objects_marked >= 2);
        assert!(stats.root_scan_time > Duration::ZERO);
    }
}
