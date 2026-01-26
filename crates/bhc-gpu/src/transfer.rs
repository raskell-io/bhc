//! Asynchronous data transfer management.
//!
//! This module provides a transfer queue for managing asynchronous
//! data transfers between host and device memory.
//!
//! # Async Transfers
//!
//! Asynchronous transfers overlap data movement with computation,
//! improving overall throughput:
//!
//! ```rust,ignore
//! let stream = ctx.create_stream("transfer")?;
//!
//! // Start async transfer
//! let handle = ctx.copy_to_device_async(&host_data, &mut device_buf, &stream)?;
//!
//! // Do other work...
//! compute_something();
//!
//! // Wait for transfer
//! handle.wait()?;
//! ```
//!
//! # Transfer Queue
//!
//! The `TransferQueue` manages pending transfers and provides
//! status tracking:
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │                Transfer Queue                    │
//! ├─────────────────────────────────────────────────┤
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
//! │  │ Transfer │──│ Transfer │──│ Transfer │──... │
//! │  │  (H→D)   │  │  (D→H)   │  │  (D→D)   │      │
//! │  └──────────┘  └──────────┘  └──────────┘      │
//! │       │             │             │             │
//! │       ▼             ▼             ▼             │
//! │    Stream 1      Stream 2      Stream 1        │
//! └─────────────────────────────────────────────────┘
//! ```

use crate::context::Stream;
use crate::device::DeviceId;
use crate::memory::{DeviceBuffer, DevicePtr};
use crate::{GpuError, GpuResult};
use bhc_ffi::FfiSafe;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// A handle to a pending transfer operation.
#[derive(Clone, Debug)]
pub struct TransferHandle {
    /// Unique transfer ID.
    pub id: u64,
    /// Transfer status.
    status: Arc<AtomicUsize>,
    /// Stream this transfer is on.
    stream_handle: u64,
}

/// Transfer status values.
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferStatus {
    /// Transfer is queued.
    Pending = 0,
    /// Transfer is in progress.
    InProgress = 1,
    /// Transfer completed successfully.
    Completed = 2,
    /// Transfer failed.
    Failed = 3,
}

impl From<usize> for TransferStatus {
    fn from(v: usize) -> Self {
        match v {
            0 => Self::Pending,
            1 => Self::InProgress,
            2 => Self::Completed,
            3 => Self::Failed,
            _ => Self::Failed,
        }
    }
}

impl TransferHandle {
    /// Create a new transfer handle.
    fn new(id: u64, stream_handle: u64) -> Self {
        Self {
            id,
            status: Arc::new(AtomicUsize::new(TransferStatus::Pending as usize)),
            stream_handle,
        }
    }

    /// Get the current transfer status.
    #[must_use]
    pub fn status(&self) -> TransferStatus {
        TransferStatus::from(self.status.load(Ordering::SeqCst))
    }

    /// Check if the transfer is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(
            self.status(),
            TransferStatus::Completed | TransferStatus::Failed
        )
    }

    /// Wait for the transfer to complete.
    pub fn wait(&self) -> GpuResult<()> {
        // Spin-wait for completion (in real impl, would use proper sync)
        while !self.is_complete() {
            std::hint::spin_loop();
        }

        match self.status() {
            TransferStatus::Completed => Ok(()),
            TransferStatus::Failed => Err(GpuError::TransferError("transfer failed".to_string())),
            _ => unreachable!(),
        }
    }

    /// Mark the transfer as complete.
    pub(crate) fn mark_complete(&self) {
        self.status
            .store(TransferStatus::Completed as usize, Ordering::SeqCst);
    }

    /// Mark the transfer as failed.
    pub(crate) fn mark_failed(&self) {
        self.status
            .store(TransferStatus::Failed as usize, Ordering::SeqCst);
    }
}

/// Direction of a transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host to device.
    HostToDevice,
    /// Device to host.
    DeviceToHost,
    /// Device to device.
    DeviceToDevice,
}

/// A transfer operation.
#[derive(Debug)]
pub struct Transfer {
    /// Transfer ID.
    pub id: u64,
    /// Transfer direction.
    pub direction: TransferDirection,
    /// Size in bytes.
    pub size: usize,
    /// Source pointer.
    src: TransferPtr,
    /// Destination pointer.
    dst: TransferPtr,
    /// Handle for status updates.
    handle: TransferHandle,
}

/// Pointer that can be either host or device.
#[derive(Debug)]
enum TransferPtr {
    Host(*const u8),
    Device(DevicePtr),
}

// Safety: TransferPtr is only used with proper synchronization
unsafe impl Send for TransferPtr {}
unsafe impl Sync for TransferPtr {}

impl Transfer {
    /// Get the transfer handle.
    #[must_use]
    pub fn handle(&self) -> TransferHandle {
        self.handle.clone()
    }
}

/// Queue for managing async transfers.
pub struct TransferQueue {
    /// Device ID.
    device: DeviceId,
    /// Pending transfers.
    pending: Mutex<VecDeque<Transfer>>,
    /// Next transfer ID.
    next_id: AtomicU64,
}

impl TransferQueue {
    /// Create a new transfer queue.
    #[must_use]
    pub fn new(device: DeviceId) -> Self {
        Self {
            device,
            pending: Mutex::new(VecDeque::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Enqueue a host to device transfer.
    pub fn enqueue_host_to_device<T: FfiSafe>(
        &self,
        src: &[T],
        dst: &mut DeviceBuffer<T>,
        stream: &Stream,
    ) -> GpuResult<TransferHandle> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let size = src.len() * std::mem::size_of::<T>();

        let handle = TransferHandle::new(id, stream.handle);

        let transfer = Transfer {
            id,
            direction: TransferDirection::HostToDevice,
            size,
            src: TransferPtr::Host(src.as_ptr() as *const u8),
            dst: TransferPtr::Device(dst.as_ptr()),
            handle: handle.clone(),
        };

        // Execute the transfer (async in real implementation)
        self.execute_transfer(&transfer, stream)?;

        self.pending.lock().push_back(transfer);

        Ok(handle)
    }

    /// Enqueue a device to host transfer.
    pub fn enqueue_device_to_host<T: FfiSafe>(
        &self,
        src: &DeviceBuffer<T>,
        dst: &mut [T],
        stream: &Stream,
    ) -> GpuResult<TransferHandle> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let size = src.len() * std::mem::size_of::<T>();

        let handle = TransferHandle::new(id, stream.handle);

        let transfer = Transfer {
            id,
            direction: TransferDirection::DeviceToHost,
            size,
            src: TransferPtr::Device(src.as_ptr()),
            dst: TransferPtr::Host(dst.as_mut_ptr() as *const u8),
            handle: handle.clone(),
        };

        // Execute the transfer (async in real implementation)
        self.execute_transfer(&transfer, stream)?;

        self.pending.lock().push_back(transfer);

        Ok(handle)
    }

    /// Enqueue a device to device transfer.
    pub fn enqueue_device_to_device<T: FfiSafe>(
        &self,
        src: &DeviceBuffer<T>,
        dst: &mut DeviceBuffer<T>,
        stream: &Stream,
    ) -> GpuResult<TransferHandle> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let size = src.len() * std::mem::size_of::<T>();

        let handle = TransferHandle::new(id, stream.handle);

        let transfer = Transfer {
            id,
            direction: TransferDirection::DeviceToDevice,
            size,
            src: TransferPtr::Device(src.as_ptr()),
            dst: TransferPtr::Device(dst.as_ptr()),
            handle: handle.clone(),
        };

        // Execute the transfer
        self.execute_transfer(&transfer, stream)?;

        self.pending.lock().push_back(transfer);

        Ok(handle)
    }

    /// Execute a transfer on a stream.
    fn execute_transfer(&self, transfer: &Transfer, stream: &Stream) -> GpuResult<()> {
        // For mock implementation, just mark as complete
        // Real implementation would call cudaMemcpyAsync / hipMemcpyAsync
        let _ = stream; // Used by real CUDA/ROCm implementations
        transfer.handle.mark_complete();
        Ok(())
    }

    /// Get the number of pending transfers.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.lock().len()
    }

    /// Wait for all pending transfers to complete.
    pub fn wait_all(&self) -> GpuResult<()> {
        let pending: Vec<_> = self.pending.lock().drain(..).collect();
        for transfer in pending {
            transfer.handle.wait()?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for TransferQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransferQueue")
            .field("device", &self.device)
            .field("pending_count", &self.pending_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_status() {
        let handle = TransferHandle::new(1, 0);
        assert_eq!(handle.status(), TransferStatus::Pending);
        assert!(!handle.is_complete());

        handle.mark_complete();
        assert_eq!(handle.status(), TransferStatus::Completed);
        assert!(handle.is_complete());
    }

    #[test]
    fn test_transfer_handle_wait() {
        let handle = TransferHandle::new(1, 0);
        handle.mark_complete();
        assert!(handle.wait().is_ok());
    }

    #[test]
    fn test_transfer_handle_failed() {
        let handle = TransferHandle::new(1, 0);
        handle.mark_failed();
        assert!(handle.wait().is_err());
    }

    #[test]
    fn test_transfer_direction() {
        assert_ne!(
            TransferDirection::HostToDevice,
            TransferDirection::DeviceToHost
        );
    }

    #[test]
    fn test_transfer_queue_new() {
        let queue = TransferQueue::new(DeviceId(0));
        assert_eq!(queue.pending_count(), 0);
    }
}
