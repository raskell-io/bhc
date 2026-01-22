//! Communication channels
//!
//! Bounded and unbounded channels for inter-task communication.

use crossbeam_channel::{self as cc, Receiver, Sender};

/// A bounded channel with fixed capacity
pub struct BoundedChannel<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
}

impl<T> BoundedChannel<T> {
    /// Create a new bounded channel with the given capacity
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = cc::bounded(capacity);
        Self { sender, receiver }
    }

    /// Send a value to the channel
    ///
    /// Blocks if the channel is full.
    pub fn send(&self, value: T) -> Result<(), cc::SendError<T>> {
        self.sender.send(value)
    }

    /// Try to send a value without blocking
    pub fn try_send(&self, value: T) -> Result<(), cc::TrySendError<T>> {
        self.sender.try_send(value)
    }

    /// Receive a value from the channel
    ///
    /// Blocks if the channel is empty.
    pub fn recv(&self) -> Result<T, cc::RecvError> {
        self.receiver.recv()
    }

    /// Try to receive a value without blocking
    pub fn try_recv(&self) -> Result<T, cc::TryRecvError> {
        self.receiver.try_recv()
    }

    /// Check if the channel is empty
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    /// Check if the channel is full
    pub fn is_full(&self) -> bool {
        self.receiver.is_full()
    }

    /// Get the number of items in the channel
    pub fn len(&self) -> usize {
        self.receiver.len()
    }
}

/// An unbounded channel with unlimited capacity
pub struct UnboundedChannel<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
}

impl<T> UnboundedChannel<T> {
    /// Create a new unbounded channel
    pub fn new() -> Self {
        let (sender, receiver) = cc::unbounded();
        Self { sender, receiver }
    }

    /// Send a value to the channel
    ///
    /// Never blocks (unbounded).
    pub fn send(&self, value: T) -> Result<(), cc::SendError<T>> {
        self.sender.send(value)
    }

    /// Receive a value from the channel
    ///
    /// Blocks if the channel is empty.
    pub fn recv(&self) -> Result<T, cc::RecvError> {
        self.receiver.recv()
    }

    /// Try to receive a value without blocking
    pub fn try_recv(&self) -> Result<T, cc::TryRecvError> {
        self.receiver.try_recv()
    }

    /// Check if the channel is empty
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    /// Get the number of items in the channel
    pub fn len(&self) -> usize {
        self.receiver.len()
    }
}

impl<T> Default for UnboundedChannel<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_channel() {
        let chan = BoundedChannel::new(2);
        assert!(chan.is_empty());

        chan.send(1).unwrap();
        chan.send(2).unwrap();
        assert!(chan.is_full());

        assert_eq!(chan.recv().unwrap(), 1);
        assert_eq!(chan.recv().unwrap(), 2);
        assert!(chan.is_empty());
    }

    #[test]
    fn test_unbounded_channel() {
        let chan = UnboundedChannel::new();

        for i in 0..100 {
            chan.send(i).unwrap();
        }

        assert_eq!(chan.len(), 100);

        for i in 0..100 {
            assert_eq!(chan.recv().unwrap(), i);
        }
    }
}
