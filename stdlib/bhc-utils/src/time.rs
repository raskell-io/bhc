//! Date and time operations
//!
//! This module provides types and functions for working with dates, times,
//! and durations.
//!
//! # Overview
//!
//! - [`Instant`] - A point in time, useful for measuring elapsed time
//! - [`Duration`] - A span of time
//! - [`Date`] - A calendar date (year, month, day)
//! - [`Time`] - A time of day (hour, minute, second)
//! - [`DateTime`] - Combined date and time
//!
//! # Example
//!
//! ```
//! use bhc_utils::time::{Duration, Instant, Date, Time};
//!
//! // Measure elapsed time
//! let start = Instant::now();
//! std::thread::sleep(std::time::Duration::from_millis(10));
//! let elapsed = start.elapsed();
//! assert!(elapsed.as_millis() >= 10);
//!
//! // Create dates and times
//! let date = Date::new(2024, 12, 25);
//! let time = Time::new(14, 30, 0);
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

/// A duration of time
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Duration {
    nanos: u128,
}

impl Duration {
    /// Zero duration
    pub const ZERO: Duration = Duration { nanos: 0 };

    /// Maximum duration
    pub const MAX: Duration = Duration { nanos: u128::MAX };

    /// Create a duration from seconds
    pub const fn from_secs(secs: u64) -> Self {
        Duration {
            nanos: secs as u128 * 1_000_000_000,
        }
    }

    /// Create a duration from milliseconds
    pub const fn from_millis(millis: u64) -> Self {
        Duration {
            nanos: millis as u128 * 1_000_000,
        }
    }

    /// Create a duration from microseconds
    pub const fn from_micros(micros: u64) -> Self {
        Duration {
            nanos: micros as u128 * 1_000,
        }
    }

    /// Create a duration from nanoseconds
    pub const fn from_nanos(nanos: u128) -> Self {
        Duration { nanos }
    }

    /// Get the total seconds (truncated)
    pub const fn as_secs(&self) -> u64 {
        (self.nanos / 1_000_000_000) as u64
    }

    /// Get the total milliseconds (truncated)
    pub const fn as_millis(&self) -> u128 {
        self.nanos / 1_000_000
    }

    /// Get the total microseconds (truncated)
    pub const fn as_micros(&self) -> u128 {
        self.nanos / 1_000
    }

    /// Get the total nanoseconds
    pub const fn as_nanos(&self) -> u128 {
        self.nanos
    }

    /// Get the fractional nanoseconds (subsec portion)
    pub const fn subsec_nanos(&self) -> u32 {
        (self.nanos % 1_000_000_000) as u32
    }

    /// Check if the duration is zero
    pub const fn is_zero(&self) -> bool {
        self.nanos == 0
    }

    /// Saturating addition
    pub const fn saturating_add(self, other: Self) -> Self {
        Duration {
            nanos: self.nanos.saturating_add(other.nanos),
        }
    }

    /// Saturating subtraction
    pub const fn saturating_sub(self, other: Self) -> Self {
        Duration {
            nanos: self.nanos.saturating_sub(other.nanos),
        }
    }

    /// Checked addition
    pub const fn checked_add(self, other: Self) -> Option<Self> {
        match self.nanos.checked_add(other.nanos) {
            Some(nanos) => Some(Duration { nanos }),
            None => None,
        }
    }

    /// Checked subtraction
    pub const fn checked_sub(self, other: Self) -> Option<Self> {
        match self.nanos.checked_sub(other.nanos) {
            Some(nanos) => Some(Duration { nanos }),
            None => None,
        }
    }

    /// Multiply by a scalar
    pub const fn mul(self, rhs: u32) -> Self {
        Duration {
            nanos: self.nanos * rhs as u128,
        }
    }

    /// Divide by a scalar
    pub const fn div(self, rhs: u32) -> Self {
        Duration {
            nanos: self.nanos / rhs as u128,
        }
    }
}

impl From<std::time::Duration> for Duration {
    fn from(d: std::time::Duration) -> Self {
        Duration::from_nanos(d.as_nanos())
    }
}

impl From<Duration> for std::time::Duration {
    fn from(d: Duration) -> Self {
        std::time::Duration::from_nanos(d.nanos as u64)
    }
}

impl std::ops::Add for Duration {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Duration {
            nanos: self.nanos + rhs.nanos,
        }
    }
}

impl std::ops::Sub for Duration {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Duration {
            nanos: self.nanos - rhs.nanos,
        }
    }
}

impl std::fmt::Display for Duration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let secs = self.as_secs();
        let nanos = self.subsec_nanos();
        if nanos == 0 {
            write!(f, "{}s", secs)
        } else {
            write!(f, "{}.{:09}s", secs, nanos)
        }
    }
}

/// An instant in time for measuring duration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant {
    inner: std::time::Instant,
}

impl Instant {
    /// Get the current instant
    pub fn now() -> Self {
        Instant {
            inner: std::time::Instant::now(),
        }
    }

    /// Get the elapsed time since this instant
    pub fn elapsed(&self) -> Duration {
        Duration::from(self.inner.elapsed())
    }

    /// Get the duration since another instant
    pub fn duration_since(&self, earlier: Instant) -> Duration {
        Duration::from(self.inner.duration_since(earlier.inner))
    }

    /// Checked addition of a duration
    pub fn checked_add(&self, duration: Duration) -> Option<Instant> {
        self.inner
            .checked_add(duration.into())
            .map(|inner| Instant { inner })
    }

    /// Checked subtraction of a duration
    pub fn checked_sub(&self, duration: Duration) -> Option<Instant> {
        self.inner
            .checked_sub(duration.into())
            .map(|inner| Instant { inner })
    }

    /// Saturating subtraction
    pub fn saturating_duration_since(&self, earlier: Instant) -> Duration {
        Duration::from(self.inner.saturating_duration_since(earlier.inner))
    }
}

impl std::ops::Add<Duration> for Instant {
    type Output = Instant;
    fn add(self, rhs: Duration) -> Self::Output {
        let std_dur: std::time::Duration = rhs.into();
        Instant {
            inner: self.inner + std_dur,
        }
    }
}

impl std::ops::Sub<Duration> for Instant {
    type Output = Instant;
    fn sub(self, rhs: Duration) -> Self::Output {
        let std_dur: std::time::Duration = rhs.into();
        Instant {
            inner: self.inner - std_dur,
        }
    }
}

impl std::ops::Sub<Instant> for Instant {
    type Output = Duration;
    fn sub(self, rhs: Instant) -> Self::Output {
        Duration::from(self.inner - rhs.inner)
    }
}

/// A calendar date
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date {
    year: i32,
    month: u8,
    day: u8,
}

impl Date {
    /// Create a new date
    ///
    /// # Panics
    ///
    /// Panics if the date is invalid.
    pub fn new(year: i32, month: u8, day: u8) -> Self {
        assert!(month >= 1 && month <= 12, "Invalid month: {}", month);
        assert!(day >= 1 && day <= 31, "Invalid day: {}", day);
        // Basic validation - could be more thorough
        Date { year, month, day }
    }

    /// Try to create a new date, returning None if invalid
    pub fn try_new(year: i32, month: u8, day: u8) -> Option<Self> {
        if month < 1 || month > 12 || day < 1 || day > 31 {
            return None;
        }

        let days_in_month = match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if Self::is_leap_year(year) {
                    29
                } else {
                    28
                }
            }
            _ => return None,
        };

        if day > days_in_month {
            return None;
        }

        Some(Date { year, month, day })
    }

    /// Get the year
    pub const fn year(&self) -> i32 {
        self.year
    }

    /// Get the month (1-12)
    pub const fn month(&self) -> u8 {
        self.month
    }

    /// Get the day (1-31)
    pub const fn day(&self) -> u8 {
        self.day
    }

    /// Check if a year is a leap year
    pub const fn is_leap_year(year: i32) -> bool {
        (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
    }

    /// Get the day of the week (0 = Sunday, 6 = Saturday)
    pub fn weekday(&self) -> u8 {
        // Zeller's congruence
        let mut y = self.year;
        let mut m = self.month as i32;

        if m < 3 {
            m += 12;
            y -= 1;
        }

        let q = self.day as i32;
        let k = y % 100;
        let j = y / 100;

        let h = (q + (13 * (m + 1)) / 5 + k + k / 4 + j / 4 - 2 * j) % 7;
        ((h + 6) % 7) as u8 // Convert to 0=Sunday
    }

    /// Get today's date
    pub fn today() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Simple conversion - days since Unix epoch
        let days = now / 86400;
        let mut year = 1970;
        let mut remaining_days = days as i32;

        loop {
            let days_in_year = if Self::is_leap_year(year) { 366 } else { 365 };
            if remaining_days < days_in_year {
                break;
            }
            remaining_days -= days_in_year;
            year += 1;
        }

        let mut month = 1u8;
        loop {
            let days_in_month = match month {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                2 => {
                    if Self::is_leap_year(year) {
                        29
                    } else {
                        28
                    }
                }
                _ => unreachable!(),
            };

            if remaining_days < days_in_month {
                break;
            }
            remaining_days -= days_in_month;
            month += 1;
        }

        Date {
            year,
            month,
            day: remaining_days as u8 + 1,
        }
    }

    /// Format as ISO 8601 (YYYY-MM-DD)
    pub fn to_iso8601(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    /// Parse from ISO 8601 (YYYY-MM-DD)
    pub fn from_iso8601(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return None;
        }

        let year = parts[0].parse().ok()?;
        let month = parts[1].parse().ok()?;
        let day = parts[2].parse().ok()?;

        Self::try_new(year, month, day)
    }
}

impl std::fmt::Display for Date {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_iso8601())
    }
}

/// A time of day
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Time {
    hour: u8,
    minute: u8,
    second: u8,
    nanos: u32,
}

impl Time {
    /// Create a new time
    pub fn new(hour: u8, minute: u8, second: u8) -> Self {
        assert!(hour < 24, "Invalid hour: {}", hour);
        assert!(minute < 60, "Invalid minute: {}", minute);
        assert!(second < 60, "Invalid second: {}", second);
        Time {
            hour,
            minute,
            second,
            nanos: 0,
        }
    }

    /// Create a time with nanoseconds
    pub fn with_nanos(hour: u8, minute: u8, second: u8, nanos: u32) -> Self {
        assert!(hour < 24, "Invalid hour: {}", hour);
        assert!(minute < 60, "Invalid minute: {}", minute);
        assert!(second < 60, "Invalid second: {}", second);
        assert!(nanos < 1_000_000_000, "Invalid nanos: {}", nanos);
        Time {
            hour,
            minute,
            second,
            nanos,
        }
    }

    /// Try to create a time
    pub fn try_new(hour: u8, minute: u8, second: u8) -> Option<Self> {
        if hour >= 24 || minute >= 60 || second >= 60 {
            return None;
        }
        Some(Time {
            hour,
            minute,
            second,
            nanos: 0,
        })
    }

    /// Get the hour (0-23)
    pub const fn hour(&self) -> u8 {
        self.hour
    }

    /// Get the minute (0-59)
    pub const fn minute(&self) -> u8 {
        self.minute
    }

    /// Get the second (0-59)
    pub const fn second(&self) -> u8 {
        self.second
    }

    /// Get the nanoseconds
    pub const fn nanosecond(&self) -> u32 {
        self.nanos
    }

    /// Get current time
    pub fn now() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let day_secs = (now % 86400) as u32;
        let hour = (day_secs / 3600) as u8;
        let minute = ((day_secs % 3600) / 60) as u8;
        let second = (day_secs % 60) as u8;

        Time {
            hour,
            minute,
            second,
            nanos: 0,
        }
    }

    /// Midnight
    pub const fn midnight() -> Self {
        Time {
            hour: 0,
            minute: 0,
            second: 0,
            nanos: 0,
        }
    }

    /// Noon
    pub const fn noon() -> Self {
        Time {
            hour: 12,
            minute: 0,
            second: 0,
            nanos: 0,
        }
    }

    /// Format as HH:MM:SS
    pub fn to_string_hms(&self) -> String {
        format!("{:02}:{:02}:{:02}", self.hour, self.minute, self.second)
    }

    /// Parse from HH:MM:SS
    pub fn from_hms(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() < 2 {
            return None;
        }

        let hour = parts[0].parse().ok()?;
        let minute = parts[1].parse().ok()?;
        let second = if parts.len() > 2 {
            parts[2].parse().ok()?
        } else {
            0
        };

        Self::try_new(hour, minute, second)
    }
}

impl std::fmt::Display for Time {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_hms())
    }
}

/// A combined date and time
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DateTime {
    date: Date,
    time: Time,
}

impl DateTime {
    /// Create a new datetime
    pub fn new(date: Date, time: Time) -> Self {
        DateTime { date, time }
    }

    /// Get the date component
    pub const fn date(&self) -> Date {
        self.date
    }

    /// Get the time component
    pub const fn time(&self) -> Time {
        self.time
    }

    /// Get current datetime
    pub fn now() -> Self {
        DateTime {
            date: Date::today(),
            time: Time::now(),
        }
    }

    /// Format as ISO 8601 (YYYY-MM-DDTHH:MM:SS)
    pub fn to_iso8601(&self) -> String {
        format!("{}T{}", self.date.to_iso8601(), self.time.to_string_hms())
    }

    /// Parse from ISO 8601
    pub fn from_iso8601(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('T').collect();
        if parts.len() != 2 {
            return None;
        }

        let date = Date::from_iso8601(parts[0])?;
        let time = Time::from_hms(parts[1])?;

        Some(DateTime { date, time })
    }

    /// Unix timestamp (seconds since epoch)
    pub fn timestamp(&self) -> i64 {
        // Simplified calculation
        let mut days: i64 = 0;

        // Years
        for y in 1970..self.date.year {
            days += if Date::is_leap_year(y) { 366 } else { 365 };
        }

        // Months
        for m in 1..self.date.month {
            days += match m {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                2 => {
                    if Date::is_leap_year(self.date.year) {
                        29
                    } else {
                        28
                    }
                }
                _ => 0,
            };
        }

        // Days
        days += (self.date.day - 1) as i64;

        // To seconds
        let secs = days * 86400
            + self.time.hour as i64 * 3600
            + self.time.minute as i64 * 60
            + self.time.second as i64;

        secs
    }
}

impl std::fmt::Display for DateTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_iso8601())
    }
}

/// Sleep for a duration
pub fn sleep(duration: Duration) {
    std::thread::sleep(duration.into());
}

/// Measure execution time of a closure
pub fn measure<F, T>(f: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    (result, elapsed)
}

// FFI exports

/// Get current Unix timestamp (FFI)
#[no_mangle]
pub extern "C" fn bhc_time_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

/// Sleep for milliseconds (FFI)
#[no_mangle]
pub extern "C" fn bhc_sleep_millis(millis: u64) {
    sleep(Duration::from_millis(millis));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duration_from_secs() {
        let d = Duration::from_secs(5);
        assert_eq!(d.as_secs(), 5);
        assert_eq!(d.as_millis(), 5000);
    }

    #[test]
    fn test_duration_from_millis() {
        let d = Duration::from_millis(1500);
        assert_eq!(d.as_secs(), 1);
        assert_eq!(d.as_millis(), 1500);
        assert_eq!(d.subsec_nanos(), 500_000_000);
    }

    #[test]
    fn test_duration_add() {
        let d1 = Duration::from_secs(1);
        let d2 = Duration::from_millis(500);
        let sum = d1 + d2;
        assert_eq!(sum.as_millis(), 1500);
    }

    #[test]
    fn test_instant_elapsed() {
        let start = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn test_date_new() {
        let date = Date::new(2024, 12, 25);
        assert_eq!(date.year(), 2024);
        assert_eq!(date.month(), 12);
        assert_eq!(date.day(), 25);
    }

    #[test]
    fn test_date_try_new() {
        assert!(Date::try_new(2024, 2, 29).is_some()); // Leap year
        assert!(Date::try_new(2023, 2, 29).is_none()); // Not leap year
        assert!(Date::try_new(2024, 13, 1).is_none()); // Invalid month
    }

    #[test]
    fn test_date_iso8601() {
        let date = Date::new(2024, 1, 5);
        assert_eq!(date.to_iso8601(), "2024-01-05");

        let parsed = Date::from_iso8601("2024-01-05").unwrap();
        assert_eq!(parsed, date);
    }

    #[test]
    fn test_time_new() {
        let time = Time::new(14, 30, 45);
        assert_eq!(time.hour(), 14);
        assert_eq!(time.minute(), 30);
        assert_eq!(time.second(), 45);
    }

    #[test]
    fn test_time_hms() {
        let time = Time::new(9, 5, 3);
        assert_eq!(time.to_string_hms(), "09:05:03");

        let parsed = Time::from_hms("09:05:03").unwrap();
        assert_eq!(parsed, time);
    }

    #[test]
    fn test_datetime() {
        let dt = DateTime::new(Date::new(2024, 6, 15), Time::new(10, 30, 0));
        assert_eq!(dt.to_iso8601(), "2024-06-15T10:30:00");
    }

    #[test]
    fn test_datetime_parse() {
        let dt = DateTime::from_iso8601("2024-06-15T10:30:00").unwrap();
        assert_eq!(dt.date().year(), 2024);
        assert_eq!(dt.time().hour(), 10);
    }

    #[test]
    fn test_leap_year() {
        assert!(Date::is_leap_year(2024));
        assert!(!Date::is_leap_year(2023));
        assert!(Date::is_leap_year(2000));
        assert!(!Date::is_leap_year(1900));
    }

    #[test]
    fn test_weekday() {
        // January 1, 2024 was a Monday (1)
        let date = Date::new(2024, 1, 1);
        assert_eq!(date.weekday(), 1);
    }

    #[test]
    fn test_measure() {
        let (result, elapsed) = measure(|| {
            std::thread::sleep(std::time::Duration::from_millis(5));
            42
        });
        assert_eq!(result, 42);
        assert!(elapsed.as_millis() >= 5);
    }

    #[test]
    fn test_midnight_noon() {
        let midnight = Time::midnight();
        assert_eq!(midnight.hour(), 0);

        let noon = Time::noon();
        assert_eq!(noon.hour(), 12);
    }

    #[test]
    fn test_duration_display() {
        let d = Duration::from_millis(1500);
        let s = format!("{}", d);
        assert!(s.contains("1"));
    }
}
