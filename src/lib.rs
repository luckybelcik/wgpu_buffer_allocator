//! # wgpu_buffer_allocator
//!
//! This crate provides a simple, robust memory allocator and abstraction layer
//! for managing GPU buffers within wgpu applications.

pub mod allocator;
#[cfg(test)]
pub mod util;
#[cfg(test)]
mod tests;
