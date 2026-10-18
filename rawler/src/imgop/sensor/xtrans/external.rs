// SPDX-License-Identifier: LGPL-2.1-only
// Copyright 2026 RapidRAW contributors.
//! Borrowed, synchronous opt-in demosaic contract. No provider is stored globally.
use super::markesteijn::{Prepared, kernel};
use crate::{
  RawImage,
  cfa::{CFA, PlaneColor},
  imgop::Rect,
  pixarray::{Color2D, PixF32},
};
pub use kernel::{Error as DemosaicError, Passes as XTransPasses};

pub struct XTransDemosaicOptions<'a> {
  pub(crate) passes: XTransPasses,
  pub(crate) to_white_one: f32,
  pub(crate) max_workers: usize,
  pub(crate) scratch_budget_bytes: usize,
  pub(crate) cancelled: &'a (dyn Fn() -> bool + Sync),
  pub(crate) external: Option<&'a dyn ExternalDemosaic>,
}
impl<'a> XTransDemosaicOptions<'a> {
  pub fn new(
    passes: XTransPasses,
    to_white_one: f32,
    max_workers: usize,
    scratch_budget_bytes: usize,
    cancelled: &'a (dyn Fn() -> bool + Sync),
  ) -> Result<Self, DemosaicError> {
    if !to_white_one.is_finite() || to_white_one <= 0.0 {
      return Err(DemosaicError::InvalidScale);
    }
    if max_workers == 0 || scratch_budget_bytes < passes.scratch_bytes() {
      return Err(DemosaicError::ScratchBudget);
    }
    Ok(Self {
      passes,
      to_white_one,
      max_workers,
      scratch_budget_bytes,
      cancelled,
      external: None,
    })
  }
  pub fn with_external(mut self, provider: &'a dyn ExternalDemosaic) -> Self {
    self.external = Some(provider);
    self
  }
  pub(crate) fn check(&self) -> Result<(), DemosaicError> {
    if (self.cancelled)() { Err(DemosaicError::Cancelled) } else { Ok(()) }
  }
}
/// Post-Rescale input and original source/level descriptors have distinct
/// borrows. Output must undo interpolation WB and to_white_one scaling.
pub struct XTransDemosaicRequest<'a> {
  pub(crate) pixels: &'a PixF32,
  pub(crate) original: &'a RawImage,
  pub(crate) cfa: &'a CFA,
  pub(crate) colors: &'a PlaneColor,
  pub(crate) roi: Rect,
  pub(crate) prepared: &'a Prepared,
  pub(crate) options: &'a XTransDemosaicOptions<'a>,
}
impl XTransDemosaicRequest<'_> {
  pub fn pixels(&self) -> &PixF32 {
    self.pixels
  }
  pub fn original(&self) -> &RawImage {
    self.original
  }
  pub fn cfa(&self) -> &CFA {
    self.cfa
  }
  pub fn colors(&self) -> &PlaneColor {
    self.colors
  }
  pub fn roi(&self) -> Rect {
    self.roi
  }
  pub fn passes(&self) -> XTransPasses {
    self.options.passes
  }
  pub fn to_white_one(&self) -> f32 {
    self.options.to_white_one
  }
  pub fn interpolation_wb(&self) -> [f32; 3] {
    self.prepared.wb
  }
  pub fn camera_to_metric(&self) -> [[f32; 3]; 3] {
    self.prepared.matrix
  }
  pub fn is_cancelled(&self) -> bool {
    (self.options.cancelled)()
  }
}
pub enum ExternalDemosaicResult {
  Ready(Color2D<f32, 3>),
  Unavailable { reason: String },
}
pub trait ExternalDemosaic: Sync {
  fn demosaic(&self, request: &XTransDemosaicRequest<'_>) -> Result<ExternalDemosaicResult, DemosaicError>;
}
