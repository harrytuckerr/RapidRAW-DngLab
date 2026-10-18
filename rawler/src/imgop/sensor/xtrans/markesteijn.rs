// SPDX-License-Identifier: LGPL-2.1-only
// Copyright 2026 RapidRAW contributors.
//! Opt-in adapter for the licensed floating LibRaw-derived scalar kernel.
//! The default rawler Quality/Speed paths remain unchanged.
pub mod kernel;
use super::external::{DemosaicError, ExternalDemosaicResult, XTransDemosaicOptions, XTransDemosaicRequest, XTransPasses};
use crate::{
  RawImage,
  cfa::{CFA, PlaneColor},
  imgop::{
    Rect,
    matrix::{multiply, normalize, pseudo_inverse},
    xyz::{Illuminant, SRGB_TO_XYZ_D65},
  },
  pixarray::{Color2D, PixF32},
  rawimage::RawPhotometricInterpretation,
};
use rayon::prelude::*;
use std::panic::{AssertUnwindSafe, catch_unwind};

pub struct XTransMarkesteijnDemosaic {
  pub passes: XTransPasses,
}
pub(crate) struct Prepared {
  pub(crate) wb: [f32; 3],
  pub(crate) matrix: [[f32; 3]; 3],
  pattern: kernel::Pattern,
  metric: kernel::Metric,
}
impl Prepared {
  /// Called before RawDevelop clones/scales and resets level metadata. Scale is
  /// explicit caller-captured policy, never inferred from reset whitelevel=1.
  pub(crate) fn new(raw: &RawImage, options: &XTransDemosaicOptions<'_>) -> Result<Self, DemosaicError> {
    options.check()?;
    if raw.cpp != 1 || raw.width < 512 || raw.height < 512 || raw.width % 2 != 0 || raw.height % 2 != 0 {
      return Err(DemosaicError::InvalidGeometry);
    }
    let samples = match &raw.data {
      crate::rawimage::RawImageData::Integer(v) => v.len(),
      crate::rawimage::RawImageData::Float(v) => v.len(),
    };
    if raw.width.checked_mul(raw.height) != Some(samples) {
      return Err(DemosaicError::InvalidGeometry);
    }
    let RawPhotometricInterpretation::Cfa(config) = &raw.photometric else {
      return Err(DemosaicError::InvalidCfa);
    };
    if config.cfa.width != 6 || config.cfa.height != 6 || !config.cfa.is_rgb() {
      return Err(DemosaicError::InvalidCfa);
    }
    let pattern = kernel::Pattern::new(std::array::from_fn(|i| config.cfa.color_at(i / 6, i % 6) as u8))?;
    if raw.wb_coeffs[0].is_nan() {
      log::warn!("Markesteijn: missing R WB; using unit interpolation WB");
    }
    let wb = kernel::effective_wb(raw.wb_coeffs)?;
    let matrix = camera_metric(raw)?;
    let metric = kernel::Metric::new(matrix)?;
    Ok(Self { wb, matrix, pattern, metric })
  }
}
fn camera_metric(raw: &RawImage) -> Result<[[f32; 3]; 3], DemosaicError> {
  let c = raw.color_matrix.get(&Illuminant::D65).ok_or(DemosaicError::InvalidMetric)?;
  if c.len() != 9 || c.iter().any(|x| !x.is_finite()) {
    return Err(DemosaicError::InvalidMetric);
  }
  let camera: [[f32; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| c[i * 3 + j]));
  let a = multiply(&camera, &SRGB_TO_XYZ_D65);
  if a.iter().any(|r| {
    let sum: f32 = r.iter().sum();
    !sum.is_finite() || sum <= 0.0
  }) {
    return Err(DemosaicError::InvalidMetric);
  }
  let a = normalize(a);
  let p = pseudo_inverse(a);
  let check = multiply(&a, &p);
  if p.iter().flatten().any(|v| !v.is_finite())
    || check.iter().enumerate().any(|(i, r)| {
      r.iter()
        .enumerate()
        .any(|(j, v)| !v.is_finite() || (*v - if i == j { 1.0 } else { 0.0 }).abs() > 1e-4)
    })
  {
    return Err(DemosaicError::InvalidMetric);
  }
  let mut matrix = multiply(&SRGB_TO_XYZ_D65, &p);
  for (row, white) in matrix.iter_mut().zip([0.95047, 1.0, 1.08883]) {
    for v in row {
      *v /= white;
    }
  }
  if matrix.iter().flatten().any(|v| !v.is_finite()) {
    return Err(DemosaicError::InvalidMetric);
  }
  Ok(matrix)
}
impl XTransMarkesteijnDemosaic {
  pub fn try_demosaic(&self, request: &XTransDemosaicRequest<'_>) -> Result<Color2D<f32, 3>, DemosaicError> {
    if self.passes != request.options.passes {
      return Err(DemosaicError::InvalidInput("request/pass mismatch"));
    }
    let context = context(request)?;
    let len = request.roi.width().checked_mul(request.roi.height()).ok_or(DemosaicError::InvalidGeometry)?;
    let mut out = kernel::filled(len, [f32::NAN; 3])?;
    let bands = context.bands();
    let budget_workers = request.options.scratch_budget_bytes / self.passes.scratch_bytes();
    let workers = request.options.max_workers.min(budget_workers).min(bands.len());
    if workers == 0 {
      return Err(DemosaicError::ScratchBudget);
    }
    // Split mutable rows before scheduling. Tasks only borrow their exclusive
    // output band; each active task owns one scratch allocation, reused across X.
    let mut tasks = Vec::new();
    let mut rest = out.as_mut_slice();
    for band in bands {
      let n = (band.end - band.begin) * request.roi.width();
      let (part, tail) = rest.split_at_mut(n);
      rest = tail;
      tasks.push((band, part));
    }
    if !rest.is_empty() {
      return Err(DemosaicError::InvalidGeometry);
    }
    let pool = rayon::ThreadPoolBuilder::new()
      .num_threads(workers)
      .build()
      .map_err(|e| DemosaicError::Backend(e.to_string()))?;
    pool.install(|| tasks.into_par_iter().try_for_each(|(band, part)| context.render_band(band, part)))?;
    request.options.check()?;
    validate_output(&out, request.roi)?;
    Ok(Color2D::new_with(out, request.roi.width(), request.roi.height()))
  }
}
fn context<'a>(request: &'a XTransDemosaicRequest<'_>) -> Result<kernel::Context<'a>, DemosaicError> {
  kernel::Context::new(
    &request.pixels.data,
    request.pixels.width,
    request.pixels.height,
    kernel::Roi {
      x: request.roi.x(),
      y: request.roi.y(),
      width: request.roi.width(),
      height: request.roi.height(),
    },
    request.prepared.pattern.clone(),
    request.options.to_white_one,
    request.prepared.wb,
    request.options.passes,
    &request.prepared.metric,
    request.options.cancelled,
  )
}
fn validate_output(data: &[[f32; 3]], roi: Rect) -> Result<(), DemosaicError> {
  if roi.width().checked_mul(roi.height()) != Some(data.len()) || data.iter().flatten().any(|v| !v.is_finite()) {
    return Err(DemosaicError::InvalidInput("invalid external output"));
  }
  Ok(())
}
pub(crate) fn render(
  original: &RawImage,
  pixels: &PixF32,
  cfa: &CFA,
  colors: &PlaneColor,
  roi: Rect,
  prepared: &Prepared,
  options: &XTransDemosaicOptions<'_>,
) -> Result<Color2D<f32, 3>, DemosaicError> {
  let request = XTransDemosaicRequest {
    pixels,
    original,
    cfa,
    colors,
    roi,
    prepared,
    options,
  };
  // Validate all source samples/ROI even before an external provider. A backend
  // cannot turn an invalid CPU context into a fabricated successful image.
  context(&request)?;
  if let Some(provider) = options.external {
    options.check()?;
    let attempt = catch_unwind(AssertUnwindSafe(|| provider.demosaic(&request)));
    options.check()?;
    match attempt {
      Ok(Ok(ExternalDemosaicResult::Ready(out))) if out.width == roi.width() && out.height == roi.height() && validate_output(&out.data, roi).is_ok() => {
        return Ok(out);
      }
      Ok(Err(DemosaicError::Cancelled)) => return Err(DemosaicError::Cancelled),
      Ok(Ok(ExternalDemosaicResult::Unavailable { reason })) => log::warn!("External XTrans unavailable: {}", reason.chars().take(160).collect::<String>()),
      Ok(Err(error)) => log::warn!("External XTrans failed: {}", error.to_string().chars().take(160).collect::<String>()),
      _ => log::warn!("External XTrans invalid output or unwind; using CPU Markesteijn"),
    }
  }
  options.check()?;
  XTransMarkesteijnDemosaic { passes: options.passes }.try_demosaic(&request)
}
