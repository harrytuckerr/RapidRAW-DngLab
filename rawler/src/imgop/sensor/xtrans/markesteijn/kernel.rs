// SPDX-License-Identifier: LGPL-2.1-only
/*
 * Copyright 2019-2025 LibRaw LLC (info@libraw.org)
 *
 * LibRaw uses code from dcraw.c -- Dave Coffin's raw photo decoder,
 * dcraw.c is copyright 1997-2018 by Dave Coffin, dcoffin a cybercom o net.
 * LibRaw do not use RESTRICTED code from dcraw.c
 *
 * LibRaw is free software; you can redistribute it and/or modify
 * it under the terms of the one of two licenses as you choose:
 * 1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
 *    (See file LICENSE.LGPL provided in LibRaw distribution archive for details).
 * 2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
 *    (See file LICENSE.CDDL provided in LibRaw distribution archive for details).
 *
 * This Rust derivative selects LGPL version 2.1. Frank Markesteijn's algorithm
 * for Fuji X-Trans sensors, ported from LibRaw 0.22.2 commit
 * b93f6e45c194f5df9b02a43b1af9a54b4f41f33f, xtrans_demosaic.cpp and
 * ahd_demosaic.cpp::cielab. Rust/floating adaptation, 2026 RapidRAW contributors.
 * Changes: immutable source, checked indexing/allocation, disjoint tile cores,
 * cancellation, floating reconstruction with no SDR clip, metric-only XYZ
 * quantization, complete borders. This is not integer LibRaw bit equivalence.
 */

use std::fmt;

pub const TILE: usize = 512;
pub const STRIDE: usize = TILE - 16;
const AREA: usize = TILE * TILE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Passes {
  One,
  Three,
}
impl TryFrom<u8> for Passes {
  type Error = Error;
  fn try_from(value: u8) -> Result<Self, Error> {
    match value {
      1 => Ok(Self::One),
      3 => Ok(Self::Three),
      _ => Err(Error::InvalidInput("passes must be 1 or 3")),
    }
  }
}
impl Passes {
  pub fn directions(self) -> usize {
    if self == Self::One { 4 } else { 8 }
  }
  fn count(self) -> usize {
    if self == Self::One { 1 } else { 3 }
  }
  /// Actual vector payload, excluding allocator/Vec headers and shared LUT.
  pub fn scratch_bytes(self) -> usize {
    AREA * (self.directions() * (12 + 4 + 1) + 12)
  }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
  Cancelled,
  InvalidInput(&'static str),
  InvalidGeometry,
  InvalidCfa,
  InvalidWb,
  InvalidMetric,
  InvalidScale,
  ScratchBudget,
  Allocation,
  Backend(String),
}
impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{self:?}")
  }
}
impl std::error::Error for Error {}

pub fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, Error> {
  len.checked_mul(std::mem::size_of::<T>()).ok_or(Error::Allocation)?;
  let mut out = Vec::new();
  out.try_reserve_exact(len).map_err(|_| Error::Allocation)?;
  out.resize(len, value);
  Ok(out)
}
pub fn effective_wb(wb: [f32; 4]) -> Result<[f32; 3], Error> {
  if wb[0].is_nan() {
    return Ok([1.0; 3]);
  }
  if wb[..3].iter().any(|x| !x.is_finite() || *x <= 0.0) {
    return Err(Error::InvalidWb);
  }
  let out = [wb[0] / wb[1], 1.0, wb[2] / wb[1]];
  if out.iter().any(|x| !x.is_finite() || *x <= 0.0) {
    return Err(Error::InvalidWb);
  }
  Ok(out)
}
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Delta {
  y: isize,
  x: isize,
}
impl Delta {
  fn times(self, n: isize) -> Self {
    Self { y: self.y * n, x: self.x * n }
  }
  fn offset(self, y: usize, x: usize) -> (usize, usize) {
    ((y as isize + self.y) as usize, (x as isize + self.x) as usize)
  }
  fn linear(self) -> isize {
    self.y * TILE as isize + self.x
  }
}
#[derive(Clone)]
pub struct Pattern {
  colors: [u8; 36],
  hex: [[[Delta; 8]; 3]; 3],
  solitary: (usize, usize),
}
impl Pattern {
  pub fn new(colors: [u8; 36]) -> Result<Self, Error> {
    // The directional R/B stencils require the complete supported topology,
    // not merely counts, green geometry or the 3px color-swap relation.
    // Accept every sensor-coordinate translation of the genuine pattern;
    // never whitelist an inconsistent camera-table fallback as a new layout.
    const CANONICAL: &[u8; 36] = b"RBGBRGGGRGGBGGBGGRBRGRBGGGBGGRGGRGGB";
    let supported = (0..6).any(|dy| {
      (0..6).any(|dx| {
        (0..36).all(|i| {
          let color = match CANONICAL[((i / 6 + dy) % 6) * 6 + (i % 6 + dx) % 6] {
            b'R' => 0,
            b'G' => 1,
            _ => 2,
          };
          colors[i] == color
        })
      })
    });
    if !supported {
      return Err(Error::InvalidCfa);
    }
    let mut counts = [0usize; 3];
    for c in colors {
      if c > 2 {
        return Err(Error::InvalidCfa);
      }
      counts[c as usize] += 1;
    }
    if !(6..=10).contains(&counts[0]) || !(16..=24).contains(&counts[1]) || !(6..=10).contains(&counts[2]) {
      return Err(Error::InvalidCfa);
    }
    let mut result = Self {
      colors,
      hex: [[[Delta { y: 32700, x: 32700 }; 8]; 3]; 3],
      solitary: (usize::MAX, usize::MAX),
    };
    let orth = [1isize, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1];
    let patt = [
      [0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0],
      [0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1],
    ];
    for row in 0..3 {
      for col in 0..3 {
        let g = usize::from(result.color(row, col) == 1);
        let mut ng = 0;
        for d in (0..10).step_by(2) {
          if result.color_signed(row as isize + orth[d], col as isize + orth[d + 2]) == 1 {
            ng = 0;
          } else {
            ng += 1;
          }
          if ng == 4 {
            result.solitary = (row, col);
          }
          if ng == g + 1 {
            for c in 0..8 {
              result.hex[row][col][c ^ (g * 2 & d)] = Delta {
                y: orth[d] * patt[g][c * 2] + orth[d + 1] * patt[g][c * 2 + 1],
                x: orth[d + 2] * patt[g][c * 2] + orth[d + 3] * patt[g][c * 2 + 1],
              };
            }
          }
        }
      }
    }
    if result.solitary.0 == usize::MAX {
      return Err(Error::InvalidCfa);
    }
    // The Markesteijn stencils repeat every 3 cells (the measured R/B sites
    // repeat every 6). Reject a merely count-compatible arbitrary RGB map.
    for y in 0..6 {
      for x in 0..6 {
        let green = result.color(y, x) == 1;
        // A 3-cell translation preserves green and swaps R/B. The
        // interpolation stencils rely on this full-color relation, not just
        // green geometry/counts; reject a locally swapped R/B map.
        if result.color((y + 3) % 6, x) != 2 - result.color(y, x) || result.color(y, (x + 3) % 6) != 2 - result.color(y, x) {
          return Err(Error::InvalidCfa);
        }
        if green != (result.color((y + 3) % 6, x) == 1) || green != (result.color(y, (x + 3) % 6) == 1) {
          return Err(Error::InvalidCfa);
        }
        for delta in result.hex[y % 3][x % 3] {
          if delta.y.abs() > 2 || delta.x.abs() > 2 {
            return Err(Error::InvalidCfa);
          }
        }
        if !green {
          for d in &result.hex[y % 3][x % 3][..6] {
            if result.color_signed(y as isize + d.y, x as isize + d.x) != 1 {
              return Err(Error::InvalidCfa);
            }
          }
          // All initial-green source accesses must fit the source's 3px halo.
          let hex = result.hex[y % 3][x % 3];
          for d in [hex[0].times(2), hex[1].times(2), hex[2], hex[3], hex[4].times(3), hex[5].times(3)] {
            if d.y.abs() > 3 || d.x.abs() > 3 {
              return Err(Error::InvalidCfa);
            }
          }
        }
        let (sy, sx) = result.solitary;
        if (y + 3 - sy) % 3 == 0 && (x + 3 - sx) % 3 == 0 && (!green || result.color(y, x + 1) == 1) {
          return Err(Error::InvalidCfa);
        }
      }
    }
    Ok(result)
  }
  pub fn color(&self, y: usize, x: usize) -> usize {
    self.colors[(y % 6) * 6 + x % 6] as usize
  }
  fn color_signed(&self, y: isize, x: isize) -> usize {
    self.color(y.rem_euclid(6) as usize, x.rem_euclid(6) as usize)
  }
}

pub struct Metric {
  matrix: [[f32; 3]; 3],
  lut: Vec<f32>,
}
impl Metric {
  pub fn new(matrix: [[f32; 3]; 3]) -> Result<Self, Error> {
    if matrix.iter().flatten().any(|x| !x.is_finite()) {
      return Err(Error::InvalidMetric);
    }
    let mut lut = filled(65536, 0.0)?;
    for (i, v) in lut.iter_mut().enumerate() {
      let t = i as f32 / 65535.0;
      *v = if t > 0.008856 { t.powf(1.0 / 3.0) } else { 7.787 * t + 16.0 / 116.0 };
    }
    Ok(Self { matrix, lut })
  }
  pub fn lab(&self, rgb: [f32; 3]) -> [f32; 3] {
    let mut f = [0.0; 3];
    for (i, out) in f.iter_mut().enumerate() {
      let mut xyz = 0.0;
      for (j, sample) in rgb.iter().enumerate() {
        xyz += self.matrix[i][j] * sample;
      }
      let index = (65535.0 * xyz + 0.5).floor().clamp(0.0, 65535.0) as usize;
      *out = self.lut[index];
    }
    [
      (64.0 * (116.0 * f[1] - 16.0)).trunc(),
      (64.0 * 500.0 * (f[0] - f[1])).trunc(),
      (64.0 * 200.0 * (f[1] - f[2])).trunc(),
    ]
  }
}
#[derive(Clone, Copy, Debug)]
pub struct Roi {
  pub x: usize,
  pub y: usize,
  pub width: usize,
  pub height: usize,
}
/// Tile-row ownership includes outer borders, but the tile writes only its
/// disjoint interior core. No writes to another band's slice are possible.
#[derive(Clone, Copy, Debug)]
pub struct Band {
  pub top: usize,
  pub begin: usize,
  pub end: usize,
}
pub struct Context<'a> {
  mosaic: &'a [f32],
  pub width: usize,
  pub height: usize,
  pub roi: Roi,
  pub pattern: Pattern,
  scale: f32,
  wb: [f32; 3],
  pub passes: Passes,
  pub metric: &'a Metric,
  cancelled: &'a (dyn Fn() -> bool + Sync),
}
impl<'a> Context<'a> {
  pub fn new(
    mosaic: &'a [f32],
    width: usize,
    height: usize,
    roi: Roi,
    pattern: Pattern,
    scale: f32,
    wb: [f32; 3],
    passes: Passes,
    metric: &'a Metric,
    cancelled: &'a (dyn Fn() -> bool + Sync),
  ) -> Result<Self, Error> {
    if !scale.is_finite() || scale <= 0.0 {
      return Err(Error::InvalidScale);
    }
    if width < 512
      || height < 512
      || width > isize::MAX as usize / TILE
      || height > isize::MAX as usize / TILE
      || width.checked_mul(height) != Some(mosaic.len())
      || roi.width == 0
      || roi.height == 0
      || roi.x.checked_add(roi.width).is_none_or(|n| n > width)
      || roi.y.checked_add(roi.height).is_none_or(|n| n > height)
    {
      return Err(Error::InvalidGeometry);
    }
    if wb.iter().any(|v| !v.is_finite() || *v <= 0.0) {
      return Err(Error::InvalidWb);
    }
    let this = Self {
      mosaic,
      width,
      height,
      roi,
      pattern,
      scale,
      wb,
      passes,
      metric,
      cancelled,
    };
    for y in 0..height {
      this.check()?;
      for x in 0..width {
        let m = mosaic[y * width + x];
        if !m.is_finite() || !(m * scale * wb[this.pattern.color(y, x)]).is_finite() {
          return Err(Error::InvalidInput("nonfinite scaled mosaic"));
        }
      }
    }
    Ok(this)
  }
  pub fn check(&self) -> Result<(), Error> {
    if (self.cancelled)() { Err(Error::Cancelled) } else { Ok(()) }
  }
  fn known(&self, y: usize, x: usize, c: usize) -> f32 {
    if self.pattern.color(y, x) == c {
      self.mosaic[y * self.width + x] * self.scale * self.wb[c]
    } else {
      0.0
    }
  }
  fn sample(&self, y: usize, x: usize, d: Delta, c: usize) -> f32 {
    let (y, x) = d.offset(y, x);
    self.known(y, x, c)
  }
  fn green_bounds(&self, y: usize, x: usize) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for d in &self.pattern.hex[y % 3][x % 3][..6] {
      let v = self.sample(y, x, *d, 1);
      lo = lo.min(v);
      hi = hi.max(v);
    }
    (lo, hi)
  }
  pub fn bands(&self) -> Vec<Band> {
    let mut bands = Vec::new();
    for top in (3..self.height - 19).step_by(STRIDE) {
      let begin = if top == 3 { 0 } else { top + 8 };
      let end = if top + STRIDE >= self.height - 19 { self.height } else { top + TILE - 8 };
      let begin = begin.max(self.roi.y);
      let end = end.min(self.roi.y + self.roi.height);
      if begin < end {
        bands.push(Band { top, begin, end });
      }
    }
    bands
  }
  fn border(&self, y: usize, x: usize) -> Result<[f32; 3], Error> {
    let mut out = [0.0; 3];
    let own = self.pattern.color(y, x);
    for (c, v) in out.iter_mut().enumerate() {
      if c == own {
        *v = self.mosaic[y * self.width + x];
        continue;
      }
      let mut found = false;
      for radius in 1usize..=5 {
        let mut sum = 0.0;
        let mut count = 0;
        for yy in y.saturating_sub(radius)..=(y + radius).min(self.height - 1) {
          for xx in x.saturating_sub(radius)..=(x + radius).min(self.width - 1) {
            if self.pattern.color(yy, xx) == c {
              sum += self.known(yy, xx, c);
              count += 1;
            }
          }
        }
        if count > 0 {
          *v = (sum / count as f32) / self.wb[c] / self.scale;
          found = true;
          break;
        }
      }
      if !found {
        return Err(Error::InvalidCfa);
      }
    }
    Ok(out)
  }
  pub fn render_band(&self, band: Band, out: &mut [[f32; 3]]) -> Result<(), Error> {
    self.check()?;
    if !self
      .bands()
      .iter()
      .any(|valid| valid.top == band.top && valid.begin == band.begin && valid.end == band.end)
    {
      return Err(Error::InvalidGeometry);
    }
    if band.begin < self.roi.y || band.end > self.roi.y + self.roi.height || band.begin >= band.end || out.len() != (band.end - band.begin) * self.roi.width {
      return Err(Error::InvalidGeometry);
    }
    for y in band.begin..band.end {
      self.check()?;
      for x in self.roi.x..self.roi.x + self.roi.width {
        if y < 8 || y >= self.height - 8 || x < 8 || x >= self.width - 8 {
          out[(y - band.begin) * self.roi.width + x - self.roi.x] = self.border(y, x)?;
        }
      }
    }
    let mut scratch = Scratch::new(self.passes)?;
    for left in (3..self.width - 19).step_by(STRIDE) {
      let x0 = (if left == 3 { 8 } else { left + 8 }).max(self.roi.x);
      let x1 = (if left + STRIDE >= self.width - 19 { self.width - 8 } else { left + TILE - 8 }).min(self.roi.x + self.roi.width);
      if x0 >= x1 {
        continue;
      }
      self.tile(band.top, left, &mut scratch)?;
      let y0 = band.begin.max(8);
      let y1 = band.end.min(self.height - 8);
      let nd = self.passes.directions();
      for y in y0..y1 {
        self.check()?;
        for x in x0..x1 {
          let q = (y - band.top) * TILE + x - left;
          let mut hm = [0u32; 8];
          for (d, v) in hm[..nd].iter_mut().enumerate() {
            for dy in -2isize..=2 {
              for dx in -2isize..=2 {
                *v += scratch.homo[d * AREA + (q as isize + dy * TILE as isize + dx) as usize] as u32;
              }
            }
          }
          for d in 0..nd - 4 {
            if hm[d] < hm[d + 4] {
              hm[d] = 0;
            } else if hm[d] > hm[d + 4] {
              hm[d + 4] = 0;
            }
          }
          let max = *hm[..nd].iter().max().ok_or(Error::InvalidInput("empty directions"))?;
          let threshold = max - (max >> 3);
          let mut sum = [0.0; 3];
          let mut n = 0;
          for (d, score) in hm[..nd].iter().enumerate() {
            if *score >= threshold {
              for (c, v) in sum.iter_mut().enumerate() {
                *v += scratch.rgb[d * AREA + q][c];
              }
              n += 1;
            }
          }
          let mut pixel = [0.0; 3];
          for c in 0..3 {
            pixel[c] = (sum[c] / n as f32) / self.wb[c] / self.scale;
          }
          let c = self.pattern.color(y, x);
          pixel[c] = self.mosaic[y * self.width + x];
          if pixel.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidInput("interpolation overflow"));
          }
          out[(y - band.begin) * self.roi.width + x - self.roi.x] = pixel;
        }
      }
    }
    self.check()?;
    if out.iter().flatten().any(|v| !v.is_finite()) {
      return Err(Error::InvalidInput("nonfinite output"));
    }
    Ok(())
  }
  fn tile(&self, top: usize, left: usize, s: &mut Scratch) -> Result<(), Error> {
    let rows = TILE.min(self.height - 3 - top);
    let cols = TILE.min(self.width - 3 - left);
    s.rgb.fill([0.0; 3]);
    s.drv.fill(0.0);
    s.homo.fill(0);
    for r in 0..rows {
      self.check()?;
      for c in 0..cols {
        let y = top + r;
        let x = left + c;
        let own = self.pattern.color(y, x);
        let mut pixel = [0.0; 3];
        pixel[own] = self.known(y, x, own);
        if own != 1 {
          pixel[1] = self.green_bounds(y, x).0;
        }
        for d in 0..4 {
          s.rgb[d * AREA + r * TILE + c] = pixel;
        }
      }
    }
    let sg = self.pattern.solitary;
    for r in 0..rows {
      self.check()?;
      for c in 0..cols {
        let y = top + r;
        let x = left + c;
        let f = self.pattern.color(y, x);
        if f == 1 {
          continue;
        }
        let hx = self.pattern.hex[y % 3][x % 3];
        let v = |d, c| self.sample(y, x, d, c);
        let center = self.known(y, x, f);
        let mut green = [0.0; 4];
        green[0] = (174.0 * (v(hx[1], 1) + v(hx[0], 1)) - 46.0 * (v(hx[1].times(2), 1) + v(hx[0].times(2), 1))) / 256.0;
        green[1] = (223.0 * v(hx[3], 1) + 33.0 * v(hx[2], 1) + 92.0 * (center - v(hx[2].times(-1), f))) / 256.0;
        for k in 0..2 {
          let d = hx[4 + k];
          green[2 + k] = (164.0 * v(d, 1) + 92.0 * v(d.times(-2), 1) + 33.0 * (2.0 * center - v(d.times(3), f) - v(d.times(-3), f))) / 256.0;
        }
        let (lo, hi) = self.green_bounds(y, x);
        for (d, g) in green.iter().enumerate() {
          if !g.is_finite() {
            return Err(Error::InvalidInput("green overflow"));
          }
          s.rgb[(d ^ usize::from((y + 3 - sg.0) % 3 == 0)) * AREA + r * TILE + c][1] = g.clamp(lo, hi);
        }
      }
    }
    let mut base = 0;
    for pass in 0..self.passes.count() {
      if pass == 1 {
        let (a, b) = s.rgb.split_at_mut(4 * AREA);
        b.copy_from_slice(a);
        base = 4;
      }
      if pass > 0 {
        for r in 2..rows - 2 {
          self.check()?;
          for c in 2..cols - 2 {
            let y = top + r;
            let x = left + c;
            let f = self.pattern.color(y, x);
            if f == 1 {
              continue;
            }
            let q = r * TILE + c;
            let hx = self.pattern.hex[y % 3][x % 3];
            let (lo, hi) = self.green_bounds(y, x);
            for (d, delta) in hx.iter().enumerate().take(6).skip(3) {
              let plane = base + ((d - 2) ^ usize::from((y + 3 - sg.0) % 3 == 0));
              let o = delta.linear();
              let a = s.rgb[plane * AREA + (q as isize - 2 * o) as usize];
              let b = s.rgb[plane * AREA + (q as isize + o) as usize];
              let center = s.rgb[plane * AREA + q];
              let g = (a[1] + 2.0 * b[1] - a[f] - 2.0 * b[f] + 3.0 * center[f]) / 3.0;
              if !g.is_finite() {
                return Err(Error::InvalidInput("refinement overflow"));
              }
              s.rgb[plane * AREA + q][1] = g.clamp(lo, hi);
            }
          }
        }
      }
      // Solitary green: six directional estimates yield four RGB candidates.
      let start_r = ((top + 4 - sg.0) / 3) * 3 + sg.0;
      let start_c = ((left + 4 - sg.1) / 3) * 3 + sg.1;
      for y in (start_r..top + rows - 2).step_by(3) {
        self.check()?;
        for x in (start_c..left + cols - 2).step_by(3) {
          let q = (y - top) * TILE + x - left;
          let mut h = self.pattern.color(y, x + 1);
          let mut colors = [[0.0; 6]; 3];
          let mut diff = [0.0f32; 6];
          let mut i = 1isize;
          let mut plane = base;
          for d in 0..6 {
            for c in 0..2 {
              let step = i << c;
              let center = s.rgb[plane * AREA + q];
              let a = s.rgb[plane * AREA + (q as isize + step) as usize];
              let b = s.rgb[plane * AREA + (q as isize - step) as usize];
              let g = 2.0 * center[1] - a[1] - b[1];
              colors[h][d] = g + a[h] + b[h];
              if d > 1 {
                diff[d] += (a[1] - b[1] - a[h] + b[h]).powi(2) + g * g;
              }
              h ^= 2;
            }
            if d > 1 && d & 1 == 1 && diff[d - 1] < diff[d] {
              colors[0][d] = colors[0][d - 1];
              colors[2][d] = colors[2][d - 1];
            }
            if d < 2 || d & 1 == 1 {
              s.rgb[plane * AREA + q][0] = colors[0][d] / 2.0;
              s.rgb[plane * AREA + q][2] = colors[2][d] / 2.0;
              plane += 1;
            }
            i ^= TILE as isize ^ 1;
            h ^= 2;
          }
        }
      }
      // Opposite red/blue sites, keeping the source branch/tie order.
      for r in 3..rows - 3 {
        self.check()?;
        for c in 3..cols - 3 {
          let y = top + r;
          let x = left + c;
          let f = 2 - self.pattern.color(y, x);
          if f == 1 {
            continue;
          }
          let q = r * TILE + c;
          let step = if (y + 3 - sg.0) % 3 != 0 { TILE } else { 1 };
          let h = 3 * (step ^ TILE ^ 1);
          for d in 0..4 {
            let p = (base + d) * AREA;
            let center = s.rgb[p + q];
            let i = if d > 1
              || ((d ^ step) & 1) != 0
              || ((center[1] - s.rgb[p + q + step][1]).abs() + (center[1] - s.rgb[p + q - step][1]).abs())
                < 2.0 * ((center[1] - s.rgb[p + q + h][1]).abs() + (center[1] - s.rgb[p + q - h][1]).abs())
            {
              step
            } else {
              h
            };
            s.rgb[p + q][f] = (s.rgb[p + q + i][f] + s.rgb[p + q - i][f] + 2.0 * center[1] - s.rgb[p + q + i][1] - s.rgb[p + q - i][1]) / 2.0;
          }
        }
      }
      // 2x2 green blocks.
      for r in 2..rows - 2 {
        self.check()?;
        let y = top + r;
        if (y + 3 - sg.0) % 3 == 0 {
          continue;
        }
        for c in 2..cols - 2 {
          let x = left + c;
          if (x + 3 - sg.1) % 3 == 0 {
            continue;
          }
          let q = r * TILE + c;
          let hx = self.pattern.hex[y % 3][x % 3];
          for d in 0..4 {
            let p = (base + d) * AREA;
            let a = hx[d * 2].linear();
            let b = hx[d * 2 + 1].linear();
            let aa = s.rgb[p + (q as isize + a) as usize];
            let bb = s.rgb[p + (q as isize + b) as usize];
            let center = s.rgb[p + q];
            for color in [0, 2] {
              s.rgb[p + q][color] = if a + b != 0 {
                (3.0 * center[1] - 2.0 * aa[1] - bb[1] + 2.0 * aa[color] + bb[color]) / 3.0
              } else {
                (2.0 * center[1] - aa[1] - bb[1] + aa[color] + bb[color]) / 2.0
              };
            }
          }
        }
      }
      for d in base..base + 4 {
        for r in 0..rows {
          self.check()?;
          if s.rgb[d * AREA + r * TILE..d * AREA + r * TILE + cols].iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::InvalidInput("RGB overflow"));
          }
        }
      }
    }
    let dirs = [1isize, TILE as isize, TILE as isize + 1, TILE as isize - 1];
    let nd = self.passes.directions();
    for d in 0..nd {
      for r in 2..rows - 2 {
        self.check()?;
        for c in 2..cols - 2 {
          let q = r * TILE + c;
          s.lab[q] = self.metric.lab(s.rgb[d * AREA + q]);
        }
      }
      for r in 3..rows - 3 {
        self.check()?;
        for c in 3..cols - 3 {
          let q = r * TILE + c;
          let f = dirs[d & 3];
          let center = s.lab[q];
          let a = s.lab[(q as isize + f) as usize];
          let b = s.lab[(q as isize - f) as usize];
          s.drv[d * AREA + q] = directional_derivative(center, a, b);
        }
      }
    }
    for r in 4..rows - 4 {
      self.check()?;
      for c in 4..cols - 4 {
        let q = r * TILE + c;
        let mut threshold = f32::MAX;
        for d in 0..nd {
          threshold = threshold.min(s.drv[d * AREA + q]);
        }
        threshold *= 8.0;
        for d in 0..nd {
          let mut n = 0;
          for dy in -1isize..=1 {
            for dx in -1isize..=1 {
              if s.drv[d * AREA + (q as isize + dy * TILE as isize + dx) as usize] <= threshold {
                n += 1;
              }
            }
          }
          s.homo[d * AREA + q] = n;
        }
      }
    }
    Ok(())
  }
}
// This private helper is the actual production derivative used by tile().
fn directional_derivative(center: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
  let g = 2.0 * center[0] - a[0] - b[0];
  // Lab samples are integral. Match the pinned signed integer quotients
  // (toward zero), while retaining float sums/squares to avoid i32 overflow.
  let red_quotient = (g * 500.0 / 232.0).trunc();
  let blue_quotient = (g * 500.0 / 580.0).trunc();
  g * g + (2.0 * center[1] - a[1] - b[1] + red_quotient).powi(2) + (2.0 * center[2] - a[2] - b[2] - blue_quotient).powi(2)
}

struct Scratch {
  rgb: Vec<[f32; 3]>,
  lab: Vec<[f32; 3]>,
  drv: Vec<f32>,
  homo: Vec<u8>,
}
impl Scratch {
  fn new(passes: Passes) -> Result<Self, Error> {
    let n = passes.directions() * AREA;
    Ok(Self {
      rgb: filled(n, [0.0; 3])?,
      lab: filled(AREA, [0.0; 3])?,
      drv: filled(n, 0.0)?,
      homo: filled(n, 0)?,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  fn pattern() -> [u8; 36] {
    let text = b"RBGBRGGGRGGBGGBGGRBRGRBGGGBGGRGGRGGB";
    std::array::from_fn(|i| match text[i] {
      b'R' => 0,
      b'G' => 1,
      _ => 2,
    })
  }
  fn identity() -> Metric {
    Metric::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unwrap()
  }
  fn flat(passes: Passes, k: f32, wb: [f32; 3], rgb: [f32; 3], width: usize, height: usize) -> Vec<[f32; 3]> {
    let p = Pattern::new(pattern()).unwrap();
    let m: Vec<_> = (0..width * height).map(|i| rgb[p.color(i / width, i % width)] / k).collect();
    let metric = identity();
    let c = Context::new(&m, width, height, Roi { x: 0, y: 0, width, height }, p, k, wb, passes, &metric, &|| false).unwrap();
    let mut out = vec![[f32::NAN; 3]; width * height];
    for b in c.bands() {
      c.render_band(b, &mut out[b.begin * width..b.end * width]).unwrap();
    }
    out
  }
  #[test]
  fn passes_and_checked_storage() {
    assert_eq!(Passes::try_from(1), Ok(Passes::One));
    assert_eq!(Passes::try_from(3), Ok(Passes::Three));
    for n in [0, 2, 255] {
      assert!(Passes::try_from(n).is_err());
    }
    assert_eq!(Passes::One.scratch_bytes(), AREA * 80);
    assert_eq!(Passes::Three.scratch_bytes(), AREA * 148);
    assert_eq!(filled::<u64>(usize::MAX, 0), Err(Error::Allocation));
  }
  #[test]
  fn all_cfa_phases_and_reject_malformed() {
    let p = pattern();
    for y in 0..6 {
      for x in 0..6 {
        let shifted = std::array::from_fn(|i| p[((i / 6 + y) % 6) * 6 + (i % 6 + x) % 6]);
        let pattern = Pattern::new(shifted).unwrap();
        let mosaic: Vec<f32> = (0..512 * 512).map(|i| [0.125, 0.25, 0.5][pattern.color(i / 512, i % 512)]).collect();
        let metric = identity();
        let roi = Roi {
          x: 11,
          y: 13,
          width: 7,
          height: 9,
        };
        let context = Context::new(&mosaic, 512, 512, roi, pattern, 1.0, [1.0; 3], Passes::One, &metric, &|| false).unwrap();
        let mut output = vec![[f32::NAN; 3]; roi.width * roi.height];
        context.render_band(context.bands()[0], &mut output).unwrap();
        for pixel in output {
          for (value, truth) in pixel.into_iter().zip([0.125, 0.25, 0.5]) {
            assert!((value - truth).abs() <= 1e-6);
          }
        }
      }
    }
    let mut bad = p;
    bad.swap(0, 1);
    assert!(Pattern::new(bad).is_err());
    assert!(Pattern::new([1; 36]).is_err());
  }
  #[test]
  fn wb_missing_channels_and_lab_known_answers() {
    assert_eq!(effective_wb([2.0, 1.0, 4.0, f32::NAN]), Ok([2.0, 1.0, 4.0]));
    assert_eq!(effective_wb([f32::NAN, 2.0, 3.0, f32::NAN]), Ok([1.0; 3]));
    for w in [[1.0, f32::NAN, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, f32::INFINITY, 0.0]] {
      assert_eq!(effective_wb(w), Err(Error::InvalidWb));
    }
    let metric = identity();
    assert_eq!(metric.lab([0.0; 3]), [0.0; 3]);
    assert_eq!(metric.lab([1.0; 3]), [6400.0, 0.0, 0.0]);
    assert_eq!(metric.lab([4.0; 3]), [6400.0, 0.0, 0.0]);
  }
  #[test]
  fn flat_one_pass_scale_wb_and_border() {
    let out = flat(Passes::One, 262144.0, [2.0, 1.0, 4.0], [0.125, 0.25, 0.5], 512, 512);
    for p in out {
      for (v, t) in p.into_iter().zip([0.125, 0.25, 0.5]) {
        assert!((v * 262144.0 - t).abs() <= 1e-6);
      }
    }
  }
  #[test]
  fn flat_three_pass_hdr_is_unclipped() {
    let out = flat(Passes::Three, 1.0, [2.0, 1.0, 4.0], [2.0, 1.5, 4.0], 512, 512);
    for p in out {
      for (v, t) in p.into_iter().zip([2.0, 1.5, 4.0]) {
        assert!((v - t).abs() <= 1e-6 * t);
      }
    }
  }
  #[test]
  fn roi_source_phase_and_disjoint_seam_ownership() {
    let width = 1024;
    let height = 528;
    let p = Pattern::new(pattern()).unwrap();
    let m: Vec<_> = (0..width * height).map(|i| [0.125, 0.25, 0.5][p.color(i / width, i % width)]).collect();
    let metric = identity();
    let roi = Roi {
      x: 503,
      y: 501,
      width: 15,
      height: 19,
    };
    let c = Context::new(&m, width, height, roi, p, 1.0, [1.0; 3], Passes::Three, &metric, &|| false).unwrap();
    let bands = c.bands();
    assert_eq!(bands.len(), 2);
    assert_eq!(bands[0].end, bands[1].begin);
    let mut output = vec![[f32::NAN; 3]; roi.width * roi.height];
    let mut rest = output.as_mut_slice();
    for b in bands {
      let (n, r) = rest.split_at_mut((b.end - b.begin) * roi.width);
      c.render_band(b, n).unwrap();
      rest = r;
    }
    assert!(rest.is_empty());
    for pix in output {
      for (v, t) in pix.into_iter().zip([0.125, 0.25, 0.5]) {
        assert!((v - t).abs() < 1e-6);
      }
    }
  }
  #[test]
  fn deterministic_cancellation_and_validation() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let p = Pattern::new(pattern()).unwrap();
    let metric = identity();
    let m = vec![0.5; 512 * 512];
    let roi = Roi {
      x: 0,
      y: 0,
      width: 512,
      height: 512,
    };
    let calls = AtomicUsize::new(0);
    let cancelled = || calls.fetch_add(1, Ordering::SeqCst) > 520;
    let c = Context::new(&m, 512, 512, roi, p.clone(), 1.0, [1.0; 3], Passes::One, &metric, &cancelled).unwrap();
    let mut out = vec![[f32::NAN; 3]; 512 * 512];
    assert_eq!(c.render_band(c.bands()[0], &mut out), Err(Error::Cancelled));
    assert!(Context::new(&m, 512, 512, roi, p.clone(), 0.0, [1.0; 3], Passes::One, &metric, &|| false).is_err());
    let mut bad = m;
    bad[10] = f32::NAN;
    assert!(Context::new(&bad, 512, 512, roi, p, 1.0, [1.0; 3], Passes::One, &metric, &|| false).is_err());
  }
  #[test]
  fn varying_mosaic_preserves_measured_samples_and_repeats_bits() {
    let p = Pattern::new(pattern()).unwrap();
    let m: Vec<_> = (0..512 * 512)
      .map(|i| (((i % 512) * 7 + (i / 512) * 11 + (i % 17) * 3) % 1024) as f32 / 512.0)
      .collect();
    let metric = identity();
    let c = Context::new(
      &m,
      512,
      512,
      Roi {
        x: 0,
        y: 0,
        width: 512,
        height: 512,
      },
      p,
      1.0,
      [1.0; 3],
      Passes::Three,
      &metric,
      &|| false,
    )
    .unwrap();
    let mut a = vec![[f32::NAN; 3]; m.len()];
    let mut b = a.clone();
    for band in c.bands() {
      c.render_band(band, &mut a[band.begin * 512..band.end * 512]).unwrap();
      c.render_band(band, &mut b[band.begin * 512..band.end * 512]).unwrap();
    }
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
      for j in 0..3 {
        assert_eq!(x[j].to_bits(), y[j].to_bits());
        assert!(x[j].is_finite());
      }
      assert_eq!(x[c.pattern.color(i / 512, i % 512)].to_bits(), m[i].to_bits());
    }
  }
  #[test]
  fn metric_signed_quotient_known_answers() {
    // Independent signed-integer quotient answers, frozen before this test.
    let rows = [
      ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 5.0),
      ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 5.0),
      ([2.0, 3.0, -1.0], [1.0, -2.0, 4.0], [0.0, 1.0, -3.0], 203.0),
      ([-1.0, -2.0, 3.0], [1.0, 4.0, -1.0], [0.0, -3.0, 2.0], 179.0),
      ([1.0, 4.0, -2.0], [1.0, -1.0, 3.0], [1.0, 2.0, -1.0], 85.0),
      ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 21.0),
    ];
    for (center, plus, minus, expected) in rows {
      assert_eq!(
        directional_derivative(center, plus, minus),
        expected,
        "fixed signed quotient: center={center:?}, plus={plus:?}, minus={minus:?}"
      );
    }
  }
  #[test]
  fn coherent_rb_layout_rejected() {
    let original = pattern();
    let mut coherent = original;
    for i in [8, 11, 26, 29] {
      coherent[i] = 2 - coherent[i];
    }
    // This negative retains the old count, green-mask and 3px constraints.
    for color in 0..3 {
      assert_eq!(
        original.iter().filter(|&&c| c == color).count(),
        coherent.iter().filter(|&&c| c == color).count()
      );
    }
    for y in 0..6 {
      for x in 0..6 {
        assert_eq!(original[y * 6 + x] == 1, coherent[y * 6 + x] == 1);
        assert_eq!(coherent[((y + 3) % 6) * 6 + x], 2 - coherent[y * 6 + x]);
        assert_eq!(coherent[y * 6 + (x + 3) % 6], 2 - coherent[y * 6 + x]);
      }
    }
    assert!(matches!(Pattern::new(coherent), Err(Error::InvalidCfa)));
    // Shifted forms are invalid too; genuine positive phases remain exercised
    // by the original all_cfa_phases_and_reject_malformed renderer test.
    for y in 0..6 {
      for x in 0..6 {
        let shifted = std::array::from_fn(|i| coherent[((i / 6 + y) % 6) * 6 + (i % 6 + x) % 6]);
        assert!(matches!(Pattern::new(shifted), Err(Error::InvalidCfa)));
      }
    }
  }
}
