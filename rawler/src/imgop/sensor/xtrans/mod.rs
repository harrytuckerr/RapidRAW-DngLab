// SPDX-License-Identifier: LGPL-2.1
// Copyright 2021 Daniel Vogelbacher <daniel@chaospixel.com>

// This file is reserved for X-Trans specific sensor code.

pub mod demosaic;

/// Explicit borrowed opt-in demosaic; legacy dispatch is unchanged.
pub mod external;
pub mod markesteijn;
