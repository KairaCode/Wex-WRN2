// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! WNR4 record decoders (docs/WNR4-SPEC.md, DRAFT 0.1), written from the specification pages only: `core`
//! holds what every kind shares (sections 1, 3 and 4, and the WNM4 header checks of section 2), `gg` kind 2
//! (section 5), `i4` kind 3 (section 6). Kinds 0 (MX) and 1 (NV) have no decoder here: no WEX4 container may
//! hold them (WEX4-SPEC section 7.3). `testenc` is a test-only encoder.
//!
//! `check_file` (wex4.rs) calls `gg::decode` and `i4::decode` for WEX4 codec 8, after the container's
//! checks of WEX4-SPEC 7.3 (blob kind, model id, kind 2 or 3, the geometry from the tensors).

pub mod core;
pub mod gg;
pub mod i4;
#[cfg(test)]
mod testenc;
