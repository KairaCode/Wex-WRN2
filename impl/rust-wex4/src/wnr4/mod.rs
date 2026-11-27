// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! WNR4 record decoders (docs/WNR4-SPEC.md): `core` holds what every kind shares (sections 1, 3 and 4,
//! and the WNM4 header checks of section 2), `mxnv` kinds 0 (MX) and 1 (NV) for EXPERT records under
//! `REQ_WNR4_EXPERT`, `gg` kind 2 (section 5), `i4` kind 3 (section 6). `testenc` is a test-only encoder.

pub mod core;
pub mod gg;
pub mod i4;
pub mod mxnv;
#[cfg(test)]
pub(crate) mod testenc;
