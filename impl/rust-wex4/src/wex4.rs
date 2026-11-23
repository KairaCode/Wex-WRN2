// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! The WEX4 container: the header page (WEX4-SPEC section 4), the index (section 5) and the three
//! steps of validation (section 6), with the record codecs of section 7.

use crate::sha256::{hex, sha256, Sha256};
use crate::wnr4::{gg, i4};
use crate::{wbp1, wnr3};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub const PAGE: u64 = 4096;
pub const MAGIC: u32 = 0x5745_5834;
const MIB_256: u64 = 1 << 28;
const GIB: u64 = 1 << 30;

/// Why a file is refused. The names are the `WEX4_E_*` names of `MANIFEST.tsv`; which class a rule
/// falls in is this reader's choice where the specification does not say (SPEC-NOTES.md, part B).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Magic,
    Hash,
    Version,
    Feature,
    Header,
    Index,
    Name,
    Tensor,
    Blob,
    Record,
    Expert,
    Layout,
    Data,
    Io,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::Magic => "WEX4_E_MAGIC",
            Class::Hash => "WEX4_E_HASH",
            Class::Version => "WEX4_E_VERSION",
            Class::Feature => "WEX4_E_FEATURE",
            Class::Header => "WEX4_E_HEADER",
            Class::Index => "WEX4_E_INDEX",
            Class::Name => "WEX4_E_NAME",
            Class::Tensor => "WEX4_E_TENSOR",
            Class::Blob => "WEX4_E_BLOB",
            Class::Record => "WEX4_E_RECORD",
            Class::Expert => "WEX4_E_EXPERT",
            Class::Layout => "WEX4_E_LAYOUT",
            Class::Data => "WEX4_E_DATA",
            Class::Io => "WEX4_E_IO",
        }
    }
}

#[derive(Debug)]
pub struct Reject {
    pub class: Class,
    pub why: String,
}

pub type Res<T> = Result<T, Reject>;

macro_rules! ensure {
    ($cond:expr, $class:ident, $($fmt:tt)+) => {
        if !$cond {
            return Err(Reject { class: Class::$class, why: format!($($fmt)+) });
        }
    };
}

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn u64le(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}
fn sha_at(b: &[u8], o: usize) -> [u8; 32] {
    let mut a = [0u8; 32];
    a.copy_from_slice(&b[o..o + 32]);
    a
}

/// n rounded up to a multiple of 4096; n must be at most 2^62.
pub fn align(n: u64) -> u64 {
    (n + (PAGE - 1)) & !(PAGE - 1)
}

fn div_ceil(a: u64, b: u64) -> u64 {
    a / b + u64::from(a % b != 0)
}

/// Bits per element of the 22 known dtype codes (section 5.4); None for an opaque code.
pub fn dtype_bits(code: u8) -> Option<u64> {
    const BITS: [u64; 22] = [8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 64, 64, 64, 64, 4, 6, 6, 8, 8];
    if (1..=22).contains(&code) {
        Some(BITS[code as usize - 1])
    } else {
        None
    }
}

pub const DTYPE_NAMES: [&str; 22] = [
    "BOOL",
    "U8",
    "I8",
    "F8_E5M2",
    "F8_E4M3",
    "F8_E8M0",
    "I16",
    "U16",
    "F16",
    "BF16",
    "I32",
    "U32",
    "F32",
    "I64",
    "U64",
    "F64",
    "C64",
    "F4",
    "F6_E2M3",
    "F6_E3M2",
    "F8_E4M3FNUZ",
    "F8_E5M2FNUZ",
];

const DT_U8: u8 = 2;
const DT_I8: u8 = 3;
const DT_F8_E8M0: u8 = 6;
const DT_F32: u8 = 13;

pub const CODEC_RAW: u16 = 0;
pub const CODEC_WNR3: u16 = 5;
pub const CODEC_WBP1: u16 = 6;
/// WNR4 records (draft, section 7.3), kinds GG and I4.
pub const CODEC_WNR4: u16 = 8;
/// A WNM4 model blob (section 5.1).
const BLOB_WNM4: u32 = 3;
pub const FLAG_VALUE_EXACT: u16 = 1;
pub const FLAG_EXPERT: u16 = 2;

#[derive(Clone, Debug)]
pub struct Header {
    pub major: u16,
    pub minor: u16,
    pub required: u64,
    pub optional: u64,
    pub file_bytes: u64,
    pub index_offset: u64,
    pub index_bytes: u64,
    pub blob_count: u32,
    pub record_count: u32,
    pub tensor_count: u32,
    pub num_layers: u32,
    pub num_experts: u32,
    pub max_decoded: u32,
    pub names_bytes: u64,
    pub meta_bytes: u64,
    pub total_decoded: u64,
    pub index_sha256: [u8; 32],
    pub root: [u8; 32],
}

/// Step 1 of section 6, in the order the section gives.
pub fn parse_header(page: &[u8], real_size: u64) -> Res<Header> {
    ensure!(page.len() >= PAGE as usize, Header, "the file has {} bytes, less than a header page", real_size);
    let magic = u32le(page, 0);
    ensure!(magic == MAGIC, Magic, "magic {:#010x} is not WEX4", magic);
    ensure!(sha256(&page[..4064]) == page[4064..4096], Hash, "header_sha256 does not match bytes [0, 4064)");
    let h = Header {
        major: u16le(page, 4),
        minor: u16le(page, 6),
        required: u64le(page, 8),
        optional: u64le(page, 16),
        file_bytes: u64le(page, 24),
        index_offset: u64le(page, 32),
        index_bytes: u64le(page, 40),
        blob_count: u32le(page, 48),
        record_count: u32le(page, 52),
        tensor_count: u32le(page, 56),
        num_layers: u32le(page, 60),
        num_experts: u32le(page, 64),
        max_decoded: u32le(page, 68),
        names_bytes: u64le(page, 72),
        meta_bytes: u64le(page, 80),
        total_decoded: u64le(page, 88),
        index_sha256: sha_at(page, 96),
        root: sha_at(page, 4064),
    };
    ensure!(h.major == 4, Version, "version_major {} is not 4", h.major);
    ensure!(h.required & !1 == 0, Feature, "unknown required feature bits {:#x}", h.required & !1);
    ensure!(h.file_bytes == real_size, Header, "file_bytes {} but the file has {} bytes", h.file_bytes, real_size);
    ensure!(h.file_bytes % PAGE == 0, Header, "file_bytes {} is not a multiple of 4096", h.file_bytes);
    ensure!(h.blob_count <= 1 << 16, Header, "blob_count {} above 2^16", h.blob_count);
    ensure!(h.record_count <= 1 << 24, Header, "record_count {} above 2^24", h.record_count);
    ensure!(h.tensor_count <= 1 << 24, Header, "tensor_count {} above 2^24", h.tensor_count);
    ensure!(h.names_bytes <= MIB_256, Header, "names_bytes {} above 2^28", h.names_bytes);
    ensure!(h.meta_bytes <= MIB_256, Header, "meta_bytes {} above 2^28", h.meta_bytes);
    ensure!(
        h.num_layers <= 65536 && h.num_experts <= 65536,
        Header,
        "expert grid {} x {} above 65536",
        h.num_layers,
        h.num_experts
    );
    ensure!(
        (h.num_layers == 0) == (h.num_experts == 0),
        Header,
        "expert grid {} x {}: one side is zero",
        h.num_layers,
        h.num_experts
    );
    ensure!(
        h.num_layers as u64 * h.num_experts as u64 <= 1 << 24,
        Header,
        "expert grid {} x {} above 2^24 slots",
        h.num_layers,
        h.num_experts
    );
    ensure!(h.max_decoded as u64 <= GIB, Header, "max_decoded {} above 2^30", h.max_decoded);
    ensure!(
        h.record_count != 0 || (h.max_decoded == 0 && h.total_decoded == 0),
        Header,
        "no records, but max_decoded {} and total_decoded {}",
        h.max_decoded,
        h.total_decoded
    );
    let sum = 64 * h.blob_count as u64
        + 72 * h.record_count as u64
        + 100 * h.tensor_count as u64
        + h.names_bytes
        + h.meta_bytes;
    ensure!(h.index_bytes == sum, Header, "index_bytes {} but the tables and areas add up to {}", h.index_bytes, sum);
    ensure!(h.index_bytes <= MIB_256, Header, "index_bytes {} above 2^28", h.index_bytes);
    ensure!(h.index_offset % PAGE == 0, Header, "index_offset {} not aligned", h.index_offset);
    ensure!(h.index_offset >= PAGE, Header, "index_offset {} inside the header page", h.index_offset);
    ensure!(
        h.index_offset <= h.file_bytes && align(h.index_bytes) <= h.file_bytes - h.index_offset,
        Header,
        "the index [{}, +{}) does not fit before file_bytes {}",
        h.index_offset,
        align(h.index_bytes),
        h.file_bytes
    );
    // section 4: "index_offset is 4096 when the index is empty"
    ensure!(h.index_bytes != 0 || h.index_offset == PAGE, Header, "an empty index at {}, not 4096", h.index_offset);
    Ok(h)
}

#[derive(Clone, Copy, Debug)]
pub struct Blob {
    pub kind: u32,
    pub id: u32,
    pub offset: u64,
    pub bytes: u64,
    pub sha: [u8; 32],
    pub reserved: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Rec {
    pub offset: u64,
    pub stored: u32,
    pub decoded: u32,
    pub codec: u16,
    pub flags: u16,
    pub blob: u32,
    pub owner: u32,
    pub first_tensor: u32,
    pub tensor_count: u32,
    pub sha: [u8; 32],
    pub reserved: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Ten {
    pub name_offset: u32,
    pub name_bytes: u16,
    pub dtype: u8,
    pub rank: u8,
    pub shape: [u64; 8],
    pub record: u32,
    pub record_count: u32,
    pub offset: u64,
    pub nbytes: u64,
}

/// The index bytes with the offsets of its six parts; entries are read from the bytes on demand.
pub struct Index {
    raw: Vec<u8>,
    pub nb: usize,
    pub nr: usize,
    pub nt: usize,
    rec_at: usize,
    ten_at: usize,
    ord_at: usize,
    names_at: usize,
    meta_at: usize,
    meta_len: usize,
}

impl Index {
    pub fn new(raw: Vec<u8>, h: &Header) -> Index {
        let (nb, nr, nt) = (h.blob_count as usize, h.record_count as usize, h.tensor_count as usize);
        let rec_at = 64 * nb;
        let ten_at = rec_at + 72 * nr;
        let ord_at = ten_at + 96 * nt;
        let names_at = ord_at + 4 * nt;
        let meta_at = names_at + h.names_bytes as usize;
        Index { raw, nb, nr, nt, rec_at, ten_at, ord_at, names_at, meta_at, meta_len: h.meta_bytes as usize }
    }

    pub fn blob(&self, i: usize) -> Blob {
        let b = &self.raw[64 * i..64 * i + 64];
        Blob {
            kind: u32le(b, 0),
            id: u32le(b, 4),
            offset: u64le(b, 8),
            bytes: u64le(b, 16),
            sha: sha_at(b, 24),
            reserved: u64le(b, 56),
        }
    }

    pub fn record(&self, i: usize) -> Rec {
        let o = self.rec_at + 72 * i;
        let b = &self.raw[o..o + 72];
        Rec {
            offset: u64le(b, 0),
            stored: u32le(b, 8),
            decoded: u32le(b, 12),
            codec: u16le(b, 16),
            flags: u16le(b, 18),
            blob: u32le(b, 20),
            owner: u32le(b, 24),
            first_tensor: u32le(b, 28),
            tensor_count: u32le(b, 32),
            sha: sha_at(b, 36),
            reserved: u32le(b, 68),
        }
    }

    pub fn tensor(&self, i: usize) -> Ten {
        let o = self.ten_at + 96 * i;
        let b = &self.raw[o..o + 96];
        let mut shape = [0u64; 8];
        for (k, s) in shape.iter_mut().enumerate() {
            *s = u64le(b, 8 + 8 * k);
        }
        Ten {
            name_offset: u32le(b, 0),
            name_bytes: u16le(b, 4),
            dtype: b[6],
            rank: b[7],
            shape,
            record: u32le(b, 72),
            record_count: u32le(b, 76),
            offset: u64le(b, 80),
            nbytes: u64le(b, 88),
        }
    }

    pub fn order(&self, k: usize) -> u32 {
        u32le(&self.raw, self.ord_at + 4 * k)
    }

    /// The name of a tensor whose name range was checked.
    pub fn name(&self, t: &Ten) -> &[u8] {
        let o = self.names_at + t.name_offset as usize;
        &self.raw[o..o + t.name_bytes as usize]
    }

    pub fn metadata(&self) -> &[u8] {
        &self.raw[self.meta_at..self.meta_at + self.meta_len]
    }
}

/// What step 2 found out: the number of data tensors (the zero-size ones follow them) and of expert records.
pub struct Checked {
    pub data_tensors: usize,
    pub expert_records: usize,
}

/// The scale counts (rows x cols) of the three matrices of an expert record whose tensors passed 5.5.
pub fn expert_scales(ix: &Index, r: &Rec) -> [u64; 3] {
    let f = r.first_tensor as usize;
    let mut s = [0u64; 3];
    for (m, v) in s.iter_mut().enumerate() {
        let t = ix.tensor(f + m);
        *v = t.shape[0] * t.shape[1];
    }
    s
}

/// Step 2 of section 6 on an index whose bytes were read: every rule of sections 5.1 to 5.5 and the
/// tiling of section 3.
pub fn check_index(h: &Header, ix: &Index, raw_padded: &[u8]) -> Res<Checked> {
    let ib = h.index_bytes as usize;
    ensure!(sha256(&raw_padded[..ib]) == h.index_sha256, Hash, "index_sha256 does not match the index");
    ensure!(raw_padded[ib..].iter().all(|&b| b == 0), Index, "a non-zero byte in the padding after the index");
    let (nb, nr, nt) = (ix.nb, ix.nr, ix.nt);

    // 5.3, every tensor entry
    let mut name_end = 0u64;
    let mut data_tensors = nt;
    for i in 0..nt {
        let t = ix.tensor(i);
        ensure!(
            t.name_offset as u64 == name_end,
            Name,
            "tensor {}: name_offset {} is not {}",
            i,
            t.name_offset,
            name_end
        );
        ensure!(
            t.name_bytes as u64 <= h.names_bytes - name_end,
            Name,
            "tensor {}: name [{}, +{}) runs past the name pool of {} bytes",
            i,
            t.name_offset,
            t.name_bytes,
            h.names_bytes
        );
        name_end += t.name_bytes as u64;
        let name = ix.name(&t);
        ensure!(!name.contains(&0), Name, "tensor {}: NUL byte in its name", i);
        ensure!(std::str::from_utf8(name).is_ok(), Name, "tensor {}: name is not valid UTF-8", i);
        ensure!(t.rank <= 8, Tensor, "tensor {}: rank {}", i, t.rank);
        ensure!(
            t.shape[t.rank as usize..].iter().all(|&d| d == 0),
            Tensor,
            "tensor {}: shape entries after rank {} not 0",
            i,
            t.rank
        );
        let mut product = 1u64;
        for &d in &t.shape[..t.rank as usize] {
            product = match product.checked_mul(d) {
                Some(p) => p,
                None => {
                    return Err(Reject {
                        class: Class::Tensor,
                        why: format!("tensor {}: shape product overflows 64 bits", i),
                    })
                }
            };
        }
        if let Some(bits) = dtype_bits(t.dtype) {
            let have = t.nbytes.checked_mul(8);
            let want = product.checked_mul(bits);
            ensure!(
                have.is_some() && have == want,
                Tensor,
                "tensor {}: {} bytes for {} elements of {} bits",
                i,
                t.nbytes,
                product,
                bits
            );
        }
        if t.record_count == 0 {
            ensure!(t.nbytes == 0, Tensor, "tensor {}: record_count 0 with {} bytes", i, t.nbytes);
            ensure!(
                t.record == 0 && t.offset == 0,
                Tensor,
                "tensor {}: zero-size tensor names record {} offset {}",
                i,
                t.record,
                t.offset
            );
            if data_tensors == nt {
                data_tensors = i;
            }
        } else {
            ensure!(
                data_tensors == nt,
                Tensor,
                "tensor {}: a data tensor after the zero-size tensor {}",
                i,
                data_tensors
            );
            ensure!(t.nbytes > 0, Tensor, "tensor {}: 0 bytes inside a record", i);
        }
        // where the tensor lies is checked with the records it tiles (section 6, step 2, "its tensors tile it")
    }
    ensure!(
        name_end == h.names_bytes,
        Name,
        "the names fill {} of the {} bytes of the name pool",
        name_end,
        h.names_bytes
    );

    // the name order: every tensor once, strictly increasing names
    for k in 0..nt {
        let a = ix.order(k) as usize;
        ensure!(a < nt, Name, "name order entry {}: tensor {} of {}", k, a, nt);
        if k > 0 {
            let p = ix.order(k - 1) as usize;
            ensure!(
                ix.name(&ix.tensor(p)) < ix.name(&ix.tensor(a)),
                Name,
                "name order entries {} and {} are not strictly increasing",
                k - 1,
                k
            );
        }
    }

    // 5.1, every blob entry
    for i in 0..nb {
        let b = ix.blob(i);
        ensure!(b.bytes >= 1 && b.bytes <= GIB, Blob, "blob {}: {} bytes", i, b.bytes);
        ensure!(b.reserved == 0, Blob, "blob {}: reserved field {:#x}", i, b.reserved);
        if b.kind == 1 {
            ensure!(b.bytes == wnr3::WNM2_BYTES as u64, Blob, "blob {}: a WNM2 model of {} bytes", i, b.bytes);
            ensure!(b.id != 0, Blob, "blob {}: WNM2 model with id 0", i);
        }
    }

    // 5.2 and 5.5, every record entry, in table order
    let vexact_ok = h.required & 1 != 0;
    let mut next_tensor = 0usize;
    let mut split: Option<(usize, u32, u64)> = None; // tensor, parts still to come, bytes so far
    let mut prev_owner: Option<u32> = None;
    let mut experts = 0usize;
    let mut max_dec = 0u32;
    let mut total_dec = 0u64;
    for r in 0..nr {
        let e = ix.record(r);
        ensure!(e.decoded >= 1 && e.decoded as u64 <= GIB, Record, "record {}: decoded_bytes {}", r, e.decoded);
        let raw_size = align(e.decoded as u64);
        ensure!(
            e.stored > 0 && e.stored as u64 % PAGE == 0 && e.stored as u64 <= raw_size,
            Record,
            "record {}: stored_bytes {} for {} decoded",
            r,
            e.stored,
            e.decoded
        );
        ensure!(
            e.flags & !(FLAG_VALUE_EXACT | FLAG_EXPERT) == 0,
            Record,
            "record {}: unknown flag bits {:#x}",
            r,
            e.flags
        );
        ensure!(e.blob as usize <= nb, Record, "record {}: blob {} of {}", r, e.blob, nb);
        ensure!(e.reserved == 0, Record, "record {}: reserved field {:#x}", r, e.reserved);
        let expert = e.flags & FLAG_EXPERT != 0;
        let vexact = e.flags & FLAG_VALUE_EXACT != 0;
        if e.codec == CODEC_RAW {
            ensure!(
                e.stored as u64 == raw_size,
                Record,
                "record {}: RAW with {} stored bytes, not {}",
                r,
                e.stored,
                raw_size
            );
            ensure!(e.blob == 0, Record, "record {}: RAW with a blob", r);
        } else {
            ensure!((e.stored as u64) < raw_size, Record, "record {}: codec {} not smaller than raw", r, e.codec);
        }
        if e.codec == CODEC_WNR3 {
            ensure!(expert, Record, "record {}: WNR3 without the EXPERT flag", r);
            ensure!(e.blob != 0, Record, "record {}: WNR3 without a model blob", r);
            ensure!(
                ix.blob(e.blob as usize - 1).kind == 1,
                Record,
                "record {}: WNR3 whose blob is not a WNM2 model",
                r
            );
        }
        if e.codec == CODEC_WBP1 {
            ensure!(e.blob == 0 && e.flags == 0, Record, "record {}: WBP1 with a blob or flags", r);
            let min = align(16 + 6 * div_ceil(e.decoded as u64, 32768));
            ensure!(
                e.stored as u64 >= min,
                Record,
                "record {}: WBP1 of {} bytes below its minimum {}",
                r,
                e.stored,
                min
            );
        }
        ensure!(!vexact || (e.codec == CODEC_WNR3 && vexact_ok), Record, "record {}: VALUE_EXACT not allowed here", r);
        if !expert {
            ensure!(e.owner as usize == r, Record, "record {}: owner {} is not its index", r, e.owner);
        }

        // the tensors that tile this record, or the part of a split tensor that it is
        if e.tensor_count == 0 {
            let (t, left, sum) = match split {
                Some(s) => s,
                None => {
                    return Err(Reject {
                        class: Class::Record,
                        why: format!("record {}: no tensor and no split tensor before it", r),
                    })
                }
            };
            ensure!(
                e.first_tensor as usize == t,
                Record,
                "record {}: continuation of tensor {}, not of {}",
                r,
                e.first_tensor,
                t
            );
            let sum = sum + e.decoded as u64;
            if left == 1 {
                let n = ix.tensor(t).nbytes;
                ensure!(sum == n, Record, "tensor {}: its parts hold {} bytes, it has {}", t, sum, n);
                split = None;
            } else {
                split = Some((t, left - 1, sum));
            }
        } else {
            let f = e.first_tensor as usize;
            ensure!(f == next_tensor, Record, "record {}: first_tensor {} is not {}", r, f, next_tensor);
            if let Some((t, left, _)) = split {
                // 5.3: the records after the first part of a split tensor are its parts. A tensor class, as the
                // C reader gives on most such files (SPEC-NOTES, part E)
                return Err(Reject {
                    class: Class::Tensor,
                    why: format!("tensor {}: {} more parts expected, record {} holds tensors", t, left, r),
                });
            }
            ensure!(
                e.tensor_count as usize <= nt - f,
                Record,
                "record {}: {} tensors from {} of {}",
                r,
                e.tensor_count,
                f,
                nt
            );
            let mut at = 0u64;
            let d = e.decoded as u64;
            for j in f..f + e.tensor_count as usize {
                let t = ix.tensor(j);
                ensure!(t.record as usize == r, Tensor, "tensor {}: in record {}, listed by record {}", j, t.record, r);
                ensure!(t.record_count != 0, Tensor, "tensor {}: a zero-size tensor listed by record {}", j, r);
                if t.record_count >= 2 {
                    // 5.3, split tensor: alone in its first record, and the next records are its parts
                    ensure!(
                        e.tensor_count == 1,
                        Tensor,
                        "tensor {}: a split tensor that is not alone in record {}",
                        j,
                        r
                    );
                    ensure!(
                        t.record_count as usize <= nr - r,
                        Tensor,
                        "tensor {}: {} parts from record {} of {}",
                        j,
                        t.record_count,
                        r,
                        nr
                    );
                    ensure!(t.offset == 0, Tensor, "tensor {}: split tensor at offset {}", j, t.offset);
                    split = Some((j, t.record_count - 1, d));
                    at = d;
                } else {
                    ensure!(
                        t.offset == at,
                        Tensor,
                        "tensor {}: offset {}, the previous tensor ends at {}",
                        j,
                        t.offset,
                        at
                    );
                    ensure!(
                        t.nbytes <= d - at,
                        Tensor,
                        "tensor {}: [{}, +{}) runs past record {} of {} bytes",
                        j,
                        t.offset,
                        t.nbytes,
                        r,
                        d
                    );
                    at += t.nbytes;
                }
            }
            ensure!(at == e.decoded as u64, Tensor, "record {}: its tensors cover {} of {} bytes", r, at, e.decoded);
            next_tensor = f + e.tensor_count as usize;
        }

        if expert {
            check_expert(ix, h, r, &e)?;
            if let Some(p) = prev_owner {
                ensure!(
                    e.owner > p,
                    Expert,
                    "record {}: owner {:#x} not above the previous expert's {:#x}",
                    r,
                    e.owner,
                    p
                );
            }
            prev_owner = Some(e.owner);
            experts += 1;
            if e.codec == CODEC_WNR3 {
                let streams: u64 =
                    expert_scales(ix, &e).iter().map(|&s| div_ceil(s, 16384) + div_ceil(32 * s, 16384)).sum();
                let min = align(32 + 6 * streams);
                // EXPERT, the class of the manifest (vector wnr3-record-smaller-than-its-streams-need, 2026-11-15):
                // the minimum is a rule of 5.2 that needs the shapes of 5.5 (SPEC-NOTES, A5 and F)
                ensure!(
                    e.stored as u64 >= min,
                    Expert,
                    "record {}: WNR3 of {} bytes below its minimum {}",
                    r,
                    e.stored,
                    min
                );
            }
        }
        max_dec = max_dec.max(e.decoded);
        total_dec += e.decoded as u64;
    }
    ensure!(split.is_none(), Record, "a split tensor lacks parts at the end of the record table");
    ensure!(
        next_tensor == data_tensors,
        Record,
        "the records hold {} of the {} data tensors",
        next_tensor,
        data_tensors
    );
    ensure!(
        max_dec == h.max_decoded,
        Record,
        "max_decoded {} but the largest record decodes to {}",
        h.max_decoded,
        max_dec
    );
    ensure!(
        total_dec == h.total_decoded,
        Record,
        "total_decoded {} but the records decode to {}",
        h.total_decoded,
        total_dec
    );
    ensure!(
        h.num_layers == 0 || experts > 0,
        Expert,
        "expert grid {} x {} without an expert record",
        h.num_layers,
        h.num_experts
    );

    check_tiling(h, ix)?;
    Ok(Checked { data_tensors, expert_records: experts })
}

/// 5.5 on an EXPERT record, and its owner inside the grid.
fn check_expert(ix: &Index, h: &Header, r: usize, e: &Rec) -> Res<()> {
    ensure!(e.codec == CODEC_RAW || e.codec == CODEC_WNR3, Expert, "record {}: EXPERT with codec {}", r, e.codec);
    ensure!(e.tensor_count == 6, Expert, "record {}: an expert record with {} tensors", r, e.tensor_count);
    let f = e.first_tensor as usize;
    let dim_ok = |d: u64| (1..=65535).contains(&d);
    for m in 0..3 {
        let s = ix.tensor(f + m);
        let w = ix.tensor(f + 3 + m);
        ensure!(
            s.dtype == DT_U8 || s.dtype == DT_F8_E8M0,
            Expert,
            "record {}: scale tensor {} has dtype {}",
            r,
            f + m,
            s.dtype
        );
        ensure!(
            w.dtype == DT_U8 || w.dtype == DT_I8,
            Expert,
            "record {}: weight tensor {} has dtype {}",
            r,
            f + 3 + m,
            w.dtype
        );
        ensure!(s.rank == 2 && w.rank == 2, Expert, "record {}: matrix {} has ranks {} and {}", r, m, s.rank, w.rank);
        let (rows, cols) = (s.shape[0], s.shape[1]);
        ensure!(dim_ok(rows) && dim_ok(cols), Expert, "record {}: scale shape [{}, {}]", r, rows, cols);
        ensure!(
            w.shape[0] == rows && w.shape[1] == 16 * cols && dim_ok(w.shape[1]),
            Expert,
            "record {}: weight shape [{}, {}] for scales [{}, {}]",
            r,
            w.shape[0],
            w.shape[1],
            rows,
            cols
        );
    }
    let (layer, expert) = (e.owner >> 16, e.owner & 0xffff);
    ensure!(
        layer < h.num_layers && expert < h.num_experts,
        Expert,
        "record {}: owner layer {} expert {} outside the grid {} x {}",
        r,
        layer,
        expert,
        h.num_layers,
        h.num_experts
    );
    Ok(())
}

/// Section 3: the header page, the blobs in table order, the records in table order and the index
/// cover [0, file_bytes) exactly once. A merge of the two ordered lists and the index.
fn check_tiling(h: &Header, ix: &Index) -> Res<()> {
    let mut at = PAGE;
    let (mut bi, mut ri) = (0usize, 0usize);
    let index_len = align(h.index_bytes);
    let mut index_done = index_len == 0; // then it is at 4096 (step 1)
    while at < h.file_bytes || bi < ix.nb || ri < ix.nr || !index_done {
        let len;
        if !index_done && h.index_offset == at {
            len = index_len;
            index_done = true;
        } else if bi < ix.nb && ix.blob(bi).offset == at {
            len = align(ix.blob(bi).bytes);
            bi += 1;
        } else if ri < ix.nr && ix.record(ri).offset == at {
            len = ix.record(ri).stored as u64;
            ri += 1;
        } else {
            let what = if at >= h.file_bytes {
                "an extent outside the file or out of table order".to_string()
            } else {
                format!("no extent starts at {} (gap, overlap or out of table order)", at)
            };
            return Err(Reject { class: Class::Layout, why: what });
        }
        ensure!(
            len <= h.file_bytes - at,
            Layout,
            "an extent at {} of {} bytes runs past file_bytes {}",
            at,
            len,
            h.file_bytes
        );
        at += len;
    }
    Ok(())
}

fn read_at<R: Read + Seek>(f: &mut R, offset: u64, len: u64) -> Res<Vec<u8>> {
    let mut buf = vec![0u8; len as usize];
    f.seek(SeekFrom::Start(offset))
        .and_then(|_| f.read_exact(&mut buf))
        .map_err(|e| Reject { class: Class::Io, why: format!("read of {} bytes at {}: {}", len, offset, e) })?;
    Ok(buf)
}

/// What a full check found.
pub struct Report {
    pub header: Header,
    pub records: usize,
    pub tensors: usize,
    pub blobs: usize,
    pub expert_records: usize,
    pub decoded: usize,
    /// The records not decoded: of a codec this reader does not know, or codec-8 records that failed a
    /// check of section 7.3 (undecodable, not the file invalid).
    pub undecodable: Vec<usize>,
    /// For each of them, `record N: ` and the reason.
    pub why: Vec<String>,
    /// The content digest of section 5, when every record was decoded.
    pub digest: Option<[u8; 32]>,
    /// With `Options::list`: the SHA-256 of every decoded record and tensor.
    pub listing: Vec<String>,
}

pub struct Options {
    pub decode: bool,
    pub list: bool,
}

/// Steps 1, 2 and 3 of section 6 on a whole file: every hash, every blob and, with `decode`, every
/// record decoded into scratch.
pub fn check_file(path: &str, opt: &Options) -> Res<Report> {
    let mut f = File::open(path).map_err(|e| Reject { class: Class::Io, why: format!("{}: {}", path, e) })?;
    let real = f.metadata().map_err(|e| Reject { class: Class::Io, why: e.to_string() })?.len();
    check(&mut f, real, opt)
}

/// `check_file` on any seekable reader of `real` bytes.
pub fn check<R: Read + Seek>(f: &mut R, real: u64, opt: &Options) -> Res<Report> {
    let page = read_at(f, 0, real.min(PAGE))?;
    let h = parse_header(&page, real)?;
    let raw = read_at(f, h.index_offset, align(h.index_bytes))?;
    let ix = Index::new(raw[..h.index_bytes as usize].to_vec(), &h);
    let checked = check_index(&h, &ix, &raw)?;
    drop(raw);

    // step 3: blobs (SHA-256, zero padding, the WNM2 model and its id), then records. The bytes of a WNM4
    // blob (kind 3) that a codec-8 record names are kept: it is opened as a model when such a record is
    // first decoded (sections 5.1 and 7.3); any other kind-3 blob gets the SHA-256 and padding checks only
    let mut models: Vec<Option<wnr3::Model>> = Vec::with_capacity(ix.nb);
    let mut wnr4 = Wnr4Blobs::new(&ix);
    for i in 0..ix.nb {
        let b = ix.blob(i);
        let bytes = read_at(f, b.offset, align(b.bytes))?;
        let n = b.bytes as usize;
        ensure!(sha256(&bytes[..n]) == b.sha, Hash, "blob {}: SHA-256 does not match", i);
        ensure!(bytes[n..].iter().all(|&v| v == 0), Blob, "blob {}: a non-zero byte in its padding", i);
        if b.kind == 1 {
            let m = wnr3::model_open(&bytes[..n])
                .map_err(|e| Reject { class: Class::Blob, why: format!("blob {}: {}", i, e) })?;
            ensure!(
                m.id == b.id,
                Blob,
                "blob {}: the entry says id {:#010x}, the model's FNV-1a is {:#010x}",
                i,
                b.id,
                m.id
            );
            models.push(Some(m));
        } else {
            models.push(None);
        }
        if wnr4.named[i] {
            wnr4.bytes[i] = Some(bytes[..n].to_vec());
        }
    }
    let mut digest = Some(Sha256::new());
    let mut undecodable = Vec::new();
    let mut why = Vec::new();
    let mut decoded = 0usize;
    let mut listing = Vec::new();
    let mut part: Option<(usize, u32, Sha256)> = None; // a split tensor: its index, parts to come, its hash
    for r in 0..ix.nr {
        let e = ix.record(r);
        let stored = read_at(f, e.offset, e.stored as u64)?;
        // checked for every record before it is decoded: for a codec-8 record of kind GG the only check of
        // its verbatim slices, which no tile hash covers (section 7.3, "Integrity")
        ensure!(sha256(&stored) == e.sha, Hash, "record {}: SHA-256 does not match", r);
        if !opt.decode {
            digest = None;
            continue;
        }
        let out = match decode_record(&ix, r, &e, &stored, &models, &mut wnr4) {
            Ok(Decoded::Out(out)) => out,
            Ok(Decoded::Undecodable(reason)) => {
                undecodable.push(r);
                why.push(format!("record {}: {}", r, reason));
                digest = None;
                part = None; // a split tensor with an undecodable part has no hash to list
                continue;
            }
            Err(why) => {
                return Err(Reject { class: Class::Data, why: format!("record {} (codec {}): {}", r, e.codec, why) })
            }
        };
        decoded += 1;
        if opt.list {
            listing.push(format!(
                "  record {} codec {} decoded {} sha256 {}",
                r,
                e.codec,
                out.len(),
                hex(&sha256(&out))
            ));
            if e.tensor_count == 0 {
                if let Some((j, left, mut s)) = part.take() {
                    s.update(&out);
                    if left == 1 {
                        listing.push(tensor_line(&ix, j, s.finish()));
                    } else {
                        part = Some((j, left - 1, s));
                    }
                }
            }
            for j in e.first_tensor as usize..(e.first_tensor + e.tensor_count) as usize {
                let t = ix.tensor(j);
                if t.record_count == 1 {
                    listing.push(tensor_line(&ix, j, sha256(&out[t.offset as usize..(t.offset + t.nbytes) as usize])));
                } else {
                    let mut s = Sha256::new();
                    s.update(&out);
                    part = Some((j, t.record_count - 1, s));
                }
            }
        }
        if let Some(d) = digest.as_mut() {
            if e.tensor_count == 0 {
                d.update(&out);
            }
            for j in e.first_tensor as usize..(e.first_tensor + e.tensor_count) as usize {
                let t = ix.tensor(j);
                digest_fields(d, &ix, &t);
                if t.record_count == 1 {
                    d.update(&out[t.offset as usize..(t.offset + t.nbytes) as usize]);
                } else {
                    d.update(&out);
                }
            }
        }
    }
    let digest = digest.map(|mut d| {
        for j in checked.data_tensors..ix.nt {
            digest_fields(&mut d, &ix, &ix.tensor(j));
        }
        d.finish()
    });
    if opt.list && opt.decode {
        for j in checked.data_tensors..ix.nt {
            listing.push(tensor_line(&ix, j, sha256(&[])));
        }
    }
    Ok(Report {
        header: h,
        records: ix.nr,
        tensors: ix.nt,
        blobs: ix.nb,
        expert_records: checked.expert_records,
        decoded,
        undecodable,
        why,
        digest,
        listing,
    })
}

fn tensor_line(ix: &Index, j: usize, sha: [u8; 32]) -> String {
    let t = ix.tensor(j);
    format!("  tensor {} {:?} nbytes {} sha256 {}", j, String::from_utf8_lossy(ix.name(&t)), t.nbytes, hex(&sha))
}

/// The fields of a tensor in the content digest: name length u16, name, dtype u8, rank u8, the 8
/// shape slots u64, nbytes u64 (section 5).
fn digest_fields(d: &mut Sha256, ix: &Index, t: &Ten) {
    d.update(&t.name_bytes.to_le_bytes());
    d.update(ix.name(t));
    d.update(&[t.dtype, t.rank]);
    for s in t.shape {
        d.update(&s.to_le_bytes());
    }
    d.update(&t.nbytes.to_le_bytes());
}

/// What decoding a record gave, when it did not refuse the file.
enum Decoded {
    Out(Vec<u8>),
    /// A codec this reader does not know, or a codec-8 record that failed a check of section 7.3: the
    /// record is undecodable, the file is not invalid.
    Undecodable(String),
}

/// Decodes one record into scratch; an Err refuses the file.
fn decode_record(
    ix: &Index,
    r: usize,
    e: &Rec,
    stored: &[u8],
    models: &[Option<wnr3::Model>],
    wnr4: &mut Wnr4Blobs,
) -> Result<Decoded, String> {
    let n = e.decoded as usize;
    match e.codec {
        CODEC_RAW => {
            if stored[n..].iter().any(|&b| b != 0) {
                return Err("a non-zero byte in the padding of a RAW record".into());
            }
            Ok(Decoded::Out(stored[..n].to_vec()))
        }
        CODEC_WBP1 => {
            let mut out = vec![0u8; n];
            wbp1::decode(stored, r as u32, e.decoded, &mut out)?;
            Ok(Decoded::Out(out))
        }
        CODEC_WNR4 => Ok(match decode_wnr4(ix, e, stored, wnr4) {
            Ok(out) => Decoded::Out(out),
            Err(why) => Decoded::Undecodable(why),
        }),
        CODEC_WNR3 => {
            let model = models[e.blob as usize - 1].as_ref().ok_or("the record's blob is not a WNM2 model")?;
            // 7.1: owner, CZ bit and model id repeated by the index
            if stored.len() < 32 {
                return Err("shorter than a WNR3 header".into());
            }
            let owner = u32le(stored, 28);
            if owner != e.owner {
                return Err(format!("record owner {:#x}, index owner {:#x}", owner, e.owner));
            }
            let cz = stored[27] & 0x08 != 0;
            if cz != (e.flags & FLAG_VALUE_EXACT != 0) {
                return Err(format!("CZ bit {} but VALUE_EXACT flag {}", cz as u8, e.flags & FLAG_VALUE_EXACT));
            }
            if u32le(stored, 4) != ix.blob(e.blob as usize - 1).id {
                return Err("model id differs from the id of the record's blob".into());
            }
            let mut out = vec![0u8; n];
            wnr3::decode(stored, model, expert_scales(ix, e), &mut out)?;
            Ok(Decoded::Out(out))
        }
        c => Ok(Decoded::Undecodable(format!("codec {}, which this reader does not know", c))),
    }
}

/// The WNM4 blobs that codec-8 records name (0-based): their bytes, kept by step 3 after the SHA-256 and
/// padding checks, and the models opened from them, each at most once per kind, at first use.
struct Wnr4Blobs {
    named: Vec<bool>,
    bytes: Vec<Option<Vec<u8>>>,
    gg: Vec<Option<Result<gg::Model, String>>>,
    i4: Vec<Option<Result<i4::Model, String>>>,
}

impl Wnr4Blobs {
    fn new(ix: &Index) -> Wnr4Blobs {
        let mut named = vec![false; ix.nb];
        for r in 0..ix.nr {
            let e = ix.record(r);
            if e.codec == CODEC_WNR4 && e.blob != 0 && ix.blob(e.blob as usize - 1).kind == BLOB_WNM4 {
                named[e.blob as usize - 1] = true;
            }
        }
        let nb = ix.nb;
        Wnr4Blobs {
            named,
            bytes: vec![None; nb],
            gg: (0..nb).map(|_| None).collect(),
            i4: (0..nb).map(|_| None).collect(),
        }
    }
}

/// The ggml type a GG slice of WEX4 dtype `d` has in the record's geometry (7.3; SPEC-NOTES G1 item 4):
/// 42 for dtype 170, 20 for dtype 148, and for any other dtype a type that is neither, so that a tensor of
/// WEX4 dtype 42 or 20 is a verbatim slice.
fn ggml_type(d: u8) -> u32 {
    match d {
        170 => gg::TYPE_Q2_0,
        148 => gg::TYPE_IQ4_NL,
        _ => 0x100 | d as u32,
    }
}

/// The geometry of a GG record (7.3): `tensor_count` 1 to 4, each tensor inside this record only; slice s
/// is the record's s-th tensor.
fn gg_slices(ix: &Index, e: &Rec) -> Result<Vec<gg::Slice>, String> {
    if e.tensor_count < 1 || e.tensor_count > 4 {
        return Err(format!("GG: a record of {} tensors, 1 to 4 allowed", e.tensor_count));
    }
    let mut slices = Vec::with_capacity(e.tensor_count as usize);
    for j in e.first_tensor..e.first_tensor + e.tensor_count {
        let t = ix.tensor(j as usize);
        if t.record_count != 1 {
            return Err(format!("GG: tensor {} is split over {} records", j, t.record_count));
        }
        slices.push(gg::Slice { ggml_type: ggml_type(t.dtype), bytes: t.nbytes });
    }
    Ok(slices)
}

/// The geometry of an I4 record (7.3): `tensor_count` 2, 4 or 6, each tensor inside this record only and
/// of rank 2; with n = `tensor_count / 2`, tensor m < n is U8 [rows, cols / 2] and tensor n + m is F32
/// [rows, cols / 64], cols a multiple of 64. The tiling of section 5.3 already puts each tensor where the
/// one before it ends. The packing is record byte 13, which `i4::decode` reads (pairs = None).
fn i4_matrices(ix: &Index, e: &Rec) -> Result<Vec<i4::Matrix>, String> {
    if !matches!(e.tensor_count, 2 | 4 | 6) {
        return Err(format!("I4: a record of {} tensors, 2, 4 or 6 allowed", e.tensor_count));
    }
    let n = e.tensor_count / 2;
    let mut mats = Vec::with_capacity(n as usize);
    for m in 0..n {
        let (ci, si) = (e.first_tensor + m, e.first_tensor + n + m);
        let (c, s) = (ix.tensor(ci as usize), ix.tensor(si as usize));
        if c.record_count != 1 || s.record_count != 1 || c.rank != 2 || s.rank != 2 {
            return Err(format!("I4: tensors {} and {} are not two tensors of rank 2 inside the record", ci, si));
        }
        if c.dtype != DT_U8 || s.dtype != DT_F32 {
            return Err(format!(
                "I4: matrix {} has codes of dtype {} and scales of dtype {} (U8 and F32 expected)",
                m, c.dtype, s.dtype
            ));
        }
        let (rows, half) = (c.shape[0], c.shape[1]);
        let ok = rows <= u32::MAX as u64
            && half <= (u32::MAX / 2) as u64
            && half % 32 == 0
            && s.shape[0] == rows
            && s.shape[1] <= u32::MAX as u64
            && 2 * half == 64 * s.shape[1];
        if !ok {
            return Err(format!(
                "I4: matrix {} has codes [{}, {}] and scales [{}, {}]",
                m, rows, half, s.shape[0], s.shape[1]
            ));
        }
        mats.push(i4::Matrix { rows: rows as u32, cols: 2 * half as u32 });
    }
    Ok(mats)
}

/// A codec-8 record (section 7.3, "At decode"): record kind 2 or 3, a blob of kind 3 that is a valid WNM4
/// model of that kind whose id is the blob entry's, then the decoder of the kind, which compares the
/// record's model id and owner. Every Err makes the record undecodable.
fn decode_wnr4(ix: &Index, e: &Rec, stored: &[u8], w: &mut Wnr4Blobs) -> Result<Vec<u8>, String> {
    let kind = stored[12]; // stored_bytes is at least 4096: the 64-byte header is there
    if kind != gg::KIND && kind != i4::KIND {
        return Err(format!(
            "WNR4 kind {}: a WEX4 record that is not EXPERT may be of kind 2 (GG) or 3 (I4) only",
            kind
        ));
    }
    if e.blob == 0 {
        return Err("codec 8 without a WNM4 blob".into());
    }
    let b = e.blob as usize - 1;
    let be = ix.blob(b);
    if be.kind != BLOB_WNM4 {
        return Err(format!("codec 8 with blob {} of kind {}, not a WNM4 model (kind 3)", b, be.kind));
    }
    let bytes = w.bytes[b].as_deref().unwrap_or(&[]);
    let bad_id = |id: u32| {
        format!("blob {}: a WNM4 model of id {:#010x} in the index, the model's FNV-1a is {:#010x}", b, be.id, id)
    };
    let mut out = vec![0u8; e.decoded as usize];
    if kind == gg::KIND {
        let m = w.gg[b]
            .get_or_insert_with(|| gg::model_open(bytes))
            .as_ref()
            .map_err(|why| format!("blob {}: {}", b, why))?;
        if m.id != be.id {
            return Err(bad_id(m.id));
        }
        gg::decode(stored, m, e.owner, &gg_slices(ix, e)?, &mut out)?;
    } else {
        let m = w.i4[b]
            .get_or_insert_with(|| i4::model_open(bytes))
            .as_ref()
            .map_err(|why| format!("blob {}: {}", b, why))?;
        if m.id != be.id {
            return Err(bad_id(m.id));
        }
        i4::decode(stored, m, e.owner, &i4_matrices(ix, e)?, None, &mut out)?;
    }
    Ok(out)
}

/// Prints the header page and the tables of a file that passed step 1 (for `dump`).
pub fn dump(path: &str) -> Res<()> {
    let mut f = File::open(path).map_err(|e| Reject { class: Class::Io, why: format!("{}: {}", path, e) })?;
    let real = f.metadata().map_err(|e| Reject { class: Class::Io, why: e.to_string() })?.len();
    let page = read_at(&mut f, 0, real.min(PAGE))?;
    let h = parse_header(&page, real)?;
    println!("{:#?}", h);
    println!(
        "version {}.{}, required features {:#x}, optional features {:#x}",
        h.major, h.minor, h.required, h.optional
    );
    println!("root {}", hex(&h.root));
    let raw = read_at(&mut f, h.index_offset, align(h.index_bytes))?;
    let ix = Index::new(raw[..h.index_bytes as usize].to_vec(), &h);
    for i in 0..ix.nb {
        let b = ix.blob(i);
        println!(
            "blob {}: kind {} id {:#010x} offset {} bytes {} reserved {}",
            i, b.kind, b.id, b.offset, b.bytes, b.reserved
        );
    }
    for i in 0..ix.nr {
        let e = ix.record(i);
        println!(
            "record {}: offset {} stored {} decoded {} codec {} flags {} blob {} owner {:#x} first_tensor {} tensor_count {} reserved {}",
            i, e.offset, e.stored, e.decoded, e.codec, e.flags, e.blob, e.owner, e.first_tensor, e.tensor_count, e.reserved
        );
    }
    let pool = h.names_bytes;
    for i in 0..ix.nt {
        let t = ix.tensor(i);
        let name = if t.name_offset as u64 + t.name_bytes as u64 <= pool {
            String::from_utf8_lossy(ix.name(&t)).into_owned()
        } else {
            "<outside the pool>".into()
        };
        let dt = dtype_bits(t.dtype).map(|_| DTYPE_NAMES[t.dtype as usize - 1]).unwrap_or("opaque");
        let rank = (t.rank as usize).min(8);
        println!(
            "tensor {}: {:?} name@{}+{} dtype {} ({}) shape {:?} record {} x{} offset {} nbytes {}",
            i,
            name,
            t.name_offset,
            t.name_bytes,
            t.dtype,
            dt,
            &t.shape[..rank],
            t.record,
            t.record_count,
            t.offset,
            t.nbytes
        );
    }
    let order: Vec<u32> = (0..ix.nt).map(|k| ix.order(k)).collect();
    println!("name order {:?}", order);
    println!("metadata {} bytes", ix.metadata().len());
    Ok(())
}

/// Codec 8 in whole files: the codec-8 files of impl/go-wex4/testdata (made with the repository's codec-8
/// tools) by their wnr4-MANIFEST.tsv, and changed copies of them with every hash recomputed, which must never
/// be refused as files (WEX4-SPEC 7.3).
#[cfg(test)]
mod wnr4_container_tests {
    use super::*;
    use std::io::Cursor;

    fn testdata(name: &str) -> Vec<u8> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../go-wex4/testdata").join(name);
        std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {}", p.display(), e))
    }

    fn check_bytes(b: &[u8]) -> Res<Report> {
        check(&mut Cursor::new(b), b.len() as u64, &Options { decode: true, list: false })
    }

    /// The record SHA-256 of record `rec` and the SHA-256 of blob `blob` (when given), then the index and the
    /// header hashes.
    fn rehash(b: &mut [u8], rec: Option<usize>, blob: Option<usize>) {
        let (io, ib, nb) = (u64le(b, 32) as usize, u64le(b, 40) as usize, u32le(b, 48) as usize);
        if let Some(i) = blob {
            let at = io + 64 * i;
            let (off, n) = (u64le(b, at + 8) as usize, u64le(b, at + 16) as usize);
            let s = sha256(&b[off..off + n]);
            b[at + 24..at + 56].copy_from_slice(&s);
        }
        if let Some(r) = rec {
            let at = io + 64 * nb + 72 * r;
            let (off, n) = (u64le(b, at) as usize, u32le(b, at + 8) as usize);
            let s = sha256(&b[off..off + n]);
            b[at + 36..at + 68].copy_from_slice(&s);
        }
        let s = sha256(&b[io..io + ib]);
        b[96..128].copy_from_slice(&s);
        let s = sha256(&b[..4064]);
        b[4064..4096].copy_from_slice(&s);
    }

    /// Every row: accept with the manifest's root and content digest; bad-sha refused (WEX4_E_HASH); every
    /// other changed copy opens with its record 0 undecodable.
    #[test]
    fn wnr4_manifest_rows() {
        let text = String::from_utf8(testdata("wnr4-MANIFEST.tsv")).unwrap();
        let mut rows = 0;
        for line in text.lines().filter(|l| !l.is_empty() && !l.starts_with('#')) {
            let c: Vec<&str> = line.split('\t').collect();
            let (name, root, verdict, what) = (c[0], c[2], c[3], c[4]);
            rows += 1;
            let res = check_bytes(&testdata(name));
            match (verdict, res) {
                ("accept", Ok(r)) => {
                    assert!(r.undecodable.is_empty(), "{}: {:?}", name, r.why);
                    assert_eq!(hex(&r.header.root), root, "{}", name);
                    assert_eq!(hex(&r.digest.unwrap()), what.split(' ').next().unwrap(), "{}", name);
                }
                ("refuse", Err(e)) if what == "record hash mismatch" => assert_eq!(e.class, Class::Hash, "{}", name),
                ("refuse", Ok(r)) if what != "record hash mismatch" => {
                    assert_eq!(hex(&r.header.root), root, "{}", name);
                    assert_eq!(r.undecodable, vec![0], "{}: {:?}", name, r.why);
                    assert!(r.digest.is_none(), "{}", name);
                }
                (_, res) => panic!("{}: {:?}, the manifest says {} ({})", name, res.err(), verdict, what),
            }
        }
        assert_eq!(rows, 23);
    }

    /// Kind 1 (NV), a GG record read as kind 3, a blob of another kind, a changed model byte and G 0: the
    /// record is undecodable, the file opens.
    #[test]
    fn wnr4_bindings() {
        let orig = testdata("wnr4-gg-one.wex4");
        let io = u64le(&orig, 32) as usize;
        let nb = u32le(&orig, 48) as usize;
        let rat = io + 64 * nb;
        let (off, blob) = (u64le(&orig, rat) as usize, u32le(&orig, rat + 20) as usize - 1);
        let boff = u64le(&orig, io + 64 * blob + 8) as usize;
        type Edit = Box<dyn Fn(&mut Vec<u8>) -> (Option<usize>, Option<usize>)>;
        let cases: Vec<(&str, Edit)> = vec![
            (
                "WNR4 kind 1",
                Box::new(move |b| {
                    b[off + 12] = 1;
                    (Some(0), None)
                }),
            ),
            (
                "blob 0",
                Box::new(move |b| {
                    b[off + 12] = 3;
                    (Some(0), None)
                }),
            ),
            (
                "not a WNM4 model",
                Box::new(move |b| {
                    b[io + 64 * blob] = 4;
                    (None, None)
                }),
            ),
            (
                "blob 0",
                Box::new(move |b| {
                    b[boff + 100] ^= 1;
                    (None, Some(blob))
                }),
            ),
            (
                "G",
                Box::new(move |b| {
                    b[off + 14] = 0;
                    b[off + 15] = 0;
                    (Some(0), None)
                }),
            ),
        ];
        for (want, edit) in cases {
            let mut b = orig.clone();
            let (r, bl) = edit(&mut b);
            rehash(&mut b, r, bl);
            let rep = check_bytes(&b).unwrap_or_else(|e| panic!("{}: refused: {:?}", want, e));
            assert_eq!(rep.undecodable, vec![0], "{}", want);
            assert!(rep.why[0].contains(want), "{}: {:?}", want, rep.why);
        }
    }

    /// Every bit of the first 256 bytes of the codec-8 record of wnr4-gg-one and wnr4-i4-one, the hashes
    /// recomputed: never refused; accepted only inside the GG verbatim slices (another digest) or an I4 base
    /// that no scale reads (the same digest).
    #[test]
    fn wnr4_flips() {
        for name in ["wnr4-gg-one.wex4", "wnr4-i4-one.wex4"] {
            let orig = testdata(name);
            let d0 = check_bytes(&orig).unwrap().digest.unwrap();
            let io = u64le(&orig, 32) as usize;
            let rat = io + 64 * u32le(&orig, 48) as usize;
            let off = u64le(&orig, rat) as usize;
            let gg = orig[off + 12] == gg::KIND;
            let area = 64 + 4 * (u32le(&orig, off + 20) as usize + 1);
            let mut accepted = 0;
            for k in 0..256 * 8 {
                let mut b = orig.clone();
                b[off + k / 8] ^= 1 << (k % 8);
                rehash(&mut b, Some(0), None);
                let rep = check_bytes(&b).unwrap_or_else(|e| panic!("{}: bit {} refused: {:?}", name, k, e));
                if rep.undecodable.is_empty() {
                    accepted += 1;
                    let same = rep.digest.unwrap() == d0;
                    assert!(k / 8 >= area && same != gg, "{}: bit {} accepted, same digest {}", name, k, same);
                }
            }
            assert!(accepted > 0 || !gg, "{}: no verbatim flip accepted", name);
        }
    }
}
