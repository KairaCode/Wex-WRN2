// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ludovico Zanichelli
//! wex4r: a WEX4 reader and verifier written from docs/WEX4-SPEC.md, docs/WNR3-SPEC.md and
//! docs/WNR2-REFERENCE.md, and for codec 8 (WNR4 kinds GG and I4, draft) from docs/WNR4-SPEC.md, standard
//! library only.

// the code keeps to old Rust (by reading, nothing newer than 1.61): no `is_multiple_of`, no `div_ceil`
#![allow(clippy::manual_is_multiple_of, clippy::manual_div_ceil)]

mod hash;
mod rans;
mod sha256;
#[cfg(test)]
mod testenc;
mod wbp1;
mod wex4;
mod wnr3;
mod wnr4;

use sha256::{hex, sha256};
use std::panic;
use std::process::ExitCode;

const USAGE: &str = "usage:
  wex4r check [-v] [--no-decode] FILE...
                                      check each file in full; one line per file, and with -v the
                                      SHA-256 of every decoded record and tensor
  wex4r manifest [-v] MANIFEST.tsv    every row: file SHA-256, verdict, class, root and content digest
  wex4r dump FILE                     the header page and the index tables, as read
exit status: 0 every file accepted (check) or every row as the manifest says (manifest);
             1 a file refused or a row that disagrees; 2 usage; 3 a file whose records are not all decodable
             (a codec this reader does not know, or a codec-8 record that fails WEX4-SPEC 7.3)";

/// The verdict on one file: accepted (with the report), refused, or a fault of this reader.
enum Verdict {
    Accept(Box<wex4::Report>),
    Reject(wex4::Reject),
    Panic(String),
}

fn judge(path: &str, opt: &wex4::Options) -> Verdict {
    let result = panic::catch_unwind(|| wex4::check_file(path, opt));
    match result {
        Ok(Ok(r)) => Verdict::Accept(Box::new(r)),
        Ok(Err(e)) => Verdict::Reject(e),
        Err(p) => Verdict::Panic(
            p.downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default(),
        ),
    }
}

fn line(path: &str, v: &Verdict) -> String {
    match v {
        Verdict::Accept(r) => {
            let digest = r.digest.map(|d| hex(&d)).unwrap_or_else(|| "-".into());
            let mut s = format!(
                "accept {} root {} digest {} records {} ({} decoded, {} expert) tensors {} blobs {}",
                path,
                hex(&r.header.root),
                digest,
                r.records,
                r.decoded,
                r.expert_records,
                r.tensors,
                r.blobs
            );
            if !r.undecodable.is_empty() {
                // a codec this reader does not know, or a codec-8 record that failed a check of
                // WEX4-SPEC 7.3 (the record is undecodable, the file is not invalid)
                s = format!(
                    "undecodable {} root {}: records {:?} not decoded ({})",
                    path,
                    hex(&r.header.root),
                    r.undecodable,
                    r.why.join("; ")
                );
            }
            s
        }
        Verdict::Reject(e) => format!("reject {} {}: {}", path, e.class.name(), e.why),
        Verdict::Panic(m) => format!("FAULT {}: the reader panicked: {}", path, m),
    }
}

fn cmd_check(args: &[String]) -> ExitCode {
    let mut opt = wex4::Options { decode: true, list: false };
    let mut files = Vec::new();
    for a in args {
        match a.as_str() {
            "--no-decode" => opt.decode = false,
            "-v" => opt.list = true,
            _ => files.push(a.clone()),
        }
    }
    if files.is_empty() {
        eprintln!("{}", USAGE);
        return ExitCode::from(2);
    }
    let (mut refused, mut undecodable) = (false, false);
    for f in &files {
        let v = judge(f, &opt);
        println!("{}", line(f, &v));
        match &v {
            Verdict::Accept(r) => {
                undecodable |= !r.undecodable.is_empty();
                for l in &r.listing {
                    println!("{}", l);
                }
            }
            _ => refused = true,
        }
    }
    ExitCode::from(if refused {
        1
    } else if undecodable {
        3
    } else {
        0
    })
}

fn cmd_manifest(args: &[String]) -> ExitCode {
    let verbose = args.iter().any(|a| a == "-v");
    let path = match args.iter().find(|a| *a != "-v") {
        Some(p) => p.clone(),
        None => {
            eprintln!("{}", USAGE);
            return ExitCode::from(2);
        }
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: {}", path, e);
            return ExitCode::from(2);
        }
    };
    let dir = std::path::Path::new(&path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let opt = wex4::Options { decode: true, list: false };
    let (mut rows, mut bad, mut roots, mut classes, mut rejects, mut accepts) = (0, 0, 0, 0, 0, 0);
    let (mut digests, mut with_digest) = (0, 0);
    for raw in text.lines() {
        if raw.starts_with('#') || raw.trim().is_empty() {
            continue;
        }
        let col: Vec<&str> = raw.split('\t').collect();
        if col.len() < 5 {
            println!("BAD ROW {:?}", raw);
            bad += 1;
            continue;
        }
        rows += 1;
        let (file, verdict, class, root, sha) = (col[0], col[1], col[2], col[3], col[4]);
        let full = dir.join(file);
        let full = full.to_string_lossy().into_owned();
        match std::fs::read(&full) {
            Ok(bytes) if hex(&sha256(&bytes)) == sha => {}
            Ok(_) => {
                println!("DAMAGED {}: its SHA-256 is not the manifest's (a damaged copy, not a verdict)", file);
                bad += 1;
                continue;
            }
            Err(e) => {
                println!("MISSING {}: {}", file, e);
                bad += 1;
                continue;
            }
        }
        let v = judge(&full, &opt);
        let mut problems = Vec::new();
        if verdict == "accept" {
            accepts += 1;
        }
        match (&v, verdict) {
            (Verdict::Accept(r), "accept") => {
                if !r.undecodable.is_empty() {
                    problems.push("not every record decoded".to_string());
                }
                if hex(&r.header.root) == root {
                    roots += 1;
                } else {
                    problems.push(format!("root {} is not the manifest's {}", hex(&r.header.root), root));
                }
                // the 7th column, when the manifest has it: the content digest of WEX4-SPEC section 5
                if let Some(&content) = col.get(6).filter(|c| **c != "-") {
                    with_digest += 1;
                    match r.digest {
                        Some(d) if hex(&d) == content => digests += 1,
                        Some(d) => problems.push(format!("digest {} is not the manifest's {}", hex(&d), content)),
                        None => problems.push("no content digest".to_string()),
                    }
                }
            }
            (Verdict::Reject(e), "reject") => {
                rejects += 1;
                if e.class.name() == class {
                    classes += 1;
                } else {
                    problems.push(format!("class {} where the manifest says {}", e.class.name(), class));
                }
            }
            _ => problems.push(format!("the manifest says {}", verdict)),
        }
        if !problems.is_empty() || verbose {
            println!(
                "{}{}",
                line(file, &v),
                if problems.is_empty() { String::new() } else { format!("  <-- {}", problems.join("; ")) }
            );
        }
        let verdict_wrong = problems.iter().any(|p| !p.starts_with("class "));
        if verdict_wrong {
            bad += 1;
        }
    }
    println!(
        "rust reader: {} rows, {} disagree in verdict, root or content digest; root as the manifest's on {} of {} accepted rows; content digest as the manifest's on {} of {} rows that give one; class as the manifest's on {} of {} rejected rows",
        rows, bad, roots, accepts, digests, with_digest, classes, rejects
    );
    // a manifest with no rows checks nothing: refused, as the Go reader does
    if rows != 0 && bad == 0 && classes == rejects {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn main() -> ExitCode {
    // a panic is reported per file by `judge`; keep the default message off the output
    panic::set_hook(Box::new(|_| {}));
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("check") => cmd_check(&args[1..]),
        Some("manifest") => cmd_manifest(&args[1..]),
        Some("dump") if args.len() == 2 => match wex4::dump(&args[1]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                println!("reject {} {}: {}", args[1], e.class.name(), e.why);
                ExitCode::from(1)
            }
        },
        _ => {
            eprintln!("{}", USAGE);
            ExitCode::from(2)
        }
    }
}
