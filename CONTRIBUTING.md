# Contributing

Seven rules, after a section for people new to the repository. The rules exist because the engine-side code under `c/` is also carried in other trees that must stay in step with it, and because the rest of this repository is read by people who were not there when the measurements were made.

## 0. Outside contributors

Thank you for looking. Most of this repository is specifications and tests, so the most useful contributions are a reader or writer in another language that passes the conformance vectors, a bug found by them, and a sentence of a specification that can be read two ways.

- **Open an issue first** for any change to a format (a field, a flag, a codec, a rule of validation): it changes what other readers must do. A bug fix or a documentation fix can go straight to a pull request.
- **Set up.** Windows: [MSYS2](https://www.msys2.org/) with the UCRT64 packages `mingw-w64-ucrt-x86_64-gcc`, `mingw-w64-ucrt-x86_64-libgomp` and `mingw-w64-ucrt-x86_64-python` (or `gcc` on `PATH` from Git Bash), and `git`. Linux: `gcc`, `python3`, `git`, and `export PYTHON=python3`. An x86-64 CPU with AVX2. No GPU, no model and no Python package outside the standard library.
- **Check.** `bash scripts/check_all.sh` builds and runs every CPU test and the repository check. It must end with `all checks passed`.
- **Style.** C99 that builds without warnings under `-Wall -Wextra`; Python 3 standard library only, run with `python -I`; plain English in the pages, with numbers stated as rule 5 says. Add no dependency.
- **A format change** brings, in the same pull request: the specification diff, new or changed files in [wex4/vectors/](wex4/vectors/README.md) with their `MANIFEST.tsv` rows, matching changes in both WEX4 readers (`c/wex4.h` and `wex4/wex4.py`), and a `CHANGELOG.md` entry.
- **`c/` is engine code** (rule 1): a pull request may change it like any other folder; run the CPU tests, and say in the description what engine behaviour the change touches.
- **Sign-off**: rule 7.

## 1. `c/` is this repository's engine-side code

`c/` holds the engine-side WEX code: format headers, decoders, the expert store, the tools and their tests, in the colibri `c/` layout. Pull requests may change it normally. The same files are also carried in the maintainer's application tree (under `src/colibri/c/`) and are being integrated upstream into [colibri](https://github.com/JustVugg/colibri) (`dev` branch); the maintainer keeps those copies in step with this one. Nothing else goes into `c/`: code that is only used here belongs in `scripts/` or `wex4/` (`archive/` takes no new code). Every file under `c/` carries the licence header of rule 7 and has a row in `sync-map.tsv`; a new file gets a row `c/<path><TAB>src/colibri/c/<path>` and a `git add` (the check reads tracked files only). A deleted file loses its row; the check then lists the Markdown links to it, and `grep -rn <file name> scripts wex4 archive integration` finds the scripts that build it.

**Maintainer only: comparing `c/` with another tree.** `python -I check_repo.py --source <TREE> [--ref master]` compares every row of `sync-map.tsv` with `<ref>:src/colibri/c/<path>` of a git checkout `<TREE>` (the committed bytes, after removing this repository's licence header when the other tree lacks it), lists format files there that have no row, and checks that line 1 of `sync-map.tsv`, `# source_commit <full sha> <YYYY-MM-DD>` (single spaces, the date of that commit), names a commit of `<TREE>` whose files are the ones of `<ref>`. `git -C <TREE> show -s --format='# source_commit %H %cs' master` prints that line. When copying a file from another tree, take the committed file, not the working tree: in Git Bash, `git -C <TREE> show master:src/colibri/c/<path> > c/<path>` (not in PowerShell, whose `>` re-encodes the bytes), then put back the licence header the file had. `.gitattributes` (`* -text`) keeps the bytes as they are; leave it alone. After a sync, run `bash scripts/check_all.sh`, update the counts in `README.md` and `STATUS.md` when the number of files under `c/` changed, and add a `CHANGELOG.md` entry for a change of a format, a flag or a tool.

Two paths are named in comments of files under `c/` and must keep resolving: `docs/NEXT-FORMAT-SPEC.md` (a stub; do not delete it) and `research/wnr3-5090-20261107/g0/t2_g0.py`.

## 2. Privacy

Nothing tracked may contain a real name (except the names of copyright holders and authors in licence notices, attribution notices and citations, which must stay), a personal e-mail address, the Windows user name, a quoted chat message, the id of a work session, a scratch or temp path, or a list of the programs running on the machine. That covers prose, scripts and raw logs alike. Write paths with placeholders:

| Placeholder | Stands for |
|---|---|
| `<TREE>` | a checkout of the maintainer's application tree (older logs write `<KAIRA>`) |
| `<worktree>` | a git worktree of that tree (older logs write `<KAIRA-WT>`) |
| `<Wex-WRN2>` | a checkout of this repository |
| `<SCRATCH>`, `<local scratch dir>` | a scratch folder named in a raw log |
| `<HOME>` | the user profile |
| `<TMP>` | any temporary folder |
| `<RESEARCH-PC>` | the working folder on the research PC |

Drive letters of the lab machine (`U:`, `C:`) are fine. In a shell script quote the placeholder (`'<TREE>'/lab`): bare, the shell reads it as a redirection. Commit with a noreply address. Before committing, and after `git add` (the check reads tracked files only), run `python -I check_repo.py`. A `privacy` line that says FAIL (a user-profile path, also inside a binary file; an e-mail address that is not a noreply one; an access token) fails the check and must be fixed. The other `privacy` lines are warnings and must not grow. They look for scratch-folder words and identifiers of work sessions; session and temp paths; assistant and session words; a private IPv4 address (10.x, 172.16-31.x, 192.168.x); the login name and the machine name of whoever runs the check, both read from the environment at run time. Re-measured on 2026-11-13 on this tree, they read, in the order printed, 24 matches in 15 files, 7 in 7, 58 in 26, 0, 0 and 0. All of those are in research run folders, except one line of `wex4/review/REVIEW.md` and the trailer line of the three `engine-*` patches in `integration/` (byte copies of engine commits). The check does not look for real names, quoted messages or lists of programs: read your diff for them.

## 3. What does not go into the repository

Real containers and checkpoints never enter the repository (`*.wex` is git-ignored), nor do model weights of any kind, nor third-party code or data whose licence is not compatible with Apache-2.0. What is tracked is small: the 59 KB WNM2 models, the tiny fixtures, the conformance vectors of `wex4/vectors/`, and the trigger files of the WEX4 security review. A file computed from real weights (a model table, a fixture, a sample) goes in only with a row in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) that names the checkpoint and its licence.

Upstream integration into colibri is done by the maintainer, not from pull requests here; `integration/upstream-561966ec-levelA.patch` is for a user's own colibri tree.

## 4. Adding a research run

1. One folder: `research/<topic>-YYYYMMDD/`, lower case with hyphens, the date the run started.
2. A `README.md` in it with: the **question**; the **setup** (machine, model, container form and drives, RAM budget, engine build with its commit, the command); the **result table**; the **limits** (what was not run, not repeated, not verified); the list of **raw files**.
3. Scripts and raw output go in as they are, with paths shortened to the placeholders of rule 2. No containers, no executables. `out/` folders and files such as `*.npz`, `*.dll`, `*.o` are git-ignored everywhere: `git status --ignored research/<run>` shows what `git add` skipped, and `git add -f` takes what belongs.
4. Add a row at the top of the table in [research/README.md](research/README.md), with the verdict copied from the run's README.
5. Add the result to [docs/BENCHMARKS.md](docs/BENCHMARKS.md) when another page quotes it, and update [STATUS.md](STATUS.md) only if what runs in production changed.

A committed run folder is frozen. A later correction is a dated note at the top of its README, or a new run; the numbers are not edited. The folders older than 2026-11-11 predate the README rule: their row in `research/README.md` gives their question, verdict and entry file instead. `research/wex4-qwen38/` has no date in its name; its row gives the date.

## 5. How results are stated

- A number carries its conditions: model, container form, drives, RAM budget, prompts x tokens, build.
- Say how many runs stand behind it: "one pair", "ABBA, 4 pairs", "one run each", "not repeated". Give an interval only when pairs stand behind it.
- Keep identity and speed apart: "tokens identical" or "logits bit-identical" is one statement, tok/s another.
- Say what was not checked, in the same place as the result.
- Keep the outcomes that went against the idea: "stays off", "not adopted", "not run".
- An estimate or a model's output is labelled as one, never written as a measurement.
- A number quoted on another page names the file it comes from. If it cannot be traced to a file, write that it is not verified.

## 6. Where a page goes

| Kind | Where | Rule |
|---|---|---|
| Living page | `README.md`, `STATUS.md`, `CHANGELOG.md`, `SECURITY.md`, the folder READMEs, `docs/*.md` (index, architecture, glossary, benchmarks, integration guide, WNR2 reference, WNR3 and WEX4 specifications) | says what is true today; edited whenever that changes. What runs in production is decided in `STATUS.md`; another page may repeat it only as a dated summary with a link to `STATUS.md`, and is corrected from it |
| Dated report | `docs/reports/<topic>-YYYYMMDD.md` (`.it.md` when Italian) | records one moment; not edited afterwards, except for a note on top that names what superseded it |
| Research run | `research/<topic>-YYYYMMDD/` | rule 4 |

New pages are written in English. The Italian reports stay in Italian. New report names are lower case; the upper-case names given before 2026-11-13 (`WNR3-VERIFY.md`, `WNR3-PRODUCTION-20261110.md`, `QWEN38-MXFP4-20261111.md`) stay as they are, because frozen pages link them.

## 7. Licence and sign-off

**Inbound = outbound.** Every contribution is licensed under the Apache License 2.0, the licence of the repository (section 5 of [LICENSE](LICENSE)). There is no separate contributor agreement.

**Sign-off.** Every commit of a contribution from outside (anyone but the maintainer, who holds the copyright of their own commits) ends with a line

```text
Signed-off-by: Your Name <your e-mail>
```

(`git commit -s` writes it). With it you certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/): you wrote the change, or you have the right to submit it under this licence. A GitHub noreply address is accepted, and keeps your e-mail out of the history.

**Headers.** A new source file starts with two lines, in the comment syntax of its language:

```text
SPDX-License-Identifier: Apache-2.0
Copyright <year> <copyright holder>
```

A file derived from someone else's code keeps the notices it came with, names its source, and says who changed it and when (Apache-2.0 section 4(b)); a file derived from colibri is also added to the list in [NOTICE](NOTICE). Code or data under another licence goes in only when that licence is compatible with Apache-2.0, with a section in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) that holds its licence text. Under `research/`, frozen run folders are covered by the README's licence paragraph and need no headers, except files derived from third-party code.
