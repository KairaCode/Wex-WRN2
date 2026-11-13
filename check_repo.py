#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Repository self-check. Python 3 standard library only.

    python -I check_repo.py [--kaira <KAIRA checkout> [--ref master]] [--self-test]

One line per check, details indented below it; exit code 1 when links, mirror, licence or a strict
privacy pattern fail. Tracked files only: run it after `git add`.
  links    every relative Markdown link (and html href/src/srcset) in every tracked .md resolves to
           a tracked path, and its #anchor, when it points into a Markdown page, to a heading of that
           page (GitHub's slugs) or an <a id/name>. Links inside research/<run>/ folders are frozen:
           counted and listed, never a failure.
  mirror   every row of kaira-sync.tsv is `c/<path><TAB>src/colibri/c/<path>` of a tracked file and
           every tracked file under c/ has a row; line 1 of STATUS.md and the text above the first
           section of integration/README.md name the recorded sync (7 hex digits). With --kaira a
           KAIRA file under src/colibri/c whose name looks like format code and that has no row is
           listed (a warning: a new file to mirror?), and every row has the git blob id of
           <ref>:<KAIRA path> (the file bytes, no filter applied), and line 1 names a KAIRA commit
           in which the mirrored files are the ones of <ref>. Until KAIRA carries the licence header
           too, a file equal to KAIRA's once that header is removed counts as identical (and is counted).
  licence  LICENSE and NOTICE are tracked, and every source file outside research/ (c/ included)
           names its licence with `SPDX-License-Identifier:` in its first 5 lines.
  privacy  one line per pattern, with count and first files. A user-profile path (binary files
           included), an e-mail address that is not a noreply one or an access token fails the
           check. Warnings: the words scratchpad / session id, session and temp paths, assistant
           and session words, a private IPv4 address, the login name and the machine name of whoever
           runs the check.
"""
import argparse, hashlib, os, re, socket, subprocess, sys, urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))
FENCE = re.compile(r'^[ \t]*(```|~~~).*?^[ \t]*\1[^\n]*$', re.S | re.M)
CODE = re.compile(r'`[^`\n]*`')
# shortcut: a srcset with several candidates is read as one path, split it when a page uses one
LINK = re.compile(r'\]\(\s*(<[^>\n]+>|[^)\s]+)'                 # [text](target) and ![alt](target)
                  r'|^[ ]{0,3}\[[^\]\n]+\]:[ \t]*(\S+)'          # [label]: target
                  r'|\b(?:href|src|srcset)="([^"\n]+)"', re.M)   # html in Markdown
HEADING = re.compile(r'^[ ]{0,3}#{1,6}[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$', re.M)
ANCHOR_TAG = re.compile(r'<a\s+(?:id|name)="([^"]+)"', re.I)
WEX_NAME = re.compile(r'(?i)wex|nr3|wnr|neural_rans|palette')   # KAIRA files that look like format code
HOLDER = b'Ludovico Zanichelli'   # the copyright holder named in the licence header of every source file
OCTET = r'(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)'
# names read from the environment at run time, so that they are never written into this file
LOGIN = os.environ.get('USERNAME') or os.environ.get('USER') or '\0'
HOST = os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME') or socket.gethostname() or '\0'
STRICT = 3   # the first three patterns have no match in the repository: one match fails the check
PRIVACY = (
    ('profile path', re.compile(r'(?i)[\\/]users[\\/]+[\w.-]+')),
    ('e-mail', re.compile(r'(?i)(?<![\w.+-])(?![\w.+@-]*noreply)[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}')),
    ('token', re.compile(r'ghp_[A-Za-z0-9]{20,}|github_pat_\w{20,}|\bsk-[A-Za-z0-9_-]{20,}|\bAKIA[0-9A-Z]{16}\b'
                         r'|Bearer [A-Za-z0-9._~+/-]{20,}')),
    ('scratchpad / session id', re.compile(r'(?i)scratchpad|session[ _-]?id')),
    ('session path', re.compile(r'(?i)\.claude[\\/]\w|[\\/]worktrees[\\/]|\bappdata\b|%temp%|[\\/]temp[\\/]')),
    ('assistant / session words', re.compile(
        r'(?i)(?<![./\\])\b(?:claude|anthropic|opus|sonnet|haiku|codex|chatgpt)\b|\bmain[ -]session\b|\bsub-?agents?\b'
        r'|\bmemory notes?\b|MEMORY\.md|\btoolu_\w+|\bwf_[0-9a-f]{6,}')),
    # octets are capped at 255, so that a version such as 10.0.26300.1 is not an address
    ('private IPv4 address', re.compile(r'(?<![\d.])(?:10\.%s|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.%s\.%s(?!\.?\d)'
                                        % (OCTET, OCTET, OCTET))),
    ('user name', re.compile(r'(?i)\b%s\b' % re.escape(LOGIN))),
    ('machine name', re.compile(r'(?i)\b%s\b' % re.escape(HOST))),
)


def git(*args, cwd=ROOT):   # stderr is not captured: git's own message reaches the terminal
    return subprocess.run(('git',) + args, cwd=cwd, stdout=subprocess.PIPE, check=True).stdout.decode('utf-8')


def read(path):
    with open(os.path.join(ROOT, path), 'rb') as f:
        return f.read()


def link_targets(text):
    """(line, target) of every link in Markdown text; code blocks and code spans are left out."""
    def blank(m):
        return re.sub(r'[^\n]', ' ', m.group(0))
    text = CODE.sub(blank, FENCE.sub(blank, text))
    for m in LINK.finditer(text):
        yield text.count('\n', 0, m.start()) + 1, next(g for g in m.groups() if g).strip('<>')


def resolve(md, target):
    """Repo path a link target points to, or None for external and same-page links."""
    if re.match(r'[a-z][a-z0-9+.-]*:|#|//', target, re.I):
        return None
    path = urllib.parse.unquote(re.split(r'[#?]', target)[0])
    base = '' if path.startswith('/') else os.path.dirname(md)
    return os.path.normpath(os.path.join(base, path.lstrip('/'))).replace(os.sep, '/')


def slug(text):
    """GitHub's anchor of a heading: its text as rendered, lower case, punctuation other than - and _ removed,
    each space a -."""
    text = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', text)   # links and images keep their text
    text = re.sub(r'<[^>]+>|`', '', text).lower()
    return re.sub(r'[^\w\- ]', '', text).replace(' ', '-')


def anchors(text):
    """Every #anchor a Markdown page defines: heading slugs (a repeated one gets -1, -2, ...) and <a id/name>."""
    text = FENCE.sub(lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text)
    seen, found = {}, set(ANCHOR_TAG.findall(text))
    for m in HEADING.finditer(text):
        base = slug(m.group(1))
        found.add(f'{base}-{seen[base]}' if base in seen else base)
        seen[base] = seen.get(base, 0) + 1
    return found


def check_links(files):
    known = set(files) | {'.'}
    for f in files:
        while '/' in f:
            f = f.rsplit('/', 1)[0]
            known.add(f)
    pages = [f for f in files if f.lower().endswith('.md')]
    cache = {}

    def anchor_missing(md, target):   # an #anchor into a tracked Markdown page that the page does not define
        frag = urllib.parse.unquote(target.split('#', 1)[1]) if '#' in target else ''
        page = md if target.startswith('#') else resolve(md, target)
        if not frag or page not in files or not page.lower().endswith('.md'):
            return False
        if page not in cache:
            cache[page] = anchors(read(page).decode('utf-8', 'replace'))
        return frag not in cache[page]
    count, broken, frozen, frozen_broken = 0, [], 0, []
    for md in pages:
        in_run = md.startswith('research/') and md.count('/') >= 2
        for line, target in link_targets(read(md).decode('utf-8', 'replace')):
            path = resolve(md, target)
            bad = (path is not None and path not in known) or anchor_missing(md, target)
            if path is None and not target.startswith('#'):
                continue
            if in_run:
                frozen += 1
                frozen_broken += [f'{md}:{line} -> {target} (frozen, not a failure)'] * bad
            else:
                count += 1
                broken += [f'{md}:{line} -> {target}'] * bad
    print(f'links    {"FAIL" if broken else "ok"}: {len(pages)} .md files, {count} relative links and anchors, '
          f'{len(broken)} broken; research run folders (frozen): {frozen} links, {len(frozen_broken)} unresolved')
    print(''.join(f'    {b}\n' for b in broken + frozen_broken), end='')
    return not broken


def parse_sync(text):
    """(sha, date), rows and errors of kaira-sync.tsv."""
    lines = text.splitlines() or ['']
    head = re.fullmatch(r'# kaira_commit ([0-9a-f]{40}) (\d{4}-\d\d-\d\d)', lines[0])
    errors = [] if head else ['line 1 is not "# kaira_commit <40-hex sha> <YYYY-MM-DD>"']
    rows = [l.split('\t') for l in lines[1:] if l.strip() and not l.startswith('#')]
    errors += [f'row is not c/<path><TAB>src/colibri/c/<path>: {" | ".join(r)}' for r in rows
               if len(r) != 2 or not r[0].startswith('c/') or r[1] != 'src/colibri/' + r[0]]
    return head.groups() if head else ('?', '?'), [r for r in rows if len(r) == 2], errors


def without_licence(data):
    """The bytes with this repository's licence header removed: from the SPDX line (in the first 5 lines)
    to the line naming the copyright holder, 3 lines at most. Any other file comes back unchanged."""
    lines = data.split(b'\n')
    for i, line in enumerate(lines[:5]):
        if b'SPDX-License-Identifier:' in line:
            for j in range(i, min(i + 3, len(lines))):
                if HOLDER in lines[j]:
                    return b'\n'.join(lines[:i] + lines[j + 1:])
            break
    return data


def blob_id(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def sync_named(sha, status, integration):
    """Errors when STATUS.md line 1, or integration/README.md above its first section, does not name the sync."""
    short = sha[:7]
    errors = [] if short in status.split('\n', 1)[0] else [f'STATUS.md line 1 does not name the recorded sync {short}']
    if short not in integration.split('\n## ', 1)[0]:
        errors.append(f'integration/README.md does not name the recorded sync {short} above its first section')
    return errors


def check_mirror(files, kaira, ref):
    (sha, date), rows, errors = parse_sync(read('kaira-sync.tsv').decode('utf-8'))
    listed = [r[0] for r in rows]
    errors += [f'listed twice: {p}' for p in sorted(set(listed)) if listed.count(p) > 1]
    errors += [f'not tracked: {p}' for p in listed if p not in files]
    mirrored = [f for f in files if f.startswith('c/')]
    errors += [f'under c/ but not in kaira-sync.tsv: {f}' for f in mirrored if f not in listed]
    if sha != '?' and {'STATUS.md', 'integration/README.md'} <= set(files):
        errors += sync_named(sha, read('STATUS.md').decode('utf-8', 'replace'),
                             read('integration/README.md').decode('utf-8', 'replace'))
    note, warnings = f'recorded sync {sha[:12]} {date}; identity not checked (no --kaira)', []
    if kaira:
        def blobs(rev):
            tree = git('ls-tree', '-r', '-z', rev, cwd=kaira).split('\0')
            return {l.split('\t', 1)[1]: l.split()[2] for l in tree if l}
        theirs = blobs(ref)
        rows_of = {r[1] for r in rows}
        warnings = [f'not mirrored (warning: a new file to sync?): {t}' for t in sorted(theirs)
                    if t.startswith('src/colibri/c/') and WEX_NAME.search(t.rsplit('/', 1)[1]) and t not in rows_of]
        present = [r for r in rows if r[0] in files]
        ours = git('hash-object', '--no-filters', '--', *[r[0] for r in present]).split()
        # shortcut: KAIRA does not carry the licence header yet; drop `headed` once its files have it
        headed = [m for (m, t), blob in zip(present, ours)
                  if theirs.get(t) != blob and theirs.get(t) == blob_id(without_licence(read(m)))]
        differ = [(mine, their) for (mine, their), blob in zip(present, ours)
                  if theirs.get(their) != blob and mine not in headed]
        errors += [f'{"differs from" if t in theirs else "missing in"} KAIRA {ref}: {m} <- {t}' for m, t in differ]
        try:   # line 1 must name a commit in which the mirrored files are the ones of <ref>
            then = blobs(sha) if sha != '?' else theirs
            errors += [f'line 1 is stale: {t} changed in KAIRA after {sha[:12]}' for _, t in rows if then.get(t) != theirs.get(t)]
        except subprocess.CalledProcessError:
            errors.append(f'line 1: KAIRA has no commit {sha}')
        note = (f'{len(present) - len(differ)} identical to KAIRA {ref} {git("rev-parse", ref, cwd=kaira)[:12]} '
                f'(recorded sync {sha[:12]} {date}; {len(headed)} of them once the licence header KAIRA lacks '
                f'is removed)')
    print(f'mirror   {"FAIL" if errors else "ok"}: {len(rows)} rows in kaira-sync.tsv, {len(mirrored)} files under c/, '
          f'{note}, {len(errors)} errors, {len(warnings)} warnings')
    print(''.join(f'    {e}\n' for e in errors + warnings), end='')
    return not errors


SOURCE = ('.c', '.h', '.cu', '.cuh', '.inc', '.cpp', '.py', '.sh', '.mk', '.ps1', '.cmd', '.bat', '.def', '.cs', '.js')


def has_spdx(data):
    return b'SPDX-License-Identifier:' in b'\n'.join(data.split(b'\n')[:5])


def check_licence(files):
    errors = [f'not tracked: {f}' for f in ('LICENSE', 'NOTICE') if f not in files]
    code = [f for f in files if f.lower().endswith(SOURCE) and not f.startswith('research/')]
    errors += [f'no SPDX header in the first 5 lines: {f}' for f in code if not has_spdx(read(f))]
    print(f'licence  {"FAIL" if errors else "ok"}: {len(code)} source files outside research/, {len(errors)} errors')
    print(''.join(f'    {e}\n' for e in errors), end='')
    return not errors


def privacy_counts(data):
    """{pattern name: matches} in one file's bytes; a binary file is searched for profile paths only."""
    text = data.decode('utf-8', 'replace')
    return {name: len(p.findall(text)) for name, p in (PRIVACY[:1] if b'\0' in data[:8192] else PRIVACY)}


def check_privacy(files):
    hits = {name: [0, []] for name, _ in PRIVACY}
    for f in files:
        if f == os.path.basename(__file__):   # this file names the patterns and holds their test strings
            continue
        for name, n in privacy_counts(read(f)).items():
            if n:
                hits[name][0] += n
                hits[name][1].append(f)
    strict = [name for name, _ in PRIVACY[:STRICT]]
    for name, (n, where) in hits.items():
        where.sort(key=lambda f: f.startswith('research/'))   # living files first: a new one is among the five shown
        first = ', '.join(where[:5]) + (', ...' if len(where) > 5 else '')
        state = 'ok' if not n else 'FAIL' if name in strict else 'warn'
        print(f'privacy  {state}: {name}: {n} matches in {len(where)} files' + (f' ({first})' if n else ''))
    return not any(hits[name][0] for name in strict)


def self_test():
    md = ('a [x](../b%20c.md#s) `[no](code.md)` <img src="i.svg"> <source srcset="j.svg">\n```\n[no](fence.md)\n```\n'
          '[r]: d/e.md\n[web](https://x.y/z)')
    assert [t for _, t in link_targets(md)] == ['../b%20c.md#s', 'i.svg', 'j.svg', 'd/e.md', 'https://x.y/z']
    assert resolve('docs/p.md', '../b%20c.md#s') == 'b c.md' and resolve('p.md', 'https://x.y/z') is None
    profile, mail, token, words, spath, assistant = (p for _, p in PRIVACY[:6])
    for hit in ('C:\\Users\\someone\\x', 'C:\\\\Users\\\\someone', 'D:/Users/someone/x', '/c/Users/someone/x', '\\Users\\someone'):
        assert profile.search(hit), hit
    assert not profile.search('<HOME>\\x') and mail.search('a.b@example.org')
    assert not mail.search('1+bot@users.noreply.github.com') and not mail.search('noreply@example.org')
    assert words.search('Session-ID') and words.search('ScratchPad')
    assert token.search('ghp_' + 'a1' * 18) and token.search('Bearer ' + 'x' * 24) and not token.search('task-' + 'a' * 24)
    for hit in ('<KAIRA>\\.claude\\worktrees\\x', 'C:\\Temp\\x', '%TEMP%\\x', 'AppData\\Local'):
        assert spath.search(hit), hit
    assert not spath.search('temperature 0.6') and not spath.search('prefix-store/template')
    for hit in ('asked Claude to', 'the main session', 'two sub-agents', 'a memory note', 'wf_2af1fd28-820'):
        assert assistant.search(hit), hit
    assert not assistant.search('x/.claude/y') and not assistant.search('magnum opuses') and not assistant.search('wf_run')
    binary, text = b'\0\0D:\\Users\\someone\\x a.b@example.org', b'D:\\Users\\someone\\x a.b@example.org'
    assert privacy_counts(binary) == {'profile path': 1}, 'binary files: profile paths only'
    assert [privacy_counts(text)[k] for k in ('profile path', 'e-mail', 'token')] == [1, 1, 0]
    sha = '0123456789abcdef' * 2 + '01234567'
    good = f'# kaira_commit {sha} 2026-11-11\nc/a.h\tsrc/colibri/c/a.h\n\n# note\nc/t/b.c\tsrc/colibri/c/t/b.c\n'
    assert parse_sync(good) == ((sha, '2026-11-11'), [['c/a.h', 'src/colibri/c/a.h'], ['c/t/b.c', 'src/colibri/c/t/b.c']], [])
    assert len(parse_sync('')[2]) == 1 and len(parse_sync('# kaira_commit notasha yesterday\n')[2]) == 1
    bad = f'# kaira_commit {sha} 2026-11-11\nscripts/a.h\tsrc/colibri/c/a.h\nc/a.h\tsrc/colibri/c/b.h\nc/a.h\n'
    assert len(parse_sync(bad)[2]) == 3 and len(parse_sync(bad)[1]) == 2
    head = b'#!/bin/sh\r\n# SPDX-License-Identifier: Apache-2.0\r\n# Copyright 2026 ' + HOLDER + b'\r\nx\r\n'
    assert without_licence(head) == b'#!/bin/sh\r\nx\r\n', 'licence header removed, line ends kept'
    assert without_licence(b'x\n# SPDX-License-Identifier: X\ny\nz\n') == b'x\n# SPDX-License-Identifier: X\ny\nz\n'
    assert blob_id(b'') == 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
    assert has_spdx(b'#!/bin/sh\n# SPDX-License-Identifier: Apache-2.0\n') and not has_spdx(b'x\n' * 5 + b'# SPDX-License-Identifier: X')
    assert slug('13.3 `app\\kaira.json`') == '133-appkairajson' and slug('The [`c/`](c/) mirror') == 'the-c-mirror'
    page = '# A\n## 5. Index (v4)\n```\n# not a heading\n```\n## 5. Index (v4)\n<a id="here"></a>\n## Done ##\n'
    assert anchors(page) == {'a', '5-index-v4', '5-index-v4-1', 'here', 'done'}
    lan = PRIVACY[6][1]
    for hit in ('host 192.168.1.20', 'at 10.0.0.1:80', '172.16.0.1'):
        assert lan.search(hit), hit
    for miss in ('Windows 10.0.26300.1234', '172.15.0.1', 'v1.10.0.1.5', '10.1.2.3.4', '127.0.0.1'):
        assert not lan.search(miss), miss
    status, integ = 'Last verified: 2026-11-13, KAIRA master 0123456789ab\n', 'x\nrecords master 0123456 here\n## T\n'
    assert sync_named(sha, status, integ) == [] and len(sync_named('f' * 40, status, integ)) == 2
    assert len(sync_named(sha, status, 'x\n## T\nmaster 0123456\n')) == 1, 'the sync must be named above the first section'
    print('self-test ok')
    return 0


def main(argv):
    ap = argparse.ArgumentParser(usage='python -I check_repo.py [--kaira <KAIRA checkout> [--ref master]] [--self-test]',
                                 allow_abbrev=False)
    ap.add_argument('--kaira')
    ap.add_argument('--ref', default='master')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        files = sorted(f for f in git('ls-files', '-z').split('\0') if f and os.path.isfile(os.path.join(ROOT, f)))
        ok = [check_links(files), check_mirror(files, args.kaira, args.ref), check_licence(files), check_privacy(files)]
    except (subprocess.CalledProcessError, OSError) as e:
        sys.exit(f'check_repo.py: {e}')
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
