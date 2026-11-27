#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli
"""Repository self-check. Python 3 standard library only.

    python -I check_repo.py [--source <TREE> [--ref master]] [--self-test] [--line-id FILE N] [--commits RANGE]

One line per check, details indented below it; exit code 1 when links, mirror, licence or a strict
privacy pattern fail. Tracked files only: run it after `git add`.
  commits  with --commits RANGE (a git revision range, `origin/main..HEAD`), in place of the checks below:
           every commit of the range has a GitHub noreply address as author and as committer
           (`...@users.noreply.github.com`, or GitHub's own `noreply@github.com` on web merges and edits)
           and no other address in its message (trailers included). A commit that fails is listed by
           its short id and the fields at fault, never by the address. CI runs it on the commits of
           every pull request.
  links    every relative Markdown link (and html href/src/srcset) in every tracked .md resolves to
           a tracked path, and its #anchor, when it points into a Markdown page, to a heading of that
           page (GitHub's slugs) or an <a id/name>. Links inside research/<run>/ folders are frozen:
           counted and listed, never a failure.
  mirror   every row of sync-map.tsv is `c/<path><TAB>src/colibri/c/<path>` of a tracked file and
           every tracked file under c/ has a row. With --source TREE (maintainer only: a git checkout
           of an application tree that carries the same engine files under src/colibri/c) a file
           there whose name looks like format code and that has no row is listed (a warning: a new
           file to sync?), every row has the git blob id of <ref>:<tree path> (the file bytes, no
           filter applied), and line 1 names a commit of that tree in which the synced files are the
           ones of <ref>. A file equal to the tree's once this repository's licence header is
           removed counts as identical (and is counted).
  licence  LICENSE and NOTICE are tracked, and every source file outside research/ (c/ included)
           names its licence with `SPDX-License-Identifier:` in its first 5 lines.
  privacy  one line per pattern, with count and first files. A user-profile path (binary files
           included), an e-mail address that is not a noreply one or an access token fails the
           check. Warnings: the words scratchpad / session id, session and temp paths, assistant
           and session words, a private IPv4 address, the login name and the machine name of whoever
           runs the check. The warning patterns also search the tracked path of every file.
           ALLOWED lists the few warning matches that stay, as (file, pattern, line id): the matches
           of that pattern on the first line of that file whose line_id() is that id are not counted.
           The id is a hash of the whole line (line end left out), so that this file does not repeat
           the words it looks for: a line changed in any way counts again. Every other match, in that
           file or any other, counts; a strict pattern cannot be listed; an entry that leaves out
           nothing is reported (update or remove it). `--line-id FILE N` prints the id of line N.
           The entries: the trailer line of the three engine-* patches (byte copies of engine
           commits), one line of a third-party model card, the dump path in the code of a frozen run.
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
WEX_NAME = re.compile(r'(?i)wex|nr3|wnr|neural_rans|palette')   # source-tree files that look like format code
HOLDER = b'Ludovico Zanichelli'   # the copyright holder named in the licence header of every source file
OCTET = r'(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)'
# names read from the environment at run time, so that they are never written into this file
LOGIN = os.environ.get('USERNAME') or os.environ.get('USER') or '\0'
HOST = os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME') or socket.gethostname() or '\0'
STRICT = 3   # the first three patterns have no match in the repository: one match fails the check
PRIVACY = (
    ('profile path', re.compile(r'(?i)[\\/](?:users|home)[\\/]+[\w.-]+')),
    ('e-mail', re.compile(r'(?i)(?<![\w.+-])(?![\w.+@-]*noreply)[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}')),
    ('token', re.compile(r'ghp_[A-Za-z0-9]{20,}|github_pat_\w{20,}|\bsk-[A-Za-z0-9_-]{20,}|\bAKIA[0-9A-Z]{16}\b'
                         r'|Bearer [A-Za-z0-9._~+/-]{20,}|\bhf_[A-Za-z0-9]{30,}\b')),
    ('scratchpad / session id', re.compile(r'(?i)scratchpad|session[ _-]?id')),
    ('session path', re.compile(r'(?i)\.claude[\\/]\w|[\\/]worktrees[\\/]|\bappdata\b|%temp%|[\\/]temp[\\/]'
                                r'|(?<![\w.~})-])/tmp/')),
    ('assistant / session words', re.compile(
        r'(?i)(?<![./\\])\b(?:claude|anthropic|opus|sonnet|haiku|codex|chatgpt)\b|\bmain[ -]session\b'
        r'|\bsessione principale\b|\bsub-?agents?\b'
        r'|\bmemory notes?\b|MEMORY\.md|\btoolu_\w+|\bwf_[0-9a-f]{6,}')),
    # octets are capped at 255, so that a version such as 10.0.26300.1 is not an address
    ('private IPv4 address', re.compile(r'(?<![\d.])(?:10\.%s|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.%s\.%s(?!\.?\d)'
                                        % (OCTET, OCTET, OCTET))),
    ('user name', re.compile(r'(?i)\b%s\b' % re.escape(LOGIN))),
    ('machine name', re.compile(r'(?i)\b%s\b' % re.escape(HOST))),
)
STRICT_NAMES = [name for name, _ in PRIVACY[:STRICT]]
# the only addresses a commit may carry as author or committer; the second is GitHub's own committer (web merges and edits)
GITHUB_NOREPLY = re.compile(r'(?i)[\w.+\[\]-]+@users\.noreply\.github\.com|noreply@github\.com')
# Warning matches that stay (header): (file, pattern, line_id() of the whole line).
ALLOWED = (
    # byte copies of engine commits (`git format-patch`, see integration/README.md): the commit's
    # `Co-Authored-By:` trailer line (line 12, 14 and 17)
    ('integration/engine-q38-wex4-loader-de17198.patch', 'assistant / session words', 'ce8572cfc0a64238'),
    ('integration/engine-wex4-first-read-4ba5568.patch', 'assistant / session words', 'ce8572cfc0a64238'),
    ('integration/engine-wex4-loader-75932c1.patch', 'assistant / session words', 'ce8572cfc0a64238'),
    # a third-party model card copied unchanged (THIRD_PARTY_NOTICES.md): the line on its calibration data,
    # `1. **Calibration.** 552 rows ...` (line 141)
    ('research/wnr3-20261107/team3/other-tensors/data/code/README.md.txt', 'assistant / session words',
     '118063280aa738ff'),
    # code of a frozen run: the folder its logit dumps went to, `OUT=$HERE/runs/$STAMP DUMP=...` (line 80)
    ('research/wnr2-fastbench-20261109/short-abba/short_abba.sh', 'session path', '6e050a121b164acc'),
)


def fold(s):
    """Lower case, with the three non-ASCII letters that (?i) matches against i and s written as those."""
    return s.replace('İ', 'i').lower().replace('ı', 'i').replace('ſ', 's')


def line_id(line):
    """The id of a line in ALLOWED: the first 16 hex digits of the SHA-256 of its text, line end left out."""
    return hashlib.sha256(line.rstrip('\r\n').encode('utf-8')).hexdigest()[:16]


# Lower-case literals of each pattern: every match holds one of them, so a file that holds none is not
# searched. The nine searches of every file took 14 of the check's 15 seconds; the counts do not change.
LITERALS = {
    'profile path': ('users', 'home'),
    'e-mail': ('@',),
    'token': ('ghp_', 'github_pat_', 'sk-', 'akia', 'bearer ', 'hf_'),
    'scratchpad / session id': ('scratchpad', 'session'),
    'session path': ('.claude', 'worktrees', 'appdata', 'temp', '/tmp/'),
    'assistant / session words': ('claude', 'anthropic', 'opus', 'sonnet', 'haiku', 'codex', 'chatgpt', 'session',
                                  'agent', 'memory', 'toolu_', 'wf_'),
    'private IPv4 address': ('10.', '192.168', '172.'),
    'user name': (fold(LOGIN) if LOGIN.isascii() else '',),
    'machine name': (fold(HOST) if HOST.isascii() else '',),
}


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
    """(sha, date), rows and errors of sync-map.tsv."""
    lines = text.splitlines() or ['']
    head = re.fullmatch(r'# source_commit ([0-9a-f]{40}) (\d{4}-\d\d-\d\d)', lines[0])
    errors = [] if head else ['line 1 is not "# source_commit <40-hex sha> <YYYY-MM-DD>"']
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


def check_mirror(files, source, ref):
    (sha, date), rows, errors = parse_sync(read('sync-map.tsv').decode('utf-8'))
    listed = [r[0] for r in rows]
    errors += [f'listed twice: {p}' for p in sorted(set(listed)) if listed.count(p) > 1]
    errors += [f'not tracked: {p}' for p in listed if p not in files]
    mirrored = [f for f in files if f.startswith('c/')]
    errors += [f'under c/ but not in sync-map.tsv: {f}' for f in mirrored if f not in listed]
    note, warnings = f'recorded sync {sha[:12]} {date}; identity not checked (no --source)', []
    if source:
        def blobs(rev):
            tree = git('ls-tree', '-r', '-z', rev, cwd=source).split('\0')
            return {l.split('\t', 1)[1]: l.split()[2] for l in tree if l}
        theirs = blobs(ref)
        rows_of = {r[1] for r in rows}
        warnings = [f'not mirrored (warning: a new file to sync?): {t}' for t in sorted(theirs)
                    if t.startswith('src/colibri/c/') and WEX_NAME.search(t.rsplit('/', 1)[1]) and t not in rows_of]
        present = [r for r in rows if r[0] in files]
        ours = git('hash-object', '--no-filters', '--', *[r[0] for r in present]).split()
        # shortcut: the source tree does not carry the licence header yet; drop `headed` once its files have it
        headed = [m for (m, t), blob in zip(present, ours)
                  if theirs.get(t) != blob and theirs.get(t) == blob_id(without_licence(read(m)))]
        differ = [(mine, their) for (mine, their), blob in zip(present, ours)
                  if theirs.get(their) != blob and mine not in headed]
        errors += [f'{"differs from" if t in theirs else "missing in"} source {ref}: {m} <- {t}' for m, t in differ]
        try:   # line 1 must name a commit in which the synced files are the ones of <ref>
            then = blobs(sha) if sha != '?' else theirs
            errors += [f'line 1 is stale: {t} changed in the source tree after {sha[:12]}' for _, t in rows if then.get(t) != theirs.get(t)]
        except subprocess.CalledProcessError:
            errors.append(f'line 1: the source tree has no commit {sha}')
        note = (f'{len(present) - len(differ)} identical to source {ref} {git("rev-parse", ref, cwd=source)[:12]} '
                f'(recorded sync {sha[:12]} {date}; {len(headed)} of them once the licence header the source lacks '
                f'is removed)')
    print(f'mirror   {"FAIL" if errors else "ok"}: {len(rows)} rows in sync-map.tsv, {len(mirrored)} files under c/, '
          f'{note}, {len(errors)} errors, {len(warnings)} warnings')
    print(''.join(f'    {e}\n' for e in errors + warnings), end='')
    return not errors


SOURCE = ('.c', '.h', '.cu', '.cuh', '.inc', '.cpp', '.py', '.sh', '.mk', '.ps1', '.cmd', '.bat', '.def', '.cs', '.js', '.go', 'go.mod', '.comp', '.rs', '.toml')


def has_spdx(data):
    return b'SPDX-License-Identifier:' in b'\n'.join(data.split(b'\n')[:5])


def check_licence(files):
    errors = [f'not tracked: {f}' for f in ('LICENSE', 'NOTICE') if f not in files]
    code = [f for f in files if f.lower().endswith(SOURCE) and not f.startswith('research/')]
    errors += [f'no SPDX header in the first 5 lines: {f}' for f in code if not has_spdx(read(f))]
    print(f'licence  {"FAIL" if errors else "ok"}: {len(code)} source files outside research/, {len(errors)} errors')
    print(''.join(f'    {e}\n' for e in errors), end='')
    return not errors


def privacy_counts(data, path='', allowed=ALLOWED):
    """{pattern name: matches} in one file's bytes, plus the warning matches in its path (a tracked path is
    public too); a binary file's bytes are searched for profile paths only. The entries of `allowed` for
    `path` leave out the matches of their line (header)."""
    text = data.decode('utf-8', 'replace')
    low = fold(text)
    mine = [(name, lid) for f, name, lid in allowed if f == path and name not in STRICT_NAMES]
    first, a = {}, 0   # line id -> (start, end) of the first line that has it
    for line in (text.split('\n') if mine else ()):
        first.setdefault(line_id(line), (a, a + len(line)))
        a += len(line) + 1
    spans = [(name,) + first[lid] for name, lid in mine if lid in first]
    counts = {}
    for name, p in (PRIVACY[:1] if b'\0' in data[:8192] else PRIVACY):
        starts = [m.start() for m in p.finditer(text)] if any(s in low for s in LITERALS[name]) else []
        for _, a, b in (sp for sp in spans if sp[0] == name):
            starts = [s for s in starts if not a <= s < b]
        counts[name] = len(starts)
    for name, p in PRIVACY[STRICT:]:
        n = len(p.findall(path))
        if n:
            counts[name] = counts.get(name, 0) + n
    return counts


def check_privacy(files):
    hits = {name: [0, []] for name, _ in PRIVACY}
    skipped, unused = dict.fromkeys(hits, 0), []
    for f in files:
        if f == os.path.basename(__file__):   # this file names the patterns and holds their test strings
            continue
        data = read(f)
        counts = privacy_counts(data, f)
        mine = [e for e in ALLOWED if e[0] == f]
        if mine:
            full = privacy_counts(data, f, ())
            unused += [f'{f} ({e[1]})' for e in mine if privacy_counts(data, f, (e,)).get(e[1]) == full.get(e[1])]
            for name in full:
                skipped[name] += full[name] - counts[name]
        for name, n in counts.items():
            if n:
                hits[name][0] += n
                hits[name][1].append(f)
    unused += [f'{f} ({name}): not tracked' for f, name, _ in ALLOWED if f not in files]
    for name, (n, where) in hits.items():
        where.sort(key=lambda f: f.startswith('research/'))   # living files first: a new one is among the five shown
        first = ', '.join(where[:5]) + (', ...' if len(where) > 5 else '')
        state = 'ok' if not n else 'FAIL' if name in STRICT_NAMES else 'warn'
        print(f'privacy  {state}: {name}: {n} matches in {len(where)} files' + (f' ({first})' if n else '')
              + (f'; {skipped[name]} listed in ALLOWED not counted' if skipped[name] else ''))
    if unused:
        print(f'privacy  warn: {len(unused)} ALLOWED entries leave out nothing (update or remove them):')
        print(''.join(f'    {u}\n' for u in unused), end='')
    return not any(hits[name][0] for name in STRICT_NAMES)


def commit_mail_errors(author, committer, message):
    """The fields of one commit that expose an address: author / committer when its address is not a GitHub
    noreply one, message when it holds an address the e-mail pattern counts (a trailer, for one)."""
    return ([name for name, mail in (('author', author), ('committer', committer)) if not GITHUB_NOREPLY.fullmatch(mail)]
            + ['message'] * bool(PRIVACY[1][1].search(message)))


def check_commits(rev_range):
    """Every commit of the range. The address itself is never printed: the output of CI is public too."""
    try:
        log = git('log', '-z', '--format=%h%x1f%ae%x1f%ce%x1f%B', rev_range, '--')
    except subprocess.CalledProcessError:
        print(f'commits  FAIL: git cannot list {rev_range} (a shallow clone? fetch the full history)')
        return False
    commits = [c.split('\x1f', 3) for c in log.split('\0') if c]
    bad = [(c[0], commit_mail_errors(*c[1:])) for c in commits]
    bad = [f'{sha}: {", ".join(fields)}' for sha, fields in bad if fields]
    print(f'commits  {"FAIL" if bad else "ok"}: {len(commits)} commits in {rev_range}, {len(bad)} with an author or '
          f'committer address that is not a GitHub noreply one, or another address in the message')
    print(''.join(f'    {b}\n' for b in bad), end='')
    return not bad


def self_test():
    md = ('a [x](../b%20c.md#s) `[no](code.md)` <img src="i.svg"> <source srcset="j.svg">\n```\n[no](fence.md)\n```\n'
          '[r]: d/e.md\n[web](https://x.y/z)')
    assert [t for _, t in link_targets(md)] == ['../b%20c.md#s', 'i.svg', 'j.svg', 'd/e.md', 'https://x.y/z']
    assert resolve('docs/p.md', '../b%20c.md#s') == 'b c.md' and resolve('p.md', 'https://x.y/z') is None
    profile, mail, token, words, spath, assistant = (p for _, p in PRIVACY[:6])
    for hit in ('C:\\Users\\someone\\x', 'C:\\\\Users\\\\someone', 'D:/Users/someone/x', '/c/Users/someone/x', '\\Users\\someone',
                '/home/someone/x'):
        assert profile.search(hit), hit
    assert not profile.search('<HOME>\\x') and mail.search('a.b@example.org')
    assert not mail.search('1+bot@users.noreply.github.com') and not mail.search('noreply@example.org')
    assert words.search('Session-ID') and words.search('ScratchPad')
    assert token.search('ghp_' + 'a1' * 18) and token.search('Bearer ' + 'x' * 24) and not token.search('task-' + 'a' * 24)
    assert privacy_counts(b'hf_' + b'a1' * 17)['token'] == 1 and not token.search('hf_hub_download')
    for hit in ('<TREE>\\.claude\\worktrees\\x', 'C:\\Temp\\x', '%TEMP%\\x', 'AppData\\Local', 'cd /tmp/x', 'C:/tmp/x'):
        assert spath.search(hit), hit
    assert not spath.search('temperature 0.6') and not spath.search('prefix-store/template')
    assert not spath.search('$W/tmp/x') and not spath.search('./tmp/x')
    for hit in ('asked Claude to', 'the main session', 'la Sessione principale', 'two sub-agents', 'a memory note',
                'wf_2af1fd28-820'):
        assert assistant.search(hit), hit
    assert not assistant.search('x/.claude/y') and not assistant.search('magnum opuses') and not assistant.search('wf_run')
    binary, text = b'\0\0D:\\Users\\someone\\x a.b@example.org', b'D:\\Users\\someone\\x a.b@example.org'
    assert privacy_counts(binary) == {'profile path': 1}, 'binary files: profile paths only'
    assert privacy_counts(b'\0', 'r/scratchpad-a/x.bin') == {'profile path': 0, 'scratchpad / session id': 1}, \
        'tracked paths are searched, binary files too'
    assert [privacy_counts(text)[k] for k in ('profile path', 'e-mail', 'token')] == [1, 1, 0]
    mixed = 'C:\\U\u017fers\\x a@b.org ghp_' + 'a1' * 18 + ' hf_' + 'b2' * 17 + ' Se\u017f\u017fion-\u0130D C:\\Temp\\x asked Claude wf_2af1fd28 192.168.1.20'
    assert privacy_counts(mixed.encode()) == {name: len(p.findall(mixed)) for name, p in PRIVACY}, 'LITERALS hide a match'
    assert all(privacy_counts(mixed.encode())[name] for name, _ in PRIVACY[:7])
    line = b'Trailer: the main session, two sub-agents'
    entry = ('p/x.patch', 'assistant / session words', line_id(line.decode()))
    def counted(data, path):
        return privacy_counts(data, path, (entry,))['assistant / session words']
    assert counted(b'x\n' + line + b'\r\na memory note\n', 'p/x.patch') == 1, 'ALLOWED: the line is left out'
    assert counted(b'x\n' + line + b'\na memory note\n', 'p/y.patch') == 3, 'ALLOWED: in that file only'
    assert counted(line + b', a memory note\n', 'p/x.patch') == 3, 'ALLOWED: a longer line counts'
    assert counted(line.replace(b'two sub-agents', b'one subagent') + b'\n', 'p/x.patch') == 2, \
        'ALLOWED: other words, same number of matches, count'
    assert counted(b'x ' + line + b'\n', 'p/x.patch') == 2, 'ALLOWED: the text inside another line counts'
    assert counted(line + b'\n' + line, 'p/x.patch') == 2, 'ALLOWED: the first such line only'
    assert privacy_counts(b'a.b@example.org\n', 'p/x', (('p/x', 'e-mail', line_id('a.b@example.org')),))['e-mail'] == 1, \
        'strict stays strict'
    assert all(name in dict(PRIVACY) and name not in STRICT_NAMES and re.fullmatch('[0-9a-f]{16}', lid)
               for _, name, lid in ALLOWED)
    sha = '0123456789abcdef' * 2 + '01234567'
    good = f'# source_commit {sha} 2026-11-11\nc/a.h\tsrc/colibri/c/a.h\n\n# note\nc/t/b.c\tsrc/colibri/c/t/b.c\n'
    assert parse_sync(good) == ((sha, '2026-11-11'), [['c/a.h', 'src/colibri/c/a.h'], ['c/t/b.c', 'src/colibri/c/t/b.c']], [])
    assert len(parse_sync('')[2]) == 1 and len(parse_sync('# source_commit notasha yesterday\n')[2]) == 1
    assert len(parse_sync(f'# kaira_commit {sha} 2026-11-11\n')[2]) == 1, 'the old header name is refused'
    bad = f'# source_commit {sha} 2026-11-11\nscripts/a.h\tsrc/colibri/c/a.h\nc/a.h\tsrc/colibri/c/b.h\nc/a.h\n'
    assert len(parse_sync(bad)[2]) == 3 and len(parse_sync(bad)[1]) == 2
    head = b'#!/bin/sh\r\n# SPDX-License-Identifier: Apache-2.0\r\n# Copyright 2026 ' + HOLDER + b'\r\nx\r\n'
    assert without_licence(head) == b'#!/bin/sh\r\nx\r\n', 'licence header removed, line ends kept'
    assert without_licence(b'x\n# SPDX-License-Identifier: X\ny\nz\n') == b'x\n# SPDX-License-Identifier: X\ny\nz\n'
    assert blob_id(b'') == 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'
    assert has_spdx(b'#!/bin/sh\n# SPDX-License-Identifier: Apache-2.0\n') and not has_spdx(b'x\n' * 5 + b'# SPDX-License-Identifier: X')
    assert slug('13.3 `app\\config.json`') == '133-appconfigjson' and slug('The [`c/`](c/) mirror') == 'the-c-mirror'
    page = '# A\n## 5. Index (v4)\n```\n# not a heading\n```\n## 5. Index (v4)\n<a id="here"></a>\n## Done ##\n'
    assert anchors(page) == {'a', '5-index-v4', '5-index-v4-1', 'here', 'done'}
    lan = PRIVACY[6][1]
    for hit in ('host 192.168.1.20', 'at 10.0.0.1:80', '172.16.0.1'):
        assert lan.search(hit), hit
    for miss in ('Windows 10.0.26300.1234', '172.15.0.1', 'v1.10.0.1.5', '10.1.2.3.4', '127.0.0.1'):
        assert not lan.search(miss), miss
    gh = '1+some-one@users.noreply.github.com'
    assert commit_mail_errors(gh, 'noreply@github.com', 's\n\nCo-Authored-By: A <noreply@example.org>\n') == []
    assert commit_mail_errors('49699333+dependabot[bot]@users.noreply.github.com', gh, 's') == []
    assert commit_mail_errors('a.b@example.org', gh, 's') == ['author']
    assert commit_mail_errors(gh, 'a.b@example.org', 's') == ['committer']
    assert commit_mail_errors(gh, gh, 's\n\nSigned-off-by: A <a.b@example.org>\n') == ['message']
    assert commit_mail_errors('noreply@example.org', gh + '.example.org', 's') == ['author', 'committer'], \
        'a noreply address of another host is not a GitHub one'
    assert commit_mail_errors('', gh, 's') == ['author']
    print('self-test ok')
    return 0


def main(argv):
    ap = argparse.ArgumentParser(usage='python -I check_repo.py [--source <TREE> [--ref master]] [--self-test]'
                                       ' [--line-id FILE N] [--commits RANGE]', allow_abbrev=False)
    ap.add_argument('--commits', metavar='RANGE')
    ap.add_argument('--source')
    ap.add_argument('--kaira', dest='source', help=argparse.SUPPRESS)   # old name, kept so existing commands work
    ap.add_argument('--ref', default='master')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--line-id', nargs=2, metavar=('FILE', 'N'))
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.commits:
        return 0 if check_commits(args.commits) else 1
    if args.line_id:   # FILE is relative to the repository root, as in ALLOWED; N counts from 1
        f, n = args.line_id
        try:
            lines = read(f).decode('utf-8', 'replace').split('\n')
        except OSError as e:
            sys.exit(f'check_repo.py: {e}')
        if not n.isdigit() or not 1 <= int(n) <= len(lines):
            sys.exit(f'check_repo.py: {f} has no line {n}')
        print(line_id(lines[int(n) - 1]))
        return 0
    try:
        files = sorted(f for f in git('ls-files', '-z').split('\0') if f and os.path.isfile(os.path.join(ROOT, f)))
        ok = [check_links(files), check_mirror(files, args.source, args.ref), check_licence(files), check_privacy(files)]
    except (subprocess.CalledProcessError, OSError) as e:
        sys.exit(f'check_repo.py: {e}')
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
