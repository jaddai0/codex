# Archive retention and storage pressure

Mavis owns this local archive policy. It never infers that a project is closed from
silence. Register the conversation IDs and their objective IDs, then close the
project only after its objectives are accepted or cancelled. A bound conversation
whose objective is omitted blocks closure. A conversation may belong to only
one archive project; the registry lock enforces that rule across concurrent
registrations.

```text
mavis archive-retention register PROJECT --conversation SESSION [--objective OBJECTIVE ...]
mavis archive-retention close PROJECT [--protect /absolute/evidence/path ...]
mavis archive-retention compact PROJECT
mavis archive-retention restore PROJECT
mavis archive-retention reopen PROJECT
mavis storage-pressure [--prune-reproducible-cache]
```

`compact` requires 30 days since closure and no changed segment set, handoff,
objective, or raw file. It scans known handoff and objective references, leaving
those raw segments at their original paths. It also protects live librarian
follow-up citations. Each other segment is gzip compressed
in the archive's own directory. The original SHA-256 and compressed SHA-256 stay
in the manifest. Mavis reconstructs and hashes the exact original bytes before
changing the manifest, then removes only the verified raw duplicate. Search and
librarian citation checks verify both hashes and read compressed segments directly.
Compression and restoration pin the archive's segments directory with a no-follow
directory handle. Reads, generated files, and removals use that handle and check
the file's inode and hash. If a directory change is detected before duplicate
removal, Mavis reports failure and rolls back the manifest while retaining both
copies. A change detected just after removal is reported without changing the
manifest that points to the retained copy.
An interrupted operation can reuse an already verified gzip or remove a verified
raw duplicate. `restore` reconstructs raw files and checks hashes; `reopen`
restores them before accepting new project activity.
The existing idle maintenance claim runs a due-project sweep automatically, at
most once per project per day. It records validation failures and leaves those
archives intact. No independent clock-based scheduler is installed; if no idle
maintenance claim occurs, no sweep occurs.

Before transcript or command-output evidence is written, Mavis checks free space.
At less than 20 GiB free or 10% of disk capacity, whichever is greater, it emits
a warning, saves `storage-pressure.json`, and prunes registered reproducible cache
files first. A cache file is eligible only while its retained source still has
the registered hash. Unknown cache files, transcripts, logs, model weights, and
other unique evidence are never pruned. The registry is optional and malformed
entries cause a warning rather than blocking evidence writes. No production
cache producer currently registers files in this namespace, so the current
production behavior is a capacity warning, not guaranteed space recovery.
Future cache producers can use `register_reproducible_cache` after successful
generation and retained-source verification.

The disable path is to leave projects open and omit `compact`, and to omit
`--prune-reproducible-cache` for manual pressure checks. The rollback path is
`restore` or `reopen`; no external service or migration of old transcripts is
required. Existing uncompressed manifests continue to search as before.

Limits: a reference outside Mavis handoffs, objective records, or the explicit
`--protect` list cannot be discovered. Such paths should be declared at closure.
Storage pressure does not delete project search indexes or other caches whose
regeneration has not been registered. It cannot guarantee a future write if the
disk remains full after safe cache pruning; the warning and receipt expose that
condition.
