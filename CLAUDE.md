# Knowledge base — schema & conventions

This file is loaded automatically into every Claude Code session in this
repo, but it currently only documents one subsystem: a personal research
knowledge base for the thesis (`raw/` + `wiki/` at repo root), following
the "LLM wiki" pattern (Karpathy:
https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

`raw/` + `wiki/` are **separate from `docs/`** — `docs/` holds this
project's own generated reports/logs (development log, Stage A/B results,
capture-pipeline plan); `raw/`/`wiki/` hold synthesized *reading* — vendor
documentation, papers, background on techniques used in the thesis
(SoilingNet, focal loss, YOLO backbones, camera/lens optics, etc.).

Note: repo root also has `data/raw/` (training images for Stage A/B) — a
different `raw/` with an unrelated meaning. Don't confuse the two; this
knowledge base's `raw/` is source *reading material*, `data/raw/` is
*training data*.

If this file later grows to cover general project conventions (build/test
commands, coding style, etc.) beyond the knowledge base, split those into
their own section rather than mixing them into the schema below.

## Structure

```
raw/                <- immutable source materials, the actual files
└── <topic>/
    ├── README.md     <- one row per file: name, one-line description, ingest status
    └── <file>.md|.pdf|.txt   <- the real source file
wiki/               <- everything below is LLM-maintained, humans don't hand-edit these
├── index.md         <- catalog of every wiki page, one line each, organized by category
├── log.md            <- chronological append-only operation log
├── overview.md       <- synthesis: what this knowledge base currently "knows", in prose
├── conventions.md    <- working preferences/rules for this session (frontmatter fields, etc.)
├── sources/          <- one page per raw source: summary + key takeaways + citations
├── entities/          <- people, organizations, hardware, tools, papers-as-things
├── concepts/          <- theories, methods, techniques, patterns
└── analyses/          <- comparisons, syntheses, "how does X relate to Y" pages
```

## raw/ convention

Source files live here **as real files**, not references to somewhere else
on disk — this is the single copy, so everything needed to rebuild the
wiki is inside the repo itself (portable to another machine, the HPC, etc.).
When adding a new source: drop it into a topic subfolder of `raw/` (make a
new subfolder if it doesn't fit an existing one) and add a row to that
subfolder's `README.md`.

raw/ entries are **immutable** once added — never edited or re-worded. If a
source is superseded, add the new one alongside it and note the
supersession in the wiki page that covers it; don't overwrite or delete the
old one.

Note: this does mean binary sources (PDFs etc.) are tracked in git, which
grows repo size over time — accepted tradeoff for having one authoritative
copy that's easy to keep updating.

## wiki/ conventions

- All pages are markdown, cross-referenced with `[[wikilinks]]` (the link
  target is the other page's filename without `.md`, e.g. `[[alvium-1800-u-1240c]]`).
  This mirrors the `[[name]]` convention already used in this Claude
  installation's own memory files, so it should feel familiar.
- Every wiki page's frontmatter cites the raw/ source(s) it was built from,
  as a path relative to repo root:
  ```yaml
  ---
  title: <page title>
  sources: [raw/camera_docs/Alvium-USB-Cameras_User-Guide.pdf]
  updated: YYYY-MM-DD
  ---
  ```
- `log.md` entries use one line per operation, newest entry on top:
  `## [YYYY-MM-DD] <ingest|query|lint|reorg> | <what happened>`
- `index.md` is regenerated/updated whenever a page is added or removed —
  it's the fast lookup table, not prose.

## Workflows

**Ingest** (adding a new source): read the raw material in full, drop the
file into `raw/<topic>/` and add/update a row in that subfolder's
`README.md`, write or extend the relevant `wiki/sources/`,
`wiki/entities/`, and `wiki/concepts/` pages, cross-link them, append a
`log.md` entry, and update `index.md`. Flag any contradiction with existing
wiki content explicitly in the page rather than silently overwriting it.

**Query** (answering a question from the knowledge base): consult
`index.md` first to find candidate pages, read the relevant pages in full,
answer with citations back to the specific wiki page (and its source). If
the answer produces a new synthesis worth keeping, offer to file it as a
new `wiki/analyses/` page rather than letting it disappear into chat.

**Lint** (periodic health check, run when asked): scan for contradictions
between pages, claims that a newer source has superseded, orphan pages with
no inbound `[[links]]`, and gaps where `index.md` references something that
was never actually written.

## Optional tooling

`wiki/` is plain markdown with standard `[[wikilink]]` syntax, so the
folder can be opened directly as an **Obsidian** vault if you want visual
graph/backlink browsing (already done, as of this writing — `wiki/.obsidian/`
exists on disk). Useful (optional, not required) plugins
from the original pattern:
- **Dataview** — query pages by their YAML frontmatter (e.g. list every
  `entities/` page with `sources` citing a given file).
- **Web Clipper** — browser extension to drop new articles straight into
  `raw/` as markdown.

None of this is installed or required by Claude Code's own workflow above —
they're purely for browsing the vault yourself.
