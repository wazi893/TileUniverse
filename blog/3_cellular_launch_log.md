# Cellular launch log

Started: 2026-07-08

## Pre-flight (step 1) — 2026-07-08

Status: **BLOCKED — live site contradicts the pack. No posting until resolved.**

Pages checked:

- `demos/cellular.html` — loads. Shows live throughput ("Conway cell-updates/sec", "bit-identical to CPU"). No hardcoded wrong number in body text. OK.
- `demos/index.html` (demo gallery) — loads. Shows **6.7 trillion cell-updates/sec** for the cellular demo. OK, matches pack.
- `github.com/wazi893/TileUniverse` — loads. README shows **115T tiles/sec** and **15.8 TCOPS**. No 63.6P. OK for launch numbers. (Note: README also lists unrelated pre-existing numbers — 200B evals/sec, 2.5 TCOPS on RTX 4070 — not launch numbers, not blocking.)
- `index.html` (site overview / landing page) — loads, but **CONTRADICTS THE PACK**:
  1. **Game-of-Life number wrong:** landing page says **"9T" / "~9 trillion" cell-updates/sec** in three places. Pack + gallery say **6.7T**. Direct contradiction.
  2. **Wrong GitHub links:** "View on GitHub" (hero) and footer "GitHub" point to **github.com/tileuniverse/tileuniverse**, not **github.com/wazi893/TileUniverse**. Would send launch traffic to the wrong repo.
  3. **Retired number in page meta:** meta-description says **"2.5 trillion operations per second on consumer GPUs"** — a retired/never-defensible general-purpose-compute framing (on the "never use" list).

Checklist item "landing page shows 115T + 15.8T (not 63.6P)": PARTIAL — 115T ✓, 15.8T ✓, no 63.6P ✓, but the Game-of-Life figure and GitHub links are wrong (see above).

Action: reported to W. Awaiting fix of the landing page before proceeding to r/rust.

### Re-check after fixes (commits a7d1ec0, 22d530a, df3482a) — 2026-07-08

Status: **CLEAN — cleared to post.** Hard-refreshed via cache-buster query.

- Landing `index.html`: hero + Performance both show **6.7T** GoL, **115T** tiles/sec, **15.8T** quantum; "View on GitHub" + footer → **github.com/wazi893/TileUniverse**; meta description = clean 115T/bit-exact line (no 2.5T); nav has **no whitepaper link**.
- `demos/index.html`: 6.7T, OK.
- `demos/cellular.html`: loads, live throughput + bit-identical badge, OK.
- No 9T / 2.5T / 63.6P patterns on any of the three URLs.

Proceeding to step 2 (r/rust).

## Posts

### r/rust — 2026-07-08

- URL: https://www.reddit.com/r/rust/comments/1ur8ijo/a_packedbit_gpu_cellular_fabric_in_rust_conways/
- Title: "A packed-bit GPU cellular fabric in Rust: Conway's Life at 6.7T cell-updates/sec, bit-exact vs CPU" (matches pack)
- **Incognito/logged-out check: PASS.** Verified via logged-out screenshot ~11 min after posting — post renders fully (title + body + link comment), not `[removed]`. AutoMod did not silently remove it.
- Numbers on post correct: 6.7T cell-updates/sec, ~115T tile-evals. Honest-scope paragraph present. First comment links: cellular demo, github.com/wazi893/TileUniverse, repro command — all correct.
- Deviation from pack: links moved into first comment (body says "link in the first comment") rather than inline verbatim. Benign; numbers/scope/links all correct. Noted to W.
- Engagement at check: 0 votes, 1 comment (OP's own link comment).

## Notable comments / follow-ups

(none yet)
