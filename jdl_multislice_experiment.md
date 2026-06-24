# Dynamic-lookup rendering experiment (branch `jdl-experiment`)

Goal: evaluate whether replacing the NEWS Manhattan routing with a **dynamic
lookup** (each tile pulls the Gaussians it needs directly from wherever they
live) makes rendering faster and removes the channel-saturation flicker.

This file is the running log + design for the experiment so we can pick it up
or write it into the rebuttal.

---

## Two candidate mechanisms

### 1. JDL (`external/jit-dynamic-lookup`)
Graphcore Research's prototype. `JDL::createPrograms(graph, data, tileSelector,
elementSelector, result)` lets a **single receiver tile** pull a contiguous
slice from any **sender tile**, with the source tile/offset chosen at runtime.

**Showstopper for us** (`JDL.hpp:78`): a tile that holds `data` may **not**
also be a `result` (receiver) tile — it's an `assert`, and `JDL.gp` is a
precompiled Mk2 binary we can't patch. In our renderer every one of the 1440
tiles both *holds* Gaussians and *needs* to receive them. Proxying through the
~32 spare tiles means ~45 batched rounds/frame and a huge graph. Not viable as
a drop-in. (Authors note the receiver=sender fix "would be easy" — worth an
email if we pursue this.)

### 2. `popops::multiSlice` (standard Poplar)
A planned gather: pull `M` rows from a `[N, width]` table by a runtime
`offsets` tensor. No sender≠receiver restriction. This is the path we test
first.

---

## The real blocker: discovery, not transport

Either mechanism only *moves* a Gaussian once a tile knows its **global
index**. Today the NEWS protocol *is* the discovery mechanism (a Gaussian
walks toward its anchor tile and blooms into neighbours it straddles).

To use a gather instead, each tile must compute the indices of all Gaussians
overlapping its screen region. The naive "every tile sees every Gaussian's
2D footprint" needs ~44k × (8 B mean + 4 B radius + 4 B index) ≈ **700 KB per
tile** — over the 624 KB tile budget. So a full metadata AllGather does **not
fit**. This memory wall is exactly why the paper chose local message passing.

Discovery options if the gather proves cheap:
- **Spatial-hash metadata**: only exchange footprints for nearby screen
  regions (bounded per tile).
- **Host-computed assignment**: project on the host (already done for CPU
  mode), build per-tile index lists, stream them down. Defeats the "all
  on-chip" story but isolates gather perf and gives zero flicker / instant
  convergence — a clean baseline.
- **Two-level multiSlice**: gather compact metadata first, then full structs.

---

## Stage 1 — microbenchmark (current)

Before any renderer change, measure the **raw `popops::multiSlice` gather
cost** for a representative 3DGS workload. This is the go/no-go number.

`tools/multislice_bench.cpp` (target `multislice_bench`): gathers
`numTiles × perTile` rows of 15 floats (one `Gaussian3D`) from a
`numGaussians × 15` table and times it, amortised over many repeats.

```bash
# in the container build dir
ninja multislice_bench
./multislice_bench                 # defaults: 44000 entries, 400/tile, 1440 tiles
./multislice_bench 44000 400       # numGaussians perTile
```

Compare the per-gather ms against the current NEWS exchange cost (the
`single_exchange` timing from `--benchmark`, typically a few ms). 

**Decision rule**: if one multiSlice gather of the full per-frame working set
is comparable to or cheaper than the NEWS exchange *and* the result fits in
SRAM, proceed to Stage 2. If it's much slower, the experiment validates the
paper's local-message-passing choice — also a useful rebuttal result.

### Stage 1 result (SDK 3.3, C600, 2026-06-24) — GREEN
```
table:   44000 entries x 15 floats
lookups: 576000 (1440 tiles x 400)   # full per-frame working set
per gather: 0.238 ms   (32.96 MiB/gather, ~138 GB/s — near exchange-fabric peak)
```
Transport is ~40x under the ~9 ms/frame compute and well under the NEWS
exchange. **Dynamic-lookup transport is not the bottleneck.** Proceed to
Stage 2 — but note this measures *only* the gather, with indices supplied. It
does NOT yet include:
1. **Discovery** — computing each tile's needed global indices (the memory
   wall above is still the open problem).
2. **Output mapping** — `multiSlice`'s output layout is plan-controlled. For
   blending, each tile's gathered Gaussians must land *on that tile*. Whether
   the plan can pin per-tile outputs (or at what cost) is the next unknown to
   test before/with Stage 2.

## Stage 2 — renderer integration (only if Stage 1 is green)
Behind a runtime flag `--gather-mode news|multislice` (default `news`, so the
working renderer is untouched). Replace `broadcastPoints` + the routing half of
`RouteVertex` with: project locally → exchange compact metadata → multiSlice
the needed structs → sort → blend.

---

## Switching back
Stage 1 is a **separate binary** — the renderer is not modified, so there is
nothing to switch back. To drop the experiment entirely:
`git checkout rebuttal-experiments`.
