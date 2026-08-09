# AED Siting — Singapore

> **Prototype for planning and simulation only—not for emergency use.**
> In an emergency in Singapore, call **995** immediately and follow SCDF
> instructions. Use official SCDF/myResponder channels. Do not delay emergency
> action to use this prototype.

A planning tool that produces a **ranked install queue** for additional
public-access AEDs across four Singapore planning areas: given a staged budget of
*k* devices, which site first, second, third — and what modelled coverage each one
buys. Recommendations name a real place ("Blk 102B Bidadari Park Drive"), and
coverage is drawn on a map rather than asserted in a number.

It is not a map of pins, and it is not a "find my nearest AED" tool.

---

## Live demo

**▶ [Open the live app](https://sgtdp-aed-intelligence-mqhejpdxwgtefbtcwz4gzy.streamlit.app/)**
— running on Streamlit Community Cloud, all four planning areas ready.

**▶ [Watch the demo video](https://drive.google.com/drive/folders/16GgX-GL3ixS64YvcARlywA2eMAXyUuKW?usp=sharing)**

Running it locally takes two commands — see [Reproduce](#reproduce).

---

## The decision

| | |
|---|---|
| **User** | A community planner allocating a staged AED budget. |
| **Decision** | Given budget for *k* devices this cycle, which sites, in what order? |
| **Success** | More demand-weighted coverage per device than a population-ranked rule, with the ranking stable across demand assumptions. |
| **Not this tool's job** | Telling anyone where the nearest AED is right now. That is a question about one person at one moment; this tool answers questions about the network in aggregate. |

Out-of-hospital cardiac arrest is time-critical, and the American Heart
Association notes that immediate CPR can double or triple the chance of
survival, with its adult basic life support guidance emphasising prompt
emergency activation, high-quality CPR, and early defibrillation. These facts
establish the importance of preparedness; they do not mean this prototype can
predict or claim improved survival. Every number this tool reports is a planning
estimate under stated assumptions.

Singapore already has an official emergency-response system. Members of the
public should call **995** and follow SCDF instructions; SCDF's **myResponder**
app alerts registered Community First Responders, connects users to 995, and
provides access to the national AED registry. This prototype does not connect
to, replace, or compete with those services.

---

## Status

Complete and running on four planning areas. Every claim below is gated by
`make check` — 142 tests covering the safety banner, the data contract, licence
and provenance, and regressions for defects this project caught in its own code.

**Documents:** [method card](docs/method_card.md) ·
[evaluation report](docs/evaluation.md) · [safety statement](docs/safety.md) ·
[data quality](docs/data_quality.md) · [sources](docs/sources.md) ·
[primary metric](docs/primary_metric.md)

---

## Architecture

```
demand proxies ─┐
                ├─→ distance matrix ─→ optimizers ─→ ranked queue ─→ scenarios
candidate set ──┘
```

The governing commitment: **constraints live in the feasible set, never in a
post-hoc corrector.** Candidate sites are real OSM building footprints that pass
a tag allowlist and a feasibility filter, so no optimiser can propose a site in
the middle of a road — no such point exists in the candidate set to begin with.

---

## Method

483 H3 res-10 cells over Toa Payoh (the other three areas scale with their size). Candidates are **real OSM building
footprints** passing an eligibility filter, located at each polygon's
representative point — guaranteed inside the building. Distance is straight-line
in EPSG:3414 scaled by a 1.35 detour factor, giving one-way network-estimate
metres; the 126 m threshold is *derived* from a 3-minute round-trip budget at
1.4 m/s, not asserted. Placement is marginal-coverage greedy on a maximal-covering
objective, pre-seeded with the existing 284 devices, with an ILP certifying
optimality. Full detail: [method card](docs/method_card.md).

**The planner stays in the loop.** Sites can be *pinned* (must include) or
*vetoed* (must exclude), and the model re-solves around them rather than editing
a finished list — so every later pick is optimised against the state the
override actually creates. The tool then re-solves a second time without the
constraints and reports the **cost of the override** in coverage. That cost is
occasionally negative, because greedy is a heuristic and a human pin can beat
it; the number is shown either way.

### The optimiser, in more detail

**The objective is maximal covering (MCLP):** choose *k* sites that maximise the
demand-weighted share of cells lying within the 126 m reach. It is **pre-seeded
with the existing network**, so the queue answers "what should we add" rather
than redesigning a network that already exists — a candidate scores only the
demand it covers that nothing already reaches.

**Why greedy, given an exact solver exists.** Marginal coverage is
*submodular* — each device adds less once its neighbours are placed — so greedy
carries the standard **1 − 1/e ≈ 63%** worst-case guarantee against the optimum,
and in practice does much better here. More importantly its queue is **nested**:
`top(k)` is always a prefix of `top(k+1)`. That is what makes a staged budget
safe. Funding 10 devices this cycle and 10 next never invalidates the first 10,
which an exact re-solve at each budget would not guarantee.

**The ILP certifies rather than replaces it.** An ILP (PuLP + CBC) solves the
same objective exactly, but returns an unordered *set* per budget with no
internal order — useless as an install queue. It is used as an oracle instead,
at k = 1, 3, 5 and 30. Greedy matches the certified optimum **exactly at
k = 1, 3 and 5** in three of the four areas, and its largest shortfall anywhere
is **0.31 pp at k = 30** (Bedok) — at worst **99.3% of optimal**. The nested,
staged-budget queue is therefore very nearly free, but it is not free at every
budget, and `nesting_loss.csv` reports the gap per area rather than rounding it
away.

**Priorities are an ordering, not a single objective.** The planner ranks the
criteria — demand covered, worst gap, average trip, relief of busy devices,
worst subzone — and they are applied lexicographically at each step. Each key
admits a **2% relative tolerance band**: candidates within the band of the best
score count as tied and are separated by the next key. Without banding, exact
float ties essentially never occur, so the ranking silently collapses onto the
first key and every ordering yields the same plan. Banding also fixes a real
defect this project caught in its own code: an earlier k-center returned
`[184, 0, 1, 2, 3, …]` because all 791 candidates scored identically once the
worst-served cell became unreachable, and `argmin` simply walked the index —
nine of ten picks were arbitrary buildings presented as an equity plan.

**Where the queue stops.** It runs to *saturation* rather than to a fixed *k*,
and halts when no criterion can separate the remaining candidates. That makes
the queue length meaningful: Toa Payoh saturates at 94 devices and 80.2%
coverage, and the budget control accepts any value up to that.

**Measured against:** the required demand-ranked top-*k* baseline and a random-*k*
band over 50 seeds, both reported in [Results](#results-vs-baseline). Full
derivations: [method card](docs/method_card.md).

**Gate:** `make gates`

---

## Results vs baseline

At the pre-registered 3-minute threshold (126 m one-way), k = 10:

| Method | Coverage | Median | p90 |
|---|---|---|---|
| Do nothing (284 existing devices) | 47.04% | 145 m | 396 m |
| Demand-ranked top-k *(required baseline)* | 47.67% | 136 m | 390 m |
| Random-k, 50 seeds (p95) | 50.52% | — | — |
| **Marginal-coverage greedy** *(method)* | **56.90%** | **111 m** | **325 m** |

**+9.86 pp over doing nothing, +9.23 pp over the baseline.** An ILP certifies
greedy as exactly optimal at k = 1, 3 and 5 here, and 0.19 pp short of optimal at
k = 30, so the nested staged-budget queue costs almost nothing.

**The same pipeline, unchanged, on four areas:**

| Area | Devices | Coverage now | With 10 more | Ceiling (queue length) |
|---|---|---|---|---|
| Toa Payoh | 284 | 47.0% | **56.9%** | 80.2% (94) |
| Ang Mo Kio | 354 | 46.5% | 53.3% | 67.9% (85) |
| Bedok | 469 | 31.2% | 35.8% | 69.1% (234) |
| Jurong East | 243 | 32.4% | 36.4% | 38.8% (49) |

Gain tracks candidate availability and geography, not the size of the deficit —
Jurong East is worst covered and gains least, because it has the fewest eligible
buildings to work with.

**k = 10 is a budget, not a ceiling.** The queue is built to *saturation* — the
point where no remaining candidate improves coverage — and the budget control
accepts any value up to that length. The ceiling column is what the candidate
pool can reach if fully funded; the gap between it and 100% is ground no
eligible building can cover, not a modelling shortfall to be tuned away.

**The most important caveat:** across nine scenarios, **no site is chosen in ≥90%
of them** — the consensus set is empty, and the three objectives agree on zero
sites. The *size* of the achievable gain is robust; the *identity* of the ten
sites is not. Every recommendation therefore ships with its stability share.
Full analysis, including negative results: [evaluation report](docs/evaluation.md).

---

## Scope and limits

- A record in the source registry is **not** proof that a device is present,
  accessible, inspected, or working. The data is a **February 2020** snapshot.
- **The snapshot is measurably stale.** SCDF states the live national AED
  registry now holds [more than 11,000 devices](https://www.scdf.gov.sg/savealife);
  this dataset has **9,644** records. Every coverage figure here describes the
  2020 network, not today's.
- Coverage is **modelled**: straight-line distance × 1.35 detour factor,
  round-trip, at 1.4 m/s, travel time only.
- Operating hours are taken **as stated**, not as verified present access.
- Results are planning estimates, not predictions of cardiac arrest or survival.
- **Four planning areas, not Singapore.** MP2019 defines **55** planning areas;
  this models 4 of them, holding **1,350 of the registry's 9,644 devices (14%)**.
  Nothing here is an island-wide result, and the four areas were chosen to be
  demonstrative, not representative.

**Each area is solved in isolation, and that biases the edges.** Devices are
clipped to the study-area polygon, so an AED a few metres across the boundary is
invisible to the model even though it would genuinely cover demand inside it:

| Area | Devices used | Ignored, but within 126 m of the border | as % of used |
|---|---|---|---|
| Jurong East | 243 | 37 | 15.2% |
| Bedok | 469 | 50 | 10.7% |
| Toa Payoh | 284 | 19 | 6.7% |
| Ang Mo Kio | 354 | 3 | 0.8% |

The error has a direction worth stating: existing coverage near a boundary is
**understated**, so the tool **over-recommends** placements there. It is worst in
Jurong East, which is also the weakest area in the results table — so some of
that area's apparent deficit is an artefact of where the border falls, not a real
gap. Treat border recommendations with more suspicion than interior ones.

---

## Data and licences

| Source | Publisher | Licence | Data date |
|---|---|---|---|
| [Public Access AEDs](https://data.gov.sg/datasets/d_4e6b82c58a8a832f6f1fee5dfa6d47ea/view) | SCDF | Singapore Open Data Licence | Feb 2020 (page updated 13 Nov 2025) |
| [MP2019 Planning Area Boundary (No Sea)](https://data.gov.sg/datasets/d_4765db0e87b9c86336792efe8a1f7a66/view) | URA | Singapore Open Data Licence | 2019 |
| [MP2019 Subzone Boundary (No Sea)](https://data.gov.sg/datasets/d_8594ae9ff96d0c708bc2af633048edfb/view) | URA | Singapore Open Data Licence | 2019 |
| OpenStreetMap buildings and POIs | OpenStreetMap contributors | **ODbL 1.0** — attribution and share-alike | frozen extract, see manifest |

**This repository:** code is [MIT](LICENSE); the derived data under
`data/processed/` is a Derivative Database of the OpenStreetMap extract and is
released under **ODbL 1.0** in turn. See [`LICENSE`](LICENSE).

Full provenance, checksums, transformations, excluded fields, and the role each of
the challenge brief's six [official resources](docs/sources.md#2-official-resources-from-the-challenge-brief)
plays in this submission: [`docs/sources.md`](docs/sources.md) and
[`data/manifest.json`](data/manifest.json).

**Official emergency channels this tool does not replace:**
[myResponder](https://www.scdf.gov.sg/home/community-and-volunteers/community-resources/myresponder-app)
· [SCDF Emergency Medical Services / 995](https://www.scdf.gov.sg/home/about-scdf/emergency-medical-services)
· [the national AED registry](https://www.scdf.gov.sg/savealife). No integration
with any of them is performed or attempted.

---

## Reproduce

**Two commands to a running tool.** The pre-computed artifacts for all four
areas are committed, so nothing has to be built first:

```bash
uv sync --python 3.12          # Python 3.12 is pinned; 3.14 lacks wheels
uv run streamlit run app.py    # the planning tool, all four areas ready
```

To verify rather than just run it:

```bash
make check                                                   # ruff format, ruff check, 142 tests
uv run python -m aed_siting.pipeline --all-towns --seed 42   # rebuilds every artifact (~25 min)
```

Every raw input is already committed, so **all of the above runs with no network
access** — enforced by a socket-blocking fixture in the test suite. Re-freezing
the inputs from source is optional and is the only step that touches a network:

```bash
uv run python scripts/fetch_sources.py   # re-downloads and re-hashes data/raw/
```

Running the pipeline twice with the same seed produces byte-identical output on
the greedy path. The ILP is asserted only on objective value, since CBC may return
any of several equally-optimal site sets.

---

## Repository structure

```
src/aed_siting/     pipeline package
  config.py         pinned constants — CRS, thresholds, dataset ids, seed
  safety.py         verbatim safety banner and point-of-use disclosures
  io/               frozen-input loading: manifest, AED registry, boundaries
scripts/            one-shot source freezing
docs/               primary metric, method card, evaluation, safety statement
data/raw/           frozen inputs, committed
tests/              gates (`-m gate`) plus unit tests
```

---

## What I would improve with more time

1. **More eligible candidate sites.** The worst-served cell is 785 m from help and
   a dedicated worst-case objective moves that by under 9 metres, because no
   eligible building lies near it. The binding constraint is the candidate pool,
   not the algorithm — so widening it (a learned eligibility classifier, or
   surveyed sites) would do more than any optimiser change.
2. **Walking-network distances.** The 1.35 detour factor cannot see barriers — a
   canal or expressway between two points is invisible, so the model is optimistic
   exactly where the real network is worst. The `DistanceModel` protocol exists so
   this is a one-line swap with no threshold change.
3. **A learned demand surface.** Recommendations are unstable across demand
   proxies (`area` vs `poi` rank-correlate at only 0.241), and that instability is
   the largest source of uncertainty in the output. Training a footfall surface on
   public transit ridership would replace the weakest assumption in the pipeline.
4. **A siting-cost term.** Permission difficulty and installation cost by building
   type would make the ranking operational rather than purely geometric. It was
   cut because a synthetic term needs its own sensitivity treatment.
5. **Correlated outage modelling.** The stress test drops devices uniformly; real
   failures cluster, so clustered risk is understated.
6. **All 55 planning areas, and allocation between them.** Each area is currently
   solved on its own, so the tool cannot answer the question a national planner
   actually has — *where in Singapore do the next 50 devices go?* That needs a
   budget split across areas, not four separate queues.
7. **Cross-boundary solving.** Including a halo of devices from neighbouring
   areas in the coverage calculation would remove the edge effect measured in
   [Scope and limits](#scope-and-limits) — border cells would stop being counted
   as gaps when a device across the line already reaches them. This is the
   single largest correctness gap in the current model.
8. **Better human and manual correction.** Overrides work today through the
   sidebar pin/veto lists, and the model genuinely re-solves around them. Direct
   manipulation on the map still needs work: dependable click targets, placing a
   device at an arbitrary point rather than only on a vetted candidate building,
   and dragging a proposal to reposition it. A planner's local knowledge should
   be expressible by pointing at the map, not by picking an ID from a list.
9. **Harden the rough edges.** Known usability issues — map-click reliability
   among them — would come before any new modelling work. The model is further
   along than the interface around it.

---

## Acknowledgements

**Data.** [SCDF](https://www.scdf.gov.sg) for the Public Access AED registry and
[URA](https://www.ura.gov.sg) for the MP2019 planning-area and subzone
boundaries, both via [data.gov.sg](https://data.gov.sg) under the Singapore Open
Data Licence. Building footprints and points of interest are
© [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, used
under ODbL 1.0. Per-dataset ids, dates, checksums and the exact fields used are
in [`docs/sources.md`](docs/sources.md) and
[`data/manifest.json`](data/manifest.json); licence terms are in
[`LICENSE`](LICENSE).

**Basemap.** Map tiles by [CARTO](https://carto.com/attributions), with map data
© OpenStreetMap contributors. The attribution is also rendered in the map itself.

**Open-source software.** This project is built on work by others, each used
under its own licence: [GeoPandas](https://geopandas.org),
[Shapely](https://shapely.readthedocs.io), [PyProj](https://pyproj4.github.io/pyproj/),
[H3](https://h3geo.org), [OSMnx](https://osmnx.readthedocs.io),
[NumPy](https://numpy.org), [pandas](https://pandas.pydata.org),
[SciPy](https://scipy.org), [PuLP](https://coin-or.github.io/pulp/) with the
[CBC](https://github.com/coin-or/Cbc) solver,
[scikit-learn](https://scikit-learn.org), [Streamlit](https://streamlit.io),
[pydeck](https://deckgl.readthedocs.io) / [deck.gl](https://deck.gl),
[Altair](https://altair-viz.github.io), [Ruff](https://docs.astral.sh/ruff/),
[pytest](https://pytest.org) and [uv](https://docs.astral.sh/uv/). Exact pinned
versions are in [`requirements.txt`](requirements.txt) and `uv.lock`.

**Clinical context.** American Heart Association Basic Life Support guidance is
cited for context on why AED access is time-critical. Those facts establish the
importance of preparedness; they **do not mean** this prototype predicts or
improves survival. They motivate the problem and are not evidence for any number
reported here, and no survival claim is made anywhere in this project.

**AI-assisted development.** Built with AI assistance
([Claude Code](https://claude.com/claude-code)) for implementation, refactoring
and code review. The problem framing, method, and modelling decisions are the
author's, and every reported figure was verified against the committed artifacts
before publication.
