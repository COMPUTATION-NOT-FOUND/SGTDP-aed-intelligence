# Baseline and evaluation report

> **Prototype for planning and simulation only—not for emergency use.**
> In an emergency in Singapore, call **995** immediately and follow SCDF instructions.

All figures reproduce from a clean clone with `make check` then
`uv run python -m aed_siting.pipeline --seed 42`, with no network access.
Four planning areas are built. Toa Payoh is the primary study area (8.18 km²,
483 H3 res-10 cells, **284 existing devices**, **791 candidate sites**); the other
three test whether the method transfers.

**Primary metric**, nominated in [`primary_metric.md`](primary_metric.md) in a
dated commit *before any optimiser existed*: demand-weighted coverage within a
**3-minute round-trip** retrieval budget (**126 m one-way**).

---

## 1. Method versus baselines — k = 10

| Method | Coverage | Median | p90 | p95 | Max |
|---|---|---|---|---|---|
| Do nothing (existing network) | 47.04% | 145 m | 396 m | 499 m | 785 m |
| **Demand-ranked top-k** *(required baseline)* | 47.67% | 136 m | 390 m | 456 m | 785 m |
| Random-k, 50 seeds (mean / p95) | 49.11% / 50.52% | — | — | — | — |
| **Marginal-coverage greedy** *(method)* | **56.90%** | **111 m** | **325 m** | **412 m** | 785 m |

**+9.86 pp over doing nothing, +9.23 pp over the required baseline**, and clear of
the random-placement band's p95. p90 retrieval distance falls 396 m → 325 m
(566 s → 464 s round-trip at 1.4 m/s).

**Why the baseline barely moves.** Demand-ranked top-k adds only 0.63 pp for ten
devices because it scores each candidate against total demand in isolation. It
therefore stacks devices into the densest cells, which the existing network
already covers. That is not a strawman — it is the transparent rule the brief
names, and its failure mode is the whole argument for a marginal objective.

**Max distance is unchanged at 785 m.** The worst-served cell is not reachable by
any of the 791 candidates within the threshold. Coverage-maximising objectives do
not fix worst-case gaps; that would need a p-centre objective, which is out of
scope here and named as future work.

## 1a. Does it transfer? Four planning areas

The same pipeline, unchanged, on four areas spanning the coverage range measured
across 15:

| Area | Devices | Candidates | Coverage now | With 10 more | Gain |
|---|---|---|---|---|---|
| **Toa Payoh** | 284 | 791 | 47.0% | 56.9% | **+9.9 pp** |
| Ang Mo Kio | 354 | 811 | 46.5% | 53.3% | +6.8 pp |
| Bedok | 469 | 4,185 | 31.2% | 35.8% | +4.6 pp |
| Jurong East | 243 | 360 | 32.4% | 36.4% | +4.0 pp |

The method helps everywhere, but **not equally, and not most where coverage is
worst** — which is the opposite of the intuitive expectation. Jurong East has the
lowest coverage and gains the least, because it has the smallest candidate pool
(360 sites): there is little to work with. Bedok has 4,185 candidates and still
gains only 4.6 pp, because its demand is spread over 21.7 km² rather than
concentrated. Gain tracks the *geography*, not the deficit.

## 1b. Alternative objectives — and the limit they expose

| Objective | Coverage | Max (m) | Mean (m) | Guarantee |
|---|---|---|---|---|
| Do nothing | — | 784.5 | 188.9 | — |
| **Max coverage (greedy)** | **56.9%** | 784.5 | 157.5 | 1−1/e; ILP-certified optimal at k ≤ 5 |
| Min worst case (k-center, minimax) | 49.7% | **775.8** | 181.8 | heuristic |
| Min worst case (k-center, Gonzalez) | 48.0% | **775.8** | 186.0 | 2-approx in a metric space |
| Min average (p-median) | 52.3% | 784.5 | **146.5** | heuristic |

Each objective wins on the quantity it optimises, and the trade-off is real:
buying worst-case improvement costs 7.2 pp of coverage.

**But the honest headline is how little k-center achieves: 784.5 → 775.8 m, under
9 metres.** The worst-served cell cannot be fixed by choosing a better objective,
because *no eligible candidate building lies within reach of it*. The binding
constraint is candidate availability, not algorithm choice. Adding a worst-case
objective did not remove the limitation this report previously flagged — it
diagnosed it, which is more useful.

**The objectives agree on zero sites.** Combined with the empty scenario
consensus (§4), that is a consistent message: which ten sites you pick depends
heavily on what you are optimising for and what you assume about demand. The
*size* of the achievable gain is robust; the *identity* of the sites is not.

## 1b-bis. Priority ordering is a policy choice

Placement criteria are applied as an **ordering**, lexicographically, with a 2%
tolerance band per key so a saturated key hands over to the next rather than to
the candidate index. Five orderings are precomputed per town:

| Ordering | Toa Payoh | Ang Mo Kio | Bedok | Jurong East |
|---|---|---|---|---|
| Most people reached *(default)* | **56.9%** | **53.3%** | **35.7%** | 36.4% |
| Worst-served areas first | 56.5% | 53.0% | 35.0% | 36.4% |
| Worst subzone first | 54.6% | 53.3% | 35.1% | 36.4% |
| Shortest average trip | 54.0% | 50.8% | 34.0% | 35.4% |
| Relieve busiest devices | 50.5% | 49.7% | 33.4% | 34.3% |

**Equity is nearly free.** Ranking by the worst gap first costs **0.4 pp** of
coverage in Toa Payoh and places a device **587 m** from the worst-served point
instead of 1,179 m — the closest any eligible building sits.

*This corrects v3.* The earlier report gave the equity objective a 7.2 pp cost.
That figure came from a defective `kcenter_minimax`: once the worst cell became
unreachable, all 791 candidates scored identically and `argmin` walked the index,
so nine of its ten picks were arbitrary. Most of the "cost of equity" was the cost
of nine random sites.

**In Jurong East every ordering converges on 36.4%** — with only 360 candidates
there is nothing to choose between. Where the candidate pool is thin, policy is
irrelevant; the binding constraint is availability, not the algorithm.

## 1c. Simulated incidents and outage stress

1,000 synthetic incidents drawn from the demand surface (seed 42). Illustrative
only — every reported metric uses the exact analytic quantile, and
`test_simulation_matches_analytic_quantiles` asserts the sample tracks it.

| | Reached in time | Median | p90 |
|---|---|---|---|
| Existing network | 47.8% | 145 m | 395 m |
| After 10 devices | 57.3% | 106 m | 319 m |

Coverage when devices are randomly unavailable (100 trials per size). **This
table stresses the 284 existing devices only**, so the shares below are
fractions of the network as it stands today. The tool's Outages tab instead
fails devices drawn from the **existing plus funded** network and captions
itself accordingly — a device the plan recommends can go out too, and assuming
otherwise would flatter the plan exactly where the chart exists to test it.
The two therefore report different denominators by design.

| Devices lost | Share of network | Mean coverage | Bad draw (p05) |
|---|---|---|---|
| 1 | 0.4% | 47.01% | 46.76% |
| 5 | 1.8% | 46.72% | 45.98% |
| 10 | 3.5% | 46.41% | 45.58% |
| 25 | 8.8% | 45.31% | 43.44% |

Degradation is gentle and close to linear: losing 8.8% of the network costs
1.7 pp of coverage. The network is **redundant rather than fragile** — no small
set of devices is load-bearing. Outages are drawn uniformly, so this understates
clustered failures such as a power cut taking out a whole block.

## 1d. Layout quality — Voronoi diagnostics

Coverage percentage says how many people are within reach. It says nothing about
whether the load is spread sensibly, or where the single worst spot is. Service
areas answer both.

| | Before | After 10 devices |
|---|---|---|
| Load Gini | **0.679** | 0.655 |
| Busiest device serves | 3.07% of demand | 3.14% |
| Median device serves | 0.19% | 0.20% |
| Largest empty circle | **593 m** | **593 m** |
| Cell area spread (max ÷ median) | 15.9× | 15.2× |

**Load is heavily unbalanced.** The busiest device is nearest for 16× the demand
of the median device. That is invisible in a coverage figure — two networks with
identical coverage can have completely different load distributions — and it is
the argument for reading catchment load alongside coverage rather than instead
of it. The plan improves balance slightly (0.679 → 0.655) as a side effect of
targeting uncovered demand, not because it optimises for balance.

**The largest empty circle does not move at all.** 593 m before and after. This
is independent confirmation of §1b from an entirely different construction: the
worst-served point is a Voronoi vertex, and no eligible candidate building lies
near it. Coverage improves; the worst case does not.

*(593 m is straight-line. Scaled by the 1.35 detour factor it is ≈800 m, consistent
with the 784.5 m network-estimate maximum in §1.)*

Voronoi is used here strictly as a **diagnostic**. A cell vertex is not a
candidate site — it frequently lands in the middle of a road, which is precisely
the failure the candidate-set design exists to prevent.

## 2. Greedy versus proven optimal

| k | ILP (proven optimal) | Greedy | Nesting loss | Greedy / optimal |
|---|---|---|---|---|
| 1 | 48.38% | 48.38% | **0.0 pp** | 1.000 |
| 3 | 50.85% | 50.85% | **0.0 pp** | 1.000 |
| 5 | 52.84% | 52.84% | **0.0 pp** | 1.000 |
| 30 | 67.90% | 67.71% | **0.193 pp** | 0.997 |

CBC proved optimality at every budget. Greedy is **exactly optimal at k = 1, 3
and 5**, and at k = 30 it gives up **0.193 pp** — 99.7% of the certified optimum.
So the nested, staged-budget queue is very nearly free, but not literally free at
every budget: a planner funding three devices now and two later lands exactly
where a five-device plan would have put them, while at thirty devices the staged
route costs a fifth of a percentage point of coverage.

Across all four areas the pattern holds and the worst case is small. Bedok is the
only area where greedy is not exactly optimal at small *k* (0.01–0.04 pp), and
the largest shortfall anywhere is **0.309 pp at k = 30**, also Bedok. Jurong East
is exactly optimal at all four budgets. Greedy therefore retains **≥99.3% of the
certified optimum everywhere tested** — comfortably better than the 1−1/e ≈ 63%
worst case the guarantee alone promises.

*An earlier version of this table reported k = 10 as the fourth budget with a
0.0 pp loss. That row no longer exists: the pipeline certifies at
`[1, 3, 5, k]` where `k` is the run budget, which moved from 10 to 30, and the
k = 30 comparison is the one where greedy stops being exactly optimal. The
figures above come from `nesting_loss.csv`.*

*This table previously reported a constant 47.044 pp nesting loss. That was a
defect: greedy was scored on marginal coverage while the ILP objective included
existing coverage, so the "loss" was simply the baseline. Both are now measured
on the same basis.*

## 3. Sensitivity — how recommendations change

### 3.1 Demand proxy (cross-proxy sensitivity, not validation)

Marginal coverage added beyond the existing network. Rows: proxy solved under.
Columns: proxy scored under.

| solved ↓ / scored → | area | area_levels | poi | uniform |
|---|---|---|---|---|
| **area** | 9.86% | 7.75% | 10.81% | 4.97% |
| **area_levels** | 8.31% | 10.54% | 8.47% | 5.18% |
| **poi** | 7.26% | 6.45% | 14.26% | 4.55% |
| **uniform** | 6.52% | 6.51% | 10.47% | 5.59% |

There are no labels, so this measures robustness to an assumption, never
accuracy. Off-diagonal degradation is real but bounded: solving under the wrong
proxy typically costs 2–4 pp of marginal coverage rather than collapsing.

Proxy rank agreement (Spearman): `area`↔`area_levels` **0.729**,
`area`↔`poi` **0.241**, `area_levels`↔`poi` **0.317**.

### 3.2 Retrieval-time budget

| Budget | Threshold | Existing | After 10 |
|---|---|---|---|
| 2 min | 84 m | 30.78% | 38.46% |
| **3 min** *(pre-registered)* | **126 m** | **47.04%** | **56.90%** |
| 4 min | 168 m | 55.28% | 66.76% |

The gain is 7.7–11.5 pp across the sweep, so the *conclusion* survives even though
every absolute number moves substantially with the threshold.

### 3.3 Hours of access

| Scenario | Devices | Existing | After 10 |
|---|---|---|---|
| All hours | 284 | 47.04% | 56.90% |
| **Stated 24/7 only** | **161** | **35.24%** | 46.56% |

Counting only devices that state round-the-clock access costs **11.8 pp**. Even
after ten placements the 24/7 network reaches 46.56% — below what the all-hours
network already achieves without any investment. These are *stated* hours from a
2020 snapshot, not verified access.

## 4. Negative results and failure cases

**The recommendations are not stable, and this is the most important caveat.**
Across nine scenarios (4 demand proxies × 3 thresholds × 2 hours regimes):

- **No site is chosen in ≥90% of scenarios. The consensus set is empty.**
- The most robust sites (49, 83, 388) appear in 67–78% of scenarios.
- **47 distinct sites** are chosen at least once for ten slots.

Presenting a single "optimal 10" without this would misrepresent the available
confidence. Every recommendation therefore ships with its stability share.

**Worst-case distance is untouched** (§1) — the objective cannot fix it.

**Site labels are mostly constructed, not found.** Only 6.3% of candidate
buildings carry an OSM name; 4.3% have a housename and 75.6% are labelled from
their address ("Blk 102B Bidadari Park Drive"). The remaining 13.1% fall through
to a building type plus an id. A constructed label says where a building is, not
what it is called, and `label_source` is reported alongside every recommendation
so the difference stays visible.

**The eligibility rules failed their first calibration.** Validated against the
network SCDF actually built:

| Verdict on the host building | Existing devices | Share |
|---|---|---|
| eligible | 177 | 62.3% |
| no OSM building | 51 | 18.0% |
| abstain | 42 | 14.8% |
| excluded | 16 | 5.6% |

A first version abstained on `building=residential`, giving only 19% recall — it
would have refused the HDB void deck, the most common real AED location in
Singapore. 18% of devices sit in buildings OSM does not contain at all, which is
a data-coverage gap no rule change fixes.

**Fuzzy near-duplicate detection was built and deleted.** It flagged 24 confident
duplicates, all wrong — schools and community clubs hosting several devices, none
sharing an `AED_ID`.

**A dedup bug deleted 33 live devices.** Keying identity on geometry rather than
`AED_ID` discarded distinct devices geocoded to one building centroid. The error
was directional: understating the existing network inflates every recommendation.

## 5. Study-area selection

Toa Payoh was chosen after measuring headroom across 15 planning areas, not
assumed. It is Singapore's **best-covered** area — the hardest case for showing
improvement — and still leaves 56% headroom.

| Area | km² | AEDs | /km² | Coverage @3 min |
|---|---|---|---|---|
| **Toa Payoh** | 8.18 | 284 | 34.7 | **43.6%** |
| Tampines | 20.80 | 691 | 33.2 | 38.8% |
| Ang Mo Kio | 13.94 | 354 | 25.4 | 36.7% |
| Bedok | 21.73 | 469 | 21.6 | 32.6% |
| Jurong East | 17.86 | 243 | 13.6 | 20.4% |

*(Uniform-demand first pass; the demand-weighted figure for Toa Payoh is 47.04%.)*
Jurong East, with 79.6% headroom, is the natural contrast area for a transfer test.

## 6. Reproducibility

Downloaded from data.gov.sg rather than supplied as an event-frozen copy, so
**judges reproducing against a different snapshot may see different numbers.**
The dataset id, retrieval timestamp and SHA256 of every input are recorded in
`data/manifest.json`; `io/aed.py` accepts either file, so substituting an
organiser copy is a one-line change and a re-run.
