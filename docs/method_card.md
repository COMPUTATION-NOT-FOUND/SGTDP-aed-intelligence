# Method card

> **Prototype for planning and simulation only—not for emergency use.**
> In an emergency in Singapore, call **995** immediately and follow SCDF instructions.

**Nominated lane:** Lane 2 — coverage and placement. Techniques are borrowed from
Lanes 1 and 3 where they serve the Lane 2 decision, with organiser confirmation;
Lane 2's metrics and minimum baseline remain the ones being scored.

---

## 1. The decision

| | |
|---|---|
| **User** | A community planner allocating a staged AED budget. |
| **Decision** | Given budget for *k* devices this cycle, which sites, in what order? |
| **Output** | A ranked install queue, each row stating the coverage it adds and why. |
| **Not this tool** | Anything answering "where is the nearest AED to me, now?" |
| **Claim ceiling** | Planning estimates under stated assumptions. Never survival, readiness, or availability. |

## 2. Architecture

```
frozen inputs ─→ H3 grid ─┐
                          ├─→ one-way distance matrix ─→ optimisers ─→ ranked queue
OSM footprints ─→ candidates ─┘                              │
                                                             └─→ scenario sweep ─→ stability
```

**The central commitment: constraints live in the feasible set, never in a
corrector applied afterwards.** Candidates are real OSM building footprints that
pass an eligibility filter, and a site's location is its polygon's
*representative point* — guaranteed inside the building, unlike a centroid, which
falls outside concave and L-shaped footprints. No optimiser can propose a site in
the middle of a road because no such site exists to be chosen.

## 3. Models and solvers

| Component | Method | Guarantee |
|---|---|---|
| Placement (shipped) | Marginal-coverage greedy over a maximal-covering objective, pre-seeded with the existing network | Submodular → within 1−1/e (~63%) |
| Placement (exact) | MCLP as an integer program, PuLP + CBC | Proven optimal, certified |
| Oracle | Brute force over all candidates at k=1 and k=2 | Exact |
| Baseline | Demand-ranked top-k; random-k over 50 seeds; do-nothing | — |
| Equity objective | k-center, minimax and Gonzalez farthest-point | heuristic / 2-approx |
| Average objective | p-median, greedy over the same discrete candidates | heuristic |
| Demonstration | Monte Carlo incidents + random-outage stress, seeded | display only |
| Eligibility | Deterministic OSM tag allowlist/denylist with an abstention band | — |
| Hours | Deterministic rule parser into 5 buckets | 100% parse rate |

**Greedy is the method; demand-ranked top-k is the baseline.** They are named
distinctly throughout because the brief lists greedy placement as an acceptable
baseline, and conflating the two would make the headline appear to beat itself.
Demand-ranked scores each candidate against total demand in isolation, so it
stacks devices into dense cores and re-covers covered ground; marginal-coverage
greedy scores against demand *not already covered*.

## 4. Input features and assumptions

**Distance.** Straight-line separation in EPSG:3414 (SVY21) metres, scaled by a
**1.35 detour factor**, giving one-way network-estimate metres. A walking-network
implementation behind the same `DistanceModel` protocol is the documented upgrade
path; because both return the same unit, swapping it requires no threshold change.

**Threshold.** Derived, not asserted: a **3-minute round-trip** retrieval budget
at 1.4 m/s gives 252 m of walked path, halved for one way → **126 m**. Travel time
only; recognition and cabinet-unlocking time are excluded and documented as
excluded. Round-trip is accounted for in exactly one place, the `/2` in
`config.threshold_m` — nothing else doubles anything.

**Floor penalty.** Existing devices above or below ground level carry +63 m
(45 s lift-and-return × 1.4 m/s). **Candidates carry zero**, because a new install
is specified at ground level. *This asymmetry flatters new installs and is
disclosed here rather than left to be discovered.* 70 of 284 devices (24.6%) are
affected; 10 have unparseable floor levels and are penalised conservatively.

**Demand.** No observed demand exists — the brief forbids incident data and none
is supplied. Four surfaces are computed and all are run:

| Surface | Basis | Imputation |
|---|---|---|
| `area` *(primary)* | building footprint area per cell | **none — fully observed** |
| `area_levels` | footprint area × storeys | 1,660 of 2,492 storey counts imputed |
| `poi` | count of activity tags per cell | none |
| `uniform` | reference case | n/a |

`building:levels` is present on only **33.4%** of footprints. Rather than bury
that behind a single imputed proxy, `area` and `area_levels` are run separately
so the imputation's influence is *measured*: their Spearman agreement is **0.729**.
`area` versus `poi` is only **0.241** — the demand proxy materially drives the
answer, which is reported as a result, not a footnote.

## 4a. Site identity and labels

Only 6.3% of candidate buildings carry an OSM `name`. Labels are resolved by a
ladder — name → housename → `Blk {number} {street}` → `{type} on {street}` →
type + id — and `label_source` is reported with every recommendation, because a
constructed label says where a building is, not what it is called. 75.6% come
from an address, 13.1% fall through to type + id. Every site also carries a
stable id (`TP-041`) that artifacts key on, and `sites.csv` maps every id to a
label and a coordinate.

## 4b. Simulation

Reported metrics are exact analytic weighted quantiles. The Monte Carlo layer
exists for legibility and for stress tests that have no closed form, and
`test_simulation_matches_analytic_quantiles` asserts the sample tracks the exact
figure. Incidents are synthetic, seeded, and labelled as such. Sampling incidents
proportional to a demand proxy is a modelling choice for illustration, not an
epidemiological claim — no individual medical risk is inferred anywhere.

## 5. Human approval points

- **The planner can pin and veto sites, and the model re-solves around them.**
  Local knowledge the model cannot see — a closure, a landlord, a promise — is
  entered as a *constraint on the solve*, never as an edit to a finished plan.
  Pins are honoured first and in order; every subsequent pick is then optimised
  against the state those pins actually create. Layering an override onto a
  completed queue would leave the remainder optimised for a plan that no longer
  exists.
- **The cost of every override is displayed.** The tool re-solves a second time
  without the constraints and reports the difference in coverage, so a planner
  can see the price of their own judgement before accepting it. That price is
  sometimes **negative** — the greedy queue is a heuristic, not an optimum, so a
  human pin occasionally improves it. The tool shows the number either way
  rather than presenting the model as the thing to be defended.
- **Abstained candidate sites (972)** are surfaced as a field-verification queue,
  never silently admitted or dropped.
- **Every recommendation carries a stability share** — how often it survives the
  scenario sweep. Low stability means the answer depends on assumptions.
- **Replacement-cycle guidance only.** Leave-one-out criticality identifies
  redundant devices, but the tool never recommends removing a working device: the
  snapshot is years old, carries no ownership or authorisation data, and
  asserting redundancy would imply knowing a device is present and working.

## 6. Confidence handling

Coverage is binary at the threshold — no distance decay — which overstates the
sharpness of a real response and is why the threshold is swept at 2, 3 and 4
minutes. Uncertainty is reported three ways: the cross-proxy sensitivity matrix,
the threshold sweep, and the per-site stability share. Where a scenario cannot be
resolved, the tool abstains rather than guessing.

## 7. Supplemental-data compliance

| # | Rule | Satisfied by |
|---|---|---|
| 1 | Public, lawfully obtained, licensed | Singapore Open Data Licence ×3, ODbL ×1 — see [`sources.md`](sources.md) |
| 2 | Manifest with URL, publisher, dates, licence, fields, transformations | [`data/manifest.json`](../data/manifest.json) |
| 3 | **Freeze API-derived inputs; reproduce without a live service** | All four inputs frozen and committed; socket-blocking fixture in the test suite; the UI reads only `data/processed/` |
| 4 | Label synthetic data, keep separate, test across scenarios | The only synthetic data is the simulated incident set: seeded, labelled `scenario`, kept in its own artifacts, and used for display and stress testing only — never as a reported metric. A siting-cost term was cut precisely to avoid an undocumented synthetic input |
| 5 | Aggregate demand; no individual risk inference | Demand is an H3-cell aggregate from building geometry; nothing person-level exists |
| 6 | No live SCDF incidents, dispatch, patient or responder data | Nothing beyond the public dataset is touched |

## 8. What is deliberately out of scope

- `AED_LOCATION_DESCRIPTION` free-text parsing (Lane 1's product).
- A siting cost/permission term — it would require a synthetic heuristic tested
  across plausible values, and the priority ranking stands without it. **This
  would be the first term added with more time.**
- Any integration with SCDF, 995, myResponder, or the national AED registry.
- Any claim that a device is present, accessible, inspected, or working.
