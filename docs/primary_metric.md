# Primary metric — nominated before any optimiser exists

The challenge brief requires teams to *"nominate one primary metric before final
evaluation."* This file is committed in Phase 1, before any optimiser, scenario
sweep, or result exists, so the nomination is verifiable in git history rather
than asserted after the fact.

## Primary (effectiveness)

**Demand-weighted coverage: the share of modelled demand lying within the
declared round-trip retrieval threshold of an AED.**

- **Declared threshold:** a **3-minute round-trip** retrieval budget.
- Derived, not asserted: `3 min × 60 × 1.4 m/s = 252 m` of walked path, halved
  for one way → **126 m**, compared directly against a distance matrix that is
  already stored in one-way network-estimate metres.
- Travel time only. Recognition and cabinet-unlocking time are excluded and
  documented as excluded.

The threshold is **swept at 2, 3 and 4 minutes** and all three are reported
regardless of which flatters the result. Pre-registering 3 minutes here is what
prevents the headline from being chosen after seeing the numbers.

## Secondary metrics

The brief additionally requires at least one safety/error metric and one
performance or usability metric, and rejects averages alone.

| Class | Metric |
|---|---|
| Effectiveness | Marginal coverage per added AED; cumulative coverage curve |
| Effectiveness (tail) | Median **and p90** modelled retrieval distance |
| Equity | Worst-served subzone coverage; maximum retrieval distance; catchment load per existing AED |
| Safety / error | Operating-hours parse rate; candidate abstention rate; count of records excluded rather than guessed |
| Performance | Solve time; end-to-end pipeline runtime |
| Robustness | Share of recommended sites stable across demand proxies and thresholds |

## What this metric is not

It is a **planning estimate under stated assumptions**. It is not a measurement
of real-world access, not evidence that any device is present or working, and not
a prediction of cardiac arrest incidence, survival, or clinical outcome.

Coverage is computed on modelled distance, on a February 2020 registry snapshot,
over a demand surface built from proxies rather than observed demand. Every one
of those is a source of error that the sensitivity analysis is designed to expose
rather than hide.
