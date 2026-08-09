# Registry data quality

> **Prototype for planning and simulation only—not for emergency use.**
> In an emergency in Singapore, call **995** immediately and follow SCDF instructions.

Techniques borrowed from challenge Lane 3, kept only where they serve the Lane 2
placement decision. Every figure below is produced by the pipeline, not by hand.

---

## 1. Findings

| Check | Result | Why it matters to placement |
|---|---|---|
| Postal codes repaired | **456 of 9,644** left-padded to six characters | Address integrity; see §2 |
| Cross-field postal agreement | **9,644 / 9,644 (100%)** | Confirms the repair is exact |
| `AED_ID` uniqueness | 9,644 distinct of 9,644 | Establishes the device identity key |
| Records with unusable coordinates | **0** | No geometry is guessed |
| Operating-hours parse rate | **100%**, 0 unparseable | Enables the hours-of-access scenario |
| Floor levels unparseable | **10 of 284** | Penalised conservatively and flagged |
| Co-located devices | **57 records across 24 coordinates**, largest holds 9 | Affects catchment load and redundancy |
| Candidate sites needing field verification | **972 of 2,492 buildings (39%)** | Abstained, never silently admitted |

## 2. The leading-zero problem, and why it is *solved* rather than mitigated

The challenge brief warns that address-like numeric fields must be handled so
leading zeroes are not lost. In this dataset **the loss had already happened
upstream**: the published legend types `POSTAL_CODE` as Numeric, and Singapore
postal sectors 01–09 exist, so 456 six-digit codes were serialised as five
characters before publication. Careful reading cannot recover what the file no
longer contains.

It is recoverable anyway, because **`AED_ID` embeds the postal code**: the format
is `<6-digit postal>-<sequence>`, e.g. `098136-004`. That gives every row a
second, independent copy of the same value — and it is exactly the field that did
not lose its zero.

Padding `POSTAL_CODE` to six characters and comparing against the `AED_ID` prefix
gives **100% agreement across all 9,644 records, with zero unresolved
mismatches**, and confirms all 456 repairs individually. The repair is therefore a
**verified reconstruction, not an inference** — a stronger claim than this project
originally made, and the check is capable of failing
(`test_postal_validation_detects_a_planted_mismatch`).

## 3. What was built and then deleted

**Fuzzy near-duplicate detection.** Proximity blocking plus `difflib` name
similarity plus postal match, with a three-way duplicate / review / distinct
banding and an abstention region. It flagged 74 candidate pairs and confidently
called 24 of them duplicates.

**All 24 were wrong.** They were schools and community clubs legitimately hosting
several devices — Maris Stella High School 6, Bartley Secondary 5, Stamford
American 5 — and **none shared an `AED_ID`**. Name-plus-postal identifies a *site*,
not a *device*. With an authoritative unique key present, fuzzy matching has
nothing to add and real potential to subtract.

It was deleted rather than tuned. The negative result is more useful than the
feature would have been: **this registry contains no detectable duplication.**

## 4. A defect this found in our own pipeline

The first implementation keyed device identity on **geometry**, dropping 33
records that shared a coordinate as duplicates. They were not duplicates:
`310480-002` sits on level 2 and `310480-003` on level 1 of HDB Hub — two distinct
registered devices geocoded to one building centroid.

The corrected in-town count is **284, not 251**. The error was directional and
self-serving: understating the existing network would have inflated the apparent
value of every placement the optimiser recommends. Identity now comes from
`AED_ID`, and `test_colocated_records_are_retained_not_dropped` prevents a
regression.

## 5. Fields deliberately not used

`XVAL`, `YVAL`, `INC_CRC` and `FMEL_UPD_D` have no authoritative data dictionary,
and the brief forbids assigning them operational meaning. They are **dropped
inside the loader** so nothing downstream can consume them even accidentally.

`XVAL`/`YVAL` were nevertheless *verified* to be EPSG:3414: reprojecting
`LATITUDE`/`LONGITUDE` and differencing gives a median offset of **0.083 m** over
3,000 records. That is recorded as a finding only — all geometry is still derived
from `LATITUDE`/`LONGITUDE`.

`AED_LOCATION_DESCRIPTION` is free text belonging to Lane 1 and is not parsed.

## 6. Review queue

Two queues are produced for a human reviewer, both ranked:

1. **972 candidate buildings** whose OSM tags do not establish public access
   (`building=yes`, dormitory, construction). Excluded from optimisation and
   flagged "needs field verification" rather than guessed either way.
2. **Existing devices by leave-one-out criticality** (`criticality.csv`), ranking
   how much demand loses its only cover if each device disappears. This informs
   the **replacement cycle** — it is not a recommendation to remove any working
   device, which this dataset cannot support.
