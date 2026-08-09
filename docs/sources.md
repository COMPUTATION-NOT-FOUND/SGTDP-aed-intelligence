# Sources and official resources

Two registers. The first lists the data this tool consumes; the second lists the
challenge brief's official resources and the role each plays here.

---

## 1. Data sources

Every input is frozen to `data/raw/` and checksummed in
[`data/manifest.json`](../data/manifest.json). Nothing after
`scripts/fetch_sources.py` touches a network.

| Source | Dataset ID | Publisher | Licence | Dates | Fields used |
|---|---|---|---|---|---|
| [Public Access AEDs](https://data.gov.sg/datasets/d_4e6b82c58a8a832f6f1fee5dfa6d47ea/view) | `d_4e6b82c58a8a832f6f1fee5dfa6d47ea` | SCDF | Singapore Open Data Licence | data **Feb 2020**; page updated **13 Nov 2025** | `AED_ID`, `OPERATING_HOURS`, `POSTAL_CODE`, `BUILDING_NAME`, `AED_LOCATION_FLOOR_LEVEL`, `LATITUDE`, `LONGITUDE` |
| [MP2019 Planning Area Boundary (No Sea)](https://data.gov.sg/datasets/d_4765db0e87b9c86336792efe8a1f7a66/view) | `d_4765db0e87b9c86336792efe8a1f7a66` | URA | Singapore Open Data Licence | 2019; page updated 5 Dec 2025 | `PLN_AREA_N`, geometry |
| [MP2019 Subzone Boundary (No Sea)](https://data.gov.sg/datasets/d_8594ae9ff96d0c708bc2af633048edfb/view) | `d_8594ae9ff96d0c708bc2af633048edfb` | URA | Singapore Open Data Licence | 2019 | `PLN_AREA_N`, `SUBZONE_N`, geometry |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) buildings + POIs | — | OpenStreetMap contributors | **ODbL 1.0** — attribution **and** share-alike | frozen Overpass extract; OSM is **unversioned** | `building`, `building:levels`, `amenity`, `shop`, `public_transport`, `name`, geometry |

**Fields deliberately not used.** `XVAL`, `YVAL`, `INC_CRC` and `FMEL_UPD_D` have
no authoritative data dictionary, and the brief forbids assigning them
operational meaning. They are dropped inside the loader so nothing downstream can
consume them. `AED_LOCATION_DESCRIPTION` is free text belonging to challenge
Lane 1 and is not parsed.

**ODbL obligations.** OpenStreetMap data is used under the Open Database Licence.
Attribution is given here and in the README; share-alike is satisfied by this
repository being public with the derived extract committed.

---

## 2. Official resources from the challenge brief

The brief lists six. Each has a defined role; none is decorative.

### [Public Access AEDs — SCDF, data.gov.sg](https://data.gov.sg/datasets/d_4e6b82c58a8a832f6f1fee5dfa6d47ea/view)

The primary dataset — see register 1. Treated as a **historical registry
snapshot**, not a live copy of the national registry.

### [myResponder app — Singapore Civil Defence Force](https://www.scdf.gov.sg/home/community-and-volunteers/community-resources/myresponder-app)

SCDF's app alerts registered Community First Responders to nearby medical
emergencies, connects users to 995, and shows AED locations. **This prototype
does not connect to, replace, duplicate, or compete with myResponder.** It has no
incident reporting, no dispatch, no alerting, and no responder-facing surface. A
planner using this tool is making a budget decision; a member of the public in an
emergency should be using myResponder and 995.

### [myResponder and the national AED registry — Singapore Civil Defence Force](https://www.scdf.gov.sg/savealife)

The national registry of publicly accessible AEDs is maintained by SCDF together
with the Singapore Heart Foundation, and SCDF states it now holds **more than
11,000** registered AEDs.

That figure is load-bearing for this submission's honesty. The dataset used here
contains **9,644** records from a **February 2020** snapshot — materially smaller
than the live registry. Any coverage number this tool produces therefore
describes the 2020 network, not today's. This is quantified evidence of staleness
rather than a generic caveat, and it appears in the UI's point-of-use disclosures.
**No integration with the national registry is performed or attempted** — the
brief prohibits it.

### [Emergency Medical Services and calling 995 — Singapore Civil Defence Force](https://www.scdf.gov.sg/home/about-scdf/emergency-medical-services)

The basis of the mandated safety banner. SCDF's guidance is that 995 is for
life-threatening emergencies, cardiac arrest among them. The banner reproduced
verbatim on every user-facing surface directs anyone in an emergency to call 995
immediately and follow SCDF instructions, and warns against delaying emergency
action to use this prototype.

### [Adult Basic Life Support — American Heart Association](https://cpr.heart.org/en/resuscitation-science/cpr-and-ecc-guidelines/adult-basic-life-support)

Frames the problem: AHA adult basic life support guidance emphasises prompt
emergency activation, high-quality CPR, and early defibrillation. Early
defibrillation is why AED *proximity* is worth planning for at all.

> These facts establish the importance of preparedness; they **do not mean** that
> this prototype can predict or claim improved survival.

### [CPR facts and statistics — American Heart Association](https://cpr.heart.org/en/resources/cpr-facts-and-stats)

Also frames the problem: the AHA notes that immediate CPR can double or triple
the chance of survival after cardiac arrest, and that out-of-hospital cardiac
arrest most often occurs in residential settings.

> These facts establish the importance of preparedness; they **do not mean** that
> this prototype can predict or claim improved survival.

**Why the caveat is repeated rather than stated once.** The challenge brief is
explicit that these statistics justify preparedness and license no claim about
outcomes. Repeating it adjacent to each citation is enforced by
`test_aha_citations_carry_caveat`, so a future edit cannot separate a statistic
from its qualifier.

---

## 3. What this tool never claims

- That any AED is currently present, accessible, inspected, or working.
- That any placement changes survival, or that any coverage figure is a clinical outcome.
- That its modelled distances describe real walking routes, or that stated
  operating hours describe present access.

Every output is a **planning estimate under the assumptions recorded in
[`primary_metric.md`](primary_metric.md) and the method card.**
