# Safety and privacy statement

> **Prototype for planning and simulation only—not for emergency use.**
> In an emergency in Singapore, call **995** immediately and follow SCDF
> instructions. Use official SCDF/myResponder channels. Do not delay emergency
> action to use this prototype.

---

## 1. The mandatory banner

The wording above is stored **once**, verbatim, as `aed_siting.safety.BANNER`.
Every user-facing surface renders that constant rather than its own paraphrase:
the first lines of the README, every tab of the Streamlit app, and the demo video.

Enforced by `test_banner_is_verbatim` (character-for-character),
`test_banner_in_readme_first_lines`, `test_banner_on_every_screen` and
`test_app_uses_the_verbatim_banner_constant`, which fails if the app contains the
wording as a literal instead of importing the constant.

## 2. Safety-gate checklist

| # | Requirement | How it is met |
|---|---|---|
| 1 | Only scripted, historical or synthetic scenarios | Every demand cell centroid is a scripted scenario weighted by demand. No incident data of any kind is used. |
| 2 | No live incident reporting, dispatch, alerts, or SCDF/995/myResponder/registry integration | The tool has no network access at runtime. `test_app_reads_only_frozen_artifacts` fails on any reference to an HTTP client or Overpass. |
| 3 | No diagnosis, medical advice, or CPR coaching | The tool outputs building rankings and coverage percentages. It contains no clinical content. |
| 4 | Never label an AED available, accessible, inspected or working | `test_recommendations_never_claim_availability` scans the UI for such phrasing. Every disclosure states a record is not proof of presence. |
| 5 | Identify dataset date, source dates, assumptions, uncertainty and failure modes **at the point of use** | Rendered in the UI's disclosure panel on every run, not only in this file. |
| 6 | Safe failure state when data or confidence is insufficient | `safe_failure()` abstains explicitly. Used for missing artifacts, empty candidate sets, empty demand and unresolvable scenarios. |
| 7 | Synthetic start points in demonstrations | No start points exist — there is no location entry surface at all (§4). |
| 8 | Collect no names, contacts, health, responder or location-history data | The tool collects nothing. It has no input fields, no accounts, no logging of user activity. |
| 9 | Consent, minimisation, protection and deletion of device-location data | Not applicable: no device location is requested, received or stored. |
| 10 | No credentials or private API keys in the prototype, logs, repo or slides | No API requires authentication. `scripts/check_secrets.py` scans the tree; `test_no_secrets_in_repo` gates it. |

## 3. What this tool never claims

- That any AED is currently present, accessible, inspected, or working.
- That any placement changes survival or any clinical outcome.
- That modelled distances describe real walking routes.
- That stated operating hours describe present access.

The source registry is a **February 2020 snapshot**. SCDF states the live national
registry now holds [more than 11,000 devices](https://www.scdf.gov.sg/savealife)
against this dataset's 9,644 — the snapshot is measurably out of date, and this is
disclosed in the UI rather than buried here.

## 4. The Lane 1 boundary, and why there is no lookup

The app includes an explainability panel answering *"why was this site chosen?"*.
It deliberately does **not** include a location box answering *"where is my
nearest AED?"* — the same machinery, but that framing is live emergency guidance
in all but name, which the gate forbids.

The absence is enforced structurally: `test_no_nearest_aed_lookup` parses
`app.py` and fails if any free-entry widget (`st.text_input`, `st.number_input`,
`st.text_area`, `st.chat_input`) appears anywhere in the call graph.

## 5. Official channels this tool does not replace

Singapore's emergency response is **SCDF 995** and the **myResponder** app, which
alerts registered Community First Responders, connects users to 995, and provides
access to the national AED registry. This prototype performs no incident
reporting, dispatch, or alerting, and integrates with none of them. Full citations
in [`sources.md`](sources.md).

## 6. Privacy

No personal data is collected, processed, or stored. There are no user accounts,
no input fields, no telemetry and no cookies beyond what Streamlit requires to
render a session. All inputs are public open data committed to the repository;
none of it identifies a person.

## 7. Known failure modes

| Failure mode | Consequence | Handling |
|---|---|---|
| OSM lacks a building | A real, suitable site is invisible | 18% of existing devices sit in such buildings; reported in the evaluation |
| Detour factor is a global constant | Barriers (canals, expressways) are invisible, so coverage is optimistic where the walking network is worst | Named as the largest modelled-versus-real gap; walking-network upgrade path documented |
| `building:levels` at 33.4% | The levels-based demand proxy is imputation-dominated | Run as a *separate* proxy so its influence is measured, not buried |
| Recommendations unstable across scenarios | A single "optimal 10" would overstate confidence | Consensus set is empty and reported as such; every site carries a stability share |
| Candidates carry no floor penalty | New installs are flattered relative to existing devices | Disclosed in the method card and in the function's own docstring |
| Snapshot is from 2020 | Coverage describes the 2020 network | Quantified against SCDF's current registry size |
