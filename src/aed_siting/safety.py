"""The mandatory safety gate.

The challenge brief's safety gate is pass/fail and applied *before* scoring: a
submission that fails it is ineligible regardless of technical quality. The
banner wording is therefore stored once, verbatim, and every user-facing surface
renders this constant rather than its own paraphrase.
"""

from __future__ import annotations

from typing import Final

#: Verbatim from the challenge brief. Do not reword, reflow, or paraphrase.
BANNER: Final[str] = (
    "Prototype for planning and simulation only—not for emergency use. "
    "In an emergency in Singapore, call 995 immediately and follow SCDF "
    "instructions. Use official SCDF/myResponder channels. Do not delay "
    "emergency action to use this prototype."
)

#: Shown at the point of use, not only in the README. The brief requires the
#: dataset date, assumptions, uncertainty and known failure modes to be visible
#: where a recommendation is read.
DISCLOSURES: Final[tuple[str, ...]] = (
    "Source data: SCDF Public Access AEDs, data.gov.sg — underlying registry "
    "snapshot dated February 2020 (page last updated 13 November 2025). This is "
    "a historical snapshot, not a live copy of the national AED registry.",
    "A record in this dataset is not proof that a device is currently present, "
    "accessible, inspected, or working. Nothing here reports device readiness.",
    "Coverage is modelled, not measured: straight-line distance scaled by a 1.35 "
    "detour factor, round-trip, at 1.4 m/s, with travel time only — recognition "
    "and cabinet-unlocking time are excluded.",
    "Operating hours are taken from a free-text field as stated, not as verified "
    "present access. Records whose hours cannot be parsed are excluded and "
    "flagged, never guessed.",
    "Results are planning estimates. They are not predictions of cardiac arrest, "
    "survival, or clinical outcome.",
)

#: The official channels this tool does not replace or integrate with.
OFFICIAL_CHANNELS: Final[str] = (
    "Singapore's official emergency response is SCDF 995 and the myResponder "
    "app, which connects users to 995 and provides access to the national AED "
    "registry. This prototype does not connect to, replace, or compete with "
    "those services, and performs no incident reporting, dispatch, or alerting."
)

#: Required adjacent to any citation of AHA preparedness statistics. The brief is
#: explicit that those facts establish the importance of preparedness and do not
#: license any claim of improved survival.
AHA_CAVEAT: Final[str] = (
    "These facts establish the importance of preparedness; they do not mean this "
    "prototype can predict or claim improved survival."
)


def safe_failure(reason: str) -> str:
    """Render a safe failure state for a case the tool refuses to answer.

    Args:
        reason: Plain-language explanation of what could not be established.

    Returns:
        A message that abstains rather than guessing.

    Limits:
        Abstention is a required behaviour, not a fallback. The brief mandates a
        safe failure state when location, operating hours, routing, or confidence
        is insufficient — a plausible guess in those cases is a gate failure.
    """
    return (
        f"No recommendation: {reason} "
        "This tool abstains rather than presenting an unsupported estimate."
    )
