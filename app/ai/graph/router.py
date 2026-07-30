from typing import Literal

from app.ai.graph.state import (
    ComplaintGraphState,
)


EntryRoute = Literal[
    "extract_complaint",
    "update_from_chat",
]

ProcessingRoute = Literal[
    "continue",
    "build_final_output",
]

ValidationRoute = Literal[
    "classify_complaint",
    "build_final_output",
]


def decide_entry_point(
    state: ComplaintGraphState,
) -> EntryRoute:
    """
    Select the initial workflow route.
    """

    trigger_type = (
        state.get("trigger_type")
        or "TEXT_SUBMISSION"
    )

    if trigger_type == "CHAT_CORRECTION":
        return "update_from_chat"

    return "extract_complaint"


def processing_router(
    state: ComplaintGraphState,
) -> ProcessingRoute:
    """
    Route failed nodes directly to final-output creation.
    """

    if state.get("has_error"):
        return "build_final_output"

    return "continue"


def validation_router(
    state: ComplaintGraphState,
) -> ValidationRoute:
    """
    Continue to classification only when validation succeeds
    and the complaint contains all required information.
    """

    if state.get("has_error"):
        return "build_final_output"

    if state.get(
        "clarification_required",
        False,
    ):
        return "build_final_output"

    if not state.get(
        "is_complete",
        False,
    ):
        return "build_final_output"

    return "classify_complaint"