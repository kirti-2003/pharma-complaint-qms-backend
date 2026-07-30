from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.ai.graph.router import (
    decide_entry_point,
    processing_router,
    validation_router,
)
from app.ai.graph.state import (
    ComplaintGraphState,
)
from app.ai.nodes.assess_risk import (
    assess_risk_node,
)
from app.ai.nodes.build_final_output import (
    build_final_output_node,
)
from app.ai.nodes.classify_complaint import (
    classify_complaint_node,
)
from app.ai.nodes.extract_complaint import (
    extract_complaint_node,
)
from app.ai.nodes.update_from_chat import (
    update_from_chat_node,
)
from app.ai.nodes.validate_fields import (
    validate_fields_node,
)


def create_complaint_graph():
    """
    Build and compile the pharmaceutical complaint
    processing LangGraph.
    """

    workflow = StateGraph(
        ComplaintGraphState,
    )

    # --------------------------------------------------
    # Nodes
    # --------------------------------------------------

    workflow.add_node(
        "extract_complaint",
        extract_complaint_node,
    )

    workflow.add_node(
        "update_from_chat",
        update_from_chat_node,
    )

    workflow.add_node(
        "validate_fields",
        validate_fields_node,
    )

    workflow.add_node(
        "classify_complaint",
        classify_complaint_node,
    )

    workflow.add_node(
        "assess_risk",
        assess_risk_node,
    )

    workflow.add_node(
        "build_final_output",
        build_final_output_node,
    )

    # --------------------------------------------------
    # Entry routing
    # --------------------------------------------------

    workflow.add_conditional_edges(
        START,
        decide_entry_point,
        {
            "extract_complaint":
                "extract_complaint",
            "update_from_chat":
                "update_from_chat",
        },
    )

    # --------------------------------------------------
    # Extraction routing
    # --------------------------------------------------

    workflow.add_conditional_edges(
        "extract_complaint",
        processing_router,
        {
            "continue":
                "validate_fields",
            "build_final_output":
                "build_final_output",
        },
    )

    # --------------------------------------------------
    # Chat update routing
    # --------------------------------------------------

    workflow.add_conditional_edges(
        "update_from_chat",
        processing_router,
        {
            "continue":
                "validate_fields",
            "build_final_output":
                "build_final_output",
        },
    )

    # --------------------------------------------------
    # Validation routing
    # --------------------------------------------------

    workflow.add_conditional_edges(
        "validate_fields",
        validation_router,
        {
            "classify_complaint":
                "classify_complaint",
            "build_final_output":
                "build_final_output",
        },
    )

    # --------------------------------------------------
    # Classification routing
    # --------------------------------------------------

    workflow.add_conditional_edges(
        "classify_complaint",
        processing_router,
        {
            "continue":
                "assess_risk",
            "build_final_output":
                "build_final_output",
        },
    )

    # --------------------------------------------------
    # Risk routing
    # --------------------------------------------------

    workflow.add_conditional_edges(
        "assess_risk",
        processing_router,
        {
            "continue":
                "build_final_output",
            "build_final_output":
                "build_final_output",
        },
    )

    workflow.add_edge(
        "build_final_output",
        END,
    )

    return workflow.compile()


complaint_graph = (
    create_complaint_graph()
)