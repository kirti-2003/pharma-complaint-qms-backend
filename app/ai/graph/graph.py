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

from app.ai.nodes.assess_complaint import (
    assess_complaint_node,
)
from app.ai.nodes.build_final_output import (
    build_final_output_node,
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

    Optimized workflow:
    - One Groq call for extraction
    - One Groq call for classification and risk assessment
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
        "assess_complaint",
        assess_complaint_node,
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
            "assess_complaint":
                "assess_complaint",
            "build_final_output":
                "build_final_output",
        },
    )

    # --------------------------------------------------
    # Combined classification and risk routing
    # --------------------------------------------------

    workflow.add_conditional_edges(
        "assess_complaint",
        processing_router,
        {
            "continue":
                "build_final_output",
            "build_final_output":
                "build_final_output",
        },
    )

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    workflow.add_edge(
        "build_final_output",
        END,
    )

    return workflow.compile()


complaint_graph = create_complaint_graph()