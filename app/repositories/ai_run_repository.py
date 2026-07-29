from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.complaint_ai_run import ComplaintAIRun
from app.schemas.ai_analysis import AIRunCreate, AIRunUpdate


class AIRunRepository:
    """
    Handles database operations related to complaint AI runs.

    Each LangGraph execution creates a separate AI run record.
    This repository stores run status, model information, extracted
    fields, classification results, risk assessment, final output,
    token usage, and processing errors.
    """

    def create_ai_run(
        self,
        db: Session,
        ai_run_data: AIRunCreate,
    ) -> ComplaintAIRun:
        """
        Create a new AI run record.
        """

        ai_run = ComplaintAIRun(
            **ai_run_data.model_dump(exclude_unset=True)
        )

        db.add(ai_run)
        db.flush()
        db.refresh(ai_run)

        return ai_run

    def get_ai_run_by_id(
        self,
        db: Session,
        ai_run_id: UUID,
    ) -> Optional[ComplaintAIRun]:
        """
        Retrieve an AI run using its UUID.
        """

        return (
            db.query(ComplaintAIRun)
            .filter(
                ComplaintAIRun.ai_run_id == ai_run_id
            )
            .first()
        )

    def get_ai_runs_by_complaint_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> List[ComplaintAIRun]:
        """
        Retrieve all AI runs belonging to a complaint.

        The latest AI run is returned first.
        """

        return (
            db.query(ComplaintAIRun)
            .filter(
                ComplaintAIRun.complaint_id == complaint_id
            )
            .order_by(
                ComplaintAIRun.started_at.desc()
            )
            .all()
        )

    def get_latest_ai_run(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> Optional[ComplaintAIRun]:
        """
        Retrieve the most recent AI run for a complaint.
        """

        return (
            db.query(ComplaintAIRun)
            .filter(
                ComplaintAIRun.complaint_id == complaint_id
            )
            .order_by(
                ComplaintAIRun.started_at.desc()
            )
            .first()
        )

    def get_ai_runs_by_status(
        self,
        db: Session,
        run_status: str,
        skip: int = 0,
        limit: int = 50,
    ) -> List[ComplaintAIRun]:
        """
        Retrieve AI runs using their current run status.

        Supported statuses:

        STARTED
        PROCESSING
        WAITING_FOR_USER
        COMPLETED
        FAILED
        """

        return (
            db.query(ComplaintAIRun)
            .filter(
                ComplaintAIRun.run_status == run_status
            )
            .order_by(
                ComplaintAIRun.started_at.desc()
            )
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_ai_runs_by_trigger_type(
        self,
        db: Session,
        complaint_id: UUID,
        trigger_type: str,
    ) -> List[ComplaintAIRun]:
        """
        Retrieve AI runs for a specific trigger type.

        Supported trigger types:

        TEXT_SUBMISSION
        FILE_UPLOAD
        CHAT_CORRECTION
        REANALYSIS
        """

        return (
            db.query(ComplaintAIRun)
            .filter(
                ComplaintAIRun.complaint_id == complaint_id,
                ComplaintAIRun.trigger_type == trigger_type,
            )
            .order_by(
                ComplaintAIRun.started_at.desc()
            )
            .all()
        )

    def get_ai_run_by_langgraph_run_id(
        self,
        db: Session,
        langgraph_run_id: str,
    ) -> Optional[ComplaintAIRun]:
        """
        Retrieve an AI run using its LangGraph run ID.
        """

        return (
            db.query(ComplaintAIRun)
            .filter(
                ComplaintAIRun.langgraph_run_id == langgraph_run_id
            )
            .first()
        )

    def get_ai_runs_by_langgraph_thread_id(
        self,
        db: Session,
        langgraph_thread_id: str,
    ) -> List[ComplaintAIRun]:
        """
        Retrieve all AI runs belonging to a LangGraph thread.
        """

        return (
            db.query(ComplaintAIRun)
            .filter(
                ComplaintAIRun.langgraph_thread_id
                == langgraph_thread_id
            )
            .order_by(
                ComplaintAIRun.started_at.asc()
            )
            .all()
        )

    def update_ai_run(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        ai_run_data: AIRunUpdate,
    ) -> ComplaintAIRun:
        """
        Update only the AI run fields provided by the caller.
        """

        update_data = ai_run_data.model_dump(
            exclude_unset=True
        )

        for field_name, field_value in update_data.items():
            if hasattr(ai_run, field_name):
                setattr(
                    ai_run,
                    field_name,
                    field_value,
                )

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def update_langgraph_details(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        langgraph_thread_id: Optional[str] = None,
        langgraph_run_id: Optional[str] = None,
    ) -> ComplaintAIRun:
        """
        Save LangGraph thread and run identifiers.
        """

        if langgraph_thread_id is not None:
            ai_run.langgraph_thread_id = langgraph_thread_id

        if langgraph_run_id is not None:
            ai_run.langgraph_run_id = langgraph_run_id

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def mark_ai_run_as_processing(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
    ) -> ComplaintAIRun:
        """
        Mark an AI run as currently processing.
        """

        ai_run.run_status = "PROCESSING"
        ai_run.error_message = None

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def mark_ai_run_as_waiting_for_user(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        missing_fields: Optional[dict] = None,
        final_output: Optional[dict] = None,
    ) -> ComplaintAIRun:
        """
        Mark an AI run as waiting for additional information
        or clarification from the user.
        """

        ai_run.run_status = "WAITING_FOR_USER"

        if missing_fields is not None:
            ai_run.missing_fields = missing_fields

        if final_output is not None:
            ai_run.final_output = final_output

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def update_input_payload(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        input_payload: dict,
    ) -> ComplaintAIRun:
        """
        Store the input sent to the LangGraph workflow.
        """

        ai_run.input_payload = input_payload

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def update_extracted_fields(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        extracted_fields: dict,
    ) -> ComplaintAIRun:
        """
        Store structured complaint fields extracted by the AI.
        """

        ai_run.extracted_fields = extracted_fields

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def update_missing_fields(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        missing_fields: dict,
    ) -> ComplaintAIRun:
        """
        Store required complaint fields that are still missing.
        """

        ai_run.missing_fields = missing_fields

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def update_classification_result(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        classification_result: dict,
    ) -> ComplaintAIRun:
        """
        Store the AI-generated complaint classification result.
        """

        ai_run.classification_result = classification_result

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def update_risk_assessment_result(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        risk_assessment_result: dict,
    ) -> ComplaintAIRun:
        """
        Store the AI-generated risk assessment result.
        """

        ai_run.risk_assessment_result = risk_assessment_result

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def update_final_output(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        final_output: dict,
    ) -> ComplaintAIRun:
        """
        Store the final structured output generated by LangGraph.
        """

        ai_run.final_output = final_output

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def update_token_usage(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> ComplaintAIRun:
        """
        Update model token usage for an AI run.
        """

        if prompt_tokens is not None:
            ai_run.prompt_tokens = prompt_tokens

        if completion_tokens is not None:
            ai_run.completion_tokens = completion_tokens

        if total_tokens is not None:
            ai_run.total_tokens = total_tokens

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def mark_ai_run_as_completed(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        extracted_fields: Optional[dict] = None,
        missing_fields: Optional[dict] = None,
        classification_result: Optional[dict] = None,
        risk_assessment_result: Optional[dict] = None,
        final_output: Optional[dict] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> ComplaintAIRun:
        """
        Mark an AI run as completed and save its final results.
        """

        ai_run.run_status = "COMPLETED"
        ai_run.completed_at = datetime.now(timezone.utc)
        ai_run.error_message = None

        if extracted_fields is not None:
            ai_run.extracted_fields = extracted_fields

        if missing_fields is not None:
            ai_run.missing_fields = missing_fields

        if classification_result is not None:
            ai_run.classification_result = classification_result

        if risk_assessment_result is not None:
            ai_run.risk_assessment_result = risk_assessment_result

        if final_output is not None:
            ai_run.final_output = final_output

        if prompt_tokens is not None:
            ai_run.prompt_tokens = prompt_tokens

        if completion_tokens is not None:
            ai_run.completion_tokens = completion_tokens

        if total_tokens is not None:
            ai_run.total_tokens = total_tokens

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def mark_ai_run_as_failed(
        self,
        db: Session,
        ai_run: ComplaintAIRun,
        error_message: str,
    ) -> ComplaintAIRun:
        """
        Mark an AI run as failed and save the error message.
        """

        ai_run.run_status = "FAILED"
        ai_run.error_message = error_message
        ai_run.completed_at = datetime.now(timezone.utc)

        db.flush()
        db.refresh(ai_run)

        return ai_run

    def count_ai_runs_by_complaint_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> int:
        """
        Count AI runs associated with a complaint.
        """

        return (
            db.query(ComplaintAIRun)
            .filter(
                ComplaintAIRun.complaint_id == complaint_id
            )
            .count()
        )

    def ai_run_belongs_to_complaint(
        self,
        db: Session,
        ai_run_id: UUID,
        complaint_id: UUID,
    ) -> bool:
        """
        Check whether an AI run belongs to a specific complaint.
        """

        ai_run = (
            db.query(ComplaintAIRun.ai_run_id)
            .filter(
                ComplaintAIRun.ai_run_id == ai_run_id,
                ComplaintAIRun.complaint_id == complaint_id,
            )
            .first()
        )

        return ai_run is not None