from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.graph import (
    complaint_graph,
    create_initial_complaint_state,
)
from app.core.config import settings
from app.models.complaint import Complaint
from app.models.complaint_ai_run import ComplaintAIRun
from app.repositories.ai_run_repository import AIRunRepository
from app.repositories.audit_repository import AuditRepository
from app.repositories.complaint_repository import ComplaintRepository
from app.repositories.field_change_repository import (
    FieldChangeRepository,
)
from app.repositories.message_repository import MessageRepository
from app.schemas.ai_analysis import AIRunCreate


class AIService:
    """
    Coordinate LangGraph complaint processing and persistence.

    Repositories only flush and refresh. This service owns commit and
    rollback for the complete AI workflow transaction.
    """

    GRAPH_TO_COMPLAINT_FIELD_MAP: dict[str, str] = {
        "complainant_name": "customer_name",
        "product_name": "product_name",
        "product_strength_grade": "product_strength_grade",
        "batch_lot_number": "batch_lot_number",
        "quantity_affected": "affected_quantity_text",
        "manufacturing_date": "manufacturing_date_text",
        "expiry_date": "expiry_date_text",
        "complaint_description": "complaint_description",
    }

    def __init__(self) -> None:
        self.complaint_repository = ComplaintRepository()
        self.ai_run_repository = AIRunRepository()
        self.message_repository = MessageRepository()
        self.field_change_repository = FieldChangeRepository()
        self.audit_repository = AuditRepository()

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_ai_run(
        self,
        db: Session,
        ai_run_id: UUID,
    ) -> ComplaintAIRun:
        ai_run = self.ai_run_repository.get_ai_run_by_id(
            db=db,
            ai_run_id=ai_run_id,
        )

        if ai_run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI run not found.",
            )

        return ai_run

    def get_complaint_ai_runs(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> list[ComplaintAIRun]:
        self._get_complaint_or_404(
            db=db,
            complaint_id=complaint_id,
        )

        return self.ai_run_repository.get_ai_runs_by_complaint_id(
            db=db,
            complaint_id=complaint_id,
        )

    # ------------------------------------------------------------------
    # Initial complaint processing
    # ------------------------------------------------------------------

    def process_complaint(
        self,
        db: Session,
        complaint_id: UUID,
        trigger_type: str = "TEXT_SUBMISSION",
    ) -> ComplaintAIRun:
        """
        Process an existing complaint through the complete AI graph.

        Supported trigger types for this method are TEXT_SUBMISSION,
        FILE_UPLOAD, and REANALYSIS.
        """

        allowed_triggers = {
            "TEXT_SUBMISSION",
            "FILE_UPLOAD",
            "REANALYSIS",
        }

        if trigger_type not in allowed_triggers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid AI processing trigger type.",
            )

        complaint = self._get_editable_complaint(
            db=db,
            complaint_id=complaint_id,
        )

        raw_text = (
            complaint.raw_complaint_text
            or ""
        ).strip()

        if not raw_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Complaint text is required before AI processing."
                ),
            )

        previous_status = complaint.status

        try:
            ai_run = self._create_ai_run(
                db=db,
                complaint=complaint,
                trigger_type=trigger_type,
                input_payload={
                    "raw_text": raw_text,
                    "input_type": complaint.input_type,
                },
            )

            self.complaint_repository.update_complaint_status(
                db=db,
                complaint=complaint,
                new_status="PROCESSING",
            )

            self.audit_repository.create_ai_processing_started_log(
                db=db,
                complaint_id=complaint.complaint_id,
                ai_run_id=ai_run.ai_run_id,
                trigger_type=trigger_type,
            )

            graph_state = create_initial_complaint_state(
                complaint_id=str(complaint.complaint_id),
                ai_run_id=str(ai_run.ai_run_id),
                trigger_type=trigger_type,
                input_type=complaint.input_type,
                raw_text=raw_text,
            )

            result = complaint_graph.invoke(
                graph_state
            )

            self._persist_graph_result(
                db=db,
                complaint=complaint,
                ai_run=ai_run,
                result=result,
                previous_status=previous_status,
                change_source="INITIAL_EXTRACTION",
                message_id=None,
            )

            db.commit()
            db.refresh(ai_run)
            db.refresh(complaint)

            return ai_run

        except HTTPException:
            db.rollback()
            raise

        except SQLAlchemyError as exc:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "A database error occurred while saving "
                    "the AI complaint analysis."
                ),
            ) from exc

        except Exception as exc:
            return self._handle_unexpected_processing_error(
                db=db,
                complaint=complaint,
                previous_status=previous_status,
                error=exc,
                ai_run=locals().get("ai_run"),
            )

    # ------------------------------------------------------------------
    # Chat correction processing
    # ------------------------------------------------------------------

    def process_chat_correction(
        self,
        db: Session,
        complaint_id: UUID,
        message_text: str,
    ) -> ComplaintAIRun:
        """
        Apply a user chat correction and rerun validation,
        classification, risk assessment, and final-output creation.
        """

        complaint = self._get_editable_complaint(
            db=db,
            complaint_id=complaint_id,
        )

        cleaned_message = message_text.strip()

        if not cleaned_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Chat message cannot be empty.",
            )

        existing_fields = self._get_existing_graph_fields(
            db=db,
            complaint=complaint,
        )

        if not existing_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "The complaint must be processed at least once "
                    "before chat corrections can be applied."
                ),
            )

        previous_status = complaint.status

        try:
            user_message = self.message_repository.create_user_message(
                db=db,
                complaint_id=complaint.complaint_id,
                message_text=cleaned_message,
                message_metadata={
                    "workflow": "CHAT_CORRECTION",
                },
            )

            ai_run = self._create_ai_run(
                db=db,
                complaint=complaint,
                trigger_type="CHAT_CORRECTION",
                input_payload={
                    "chat_message": cleaned_message,
                    "existing_fields": existing_fields,
                    "message_id": str(user_message.message_id),
                },
            )

            self.complaint_repository.update_complaint_status(
                db=db,
                complaint=complaint,
                new_status="PROCESSING",
            )

            self.audit_repository.create_ai_processing_started_log(
                db=db,
                complaint_id=complaint.complaint_id,
                ai_run_id=ai_run.ai_run_id,
                trigger_type="CHAT_CORRECTION",
            )

            graph_state = create_initial_complaint_state(
                complaint_id=str(complaint.complaint_id),
                ai_run_id=str(ai_run.ai_run_id),
                trigger_type="CHAT_CORRECTION",
                input_type=complaint.input_type,
                raw_text=complaint.raw_complaint_text or "",
                chat_message=cleaned_message,
                existing_fields=existing_fields,
            )

            result = complaint_graph.invoke(
                graph_state
            )

            self._persist_graph_result(
                db=db,
                complaint=complaint,
                ai_run=ai_run,
                result=result,
                previous_status=previous_status,
                change_source="CHAT_CORRECTION",
                message_id=user_message.message_id,
            )

            db.commit()
            db.refresh(ai_run)
            db.refresh(complaint)

            return ai_run

        except HTTPException:
            db.rollback()
            raise

        except SQLAlchemyError as exc:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "A database error occurred while saving "
                    "the AI chat correction."
                ),
            ) from exc

        except Exception as exc:
            return self._handle_unexpected_processing_error(
                db=db,
                complaint=complaint,
                previous_status=previous_status,
                error=exc,
                ai_run=locals().get("ai_run"),
            )

    # ------------------------------------------------------------------
    # Workflow persistence
    # ------------------------------------------------------------------

    def _persist_graph_result(
        self,
        db: Session,
        complaint: Complaint,
        ai_run: ComplaintAIRun,
        result: dict[str, Any],
        previous_status: str,
        change_source: str,
        message_id: UUID | None,
    ) -> None:
        processing_status = result.get(
            "processing_status",
            "FAILED",
        )

        extracted_fields = result.get(
            "extracted_fields",
            {},
        ) or {}

        classification_result = result.get(
            "classification_result",
            {},
        ) or {}

        risk_assessment_result = result.get(
            "risk_assessment_result",
            {},
        ) or {}

        missing_fields = result.get(
            "missing_fields",
            [],
        ) or []

        final_output = result.get(
            "final_output",
            {},
        ) or {}

        self.ai_run_repository.update_langgraph_details(
            db=db,
            ai_run=ai_run,
            langgraph_thread_id=result.get(
                "langgraph_thread_id"
            ),
            langgraph_run_id=result.get(
                "langgraph_run_id"
            ),
        )

        complaint_updates = self._build_complaint_updates(
            extracted_fields=extracted_fields,
            classification_result=classification_result,
            risk_assessment_result=risk_assessment_result,
        )

        changes = self._build_field_changes(
            complaint=complaint,
            update_data=complaint_updates,
        )

        if complaint_updates:
            self.complaint_repository.update_complaint_fields(
                db=db,
                complaint=complaint,
                update_data=complaint_updates,
            )

        assistant_message = (
            result.get("assistant_message")
            or final_output.get("assistant_message")
            or "AI complaint processing finished."
        )

        assistant_db_message = self._create_result_message(
            db=db,
            complaint=complaint,
            ai_run=ai_run,
            result=result,
            assistant_message=assistant_message,
            processing_status=processing_status,
        )

        if changes:
            self.field_change_repository.create_multiple_field_changes(
                db=db,
                complaint_id=complaint.complaint_id,
                changes=changes,
                changed_by="AI",
                change_source=change_source,
                ai_run_id=ai_run.ai_run_id,
                message_id=(
                    message_id
                    or assistant_db_message.message_id
                ),
            )

        if processing_status == "COMPLETED":
            self.ai_run_repository.mark_ai_run_as_completed(
                db=db,
                ai_run=ai_run,
                extracted_fields=extracted_fields,
                missing_fields=missing_fields,
                classification_result=classification_result,
                risk_assessment_result=risk_assessment_result,
                final_output=final_output,
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get(
                    "completion_tokens",
                    0,
                ),
                total_tokens=result.get("total_tokens", 0),
            )

            new_complaint_status = "READY_TO_COMMIT"

            self.audit_repository.create_ai_processing_completed_log(
                db=db,
                complaint_id=complaint.complaint_id,
                ai_run_id=ai_run.ai_run_id,
                audit_metadata={
                    "processing_status": processing_status,
                    "model_name": result.get("model_name"),
                    "total_tokens": result.get("total_tokens", 0),
                },
            )

        elif processing_status == "WAITING_FOR_USER":
            self.ai_run_repository.mark_ai_run_as_waiting_for_user(
                db=db,
                ai_run=ai_run,
                missing_fields=missing_fields,
                final_output=final_output,
            )

            self.ai_run_repository.update_extracted_fields(
                db=db,
                ai_run=ai_run,
                extracted_fields=extracted_fields,
            )

            self.ai_run_repository.update_token_usage(
                db=db,
                ai_run=ai_run,
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get(
                    "completion_tokens",
                    0,
                ),
                total_tokens=result.get("total_tokens", 0),
            )

            new_complaint_status = "NEEDS_INFORMATION"

            self.audit_repository.create_ai_audit_log(
                db=db,
                complaint_id=complaint.complaint_id,
                action="AI_WAITING_FOR_USER",
                description=assistant_message,
                audit_metadata={
                    "ai_run_id": str(ai_run.ai_run_id),
                    "missing_fields": missing_fields,
                },
            )

        else:
            error_message = (
                result.get("error_message")
                or final_output.get("error_message")
                or "AI complaint processing failed."
            )

            self.ai_run_repository.update_extracted_fields(
                db=db,
                ai_run=ai_run,
                extracted_fields=extracted_fields,
            )

            self.ai_run_repository.update_missing_fields(
                db=db,
                ai_run=ai_run,
                missing_fields=missing_fields,
            )

            self.ai_run_repository.update_classification_result(
                db=db,
                ai_run=ai_run,
                classification_result=classification_result,
            )

            self.ai_run_repository.update_risk_assessment_result(
                db=db,
                ai_run=ai_run,
                risk_assessment_result=risk_assessment_result,
            )

            self.ai_run_repository.update_final_output(
                db=db,
                ai_run=ai_run,
                final_output=final_output,
            )

            self.ai_run_repository.update_token_usage(
                db=db,
                ai_run=ai_run,
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get(
                    "completion_tokens",
                    0,
                ),
                total_tokens=result.get("total_tokens", 0),
            )

            self.ai_run_repository.mark_ai_run_as_failed(
                db=db,
                ai_run=ai_run,
                error_message=error_message,
            )

            new_complaint_status = "PROCESSING_FAILED"

            self.audit_repository.create_ai_processing_failed_log(
                db=db,
                complaint_id=complaint.complaint_id,
                error_message=error_message,
                ai_run_id=ai_run.ai_run_id,
            )

        self.complaint_repository.update_complaint_status(
            db=db,
            complaint=complaint,
            new_status=new_complaint_status,
        )

        if previous_status != new_complaint_status:
            self.audit_repository.create_status_change_log(
                db=db,
                complaint_id=complaint.complaint_id,
                previous_status=previous_status,
                new_status=new_complaint_status,
                performed_by="AI",
                description=(
                    "Complaint status updated after AI processing."
                ),
                audit_metadata={
                    "ai_run_id": str(ai_run.ai_run_id),
                    "graph_status": processing_status,
                },
            )

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _build_complaint_updates(
        self,
        extracted_fields: dict[str, Any],
        classification_result: dict[str, Any],
        risk_assessment_result: dict[str, Any],
    ) -> dict[str, Any]:
        update_data: dict[str, Any] = {}

        for graph_field, complaint_field in (
            self.GRAPH_TO_COMPLAINT_FIELD_MAP.items()
        ):
            value = extracted_fields.get(
                graph_field
            )

            if value is not None:
                update_data[complaint_field] = value

        category = classification_result.get(
            "complaint_category"
        )

        severity = classification_result.get(
            "suggested_severity"
        )

        suggested_next_action = risk_assessment_result.get(
            "suggested_next_action"
        )

        risk_reasoning = risk_assessment_result.get(
            "risk_reasoning"
        )

        if category is not None:
            update_data["complaint_category"] = category

        if severity in {
            "MINOR",
            "MAJOR",
            "CRITICAL",
        }:
            update_data["suggested_severity"] = severity

        if suggested_next_action is not None:
            update_data[
                "suggested_next_action"
            ] = suggested_next_action

        if risk_reasoning is not None:
            update_data[
                "initial_risk_assessment"
            ] = risk_reasoning

        return update_data

    def _build_field_changes(
        self,
        complaint: Complaint,
        update_data: dict[str, Any],
    ) -> list[dict[str, str | None]]:
        changes: list[dict[str, str | None]] = []

        for field_name, new_value in update_data.items():
            old_value = getattr(
                complaint,
                field_name,
                None,
            )

            if old_value == new_value:
                continue

            changes.append(
                {
                    "field_name": field_name,
                    "old_value": self._serialize_value(
                        old_value
                    ),
                    "new_value": self._serialize_value(
                        new_value
                    ),
                }
            )

        return changes

    def _get_existing_graph_fields(
        self,
        db: Session,
        complaint: Complaint,
    ) -> dict[str, Any]:
        latest_run = self.ai_run_repository.get_latest_ai_run(
            db=db,
            complaint_id=complaint.complaint_id,
        )

        if (
            latest_run is not None
            and latest_run.extracted_fields
        ):
            return dict(
                latest_run.extracted_fields
            )

        return {
            "complainant_name": complaint.customer_name,
            "product_name": complaint.product_name,
            "product_strength_grade": (
                complaint.product_strength_grade
            ),
            "batch_lot_number": complaint.batch_lot_number,
            "quantity_affected": (
                complaint.affected_quantity_text
            ),
            "manufacturing_date": (
                complaint.manufacturing_date_text
            ),
            "expiry_date": complaint.expiry_date_text,
            "complaint_description": (
                complaint.complaint_description
            ),
        }

    # ------------------------------------------------------------------
    # Record creation helpers
    # ------------------------------------------------------------------

    def _create_ai_run(
        self,
        db: Session,
        complaint: Complaint,
        trigger_type: str,
        input_payload: dict[str, Any],
    ) -> ComplaintAIRun:
        ai_run = self.ai_run_repository.create_ai_run(
            db=db,
            ai_run_data=AIRunCreate(
                complaint_id=complaint.complaint_id,
                trigger_type=trigger_type,
                model_provider="GROQ",
                model_name=settings.GROQ_MODEL,
                input_payload=input_payload,
            ),
        )

        return self.ai_run_repository.mark_ai_run_as_processing(
            db=db,
            ai_run=ai_run,
        )

    def _create_result_message(
        self,
        db: Session,
        complaint: Complaint,
        ai_run: ComplaintAIRun,
        result: dict[str, Any],
        assistant_message: str,
        processing_status: str,
    ):
        metadata = {
            "ai_run_id": str(ai_run.ai_run_id),
            "processing_status": processing_status,
            "clarification_required": result.get(
                "clarification_required",
                False,
            ),
            "missing_fields": result.get(
                "missing_fields",
                [],
            ),
            "updated_fields": result.get(
                "updated_fields",
                {},
            ),
        }

        if processing_status == "FAILED":
            return self.message_repository.create_error_message(
                db=db,
                complaint_id=complaint.complaint_id,
                message_text=assistant_message,
                message_metadata=metadata,
            )

        if result.get("updated_fields"):
            return self.message_repository.create_field_update_message(
                db=db,
                complaint_id=complaint.complaint_id,
                message_text=assistant_message,
                message_metadata=metadata,
            )

        return self.message_repository.create_extraction_result_message(
            db=db,
            complaint_id=complaint.complaint_id,
            message_text=assistant_message,
            message_metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Validation and error helpers
    # ------------------------------------------------------------------

    def _get_complaint_or_404(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> Complaint:
        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        return complaint

    def _get_editable_complaint(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> Complaint:
        complaint = self._get_complaint_or_404(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint.is_committed:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Committed complaints cannot be modified "
                    "or reprocessed."
                ),
            )

        return complaint

    def _handle_unexpected_processing_error(
        self,
        db: Session,
        complaint: Complaint,
        previous_status: str,
        error: Exception,
        ai_run: ComplaintAIRun | None,
    ) -> ComplaintAIRun:
        error_message = str(error) or (
            "Unexpected AI processing error."
        )

        try:
            if ai_run is None:
                ai_run = self._create_ai_run(
                    db=db,
                    complaint=complaint,
                    trigger_type="REANALYSIS",
                    input_payload={
                        "error_context": "workflow_start_failure",
                    },
                )

            self.ai_run_repository.mark_ai_run_as_failed(
                db=db,
                ai_run=ai_run,
                error_message=error_message,
            )

            self.complaint_repository.update_complaint_status(
                db=db,
                complaint=complaint,
                new_status="PROCESSING_FAILED",
            )

            self.message_repository.create_error_message(
                db=db,
                complaint_id=complaint.complaint_id,
                message_text=(
                    "Complaint processing failed. "
                    f"{error_message}"
                ),
                message_metadata={
                    "ai_run_id": str(ai_run.ai_run_id),
                },
            )

            self.audit_repository.create_ai_processing_failed_log(
                db=db,
                complaint_id=complaint.complaint_id,
                error_message=error_message,
                ai_run_id=ai_run.ai_run_id,
            )

            if previous_status != "PROCESSING_FAILED":
                self.audit_repository.create_status_change_log(
                    db=db,
                    complaint_id=complaint.complaint_id,
                    previous_status=previous_status,
                    new_status="PROCESSING_FAILED",
                    performed_by="AI",
                    description=(
                        "Complaint status updated after an unexpected "
                        "AI workflow error."
                    ),
                    audit_metadata={
                        "ai_run_id": str(ai_run.ai_run_id),
                    },
                )

            db.commit()
            db.refresh(ai_run)

            return ai_run

        except SQLAlchemyError as db_exc:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "AI processing failed and the failure record "
                    "could not be saved."
                ),
            ) from db_exc

    @staticmethod
    def _serialize_value(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        if isinstance(value, str):
            return value

        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )