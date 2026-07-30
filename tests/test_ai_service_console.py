from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models import (
    Complaint,
    ComplaintAIRun,
    ComplaintAuditLog,
    ComplaintFieldChange,
    ComplaintMessage,
)
from app.schemas.complaint import ComplaintCreate
from app.services.ai_service import AIService
from app.services.complaint_service import ComplaintService


INITIAL_COMPLAINT_TEXT = """
Apollo Pharmacy reported 12 discolored capsules in a sealed bottle
of Amoxicillin Capsules 500 mg.

Batch number: AMX240602.
Manufacturing date: March 2026.
Expiry date: February 2028.

The pharmacy requested an investigation and replacement.
There was no patient injury or adverse event reported.
""".strip()


CHAT_CORRECTION_TEXT = (
    "Correction: the affected quantity was 15 capsules, not 12. "
    "The product was stored at room temperature."
)


def print_section(
    title: str,
    payload: object | None = None,
) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if payload is not None:
        print(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )


def serialize_complaint(
    complaint: Complaint,
) -> dict:
    return {
        "complaint_id": complaint.complaint_id,
        "complaint_number": complaint.complaint_number,
        "customer_name": complaint.customer_name,
        "product_name": complaint.product_name,
        "product_strength_grade": (
            complaint.product_strength_grade
        ),
        "batch_lot_number": complaint.batch_lot_number,
        "affected_quantity_text": (
            complaint.affected_quantity_text
        ),
        "manufacturing_date_text": (
            complaint.manufacturing_date_text
        ),
        "expiry_date_text": complaint.expiry_date_text,
        "complaint_category": complaint.complaint_category,
        "complaint_description": (
            complaint.complaint_description
        ),
        "suggested_severity": complaint.suggested_severity,
        "suggested_next_action": (
            complaint.suggested_next_action
        ),
        "initial_risk_assessment": (
            complaint.initial_risk_assessment
        ),
        "input_type": complaint.input_type,
        "status": complaint.status,
        "is_committed": complaint.is_committed,
    }


def serialize_ai_run(
    ai_run: ComplaintAIRun,
) -> dict:
    return {
        "ai_run_id": ai_run.ai_run_id,
        "complaint_id": ai_run.complaint_id,
        "trigger_type": ai_run.trigger_type,
        "model_provider": ai_run.model_provider,
        "model_name": ai_run.model_name,
        "run_status": ai_run.run_status,
        "input_payload": ai_run.input_payload,
        "extracted_fields": ai_run.extracted_fields,
        "missing_fields": ai_run.missing_fields,
        "classification_result": (
            ai_run.classification_result
        ),
        "risk_assessment_result": (
            ai_run.risk_assessment_result
        ),
        "final_output": ai_run.final_output,
        "prompt_tokens": ai_run.prompt_tokens,
        "completion_tokens": ai_run.completion_tokens,
        "total_tokens": ai_run.total_tokens,
        "error_message": ai_run.error_message,
        "started_at": ai_run.started_at,
        "completed_at": ai_run.completed_at,
    }


def get_complaint_messages(
    db: Session,
    complaint_id: UUID,
) -> list[ComplaintMessage]:
    return (
        db.query(ComplaintMessage)
        .filter(
            ComplaintMessage.complaint_id == complaint_id
        )
        .order_by(ComplaintMessage.created_at.asc())
        .all()
    )


def get_field_changes(
    db: Session,
    complaint_id: UUID,
) -> list[ComplaintFieldChange]:
    return (
        db.query(ComplaintFieldChange)
        .filter(
            ComplaintFieldChange.complaint_id == complaint_id
        )
        .order_by(ComplaintFieldChange.created_at.asc())
        .all()
    )


def get_audit_logs(
    db: Session,
    complaint_id: UUID,
) -> list[ComplaintAuditLog]:
    return (
        db.query(ComplaintAuditLog)
        .filter(
            ComplaintAuditLog.complaint_id == complaint_id
        )
        .order_by(ComplaintAuditLog.created_at.asc())
        .all()
    )


def get_ai_runs(
    db: Session,
    complaint_id: UUID,
) -> list[ComplaintAIRun]:
    return (
        db.query(ComplaintAIRun)
        .filter(
            ComplaintAIRun.complaint_id == complaint_id
        )
        .order_by(ComplaintAIRun.started_at.asc())
        .all()
    )


def serialize_messages(
    messages: list[ComplaintMessage],
) -> list[dict]:
    return [
        {
            "message_id": message.message_id,
            "sender_type": message.sender_type,
            "message_type": message.message_type,
            "message_text": message.message_text,
            "message_metadata": message.message_metadata,
            "created_at": message.created_at,
        }
        for message in messages
    ]


def serialize_field_changes(
    changes: list[ComplaintFieldChange],
) -> list[dict]:
    return [
        {
            "field_change_id": change.field_change_id,
            "ai_run_id": change.ai_run_id,
            "message_id": change.message_id,
            "field_name": change.field_name,
            "old_value": change.old_value,
            "new_value": change.new_value,
            "changed_by": change.changed_by,
            "change_source": change.change_source,
            "created_at": change.created_at,
        }
        for change in changes
    ]


def serialize_audit_logs(
    logs: list[ComplaintAuditLog],
) -> list[dict]:
    return [
        {
            "audit_log_id": log.audit_log_id,
            "action": log.action,
            "performed_by": log.performed_by,
            "previous_status": log.previous_status,
            "new_status": log.new_status,
            "description": log.description,
            "audit_metadata": log.audit_metadata,
            "created_at": log.created_at,
        }
        for log in logs
    ]


def cleanup_test_complaint(
    db: Session,
    complaint_id: UUID,
) -> None:
    """
    Remove all rows created by this console integration test.

    Child records are deleted before the complaint because they
    reference the complaint through foreign keys.
    """

    try:
        db.rollback()

        db.query(ComplaintFieldChange).filter(
            ComplaintFieldChange.complaint_id == complaint_id
        ).delete(synchronize_session=False)

        db.query(ComplaintMessage).filter(
            ComplaintMessage.complaint_id == complaint_id
        ).delete(synchronize_session=False)

        db.query(ComplaintAuditLog).filter(
            ComplaintAuditLog.complaint_id == complaint_id
        ).delete(synchronize_session=False)

        db.query(ComplaintAIRun).filter(
            ComplaintAIRun.complaint_id == complaint_id
        ).delete(synchronize_session=False)

        db.query(Complaint).filter(
            Complaint.complaint_id == complaint_id
        ).delete(synchronize_session=False)

        db.commit()

        print_section(
            "TEST DATA CLEANUP",
            {
                "complaint_id": complaint_id,
                "status": "deleted",
            },
        )

    except Exception:
        db.rollback()
        print(
            "\nWarning: automatic test-data cleanup failed. "
            "The test complaint may still exist in the database."
        )
        raise


def assert_initial_processing_result(
    complaint: Complaint,
    ai_run: ComplaintAIRun,
    messages: list[ComplaintMessage],
    field_changes: list[ComplaintFieldChange],
    audit_logs: list[ComplaintAuditLog],
) -> None:
    assert ai_run.trigger_type == "TEXT_SUBMISSION"
    assert ai_run.run_status == "COMPLETED"

    assert ai_run.extracted_fields is not None
    assert ai_run.classification_result is not None
    assert ai_run.risk_assessment_result is not None
    assert ai_run.final_output is not None

    assert ai_run.total_tokens is not None
    assert ai_run.total_tokens > 0

    assert complaint.status == "READY_TO_COMMIT"
    assert complaint.product_name is not None
    assert complaint.batch_lot_number is not None
    assert complaint.complaint_description is not None

    assert messages
    assert field_changes
    assert audit_logs

    assistant_messages = [
        message
        for message in messages
        if message.sender_type == "ASSISTANT"
    ]

    assert assistant_messages

    changed_fields = {
        change.field_name
        for change in field_changes
    }

    assert "product_name" in changed_fields
    assert "batch_lot_number" in changed_fields


def assert_chat_processing_result(
    complaint: Complaint,
    ai_run: ComplaintAIRun,
    messages: list[ComplaintMessage],
    field_changes: list[ComplaintFieldChange],
    ai_runs: list[ComplaintAIRun],
) -> None:
    assert ai_run.trigger_type == "CHAT_CORRECTION"
    assert ai_run.run_status == "COMPLETED"

    assert ai_run.extracted_fields is not None
    assert ai_run.extracted_fields.get(
        "quantity_affected"
    ) == "15"

    assert ai_run.extracted_fields.get(
        "storage_conditions"
    ) == "room temperature"

    assert complaint.affected_quantity_text == "15"
    assert complaint.status == "READY_TO_COMMIT"

    assert len(ai_runs) == 2

    user_messages = [
        message
        for message in messages
        if message.sender_type == "USER"
    ]

    assert user_messages
    assert any(
        message.message_text == CHAT_CORRECTION_TEXT
        for message in user_messages
    )

    quantity_changes = [
        change
        for change in field_changes
        if (
            change.field_name == "affected_quantity_text"
            and change.change_source == "CHAT_CORRECTION"
        )
    ]

    assert quantity_changes
    assert quantity_changes[-1].new_value == "15"


def run_ai_service_database_test() -> None:
    db = SessionLocal()

    complaint_service = ComplaintService()
    ai_service = AIService()

    complaint_id: UUID | None = None

    try:
        print_section(
            "CREATING TEST COMPLAINT",
            {
                "raw_complaint_text": INITIAL_COMPLAINT_TEXT,
                "input_type": "TEXT",
                "complaint_source": "AI_SERVICE_CONSOLE_TEST",
            },
        )

        complaint = complaint_service.create_complaint(
            db=db,
            complaint_data=ComplaintCreate(
                raw_complaint_text=INITIAL_COMPLAINT_TEXT,
                input_type="TEXT",
                complaint_source="AI_SERVICE_CONSOLE_TEST",
            ),
        )

        complaint_id = complaint.complaint_id

        print_section(
            "TEST COMPLAINT CREATED",
            serialize_complaint(complaint),
        )

        # --------------------------------------------------------------
        # Initial AI processing
        # --------------------------------------------------------------

        print_section(
            "RUNNING INITIAL AI PROCESSING"
        )

        initial_ai_run = ai_service.process_complaint(
            db=db,
            complaint_id=complaint_id,
            trigger_type="TEXT_SUBMISSION",
        )

        db.expire_all()

        complaint = (
            db.query(Complaint)
            .filter(
                Complaint.complaint_id == complaint_id
            )
            .one()
        )

        messages = get_complaint_messages(
            db=db,
            complaint_id=complaint_id,
        )

        field_changes = get_field_changes(
            db=db,
            complaint_id=complaint_id,
        )

        audit_logs = get_audit_logs(
            db=db,
            complaint_id=complaint_id,
        )

        print_section(
            "INITIAL AI RUN",
            serialize_ai_run(initial_ai_run),
        )

        print_section(
            "COMPLAINT AFTER INITIAL PROCESSING",
            serialize_complaint(complaint),
        )

        print_section(
            "MESSAGES AFTER INITIAL PROCESSING",
            serialize_messages(messages),
        )

        print_section(
            "FIELD CHANGES AFTER INITIAL PROCESSING",
            serialize_field_changes(field_changes),
        )

        print_section(
            "AUDIT LOGS AFTER INITIAL PROCESSING",
            serialize_audit_logs(audit_logs),
        )

        assert_initial_processing_result(
            complaint=complaint,
            ai_run=initial_ai_run,
            messages=messages,
            field_changes=field_changes,
            audit_logs=audit_logs,
        )

        print(
            "\nInitial AI service database processing "
            "completed successfully."
        )

        # --------------------------------------------------------------
        # Chat correction
        # --------------------------------------------------------------

        print_section(
            "RUNNING CHAT CORRECTION",
            {
                "message_text": CHAT_CORRECTION_TEXT,
            },
        )

        chat_ai_run = ai_service.process_chat_correction(
            db=db,
            complaint_id=complaint_id,
            message_text=CHAT_CORRECTION_TEXT,
        )

        db.expire_all()

        complaint = (
            db.query(Complaint)
            .filter(
                Complaint.complaint_id == complaint_id
            )
            .one()
        )

        ai_runs = get_ai_runs(
            db=db,
            complaint_id=complaint_id,
        )

        messages = get_complaint_messages(
            db=db,
            complaint_id=complaint_id,
        )

        field_changes = get_field_changes(
            db=db,
            complaint_id=complaint_id,
        )

        audit_logs = get_audit_logs(
            db=db,
            complaint_id=complaint_id,
        )

        print_section(
            "CHAT-CORRECTION AI RUN",
            serialize_ai_run(chat_ai_run),
        )

        print_section(
            "COMPLAINT AFTER CHAT CORRECTION",
            serialize_complaint(complaint),
        )

        print_section(
            "ALL AI RUNS",
            [
                serialize_ai_run(ai_run)
                for ai_run in ai_runs
            ],
        )

        print_section(
            "ALL COMPLAINT MESSAGES",
            serialize_messages(messages),
        )

        print_section(
            "ALL FIELD CHANGES",
            serialize_field_changes(field_changes),
        )

        print_section(
            "ALL AUDIT LOGS",
            serialize_audit_logs(audit_logs),
        )

        assert_chat_processing_result(
            complaint=complaint,
            ai_run=chat_ai_run,
            messages=messages,
            field_changes=field_changes,
            ai_runs=ai_runs,
        )

        print_section(
            "AI SERVICE DATABASE TEST RESULT",
            {
                "status": "SUCCESS",
                "complaint_id": complaint_id,
                "initial_ai_run_id": (
                    initial_ai_run.ai_run_id
                ),
                "chat_ai_run_id": chat_ai_run.ai_run_id,
                "ai_run_count": len(ai_runs),
                "message_count": len(messages),
                "field_change_count": len(field_changes),
                "audit_log_count": len(audit_logs),
                "final_quantity": (
                    complaint.affected_quantity_text
                ),
                "final_complaint_status": complaint.status,
            },
        )

        print(
            "\nAll AI service database integration "
            "tests completed successfully."
        )

    except AssertionError as exc:
        db.rollback()

        print_section(
            "TEST ASSERTION FAILED",
            {
                "error": str(exc) or (
                    "An integration-test assertion failed."
                ),
                "complaint_id": complaint_id,
            },
        )

        raise

    except Exception as exc:
        db.rollback()

        print_section(
            "AI SERVICE DATABASE TEST FAILED",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "complaint_id": complaint_id,
            },
        )

        raise

    finally:
        if complaint_id is not None:
            cleanup_test_complaint(
                db=db,
                complaint_id=complaint_id,
            )

        db.close()


if __name__ == "__main__":
    run_ai_service_database_test()