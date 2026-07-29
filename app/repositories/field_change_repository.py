from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.complaint_field_change import ComplaintFieldChange
from app.schemas.field_change import FieldChangeCreate


class FieldChangeRepository:
    """
    Handles database operations related to complaint field changes.

    Every complaint field modification can be recorded here,
    including changes made by the user, AI, or system.
    """

    def create_field_change(
        self,
        db: Session,
        field_change_data: FieldChangeCreate,
    ) -> ComplaintFieldChange:
        """
        Create a new complaint field-change record.
        """

        field_change = ComplaintFieldChange(
            **field_change_data.model_dump(exclude_unset=True)
        )

        db.add(field_change)
        db.flush()
        db.refresh(field_change)

        return field_change

    def get_field_change_by_id(
        self,
        db: Session,
        field_change_id: UUID,
    ) -> Optional[ComplaintFieldChange]:
        """
        Retrieve a field-change record using its UUID.
        """

        return (
            db.query(ComplaintFieldChange)
            .filter(
                ComplaintFieldChange.field_change_id
                == field_change_id
            )
            .first()
        )

    def get_field_changes_by_complaint_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> List[ComplaintFieldChange]:
        """
        Retrieve all field changes belonging to a complaint.

        Changes are returned from oldest to newest.
        """

        return (
            db.query(ComplaintFieldChange)
            .filter(
                ComplaintFieldChange.complaint_id == complaint_id
            )
            .order_by(
                ComplaintFieldChange.created_at.asc()
            )
            .all()
        )

    def get_field_changes_by_field_name(
        self,
        db: Session,
        complaint_id: UUID,
        field_name: str,
    ) -> List[ComplaintFieldChange]:
        """
        Retrieve the complete history of a complaint field.
        """

        return (
            db.query(ComplaintFieldChange)
            .filter(
                ComplaintFieldChange.complaint_id == complaint_id,
                ComplaintFieldChange.field_name == field_name,
            )
            .order_by(
                ComplaintFieldChange.created_at.asc()
            )
            .all()
        )

    def get_latest_field_change(
        self,
        db: Session,
        complaint_id: UUID,
        field_name: str,
    ) -> Optional[ComplaintFieldChange]:
        """
        Retrieve the latest change made to a complaint field.
        """

        return (
            db.query(ComplaintFieldChange)
            .filter(
                ComplaintFieldChange.complaint_id == complaint_id,
                ComplaintFieldChange.field_name == field_name,
            )
            .order_by(
                ComplaintFieldChange.created_at.desc()
            )
            .first()
        )

    def get_field_changes_by_changed_by(
        self,
        db: Session,
        complaint_id: UUID,
        changed_by: str,
    ) -> List[ComplaintFieldChange]:
        """
        Retrieve field changes based on who made the change.

        Supported values:
        USER
        AI
        SYSTEM
        """

        return (
            db.query(ComplaintFieldChange)
            .filter(
                ComplaintFieldChange.complaint_id == complaint_id,
                ComplaintFieldChange.changed_by == changed_by,
            )
            .order_by(
                ComplaintFieldChange.created_at.asc()
            )
            .all()
        )

    def get_field_changes_by_source(
        self,
        db: Session,
        complaint_id: UUID,
        change_source: str,
    ) -> List[ComplaintFieldChange]:
        """
        Retrieve field changes created from a particular source.

        Supported values:
        INITIAL_EXTRACTION
        CHAT_CORRECTION
        MANUAL_EDIT
        FILE_REPROCESSING
        SYSTEM_UPDATE
        """

        return (
            db.query(ComplaintFieldChange)
            .filter(
                ComplaintFieldChange.complaint_id == complaint_id,
                ComplaintFieldChange.change_source == change_source,
            )
            .order_by(
                ComplaintFieldChange.created_at.asc()
            )
            .all()
        )

    def get_field_changes_by_ai_run_id(
        self,
        db: Session,
        ai_run_id: UUID,
    ) -> List[ComplaintFieldChange]:
        """
        Retrieve all field changes created during an AI run.
        """

        return (
            db.query(ComplaintFieldChange)
            .filter(
                ComplaintFieldChange.ai_run_id == ai_run_id
            )
            .order_by(
                ComplaintFieldChange.created_at.asc()
            )
            .all()
        )

    def get_field_changes_by_message_id(
        self,
        db: Session,
        message_id: UUID,
    ) -> List[ComplaintFieldChange]:
        """
        Retrieve all field changes linked to a complaint message.
        """

        return (
            db.query(ComplaintFieldChange)
            .filter(
                ComplaintFieldChange.message_id == message_id
            )
            .order_by(
                ComplaintFieldChange.created_at.asc()
            )
            .all()
        )

    def create_user_field_change(
        self,
        db: Session,
        complaint_id: UUID,
        field_name: str,
        old_value: Optional[str],
        new_value: Optional[str],
        message_id: Optional[UUID] = None,
        change_source: str = "MANUAL_EDIT",
    ) -> ComplaintFieldChange:
        """
        Record a complaint field change made by the user.
        """

        field_change = ComplaintFieldChange(
            complaint_id=complaint_id,
            message_id=message_id,
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            changed_by="USER",
            change_source=change_source,
        )

        db.add(field_change)
        db.flush()
        db.refresh(field_change)

        return field_change

    def create_ai_field_change(
        self,
        db: Session,
        complaint_id: UUID,
        field_name: str,
        old_value: Optional[str],
        new_value: Optional[str],
        ai_run_id: Optional[UUID] = None,
        message_id: Optional[UUID] = None,
        change_source: str = "INITIAL_EXTRACTION",
    ) -> ComplaintFieldChange:
        """
        Record a complaint field change made by the AI.
        """

        field_change = ComplaintFieldChange(
            complaint_id=complaint_id,
            ai_run_id=ai_run_id,
            message_id=message_id,
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            changed_by="AI",
            change_source=change_source,
        )

        db.add(field_change)
        db.flush()
        db.refresh(field_change)

        return field_change

    def create_system_field_change(
        self,
        db: Session,
        complaint_id: UUID,
        field_name: str,
        old_value: Optional[str],
        new_value: Optional[str],
        change_source: str = "SYSTEM_UPDATE",
    ) -> ComplaintFieldChange:
        """
        Record a complaint field change made by the system.
        """

        field_change = ComplaintFieldChange(
            complaint_id=complaint_id,
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            changed_by="SYSTEM",
            change_source=change_source,
        )

        db.add(field_change)
        db.flush()
        db.refresh(field_change)

        return field_change

    def create_multiple_field_changes(
        self,
        db: Session,
        complaint_id: UUID,
        changes: list,
        changed_by: str,
        change_source: str,
        ai_run_id: Optional[UUID] = None,
        message_id: Optional[UUID] = None,
    ) -> List[ComplaintFieldChange]:
        """
        Create multiple field-change records in one operation.

        Expected changes format:

        [
            {
                "field_name": "product_name",
                "old_value": "Old Product",
                "new_value": "New Product"
            }
        ]
        """

        field_changes = []

        for change in changes:
            field_change = ComplaintFieldChange(
                complaint_id=complaint_id,
                ai_run_id=ai_run_id,
                message_id=message_id,
                field_name=change.get("field_name"),
                old_value=change.get("old_value"),
                new_value=change.get("new_value"),
                changed_by=changed_by,
                change_source=change_source,
            )

            db.add(field_change)
            field_changes.append(field_change)

        db.flush()

        for field_change in field_changes:
            db.refresh(field_change)

        return field_changes

    def count_field_changes_by_complaint_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> int:
        """
        Count all recorded field changes for a complaint.
        """

        return (
            db.query(ComplaintFieldChange)
            .filter(
                ComplaintFieldChange.complaint_id == complaint_id
            )
            .count()
        )

    def field_change_belongs_to_complaint(
        self,
        db: Session,
        field_change_id: UUID,
        complaint_id: UUID,
    ) -> bool:
        """
        Check whether a field-change record belongs to a complaint.
        """

        field_change = (
            db.query(ComplaintFieldChange.field_change_id)
            .filter(
                ComplaintFieldChange.field_change_id
                == field_change_id,
                ComplaintFieldChange.complaint_id
                == complaint_id,
            )
            .first()
        )

        return field_change is not None