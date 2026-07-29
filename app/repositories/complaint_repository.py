from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.complaint import Complaint
from app.schemas.complaint import (
    ComplaintCreate,
    ComplaintUpdate,
)


class ComplaintRepository:
    """
    Handles database operations related to complaints.

    This repository is responsible only for reading and writing
    complaint records. Business rules and API logic should remain
    in the service and route layers.
    """

    def create_complaint(
        self,
        db: Session,
        complaint_data: ComplaintCreate,
    ) -> Complaint:
        """
        Create a new draft complaint.

        The transaction is flushed but not committed here.
        The service layer will control the final commit or rollback.
        """

        complaint = Complaint(
            **complaint_data.model_dump(exclude_unset=True)
        )

        db.add(complaint)
        db.flush()
        db.refresh(complaint)

        return complaint

    def get_complaint_by_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> Optional[Complaint]:
        """
        Retrieve a complaint using its UUID.
        """

        return (
            db.query(Complaint)
            .filter(
                Complaint.complaint_id == complaint_id
            )
            .first()
        )

    def get_complaint_by_number(
        self,
        db: Session,
        complaint_number: str,
    ) -> Optional[Complaint]:
        """
        Retrieve a complaint using its complaint number.
        """

        return (
            db.query(Complaint)
            .filter(
                Complaint.complaint_number == complaint_number
            )
            .first()
        )

    def get_all_complaints(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 20,
        status: Optional[str] = None,
        is_committed: Optional[bool] = None,
    ) -> List[Complaint]:
        """
        Retrieve complaints with optional filters and pagination.
        """

        query = db.query(Complaint)

        if status is not None:
            query = query.filter(
                Complaint.status == status
            )

        if is_committed is not None:
            query = query.filter(
                Complaint.is_committed == is_committed
            )

        return (
            query
            .order_by(Complaint.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def update_complaint(
        self,
        db: Session,
        complaint: Complaint,
        complaint_data: ComplaintUpdate,
    ) -> Complaint:
        """
        Update only the fields explicitly provided by the caller.
        """

        update_data = complaint_data.model_dump(
            exclude_unset=True
        )

        for field_name, field_value in update_data.items():
            setattr(
                complaint,
                field_name,
                field_value,
            )

        complaint.updated_at = datetime.now(timezone.utc)

        db.flush()
        db.refresh(complaint)

        return complaint

    def update_complaint_fields(
        self,
        db: Session,
        complaint: Complaint,
        update_data: dict,
    ) -> Complaint:
        """
        Update complaint fields using a dictionary.

        This method will be useful when LangGraph returns extracted
        complaint fields as structured JSON.
        """

        for field_name, field_value in update_data.items():
            if hasattr(complaint, field_name):
                setattr(
                    complaint,
                    field_name,
                    field_value,
                )

        complaint.updated_at = datetime.now(timezone.utc)

        db.flush()
        db.refresh(complaint)

        return complaint

    def update_complaint_status(
        self,
        db: Session,
        complaint: Complaint,
        new_status: str,
    ) -> Complaint:
        """
        Update the workflow status of a complaint.
        """

        complaint.status = new_status
        complaint.updated_at = datetime.now(timezone.utc)

        db.flush()
        db.refresh(complaint)

        return complaint

    def mark_complaint_as_committed(
        self,
        db: Session,
        complaint: Complaint,
    ) -> Complaint:
        """
        Mark a complaint as committed to the QMS ledger.
        """

        current_time = datetime.now(timezone.utc)

        complaint.is_committed = True
        complaint.status = "COMMITTED"
        complaint.committed_at = current_time
        complaint.updated_at = current_time

        db.flush()
        db.refresh(complaint)

        return complaint

    def count_complaints(
        self,
        db: Session,
        status: Optional[str] = None,
        is_committed: Optional[bool] = None,
    ) -> int:
        """
        Count complaints with optional filters.
        """

        query = db.query(Complaint)

        if status is not None:
            query = query.filter(
                Complaint.status == status
            )

        if is_committed is not None:
            query = query.filter(
                Complaint.is_committed == is_committed
            )

        return query.count()

    def complaint_number_exists(
        self,
        db: Session,
        complaint_number: str,
    ) -> bool:
        """
        Check whether a complaint number already exists.
        """

        complaint = (
            db.query(Complaint.complaint_id)
            .filter(
                Complaint.complaint_number == complaint_number
            )
            .first()
        )

        return complaint is not None