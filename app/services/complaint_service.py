from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.complaint import Complaint
from app.repositories.audit_repository import AuditRepository
from app.repositories.complaint_repository import ComplaintRepository
from app.repositories.field_change_repository import FieldChangeRepository
from app.schemas.complaint import ComplaintCreate, ComplaintUpdate


class ComplaintService:
    """
    Contains complaint-related business logic.

    The service coordinates multiple repositories and controls
    database transactions using commit and rollback.
    """

    def __init__(self):
        self.complaint_repository = ComplaintRepository()
        self.field_change_repository = FieldChangeRepository()
        self.audit_repository = AuditRepository()

    def create_complaint(
        self,
        db: Session,
        complaint_data: ComplaintCreate,
    ) -> Complaint:
        """
        Create a complaint draft and its corresponding audit log.
        """

        try:
            complaint = self.complaint_repository.create_complaint(
                db=db,
                complaint_data=complaint_data,
            )

            self.audit_repository.create_complaint_created_log(
                db=db,
                complaint_id=complaint.complaint_id,
                audit_metadata={
                    "complaint_number": complaint.complaint_number,
                    "input_type": complaint.input_type,
                },
            )

            db.commit()
            db.refresh(complaint)

            return complaint

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create complaint.",
            ) from exc

    def get_complaint_by_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> Complaint:
        """
        Retrieve a complaint by its ID.

        Raises a 404 error when the complaint does not exist.
        """

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

    def get_complaint_by_number(
        self,
        db: Session,
        complaint_number: str,
    ) -> Complaint:
        """
        Retrieve a complaint using its complaint number.
        """

        complaint = (
            self.complaint_repository.get_complaint_by_number(
                db=db,
                complaint_number=complaint_number,
            )
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        return complaint

    def get_all_complaints(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 50,
        complaint_status: Optional[str] = None,
    ) -> List[Complaint]:
        """
        Retrieve complaints with optional status filtering.
        """

        return self.complaint_repository.get_all_complaints(
            db=db,
            skip=skip,
            limit=limit,
            status=complaint_status,
        )

    def update_complaint(
        self,
        db: Session,
        complaint_id: UUID,
        complaint_data: ComplaintUpdate,
    ) -> Complaint:
        """
        Update complaint fields and record each changed field.

        Only fields included in the request are processed.
        """

        complaint = self.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint.is_committed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Committed complaints cannot be edited.",
            )

        update_data = complaint_data.model_dump(
            exclude_unset=True
        )

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No complaint fields were provided for update.",
            )

        try:
            field_changes = []

            for field_name, new_value in update_data.items():
                if not hasattr(complaint, field_name):
                    continue

                old_value = getattr(complaint, field_name)

                if old_value == new_value:
                    continue

                field_changes.append(
                    {
                        "field_name": field_name,
                        "old_value": (
                            str(old_value)
                            if old_value is not None
                            else None
                        ),
                        "new_value": (
                            str(new_value)
                            if new_value is not None
                            else None
                        ),
                    }
                )

            complaint = self.complaint_repository.update_complaint(
                db=db,
                complaint=complaint,
                complaint_data=complaint_data,
            )

            if field_changes:
                self.field_change_repository.create_multiple_field_changes(
                    db=db,
                    complaint_id=complaint.complaint_id,
                    changes=field_changes,
                    changed_by="USER",
                    change_source="MANUAL_EDIT",
                )

                self.audit_repository.create_user_audit_log(
                    db=db,
                    complaint_id=complaint.complaint_id,
                    action="COMPLAINT_UPDATED",
                    description="Complaint details were updated.",
                    audit_metadata={
                        "changed_fields": [
                            change["field_name"]
                            for change in field_changes
                        ]
                    },
                )

            db.commit()
            db.refresh(complaint)

            return complaint

        except HTTPException:
            db.rollback()
            raise

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update complaint.",
            ) from exc

    def update_complaint_status(
        self,
        db: Session,
        complaint_id: UUID,
        new_status: str,
        performed_by: str = "USER",
        description: Optional[str] = None,
    ) -> Complaint:
        """
        Update complaint status and record the transition.

        Supported complaint statuses:

        PENDING_TRIAGE
        PROCESSING
        NEEDS_INFORMATION
        READY_TO_COMMIT
        COMMITTED
        PROCESSING_FAILED
        """

        valid_statuses = {
            "PENDING_TRIAGE",
            "PROCESSING",
            "NEEDS_INFORMATION",
            "READY_TO_COMMIT",
            "COMMITTED",
            "PROCESSING_FAILED",
        }

        valid_performers = {
            "USER",
            "AI",
            "SYSTEM",
        }

        if new_status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid complaint status: {new_status}",
            )

        if performed_by not in valid_performers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid performed_by value: {performed_by}",
            )

        complaint = self.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint.is_committed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The complaint has already been committed.",
            )

        previous_status = complaint.status

        if previous_status == new_status:
            return complaint

        try:
            complaint = (
                self.complaint_repository.update_complaint_status(
                    db=db,
                    complaint=complaint,
                    new_status=new_status,
                )
            )

            self.audit_repository.create_status_change_log(
                db=db,
                complaint_id=complaint.complaint_id,
                previous_status=previous_status,
                new_status=new_status,
                performed_by=performed_by,
                description=description,
            )

            db.commit()
            db.refresh(complaint)

            return complaint

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update complaint status.",
            ) from exc

    def mark_as_processing(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> Complaint:
        """
        Mark a complaint as currently being processed by AI.
        """

        return self.update_complaint_status(
            db=db,
            complaint_id=complaint_id,
            new_status="PROCESSING",
            performed_by="AI",
            description="AI complaint processing started.",
        )

    def mark_as_needs_information(
        self,
        db: Session,
        complaint_id: UUID,
        missing_fields: Optional[list] = None,
    ) -> Complaint:
        """
        Mark a complaint as requiring additional information.
        """

        description = (
            "Additional complaint information is required."
        )

        complaint = self.update_complaint_status(
            db=db,
            complaint_id=complaint_id,
            new_status="NEEDS_INFORMATION",
            performed_by="AI",
            description=description,
        )

        return complaint

    def mark_as_ready_to_commit(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> Complaint:
        """
        Mark a complaint as ready for QMS ledger commitment.
        """

        return self.update_complaint_status(
            db=db,
            complaint_id=complaint_id,
            new_status="READY_TO_COMMIT",
            performed_by="AI",
            description=(
                "Complaint processing is complete and ready "
                "for QMS ledger commitment."
            ),
        )

    def mark_as_processing_failed(
        self,
        db: Session,
        complaint_id: UUID,
        error_message: str,
    ) -> Complaint:
        """
        Mark complaint processing as failed.
        """

        return self.update_complaint_status(
            db=db,
            complaint_id=complaint_id,
            new_status="PROCESSING_FAILED",
            performed_by="SYSTEM",
            description=error_message,
        )

    def commit_complaint_to_qms(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> Complaint:
        """
        Commit a completed complaint to the QMS ledger.

        Only complaints with READY_TO_COMMIT status can be committed.
        """

        complaint = self.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint.is_committed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Complaint is already committed.",
            )

        if complaint.status != "READY_TO_COMMIT":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Only complaints with READY_TO_COMMIT status "
                    "can be committed."
                ),
            )

        previous_status = complaint.status

        try:
            complaint = (
                self.complaint_repository.mark_complaint_as_committed(
                    db=db,
                    complaint=complaint,
                )
            )

            self.audit_repository.create_complaint_committed_log(
                db=db,
                complaint_id=complaint.complaint_id,
                previous_status=previous_status,
                audit_metadata={
                    "complaint_number": complaint.complaint_number,
                },
            )

            db.commit()
            db.refresh(complaint)

            return complaint

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to commit complaint to QMS ledger.",
            ) from exc

    def count_complaints(
        self,
        db: Session,
        complaint_status: Optional[str] = None,
    ) -> int:
        """
        Count complaint records with optional status filtering.
        """

        return self.complaint_repository.count_complaints(
            db=db,
            status=complaint_status,
        )