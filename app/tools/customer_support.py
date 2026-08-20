"""Validated telecom account tools with ownership enforcement independent of any LLM."""

import re
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from sqlmodel import Session, SQLModel, select

from app.call_logging import call_exception, call_log
from app.identifiers import mask_identifier, normalize_spelled_name
from app.persistence.models import (
    BillingAccount,
    Customer,
    CustomerOrder,
    Invoice,
    Payment,
    Subscription,
    SupportTicket,
    Transaction,
)
from app.persistence.session_store import engine

_CUSTOMER_ID = re.compile(r"\ACUST\d{4}\Z", re.IGNORECASE)
_RESOURCE_ID = re.compile(r"\A(?:PAY|INV|ORD|TKT)[A-Z0-9]{6,}\Z", re.IGNORECASE)
_T = TypeVar("_T", bound=SQLModel)


class ToolError(RuntimeError):
    """Base class with a safe caller-facing message; details remain in backend logs."""

    public_message = "I'm unable to retrieve that information right now."


class ToolValidationError(ToolError):
    public_message = "That reference number is not in a valid format."


class ToolAccessDenied(ToolError):
    public_message = "Access denied: that record does not belong to the verified account."


class ToolNotFound(ToolError):
    public_message = "I couldn't find that record for the verified account."


def _normalise_customer_id(customer_id: str) -> str:
    value = customer_id.strip().upper()
    if not _CUSTOMER_ID.fullmatch(value):
        raise ToolValidationError("Malformed customer ID")
    return value


def validate_customer_id(customer_id: str, *, session_id: str = "system") -> str:
    """Validate a normalized customer ID before any customer-specific lookup."""
    try:
        value = _normalise_customer_id(customer_id)
    except ToolValidationError:
        call_log(session_id, "IDENTIFIER", "validation", details={"type": "customer_id", "valid": False})
        raise
    call_log(session_id, "IDENTIFIER", "validation_success", details={"type": "customer_id", "valid": True})
    return value


def find_customer_candidates(name: str, *, session_id: str = "system") -> list[str]:
    """Return internal candidate IDs for an exact name match; callers must add a second factor."""
    supplied_name = " ".join(name.strip().casefold().split())
    if not supplied_name or len(supplied_name) > 120:
        raise ToolValidationError("Invalid account name")
    spelled_name = normalize_spelled_name(name)
    with Session(engine) as session:
        records = session.exec(select(Customer).where(Customer.full_name.is_not(None))).all()
    candidates = [
        record.customer_id
        for record in records
        if supplied_name == " ".join(record.full_name.casefold().split())
        or (spelled_name is not None and spelled_name == record.full_name.casefold().replace(" ", ""))
    ]
    call_log(session_id, "IDENTITY", "candidate_search", details={"candidate_count": len(candidates)})
    return candidates


def verify_recovery_candidate(
    candidate_ids: list[str], contact: str, *, session_id: str = "system"
) -> str | None:
    """Verify an exact phone or email factor against the name-scoped candidate set."""
    value = contact.strip().casefold()
    phone_digits = re.sub(r"\D", "", contact)
    if not value or len(value) > 160:
        raise ToolValidationError("Invalid verification factor")
    with Session(engine) as session:
        records = [session.get(Customer, customer_id) for customer_id in candidate_ids]
    matches = [
        record.customer_id
        for record in records
        if record is not None
        and ((record.email and record.email.casefold() == value) or (record.phone and re.sub(r"\D", "", record.phone) == phone_digits))
    ]
    if len(matches) == 1:
        call_log(session_id, "IDENTITY", "verification_success", details={"verification_method": "contact"})
        return matches[0]
    call_log(session_id, "IDENTITY", "verification_failed", details={"candidate_count": len(matches)})
    return None


def _normalise_resource_id(resource_id: str, prefix: str) -> str:
    value = resource_id.strip().upper()
    if not value.startswith(prefix) or not _RESOURCE_ID.fullmatch(value):
        raise ToolValidationError("Malformed resource ID")
    return value


def _authorize(session_id: str, authenticated_customer_id: str, requested_customer_id: str, resource: str) -> str:
    authenticated = _normalise_customer_id(authenticated_customer_id)
    requested = _normalise_customer_id(requested_customer_id)
    call_log(session_id, "AUTHZ", "checking", details={"customer_id": mask_identifier(requested), "resource": resource})
    if authenticated != requested:
        call_log(session_id, "AUTHZ", "denied", level=30, details={"reason": "customer_mismatch", "resource": resource})
        raise ToolAccessDenied("Authenticated customer does not own requested customer ID")
    call_log(session_id, "AUTHZ", "approved", details={"resource": resource})
    return requested


def _owned_one(
    session: Session, model: type[_T], identifier_column: Any, identifier: str, customer_id: str, resource: str
) -> _T:
    record = session.exec(select(model).where(identifier_column == identifier)).first()
    if record is None:
        raise ToolNotFound(f"{resource} was not found")
    if record.customer_id != customer_id:  # type: ignore[attr-defined]
        raise ToolAccessDenied(f"{resource} ownership mismatch")
    return record


def _run_tool(session_id: str, tool: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Emit safe observability and preserve domain errors for the workflow to phrase."""
    call_log(session_id, "TOOL", "start", details={"tool": tool})
    try:
        result = operation()
    except ToolError:
        raise
    except Exception:
        call_exception(session_id, "TOOL", "failed", details={"tool": tool})
        raise ToolError("Database lookup failed") from None
    call_log(session_id, "DB", "query_complete", details={"rows": 1, "tool": tool})
    call_log(session_id, "TOOL", "complete", details={"tool": tool})
    call_log(session_id, "TOOL", f"{tool}_complete", details={"tool": tool})
    return result


def verify_customer_identity(customer_id: str, name: str, *, session_id: str = "system") -> bool:
    """Verification-only query; returns no account information and establishes no LLM trust."""
    customer = _normalise_customer_id(customer_id)
    supplied_name = " ".join(name.strip().casefold().split())
    if not supplied_name or len(supplied_name) > 120:
        raise ToolValidationError("Invalid account name")
    spelled_name = normalize_spelled_name(name)

    def operation() -> dict[str, bool]:
        with Session(engine) as session:
            record = session.get(Customer, customer)
            stored_name = " ".join(record.full_name.casefold().split()) if record else ""
            matches = record is not None and (
                supplied_name == stored_name
                or (spelled_name is not None and spelled_name == stored_name.replace(" ", ""))
            )
            return {"verified": matches}

    return bool(_run_tool(session_id, "verify_customer_identity", operation)["verified"])


def get_customer(customer_id: str, authenticated_customer_id: str, *, session_id: str = "system") -> dict[str, Any]:
    customer = _authorize(session_id, authenticated_customer_id, customer_id, "customer")

    def operation() -> dict[str, Any]:
        with Session(engine) as session:
            record = session.get(Customer, customer)
            if record is None:
                raise ToolNotFound("Customer was not found")
            return {
                "customer_id": record.customer_id,
                "full_name": record.full_name,
                "account_status": record.account_status,
            }

    return _run_tool(session_id, "customer_lookup", operation)


def get_account_status(
    customer_id: str, authenticated_customer_id: str, *, session_id: str = "system"
) -> dict[str, Any]:
    customer = _authorize(session_id, authenticated_customer_id, customer_id, "account_status")

    def operation() -> dict[str, Any]:
        with Session(engine) as session:
            account = session.exec(select(BillingAccount).where(BillingAccount.customer_id == customer)).first()
            subscription = session.exec(select(Subscription).where(Subscription.customer_id == customer)).first()
            if account is None or subscription is None:
                raise ToolNotFound("Account was not found")
            return {
                "customer_id": customer,
                "account_status": account.status,
                "plan_id": subscription.plan_id,
                "subscription_status": subscription.status,
                "contract_type": subscription.contract_type,
                "monthly_charge": account.monthly_charge,
                "balance": account.balance,
            }

    return _run_tool(session_id, "account_status", operation)


def get_payment(
    payment_id: str, customer_id: str, authenticated_customer_id: str, *, session_id: str = "system"
) -> dict[str, Any]:
    customer = _authorize(session_id, authenticated_customer_id, customer_id, "payment")
    payment = _normalise_resource_id(payment_id, "PAY")

    def operation() -> dict[str, Any]:
        with Session(engine) as session:
            record = _owned_one(session, Payment, Payment.payment_id, payment, customer, "Payment")
            return {
                "payment_id": record.payment_id,
                "invoice_id": record.invoice_id,
                "payment_date": record.payment_date.isoformat(),
                "amount": record.amount,
                "payment_method": record.payment_method,
                "status": record.status,
                "failure_reason": record.failure_reason,
            }

    return _run_tool(session_id, "payment_lookup", operation)


def get_customer_payments(
    customer_id: str, authenticated_customer_id: str, *, session_id: str = "system"
) -> dict[str, Any]:
    customer = _authorize(session_id, authenticated_customer_id, customer_id, "payment_history")

    def operation() -> dict[str, Any]:
        with Session(engine) as session:
            records = session.exec(select(Payment).where(Payment.customer_id == customer)).all()
            return {
                "payments": [
                    {
                        "payment_id": item.payment_id,
                        "date": item.payment_date.isoformat(),
                        "amount": item.amount,
                        "status": item.status,
                    }
                    for item in records[:10]
                ]
            }

    return _run_tool(session_id, "payment_history", operation)


def get_invoice(
    invoice_id: str, customer_id: str, authenticated_customer_id: str, *, session_id: str = "system"
) -> dict[str, Any]:
    customer = _authorize(session_id, authenticated_customer_id, customer_id, "invoice")
    invoice = _normalise_resource_id(invoice_id, "INV")

    def operation() -> dict[str, Any]:
        with Session(engine) as session:
            record = _owned_one(session, Invoice, Invoice.invoice_id, invoice, customer, "Invoice")
            return {
                "invoice_id": record.invoice_id,
                "due_date": record.due_date.isoformat(),
                "total_amount": record.total_amount,
                "balance_due": record.balance_due,
                "status": record.status,
            }

    return _run_tool(session_id, "invoice_lookup", operation)


def get_customer_transactions(
    customer_id: str, authenticated_customer_id: str, *, session_id: str = "system"
) -> dict[str, Any]:
    customer = _authorize(session_id, authenticated_customer_id, customer_id, "transaction_history")

    def operation() -> dict[str, Any]:
        with Session(engine) as session:
            records = session.exec(select(Transaction).where(Transaction.customer_id == customer)).all()
            return {
                "transactions": [
                    {
                        "transaction_id": item.transaction_id,
                        "payment_id": item.payment_id,
                        "amount": item.amount,
                        "status": item.status,
                    }
                    for item in records[:10]
                ]
            }

    return _run_tool(session_id, "transaction_history", operation)


def get_order(
    order_id: str, customer_id: str, authenticated_customer_id: str, *, session_id: str = "system"
) -> dict[str, Any]:
    customer = _authorize(session_id, authenticated_customer_id, customer_id, "order")
    order = _normalise_resource_id(order_id, "ORD")

    def operation() -> dict[str, Any]:
        with Session(engine) as session:
            record = _owned_one(session, CustomerOrder, CustomerOrder.order_id, order, customer, "Order")
            return {
                "order_id": record.order_id,
                "order_date": record.order_date.isoformat(),
                "order_type": record.order_type,
                "status": record.status,
                "tracking_number": record.tracking_number,
            }

    return _run_tool(session_id, "order_lookup", operation)


def get_customer_tickets(
    customer_id: str, authenticated_customer_id: str, *, session_id: str = "system"
) -> dict[str, Any]:
    customer = _authorize(session_id, authenticated_customer_id, customer_id, "ticket_history")

    def operation() -> dict[str, Any]:
        with Session(engine) as session:
            records = session.exec(select(SupportTicket).where(SupportTicket.customer_id == customer)).all()
            return {
                "tickets": [
                    {"ticket_id": item.ticket_id, "subject": item.subject, "status": item.status}
                    for item in records[:10]
                ]
            }

    return _run_tool(session_id, "ticket_history", operation)


def get_ticket(
    ticket_id: str, customer_id: str, authenticated_customer_id: str, *, session_id: str = "system"
) -> dict[str, Any]:
    customer = _authorize(session_id, authenticated_customer_id, customer_id, "ticket")
    ticket = _normalise_resource_id(ticket_id, "TKT")

    def operation() -> dict[str, Any]:
        with Session(engine) as session:
            record = _owned_one(session, SupportTicket, SupportTicket.ticket_id, ticket, customer, "Ticket")
            return {
                "ticket_id": record.ticket_id,
                "category": record.category,
                "subject": record.subject,
                "status": record.status,
                "resolution": record.resolution,
            }

    return _run_tool(session_id, "ticket_lookup", operation)


def create_support_ticket(
    customer_id: str,
    authenticated_customer_id: str,
    category: str,
    subject: str,
    description: str,
    *,
    session_id: str = "system",
) -> dict[str, Any]:
    """State-changing tool with explicit verified-owner authorization and bounded parameters."""
    customer = _authorize(session_id, authenticated_customer_id, customer_id, "create_support_ticket")
    if (
        category not in {"billing", "technical_issue", "account_access", "order_status", "cancellation"}
        or not (1 <= len(subject) <= 120)
        or not (1 <= len(description) <= 1000)
    ):
        raise ToolValidationError("Invalid ticket details")

    def operation() -> dict[str, Any]:
        with Session(engine) as session:
            sequence = session.exec(select(SupportTicket).where(SupportTicket.customer_id == customer)).all()
            ticket_id = f"TKTNEW{len(sequence) + 1:04d}"
            now = datetime.utcnow()
            session.add(
                SupportTicket(
                    ticket_id=ticket_id,
                    customer_id=customer,
                    category=category,
                    subject=subject,
                    description=description,
                    priority="normal",
                    status="open",
                    assigned_team="Customer Support",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()
            return {"ticket_id": ticket_id, "status": "open"}

    return _run_tool(session_id, "create_support_ticket", operation)
