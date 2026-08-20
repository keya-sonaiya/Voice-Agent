"""Server-controlled customer verification and account lookup workflow.

This module—not an LLM—chooses and invokes database tools.  It only calls a
customer-specific lookup after successful name verification, and every tool repeats
the ownership check in its own backend implementation.
"""

import re
from datetime import UTC, datetime

from app.call_logging import call_exception, call_log
from app.graph.state import ConversationState
from app.identifiers import IdentifierType, mask_identifier, normalize_identifier
from app.tools import customer_support
from app.tools.customer_support import ToolAccessDenied, ToolError, ToolNotFound, ToolValidationError

_CUSTOMER_ID = re.compile(r"\b(CUST\d{4})\b", re.IGNORECASE)
_PAYMENT_ID = re.compile(r"\b(PAY[A-Z0-9]{6,})\b", re.IGNORECASE)
_INVOICE_ID = re.compile(r"\b(INV[A-Z0-9]{6,})\b", re.IGNORECASE)
_ORDER_ID = re.compile(r"\b(ORD[A-Z0-9]{6,})\b", re.IGNORECASE)
_TICKET_ID = re.compile(r"\b(TKT[A-Z0-9]{6,})\b", re.IGNORECASE)
_TOOL_INTENTS = {"billing", "order_status", "account_access", "technical_issue", "cancellation"}

_ASK_CUSTOMER_ID = "I can help with that. Could you provide your customer ID?"
_ASK_ACCOUNT_NAME = "For verification, could you tell me the name on the account?"
_VERIFIED_PAYMENT = "Thanks, I've verified your account. What payment or invoice are you having trouble with?"
_ASK_PAYMENT_ID = "Please provide the payment or invoice ID so I can look it up."
_ASK_ORDER_ID = "Please provide the order ID so I can look it up."
_VERIFICATION_FAILED = "I couldn't verify that account. Please check the customer ID and name, then try again."
_RECOVERY_NAME = "That's okay. I can help you find your account. What's the name on the account?"
_RECOVERY_CONTACT = "Thanks. To verify the account, could you provide the phone number or email associated with it?"
_RECOVERY_RETRY = "I couldn't verify that account with those details. Please provide the phone number or email associated with it."
_FORGOT_ID = re.compile(r"\b(?:forgot|don't know|do not know|lost)\b.{0,30}\b(?:customer|account)\s+id\b", re.I)
_DIFFERENT_ACCOUNT = re.compile(r"\b(?:different|another|new)\s+account\b", re.I)


def _match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).upper() if match else None


def _identifier(text: str, identifier_type: IdentifierType, session_id: str) -> str | None:
    call_log(session_id, "IDENTIFIER", "raw_received", details={"type": identifier_type, "value_length": len(text)})
    call_log(session_id, "IDENTIFIER", "normalization_start", details={"type": identifier_type})
    normalized = normalize_identifier(text, identifier_type)
    call_log(
        session_id,
        "IDENTIFIER",
        "normalized",
        details={"type": identifier_type, "value": mask_identifier(normalized), "valid_shape": normalized is not None},
    )
    call_log(session_id, "IDENTIFIER", "validation", details={"type": identifier_type, "valid": normalized is not None})
    return normalized


def _support_intent(state: ConversationState) -> str | None:
    current = state["support_intent"]
    if current:
        return current
    result = state["intent_result"]
    if result is not None and result.intent in _TOOL_INTENTS:
        return result.intent
    return None


def is_support_workflow_turn(state: ConversationState) -> bool:
    """Route short identifiers/names back to the active verified-support workflow."""
    return bool(
        _support_intent(state)
        or state["awaiting_customer_verification"]
        or state["awaiting_payment_id"]
        or _match(_CUSTOMER_ID, state["current_transcript"])
        or _match(_PAYMENT_ID, state["current_transcript"])
        or _match(_INVOICE_ID, state["current_transcript"])
        or _match(_ORDER_ID, state["current_transcript"])
        or _match(_TICKET_ID, state["current_transcript"])
    )


def _prompt(text: str, intent: str | None, **updates: object) -> dict[str, object]:
    return {
        "draft_answer": text,
        "retrieved_excerpts": [],
        "response_mode": "support_workflow",
        "support_intent": intent,
        "system_failure": None,
        **updates,
    }


def _tool_answer(text: str, evidence: str, intent: str, **updates: object) -> dict[str, object]:
    return {
        "draft_answer": text,
        "retrieved_excerpts": [evidence],
        "response_mode": "support_workflow",
        "support_intent": intent,
        "system_failure": None,
        **updates,
    }


def _handle_tool_error(state: ConversationState, error: ToolError, intent: str) -> dict[str, object]:
    call_log(state["session_id"], "TOOL", "user_safe_failure", details={"tool_intent": intent})
    return _prompt(error.public_message, intent)


def _identity_change(state: ConversationState, current: str, **updates: object) -> dict[str, object]:
    previous = state["identity_state"]
    if previous != current:
        call_log(state["session_id"], "IDENTITY", "state_change", details={"previous": previous, "current": current})
    return updates


def handle_support_turn(state: ConversationState) -> dict[str, object] | None:
    """Advance a verified customer workflow or return ``None`` for ordinary RAG requests."""
    intent = _support_intent(state)
    if intent is None:
        return None
    transcript = state["current_transcript"]
    if state["customer_verified"] and _DIFFERENT_ACCOUNT.search(transcript):
        call_log(state["session_id"], "IDENTITY", "state", details={"identity_state": "unidentified"})
        return _prompt(
            _ASK_CUSTOMER_ID,
            intent,
            **_identity_change(
                state,
                "unidentified",
                customer_id=None,
                customer_identified=False,
                customer_verified=False,
                identity_state="unidentified",
                verification_method=None,
                verification_timestamp=None,
                awaiting_customer_verification=True,
                awaiting_customer_name=False,
                awaiting_customer_phone=False,
                awaiting_customer_email=False,
            ),
        )
    if not state["customer_verified"] and state["customer_id"] is None:
        call_log(state["session_id"], "WORKFLOW", "awaiting_identifier", details={"type": "customer_id"})
    elif state["customer_verified"] and state["awaiting_payment_id"]:
        call_log(state["session_id"], "IDENTITY", "session_reuse", details={"identity_state": state["identity_state"]})
        call_log(state["session_id"], "WORKFLOW", "awaiting_identifier", details={"type": "payment_or_invoice_id"})
    customer_id = state["customer_id"]
    try:
        if not state["customer_verified"] and not state["account_recovery_active"] and customer_id is None and _FORGOT_ID.search(transcript):
            call_log(state["session_id"], "IDENTITY", "recovery_started")
            return _prompt(
                _RECOVERY_NAME,
                intent,
                account_recovery_active=True,
                awaiting_customer_name=True,
                awaiting_customer_verification=True,
                account_recovery_attempts=0,
            )

        if state["account_recovery_active"]:
            if state["awaiting_customer_name"]:
                candidates = customer_support.find_customer_candidates(transcript, session_id=state["session_id"])
                if not candidates:
                    return _prompt(
                        "I couldn't find an account with that name. What's the name on the account?",
                        intent,
                        awaiting_customer_name=True,
                        awaiting_customer_phone=False,
                        awaiting_customer_email=False,
                        account_recovery_attempts=state["account_recovery_attempts"] + 1,
                    )
                if len(candidates) != 1:
                    return _prompt(
                        "I couldn't find one unique account with that name. Could you provide the phone number or email associated with it?",
                        intent,
                        awaiting_customer_name=False,
                        awaiting_customer_phone=True,
                        awaiting_customer_email=True,
                        recovery_candidate_ids=candidates,
                        account_recovery_attempts=state["account_recovery_attempts"] + 1,
                    )
                return _prompt(
                    _RECOVERY_CONTACT,
                    intent,
                    awaiting_customer_name=False,
                    awaiting_customer_phone=True,
                    awaiting_customer_email=True,
                    recovery_candidate_ids=candidates,
                )
            verified_id = customer_support.verify_recovery_candidate(
                state["recovery_candidate_ids"], transcript, session_id=state["session_id"]
            )
            if verified_id is None:
                return _prompt(
                    _RECOVERY_RETRY,
                    intent,
                    account_recovery_attempts=state["account_recovery_attempts"] + 1,
                    awaiting_customer_phone=True,
                    awaiting_customer_email=True,
                )
            call_log(state["session_id"], "IDENTITY", "state_change", details={"previous": "unidentified", "current": "verified"})
            return _prompt(
                "I've verified your account. How can I help you?",
                intent,
                customer_id=verified_id,
                customer_identified=True,
                customer_verified=True,
                identity_state="verified",
                verification_method="name_and_contact",
                verification_timestamp=datetime.now(UTC).isoformat(),
                account_recovery_active=False,
                awaiting_customer_name=False,
                awaiting_customer_phone=False,
                awaiting_customer_email=False,
                awaiting_customer_verification=False,
                recovery_candidate_ids=[],
            )

        if not state["customer_verified"]:
            if customer_id is None:
                supplied_id = _identifier(transcript, "customer_id", state["session_id"])
                if supplied_id is None:
                    return _prompt(
                        _ASK_CUSTOMER_ID,
                        intent,
                        awaiting_customer_verification=True,
                        awaiting_payment_id=False,
                    )
                supplied_id = customer_support.validate_customer_id(supplied_id, session_id=state["session_id"])
                return _prompt(
                    _ASK_ACCOUNT_NAME,
                    intent,
                    customer_id=supplied_id,
                    customer_identified=True,
                    identity_state="identified",
                    awaiting_customer_verification=True,
                    awaiting_payment_id=False,
                )
            # The customer ID was received on a previous turn; this turn supplies the account name.
            if customer_support.verify_customer_identity(customer_id, transcript, session_id=state["session_id"]):
                if intent == "billing":
                    return _prompt(
                        _VERIFIED_PAYMENT,
                        intent,
                        customer_verified=True,
                        customer_identified=True,
                        identity_state="verified",
                        verification_method="customer_id_and_name",
                        verification_timestamp=datetime.now(UTC).isoformat(),
                        awaiting_customer_verification=False,
                        awaiting_payment_id=True,
                    )
                if intent == "order_status":
                    return _prompt(
                        "Thanks, I've verified your account. Please provide the order ID you want me to check.",
                        intent,
                        customer_verified=True,
                        customer_identified=True,
                        identity_state="verified",
                        verification_method="customer_id_and_name",
                        verification_timestamp=datetime.now(UTC).isoformat(),
                        awaiting_customer_verification=False,
                    )
                if intent == "account_access":
                    account = customer_support.get_account_status(
                        customer_id, customer_id, session_id=state["session_id"]
                    )
                    return _tool_answer(
                        "Your account is "
                        f"{account['account_status']} and your subscription is {account['subscription_status']}.",
                        f"Authoritative account record: {account}",
                        intent,
                        customer_verified=True,
                        customer_identified=True,
                        identity_state="verified",
                        verification_method="customer_id_and_name",
                        verification_timestamp=datetime.now(UTC).isoformat(),
                        awaiting_customer_verification=False,
                    )
                return _prompt(
                    "Thanks, I've verified your account. Please describe the issue in a little more detail.",
                    intent,
                    customer_verified=True,
                    customer_identified=True,
                    identity_state="verified",
                    verification_method="customer_id_and_name",
                    verification_timestamp=datetime.now(UTC).isoformat(),
                    awaiting_customer_verification=False,
                )
            return _prompt(
                _VERIFICATION_FAILED,
                intent,
                customer_id=None,
                customer_identified=False,
                customer_verified=False,
                identity_state="unidentified",
                verification_method=None,
                verification_timestamp=None,
                awaiting_customer_verification=True,
            )

        # From this point onward every lookup receives the verified customer ID and
        # enforces ownership again inside the tool service.
        assert customer_id is not None
        payment_id = _identifier(transcript, "payment_id", state["session_id"]) if state["awaiting_payment_id"] else None
        invoice_id = _identifier(transcript, "invoice_id", state["session_id"]) if state["awaiting_payment_id"] else None
        order_id = _identifier(transcript, "order_id", state["session_id"])
        ticket_id = _identifier(transcript, "ticket_id", state["session_id"])
        if payment_id:
            payment = customer_support.get_payment(payment_id, customer_id, customer_id, session_id=state["session_id"])
            reason = f" The recorded reason is {payment['failure_reason']}" if payment["failure_reason"] else ""
            return _tool_answer(
                f"Payment {payment['payment_id']} was {payment['status']} on {payment['payment_date']}.{reason}",
                f"Authoritative payment record: {payment}",
                intent,
                current_payment_id=payment_id,
                awaiting_payment_id=False,
            )
        if invoice_id:
            invoice = customer_support.get_invoice(invoice_id, customer_id, customer_id, session_id=state["session_id"])
            return _tool_answer(
                f"Invoice {invoice['invoice_id']} is {invoice['status']} and has a balance due of "
                f"${invoice['balance_due']:.2f} on {invoice['due_date']}.",
                f"Authoritative invoice record: {invoice}",
                intent,
                current_invoice_id=invoice_id,
                awaiting_payment_id=False,
            )
        if order_id:
            order = customer_support.get_order(order_id, customer_id, customer_id, session_id=state["session_id"])
            tracking = f" Tracking number: {order['tracking_number']}." if order["tracking_number"] else ""
            return _tool_answer(
                f"Order {order['order_id']} is {order['status']}.{tracking}",
                f"Authoritative order record: {order}",
                intent,
                current_order_id=order_id,
            )
        if ticket_id:
            ticket = customer_support.get_ticket(ticket_id, customer_id, customer_id, session_id=state["session_id"])
            resolution = f" Resolution: {ticket['resolution']}" if ticket["resolution"] else ""
            return _tool_answer(
                f"Ticket {ticket['ticket_id']} is {ticket['status']}.{resolution}",
                f"Authoritative ticket record: {ticket}",
                intent,
                current_ticket_id=ticket_id,
            )
        if intent == "billing":
            return _prompt(_ASK_PAYMENT_ID, intent, awaiting_payment_id=True)
        if intent == "order_status":
            return _prompt(_ASK_ORDER_ID, intent)
    except ToolError as error:
        return _handle_tool_error(state, error, intent)
    except Exception:
        call_exception(state["session_id"], "TOOL", "workflow_failed", details={"tool_intent": intent})
        return _prompt("I'm unable to retrieve that information right now.", intent)
    return None


def is_deterministic_support_prompt(state: ConversationState) -> bool:
    """Only these fixed, non-record-specific prompts may bypass the model grounding call."""
    return state["response_mode"] == "support_workflow" and state["draft_answer"] in {
        _ASK_CUSTOMER_ID,
        _ASK_ACCOUNT_NAME,
        _VERIFIED_PAYMENT,
        _ASK_PAYMENT_ID,
        _ASK_ORDER_ID,
        _VERIFICATION_FAILED,
        "Thanks, I've verified your account. Please provide the order ID you want me to check.",
        "Thanks, I've verified your account. Please describe the issue in a little more detail.",
        ToolError.public_message,
        ToolAccessDenied.public_message,
        ToolNotFound.public_message,
        ToolValidationError.public_message,
    }
