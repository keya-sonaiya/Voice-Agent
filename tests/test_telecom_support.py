"""Integration tests for the relational IBM Telco import and authorized support tools."""

from collections.abc import Iterator

import pytest
from sqlmodel import Session

from app.graph import build_graph as graph_module
from app.graph.nodes import support_workflow
from app.graph.state import GroundingResult, IntentResult
from app.persistence.models import Customer, Payment, SourceCustomerRecord
from app.persistence.telecom_seed import build_engine, initialize_telecom_database
from app.tools import customer_support

from .helpers import state


@pytest.fixture(scope="module")
def seeded_engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    database_path = tmp_path_factory.mktemp("telecom") / "telecom.db"
    database_url = f"sqlite:///{database_path.as_posix()}"
    statistics = initialize_telecom_database(database_url)
    assert statistics["customers"] == 7043
    assert statistics["source_records"] == 7043
    assert statistics["payments"] >= 7044
    yield build_engine(database_url)


@pytest.fixture
def support_tools(monkeypatch: pytest.MonkeyPatch, seeded_engine: object) -> None:
    monkeypatch.setattr(customer_support, "engine", seeded_engine)


def test_source_rows_are_preserved_and_mapped_to_deterministic_support_customers(seeded_engine: object) -> None:
    with Session(seeded_engine) as session:
        customer = session.get(Customer, "CUST1024")
        source = session.get(SourceCustomerRecord, customer.source_customer_id if customer else "")
        payment = session.get(Payment, "PAY102938")
        foreign_key_violations = session.connection().exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    assert customer is not None
    assert customer.full_name == "Samad Sama"
    assert source is not None and "customerID" in source.raw_json
    assert payment is not None and payment.customer_id == "CUST1024"
    assert foreign_key_violations == []


def test_identity_verification_and_owned_payment_lookup(support_tools: None) -> None:
    assert customer_support.verify_customer_identity("CUST1024", "Samad Sama") is True
    assert customer_support.verify_customer_identity("CUST1024", "Wrong Name") is False
    payment = customer_support.get_payment("PAY102938", "CUST1024", "CUST1024")
    assert payment["status"] == "declined"
    assert payment["failure_reason"] == "Issuer declined the transaction."


def test_unauthorized_or_malformed_lookups_do_not_disclose_records(support_tools: None) -> None:
    with pytest.raises(customer_support.ToolAccessDenied):
        customer_support.get_payment("PAY102938", "CUST1024", "CUST0001")
    with pytest.raises(customer_support.ToolAccessDenied):
        customer_support.get_order("ORD000007", "CUST0007", "CUST0001")
    with pytest.raises(customer_support.ToolValidationError):
        customer_support.get_invoice("not-an-invoice", "CUST1024", "CUST1024")


def test_invoice_order_ticket_and_transaction_tools_enforce_customer_scope(support_tools: None) -> None:
    invoice = customer_support.get_invoice("INV001024", "CUST1024", "CUST1024")
    order = customer_support.get_order("ORD000007", "CUST0007", "CUST0007")
    ticket = customer_support.get_ticket("TKT000010", "CUST0010", "CUST0010")
    transactions = customer_support.get_customer_transactions("CUST1024", "CUST1024")
    assert invoice["invoice_id"] == "INV001024"
    assert order["status"] == "in_transit"
    assert ticket["status"] == "resolved"
    assert transactions["transactions"][0]["payment_id"] == "PAY102938"


def test_payment_conversation_requires_verification_then_uses_owned_record(
    monkeypatch: pytest.MonkeyPatch, seeded_engine: object
) -> None:
    monkeypatch.setattr(customer_support, "engine", seeded_engine)
    current = {
        **state("I am having trouble with my payment."),
        "intent_result": IntentResult(intent="billing", confidence=0.98),
    }
    first = support_workflow.handle_support_turn(current)
    assert first is not None and "customer ID" in first["draft_answer"]
    current = {**current, **first, "current_transcript": "CUST1024"}
    second = support_workflow.handle_support_turn(current)
    assert second is not None and "name on the account" in second["draft_answer"]
    current = {**current, **second, "current_transcript": "Samad Sama"}
    third = support_workflow.handle_support_turn(current)
    assert third is not None and "verified your account" in third["draft_answer"]
    current = {**current, **third, "current_transcript": "PAY102938"}
    fourth = support_workflow.handle_support_turn(current)
    assert fourth is not None
    assert "declined" in fourth["draft_answer"]
    assert fourth["retrieved_excerpts"]


def test_payment_workflow_reaches_graph_response_after_authorized_lookup(
    monkeypatch: pytest.MonkeyPatch, seeded_engine: object
) -> None:
    """Exercise intent → sentiment → workflow → grounding → response without a provider call."""
    monkeypatch.setattr(customer_support, "engine", seeded_engine)
    monkeypatch.setattr(graph_module, "record_transition", lambda *_: None)
    monkeypatch.setattr(
        graph_module.grounding_judge,
        "check_grounding",
        lambda _: {"grounding_result": GroundingResult(is_grounded=True, reason="Test support evidence")},
    )
    graph = graph_module.build_graph()
    first = graph.invoke(
        {
            **state("I am having trouble with my payment."),
            "intent_result": IntentResult(intent="billing", confidence=0.98),
        }
    )
    assert "customer ID" in first["final_response_text"]
    second = graph.invoke({**first, "current_transcript": "CUST1024"})
    assert "name on the account" in second["final_response_text"]
    third = graph.invoke({**second, "current_transcript": "Samad Sama"})
    assert "verified your account" in third["final_response_text"]
    fourth = graph.invoke({**third, "current_transcript": "PAY102938"})
    assert "Payment PAY102938 was declined" in fourth["final_response_text"]
    assert fourth["system_failure"] is None
