"""Reproducible IBM Telco import and deterministic operational demo-data seeding."""

import csv
import json
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine, delete, func, select

from app.persistence.models import (
    BillingAccount,
    Customer,
    CustomerInteraction,
    CustomerOrder,
    CustomerService,
    Invoice,
    Payment,
    Plan,
    SourceCustomerRecord,
    Subscription,
    SupportTicket,
    TicketMessage,
    Transaction,
)

CSV_PATH = Path(__file__).parents[2] / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
AS_OF_DATE = date(2026, 8, 1)
_FIRST_NAMES = ("Avery", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Jamie", "Robin")
_LAST_NAMES = ("Patel", "Smith", "Johnson", "Garcia", "Brown", "Wilson", "Lee", "Davis")


def build_engine(database_url: str) -> Engine:
    """Create a SQLite engine with mandatory foreign-key checks enabled."""
    engine = create_engine(database_url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection: object, _: object) -> None:
        cursor = cast(Any, dbapi_connection).cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def _yes(value: str) -> bool:
    return value.strip().lower() == "yes"


def _number(value: str) -> float:
    """IBM's eleven blank ``TotalCharges`` values are represented as zero, not guessed."""
    return float(value.strip()) if value.strip() else 0.0


def _customer_name(position: int) -> tuple[str, str]:
    if position == 1024:
        return "Samad", "Sama"
    return (
        _FIRST_NAMES[(position - 1) % len(_FIRST_NAMES)],
        _LAST_NAMES[((position - 1) // len(_FIRST_NAMES)) % len(_LAST_NAMES)],
    )


def _support_customer_id(position: int) -> str:
    return f"CUST{position:04d}"


def _plans() -> list[Plan]:
    return [
        Plan(
            plan_id="PLAN-VOICE",
            plan_name="Voice Essentials",
            monthly_price=25.0,
            service_type="phone",
            description="Phone service without internet.",
        ),
        Plan(
            plan_id="PLAN-DSL",
            plan_name="DSL Connect",
            monthly_price=55.0,
            service_type="dsl",
            description="DSL internet and optional services.",
        ),
        Plan(
            plan_id="PLAN-FIBER",
            plan_name="Fiber Connect",
            monthly_price=85.0,
            service_type="fiber",
            description="Fiber internet and optional services.",
        ),
    ]


def _plan_id(row: dict[str, str]) -> str:
    return {"DSL": "PLAN-DSL", "Fiber optic": "PLAN-FIBER"}.get(row["InternetService"], "PLAN-VOICE")


def _clear_telecom_data(session: Session) -> None:
    """Explicit reset path; call snapshots deliberately remain untouched."""
    for model in (
        TicketMessage,
        CustomerInteraction,
        Transaction,
        Payment,
        Invoice,
        CustomerOrder,
        SupportTicket,
        BillingAccount,
        CustomerService,
        Subscription,
        Customer,
        Plan,
        SourceCustomerRecord,
    ):
        session.execute(delete(model))


def _count(session: Session, model: type[SQLModel]) -> int:
    return int(session.exec(select(func.count()).select_from(model)).one())


def _statistics(session: Session) -> dict[str, int]:
    return {
        "customers": _count(session, Customer),
        "source_records": _count(session, SourceCustomerRecord),
        "plans": _count(session, Plan),
        "subscriptions": _count(session, Subscription),
        "services": _count(session, CustomerService),
        "billing_accounts": _count(session, BillingAccount),
        "invoices": _count(session, Invoice),
        "payments": _count(session, Payment),
        "transactions": _count(session, Transaction),
        "orders": _count(session, CustomerOrder),
        "tickets": _count(session, SupportTicket),
        "ticket_messages": _count(session, TicketMessage),
        "interactions": _count(session, CustomerInteraction),
    }


def _integrity_check(session: Session) -> None:
    """Fail initialization if SQLite reports any broken relationship."""
    violations = session.connection().exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise RuntimeError(f"Foreign-key integrity check failed: {violations[:3]}")
    if _count(session, Customer) != _count(session, SourceCustomerRecord):
        raise RuntimeError("Customer/source-record count mismatch.")


def _add_operational_scenarios(session: Session) -> None:
    """Add fixed records that make demonstrations repeatable; none come from IBM."""
    order_scenarios = (
        ("ORD000007", "CUST0007", "in_transit", "TRK000007"),
        ("ORD000008", "CUST0008", "delivered", "TRK000008"),
        ("ORD000009", "CUST0009", "cancelled", None),
    )
    for order_id, customer_id, status, tracking in order_scenarios:
        session.add(
            CustomerOrder(
                order_id=order_id,
                customer_id=customer_id,
                order_date=AS_OF_DATE - timedelta(days=10),
                order_type="router",
                status=status,
                total_amount=99.99,
                shipping_address="Demo address withheld from agent responses",
                tracking_number=tracking,
            )
        )
    ticket_scenarios = (
        ("TKT000010", "CUST0010", "technical_issue", "Internet outage", "resolved", "Network Operations"),
        ("TKT000011", "CUST0011", "billing", "Payment review", "open", "Billing Support"),
        ("TKT000012", "CUST0012", "account_access", "Login assistance", "resolved", "Account Support"),
    )
    now = datetime(2026, 7, 25, 9, 0, 0)
    for ticket_id, customer_id, category, subject, status, team in ticket_scenarios:
        session.add(
            SupportTicket(
                ticket_id=ticket_id,
                customer_id=customer_id,
                category=category,
                subject=subject,
                description="Deterministic synthetic demo support record.",
                priority="normal",
                status=status,
                assigned_team=team,
                created_at=now,
                updated_at=now,
                resolution="Resolved during deterministic demo seeding." if status == "resolved" else None,
            )
        )
        session.add(
            TicketMessage(
                message_id=f"MSG{ticket_id[3:]}",
                ticket_id=ticket_id,
                sender_type="agent",
                message="This is a deterministic synthetic ticket message.",
                created_at=now,
            )
        )
    session.add(
        CustomerInteraction(
            interaction_id="INT001024",
            customer_id="CUST1024",
            channel="voice",
            interaction_type="payment_support",
            summary="Deterministic demo payment-support scenario.",
            created_at=now,
        )
    )


def initialize_telecom_database(database_url: str, *, reset: bool = False, csv_path: Path = CSV_PATH) -> dict[str, int]:
    """Create schema and seed once; ``reset`` is explicit and never removes call snapshots."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"IBM Telco source CSV was not found: {csv_path}")
    engine = build_engine(database_url)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        existing_customers = _count(session, Customer)
        if existing_customers and not reset:
            raise RuntimeError("Telecom data already exists. Use the explicit --reset option to rebuild it.")
        if reset:
            _clear_telecom_data(session)
            session.commit()
        session.add_all(_plans())
        session.flush()
        with csv_path.open(newline="", encoding="utf-8-sig") as source:
            for position, row in enumerate(csv.DictReader(source), start=1):
                customer_id = _support_customer_id(position)
                first_name, last_name = _customer_name(position)
                full_name = f"{first_name} {last_name}"
                tenure = int(row["tenure"])
                monthly_charge = _number(row["MonthlyCharges"])
                total_charges = _number(row["TotalCharges"])
                churned = _yes(row["Churn"])
                account_status = "cancelled" if churned else "active"
                session.add(
                    SourceCustomerRecord(source_customer_id=row["customerID"], raw_json=json.dumps(row, sort_keys=True))
                )
                session.add(
                    Customer(
                        customer_id=customer_id,
                        source_customer_id=row["customerID"],
                        first_name=first_name,
                        last_name=last_name,
                        full_name=full_name,
                        email=f"{customer_id.lower()}@demo.telco.invalid",
                        phone=f"+1-555-{position:04d}",
                        gender=row["gender"],
                        senior_citizen=row["SeniorCitizen"] == "1",
                        partner=_yes(row["Partner"]),
                        dependents=_yes(row["Dependents"]),
                        account_status=account_status,
                        churned=churned,
                    )
                )
                # SQLModel models intentionally have no ORM relationships: persist the
                # parent before accumulating the source-derived child records.
                session.flush()
                session.add(
                    Subscription(
                        subscription_id=f"SUB{position:06d}",
                        customer_id=customer_id,
                        plan_id=_plan_id(row),
                        start_date=AS_OF_DATE - timedelta(days=30 * tenure),
                        end_date=AS_OF_DATE if churned else None,
                        contract_type=row["Contract"],
                        status="cancelled" if churned else "active",
                        tenure_months=tenure,
                    )
                )
                session.add(
                    CustomerService(
                        service_id=f"SVC{position:06d}",
                        customer_id=customer_id,
                        phone_service=_yes(row["PhoneService"]),
                        multiple_lines=row["MultipleLines"],
                        internet_service=row["InternetService"],
                        online_security=row["OnlineSecurity"],
                        online_backup=row["OnlineBackup"],
                        device_protection=row["DeviceProtection"],
                        tech_support=row["TechSupport"],
                        streaming_tv=row["StreamingTV"],
                        streaming_movies=row["StreamingMovies"],
                    )
                )
                invoice_id = f"INV{position:06d}"
                payment_id = f"PAY{position:06d}"
                payment_status, failure_reason = "successful", None
                invoice_status, amount_paid = "paid", monthly_charge
                if position == 2:
                    payment_status, failure_reason, invoice_status, amount_paid = (
                        "failed",
                        "Payment processor unavailable.",
                        "open",
                        0.0,
                    )
                elif position == 3:
                    payment_status, invoice_status, amount_paid = "pending", "pending", 0.0
                elif position == 4:
                    payment_status, invoice_status = "refunded", "refunded"
                elif position == 6:
                    payment_status, failure_reason, invoice_status, amount_paid = (
                        "failed",
                        "Invoice is overdue.",
                        "overdue",
                        0.0,
                    )
                elif position == 1024:
                    payment_id = "PAY102938"
                    payment_status, failure_reason, invoice_status, amount_paid = (
                        "declined",
                        "Issuer declined the transaction.",
                        "open",
                        0.0,
                    )
                balance_due = round(max(monthly_charge - amount_paid, 0.0), 2)
                session.add(
                    BillingAccount(
                        billing_account_id=f"BILL{position:06d}",
                        customer_id=customer_id,
                        billing_cycle="monthly",
                        payment_method=row["PaymentMethod"],
                        paperless_billing=_yes(row["PaperlessBilling"]),
                        monthly_charge=monthly_charge,
                        total_charges=total_charges,
                        balance=balance_due,
                        status="overdue" if invoice_status == "overdue" else "active",
                    )
                )
                session.add(
                    Invoice(
                        invoice_id=invoice_id,
                        customer_id=customer_id,
                        billing_account_id=f"BILL{position:06d}",
                        invoice_date=AS_OF_DATE - timedelta(days=15),
                        billing_period_start=AS_OF_DATE - timedelta(days=45),
                        billing_period_end=AS_OF_DATE - timedelta(days=15),
                        due_date=(
                            AS_OF_DATE - timedelta(days=1)
                            if invoice_status == "overdue"
                            else AS_OF_DATE + timedelta(days=15)
                        ),
                        subtotal=monthly_charge,
                        tax=0.0,
                        total_amount=monthly_charge,
                        amount_paid=amount_paid,
                        balance_due=balance_due,
                        status=invoice_status,
                    )
                )
                session.add(
                    Payment(
                        payment_id=payment_id,
                        customer_id=customer_id,
                        invoice_id=invoice_id,
                        payment_date=AS_OF_DATE - timedelta(days=12),
                        amount=monthly_charge,
                        payment_method=row["PaymentMethod"],
                        status=payment_status,
                        failure_reason=failure_reason,
                        reference_number=f"REF{position:08d}",
                    )
                )
                session.add(
                    Transaction(
                        transaction_id=f"TXN{position:06d}",
                        customer_id=customer_id,
                        payment_id=payment_id,
                        transaction_type="payment",
                        amount=monthly_charge,
                        timestamp=datetime(2026, 7, 20, 10, 0, 0),
                        status=payment_status,
                        description="Deterministic synthetic payment transaction.",
                        external_reference=f"EXT{position:08d}",
                    )
                )
        # A second, similarly valued payment makes the duplicate-charge demo deterministic.
        session.add(
            Payment(
                payment_id="PAYDUP0005",
                customer_id="CUST0005",
                invoice_id="INV000005",
                payment_date=AS_OF_DATE - timedelta(days=11),
                amount=10.0,
                payment_method="Electronic check",
                status="successful",
                reference_number="REFDUP0005",
            )
        )
        session.add(
            Transaction(
                transaction_id="TXNDUP0005",
                customer_id="CUST0005",
                payment_id="PAYDUP0005",
                transaction_type="payment",
                amount=10.0,
                timestamp=datetime(2026, 7, 21, 10, 0, 0),
                status="successful",
                description="Deterministic duplicate-looking demo payment.",
                external_reference="EXTDUP0005",
            )
        )
        _add_operational_scenarios(session)
        session.commit()
        _integrity_check(session)
        return _statistics(session)


def format_statistics(stats: Iterable[tuple[str, int]]) -> str:
    """Render computed initialization statistics without hardcoding counts."""
    return "\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in stats)
