"""SQLModel tables for call snapshots and the local telecom-support demo database.

IBM Telco churn fields are retained as source-derived data. Operational records are
deterministic synthetic demo data created by ``scripts/init_db.py``.
"""

from datetime import date, datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class SessionSnapshot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    node_name: str
    state_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SourceCustomerRecord(SQLModel, table=True):
    """Immutable source row for reproducibility; ``raw_json`` is IBM source data."""

    source_customer_id: str = Field(primary_key=True)
    raw_json: str
    imported_at: datetime = Field(default_factory=datetime.utcnow)


class Customer(SQLModel, table=True):
    customer_id: str = Field(primary_key=True, index=True)  # Deterministic demo support ID, e.g. CUST1024.
    source_customer_id: str = Field(index=True, unique=True)
    first_name: str  # GENERATED DEMO DATA; IBM does not provide names.
    last_name: str  # GENERATED DEMO DATA; IBM does not provide names.
    full_name: str = Field(index=True)  # GENERATED DEMO DATA.
    email: Optional[str] = Field(default=None, index=True)  # GENERATED DEMO DATA.
    phone: Optional[str] = Field(default=None, index=True)  # GENERATED DEMO DATA.
    gender: str
    senior_citizen: bool
    partner: bool
    dependents: bool
    account_status: str = Field(index=True)
    churned: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Plan(SQLModel, table=True):
    plan_id: str = Field(primary_key=True)
    plan_name: str = Field(index=True)
    monthly_price: float
    service_type: str
    description: str
    active: bool = True


class Subscription(SQLModel, table=True):
    subscription_id: str = Field(primary_key=True)
    customer_id: str = Field(foreign_key="customer.customer_id", index=True)
    plan_id: str = Field(foreign_key="plan.plan_id", index=True)
    start_date: date
    end_date: Optional[date] = None
    contract_type: str
    status: str = Field(index=True)
    tenure_months: int


class CustomerService(SQLModel, table=True):
    service_id: str = Field(primary_key=True)
    customer_id: str = Field(foreign_key="customer.customer_id", index=True)
    phone_service: bool
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str


class BillingAccount(SQLModel, table=True):
    billing_account_id: str = Field(primary_key=True)
    customer_id: str = Field(foreign_key="customer.customer_id", index=True, unique=True)
    billing_cycle: str
    payment_method: str
    paperless_billing: bool
    monthly_charge: float
    total_charges: float
    balance: float
    currency: str = "USD"
    status: str = Field(index=True)


class Invoice(SQLModel, table=True):
    invoice_id: str = Field(primary_key=True, index=True)
    customer_id: str = Field(foreign_key="customer.customer_id", index=True)
    billing_account_id: str = Field(foreign_key="billingaccount.billing_account_id", index=True)
    invoice_date: date
    billing_period_start: date
    billing_period_end: date
    due_date: date
    subtotal: float
    tax: float
    total_amount: float
    amount_paid: float
    balance_due: float
    status: str = Field(index=True)


class Payment(SQLModel, table=True):
    payment_id: str = Field(primary_key=True, index=True)
    customer_id: str = Field(foreign_key="customer.customer_id", index=True)
    invoice_id: str = Field(foreign_key="invoice.invoice_id", index=True)
    payment_date: date
    amount: float
    payment_method: str
    status: str = Field(index=True)
    failure_reason: Optional[str] = None
    reference_number: str = Field(index=True, unique=True)
    generated_demo_data: bool = True


class Transaction(SQLModel, table=True):
    transaction_id: str = Field(primary_key=True, index=True)
    customer_id: str = Field(foreign_key="customer.customer_id", index=True)
    payment_id: Optional[str] = Field(default=None, foreign_key="payment.payment_id", index=True)
    transaction_type: str
    amount: float
    timestamp: datetime
    status: str = Field(index=True)
    description: str
    external_reference: str = Field(index=True, unique=True)


class CustomerOrder(SQLModel, table=True):
    order_id: str = Field(primary_key=True, index=True)
    customer_id: str = Field(foreign_key="customer.customer_id", index=True)
    order_date: date
    order_type: str
    status: str = Field(index=True)
    total_amount: float
    shipping_address: str  # GENERATED DEMO DATA; not supplied by IBM.
    tracking_number: Optional[str] = Field(default=None, index=True)


class SupportTicket(SQLModel, table=True):
    ticket_id: str = Field(primary_key=True, index=True)
    customer_id: str = Field(foreign_key="customer.customer_id", index=True)
    category: str
    subject: str
    description: str
    priority: str
    status: str = Field(index=True)
    assigned_team: str
    created_at: datetime
    updated_at: datetime
    resolution: Optional[str] = None


class TicketMessage(SQLModel, table=True):
    message_id: str = Field(primary_key=True)
    ticket_id: str = Field(foreign_key="supportticket.ticket_id", index=True)
    sender_type: str
    message: str
    created_at: datetime


class CustomerInteraction(SQLModel, table=True):
    interaction_id: str = Field(primary_key=True)
    customer_id: str = Field(foreign_key="customer.customer_id", index=True)
    channel: str
    interaction_type: str
    summary: str
    created_at: datetime
