# Telecom Support Database

## Purpose and source boundary

`WA_Fn-UseC_-Telco-Customer-Churn.csv` is the source for customer, subscription,
service, and billing-account attributes. It has 7,043 rows, 21 columns, no duplicate
rows or duplicate source customer IDs, and 11 blank `TotalCharges` values. Those blanks
are imported as `0.0` and are documented rather than inferred. Categorical fields are
preserved from IBM (for example, contract, payment method, internet service, and churn).

**IBM Telco Customer Churn provides the source customer/account/service data;
transactional and support records are deterministic synthetic demo data added for the
customer-support application.** IBM does not provide names, contact details, invoices,
payments, orders, tickets, or addresses. Generated values use a fixed algorithm and
fixed as-of date (`2026-08-01`); they are never presented as IBM data.

## Relationships

```text
source_customer_records 1--1 customers 1--1 billing_accounts
                                  | 1--1 subscriptions --1 plans
                                  | 1--1 customer_services
                                  | 1--* invoices --1--* payments --1--* transactions
                                  | 1--* customer_orders
                                  | 1--* support_tickets --1--* ticket_messages
                                  ` 1--* customer_interactions
```

`SessionSnapshot` remains the existing call-audit table; it is not reset by telecom
seeding. SQLite foreign-key enforcement is enabled on the application and initializer
engines.

## CSV mapping

| IBM CSV field | Destination | Notes |
|---|---|---|
| `customerID` | `source_customer_records.source_customer_id`, `customers.source_customer_id` | Original identifier retained verbatim. |
| row position | `customers.customer_id` | Deterministic support-safe ID (`CUST0001` … `CUST7043`). |
| `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `Churn` | `customers` | Source derived; churn maps to `churned` and `account_status`. |
| `tenure`, `Contract` | `subscriptions.tenure_months`, `contract_type` | Subscription start/status are deterministic derivations. |
| `PhoneService` through `StreamingMovies` | `customer_services` | Source service attributes. |
| `InternetService`, `MonthlyCharges` | `plans`, `billing_accounts` | Plan is classified by service type; plan labels are generated. |
| `PaymentMethod`, `PaperlessBilling`, `MonthlyCharges`, `TotalCharges` | `billing_accounts` | Source billing attributes. |

## Generated demo records

`scripts/init_db.py` creates one invoice, payment, and transaction per customer plus
a deterministic duplicate-looking payment. Fixed scenarios include successful, failed,
pending, refunded, declined, overdue, in-transit/delivered/cancelled orders, and
open/resolved technical or billing tickets.

The documented payment demo is `CUST1024` / `Samad Sama` / `PAY102938`, which is
declined with the generated reason `Issuer declined the transaction.`

## Backend tools and authorization

Tools are server functions in `app.tools.customer_support`; no LLM is given a database
connection or SQL capability.

- `verify_customer_identity(customer_id, name)` is verification-only and returns a boolean.
- `get_customer`, `get_account_status`, `get_payment`, `get_customer_payments`,
  `get_invoice`, `get_customer_transactions`, `get_order`, `get_customer_tickets`, and
  `get_ticket` all require the requested `customer_id` to equal the verified customer.
- Resource tools additionally verify `resource.customer_id == verified_customer_id`.
- `create_support_ticket` is bounded, validates its input, and requires the same owner check.

Invalid identifiers, missing records, ownership mismatches, and database exceptions
produce safe caller messages; raw SQL errors are logged only on the backend. Tool logs
use `[CALL][stage=TOOL]`, `[stage=AUTHZ]`, and `[stage=DB]` without credentials or card data.

## Voice workflow

For customer-specific intents, the Knowledge node uses a server-controlled flow:

```text
Intent/Sentiment → verified-support workflow → authorized tool and/or RAG
→ Grounding Judge → final_response_text → WebSocket → TTS
```

The workflow asks for a customer ID, then account name, before accepting a payment,
invoice, order, or ticket ID. Dynamic tool answers include an authoritative, scoped
database excerpt for the existing grounding judge. Generic verification prompts are
fixed non-factual responses and use the existing deterministic grounding gate.

## Commands

```powershell
# First-time, non-destructive initialization. Refuses if telecom data already exists.
python scripts/init_db.py initialize

# Explicit safe seed operation for a blank telecom schema. Also refuses to overwrite data.
python scripts/init_db.py seed

# Explicitly rebuild only telecom tables; existing SessionSnapshot call history remains.
python scripts/init_db.py reset

# Use a separate local database for experiments/tests.
python scripts/init_db.py --database-url sqlite:///./telecom_demo.db
```

## Demo conversations

1. Payment: `payment trouble` → `CUST1024` → `Samad Sama` → `PAY102938`.
2. Refund: verify `CUST0004`, then supply `PAY000004` to see `refunded`.
3. Overdue invoice: verify `CUST0006`, then supply `INV000006`.
4. Router order: verify `CUST0007`, then supply `ORD000007` to see `in_transit`.
5. Ticket: verify `CUST0010`, then supply `TKT000010` to see the resolved technical ticket.

Trying any of those resource IDs with another verified customer ID is denied without
returning the other customer's record.
