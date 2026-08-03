"""Mock ticket history lookup with explicit server-side authorization."""


def get_ticket_history(account_id: str, authenticated_account_id: str) -> list[dict[str, str]]:
    """Return mock ticket history only after identity binding is verified."""
    if account_id != authenticated_account_id:
        raise PermissionError("Caller is not authorized for this account.")
    return [{"ticket_id": "demo-001", "status": "closed"}]
