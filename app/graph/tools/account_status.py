"""Mock account status lookup with explicit server-side authorization."""


def get_account_status(account_id: str, authenticated_account_id: str) -> dict[str, str]:
    """Return a mock account state only for the authenticated account."""
    if account_id != authenticated_account_id:
        raise PermissionError("Caller is not authorized for this account.")
    return {"account_id": account_id, "status": "active"}
