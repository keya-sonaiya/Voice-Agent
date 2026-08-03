"""Mock order lookup that requires server-validated caller ownership."""


def lookup_order(order_id: str, authenticated_order_ids: set[str]) -> dict[str, str]:
    """Return a mock record only when server-side identity authorizes this order."""
    if order_id not in authenticated_order_ids:
        raise PermissionError("Caller is not authorized for this order.")
    return {"order_id": order_id, "status": "processing"}
