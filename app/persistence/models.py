"""SQLModel tables used for session transition snapshots."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class SessionSnapshot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    node_name: str
    state_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
