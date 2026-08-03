import os

os.environ.setdefault("OLLAMA_API_KEY", "test-key")
os.environ.setdefault("API_AUTH_SECRET", "test-secret")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_sessions.db")
