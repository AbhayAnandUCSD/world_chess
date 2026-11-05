"""Local viewer: uv run python -m viewer  -> http://localhost:8765"""
import uvicorn
uvicorn.run("viewer.app:app", host="127.0.0.1", port=8765, reload=False)
