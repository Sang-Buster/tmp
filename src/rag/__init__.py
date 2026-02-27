"""
RAG module - Unified Qdrant storage for telemetry and logs.

Uses Qdrant for both:
- telemetry: Agent position/state data
- logs: Conversation history, commands, notifications
"""
from .qdrant import (
    # Logs
    add_log,
    # Telemetry
    add_telemetry,
    clear_all,
    clear_logs,
    # Cleanup
    clear_telemetry,
    get_all_telemetry,
    get_conversation_history,
    get_logs,
    get_telemetry_history,
    # Unified
    search_all,
    search_logs,
    search_telemetry,
    test_connection,
)

__all__ = [
    # Telemetry
    "add_telemetry",
    "get_telemetry_history",
    "search_telemetry",
    "get_all_telemetry",
    # Logs
    "add_log",
    "get_logs",
    "get_conversation_history",
    "search_logs",
    # Unified
    "search_all",
    "test_connection",
    # Cleanup
    "clear_telemetry",
    "clear_logs",
    "clear_all",
]
