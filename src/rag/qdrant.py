"""
Qdrant Store - Unified storage for telemetry and logs.

Two collections:
- telemetry: Agent position/state data
- logs: Conversation history, commands, notifications
"""
import uuid
from datetime import datetime
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from ..config import QDRANT_HOST, QDRANT_PORT, VECTOR_DIM

# Collection names
TELEMETRY_COLLECTION = "telemetry"
LOGS_COLLECTION = "logs"

# Global instances
_client: Optional[QdrantClient] = None
_model: Optional[SentenceTransformer] = None


def get_client() -> QdrantClient:
    """Get or create Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        _init_collections()
    return _client


def get_model() -> SentenceTransformer:
    """Get or create embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _init_collections():
    """Initialize Qdrant collections if they don't exist."""
    client = get_client()

    # Telemetry collection
    try:
        client.get_collection(TELEMETRY_COLLECTION)
        print(f"[Qdrant] Connected to '{TELEMETRY_COLLECTION}'")
    except Exception:
        client.create_collection(
            collection_name=TELEMETRY_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print(f"[Qdrant] Created collection '{TELEMETRY_COLLECTION}'")

    # Logs collection
    try:
        client.get_collection(LOGS_COLLECTION)
        print(f"[Qdrant] Connected to '{LOGS_COLLECTION}'")
    except Exception:
        client.create_collection(
            collection_name=LOGS_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print(f"[Qdrant] Created collection '{LOGS_COLLECTION}'")


def test_connection() -> bool:
    """Test Qdrant connection."""
    try:
        client = get_client()
        client.get_collections()
        return True
    except Exception as e:
        print(f"[Qdrant] Connection failed: {e}")
        return False


# =============================================================================
# TELEMETRY FUNCTIONS
# =============================================================================

def add_telemetry(
    agent_id: str,
    position: tuple[float, float, float],
    metadata: Optional[dict[str, Any]] = None
) -> Optional[str]:
    """
    Add agent telemetry to Qdrant.
    
    Args:
        agent_id: Agent identifier
        position: (x, y, z) coordinates
        metadata: Additional data (jammed, comm_quality, etc.)
    
    Returns:
        Point ID if successful
    """
    if metadata is None:
        metadata = {}

    try:
        client = get_client()
        model = get_model()

        # Create searchable text
        jammed = metadata.get("jammed", False)
        comm_quality = metadata.get("communication_quality", 1.0)

        text = (
            f"Agent {agent_id} at position ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) "
            f"{'JAMMED' if jammed else 'CLEAR'} comm_quality={comm_quality:.2f}"
        )

        # Generate embedding
        embedding = model.encode([text])[0].tolist()

        # Create point
        point_id = str(uuid.uuid4())
        timestamp = metadata.get("timestamp", datetime.now().isoformat())

        payload = {
            "agent_id": agent_id,
            "position_x": float(position[0]),
            "position_y": float(position[1]),
            "position_z": float(position[2]),
            "timestamp": timestamp,
            "text": text,
            "jammed": jammed,
            "communication_quality": comm_quality,
            **{k: v for k, v in metadata.items() if k not in ["timestamp", "jammed", "communication_quality"]}
        }

        # Insert
        client.upsert(
            collection_name=TELEMETRY_COLLECTION,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
        )

        return point_id

    except Exception as e:
        print(f"[Qdrant] Error adding telemetry: {e}")
        return None


def get_telemetry_history(
    agent_id: str,
    limit: int = 20
) -> list[dict[str, Any]]:
    """Get position history for an agent."""
    try:
        client = get_client()

        results = client.scroll(
            collection_name=TELEMETRY_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=agent_id))]
            ),
            limit=limit * 2,
            with_payload=True,
            with_vectors=False
        )[0]

        history = []
        for point in results:
            payload = point.payload
            history.append({
                "position": (
                    float(payload.get("position_x", 0)),
                    float(payload.get("position_y", 0)),
                    float(payload.get("position_z", 0)),
                ),
                "jammed": bool(payload.get("jammed", False)),
                "communication_quality": float(payload.get("communication_quality", 1.0)),
                "timestamp": payload.get("timestamp", ""),
                "point_id": str(point.id),
            })

        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return history[:limit]

    except Exception as e:
        print(f"[Qdrant] Error getting history: {e}")
        return []


def search_telemetry(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Semantic search through telemetry data."""
    try:
        client = get_client()
        model = get_model()

        query_vector = model.encode([query])[0].tolist()

        results = client.query_points(
            collection_name=TELEMETRY_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

        return [
            {**hit.payload, "score": hit.score}
            for hit in results.points
        ]

    except Exception as e:
        print(f"[Qdrant] Error searching telemetry: {e}")
        return []


def get_all_telemetry(limit: int = 100) -> list[dict[str, Any]]:
    """Get all recent telemetry for dashboard."""
    try:
        client = get_client()

        results = client.scroll(
            collection_name=TELEMETRY_COLLECTION,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )[0]

        records = []
        for point in results:
            payload = point.payload
            records.append({
                "agent_id": payload.get("agent_id"),
                "position": (
                    float(payload.get("position_x", 0)),
                    float(payload.get("position_y", 0)),
                    float(payload.get("position_z", 0)),
                ),
                "jammed": bool(payload.get("jammed", False)),
                "communication_quality": float(payload.get("communication_quality", 1.0)),
                "timestamp": payload.get("timestamp", ""),
            })

        records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return records

    except Exception as e:
        print(f"[Qdrant] Error getting all telemetry: {e}")
        return []


# =============================================================================
# LOG FUNCTIONS (replaces PostgreSQL)
# =============================================================================

def add_log(
    log_text: str,
    metadata: Optional[dict[str, Any]] = None,
    source: str = "system",
    message_type: str = "notification"
) -> Optional[str]:
    """
    Add a log entry to Qdrant.
    
    Args:
        log_text: Message content
        metadata: Additional data
        source: "user", "llm", "agent1", etc.
        message_type: "command", "response", "notification", "error"
    
    Returns:
        Point ID if successful
    """
    if metadata is None:
        metadata = {}

    try:
        client = get_client()
        model = get_model()

        timestamp = datetime.now().isoformat()

        # Generate embedding
        embedding = model.encode([log_text])[0].tolist()

        point_id = str(uuid.uuid4())

        payload = {
            "text": log_text,
            "source": source,
            "message_type": message_type,
            "timestamp": timestamp,
            **metadata
        }

        client.upsert(
            collection_name=LOGS_COLLECTION,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)]
        )

        return point_id

    except Exception as e:
        print(f"[Qdrant] Error adding log: {e}")
        return None


def get_logs(
    source: Optional[str] = None,
    message_type: Optional[str] = None,
    limit: int = 50
) -> list[dict[str, Any]]:
    """Get logs with optional filtering."""
    try:
        client = get_client()

        # Build filter
        filter_conditions = []
        if source:
            filter_conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))
        if message_type:
            filter_conditions.append(FieldCondition(key="message_type", match=MatchValue(value=message_type)))

        scroll_filter = Filter(must=filter_conditions) if filter_conditions else None

        results = client.scroll(
            collection_name=LOGS_COLLECTION,
            scroll_filter=scroll_filter,
            limit=limit * 2,
            with_payload=True,
            with_vectors=False
        )[0]

        logs = []
        for point in results:
            payload = point.payload
            logs.append({
                "id": str(point.id),
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "message_type": payload.get("message_type", ""),
                "timestamp": payload.get("timestamp", ""),
                "metadata": payload,
            })

        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return logs[:limit]

    except Exception as e:
        print(f"[Qdrant] Error getting logs: {e}")
        return []


def get_conversation_history(limit: int = 100) -> list[dict[str, Any]]:
    """Get user/LLM conversation history."""
    try:
        client = get_client()

        # Get user messages
        user_results = client.scroll(
            collection_name=LOGS_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value="user"))]),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )[0]

        # Get LLM messages
        llm_results = client.scroll(
            collection_name=LOGS_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value="llm"))]),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )[0]

        # Combine and sort
        messages = []
        for point in user_results + llm_results:
            payload = point.payload
            messages.append({
                "id": str(point.id),
                "text": payload.get("text", ""),
                "role": payload.get("source", ""),
                "timestamp": payload.get("timestamp", ""),
            })

        messages.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return messages[:limit]

    except Exception as e:
        print(f"[Qdrant] Error getting conversation: {e}")
        return []


def search_logs(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Semantic search through logs."""
    try:
        client = get_client()
        model = get_model()

        query_vector = model.encode([query])[0].tolist()

        results = client.query_points(
            collection_name=LOGS_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True
        )

        return [
            {**hit.payload, "score": hit.score, "id": str(hit.id)}
            for hit in results.points
        ]

    except Exception as e:
        print(f"[Qdrant] Error searching logs: {e}")
        return []


# =============================================================================
# UNIFIED SEARCH (for RAG)
# =============================================================================

def search_all(query: str, limit: int = 10) -> dict[str, list[dict]]:
    """
    Search both telemetry and logs for RAG retrieval.
    
    Args:
        query: Search query
        limit: Max results per collection
    
    Returns:
        Dict with 'telemetry' and 'logs' results
    """
    return {
        "telemetry": search_telemetry(query, limit),
        "logs": search_logs(query, limit),
    }


# =============================================================================
# CLEANUP
# =============================================================================

def clear_telemetry():
    """Clear all telemetry data."""
    try:
        client = get_client()
        client.delete_collection(TELEMETRY_COLLECTION)
        client.create_collection(
            collection_name=TELEMETRY_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print("[Qdrant] Telemetry cleared")
    except Exception as e:
        print(f"[Qdrant] Error clearing telemetry: {e}")


def clear_logs():
    """Clear all logs."""
    try:
        client = get_client()
        client.delete_collection(LOGS_COLLECTION)
        client.create_collection(
            collection_name=LOGS_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print("[Qdrant] Logs cleared")
    except Exception as e:
        print(f"[Qdrant] Error clearing logs: {e}")


def clear_all():
    """Clear all collections."""
    clear_telemetry()
    clear_logs()
