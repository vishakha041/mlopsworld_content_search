# ===== TOOL 2: SEMANTIC SEARCH =====

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from .utils import (
    get_db_connector, get_embedding_model, to_blob, safe_get,
    format_date_constraint, get_text_field_name,
    SET_TRANSCRIPT, SET_META, SET_BIO, EMBED_MODEL, EMBED_DIM
)

class SearchTalksSemanticallInput(BaseModel):
    """Input schema for semantic search across talks using embeddings."""
    
    query: str = Field(
        ...,
        description="Natural language search query. Examples: 'AI agents with memory', 'MLOps deployment strategies', 'vector databases in production'"
    )
    search_type: Optional[Literal["transcript", "meta", "bio", "all"]] = Field(
        "all",
        description="Which content to search: 'transcript' (video transcripts), 'meta' (abstracts/keywords), 'bio' (speaker bios), 'all' (search all types)"
    )
    date_from: Optional[str] = Field(
        None,
        description="Optional: filter results to talks published from this date (YYYY-MM-DD, YYYY-MM, or YYYY)"
    )
    date_to: Optional[str] = Field(
        None, 
        description="Optional: filter results to talks published up to this date (YYYY-MM-DD, YYYY-MM, or YYYY)"
    )
    category: Optional[str] = Field(
        None,
        description="Optional: filter results to specific category (e.g., 'MLOps', 'Deployment and integration')"
    )
    event_name: Optional[str] = Field(
        None,
        description="Optional: filter results to specific event (e.g., 'MLOps & GenAI World 2024')"
    )
    speaker_name: Optional[str] = Field(
        None,
        description="Optional: search only within talks by this specific speaker (exact name match)"
    )
    k_neighbors: Optional[int] = Field(
        10,
        description="Number of most similar results to return (default: 10, max recommended: 50)"
    )
    score_threshold: Optional[float] = Field(
        None,
        description="Optional: minimum similarity score threshold (0.0-1.0). Higher values = more similar results only"
    )

@tool("search_talks_semantically", args_schema=SearchTalksSemanticallInput)
def search_talks_semantically(
    query: str,
    search_type: Optional[str] = "all",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    category: Optional[str] = None,
    event_name: Optional[str] = None,
    speaker_name: Optional[str] = None,
    k_neighbors: Optional[int] = 10,
    score_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Perform semantic search across MLOps talks using natural language queries.
    Searches through video transcripts, talk abstracts/metadata, and speaker bios
    using embeddings to find semantically similar content.
    
    This tool handles all k-NN semantic search operations with optional filtering.
    It can search across different content types (transcripts, abstracts, bios) 
    and apply additional filters like dates, categories, or specific speakers.
    Results include similarity scores and matching text snippets for context.
    
    Use this tool when users ask questions like:
    - "Which talks discuss AI agents?"
    - "Find talks about MLOps deployment strategies" 
    - "Who are the experts in vector databases?"
    - "Talks about memory management in AI systems"
    - "Find content similar to RAG implementations"
    
    Args:
        query: Natural language search query
        search_type: Content type to search ('transcript', 'meta', 'bio', or 'all')
        date_from: Optional date filter start (YYYY-MM-DD format) 
        date_to: Optional date filter end (YYYY-MM-DD format)
        category: Optional category filter
        event_name: Optional event filter
        speaker_name: Optional speaker filter (exact match)
        k_neighbors: Number of similar results to return
        score_threshold: Optional minimum similarity threshold
        
    Returns:
        Dict containing:
        - 'results': List of matching talks with similarity scores and snippets
        - 'total_found': Number of results found
        - 'search_summary': Description of search performed
        - 'query_vector_info': Information about the query embedding
    """
    
    try:
        con = get_db_connector()
        model = get_embedding_model()

        query_text = f"query: {query}"
        qvec = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0].astype("<f4")
        query_blob = to_blob(qvec)

        # Resolve which sets to search
        sets_to_search = []
        if search_type == "all":
            sets_to_search = [(SET_TRANSCRIPT, "TalkHasTranscriptChunk", "transcript"),
                              (SET_META, "TalkHasMeta", "abstract/metadata"),
                              (SET_BIO, "TalkHasSpeakerBio", "speaker bio")]
        elif search_type == "transcript":
            sets_to_search = [(SET_TRANSCRIPT, "TalkHasTranscriptChunk", "transcript")]
        elif search_type == "meta":
            sets_to_search = [(SET_META, "TalkHasMeta", "abstract/metadata")]
        elif search_type == "bio":
            sets_to_search = [(SET_BIO, "TalkHasSpeakerBio", "speaker bio")]
        else:
            return {"success": False, "results": [], "total_found": 0,
                    "search_summary": f"Invalid search_type '{search_type}'", "query_vector_info": ""}

        def list_fields_for_set(set_name: str) -> List[str]:
            if set_name == SET_TRANSCRIPT:
                return ["_distance", "talk_id", "chunk_text", "seq", "start_sec", "end_sec"]
            elif set_name == SET_META:
                return ["_distance", "talk_id", "meta_text"]
            else:  # SET_BIO
                return ["_distance", "talk_id", "bio_text"]

        all_results: List[Dict[str, Any]] = []
        search_summaries: List[str] = []

        for set_name, connection_class, content_type in sets_to_search:
            # Build base query
            if speaker_name:
                q = [
                    {"FindEntity": {
                        "_ref": 1, "with_class": "Person", "unique": True,
                        "constraints": {"name": ["==", speaker_name]}
                    }},
                    {"FindEntity": {
                        "_ref": 2, "with_class": "Talk",
                        "is_connected_to": {"ref": 1, "direction": "in", "connection_class": "TalkHasSpeaker"},
                        "results": {"list": ["talk_id"]}  # ensure talk_id exists for downstream
                    }},
                    {"FindDescriptor": {
                        "_ref": 3, "set": set_name,
                        "is_connected_to": {"ref": 2, "connection_class": connection_class},
                        "k_neighbors": int(k_neighbors), "distances": True,
                        "results": {"list": list_fields_for_set(set_name)}
                    }},
                    {"FindEntity": {
                        "with_class": "Talk",
                        "is_connected_to": {"ref": 3, "connection_class": connection_class},
                        "results": {"list": ["talk_id","talk_title","speaker_name","youtube_url","event_name","category_primary"]}
                    }}
                ]
            else:
                talk_constraints: Dict[str, Any] = {}
                if date_from:
                    df = format_date_constraint(date_from)
                    if df: talk_constraints["yt_published_at"] = talk_constraints.get("yt_published_at", []) + [">=", df]
                if date_to:
                    dt = format_date_constraint(date_to)
                    if dt: talk_constraints["yt_published_at"] = talk_constraints.get("yt_published_at", []) + ["<=", dt]
                if category:
                    talk_constraints["category_primary"] = ["==", category]
                if event_name:
                    talk_constraints["event_name"] = ["==", event_name]

                if talk_constraints:
                    q = [
                        {"FindEntity": {
                            "_ref": 1, "with_class": "Talk",
                            "constraints": talk_constraints,
                            "results": {"list": ["talk_id"]}
                        }},
                        {"FindDescriptor": {
                            "_ref": 2, "set": set_name,
                            "is_connected_to": {"ref": 1, "connection_class": connection_class},
                            "k_neighbors": int(k_neighbors), "distances": True,
                            "results": {"list": list_fields_for_set(set_name)}
                        }},
                        {"FindEntity": {
                            "with_class": "Talk",
                            "is_connected_to": {"ref": 2, "connection_class": connection_class},
                            "results": {"list": ["talk_id","talk_title","speaker_name","youtube_url","event_name","category_primary"]}
                        }}
                    ]
                else:
                    q = [
                        {"FindDescriptor": {
                            "_ref": 1, "set": set_name,
                            "k_neighbors": int(k_neighbors), "distances": True,
                            "results": {"list": list_fields_for_set(set_name)}
                        }},
                        {"FindEntity": {
                            "with_class": "Talk",
                            "is_connected_to": {"ref": 1, "connection_class": connection_class},
                            "results": {"list": ["talk_id","talk_title","speaker_name","youtube_url","event_name","category_primary"]}
                        }}
                    ]

            resp, _ = con.query(q, blobs=[query_blob])

            # Pull out descriptor/talk entities robustly
            desc_resp = next((r["FindDescriptor"] for r in resp if "FindDescriptor" in r), None)
            talk_resp = next((r["FindEntity"] for r in resp if "FindEntity" in r and r["FindEntity"].get("entities")), None)

            if not desc_resp or not talk_resp:
                search_summaries.append(f"{content_type} search (no matches)")
                continue

            desc_entities = desc_resp.get("entities", []) or []
            talk_entities = talk_resp.get("entities", []) or []

            talk_map = {safe_get(t, "talk_id"): t for t in talk_entities if safe_get(t, "talk_id")}

            for d in desc_entities:
                talk_id = safe_get(d, "talk_id")
                if not talk_id:
                    continue
                talk_info = talk_map.get(talk_id, {})

                dist = safe_get(d, "_distance")
                # If distance is None, treat as low-confidence (0.0)
                similarity = 1.0 - float(dist) if dist is not None else 0.0

                if score_threshold is not None and similarity < float(score_threshold):
                    continue

                if set_name == SET_TRANSCRIPT:
                    matching_text = safe_get(d, "chunk_text", "")
                    context_info = f"Timestamp: {int(safe_get(d,'start_sec',0))}-{int(safe_get(d,'end_sec',0))}s"
                elif set_name == SET_META:
                    matching_text = safe_get(d, "meta_text", "")
                    context_info = "From talk abstract/metadata"
                else:
                    matching_text = safe_get(d, "bio_text", "")
                    context_info = "From speaker bio"

                all_results.append({
                    "talk_id": talk_id,
                    "title": safe_get(talk_info, "talk_title"),
                    "speaker": safe_get(talk_info, "speaker_name"),
                    "youtube_url": safe_get(talk_info, "youtube_url"),
                    "event": safe_get(talk_info, "event_name"),
                    "category": safe_get(talk_info, "category_primary"),
                    "similarity_score": round(similarity, 3),
                    "matching_text": (matching_text[:300] + "...") if len(matching_text) > 300 else matching_text,
                    "content_type": content_type,
                    "context_info": context_info
                })

            search_summaries.append(f"{content_type} search")

        # Sort + trim
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        final_results = all_results[: int(k_neighbors)]

        # Summary
        parts = []
        if search_summaries: parts.append(f"Searched: {', '.join(search_summaries)}")
        if date_from or date_to: parts.append(f"Date range: {date_from or 'start'} to {date_to or 'end'}")
        if category: parts.append(f"Category: {category}")
        if event_name: parts.append(f"Event: {event_name}")
        if speaker_name: parts.append(f"Speaker: {speaker_name}")
        if score_threshold is not None: parts.append(f"Min similarity: {score_threshold}")

        return {
            "success": True,
            "results": final_results,
            "total_found": len(final_results),
            "search_summary": f"Semantic search for '{query}'" + (f" ({', '.join(parts)})" if parts else ""),
            "query_vector_info": f"Generated {EMBED_DIM}D embedding using {EMBED_MODEL}"
        }

    except Exception as e:
        return {"success": False, "results": [], "total_found": 0,
                "search_summary": f"Semantic search for '{query}' failed",
                "query_vector_info": "", "error": str(e)}
