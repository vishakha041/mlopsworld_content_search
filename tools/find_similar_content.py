# ===== TOOL 5: FIND SIMILAR CONTENT =====
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Import shared utilities
from utils import (
    get_db_connector, get_embedding_model, to_blob, safe_get,
    format_date_constraint, get_text_field_name,
    SET_TRANSCRIPT, SET_META, SET_BIO, EMBED_MODEL, EMBED_DIM
)

DEBUG_SIMQ = False  # set True temporarily to log the query JSON and raw responses


class FindSimilarContentInput(BaseModel):
    """Input schema for finding similar content and content recommendations."""
    reference_talk_title: Optional[str] = Field(
        None,
        description="Title of a specific talk to find similar content for."
    )
    reference_talk_id: Optional[str] = Field(
        None,
        description="Unique ID of a specific talk to find similar content for."
    )
    reference_query: Optional[str] = Field(
        None,
        description="Natural language query to find similar content when no specific talk is referenced."
    )
    similarity_type: Optional[Literal["content", "speaker", "topic", "all"]] = Field(
        "content",
        description="Type of similarity analysis: 'content', 'speaker', 'topic', or 'all'."
    )
    date_from: Optional[str] = Field(None, description="YYYY[-MM[-DD]]")
    date_to: Optional[str] = Field(None, description="YYYY[-MM[-DD]]")
    category: Optional[str] = Field(None, description="Category filter")
    event_name: Optional[str] = Field(None, description="Event filter")
    exclude_same_speaker: Optional[bool] = Field(False, description="Exclude same speaker's talks")
    min_similarity: Optional[float] = Field(None, description="Minimum similarity (0..1)")
    limit: Optional[int] = Field(10, description="Max results to return")


@tool("find_similar_content", args_schema=FindSimilarContentInput)
def find_similar_content(
    reference_talk_title: Optional[str] = None,
    reference_talk_id: Optional[str] = None,
    reference_query: Optional[str] = None,
    similarity_type: Optional[str] = "content",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    category: Optional[str] = None,
    event_name: Optional[str] = None,
    exclude_same_speaker: Optional[bool] = False,
    min_similarity: Optional[float] = None,
    limit: Optional[int] = 10
) -> Dict[str, Any]:
    """
     Find content similar to a reference talk or query using semantic similarity analysis.
    Supports multiple similarity types including content-based, speaker-based, and topic-based
    recommendations with flexible filtering and ranking options.
    
    This tool handles all content recommendation and similarity analysis operations.
    It can find similar talks based on a reference talk, generate recommendations
    from natural language queries, and provide different types of similarity analysis.
    Results include detailed similarity scores and explanations of why content is similar.
    
    Use this tool when users ask questions like:
    - "Find talks similar to the LangChain presentation"
    - "Recommend talks based on my interest in AI agents"
    - "What other talks cover similar themes to this one?"
    - "Show me related content but exclude same speaker"
    - "Find talks with similar content from different events"
    
    Args:
        reference_talk_title: Specific talk title to find similar content for
        reference_talk_id: Alternative talk ID for reference
        reference_query: Natural language query for similarity (if no specific talk)
        similarity_type: Type of similarity ('content', 'speaker', 'topic', 'all')
        date_from: Optional start date filter (YYYY-MM-DD format)
        date_to: Optional end date filter (YYYY-MM-DD format)
        category: Optional category filter
        event_name: Optional event filter
        exclude_same_speaker: Whether to exclude same speaker's talks
        min_similarity: Optional minimum similarity threshold
        limit: Maximum results to return
        
    Returns:
        Dict containing:
        - 'similar_talks': List of similar talks with scores and similarity reasons
        - 'reference_info': Information about the reference talk or query used
        - 'similarity_analysis': Summary of similarity analysis performed
        - 'total_found': Number of similar talks found
        - 'filters_applied': List of filters applied to results
    """
    try:
        con = get_db_connector()

        # Validate reference
        if not reference_talk_title and not reference_talk_id and not reference_query:
            return {
                "similar_talks": [],
                "reference_info": {},
                "similarity_analysis": "Error: No reference provided",
                "total_found": 0,
                "filters_applied": [],
                "success": False,
                "error": "Must provide either reference_talk_title, reference_talk_id, or reference_query",
            }

        reference_count = sum(bool(x) for x in [reference_talk_title, reference_talk_id, reference_query])
        if reference_count > 1:
            return {
                "similar_talks": [],
                "reference_info": {},
                "similarity_analysis": "Error: Multiple reference inputs provided",
                "total_found": 0,
                "filters_applied": [],
                "success": False,
                "error": "Provide only one of: reference_talk_title, reference_talk_id, or reference_query",
            }

        # Build filters
        filters_applied = []
        talk_constraints: Dict[str, Any] = {}

        if date_from:
            dc = format_date_constraint(date_from)
            if dc:
                talk_constraints["yt_published_at"] = talk_constraints.get("yt_published_at", []) + [">=", dc]
                filters_applied.append(f"from {date_from}")

        if date_to:
            dc = format_date_constraint(date_to)
            if dc:
                talk_constraints["yt_published_at"] = talk_constraints.get("yt_published_at", []) + ["<=", dc]
                filters_applied.append(f"until {date_to}")

        if category:
            talk_constraints["category_primary"] = ["==", category]
            filters_applied.append(f"category '{category}'")

        if event_name:
            talk_constraints["event_name"] = ["==", event_name]
            filters_applied.append(f"event '{event_name}'")

        # Resolve reference + build query text
        reference_info: Dict[str, Any] = {}
        reference_speaker = None
        model = get_embedding_model()

        if reference_talk_title or reference_talk_id:
            ref_constraint = (
                {"talk_title": ["==", reference_talk_title]}
                if reference_talk_title else
                {"talk_id": ["==", reference_talk_id]}
            )
            ref_q = [{
                "FindEntity": {
                    "with_class": "Talk",
                    "unique": True,
                    "constraints": ref_constraint,
                    "results": {
                        "list": ["talk_id", "talk_title", "speaker_name", "abstract",
                                 "category_primary", "youtube_url"]
                    }
                }
            }]
            ref_resp, _ = con.query(ref_q)
            entities = (ref_resp or [{}])[0].get("FindEntity", {}).get("entities", [])
            if not entities:
                desc = f"title '{reference_talk_title}'" if reference_talk_title else f"ID '{reference_talk_id}'"
                return {
                    "similar_talks": [],
                    "reference_info": {},
                    "similarity_analysis": f"Reference talk not found with {desc}",
                    "total_found": 0,
                    "filters_applied": filters_applied,
                    "success": False,
                    "error": f"No talk found with {desc}",
                }
            ref_talk = entities[0]
            reference_info = {
                "talk_id": safe_get(ref_talk, "talk_id"),
                "title": safe_get(ref_talk, "talk_title"),
                "speaker": safe_get(ref_talk, "speaker_name"),
                "category": safe_get(ref_talk, "category_primary"),
                "youtube_url": safe_get(ref_talk, "youtube_url"),
                "abstract": safe_get(ref_talk, "abstract", ""),
            }
            reference_speaker = safe_get(ref_talk, "speaker_name")
            abstract_text = safe_get(ref_talk, "abstract", "")
            query_text = f"query: {abstract_text[:500]}" if abstract_text else f"query: {reference_info['title']}"
        else:
            reference_info = {"query": reference_query, "type": "natural_language_query"}
            query_text = f"query: {reference_query}"

        # Build embedding blob
        qvec = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0].astype("<f4")
        query_blob = to_blob(qvec)

        all_similar_talks = []

        if similarity_type in ["content", "all"]:
            steps = []

            if talk_constraints:
                steps.append({
                    "FindEntity": {
                        "_ref": 1,
                        "with_class": "Talk",
                        "constraints": talk_constraints,
                        "results": {"list": ["talk_id"]}
                    }
                })

            steps.append({
                "FindDescriptor": {
                    "_ref": 2 if talk_constraints else 1,
                    "set": SET_META,
                    **({"is_connected_to": {"ref": 1, "connection_class": "TalkHasMeta"}} if talk_constraints else {}),
                    "k_neighbors": int(limit) + 5,
                    "distances": True,  # always request distances
                    "results": { "list": ["_distance","talk_id","meta_text"] }
                }
            })

            steps.append({
                "FindEntity": {
                    "with_class": "Talk",
                    "is_connected_to": {
                        "ref": 2 if talk_constraints else 1,
                        "connection_class": "TalkHasMeta"
                    },
                    "results": {
                        "list": ["talk_id","talk_title","speaker_name","youtube_url",
                                 "category_primary","yt_views","abstract"]
                    }
                }
            })

            sim_q = steps

            if DEBUG_SIMQ:
                import json
                print("\n[find_similar_content] sim_q JSON:\n", json.dumps(sim_q, indent=2))
                print("[find_similar_content] blobs count:", 1 if query_blob else 0)

            sim_resp, _ = con.query(sim_q, blobs=[query_blob])

            if DEBUG_SIMQ:
                print("\n[find_similar_content] RAW sim_resp:\n", sim_resp)

            # Parse results
            desc_step = (sim_resp or [{}])[-2].get("FindDescriptor", {}) if len(sim_resp or []) >= 2 else {}
            talk_step = (sim_resp or [{}])[-1].get("FindEntity", {}) if len(sim_resp or []) >= 1 else {}
            desc_entities = desc_step.get("entities", []) or []
            talk_entities = talk_step.get("entities", []) or []

            # Build talk map for quick join
            talk_map = {safe_get(t, "talk_id"): t for t in talk_entities if safe_get(t, "talk_id")}

            for desc in desc_entities:
                talk_id = safe_get(desc, "talk_id")
                if not talk_id:
                    continue

                # Skip self
                if reference_info.get("talk_id") == talk_id:
                    continue

                t = talk_map.get(talk_id)
                if not t:
                    # If talk wasn’t materialized (e.g., filtered away), skip
                    continue

                spk = safe_get(t, "speaker_name")
                if exclude_same_speaker and reference_speaker and spk == reference_speaker:
                    continue

                # Convert distance to [0..1] similarity
                distance = safe_get(desc, "_distance", None)
                similarity_score = round(1 - float(distance), 3) if distance is not None else 0.0

                if min_similarity is not None and similarity_score < float(min_similarity):
                    continue

                all_similar_talks.append({
                    "talk_id": talk_id,
                    "title": safe_get(t, "talk_title"),
                    "speaker": spk,
                    "youtube_url": safe_get(t, "youtube_url"),
                    "category": safe_get(t, "category_primary"),
                    "views": safe_get(t, "yt_views", 0),
                    "similarity_score": similarity_score,
                    "similarity_type": "content",
                    "similarity_reason": f"Similar content themes (score: {similarity_score})",
                    "matching_content": (safe_get(desc, "meta_text", "")[:200] + "...") if safe_get(desc, "meta_text") else ""
                })

        # SPEAKER similarity (same speaker’s other talks)
        if similarity_type in ["speaker", "all"] and reference_speaker:
            speaker_find_talks = {
                "with_class": "Talk",
                "is_connected_to": {
                    "ref": 1, "direction": "in", "connection_class": "TalkHasSpeaker"
                },
                "sort": {"key": "yt_published_at", "order": "descending"},
                "limit": limit,
                "results": {
                    "list": ["talk_id","talk_title","youtube_url","category_primary","yt_views","yt_published_at"]
                }
            }
            if talk_constraints:
                speaker_find_talks["constraints"] = talk_constraints  # omit entirely when empty

            speaker_q = [
                {
                    "FindEntity": {
                        "_ref": 1,
                        "with_class": "Person",
                        "unique": True,
                        "constraints": {"name": ["==", reference_speaker]}
                    }
                },
                { "FindEntity": speaker_find_talks }
            ]

            try:
                speaker_resp, _ = con.query(speaker_q)
                talks = (speaker_resp or [{}, {}])[1].get("FindEntity", {}).get("entities", []) if len(speaker_resp or []) > 1 else []
                for t in talks:
                    tid = safe_get(t, "talk_id")
                    if not tid or reference_info.get("talk_id") == tid:
                        continue
                    if any(st["talk_id"] == tid for st in all_similar_talks):
                        continue
                    pub_date = safe_get(t, "yt_published_at")
                    if isinstance(pub_date, dict) and "_date" in pub_date:
                        pub_date = pub_date["_date"].split("T")[0]
                    all_similar_talks.append({
                        "talk_id": tid,
                        "title": safe_get(t, "talk_title"),
                        "speaker": reference_speaker,
                        "youtube_url": safe_get(t, "youtube_url"),
                        "category": safe_get(t, "category_primary"),
                        "views": safe_get(t, "yt_views", 0),
                        "similarity_score": 0.9,  # fixed score for same-speaker recs
                        "similarity_type": "speaker",
                        "similarity_reason": f"Same speaker: {reference_speaker}",
                        "published_date": pub_date
                    })
            except Exception as e:
                if DEBUG_SIMQ:
                    print("[find_similar_content] speaker branch error:", e)
                # swallow speaker errors; keep content results

        # Finalize
        all_similar_talks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        final_similar_talks = all_similar_talks[: int(limit or 10)]

        analysis_parts = []
        if similarity_type == "content":
            analysis_parts.append("content-based semantic similarity")
        elif similarity_type == "speaker":
            analysis_parts.append("speaker-based similarity")
        elif similarity_type == "topic":
            analysis_parts.append("topic-based similarity")
        else:
            analysis_parts.append("comprehensive similarity analysis")

        if reference_talk_title or reference_talk_id:
            analysis_parts.append(f"using '{reference_info.get('title','?')}' as reference")
        else:
            analysis_parts.append(f"using query '{reference_query}' as reference")

        if exclude_same_speaker:
            filters_applied.append("excluding same speaker")
        if min_similarity is not None:
            filters_applied.append(f"minimum similarity {min_similarity}")

        return {
            "similar_talks": final_similar_talks,
            "reference_info": reference_info,
            "similarity_analysis": f"Performed {', '.join(analysis_parts)}",
            "total_found": len(final_similar_talks),
            "filters_applied": filters_applied,
            "success": True
        }

    except Exception as e:
        return {
            "similar_talks": [],
            "reference_info": {},
            "similarity_analysis": "Similarity analysis failed",
            "total_found": 0,
            "filters_applied": [],
            "success": False,
            "error": str(e)
        }
