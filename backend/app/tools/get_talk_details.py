# ===== TOOL 4: GET TALK DETAILS =====

from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Import shared utilities
from .utils import (
    get_db_connector, safe_get, format_date_constraint,
    get_embedding_model, to_blob, SET_TRANSCRIPT, SET_META, SET_BIO, EMBED_MODEL
)

class GetTalkDetailsInput(BaseModel):
    """Input schema for retrieving detailed information about a specific talk."""
    
    talk_title: Optional[str] = Field(
        None,
        description="Exact title of the talk to retrieve. Use this for title-based search. Example: 'LLMs, from Playgrounds to Production-ready Pipelines'"
    )
    talk_id: Optional[str] = Field(
        None,
        description="Unique talk identifier. Use this when you have the specific talk ID from previous searches"
    )
    include_transcript: Optional[bool] = Field(
        False,
        description="Whether to include transcript chunks in the response. Set to True to get talk content for analysis"
    )
    time_start: Optional[int] = Field(
        None,
        description="Start time in seconds for transcript filtering. Only applies when include_transcript=True. Example: 300 for 5 minutes"
    )
    time_end: Optional[int] = Field(
        None,
        description="End time in seconds for transcript filtering. Only applies when include_transcript=True. Example: 900 for 15 minutes"
    )
    max_chunks: Optional[int] = Field(
        10,
        description="Maximum number of transcript chunks to return if include_transcript=True (default: 10, max: 100)"
    )
    include_related: Optional[bool] = Field(
        False,
        description="Whether to find and include semantically related talks. Useful for topic exploration"
    )
    related_count: Optional[int] = Field(
        5,
        description="Number of related talks to return if include_related=True (default: 5)"
    )

@tool("get_talk_details", args_schema=GetTalkDetailsInput)
def get_talk_details(
    talk_title: Optional[str] = None,
    talk_id: Optional[str] = None,
    include_transcript: Optional[bool] = False,
    time_start: Optional[int] = None,
    time_end: Optional[int] = None,
    max_chunks: Optional[int] = 10,
    include_related: Optional[bool] = False,
    related_count: Optional[int] = 5
) -> Dict[str, Any]:
    """
    Retrieve comprehensive details about a specific MLOps talk including metadata,
    transcript chunks, speaker information, and optionally related talks.
    
    This tool provides deep dive capabilities into individual talks with optional
    transcript content and semantic similarity finding. It handles both title-based
    and ID-based lookups, with flexible transcript filtering by time ranges.
    Use for detailed talk analysis and content exploration.
    
    Use this tool when users ask questions like:
    - "Tell me about the talk on RAG deployment"
    - "Show me the transcript of this talk"
    - "Get details for talk ID xyz"
    - "What was said in the first 5 minutes of this talk?"
    - "Find talks similar to the LangChain presentation"
    - "Get the full abstract and speaker info for this talk"
    
    Args:
        talk_title: Exact title of the talk (alternative to talk_id)
        talk_id: Unique identifier for the talk (alternative to talk_title)
        include_transcript: Whether to include transcript chunks
        time_start: Start time filter for transcript (seconds)
        time_end: End time filter for transcript (seconds)
        max_chunks: Maximum transcript chunks to return
        include_related: Whether to find semantically related talks
        related_count: Number of related talks to find
        
    Returns:
        Dict containing:
        - 'talk_info': Complete talk metadata and speaker information
        - 'transcript_chunks': List of transcript segments (if requested)
        - 'related_talks': Semantically similar talks (if requested)  
        - 'summary': Summary of the information retrieved
        - 'transcript_stats': Statistics about transcript content (if included)
    """
    
    try:
        con = get_db_connector()
        
        # Validate input - exactly one of talk_title or talk_id must be provided
        if not talk_title and not talk_id:
            return {
                "talk_info": {},
                "transcript_chunks": [],
                "related_talks": [],
                "summary": "Error: Either talk_title or talk_id must be provided",
                "transcript_stats": {},
                "success": False,
                "error": "Missing required parameter: either 'talk_title' or 'talk_id' must be specified"
            }
            
        if talk_title and talk_id:
            return {
                "talk_info": {},
                "transcript_chunks": [],
                "related_talks": [],
                "summary": "Error: Provide either talk_title OR talk_id, not both",
                "transcript_stats": {},
                "success": False,
                "error": "Conflicting parameters: specify either 'talk_title' or 'talk_id', not both"
            }
        
        # Build base query to find the talk
        if talk_title:
            talk_constraint = {"talk_title": ["==", talk_title]}
            search_description = f"title '{talk_title}'"
        else:
            talk_constraint = {"talk_id": ["==", talk_id]}
            search_description = f"ID '{talk_id}'"
        
        # Base query: Get talk details and speaker info
        q = [
            {
                "FindEntity": {
                    "_ref": 1,
                    "with_class": "Talk",
                    "unique": True,
                    "constraints": talk_constraint,
                    "results": {
                        "list": [
                            "talk_id", "talk_title", "speaker_name", "company_name",
                            "yt_views", "yt_published_at", "youtube_url", "event_name",
                            "category_primary", "tech_level", "abstract", "track",
                            "industries", "keywords_csv", "yt_duration_sec"
                        ]
                    }
                }
            },
            {
                "FindEntity": {
                    "with_class": "Person",
                    "is_connected_to": {
                        "ref": 1,
                        "direction": "out",
                        "connection_class": "TalkHasSpeaker"
                    },
                    "results": {
                        "list": ["name"]
                    }
                }
            }
        ]
        
        # Add transcript query if requested
        if include_transcript:
            transcript_constraints = {}
            
            # Apply time filtering if specified
            if time_start is not None:
                transcript_constraints["start_sec"] = transcript_constraints.get("start_sec", []) + [">=", time_start]
            if time_end is not None:
                transcript_constraints["end_sec"] = transcript_constraints.get("end_sec", []) + ["<=", time_end]
            
            fd = {
                    "FindDescriptor": {
                        "set": SET_TRANSCRIPT,
                        "is_connected_to": { "ref": 1, "connection_class": "TalkHasTranscriptChunk" },
                        "sort": {"key": "seq", "order": "ascending"},
                        "limit": int(min(max_chunks or 10, 100)),
                        "results": { "list": ["chunk_id","seq","start_sec","end_sec","chunk_text"] }
                    }
                }
            
            # Only add constraints if present
            transcript_constraints = {}
            if time_start is not None:
                transcript_constraints.setdefault("start_sec", []).extend([">=", int(time_start)])
            if time_end is not None:
                transcript_constraints.setdefault("end_sec", []).extend(["<=", int(time_end)])

            if transcript_constraints:
                fd["FindDescriptor"]["constraints"] = transcript_constraints

            q.append(fd)
                        
        # Execute base query
        resp, _ = con.query(q)
        
        # Process talk info
        if len(resp) == 0 or "FindEntity" not in resp[0] or not resp[0]["FindEntity"].get("entities"):
            return {
                "talk_info": {},
                "transcript_chunks": [],
                "related_talks": [],
                "summary": f"Talk not found with {search_description}",
                "transcript_stats": {},
                "success": False,
                "error": f"No talk found with {search_description}"
            }
        
        talk_entity = resp[0]["FindEntity"]["entities"][0]
        
        # Format publish date
        pub_date = safe_get(talk_entity, "yt_published_at")
        if pub_date and isinstance(pub_date, dict) and "_date" in pub_date:
            pub_date = pub_date["_date"].split("T")[0]
        
        # Format duration
        duration_sec = safe_get(talk_entity, "yt_duration_sec", 0)
        duration_formatted = f"{duration_sec // 60}:{duration_sec % 60:02d}" if duration_sec else "Unknown"
        
        # Build talk info
        talk_info = {
            "talk_id": safe_get(talk_entity, "talk_id"),
            "title": safe_get(talk_entity, "talk_title"),
            "speaker": safe_get(talk_entity, "speaker_name"),
            "company": safe_get(talk_entity, "company_name"),
            "views": safe_get(talk_entity, "yt_views", 0),
            "published_date": pub_date,
            "duration": duration_formatted,
            "youtube_url": safe_get(talk_entity, "youtube_url"),
            "event": safe_get(talk_entity, "event_name"),
            "category": safe_get(talk_entity, "category_primary"),
            "track": safe_get(talk_entity, "track"),
            "tech_level": safe_get(talk_entity, "tech_level"),
            "industries": safe_get(talk_entity, "industries"),
            "keywords": safe_get(talk_entity, "keywords"),
            "abstract": safe_get(talk_entity, "abstract", "")
        }
        
        # Process speaker info
        speaker_info = {}
        if len(resp) > 1 and "FindEntity" in resp[1] and resp[1]["FindEntity"].get("entities"):
            speaker_entity = resp[1]["FindEntity"]["entities"][0]
            speaker_info = {
                "name": safe_get(speaker_entity, "name"),
                "bio": safe_get(speaker_entity, "bio_text", ""),
                "company": safe_get(speaker_entity, "company")
            }
        
        talk_info["speaker_details"] = speaker_info
        
        # Process transcript chunks
        transcript_chunks = []
        transcript_stats = {}
        
        if include_transcript and len(resp) > 2 and "FindDescriptor" in resp[2]:
            chunks = resp[2]["FindDescriptor"].get("entities", [])
            
            for chunk in chunks:
                transcript_chunks.append({
                    "sequence": safe_get(chunk, "seq"),
                    "start_time": safe_get(chunk, "start_sec", 0),
                    "end_time": safe_get(chunk, "end_sec", 0),
                    "duration": safe_get(chunk, "end_sec", 0) - safe_get(chunk, "start_sec", 0),
                    "text": safe_get(chunk, "chunk_text", ""),
                    "timestamp_formatted": f"{safe_get(chunk, 'start_sec', 0)//60}:{safe_get(chunk, 'start_sec', 0)%60:02d}"
                })
            
            # Calculate transcript statistics
            total_text_length = sum(len(chunk["text"]) for chunk in transcript_chunks)
            time_range_start = min(chunk["start_time"] for chunk in transcript_chunks) if transcript_chunks else 0
            time_range_end = max(chunk["end_time"] for chunk in transcript_chunks) if transcript_chunks else 0
            
            transcript_stats = {
                "chunks_returned": len(transcript_chunks),
                "total_text_length": total_text_length,
                "time_range": f"{time_range_start//60}:{time_range_start%60:02d} - {time_range_end//60}:{time_range_end%60:02d}",
                "time_filter_applied": time_start is not None or time_end is not None
            }
        
        # Find related talks if requested
        related_talks = []
        if include_related and talk_info.get("abstract"):
            try:
                model = get_embedding_model()
                query_text = f"query: {talk_info['abstract'][:500]}"  # or use full if you like
                qvec = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0].astype("<f4")
                query_blob = to_blob(qvec)

                related_q = [
                    {
                        "FindDescriptor": {
                            "_ref": 1,
                            "set": SET_META,                          # e.g., "ds_talk_meta_v1"
                            "k_neighbors": int((related_count or 5) + 3),  # a few extra to allow skipping self/dups
                            "distances": True,                        # ensure _distance is populated
                            "results": {
                                "list": ["_distance", "talk_id", "meta_text"]
                            }
                        }
                    },
                    {
                        "FindEntity": {
                            "with_class": "Talk",
                            "is_connected_to": {
                                "ref": 1,
                                "connection_class": "TalkHasMeta"
                            },
                            "results": {
                                # IMPORTANT: include talk_id here
                                "list": ["talk_id", "talk_title", "speaker_name", "youtube_url", "yt_views", "category_primary"]
                            }
                        }
                    }
                ]

                related_resp, _ = con.query(related_q, blobs=[query_blob])

                related_talks = []
                if len(related_resp) >= 2:
                    desc_entities = related_resp[0]["FindDescriptor"].get("entities", []) or []
                    talk_entities = related_resp[1]["FindEntity"].get("entities", []) or []

                    # Build map from talk_id -> talk entity (now we actually have talk_id)
                    talk_map = {safe_get(t, "talk_id"): t for t in talk_entities if safe_get(t, "talk_id")}

                    seen = set()
                    for desc in desc_entities:
                        desc_tid = safe_get(desc, "talk_id")
                        if not desc_tid or desc_tid == talk_info["talk_id"]:
                            continue
                        if desc_tid in seen:
                            continue

                        talk_row = talk_map.get(desc_tid)
                        if not talk_row:
                            # If not found, we can still surface via desc only, but we prefer mapped rows
                            continue

                        distance = safe_get(desc, "_distance", None)
                        similarity = round(1 - float(distance), 3) if distance is not None else None

                        related_talks.append({
                            "title": safe_get(talk_row, "talk_title"),
                            "speaker": safe_get(talk_row, "speaker_name"),
                            "youtube_url": safe_get(talk_row, "youtube_url"),
                            "views": safe_get(talk_row, "yt_views", 0),
                            "category": safe_get(talk_row, "category_primary"),
                            "similarity_score": similarity
                        })
                        seen.add(desc_tid)

                        if len(related_talks) >= int(related_count or 5):
                            break
                
            except Exception as e:
                # Don't fail the entire request if related talks search fails
                related_talks = []
        
        # Build summary
        summary_parts = [f"Retrieved details for talk '{talk_info['title']}'"]
        if include_transcript:
            summary_parts.append(f"{len(transcript_chunks)} transcript chunks")
            if time_start is not None or time_end is not None:
                summary_parts.append("with time filtering")
        if include_related:
            summary_parts.append(f"{len(related_talks)} related talks")
        
        summary = " | ".join(summary_parts)
        
        return {
            "talk_info": talk_info,
            "transcript_chunks": transcript_chunks,
            "related_talks": related_talks,
            "summary": summary,
            "transcript_stats": transcript_stats,
            "success": True
        }
        
    except Exception as e:
        return {
            "talk_info": {},
            "transcript_chunks": [],
            "related_talks": [],
            "summary": "Failed to retrieve talk details",
            "transcript_stats": {},
            "success": False,
            "error": str(e)
        }