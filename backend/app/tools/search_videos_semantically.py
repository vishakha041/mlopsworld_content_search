"""
Video Semantic Search Tool

This tool performs semantic search on video content using Twelve Labs embeddings.
Unlike text-based semantic search (which searches transcripts/abstracts), this
searches at the video level using visual and audio understanding.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import numpy as np
import logging

from .utils import (
    get_db_connector,
    get_twelvelabs_client,
    VIDEO_DESCRIPTOR_SET,
    VIDEO_EMBED_MODEL,
    VIDEO_EMBED_DIM
)

# Configure logging
logging.basicConfig(level=logging.INFO)


class SearchVideosSemanticallyInput(BaseModel):
    """Input schema for semantic video search using Twelve Labs embeddings."""
    
    query: str = Field(
        ...,
        description="Natural language search query describing the video content you're looking for. Examples: 'AI agents and LLMs in production', 'Vector databases and RAG systems', 'MLOps deployment strategies'"
    )
    top_n: Optional[int] = Field(
        3,
        description="Number of most similar videos to return (default: 3, max recommended: 10)"
    )
    return_blobs: Optional[bool] = Field(
        False,
        description="Whether to return video blobs (set False for metadata-only results, True to get videos for playback)"
    )


@tool("search_videos_semantically", args_schema=SearchVideosSemanticallyInput)
def search_videos_semantically(
    query: str,
    top_n: Optional[int] = 3,
    return_blobs: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Perform semantic search on conference talk videos using Twelve Labs embeddings.
    
    This tool searches across video content using AI-powered video understanding.
    Unlike transcript-based search, this understands visual and audio content
    in videos using the Twelve Labs Marengo model.
    
    The search works by:
    1. Converting your query to a 1024-dimensional embedding using Twelve Labs
    2. Performing k-NN similarity search in ApertureDB
    3. Returning the most semantically similar videos with metadata
    
    Use this tool when users want to:
    - Find videos about specific topics visually shown in talks
    - Search based on presentation style or demo content
    - Discover similar video content (not just transcript text)
    
    Args:
        query: Natural language description of desired video content
        top_n: Number of similar videos to return (default: 3)
        return_blobs: Whether to include video blobs in results (default: False)
        
    Returns:
        Dict containing:
        - 'success': Boolean indicating if search succeeded
        - 'results': List of matching videos with metadata and similarity scores
        - 'total_found': Number of results returned
        - 'search_summary': Description of search performed
        - 'error': Error message if search failed (only present on failure)
    """
    
    try:
        # Get database connector and Twelve Labs client
        client = get_db_connector()
        tl = get_twelvelabs_client()
        
        if not tl:
            return {
                "success": False,
                "results": [],
                "total_found": 0,
                "search_summary": f"Video search for '{query}' failed",
                "error": "Twelve Labs client not available"
            }
        
        # 1. Create text embedding for the query using Twelve Labs
        logging.info(f"Creating video embedding for query: '{query}'")
        t = tl.embed.create(model_name=VIDEO_EMBED_MODEL, text=query)
        qvec = np.asarray(t.text_embedding.segments[0].float_, dtype=np.float32)
        
        # 2. Perform k-NN search in ApertureDB
        logging.info(f"Searching for top {top_n} similar videos...")
        search_q = [
            {
                "FindDescriptor": {
                    "_ref": 1,
                    "set": VIDEO_DESCRIPTOR_SET,
                    "k_neighbors": top_n,
                    "metric": "L2",
                    "distances": True
                }
            },
            {
                "FindVideo": {
                    "_ref": 2,
                    "is_connected_to": {"ref": 1},
                    "results": {"list": ["talk_title", "_uniqueid"]},
                    "blobs": return_blobs
                }
            }
        ]
        
        search_results, blobs = client.query(search_q, blobs=[qvec.tobytes()])
        
        # 3. Check and parse results
        find_video_response = search_results[1]
        if find_video_response.get("FindVideo", {}).get("returned", 0) == 0:
            logging.warning("No videos found matching search query")
            return {
                "success": True,
                "results": [],
                "total_found": 0,
                "search_summary": f"Video search for '{query}' returned no matches"
            }
        
        video_entities = search_results[1]["FindVideo"]["entities"]
        descriptor_entities = search_results[0]["FindDescriptor"]["entities"]
        distances = [entity["_distance"] for entity in descriptor_entities]
        
        # 4. For each video, query its connected Talk entity individually
        # Log distance range for debugging
        if distances:
            logging.info(f"L2 distance range: min={min(distances):.4f}, max={max(distances):.4f}")
        
        results = []
        for i, video_entity in enumerate(video_entities):
            distance = distances[i]
            # Calculate similarity score from L2 distance
            # For normalized embeddings: L2² = 2(1 - cosine_sim)
            # So: cosine_sim = 1 - (L2² / 2)
            # This gives proper 0-100% range for normalized vectors
            similarity = max(0, min(1, 1 - (distance ** 2) / 2))
            
            video_title = video_entity.get("talk_title", "N/A")
            video_uid = video_entity.get("_uniqueid")
            
            # Query the connected Talk entity for this specific video
            talk_data = {}
            if video_uid:
                talk_query = [
                    {
                        "FindVideo": {
                            "_ref": 1,
                            "constraints": {"_uniqueid": ["==", video_uid]}
                        }
                    },
                    {
                        "FindEntity": {
                            "with_class": "Talk",
                            "is_connected_to": {
                                "ref": 1,
                                "direction": "out",
                                "connection_class": "BELONGS_TO_TALK"
                            },
                            "results": {
                                "list": [
                                    "talk_id", "talk_title", "speaker_name", "company_name",
                                    "event_name", "category_primary", "track", "abstract",
                                    "youtube_url", "youtube_id", "yt_views", "yt_published_at",
                                    "tech_level", "keywords_csv", "job_title"
                                ]
                            }
                        }
                    }
                ]
                try:
                    talk_resp, _ = client.query(talk_query)
                    talk_entities = talk_resp[1].get("FindEntity", {}).get("entities", [])
                    if talk_entities:
                        talk_data = talk_entities[0]
                except Exception as e:
                    logging.warning(f"Failed to get Talk for video {video_uid}: {e}")
            
            # Format publish date
            pub_date = talk_data.get("yt_published_at")
            if pub_date and isinstance(pub_date, dict) and "_date" in pub_date:
                pub_date = pub_date["_date"].split("T")[0]
            
            result = {
                "talk_title": talk_data.get("talk_title") or video_title,
                "speaker_name": talk_data.get("speaker_name", "Unknown"),
                "company_name": talk_data.get("company_name", ""),
                "job_title": talk_data.get("job_title", ""),
                "event_name": talk_data.get("event_name", ""),
                "category_primary": talk_data.get("category_primary", ""),
                "track": talk_data.get("track", ""),
                "abstract": talk_data.get("abstract", ""),
                "youtube_url": talk_data.get("youtube_url", ""),
                "youtube_id": talk_data.get("youtube_id", ""),
                "yt_views": talk_data.get("yt_views", 0),
                "published_date": pub_date,
                "tech_level": talk_data.get("tech_level"),
                "keywords": talk_data.get("keywords_csv", ""),
                "distance": round(distance, 4),
                "similarity_score": round(similarity, 3),
            }
            
            # Add video blob if requested
            if return_blobs and blobs and i < len(blobs):
                result["video_blob"] = blobs[i]
            
            results.append(result)
        
        logging.info(f"Found {len(results)} matching videos")
        
        return {
            "success": True,
            "results": results,
            "total_found": len(results),
            "search_summary": f"Video semantic search for '{query}' using Twelve Labs {VIDEO_EMBED_MODEL} ({VIDEO_EMBED_DIM}D embeddings)"
        }
        
    except Exception as e:
        logging.error(f"Error during video semantic search: {str(e)}")
        return {
            "success": False,
            "results": [],
            "total_found": 0,
            "search_summary": f"Video search for '{query}' failed",
            "error": str(e)
        }
