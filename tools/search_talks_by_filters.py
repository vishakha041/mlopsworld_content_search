# ===== TOOL 1: SEARCH TALKS BY FILTERS =====

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from .utils import (
    get_db_connector, safe_get, format_date_constraint,
    get_sort_key, get_sort_description
)

class SearchTalksByFiltersInput(BaseModel):
    """Input schema for searching talks by various filters and criteria."""
    
    date_from: Optional[str] = Field(
        None, 
        description="Filter talks published from this date (format: YYYY-MM-DD, YYYY-MM, or YYYY). Example: '2024-01-01'"
    )
    date_to: Optional[str] = Field(
        None,
        description="Filter talks published up to this date (format: YYYY-MM-DD, YYYY-MM, or YYYY). Example: '2024-12-31'"  
    )
    min_views: Optional[int] = Field(
        None,
        description="Minimum YouTube view count required. Example: 1000 for talks with at least 1K views"
    )
    max_views: Optional[int] = Field(
        None,
        description="Maximum YouTube view count allowed. Example: 50000 for talks with under 50K views"
    )
    category: Optional[str] = Field(
        None,
        description="Filter by primary category. Common values: 'MLOps', 'Deployment and integration', 'Future trends', etc."
    )
    track: Optional[str] = Field(
        None,
        description="Filter by conference track/session type"
    )
    event_name: Optional[str] = Field(
        None,
        description="Filter by specific event name. Example: 'MLOps & GenAI World 2024'"
    )
    min_tech_level: Optional[int] = Field(
        None,
        description="Minimum technical level (1-7 scale, where 1=beginner, 7=expert). Example: 3 for intermediate+"
    )
    max_tech_level: Optional[int] = Field(
        None,
        description="Maximum technical level (1-7 scale). Example: 4 for beginner to intermediate"
    )
    company_name: Optional[str] = Field(
        None,
        description="Filter by speaker's company name. Example: 'Google', 'Microsoft', 'OpenAI'"
    )
    industries: Optional[str] = Field(
        None,
        description="Filter by relevant industries mentioned in the talk"
    )
    speaker_name: Optional[str] = Field(
        None,
        description="Filter by exact speaker name. Use full name as stored in database"
    )
    sort_by: Optional[Literal["date", "views", "title", "tech_level"]] = Field(
        "date",
        description="Sort results by: 'date' (publish date), 'views' (YouTube views), 'title' (alphabetical), 'tech_level'"
    )
    sort_order: Optional[Literal["asc", "desc"]] = Field(
        "desc", 
        description="Sort order: 'asc' (ascending/oldest first) or 'desc' (descending/newest first)"
    )
    limit: Optional[int] = Field(
        10,
        description="Maximum number of results to return (default: 10, max recommended: 100)"
    )

@tool("search_talks_by_filters", args_schema=SearchTalksByFiltersInput)
def search_talks_by_filters(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None, 
    min_views: Optional[int] = None,
    max_views: Optional[int] = None,
    category: Optional[str] = None,
    track: Optional[str] = None,
    event_name: Optional[str] = None,
    min_tech_level: Optional[int] = None,
    max_tech_level: Optional[int] = None,
    company_name: Optional[str] = None,
    industries: Optional[str] = None,
    speaker_name: Optional[str] = None,
    sort_by: Optional[str] = "date",
    sort_order: Optional[str] = "desc",
    limit: Optional[int] = 10
) -> Dict[str, Any]:
    """
    Search and filter MLOps talks by various criteria including dates, view counts, 
    categories, speakers, companies, and technical levels.
    
    This tool handles all basic filtering and sorting operations on the talks dataset.
    It can filter by publication dates, YouTube metrics, conference categories, 
    speaker information, and technical difficulty levels. Results are returned 
    with comprehensive metadata including speaker names, view counts, and URLs.
    
    Use this tool when users ask questions like:
    - "Show me talks from 2024" 
    - "Find talks with more than 1000 views"
    - "Get MLOps talks from advanced level speakers"
    - "Top 10 most viewed talks"
    - "Find talks by speakers from Google"
    - "Show me beginner-friendly talks"
    
    Args:
        date_from: Start date filter (YYYY-MM-DD, YYYY-MM, or YYYY format)
        date_to: End date filter (YYYY-MM-DD, YYYY-MM, or YYYY format)  
        min_views: Minimum YouTube view count
        max_views: Maximum YouTube view count
        category: Primary category filter (e.g., 'MLOps', 'Deployment and integration')
        track: Conference track/session type
        event_name: Specific event name (e.g., 'MLOps & GenAI World 2024')
        min_tech_level: Minimum technical level (1-7 scale)
        max_tech_level: Maximum technical level (1-7 scale)
        company_name: Speaker's company name
        industries: Relevant industries
        speaker_name: Specific speaker name (exact match)
        sort_by: Sort field ('date', 'views', 'title', 'tech_level')
        sort_order: Sort direction ('asc' or 'desc')
        limit: Maximum results to return (default 10)
        
    Returns:
        Dict containing:
        - 'results': List of matching talks with metadata
        - 'total_found': Number of talks matching the criteria  
        - 'query_summary': Human-readable description of applied filters
        - 'sort_info': Applied sorting information
    """
    
    try:
        con = get_db_connector()
        
        # Build constraints dynamically
        constraints = {}
        query_parts = []
        
        # Date range constraints
        if date_from:
            date_constraint = format_date_constraint(date_from)
            if date_constraint:
                constraints["yt_published_at"] = constraints.get("yt_published_at", []) + [">=", date_constraint]
                query_parts.append(f"published from {date_from}")
                
        if date_to:
            date_constraint = format_date_constraint(date_to)  
            if date_constraint:
                constraints["yt_published_at"] = constraints.get("yt_published_at", []) + ["<=", date_constraint]
                query_parts.append(f"published until {date_to}")
        
        # View count constraints
        if min_views is not None:
            constraints["yt_views"] = constraints.get("yt_views", []) + [">=", min_views]
            query_parts.append(f"≥{min_views:,} views")
            
        if max_views is not None:
            constraints["yt_views"] = constraints.get("yt_views", []) + ["<=", max_views]
            query_parts.append(f"≤{max_views:,} views")
            
        # Categorical constraints
        if category:
            constraints["category_primary"] = ["==", category]
            query_parts.append(f"category '{category}'")
            
        if track:
            constraints["track"] = ["==", track]
            query_parts.append(f"track '{track}'")
            
        if event_name:
            constraints["event_name"] = ["==", event_name]
            query_parts.append(f"event '{event_name}'")
            
        if company_name:
            constraints["company_name"] = ["==", company_name]
            query_parts.append(f"company '{company_name}'")
            
        if industries:
            constraints["industries"] = ["==", industries]
            query_parts.append(f"industry '{industries}'")
            
        # Technical level constraints
        if min_tech_level is not None:
            constraints["tech_level"] = constraints.get("tech_level", []) + [">=", min_tech_level]
            query_parts.append(f"tech level ≥{min_tech_level}")
            
        if max_tech_level is not None:
            constraints["tech_level"] = constraints.get("tech_level", []) + ["<=", max_tech_level]
            query_parts.append(f"tech level ≤{max_tech_level}")
        
        # Handle speaker name filter (requires join with Person entities)
        if speaker_name:
            query_parts.append(f"speaker '{speaker_name}'")
            # Multi-step query: Person -> Talk
            # Build Talk query dynamically
            talk_query = {
                "with_class": "Talk",
                "is_connected_to": {
                    "ref": 1,
                    "direction": "in", 
                    "connection_class": "TalkHasSpeaker"
                },
                "sort": {
                    "key": get_sort_key(sort_by),
                    "order": "descending" if sort_order == "desc" else "ascending"
                },
                "limit": limit,
                "results": {
                    "list": [
                        "talk_id", "talk_title", "speaker_name", "company_name",
                        "yt_views", "yt_published_at", "youtube_url", "event_name", 
                        "category_primary", "tech_level", "abstract", "track"
                    ]
                }
            }
            
            # Only add constraints if they exist
            if constraints:
                talk_query["constraints"] = constraints
            
            q = [
                {
                    "FindEntity": {
                        "_ref": 1,
                        "with_class": "Person", 
                        "unique": True,
                        "constraints": {"name": ["==", speaker_name]}
                    }
                },
                {"FindEntity": talk_query}
            ]
        else:
            # Single query for non-speaker filters
            # Build FindEntity query dynamically - only include constraints if they exist
            find_entity_query = {
                "with_class": "Talk",
                "sort": {
                    "key": get_sort_key(sort_by),
                    "order": "descending" if sort_order == "desc" else "ascending"
                },
                "limit": limit,
                "results": {
                    "list": [
                        "talk_id", "talk_title", "speaker_name", "company_name",
                        "yt_views", "yt_published_at", "youtube_url", "event_name",
                        "category_primary", "tech_level", "abstract", "track"
                    ]
                }
            }
            
            # Only add constraints if they exist
            if constraints:
                find_entity_query["constraints"] = constraints
            
            q = [{"FindEntity": find_entity_query}]
        
        # Execute query 
        resp, _ = con.query(q)
        
        # Extract results from response
        if speaker_name:
            # Multi-step query result
            if len(resp) > 1 and "FindEntity" in resp[1]:
                entities = resp[1]["FindEntity"].get("entities", [])
                total_found = resp[1]["FindEntity"].get("returned", 0)
            else:
                entities = []
                total_found = 0
        else:
            # Single query result
            if len(resp) > 0 and "FindEntity" in resp[0]:
                entities = resp[0]["FindEntity"].get("entities", [])
                total_found = resp[0]["FindEntity"].get("returned", 0)  
            else:
                entities = []
                total_found = 0
        
        # Format results
        results = []
        for entity in entities:
            # Format publish date
            pub_date = safe_get(entity, "yt_published_at")
            if pub_date and isinstance(pub_date, dict) and "_date" in pub_date:
                pub_date = pub_date["_date"].split("T")[0]  # Extract date part
            
            result = {
                "talk_id": safe_get(entity, "talk_id"),
                "title": safe_get(entity, "talk_title"),
                "speaker": safe_get(entity, "speaker_name"), 
                "company": safe_get(entity, "company_name"),
                "views": safe_get(entity, "yt_views", 0),
                "published_date": pub_date,
                "youtube_url": safe_get(entity, "youtube_url"),
                "event": safe_get(entity, "event_name"),
                "category": safe_get(entity, "category_primary"),
                "tech_level": safe_get(entity, "tech_level"),
                "track": safe_get(entity, "track"),
                "abstract": safe_get(entity, "abstract", "")[:200] + "..." if safe_get(entity, "abstract", "") else ""
            }
            results.append(result)
        
        # Create query summary
        if query_parts:
            query_summary = f"Talks filtered by: {', '.join(query_parts)}"
        else:
            query_summary = "All talks (no filters applied)"
            
        sort_desc = get_sort_description(sort_by, sort_order)
        sort_info = f"Sorted by {sort_by} ({sort_desc} first)"
        
        return {
            "results": results,
            "total_found": total_found,
            "query_summary": query_summary,
            "sort_info": sort_info,
            "success": True
        }
        
    except Exception as e:
        return {
            "results": [],
            "total_found": 0,
            "query_summary": "Query failed",
            "sort_info": "",
            "success": False,
            "error": str(e)
        }