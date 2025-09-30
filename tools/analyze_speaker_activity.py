# ===== TOOL 3: ANALYZE SPEAKER ACTIVITY =====

from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Import shared utilities
from .utils import (
    get_db_connector, safe_get, format_date_constraint, 
    get_sort_key, get_sort_description
)

class AnalyzeSpeakerActivityInput(BaseModel):
    """Input schema for analyzing speaker activity and patterns."""
    
    speaker_name: Optional[str] = Field(
        None,
        description="Specific speaker name to analyze (exact match). Leave empty to analyze all speakers or use other filters"
    )
    company_name: Optional[str] = Field(
        None,
        description="Analyze speakers from a specific company. Example: 'Google', 'Microsoft', 'OpenAI'"
    )
    date_from: Optional[str] = Field(
        None,
        description="Filter speaker activity from this date (YYYY-MM-DD, YYYY-MM, or YYYY). Example: '2024-01-01'"
    )
    date_to: Optional[str] = Field(
        None,
        description="Filter speaker activity up to this date (YYYY-MM-DD, YYYY-MM, or YYYY). Example: '2024-12-31'"
    )
    min_talk_count: Optional[int] = Field(
        None,
        description="Filter to speakers with at least this many talks. Example: 2 for repeat speakers only"
    )
    analysis_type: Optional[Literal["talk_count", "topics", "companies", "all"]] = Field(
        "all",
        description="Type of analysis: 'talk_count' (speaker frequency), 'topics' (semantic topic analysis), 'companies' (company breakdown), 'all' (comprehensive analysis)"
    )
    event_name: Optional[str] = Field(
        None,
        description="Filter to speakers from a specific event. Example: 'MLOps & GenAI World 2024'"
    )
    category: Optional[str] = Field(
        None,
        description="Filter to speakers who presented in a specific category. Example: 'MLOps', 'Deployment and integration'"
    )
    limit: Optional[int] = Field(
        20,
        description="Maximum number of speakers to analyze (default: 20, max recommended: 100)"
    )

@tool("analyze_speaker_activity", args_schema=AnalyzeSpeakerActivityInput)
def analyze_speaker_activity(
    speaker_name: Optional[str] = None,
    company_name: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_talk_count: Optional[int] = None,
    analysis_type: Optional[str] = "all",
    event_name: Optional[str] = None,
    category: Optional[str] = None,
    limit: Optional[int] = 20
) -> Dict[str, Any]:
    """
    Analyze speaker activity patterns, frequency, topics, and company representation
    across the MLOps events dataset. Provides comprehensive speaker-centric analytics
    including talk counts, topic diversity, company breakdown, and repeat speakers.
    
    This tool handles all speaker-focused analytical queries with filtering and 
    statistical aggregation. It can analyze individual speakers, company representation,
    or provide dataset-wide speaker statistics. Results include talk lists, topic
    summaries, and presentation frequency metrics.
    
    Use this tool when users ask questions like:
    - "How many times did John Smith present?"
    - "What topics does this speaker cover?"
    - "Which speakers presented most frequently?"
    - "Find speakers from Google"  
    - "Who are the repeat speakers?"
    - "Analyze company representation in talks"
    - "What topics do OpenAI speakers discuss?"
    
    Args:
        speaker_name: Specific speaker to analyze (exact match)
        company_name: Filter speakers by company affiliation
        date_from: Start date for activity analysis (YYYY-MM-DD format)
        date_to: End date for activity analysis (YYYY-MM-DD format)
        min_talk_count: Minimum number of talks for inclusion (for repeat speakers)
        analysis_type: Type of analysis ('talk_count', 'topics', 'companies', 'all')
        event_name: Filter to specific event
        category: Filter to specific category
        limit: Maximum number of speakers to analyze
        
    Returns:
        Dict containing:
        - 'speaker_stats': Speaker-level statistics and talk counts
        - 'company_breakdown': Company representation analysis (if applicable)
        - 'topic_analysis': Topic patterns for speakers (if requested)
        - 'repeat_speakers': List of speakers with multiple talks
        - 'analysis_summary': Summary of the analysis performed
        - 'total_speakers': Number of unique speakers found
    """
    
    try:
        con = get_db_connector()
        
        # Build talk constraints for filtering
        talk_constraints = {}
        filter_parts = []
        
        if date_from:
            date_constraint = format_date_constraint(date_from)
            if date_constraint:
                talk_constraints["yt_published_at"] = talk_constraints.get("yt_published_at", []) + [">=", date_constraint]
                filter_parts.append(f"from {date_from}")
                
        if date_to:
            date_constraint = format_date_constraint(date_to)
            if date_constraint:
                talk_constraints["yt_published_at"] = talk_constraints.get("yt_published_at", []) + ["<=", date_constraint]
                filter_parts.append(f"until {date_to}")
                
        if event_name:
            talk_constraints["event_name"] = ["==", event_name]
            filter_parts.append(f"event '{event_name}'")
            
        if category:
            talk_constraints["category_primary"] = ["==", category]
            filter_parts.append(f"category '{category}'")
            
        if company_name:
            talk_constraints["company_name"] = ["==", company_name]
            filter_parts.append(f"company '{company_name}'")
            
        # Handle specific speaker analysis
        if speaker_name:
            # Detailed analysis for a specific speaker
            q = [
                {
                    "FindEntity": {
                        "_ref": 1,
                        "with_class": "Person",
                        "unique": True,
                        "constraints": {"name": ["==", speaker_name]},
                        "results": {"list": ["name"]}
                    }
                },
                {
                    "FindEntity": {
                        "with_class": "Talk",
                        "is_connected_to": {
                        "ref": 1,
                        "direction": "in",
                        "connection_class": "TalkHasSpeaker"
                        },
                        **({"constraints": talk_constraints} if talk_constraints else {}),
                        "sort": {"key": "yt_published_at", "order": "descending"},
                        "results": {
                        "list": [
                            "talk_id","talk_title","yt_published_at","yt_views",
                            "youtube_url","event_name","category_primary",
                            "company_name","abstract","track"
                            ]
                        }
                    }
                }       
            ]
            
            resp, _ = con.query(q)
            
            if len(resp) > 1 and "FindEntity" in resp[1]:
                talks = resp[1]["FindEntity"].get("entities", [])
                
                # Calculate speaker statistics
                total_talks = len(talks)
                total_views = sum(safe_get(talk, "yt_views", 0) for talk in talks)
                categories = {safe_get(talk, "category_primary") for talk in talks if safe_get(talk, "category_primary")}
                events = {safe_get(talk, "event_name") for talk in talks if safe_get(talk, "event_name")}
                companies = {safe_get(talk, "company_name") for talk in talks if safe_get(talk, "company_name")}
                
                # Format talk list
                talk_list = []
                for talk in talks:
                    pub_date = safe_get(talk, "yt_published_at")
                    if pub_date and isinstance(pub_date, dict) and "_date" in pub_date:
                        pub_date = pub_date["_date"].split("T")[0]
                        
                    talk_list.append({
                        "title": safe_get(talk, "talk_title"),
                        "date": pub_date,
                        "views": safe_get(talk, "yt_views", 0),
                        "event": safe_get(talk, "event_name"),
                        "category": safe_get(talk, "category_primary"),
                        "youtube_url": safe_get(talk, "youtube_url"),
                        "abstract": safe_get(talk, "abstract", "")[:150] + "..." if safe_get(talk, "abstract", "") else ""
                    })
                
                return {
                    "speaker_stats": [{
                        "speaker_name": speaker_name,
                        "total_talks": total_talks,
                        "total_views": total_views,
                        "avg_views": round(total_views / total_talks) if total_talks > 0 else 0,
                        "categories_covered": list(categories),
                        "events_participated": list(events),
                        "companies": list(companies),
                        "talk_list": talk_list
                    }],
                    "company_breakdown": {},
                    "topic_analysis": {},
                    "repeat_speakers": [],
                    "analysis_summary": f"Detailed analysis for speaker '{speaker_name}' ({filter_parts})" if filter_parts else f"Detailed analysis for speaker '{speaker_name}'",
                    "total_speakers": 1,
                    "success": True
                }
            else:
                return {
                    "speaker_stats": [],
                    "company_breakdown": {},
                    "topic_analysis": {},
                    "repeat_speakers": [],
                    "analysis_summary": f"No talks found for speaker '{speaker_name}'",
                    "total_speakers": 0,
                    "success": True,
                    "error": f"Speaker '{speaker_name}' not found or has no talks matching the criteria"
                }
                
        else:
            # General speaker analysis across dataset
            q = [{
                "FindEntity": {
                        "with_class": "Talk",
                        **({"constraints": talk_constraints} if talk_constraints else {}),
                        "results": {
                        "list": [
                            "talk_id","talk_title","speaker_name","company_name",
                            "yt_views","yt_published_at","event_name","category_primary"
                        ]
                    }
                }
            }]
            
            resp, _ = con.query(q)
            
            if len(resp) > 0 and "FindEntity" in resp[0]:
                talks = resp[0]["FindEntity"].get("entities", [])
                
                # Group talks by speaker
                speaker_stats = {}
                company_counts = {}
                category_counts = {}
                
                for talk in talks:
                    speaker = safe_get(talk, "speaker_name")
                    company = safe_get(talk, "company_name")
                    category = safe_get(talk, "category_primary")
                    views = safe_get(talk, "yt_views", 0)
                    
                    if not speaker:
                        continue
                        
                    # Speaker statistics
                    if speaker not in speaker_stats:
                        speaker_stats[speaker] = {
                            "speaker_name": speaker,
                            "total_talks": 0,
                            "total_views": 0,
                            "companies": set(),
                            "categories": set(),
                            "events": set()
                        }
                    
                    speaker_stats[speaker]["total_talks"] += 1
                    speaker_stats[speaker]["total_views"] += views
                    
                    if company:
                        speaker_stats[speaker]["companies"].add(company)
                        company_counts[company] = company_counts.get(company, 0) + 1
                        
                    if category:
                        speaker_stats[speaker]["categories"].add(category)
                        category_counts[category] = category_counts.get(category, 0) + 1
                        
                    event = safe_get(talk, "event_name")
                    if event:
                        speaker_stats[speaker]["events"].add(event)
                
                # Convert sets to lists and calculate averages
                for stats in speaker_stats.values():
                    stats["companies"] = list(stats["companies"])
                    stats["categories"] = list(stats["categories"])
                    stats["events"] = list(stats["events"])
                    stats["avg_views"] = round(stats["total_views"] / stats["total_talks"]) if stats["total_talks"] > 0 else 0
                
                # Filter by minimum talk count
                if min_talk_count:
                    speaker_stats = {k: v for k, v in speaker_stats.items() if v["total_talks"] >= min_talk_count}
                
                # Sort speakers by talk count (descending)
                sorted_speakers = sorted(speaker_stats.values(), key=lambda x: x["total_talks"], reverse=True)[:limit]
                
                # Find repeat speakers (2+ talks)
                repeat_speakers = [s for s in sorted_speakers if s["total_talks"] >= 2]
                
                # Company breakdown
                sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:20]
                company_breakdown = [{"company": comp, "speaker_count": count} for comp, count in sorted_companies]
                
                # Analysis summary parts
                summary_parts = []
                if analysis_type in ["talk_count", "all"]:
                    summary_parts.append("speaker talk counts")
                if analysis_type in ["companies", "all"]:
                    summary_parts.append("company representation")
                if filter_parts:
                    summary_parts.append(f"filtered by {', '.join(filter_parts)}")
                
                analysis_summary = f"Speaker activity analysis: {', '.join(summary_parts)}"
                
                return {
                    "speaker_stats": sorted_speakers,
                    "company_breakdown": company_breakdown if analysis_type in ["companies", "all"] else {},
                    "topic_analysis": {},  # Could implement semantic topic analysis here
                    "repeat_speakers": repeat_speakers,
                    "analysis_summary": analysis_summary,
                    "total_speakers": len(speaker_stats),
                    "success": True
                }
                    
            else:
                return {
                    "speaker_stats": [],
                    "company_breakdown": {},
                    "topic_analysis": {},
                    "repeat_speakers": [],
                    "analysis_summary": "No talks found matching the criteria",
                    "total_speakers": 0,
                    "success": True
                }
                
    except Exception as e:
        return {
            "speaker_stats": [],
            "company_breakdown": {},
            "topic_analysis": {},
            "repeat_speakers": [],
            "analysis_summary": "Speaker analysis failed",
            "total_speakers": 0,
            "success": False,
            "error": str(e)
        }