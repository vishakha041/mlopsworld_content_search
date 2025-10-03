# ===== TOOL 7: GET UNIQUE VALUES =====

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from .utils import get_db_connector, safe_get


class GetUniqueValuesInput(BaseModel):
    """Input schema for getting unique values from Talk entity properties."""
    
    event_name: Optional[bool] = Field(
        False,
        description="Set to True to get all unique event names"
    )
    category_primary: Optional[bool] = Field(
        False,
        description="Set to True to get all unique primary categories"
    )
    track: Optional[bool] = Field(
        False,
        description="Set to True to get all unique tracks"
    )
    company_name: Optional[bool] = Field(
        False,
        description="Set to True to get all unique company names"
    )
    tech_level: Optional[bool] = Field(
        False,
        description="Set to True to get all unique technical levels"
    )
    industries: Optional[bool] = Field(
        False,
        description="Set to True to get all unique industries"
    )


@tool("get_unique_values", args_schema=GetUniqueValuesInput)
def get_unique_values(
    event_name: Optional[bool] = False,
    category_primary: Optional[bool] = False,
    track: Optional[bool] = False,
    company_name: Optional[bool] = False,
    tech_level: Optional[bool] = False,
    industries: Optional[bool] = False
) -> Dict[str, Any]:
    """
    Get all unique values for specified Talk entity properties.
    
    This tool retrieves distinct/unique values from one or more Talk entity 
    properties. Useful for understanding available options, filtering criteria,
    or exploring the dataset's categorical values.
    
    Use this tool when users ask questions like:
    - "What events are in the database?"
    - "Show me all available categories"
    - "Which companies have presented?"
    - "What are the technical level options?"
    - "List all unique tracks"
    
    Args:
        event_name: Get unique event names
        category_primary: Get unique primary categories
        track: Get unique conference tracks
        company_name: Get unique company names
        tech_level: Get unique technical levels (1-7)
        industries: Get unique industries
        
    Returns:
        Dict containing:
        - Requested property names as keys
        - Lists of unique values (sorted) for each property
        - Count of unique values per property
        - Total entities queried
    """
    
    try:
        # Collect requested properties
        requested_props = []
        if event_name:
            requested_props.append("event_name")
        if category_primary:
            requested_props.append("category_primary")
        if track:
            requested_props.append("track")
        if company_name:
            requested_props.append("company_name")
        if tech_level:
            requested_props.append("tech_level")
        if industries:
            requested_props.append("industries")
        
        # Validate at least one property is requested
        if not requested_props:
            return {
                "success": False,
                "error": "At least one property must be set to True",
                "unique_values": {},
                "counts": {}
            }
        
        # Get database connector
        con = get_db_connector()
        
        # Build query to fetch all Talk entities with only requested properties
        query = [{
            "FindEntity": {
                "with_class": "Talk",
                "results": {
                    "list": requested_props
                }
            }
        }]
        
        # Execute query
        resp, _ = con.query(query)
        
        # Extract entities from response
        if not resp or len(resp) == 0 or "FindEntity" not in resp[0]:
            return {
                "success": False,
                "error": "No response from database",
                "unique_values": {},
                "counts": {}
            }
        
        entities = resp[0]["FindEntity"].get("entities", [])
        total_entities = len(entities)
        
        if total_entities == 0:
            return {
                "success": True,
                "message": "No Talk entities found in database",
                "unique_values": {},
                "counts": {},
                "total_entities": 0
            }
        
        # Extract unique values for each requested property
        unique_values = {}
        counts = {}
        
        for prop in requested_props:
            values_set = set()
            
            for entity in entities:
                value = safe_get(entity, prop)
                
                # Handle None/null values
                if value is None:
                    values_set.add(None)
                # Handle date objects (extract date string)
                elif isinstance(value, dict) and "_date" in value:
                    values_set.add(value["_date"])
                else:
                    values_set.add(value)
            
            # Convert set to sorted list (None values at the end)
            values_list = sorted(
                [v for v in values_set if v is not None],
                key=lambda x: str(x).lower() if isinstance(x, str) else x
            )
            
            # Add None values at the end if present
            if None in values_set:
                values_list.append(None)
            
            unique_values[prop] = values_list
            counts[prop] = len(values_list)
        
        return {
            "success": True,
            "unique_values": unique_values,
            "counts": counts,
            "total_entities": total_entities,
            "properties_queried": requested_props
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "unique_values": {},
            "counts": {}
        }
