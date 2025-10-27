# ===== TOOL 6: ANALYZE TOPICS AND TRENDS =====

from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import re
from collections import defaultdict, Counter

# Import shared utilities
from .utils import (
    get_db_connector, get_embedding_model, to_blob, safe_get,
    format_date_constraint, get_text_field_name,
    SET_TRANSCRIPT, SET_META, SET_BIO, EMBED_MODEL
)

class AnalyzeTopicsAndTrendsInput(BaseModel):
    """Input schema for analyzing topics, tools, technologies, and trends."""
    
    analysis_type: Optional[Literal["tools", "topics", "technologies", "trends", "keywords"]] = Field(
        "topics",
        description="Type of analysis: 'tools' (software tools/libraries), 'topics' (discussion themes), 'technologies' (tech stack mentions), 'trends' (time-based analysis), 'keywords' (frequent terms)"
    )
    date_from: Optional[str] = Field(
        None,
        description="Optional: analyze content from this date onwards (YYYY-MM-DD, YYYY-MM, or YYYY)"
    )
    date_to: Optional[str] = Field(
        None,
        description="Optional: analyze content up to this date (YYYY-MM-DD, YYYY-MM, or YYYY)"
    )
    category: Optional[str] = Field(
        None,
        description="Optional: filter analysis to specific category (e.g., 'MLOps', 'Deployment and integration')"
    )
    event_name: Optional[str] = Field(
        None,
        description="Optional: filter analysis to specific event (e.g., 'MLOps & GenAI World 2024')"
    )
    content_source: Optional[Literal["transcripts", "abstracts", "all"]] = Field(
        "abstracts",
        description="Content source for analysis: 'transcripts' (video content), 'abstracts' (talk summaries), 'all' (comprehensive analysis)"
    )
    top_n: Optional[int] = Field(
        10,
        description="Number of top results to return (default: 10, max recommended: 50)"
    )
    time_grouping: Optional[Literal["monthly", "yearly", "quarterly", "none"]] = Field(
        "none",
        description="Time-based grouping for trend analysis: 'monthly', 'yearly', 'quarterly', or 'none' for no grouping"
    )
    min_mentions: Optional[int] = Field(
        2,
        description="Minimum number of mentions required for inclusion in results (default: 2)"
    )

@tool("analyze_topics_and_trends", args_schema=AnalyzeTopicsAndTrendsInput)
def analyze_topics_and_trends(
    analysis_type: Optional[str] = "topics",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    category: Optional[str] = None,
    event_name: Optional[str] = None,
    content_source: Optional[str] = "abstracts",
    top_n: Optional[int] = 10,
    time_grouping: Optional[str] = "none",
    min_mentions: Optional[int] = 2
) -> Dict[str, Any]:
    """
    Analyze topics, tools, technologies, and trends across the MLOps events dataset.
    Provides high-level analysis of popular topics, trending tools, technology mentions,
    and time-based trend identification with flexible filtering and grouping options.
    
    This tool handles all topic analysis and trend identification operations.
    It can analyze different types of content (tools, topics, technologies) across
    various time periods and content sources. Results include frequency counts,
    trend analysis, and time-based groupings for comprehensive insights.
    
    Use this tool when users ask questions like:
    - "What are the top 10 tools mentioned in talks?"
    - "Show me trending topics in 2024"
    - "Which technologies are most discussed?"
    - "How has the focus on AI agents changed over time?"
    - "What are the popular keywords in MLOps talks?"
    - "Analyze technology trends by quarter"
    
    Args:
        analysis_type: Type of analysis ('tools', 'topics', 'technologies', 'trends', 'keywords')
        date_from: Optional start date filter (YYYY-MM-DD format)
        date_to: Optional end date filter (YYYY-MM-DD format)
        category: Optional category filter
        event_name: Optional event filter
        content_source: Content source ('transcripts', 'abstracts', 'all')
        top_n: Number of top results to return
        time_grouping: Time grouping for trends ('monthly', 'yearly', 'quarterly', 'none')
        min_mentions: Minimum mentions required for inclusion
        
    Returns:
        Dict containing:
        - 'analysis_results': Main analysis results with counts and frequencies
        - 'time_trends': Time-based trend data (if time_grouping specified)
        - 'analysis_summary': Summary of analysis performed
        - 'total_items_found': Number of unique items found
        - 'content_stats': Statistics about content analyzed
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
                
        if category:
            talk_constraints["category_primary"] = ["==", category]
            filter_parts.append(f"category '{category}'")
            
        if event_name:
            talk_constraints["event_name"] = ["==", event_name]
            filter_parts.append(f"event '{event_name}'")
        
        # Determine content sources to analyze
        descriptor_sets = []
        content_descriptions = []
        
        if content_source == "transcripts":
            descriptor_sets = [(SET_TRANSCRIPT, "TalkHasTranscriptChunk", "transcript")]
            content_descriptions = ["video transcripts"]
        elif content_source == "abstracts":
            descriptor_sets = [(SET_META, "TalkHasMeta", "abstract")]
            content_descriptions = ["talk abstracts/metadata"]
        else:  # all
            descriptor_sets = [
                (SET_META, "TalkHasMeta", "abstract"),
                (SET_TRANSCRIPT, "TalkHasTranscriptChunk", "transcript")
            ]
            content_descriptions = ["abstracts and transcripts"]
        
        # Collect all text content with metadata
        all_content = []
        content_stats = {"total_talks": 0, "total_text_chunks": 0, "content_sources": content_descriptions}
        
        for set_name, connection_class, content_type in descriptor_sets:
            if talk_constraints:
                # Filter talks first, then get descriptors
                q = [
                    {
                        "FindEntity": {
                            "_ref": 1,
                            "with_class": "Talk",
                            "constraints": talk_constraints,
                            "results": {"list": ["talk_id", "yt_published_at", "talk_title"]}
                        }
                    },
                    {
                        "FindDescriptor": {
                            "set": set_name,
                            "is_connected_to": {
                                "ref": 1,
                                "connection_class": connection_class
                            },
                            "results": {
                                "list": [get_text_field_name(set_name), "talk_id"]
                            }
                        }
                    }
                ]
            else:
                # Get all descriptors
                q = [
                    {
                        "FindDescriptor": {
                            "_ref": 1,
                            "set": set_name,
                            "results": {
                                "list": [get_text_field_name(set_name), "talk_id"]
                            }
                        }
                    },
                    {
                        "FindEntity": {
                            "with_class": "Talk",
                            "is_connected_to": {
                                "ref": 1,
                                "connection_class": connection_class
                            },
                            "results": {"list": ["talk_id", "yt_published_at", "talk_title"]}
                        }
                    }
                ]
            
            resp, _ = con.query(q)
            
            # Process results
            if len(resp) >= 2:
                desc_resp = None
                talk_resp = None
                
                for r in resp:
                    if "FindDescriptor" in r:
                        desc_resp = r["FindDescriptor"]
                    elif "FindEntity" in r:
                        talk_resp = r["FindEntity"]
                
                if desc_resp and talk_resp:
                    desc_entities = desc_resp.get("entities", [])
                    talk_entities = talk_resp.get("entities", [])
                    
                    # Create talk lookup
                    talk_map = {safe_get(t, "talk_id"): t for t in talk_entities if safe_get(t, "talk_id")}
                    
                    content_stats["total_text_chunks"] += len(desc_entities)
                    content_stats["total_talks"] = len({safe_get(t, "talk_id") for t in talk_entities})
                    
                    for desc in desc_entities:
                        talk_id = safe_get(desc, "talk_id")
                        talk_info = talk_map.get(talk_id, {})
                        
                        text_content = safe_get(desc, get_text_field_name(set_name), "")
                        if text_content:
                            # Extract date for trend analysis
                            pub_date = safe_get(talk_info, "yt_published_at")
                            if pub_date and isinstance(pub_date, dict) and "_date" in pub_date:
                                pub_date = pub_date["_date"].split("T")[0]
                            
                            all_content.append({
                                "text": text_content,
                                "talk_id": talk_id,
                                "talk_title": safe_get(talk_info, "talk_title", ""),
                                "published_date": pub_date,
                                "content_type": content_type
                            })
        
        if not all_content:
            return {
                "analysis_results": [],
                "time_trends": {},
                "analysis_summary": "No content found matching the criteria",
                "total_items_found": 0,
                "content_stats": content_stats,
                "success": True
            }
        
        # Perform analysis based on analysis_type
        analysis_results = []
        time_trends = {}
        
        if analysis_type == "tools":
            # Analyze software tools and libraries
            tool_patterns = [
                # Popular ML/AI tools
                r'\b(LangChain|LangGraph|LangSmith)\b',
                r'\b(TensorFlow|PyTorch|Keras|scikit-learn|sklearn)\b',
                r'\b(Weights & Biases|wandb|MLflow|Kubeflow)\b',
                r'\b(Docker|Kubernetes|k8s)\b',
                r'\b(Ray|Dask|Apache Spark)\b',
                r'\b(Airflow|Prefect|Dagster)\b',
                r'\b(FastAPI|Flask|Django)\b',
                r'\b(OpenAI|Anthropic|Cohere)\b',
                r'\b(Hugging Face|HuggingFace|transformers)\b',
                r'\b(Triton|TensorRT|ONNX)\b',
                r'\b(Kafka|Redis|PostgreSQL|MongoDB)\b',
                r'\b(Grafana|Prometheus|Jaeger)\b',
                r'\b(AWS|Azure|GCP|Google Cloud)\b',
                r'\b(Jupyter|VSCode|GitHub)\b'
            ]
            
            tool_counts = Counter()
            tool_mentions = defaultdict(list)
            
            for content in all_content:
                text = content["text"].lower()
                for pattern in tool_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        tool_name = match.strip()
                        if tool_name:
                            tool_counts[tool_name] += 1
                            tool_mentions[tool_name].append({
                                "talk_title": content["talk_title"],
                                "published_date": content["published_date"]
                            })
            
            # Filter by minimum mentions and get top results
            filtered_tools = {tool: count for tool, count in tool_counts.items() if count >= min_mentions}
            top_tools = Counter(filtered_tools).most_common(top_n)
            
            analysis_results = []
            for tool, count in top_tools:
                analysis_results.append({
                    "item": tool,
                    "count": count,
                    "percentage": round((count / len(all_content)) * 100, 1),
                    "sample_mentions": tool_mentions[tool][:3]  # Show first 3 mentions
                })
        
        elif analysis_type == "technologies":
            # Analyze technology mentions
            tech_patterns = [
                r'\b(Machine Learning|ML|Deep Learning|DL|AI|Artificial Intelligence)\b',
                r'\b(Large Language Model|LLM|GPT|BERT|Transformer)\b',
                r'\b(Vector Database|Vector Search|Embedding|RAG)\b',
                r'\b(Microservices|API|REST|GraphQL)\b',
                r'\b(Cloud|Edge Computing|Serverless)\b',
                r'\b(DevOps|MLOps|DataOps|CI/CD)\b',
                r'\b(Monitoring|Observability|Logging)\b',
                r'\b(Real-time|Streaming|Batch Processing)\b',
                r'\b(Computer Vision|NLP|Natural Language Processing)\b',
                r'\b(Reinforcement Learning|Supervised Learning|Unsupervised Learning)\b'
            ]
            
            tech_counts = Counter()
            for content in all_content:
                text = content["text"]
                for pattern in tech_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        tech_name = match.strip()
                        if tech_name:
                            tech_counts[tech_name] += 1
            
            filtered_techs = {tech: count for tech, count in tech_counts.items() if count >= min_mentions}
            top_techs = Counter(filtered_techs).most_common(top_n)
            
            analysis_results = []
            for tech, count in top_techs:
                analysis_results.append({
                    "item": tech,
                    "count": count,
                    "percentage": round((count / len(all_content)) * 100, 1)
                })
        
        elif analysis_type == "keywords":
            # Analyze frequent keywords/phrases
            all_text = " ".join(content["text"].lower() for content in all_content)
            
            # Remove common stop words and extract meaningful terms
            stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about", "into", "through", "during", "before", "after", "above", "below", "between", "among", "throughout", "despite", "towards", "upon", "concerning", "a", "an", "as", "are", "was", "were", "been", "be", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "them", "their", "there", "where", "when", "why", "how", "what", "which", "who", "whose", "whom", "if", "unless", "until", "while", "although", "though", "since", "because", "so", "but", "yet", "nor", "either", "neither", "both", "all", "any", "each", "every", "some", "many", "much", "more", "most", "other", "another", "such", "only", "own", "same", "few", "little", "large", "small", "next", "last", "first", "second", "new", "old", "good", "bad", "best", "better", "worse", "worst", "high", "higher", "highest", "low", "lower", "lowest", "big", "bigger", "biggest", "small", "smaller", "smallest", "long", "longer", "longest", "short", "shorter", "shortest", "early", "earlier", "earliest", "late", "later", "latest", "young", "younger", "youngest", "old", "older", "oldest", "important", "more", "most", "less", "least", "very", "quite", "just", "still", "also", "too", "not", "no", "yes", "maybe", "perhaps", "probably", "definitely", "certainly", "surely", "indeed", "really", "truly", "actually", "exactly", "precisely", "approximately", "roughly", "about", "around", "nearly", "almost", "quite", "rather", "fairly", "pretty", "somewhat", "slightly", "a", "bit", "little", "lot", "much", "many", "several", "few", "couple", "pair", "dozen", "hundred", "thousand", "million", "billion", "trillion"}
            
            # Extract words (2+ characters, alphabetic)
            words = re.findall(r'\b[a-z]{2,}\b', all_text)
            word_counts = Counter(word for word in words if word not in stop_words and len(word) > 2)
            
            # Filter and get top keywords
            filtered_words = {word: count for word, count in word_counts.items() if count >= min_mentions}
            top_keywords = Counter(filtered_words).most_common(top_n)
            
            analysis_results = []
            for keyword, count in top_keywords:
                analysis_results.append({
                    "item": keyword,
                    "count": count,
                    "percentage": round((count / len(words)) * 100, 2)
                })
        
        else:  # topics or trends
            # For topics, we'll use keyword analysis with domain-specific terms
            topic_patterns = [
                r'\b(deployment|production|pipeline|monitoring)\b',
                r'\b(model|training|inference|evaluation)\b',
                r'\b(data|dataset|feature|pipeline)\b',
                r'\b(agent|agents|memory|reasoning)\b',
                r'\b(vector|embedding|retrieval|search)\b',
                r'\b(scaling|performance|optimization)\b',
                r'\b(governance|compliance|security|privacy)\b',
                r'\b(automation|orchestration|workflow)\b'
            ]
            
            topic_counts = Counter()
            for content in all_content:
                text = content["text"].lower()
                for pattern in topic_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        topic_counts[match.strip()] += 1
            
            filtered_topics = {topic: count for topic, count in topic_counts.items() if count >= min_mentions}
            top_topics = Counter(filtered_topics).most_common(top_n)
            
            analysis_results = []
            for topic, count in top_topics:
                analysis_results.append({
                    "item": topic,
                    "count": count,
                    "percentage": round((count / len(all_content)) * 100, 1)
                })
        
        # Time-based trend analysis
        if time_grouping != "none" and analysis_results:
            time_trends = {}
            # This would require more complex implementation based on published dates
            # For now, provide a basic structure
            time_trends = {
                "grouping": time_grouping,
                "note": "Time trend analysis requires additional implementation for date grouping"
            }
        
        # Build analysis summary
        summary_parts = [f"{analysis_type} analysis"]
        if filter_parts:
            summary_parts.append(f"filtered by {', '.join(filter_parts)}")
        summary_parts.append(f"from {', '.join(content_descriptions)}")
        
        analysis_summary = f"Performed {' '.join(summary_parts)}"
        
        return {
            "analysis_results": analysis_results,
            "time_trends": time_trends,
            "analysis_summary": analysis_summary,
            "total_items_found": len(analysis_results),
            "content_stats": content_stats,
            "success": True
        }
        
    except Exception as e:
        return {
            "analysis_results": [],
            "time_trends": {},
            "analysis_summary": "Topic and trend analysis failed",
            "total_items_found": 0,
            "content_stats": {},
            "success": False,
            "error": str(e)
        }