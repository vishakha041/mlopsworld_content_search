🏥 Health & Status Endpoints
# Test 1: Basic health check (no auth required)
curl -X GET 'http://0.0.0.0:8000/health'

# Test 2: Detailed status (no auth required)
curl -X GET 'http://0.0.0.0:8000/api/v1/status'
🤖 Agent Query Endpoint
# Test 1: Simple query
curl -X POST 'http://0.0.0.0:8000/api/v1/query' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{"query": "Show me the most popular talks from 2024"}'

# Test 2: Technical query
curl -X POST 'http://0.0.0.0:8000/api/v1/query' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{"query": "Which talks discuss AI agents with memory?"}'

# Test 3: Speaker analysis query
curl -X POST 'http://0.0.0.0:8000/api/v1/query' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{"query": "Who are the top 10 most active speakers?"}'

# Test 4: Trend analysis query
curl -X POST 'http://0.0.0.0:8000/api/v1/query' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{"query": "What tools are trending in MLOps?"}'


🔍 Talk Search Endpoint
# Test 1: Basic semantic search
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/search' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "machine learning deployment",
    "search_type": "all",
    "k_neighbors": 10
  }'

# Test 2: Search transcripts only
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/search' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "vector databases and embeddings",
    "search_type": "transcript",
    "k_neighbors": 5
  }'

# Test 3: Search with date filter
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/search' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "AI agents",
    "search_type": "all",
    "date_from": "2024-01-01",
    "k_neighbors": 10
  }'

# Test 4: Search with category filter
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/search' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "data quality",
    "search_type": "all",
    "category": "MLOps",
    "k_neighbors": 10
  }'

# Test 5: Search with similarity threshold
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/search' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "machine learning",
    "search_type": "all",
    "k_neighbors": 20,
    "score_threshold": 0.5
  }'

🎛️ Talk Filter Endpoint
# Test 1: Filter by date range
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/filter' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "date_from": "2024-01-01",
    "date_to": "2024-12-31",
    "sort_by": "date_desc",
    "limit": 10
  }'

# Test 2: Filter by minimum views
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/filter' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "min_views": 1000,
    "sort_by": "views_desc",
    "limit": 20
  }'

# Test 3: Filter by category
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/filter' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "category": "MLOps",
    "sort_by": "views_desc",
    "limit": 15
  }'

# Test 4: Filter by company
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/filter' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "company_name": "Google",
    "sort_by": "date_desc",
    "limit": 10
  }'

# Test 5: Complex multi-filter
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/filter' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "date_from": "2024-01-01",
    "min_views": 500,
    "category": "Deployment and integration",
    "sort_by": "views_desc",
    "limit": 10
  }'

📄 Talk Details Endpoint
# Test 1: Get details by title (without transcript)
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/details' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "talk_title": "MemeGPT: Creating a Large Language Model to Generate Memes",
    "include_transcript": false,
    "include_related": false
  }'

# Test 2: Get details with transcript
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/details' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "talk_title": "MemeGPT: Creating a Large Language Model to Generate Memes",
    "include_transcript": true,
    "include_related": false
  }'

# Test 3: Get details with related talks
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/details' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "talk_title": "MemeGPT: Creating a Large Language Model to Generate Memes",
    "include_transcript": false,
    "include_related": true,
    "related_count": 5
  }'

# Test 4: Get details with transcript time filter
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/details' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "talk_title": "Multimodal LLMs for product taxonomy at Shopify",
    "include_transcript": true,
    "time_start": 300,
    "time_end": 900
  }'

🔗 Similar Content Endpoint
# Test 1: Find similar by talk title
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/similar' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "reference_talk_title": "Multimodal LLMs for product taxonomy at Shopify",
    "similarity_type": "content",
    "k_neighbors": 10
  }'

# Test 2: Find similar by query
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/similar' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "reference_query": "deploying machine learning models at scale",
    "similarity_type": "content",
    "k_neighbors": 10
  }'

# Test 3: Find similar excluding same speaker
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/similar' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "reference_talk_title": "Serving GenAI Workload At Scale With LitServe",
    "similarity_type": "content",
    "exclude_same_speaker": true,
    "k_neighbors": 10
  }'

# Test 4: Find similar with filters
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/similar' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "reference_query": "machine learning",
    "similarity_type": "content",
    "date_from": "2024-01-01",
    "k_neighbors": 10,
    "min_similarity": 0.5
  }'


👥 Speaker Analysis Endpoint
# Test 1: Analyze all speakers
curl -X POST 'http://0.0.0.0:8000/api/v1/speakers/analyze' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "analysis_type": "all",
    "top_n": 10
  }'

# Test 2: Get top speakers by talk count
curl -X POST 'http://0.0.0.0:8000/api/v1/speakers/analyze' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "analysis_type": "talk_count",
    "min_talks": 2,
    "top_n": 15
  }'

# Test 3: Analyze specific speaker
curl -X POST 'http://0.0.0.0:8000/api/v1/speakers/analyze' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "speaker_name": "Marcelo Litovsky",
    "analysis_type": "all"
  }'

# Test 4: Company breakdown
curl -X POST 'http://0.0.0.0:8000/api/v1/speakers/analyze' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "analysis_type": "companies",
    "date_from": "2024-01-01"
  }'

# Test 5: Analyze with category filter
curl -X POST 'http://0.0.0.0:8000/api/v1/speakers/analyze' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "analysis_type": "all",
    "category": "MLOps",
    "top_n": 10
  }'
🎥 Video Search Endpoint
# Test 1: Basic video search
curl -X POST 'http://0.0.0.0:8000/api/v1/videos/search' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "AI agents demonstration",
    "top_n": 5,
    "include_videos": false
  }'

# Test 2: Video search with more results
curl -X POST 'http://0.0.0.0:8000/api/v1/videos/search' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "live coding session",
    "top_n": 10,
    "include_videos": false
  }'

# Test 3: Video search for visual content
curl -X POST 'http://0.0.0.0:8000/api/v1/videos/search' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "architecture diagrams and system design",
    "top_n": 8,
    "include_videos": false
  }'

📊 Trend Analysis Endpoint
# Test 1: Analyze tools
curl -X POST 'http://0.0.0.0:8000/api/v1/trends/analyze' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "analysis_type": "tools",
    "content_source": "all",
    "top_n": 20,
    "min_mentions": 2
  }'

# Test 2: Analyze topics
curl -X POST 'http://0.0.0.0:8000/api/v1/trends/analyze' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "analysis_type": "topics",
    "content_source": "transcripts",
    "top_n": 15,
    "min_mentions": 3
  }'

# Test 3: Analyze technologies
curl -X POST 'http://0.0.0.0:8000/api/v1/trends/analyze' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "analysis_type": "technologies",
    "content_source": "all",
    "top_n": 20,
    "min_mentions": 2
  }'

# Test 4: Analyze with time grouping
curl -X POST 'http://0.0.0.0:8000/api/v1/trends/analyze' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "analysis_type": "tools",
    "content_source": "all",
    "time_grouping": "monthly",
    "top_n": 15,
    "min_mentions": 2
  }'

# Test 5: Analyze with filters
curl -X POST 'http://0.0.0.0:8000/api/v1/trends/analyze' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "analysis_type": "keywords",
    "content_source": "abstracts",
    "date_from": "2024-01-01",
    "category": "Deployment and integration",
    "top_n": 30,
    "min_mentions": 5
  }'

🏷️ Unique Values Endpoint
# Test 1: Get all unique values
curl -X POST 'http://0.0.0.0:8000/api/v1/trends/unique-values' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "event_name": true,
    "category_primary": true,
    "track": true,
    "company_name": true,
    "tech_level": true,
    "industries": true
  }'

# Test 2: Get specific values
curl -X POST 'http://0.0.0.0:8000/api/v1/trends/unique-values' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "event_name": true,
    "category_primary": true
  }'

# Test 3: Get company list
curl -X POST 'http://0.0.0.0:8000/api/v1/trends/unique-values' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "company_name": true
  }'

🧪 Error Testing
# Test 1: Missing API key
curl -X POST 'http://0.0.0.0:8000/api/v1/query' \
  -H 'Content-Type: application/json' \
  -d '{"query": "test"}'

# Test 2: Invalid API key
curl -X POST 'http://0.0.0.0:8000/api/v1/query' \
  -H 'X-API-Key: wrong-key' \
  -H 'Content-Type: application/json' \
  -d '{"query": "test"}'

# Test 3: Missing required field
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/search' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "search_type": "all"
  }'

# Test 4: Invalid parameter value
curl -X POST 'http://0.0.0.0:8000/api/v1/talks/search' \
  -H 'X-API-Key: secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "test",
    "search_type": "invalid_type"
  }'