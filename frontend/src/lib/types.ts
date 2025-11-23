export interface AgentQueryRequest {
  query: string;
  stream?: boolean;
}

export interface AgentStep {
  type: string;
  content: string;
  tool_name?: string;
  timestamp?: string;
  tool_calls?: Array<{
    name: string;
    args: Record<string, any>;
    id: string;
  }>;
}

export interface StreamEvent {
  event_type: 'step' | 'answer' | 'error';
  data: any;
  timestamp: string;
}

export interface StepData {
  step_number: number;
  message_type: string;
  content?: string;
  tool_calls?: Array<{
    name: string;
    args: Record<string, any>;
    id: string;
  }>;
}

export interface AnswerData {
  answer: string;
  total_steps: number;
  query: string;
}

export interface VideoSearchRequest {
  query: string;
  top_n?: number;
  include_videos?: boolean;
}

export interface VideoMetadata {
  fps?: number;
  duration_sec?: number;
  resolution?: string;
}

export interface VideoSearchResult {
  talk_title: string;
  speaker_name: string;
  company_name?: string;
  category_primary?: string;
  youtube_url?: string;
  youtube_id?: string;
  yt_views?: number;
  distance?: number;
  similarity_score: number;
  metadata?: VideoMetadata;
  video_blob?: string; // Base64 string or similar if returned
}

export interface VideoSearchResponse {
  success: boolean;
  results: VideoSearchResult[];
  total_found: number;
  search_summary: string;
}
