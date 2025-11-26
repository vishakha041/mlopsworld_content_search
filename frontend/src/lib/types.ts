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

export interface VideoSearchResult {
  talk_title: string;
  speaker_name: string;
  company_name?: string;
  job_title?: string;
  category_primary?: string;
  event_name?: string;
  track?: string;
  abstract?: string;
  youtube_url?: string;
  youtube_id?: string;
  yt_views?: number;
  published_date?: string;
  tech_level?: number;
  keywords?: string;
  distance?: number;
  similarity_score: number;
  video_blob?: string; // Base64 string or similar if returned
}

export interface VideoSearchResponse {
  success: boolean;
  results: VideoSearchResult[];
  total_found: number;
  search_summary: string;
}
