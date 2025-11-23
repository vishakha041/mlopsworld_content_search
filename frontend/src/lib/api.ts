import { AgentQueryRequest, VideoSearchRequest, VideoSearchResponse } from './types';

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || 'change-in-production';

const headers = {
  'Content-Type': 'application/json',
  'X-API-Key': API_KEY,
};

export async function searchVideos(request: VideoSearchRequest): Promise<VideoSearchResponse> {
  const response = await fetch(`${API_BASE_URL}/videos/search`, {
    method: 'POST',
    headers,
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(error.detail || 'Failed to search videos');
  }

  return response.json();
}

export async function queryAgent(request: AgentQueryRequest) {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers,
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Network error' }));
    throw new Error(error.detail || 'Failed to query agent');
  }

  return response.json();
}

export const API_ENDPOINTS = {
  STREAM: `${API_BASE_URL}/query/stream`,
};
