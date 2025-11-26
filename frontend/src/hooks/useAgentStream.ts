import { useState, useCallback, useRef } from 'react';
import { API_ENDPOINTS } from '../lib/api';
import { StreamEvent, StepData, AnswerData } from '../lib/types';

interface UseAgentStreamReturn {
  messages: StreamEvent[];
  isLoading: boolean;
  error: string | null;
  streamAgent: (query: string) => Promise<void>;
  reset: () => void;
}

export function useAgentStream(): UseAgentStreamReturn {
  const [messages, setMessages] = useState<StreamEvent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setMessages([]);
    setError(null);
    setIsLoading(false);
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  const streamAgent = useCallback(async (query: string) => {
    reset();
    setIsLoading(true);
    
    abortControllerRef.current = new AbortController();
    const API_KEY = process.env.NEXT_PUBLIC_API_KEY || 'change-in-production';

    try {
      const response = await fetch(API_ENDPOINTS.STREAM, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY,
        },
        body: JSON.stringify({ query, stream: true }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error('Response body is null');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // Split by double newline which separates SSE events
        const parts = buffer.split('\n\n');
        
        // Keep the last part in the buffer if it's incomplete
        buffer = parts.pop() || '';

        for (const part of parts) {
          if (part.trim().startsWith('data: ')) {
            const jsonStr = part.replace('data: ', '').trim();
            if (!jsonStr) continue;

            try {
              const event: StreamEvent = JSON.parse(jsonStr);
              
              if (event.event_type === 'error') {
                throw new Error(event.data.error || 'Unknown error from agent');
              }

              setMessages((prev) => [...prev, event]);
              
              if (event.event_type === 'answer') {
                setIsLoading(false);
              }
            } catch (e) {
              console.error('Error parsing stream event:', e);
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('Stream aborted');
      } else {
        setError(err.message || 'An error occurred while streaming');
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [reset]);

  return { messages, isLoading, error, streamAgent, reset };
}
