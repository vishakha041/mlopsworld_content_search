import { motion, AnimatePresence } from 'framer-motion';
import { X, ExternalLink, Play, Calendar, Eye, User } from 'lucide-react';
import { StreamEvent } from '@/lib/types';

interface ResultsModalProps {
  isOpen: boolean;
  onClose: () => void;
  steps: StreamEvent[];
}

interface TalkResult {
  title: string;
  speaker: string;
  youtube_url?: string;
  views?: number;
  published_date?: string;
  similarity_score?: number;
  category?: string;
}

export function ResultsModal({ isOpen, onClose, steps }: ResultsModalProps) {
  // Extract results from the last tool step that has results
  const results = extractResults(steps);

  if (!results || results.length === 0) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed inset-0 flex items-center justify-center z-50 p-4 pointer-events-none"
          >
            <div className="bg-zinc-900 border border-white/10 rounded-2xl w-full max-w-4xl max-h-[80vh] flex flex-col shadow-2xl pointer-events-auto">
              {/* Header */}
              <div className="flex items-center justify-between p-6 border-b border-white/5">
                <div>
                  <h2 className="text-xl font-semibold text-white">Retrieved Content</h2>
                  <p className="text-sm text-zinc-400">Found {results.length} relevant talks</p>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-white/5 rounded-full transition-colors text-zinc-400 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Content Grid */}
              <div className="flex-1 overflow-y-auto p-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                {results.map((item, idx) => {
                  const hasLink = !!item.youtube_url;
                  const cardClassName = "group relative flex flex-col bg-white/5 hover:bg-white/10 border border-white/5 hover:border-purple-500/30 rounded-xl p-4 transition-all duration-200";
                  
                  const cardContent = (
                    <>
                      <div className="flex items-start justify-between gap-4 mb-3">
                        <h3 className="font-medium text-zinc-100 line-clamp-2 group-hover:text-purple-300 transition-colors">
                          {item.title || 'Untitled Talk'}
                        </h3>
                        {hasLink && (
                          <ExternalLink className="w-4 h-4 text-zinc-500 group-hover:text-purple-400 flex-shrink-0" />
                        )}
                      </div>

                      <div className="mt-auto space-y-3">
                        <div className="flex items-center gap-2 text-sm text-zinc-300">
                          <User className="w-4 h-4 text-zinc-500" />
                          <span>{item.speaker || 'Unknown Speaker'}</span>
                        </div>

                        <div className="flex items-center gap-4 text-xs text-zinc-500">
                          {item.views !== undefined && item.views !== null && (
                            <div className="flex items-center gap-1.5">
                              <Eye className="w-3 h-3" />
                              <span>{item.views.toLocaleString()}</span>
                            </div>
                          )}
                          {item.published_date && (
                            <div className="flex items-center gap-1.5">
                              <Calendar className="w-3 h-3" />
                              <span>{item.published_date}</span>
                            </div>
                          )}
                          {item.similarity_score !== undefined && item.similarity_score !== null && (
                            <div className="ml-auto px-2 py-0.5 bg-purple-500/10 text-purple-300 rounded-full">
                              {Math.round(item.similarity_score * 100)}% match
                            </div>
                          )}
                        </div>
                      </div>
                    </>
                  );
                  
                  return hasLink ? (
                    <a
                      key={idx}
                      href={item.youtube_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={cardClassName}
                    >
                      {cardContent}
                    </a>
                  ) : (
                    <div key={idx} className={cardClassName}>
                      {cardContent}
                    </div>
                  );
                })}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

function extractResults(steps: StreamEvent[]): TalkResult[] | null {
  // Look for the last tool step that has results
  // We iterate backwards to find the most recent relevant tool call
  for (let i = steps.length - 1; i >= 0; i--) {
    const step = steps[i];
    if (step.event_type !== 'step') continue;

    // Check content for JSON-like structure containing results
    try {
      if (step.data.content) {
        const content = step.data.content;
        
        // Heuristic: check if it looks like a tool output with results
        if (content.includes('"results":') || content.includes("'results':")) {
          let parsed;
          
          // Try to parse the content as JSON
          try {
            parsed = JSON.parse(content);
          } catch (firstError) {
            // If direct parsing fails, the content might be a Python string representation
            // Try to extract JSON from it or handle edge cases
            console.warn('Direct JSON parse failed, attempting cleanup:', firstError);
            
            // Try to find JSON object within the string
            const jsonMatch = content.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
              try {
                parsed = JSON.parse(jsonMatch[0]);
              } catch (secondError) {
                console.error('JSON extraction failed:', secondError);
                continue;
              }
            } else {
              continue;
            }
          }
          
          console.log('Parsed tool output:', parsed);
          
          // Check for results array
          if (parsed.results && Array.isArray(parsed.results)) {
            // Map semantic search results to TalkResult format
            const mapped = parsed.results.map((result: any) => {
              // Format published date - extract just YYYY-MM-DD from ISO datetime
              let formattedDate = result.published_date || result.yt_published_at;
              if (formattedDate && typeof formattedDate === 'string') {
                formattedDate = formattedDate.split('T')[0]; // Get just the date part
              }
              
              return {
                title: result.title || result.talk_title || 'Untitled',
                speaker: result.speaker || result.speaker_name || 'Unknown Speaker',
                youtube_url: result.youtube_url,
                views: result.views || result.yt_views,
                published_date: formattedDate,
                similarity_score: result.similarity_score,
                category: result.category || result.category_primary
              };
            });
            
            console.log('Mapped results:', mapped);
            return mapped;
          }
          
          if (parsed.similar_talks && Array.isArray(parsed.similar_talks)) {
            return parsed.similar_talks.map((result: any) => ({
              title: result.title || result.talk_title || 'Untitled',
              speaker: result.speaker || result.speaker_name || 'Unknown Speaker',
              youtube_url: result.youtube_url,
              views: result.views || result.yt_views,
              published_date: result.published_date || result.yt_published_at,
              similarity_score: result.similarity_score,
              category: result.category || result.category_primary
            }));
          }

          if (parsed.talk_info) {
            return [{
              title: parsed.talk_info.title || parsed.talk_info.talk_title || 'Untitled',
              speaker: parsed.talk_info.speaker || parsed.talk_info.speaker_name || 'Unknown Speaker',
              youtube_url: parsed.talk_info.youtube_url,
              views: parsed.talk_info.views || parsed.talk_info.yt_views,
              published_date: parsed.talk_info.published_date || parsed.talk_info.yt_published_at,
              similarity_score: parsed.talk_info.similarity_score,
              category: parsed.talk_info.category || parsed.talk_info.category_primary
            }];
          }
        }
      }
    } catch (e) {
      // Parsing failed, continue searching
      console.error('Error processing step:', e);
      continue;
    }
  }
  return null;
}
