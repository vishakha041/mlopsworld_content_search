import { motion, AnimatePresence } from 'framer-motion';
import { X, ExternalLink, Play, Calendar, Eye, User, Tag, Presentation } from 'lucide-react';
import { StreamEvent } from '@/lib/types';

interface ResultsSidebarProps {
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
  event_name?: string;
}

export function ResultsSidebar({ isOpen, onClose, steps }: ResultsSidebarProps) {
  const results = extractResults(steps);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ x: '100%' }}
          animate={{ x: 0 }}
          exit={{ x: '100%' }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className="fixed top-0 right-0 h-full w-96 bg-zinc-900/95 border-l border-white/10 shadow-2xl z-40 flex flex-col backdrop-blur-xl"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-white/5">
            <div>
              <h2 className="text-lg font-semibold text-white">Retrieved Content</h2>
              <p className="text-xs text-zinc-400">
                {results && results.length > 0 ? `${results.length} relevant talks` : 'No content'}
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/5 rounded-full transition-colors text-zinc-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {(!results || results.length === 0) ? (
              <div className="flex flex-col items-center justify-center h-full text-zinc-500">
                <p className="text-sm">No retrieved content available</p>
              </div>
            ) : (
              results.map((item, idx) => {
                const hasLink = !!item.youtube_url;
                const thumbnailUrl = item.youtube_url ? getYouTubeThumbnail(item.youtube_url) : null;
                const cardClassName = "group relative flex flex-col bg-white/5 hover:bg-white/10 border border-white/5 hover:border-purple-500/30 rounded-xl p-3 transition-all duration-200";
                
                const cardContent = (
                  <>
                    {thumbnailUrl && (
                      <div className="relative w-full aspect-video mb-3 rounded-lg overflow-hidden bg-zinc-800">
                        <img 
                          src={thumbnailUrl} 
                          alt={item.title}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                        />
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors" />
                        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                          <div className="w-8 h-8 rounded-full bg-purple-500/90 flex items-center justify-center backdrop-blur-sm shadow-lg transform scale-75 group-hover:scale-100 transition-transform">
                            <Play className="w-3 h-3 text-white fill-current ml-0.5" />
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="flex items-start justify-between gap-2 mb-2">
                      <h3 className="font-medium text-sm text-zinc-100 line-clamp-2 group-hover:text-purple-300 transition-colors">
                        {item.title || 'Untitled Talk'}
                      </h3>
                      {hasLink && (
                        <ExternalLink className="w-3 h-3 text-zinc-500 group-hover:text-purple-400 flex-shrink-0 mt-1" />
                      )}
                    </div>

                    <div className="mt-auto space-y-2">
                      <div className="flex items-center gap-2 text-xs text-zinc-300">
                        <User className="w-3 h-3 text-zinc-500" />
                        <span className="truncate">{item.speaker || 'Unknown Speaker'}</span>
                      </div>

                      {(item.event_name || item.category) && (
                        <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
                          {item.event_name && (
                            <div className="flex items-center gap-1">
                              <Presentation className="w-3 h-3" />
                              <span className="truncate max-w-[120px]">{item.event_name}</span>
                            </div>
                          )}
                          {item.category && (
                            <div className="flex items-center gap-1">
                              <Tag className="w-3 h-3" />
                              <span>{item.category}</span>
                            </div>
                          )}
                        </div>
                      )}

                      <div className="flex items-center gap-3 text-xs text-zinc-500">
                        {item.views !== undefined && item.views !== null && (
                          <div className="flex items-center gap-1">
                            <Eye className="w-3 h-3" />
                            <span>{item.views.toLocaleString()}</span>
                          </div>
                        )}
                        {item.published_date && (
                          <div className="flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            <span>{item.published_date}</span>
                          </div>
                        )}
                        {item.similarity_score !== undefined && item.similarity_score !== null && (
                          <div className="ml-auto px-2 py-0.5 bg-purple-500/10 text-purple-300 rounded-full text-xs">
                            {Math.round(item.similarity_score * 100)}%
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
              })
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function extractResults(steps: StreamEvent[]): TalkResult[] | null {
  // Look for the last tool step that has results
  for (let i = steps.length - 1; i >= 0; i--) {
    const step = steps[i];
    if (step.event_type !== 'step') continue;

    try {
      if (step.data.content) {
        const content = step.data.content;
        
        if (content.includes('"results":') || content.includes("'results':")) {
          let parsed;
          
          try {
            parsed = JSON.parse(content);
          } catch (firstError) {
            const jsonMatch = content.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
              try {
                parsed = JSON.parse(jsonMatch[0]);
              } catch (secondError) {
                continue;
              }
            } else {
              continue;
            }
          }
          
          if (parsed.results && Array.isArray(parsed.results)) {
            const mapped = parsed.results.map((result: any) => {
              let formattedDate = result.published_date || result.yt_published_at;
              if (formattedDate && typeof formattedDate === 'string') {
                formattedDate = formattedDate.split('T')[0];
              }
              
              return {
                title: result.title || result.talk_title || 'Untitled',
                speaker: result.speaker || result.speaker_name || 'Unknown Speaker',
                youtube_url: result.youtube_url,
                views: result.views || result.yt_views,
                published_date: formattedDate,
                similarity_score: result.similarity_score,
                category: result.category || result.category_primary,
                event_name: result.event || result.event_name
              };
            });
            
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
              category: result.category || result.category_primary,
              event_name: result.event || result.event_name
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
              category: parsed.talk_info.category || parsed.talk_info.category_primary,
              event_name: parsed.talk_info.event || parsed.talk_info.event_name
            }];
          }
        }
      }
    } catch (e) {
      continue;
    }
  }
  return null;
}

function getYouTubeThumbnail(url: string): string | null {
  if (!url) return null;
  const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
  const match = url.match(regExp);
  return (match && match[2].length === 11)
    ? `https://img.youtube.com/vi/${match[2]}/mqdefault.jpg`
    : null;
}

// Export helper to check if steps have results
export function hasRetrievedResults(steps: StreamEvent[]): boolean {
  return extractResults(steps) !== null && extractResults(steps)!.length > 0;
}
