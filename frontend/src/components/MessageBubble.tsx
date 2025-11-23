import { useState } from 'react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { User, Bot, LayoutGrid, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { StreamEvent } from '@/lib/types';
import { StepIndicator } from './StepIndicator';
import { ResultsModal } from './ResultsModal';

interface MessageBubbleProps {
  role: 'user' | 'agent';
  content?: string;
  steps?: StreamEvent[];
  isStreaming?: boolean;
}

export function MessageBubble({ role, content, steps = [], isStreaming = false }: MessageBubbleProps) {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const isUser = role === 'user';

  // Check if we have results to show
  const hasResults = !isStreaming && !isUser && steps.some(step => {
    try {
      if (step.event_type === 'step' && step.data.content) {
        const parsed = JSON.parse(step.data.content);
        return (parsed.results && Array.isArray(parsed.results) && parsed.results.length > 0) ||
               (parsed.similar_talks && Array.isArray(parsed.similar_talks) && parsed.similar_talks.length > 0) ||
               (parsed.talk_info);
      }
    } catch { return false; }
    return false;
  });

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={clsx(
          "flex w-full gap-4 mb-8",
          isUser ? "flex-row-reverse" : "flex-row"
        )}
      >
        {/* Avatar */}
        <div className={clsx(
          "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-lg",
          isUser ? "bg-white text-black" : "bg-gradient-to-br from-purple-600 to-blue-600 text-white"
        )}>
          {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
        </div>

        {/* Content */}
        <div className={clsx(
          "flex flex-col max-w-[80%]",
          isUser ? "items-end" : "items-start"
        )}>
          {/* Message Text */}
          {(content || isStreaming) && (
            <div className={clsx(
              "px-6 py-4 rounded-2xl shadow-md backdrop-blur-sm",
              isUser 
                ? "bg-white/10 text-white rounded-tr-none" 
                : "bg-zinc-900/50 border border-white/5 text-zinc-100 rounded-tl-none"
            )}>
              {content ? (
                <div className="prose prose-invert prose-sm max-w-none">
                  <ReactMarkdown>{content}</ReactMarkdown>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-zinc-400 text-sm">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Thinking...</span>
                </div>
              )}

              {/* Results Button */}
              {hasResults && (
                <motion.button
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  onClick={() => setIsModalOpen(true)}
                  className="mt-4 flex items-center gap-2 px-4 py-2 bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/20 rounded-lg text-sm text-purple-300 transition-colors"
                >
                  <LayoutGrid className="w-4 h-4" />
                  <span>View Retrieved Content</span>
                </motion.button>
              )}
            </div>
          )}

          {/* Agent Steps (only for agent) */}
          {!isUser && isStreaming && (
            <StepIndicator steps={steps} isComplete={false} />
          )}
        </div>
      </motion.div>

      {/* Results Modal */}
      <ResultsModal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)} 
        steps={steps} 
      />
    </>
  );
}
