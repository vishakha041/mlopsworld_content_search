import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, Circle, Loader2, Terminal, Brain, Database, ChevronDown, ChevronRight, Sparkles } from 'lucide-react';
import { clsx } from 'clsx';
import { StreamEvent } from '@/lib/types';

interface StepIndicatorProps {
  steps: StreamEvent[];
  isComplete: boolean;
}

export function StepIndicator({ steps, isComplete }: StepIndicatorProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  // Auto-collapse when complete
  useEffect(() => {
    if (isComplete) {
      setIsExpanded(false);
    }
  }, [isComplete]);

  // Filter relevant steps
  const relevantSteps = steps.filter(s => 
    s.event_type === 'step' && 
    ['AIMessage', 'ToolMessage'].includes(s.data.message_type)
  );

  if (relevantSteps.length === 0) return null;

  return (
    <div className="w-full max-w-2xl my-4">
      <button 
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-xs font-medium text-zinc-500 hover:text-zinc-300 transition-colors mb-2"
      >
        {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <span>
          {isComplete ? 'Thought Process' : 'Agent is thinking...'}
        </span>
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="flex flex-col gap-4 pl-2 border-l border-white/10 ml-1.5 py-2">
              {relevantSteps.map((step, idx) => (
                <StepItem key={idx} step={step} isLast={idx === relevantSteps.length - 1} isComplete={isComplete} />
              ))}
              
              {!isComplete && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex items-center gap-3 text-sm text-zinc-400"
                >
                  <div className="relative flex items-center justify-center w-6 h-6">
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-purple-400" />
                  </div>
                  <span className="text-xs text-zinc-500 animate-pulse">Processing...</span>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function StepItem({ step, isLast, isComplete }: { step: StreamEvent; isLast: boolean; isComplete: boolean }) {
  const { message_type, tool_calls, content } = step.data;

  // 1. Tool Call (AIMessage with tool_calls)
  if (message_type === 'AIMessage' && tool_calls?.length > 0) {
    return (
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        className="flex items-start gap-3 text-sm"
      >
        <div className="relative flex items-center justify-center w-6 h-6 mt-0.5 bg-blue-500/10 rounded-md border border-blue-500/20">
          <Terminal className="w-3.5 h-3.5 text-blue-400" />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-blue-300">Decided to call tool</span>
          <div className="flex flex-col gap-1">
            {tool_calls.map((tool: any, idx: number) => (
              <code key={idx} className="text-[10px] bg-black/30 px-2 py-1 rounded text-zinc-400 font-mono border border-white/5 w-fit">
                {tool.name}({Object.keys(tool.args).length > 0 ? '...' : ''})
              </code>
            ))}
          </div>
        </div>
      </motion.div>
    );
  }

  // 2. Tool Output (ToolMessage)
  if (message_type === 'ToolMessage') {
    let resultCount = 0;
    let summary = "Tool execution complete";
    try {
      const parsed = JSON.parse(content);
      if (parsed.results) resultCount = parsed.results.length;
      if (parsed.similar_talks) resultCount = parsed.similar_talks.length;
      if (parsed.total_found) summary = `Found ${parsed.total_found} results`;
      else if (parsed.success) summary = "Execution successful";
    } catch (e) {
      // content might not be JSON
    }

    return (
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        className="flex items-start gap-3 text-sm"
      >
        <div className="relative flex items-center justify-center w-6 h-6 mt-0.5 bg-green-500/10 rounded-md border border-green-500/20">
          <Database className="w-3.5 h-3.5 text-green-400" />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-green-300">Tool output received</span>
          <span className="text-xs text-zinc-500">{summary}</span>
        </div>
      </motion.div>
    );
  }

  // 3. Final Answer Generation (AIMessage with content)
  if (message_type === 'AIMessage' && content) {
    return (
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        className="flex items-start gap-3 text-sm"
      >
        <div className="relative flex items-center justify-center w-6 h-6 mt-0.5 bg-purple-500/10 rounded-md border border-purple-500/20">
          <Brain className="w-3.5 h-3.5 text-purple-400" />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-purple-300">Analyzing & Generating Answer</span>
          <span className="text-xs text-zinc-500">Synthesizing information...</span>
        </div>
      </motion.div>
    );
  }

  return null;
}
