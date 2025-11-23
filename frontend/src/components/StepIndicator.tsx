import { motion } from 'framer-motion';
import { CheckCircle2, Circle, Loader2, Terminal } from 'lucide-react';
import { clsx } from 'clsx';
import { StreamEvent } from '@/lib/types';

interface StepIndicatorProps {
  steps: StreamEvent[];
  isComplete: boolean;
}

export function StepIndicator({ steps, isComplete }: StepIndicatorProps) {
  // Filter only step events
  const toolSteps = steps.filter(s => s.event_type === 'step' && s.data.tool_calls);

  // If complete, don't show anything unless there were tool calls
  if (isComplete && toolSteps.length === 0) return null;


  if (toolSteps.length === 0) return null;

  // Group consecutive identical tool calls
  const visibleTools: { name: string; count: number; key: string }[] = [];
  
  toolSteps.forEach((step) => {
    const toolCalls = step.data.tool_calls || [];
    toolCalls.forEach((tool: any, toolIdx: number) => {
      const lastTool = visibleTools[visibleTools.length - 1];
      if (lastTool && lastTool.name === tool.name) {
        lastTool.count++;
      } else {
        visibleTools.push({
          name: tool.name,
          count: 1,
          key: `${step.data.step_number}-${toolIdx}`
        });
      }
    });
  });

  return (
    <div className="flex flex-col gap-2 my-4 pl-4 border-l-2 border-white/10">
      {visibleTools.map((tool) => (
        <motion.div
          key={tool.key}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-3 text-sm text-zinc-400"
        >
          <div className="relative flex items-center justify-center w-5 h-5">
            <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-sm" />
            <Terminal className="w-3 h-3 text-blue-400 relative z-10" />
          </div>
          <span className="font-mono text-xs text-blue-300/80">
            Running <span className="text-blue-300">{tool.name}</span>
            {tool.count > 1 && <span className="text-zinc-500 ml-1">(x{tool.count})</span>}...
          </span>
        </motion.div>
      ))}
      
    </div>
  );
}
