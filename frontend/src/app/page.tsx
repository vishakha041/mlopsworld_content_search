'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, ChevronDown, ChevronUp } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAgentStream } from '@/hooks/useAgentStream';
import { MessageBubble } from '@/components/MessageBubble';
import { ExampleQueries } from '@/components/ExampleQueries';
import { ResultsSidebar, hasRetrievedResults } from '@/components/ResultsSidebar';
import { StreamEvent } from '@/lib/types';

interface ChatMessage {
  id: string;
  role: 'user' | 'agent';
  content: string;
  steps?: StreamEvent[];
}

export default function ChatContainer() {
  const [input, setInput] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [showExamples, setShowExamples] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarSteps, setSidebarSteps] = useState<StreamEvent[]>([]);
  const { messages: streamMessages, isLoading, sessionId, streamAgent, reset, newConversation } = useAgentStream();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, streamMessages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setInput('');
    setShowExamples(false); // Collapse examples on submit
    
    // Add user message immediately
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: userQuery
    };
    
    setChatHistory(prev => [...prev, userMsg]);
    
    // Start streaming
    await streamAgent(userQuery);
  };

  // Handle showing results in sidebar for a specific message
  const handleShowResults = (steps: StreamEvent[]) => {
    setSidebarSteps(steps);
    setSidebarOpen(true);
  };

  // Handle stream completion and update history
  useEffect(() => {
    if (!isLoading && streamMessages.length > 0) {
      const answerEvent = streamMessages.find(m => m.event_type === 'answer');
      const errorEvent = streamMessages.find(m => m.event_type === 'error');
      
      if (answerEvent || errorEvent) {
        const agentMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'agent',
          content: answerEvent ? answerEvent.data.answer : `Error: ${errorEvent?.data.error}`,
          steps: streamMessages
        };
        
        setChatHistory(prev => [...prev, agentMsg]);
        
        // Auto-open sidebar if there are retrieved results
        if (hasRetrievedResults(streamMessages)) {
          setSidebarSteps(streamMessages);
          setSidebarOpen(true);
        }
        
        reset(); // Clear stream state for next interaction
      }
    }
  }, [isLoading, streamMessages, reset]);

  // Current streaming message (if active)
  const currentStreamSteps = streamMessages.filter(m => m.event_type === 'step');
  const currentAnswer = streamMessages.find(m => m.event_type === 'answer')?.data.answer;

  return (
    <>
      <div className={`flex flex-col h-screen mx-auto px-4 pt-8 pb-4 transition-all duration-300 ${sidebarOpen ? 'mr-96' : ''}`} style={{ maxWidth: sidebarOpen ? 'calc(100% - 24rem)' : '64rem' }}>
        {/* Header */}
        <div className="flex items-center justify-between mb-8 px-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-white/5 rounded-lg border border-white/10">
              <Sparkles className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-white">MLOps Content Agent</h1>
              <p className="text-sm text-zinc-400">Ask about talks, speakers, and trends</p>
            </div>
          </div>
          {chatHistory.length > 0 && (
            <button
              onClick={() => {
                newConversation();
                setChatHistory([]);
                setShowExamples(true);
              }}
              className="px-4 py-2 text-sm bg-white/5 hover:bg-white/10 text-zinc-300 hover:text-white rounded-lg border border-white/10 transition-colors"
            >
              New Conversation
            </button>
          )}
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-4 pb-4">
          {chatHistory.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-zinc-500 space-y-4 opacity-50">
              <Sparkles className="w-12 h-12" />
              <p>Start a conversation to explore the content</p>
            </div>
          ) : (
            <div className="space-y-6">
              {chatHistory.map((msg) => (
                <MessageBubble 
                  key={msg.id} 
                  role={msg.role} 
                  content={msg.content} 
                  steps={msg.steps}
                  onShowResults={handleShowResults}
                />
              ))}

              {/* Active Streaming Message */}
              {isLoading && (
                <MessageBubble 
                  role="agent" 
                  content={currentAnswer} // Will be undefined until done, which is fine
                  steps={currentStreamSteps}
                  isStreaming={true}
                />
              )}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

      {/* Input Area */}
      <div className="mt-4 px-4">
        <div className="flex justify-between items-center mb-2">
          <AnimatePresence>
            {!showExamples && (
              <motion.button
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setShowExamples(true)}
                className="text-xs text-zinc-500 hover:text-purple-400 flex items-center gap-1 transition-colors ml-auto"
              >
                <Sparkles className="w-3 h-3" />
                Show Examples
              </motion.button>
            )}
          </AnimatePresence>
        </div>

        <AnimatePresence>
          {showExamples && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <div className="relative">
                <button 
                  onClick={() => setShowExamples(false)}
                  className="absolute top-0 right-0 p-1 text-zinc-500 hover:text-zinc-300 transition-colors z-10"
                >
                  <ChevronDown className="w-4 h-4" />
                </button>
                <ExampleQueries onSelect={(q) => setInput(q)} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <form onSubmit={handleSubmit} className="relative group">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-600 to-blue-600 rounded-2xl opacity-20 group-hover:opacity-40 transition duration-500 blur"></div>
          <div className="relative flex items-center bg-zinc-900 rounded-2xl border border-white/10 shadow-2xl">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question..."
              className="w-full bg-transparent border-none px-6 py-4 text-white placeholder-zinc-500 focus:outline-none focus:ring-0"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="mr-2 p-2 rounded-xl bg-white/5 hover:bg-white/10 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </form>
        <div className="text-center mt-2">
          <p className="text-xs text-zinc-600">
            Powered by ApertureDB, Gemini & Langgraph
          </p>
        </div>
      </div>
    </div>

    {/* Results Sidebar */}
    <ResultsSidebar
      isOpen={sidebarOpen}
      onClose={() => setSidebarOpen(false)}
      steps={sidebarSteps}
    />
  </>
  );
}
