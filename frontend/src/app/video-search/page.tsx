'use client';

import { useState } from 'react';
import { Search, SlidersHorizontal, Play, Clock, Eye, User, Loader2, ExternalLink } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { searchVideos, API_BASE_URL } from '@/lib/api';
import { VideoSearchResult } from '@/lib/types';
import { VideoExampleQueries } from '@/components/VideoExampleQueries';

export default function VideoSearchPage() {
  const [query, setQuery] = useState('');
  const [topN, setTopN] = useState(10);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<VideoSearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await searchVideos({
        query,
        top_n: topN,
        include_videos: false
      });
      setResults(response.results);
    } catch (err: any) {
      setError(err.message || 'Failed to search videos');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-12">
        <h1 className="text-3xl font-bold text-white mb-2">Semantic Video Search</h1>
        <p className="text-zinc-400">Search inside video content using visual and audio understanding</p>
      </div>

      {/* Search Bar */}
      <div className="mb-8 max-w-4xl mx-auto">
        <form onSubmit={handleSearch} className="relative z-10">
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl opacity-20 group-hover:opacity-40 transition duration-500 blur"></div>
            <div className="relative flex items-center bg-zinc-900 rounded-2xl border border-white/10 shadow-2xl p-2">
              <Search className="ml-4 w-5 h-5 text-zinc-500" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Describe a scene, demo, or slide..."
                className="w-full bg-transparent border-none px-4 py-3 text-white placeholder-zinc-500 focus:outline-none focus:ring-0 text-lg"
              />
              <div className="flex items-center gap-2 pr-2">
                <button
                  type="button"
                  onClick={() => setShowFilters(!showFilters)}
                  className={`p-2 rounded-xl transition-colors ${showFilters ? 'bg-white/10 text-white' : 'text-zinc-500 hover:text-white hover:bg-white/5'}`}
                  title="Filters"
                >
                  <SlidersHorizontal className="w-5 h-5" />
                </button>
                <button
                  type="submit"
                  disabled={isLoading || !query.trim()}
                  className="px-6 py-2.5 bg-white text-black font-medium rounded-xl hover:bg-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                >
                  {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <span>Search</span>}
                </button>
              </div>
            </div>
          </div>

          {/* Filters */}
          <AnimatePresence>
            {showFilters && (
              <motion.div
                initial={{ opacity: 0, height: 0, marginTop: 0 }}
                animate={{ opacity: 1, height: 'auto', marginTop: 16 }}
                exit={{ opacity: 0, height: 0, marginTop: 0 }}
                className="overflow-hidden"
              >
                <div className="bg-zinc-900/50 border border-white/5 rounded-xl p-6 backdrop-blur-sm">
                  <div className="flex items-center gap-8">
                    <div className="flex-1">
                      <div className="flex justify-between mb-2">
                        <label className="text-sm text-zinc-400">Results Limit</label>
                        <span className="text-sm text-white font-mono">{topN}</span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max="50"
                        value={topN}
                        onChange={(e) => setTopN(parseInt(e.target.value))}
                        className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                      />
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </form>
      </div>

      {/* Example Queries */}
      <div className="max-w-4xl mx-auto mb-12">
        <VideoExampleQueries onSelect={setQuery} />
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400">
          {error}
        </div>
      )}

      {/* Results Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {results.map((video, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.05 }}
            className="bg-zinc-900/50 border border-white/5 rounded-2xl overflow-hidden hover:border-blue-500/30 transition-colors flex flex-col"
          >
            {/* Header Info */}
            <div className="p-5 border-b border-white/5">
              <div className="flex items-start justify-between gap-4 mb-4">
                <h3 className="font-medium text-lg text-white line-clamp-2">
                  {video.talk_title}
                </h3>
                {video.youtube_url && (
                  <a 
                    href={video.youtube_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-zinc-500 hover:text-blue-400 transition-colors flex-shrink-0"
                  >
                    <ExternalLink className="w-5 h-5" />
                  </a>
                )}
              </div>

              <div className="grid grid-cols-2 gap-y-4 text-sm">
                <div>
                  <div className="text-zinc-500 text-xs mb-1">Speaker</div>
                  <div className="text-zinc-200 font-medium">{video.speaker_name}</div>
                </div>
                <div>
                  <div className="text-zinc-500 text-xs mb-1">Similarity</div>
                  <div className="text-green-400 font-medium">{Math.round(video.similarity_score * 100)}%</div>
                </div>
              </div>
            </div>

            {/* Video Player */}
            <div className="aspect-video bg-black relative mt-auto">
               <video 
                 controls
                 className="w-full h-full"
                 poster={video.youtube_url ? getYouTubeThumbnail(video.youtube_url) : undefined}
                 preload="metadata"
                 src={`${API_BASE_URL}/videos/stream?talk_title=${encodeURIComponent(video.talk_title)}`}
               >
                 Your browser does not support the video tag.
               </video>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Empty State */}
      {!isLoading && results.length === 0 && query && (
        <div className="text-center py-20 text-zinc-500">
          <p>No videos found matching your query.</p>
        </div>
      )}
    </div>
  );
}

function getYouTubeThumbnail(url: string): string | undefined {
  if (!url) return undefined;
  const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
  const match = url.match(regExp);
  return (match && match[2].length === 11)
    ? `https://img.youtube.com/vi/${match[2]}/mqdefault.jpg`
    : undefined;
}
