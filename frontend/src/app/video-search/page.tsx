'use client';

import { useState } from 'react';
import { Search, SlidersHorizontal, Play, Clock, Eye, User, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { searchVideos } from '@/lib/api';
import { VideoSearchResult } from '@/lib/types';

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
      <div className="mb-12 max-w-3xl">
        <form onSubmit={handleSearch} className="relative z-10">
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl opacity-20 group-hover:opacity-40 transition duration-500 blur"></div>
            <div className="relative flex items-center bg-zinc-900 rounded-2xl border border-white/10 shadow-2xl">
              <Search className="ml-6 w-5 h-5 text-zinc-500" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Describe a scene, demo, or slide..."
                className="w-full bg-transparent border-none px-4 py-4 text-white placeholder-zinc-500 focus:outline-none focus:ring-0 text-lg"
              />
              <button
                type="button"
                onClick={() => setShowFilters(!showFilters)}
                className={`p-2 mr-2 rounded-xl transition-colors ${showFilters ? 'bg-white/10 text-white' : 'text-zinc-500 hover:text-white'}`}
              >
                <SlidersHorizontal className="w-5 h-5" />
              </button>
              <button
                type="submit"
                disabled={isLoading || !query.trim()}
                className="mr-2 px-6 py-2 bg-white text-black font-medium rounded-xl hover:bg-zinc-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Search'}
              </button>
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
                    <div className="flex-1">
                      {/* Placeholder for future filters */}
                      <p className="text-xs text-zinc-600">More filters coming soon...</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </form>
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
          <motion.a
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.05 }}
            href={video.youtube_url || '#'}
            target="_blank"
            rel="noopener noreferrer"
            className="group bg-zinc-900/50 border border-white/5 hover:border-blue-500/30 rounded-2xl overflow-hidden transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-blue-500/10"
          >
            {/* Thumbnail Placeholder / Video Preview */}
            <div className="aspect-video bg-zinc-800 relative overflow-hidden">
              {/* We could use youtube thumbnail if available, for now a gradient placeholder */}
              <div className="absolute inset-0 bg-gradient-to-br from-zinc-800 to-zinc-900 group-hover:scale-105 transition-transform duration-500" />
              
              <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-black/40 backdrop-blur-[2px]">
                <div className="w-12 h-12 rounded-full bg-white/10 backdrop-blur-md flex items-center justify-center border border-white/20">
                  <Play className="w-5 h-5 text-white fill-white" />
                </div>
              </div>

              {/* Similarity Badge */}
              <div className="absolute top-3 right-3 px-2 py-1 bg-black/60 backdrop-blur-md rounded-lg border border-white/10 text-xs font-medium text-blue-300">
                {Math.round(video.similarity_score * 100)}% Match
              </div>

              {/* Duration Badge */}
              {video.metadata?.duration_sec && (
                <div className="absolute bottom-3 right-3 px-2 py-1 bg-black/60 backdrop-blur-md rounded-lg border border-white/10 text-xs text-white flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {Math.floor(video.metadata.duration_sec / 60)}:{(video.metadata.duration_sec % 60).toString().padStart(2, '0')}
                </div>
              )}
            </div>

            {/* Content */}
            <div className="p-5">
              <h3 className="font-medium text-lg text-white mb-2 line-clamp-2 group-hover:text-blue-300 transition-colors">
                {video.talk_title}
              </h3>
              
              <div className="flex items-center gap-2 text-sm text-zinc-400 mb-4">
                <User className="w-4 h-4" />
                <span>{video.speaker_name}</span>
              </div>

              <div className="flex items-center justify-between text-xs text-zinc-500 pt-4 border-t border-white/5">
                <div className="flex items-center gap-1.5">
                  <Eye className="w-3 h-3" />
                  <span>{video.yt_views?.toLocaleString() || 0} views</span>
                </div>
                {video.category_primary && (
                  <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/5">
                    {video.category_primary}
                  </span>
                )}
              </div>
            </div>
          </motion.a>
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
