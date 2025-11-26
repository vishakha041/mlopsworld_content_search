import { MonitorPlay, Presentation, Users, Code } from 'lucide-react';

const VIDEO_EXAMPLE_CATEGORIES = [
  {
    id: 'visuals',
    label: 'Visuals & Slides',
    icon: Presentation,
    queries: [
      "Slides showing transformer architecture",
      "Diagrams of MLOps pipelines"
    ]
  },
  {
    id: 'demos',
    label: 'Demos & Code',
    icon: Code,
    queries: [
      "Live coding demos in VS Code",
      "Terminal sessions showing CLI tools"
    ]
  },
  {
    id: 'concepts',
    label: 'Concepts',
    icon: MonitorPlay,
    queries: [
      "Explanation of RAG pipelines",
      "Discussion about model quantization"
    ]
  },
  {
    id: 'scenes',
    label: 'Scenes',
    icon: Users,
    queries: [
      "Panel discussion with multiple speakers",
      "Q&A sessions with audience"
    ]
  }
];

interface VideoExampleQueriesProps {
  onSelect: (query: string) => void;
}

export function VideoExampleQueries({ onSelect }: VideoExampleQueriesProps) {
  return (
    <div className="w-full mb-8">
      <div className="flex items-center gap-2 mb-4 px-1">
        <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Suggested Video Queries</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {VIDEO_EXAMPLE_CATEGORIES.map((category) => (
          <div key={category.id} className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-medium text-zinc-400 px-1">
              <category.icon className="w-3 h-3" />
              <span>{category.label}</span>
            </div>
            <div className="flex flex-col gap-2">
              {category.queries.map((query, idx) => (
                <button
                  key={idx}
                  onClick={() => onSelect(query)}
                  className="text-left text-xs p-3 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/10 transition-all duration-200 text-zinc-300 hover:text-white hover:shadow-lg hover:shadow-blue-500/5 truncate"
                  title={query}
                >
                  {query}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
