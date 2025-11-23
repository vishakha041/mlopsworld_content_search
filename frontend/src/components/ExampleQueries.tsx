import { Search, Brain, Users, TrendingUp } from 'lucide-react';

const EXAMPLE_CATEGORIES = [
  {
    id: 'filtering',
    label: 'Filtering & Sorting',
    icon: Search,
    queries: [
      "Show me the most popular talks from 2024",
      "Find talks by speakers from Google"
    ]
  },
  {
    id: 'semantic',
    label: 'Semantic Search',
    icon: Brain,
    queries: [
      "Which talks discuss AI agents with memory?",
      "Find experts in vector databases and RAG"
    ]
  },
  {
    id: 'speakers',
    label: 'Speaker Analysis',
    icon: Users,
    queries: [
      "Who are the top 10 most active speakers?",
      "Which companies presented the most talks?"
    ]
  },
  {
    id: 'trends',
    label: 'Trends & Tools',
    icon: TrendingUp,
    queries: [
      "What are the most discussed tools in 2024?",
      "Show trending technologies in MLOps"
    ]
  }
];

interface ExampleQueriesProps {
  onSelect: (query: string) => void;
}

export function ExampleQueries({ onSelect }: ExampleQueriesProps) {
  return (
    <div className="w-full mb-4">
      <div className="flex items-center gap-2 mb-3 px-1">
        <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Suggested Queries</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {EXAMPLE_CATEGORIES.map((category) => (
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
                  className="text-left text-xs p-2.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/10 transition-all duration-200 text-zinc-300 hover:text-white hover:shadow-lg hover:shadow-purple-500/5 truncate"
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
