'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { MessageSquare, Video, Aperture } from 'lucide-react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';

const navItems = [
  {
    name: 'Agent Chat',
    href: '/',
    icon: MessageSquare,
  },
  {
    name: 'Video Search',
    href: '/video-search',
    icon: Video,
  },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="fixed left-0 top-0 h-full w-20 flex flex-col items-center py-8 bg-black/20 backdrop-blur-xl border-r border-white/5 z-50">
      <div className="mb-12">
        <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20">
          <Aperture className="w-6 h-6 text-white" />
        </div>
      </div>

      <div className="flex flex-col gap-8 w-full px-4">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className="relative group flex items-center justify-center w-full aspect-square"
            >
              {isActive && (
                <motion.div
                  layoutId="activeNav"
                  className="absolute inset-0 bg-white/10 rounded-xl"
                  initial={false}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
              
              <div className={clsx(
                "relative z-10 transition-colors duration-200",
                isActive ? "text-white" : "text-zinc-500 group-hover:text-zinc-300"
              )}>
                <Icon className="w-6 h-6" />
              </div>

              {/* Tooltip */}
              <div className="absolute left-full ml-4 px-2 py-1 bg-zinc-900 border border-white/10 rounded text-xs text-zinc-300 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                {item.name}
              </div>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
