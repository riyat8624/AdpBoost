import React, { useState } from 'react';
import {
  Zap,
  Sparkles,
  BarChart3,
  BookOpen,
  Database,
  Home,
  Menu,
  X,
  TrendingUp,
  Search,
} from 'lucide-react';

interface NavbarProps {
  activeTab: 'home' | 'prediction' | 'performance' | 'about' | 'dataset';
  setActiveTab: (tab: 'home' | 'prediction' | 'performance' | 'about' | 'dataset') => void;
  adaAccuracy: number;
  improvement: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  adaAccuracy,
  improvement,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks: { id: 'home' | 'prediction' | 'performance' | 'about' | 'dataset'; label: string; icon: React.ReactNode }[] = [
    { id: 'home', label: 'Overview', icon: <Home className="w-3.5 h-3.5" /> },
    { id: 'prediction', label: 'Predict Customer', icon: <Sparkles className="w-3.5 h-3.5" /> },
    { id: 'performance', label: 'Model Performance', icon: <BarChart3 className="w-3.5 h-3.5" /> },
    { id: 'about', label: 'About AdaBoost', icon: <BookOpen className="w-3.5 h-3.5" /> },
    { id: 'dataset', label: 'Dataset & Code', icon: <Database className="w-3.5 h-3.5" /> },
  ];

  return (
    <header className="sticky top-0 z-50 bg-white/85 backdrop-blur-md border-b border-[#EBE4F7] transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-18">
          
          {/* LEFT: Navigation Links (as seen in the reference header) */}
          <nav className="hidden lg:flex items-center gap-1 xl:gap-2">
            {navLinks.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all flex items-center gap-1.5 cursor-pointer ${
                    isActive
                      ? 'bg-[#EFEAFF] text-[#6D28D9] font-semibold shadow-xs'
                      : 'text-[#645E7A] hover:text-[#181524] hover:bg-[#F6F2FD]'
                  }`}
                >
                  <span className={isActive ? 'text-[#7C3AED]' : 'text-[#8E87A5]'}>{item.icon}</span>
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Mobile Menu Button (Left on small screens) */}
          <div className="flex lg:hidden items-center">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-xl text-[#1E1B2E] hover:bg-[#F3EEFC] transition-colors"
              aria-label="Toggle navigation"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>

          {/* CENTER: Clean Bold Brand Logo (matches NEXORA centered branding) */}
          <div
            className="flex items-center gap-2 cursor-pointer group"
            onClick={() => setActiveTab('home')}
          >
            <div className="w-8 h-8 rounded-xl bg-[#7C3AED] flex items-center justify-center text-white shadow-sm group-hover:scale-105 transition-transform">
              <Zap className="w-4 h-4 fill-white" />
            </div>
            <div className="text-center sm:text-left">
              <span className="font-heading font-black tracking-widest text-[#151226] text-lg sm:text-xl uppercase block leading-none">
                ADABOOST
              </span>
              <span className="text-[10px] tracking-wider text-[#7C3AED] font-semibold uppercase block">
                Customer AI
              </span>
            </div>
          </div>

          {/* RIGHT: Status Badge & Action Pill (matches reference right icons & pill) */}
          <div className="flex items-center gap-2.5 sm:gap-3">
            {/* Accuracy Pill (like the shopping cart pill in reference) */}
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#F3EEFD] border border-[#E3D9F8] text-xs">
              <span className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse" />
              <span className="font-semibold text-[#5B21B6]">{adaAccuracy}% Acc</span>
              <span className="text-[#9D93B8] font-mono">·</span>
              <span className="text-[#059669] font-medium text-[11px]">+{improvement}%</span>
            </div>

            {/* Primary Action Button (dark rounded-full button in the reference style) */}
            <button
              onClick={() => setActiveTab('prediction')}
              className="px-4 py-2 rounded-full bg-[#161324] hover:bg-[#28233D] text-white text-xs font-semibold shadow-sm transition-all flex items-center gap-1.5 cursor-pointer hover:shadow-md"
            >
              <Sparkles className="w-3.5 h-3.5 text-purple-300" />
              <span>Predict Now</span>
            </button>
          </div>
        </div>

        {/* Mobile Navigation Drawer */}
        {mobileMenuOpen && (
          <div className="lg:hidden py-3 px-2 border-t border-[#EBE4F7] bg-white/95 backdrop-blur-md rounded-2xl mb-2 shadow-lg space-y-1 animate-in fade-in slide-in-from-top-2 duration-200">
            {navLinks.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    setActiveTab(item.id);
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full px-3.5 py-2.5 rounded-xl text-left text-xs font-medium flex items-center justify-between transition-colors ${
                    isActive
                      ? 'bg-[#EFEAFF] text-[#6D28D9] font-semibold'
                      : 'text-[#645E7A] hover:bg-[#F6F2FD]'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    {item.icon}
                    <span>{item.label}</span>
                  </div>
                  {isActive && <span className="w-1.5 h-1.5 rounded-full bg-[#7C3AED]" />}
                </button>
              );
            })}

            <div className="pt-2 mt-2 border-t border-[#F0EAF8] flex items-center justify-between px-2 text-xs text-[#6B6482]">
              <span>Current Model Accuracy</span>
              <span className="font-mono font-bold text-[#7C3AED]">{adaAccuracy}%</span>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};
