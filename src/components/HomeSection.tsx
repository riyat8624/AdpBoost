import React from 'react';
import {
  Sparkles,
  ArrowRight,
  Database,
  SlidersHorizontal,
  GitFork,
  TreeDeciduous,
  Zap,
  CheckCircle2,
  TrendingUp,
  Award,
  ChevronRight,
  ShieldCheck,
  Cpu,
  Users,
} from 'lucide-react';
import { ModelComparisonResult } from '../ml/metrics';

interface HomeSectionProps {
  comparison: ModelComparisonResult;
  setActiveTab: (tab: 'home' | 'prediction' | 'performance' | 'about' | 'dataset') => void;
}

export const HomeSection: React.FC<HomeSectionProps> = ({ comparison, setActiveTab }) => {
  const { decisionTree, adaBoost, accuracyImprovement, sampleCount, testCount } = comparison;

  return (
    <div className="space-y-12 py-4">
      {/* 1. TOP EDITORIAL HEADLINE (matching the "Define Your STYLE · Own Your WORLD" reference) */}
      <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4 pt-2">
        <div>
          <span className="font-serif italic text-2xl sm:text-3xl text-[#181524] tracking-tight block">
            Classify Your
          </span>
          <h1 className="font-heading font-black text-5xl sm:text-7xl lg:text-8xl tracking-tight text-[#7C3AED] leading-none uppercase">
            CUSTOMERS
          </h1>
        </div>

        <div className="sm:text-right">
          <div className="inline-flex items-center gap-1.5 font-serif italic text-2xl sm:text-3xl text-[#181524] tracking-tight">
            <span>Boost Your</span>
            <span className="text-[#7C3AED] not-italic text-2xl">✦</span>
          </div>
          <h2 className="font-heading font-black text-5xl sm:text-7xl lg:text-8xl tracking-tight text-[#7C3AED] leading-none uppercase">
            ACCURACY
          </h2>
        </div>
      </div>

      {/* 2. SIGNATURE LAVENDER HERO CONTAINER (matching the purple center hero card in the reference) */}
      <section className="relative overflow-hidden rounded-[32px] bg-gradient-to-br from-[#8E78F4] via-[#8168EB] to-[#6E50E2] text-white p-8 sm:p-12 shadow-2xl shadow-purple-900/15">
        {/* Soft decorative background circles (as seen in the purple reference) */}
        <div className="absolute top-1/2 left-1/3 -translate-y-1/2 -z-0 h-96 w-96 rounded-full bg-white/10 blur-2xl pointer-events-none" />
        <div className="absolute -bottom-16 -right-16 -z-0 h-80 w-80 rounded-full bg-[#5B39D4]/40 blur-xl pointer-events-none" />

        <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left Hero Content */}
          <div className="lg:col-span-7 space-y-6">
            <div className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1 rounded-full bg-white/15 backdrop-blur-md text-white border border-white/20">
              <span>✦</span>
              <span>College AI / ML Project Edition</span>
            </div>

            <h2 className="text-3xl sm:text-5xl font-heading font-black tracking-tight leading-tight text-white">
              Where Weak Learners <br />
              Meets Strong Accuracy
            </h2>

            <p className="text-base sm:text-lg text-purple-100 font-normal leading-relaxed max-w-xl">
              Elevate customer retention prediction using the AdaBoost classification algorithm.
              Sequentially correct classification errors from standard Decision Tree baselines to achieve superior performance.
            </p>

            {/* Action Button & Metric Pill */}
            <div className="flex flex-wrap items-center gap-4 pt-2">
              <button
                onClick={() => setActiveTab('prediction')}
                className="px-7 py-3.5 rounded-full bg-[#151226] hover:bg-black text-white font-semibold text-sm shadow-xl transition-all flex items-center gap-2 cursor-pointer group hover:scale-[1.02]"
              >
                <span>Explore Prediction</span>
                <div className="w-5 h-5 rounded-full bg-white/20 flex items-center justify-center group-hover:translate-x-0.5 transition-transform">
                  <ArrowRight className="w-3 h-3 text-white" />
                </div>
              </button>

              <button
                onClick={() => setActiveTab('performance')}
                className="px-6 py-3.5 rounded-full bg-white/15 hover:bg-white/25 backdrop-blur-md text-white border border-white/25 font-medium text-sm transition-all flex items-center gap-2 cursor-pointer"
              >
                <TrendingUp className="w-4 h-4 text-purple-200" />
                <span>View Benchmarks</span>
              </button>
            </div>

            {/* Metric / User tag (matches "Loved by 20K+ Trendsetters" in reference) */}
            <div className="pt-3">
              <div className="inline-flex items-center gap-3 px-4 py-2 rounded-full bg-white/15 backdrop-blur-md border border-white/20 text-xs text-white">
                <div className="flex -space-x-2">
                  <div className="w-6 h-6 rounded-full bg-[#F59E0B] border-2 border-white flex items-center justify-center text-[10px] font-bold text-white">
                    ML
                  </div>
                  <div className="w-6 h-6 rounded-full bg-[#10B981] border-2 border-white flex items-center justify-center text-[10px] font-bold text-white">
                    ADA
                  </div>
                  <div className="w-6 h-6 rounded-full bg-[#6366F1] border-2 border-white flex items-center justify-center text-[10px] font-bold text-white">
                    DT
                  </div>
                </div>
                <span>Trained on <strong>1,000+</strong> Customer Records 💜</span>
              </div>
            </div>
          </div>

          {/* Right Hero: Showcase Card (matches "Featured Look" white card in the reference) */}
          <div className="lg:col-span-5 flex flex-col items-center lg:items-end space-y-4">
            {/* Top pill tags */}
            <div className="flex items-center gap-3 text-xs text-purple-100 font-medium">
              <div className="flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Verified Scikit-Learn</span>
              </div>
              <span>·</span>
              <div className="flex items-center gap-1">
                <Award className="w-3.5 h-3.5" />
                <span>+10% Boost</span>
              </div>
            </div>

            {/* The Floating White "Featured Model" Card */}
            <div className="w-full max-w-xs bg-white rounded-3xl p-5 text-[#181524] shadow-2xl shadow-purple-950/20 space-y-4 border border-purple-100">
              <div className="flex items-center justify-between text-xs text-[#706987]">
                <span className="font-semibold uppercase tracking-wider text-[10px] text-[#7C3AED]">
                  Featured Model
                </span>
                <span className="font-mono text-[11px] bg-[#F3EEFD] text-[#6D28D9] px-2 py-0.5 rounded-full font-semibold">
                  Ensemble
                </span>
              </div>

              {/* Visual representation of AdaBoost */}
              <div className="h-36 rounded-2xl bg-gradient-to-br from-[#F5F2FE] to-[#EBE4FC] border border-[#DDD3F8] p-3 flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <div className="w-8 h-8 rounded-xl bg-[#7C3AED] text-white flex items-center justify-center shadow-xs">
                    <Zap className="w-4 h-4 fill-white" />
                  </div>
                  <span className="text-[11px] font-mono text-[#5B21B6] font-semibold bg-white px-2 py-0.5 rounded-md shadow-xs">
                    T = 50 Stumps
                  </span>
                </div>

                <div className="space-y-1">
                  <span className="text-xs text-[#6B6482] block">Sequential Boosting Algorithm</span>
                  <div className="h-2 w-full bg-[#DDD3F8] rounded-full overflow-hidden flex">
                    <div className="h-full bg-[#7C3AED] rounded-full" style={{ width: `${adaBoost.accuracy}%` }} />
                  </div>
                  <div className="flex justify-between text-[10px] text-[#6B6482] font-mono">
                    <span>Baseline: {decisionTree.accuracy}%</span>
                    <span className="text-[#059669] font-bold">Boosted: {adaBoost.accuracy}%</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-heading font-bold text-base text-[#151226]">
                  AdaBoost.M1 Classifier
                </h4>
                <p className="text-xs text-[#6E6785]">
                  Adaptive Weighted Stumps Ensemble
                </p>
              </div>

              {/* Price / Metric pill button (matches "$79.99" purple button in reference) */}
              <button
                onClick={() => setActiveTab('performance')}
                className="w-full py-2.5 rounded-2xl bg-[#7C3AED] hover:bg-[#6D28D9] text-white font-semibold text-xs transition-all flex items-center justify-center gap-2 shadow-md shadow-purple-500/25 cursor-pointer"
              >
                <Zap className="w-3.5 h-3.5 fill-white" />
                <span>{adaBoost.accuracy}% Test Accuracy</span>
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* 3. CORE ML WORKFLOW CARDS (Styled in crisp white with soft lavender touches) */}
      <section className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2">
          <div>
            <span className="text-xs font-mono text-[#7C3AED] font-semibold uppercase tracking-wider block">
              End-to-End Pipeline
            </span>
            <h3 className="text-2xl sm:text-3xl font-heading font-black text-[#151226]">
              Machine Learning Workflow
            </h3>
          </div>
          <span className="text-xs text-[#68617F]">
            From raw customer data to boosted classification
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-3.5">
          {/* Step 1 */}
          <div className="rounded-2xl border border-[#ECE6F8] bg-white p-4.5 space-y-2 shadow-xs hover:border-[#D8CCF6] hover:shadow-md transition-all">
            <div className="w-8 h-8 rounded-xl bg-[#F0EBFE] text-[#7C3AED] flex items-center justify-center mb-2">
              <Database className="w-4 h-4" />
            </div>
            <span className="text-[10px] font-mono text-[#8E86A8] uppercase block">Step 01</span>
            <h4 className="font-heading font-bold text-sm text-[#181524]">Dataset Ingestion</h4>
            <p className="text-xs text-[#6A6382] leading-relaxed">
              1,000 records with 10 customer behavior & churn features.
            </p>
          </div>

          {/* Step 2 */}
          <div className="rounded-2xl border border-[#ECE6F8] bg-white p-4.5 space-y-2 shadow-xs hover:border-[#D8CCF6] hover:shadow-md transition-all">
            <div className="w-8 h-8 rounded-xl bg-[#FFF7ED] text-[#EA580C] flex items-center justify-center mb-2">
              <SlidersHorizontal className="w-4 h-4" />
            </div>
            <span className="text-[10px] font-mono text-[#8E86A8] uppercase block">Step 02</span>
            <h4 className="font-heading font-bold text-sm text-[#181524]">Preprocessing</h4>
            <p className="text-xs text-[#6A6382] leading-relaxed">
              Median imputation, one-hot encoding & standard scaling.
            </p>
          </div>

          {/* Step 3 */}
          <div className="rounded-2xl border border-[#ECE6F8] bg-white p-4.5 space-y-2 shadow-xs hover:border-[#D8CCF6] hover:shadow-md transition-all">
            <div className="w-8 h-8 rounded-xl bg-[#ECFDF5] text-[#059669] flex items-center justify-center mb-2">
              <GitFork className="w-4 h-4" />
            </div>
            <span className="text-[10px] font-mono text-[#8E86A8] uppercase block">Step 03</span>
            <h4 className="font-heading font-bold text-sm text-[#181524]">Train/Test Split</h4>
            <p className="text-xs text-[#6A6382] leading-relaxed">
              Stratified 80/20 train/test split preserving churn ratio.
            </p>
          </div>

          {/* Step 4 */}
          <div className="rounded-2xl border border-[#ECE6F8] bg-white p-4.5 space-y-2 shadow-xs hover:border-[#D8CCF6] hover:shadow-md transition-all">
            <div className="w-8 h-8 rounded-xl bg-[#F1F5F9] text-[#475569] flex items-center justify-center mb-2">
              <TreeDeciduous className="w-4 h-4" />
            </div>
            <span className="text-[10px] font-mono text-[#8E86A8] uppercase block">Step 04</span>
            <h4 className="font-heading font-bold text-sm text-[#181524]">Decision Tree</h4>
            <p className="text-xs text-[#6A6382] leading-relaxed">
              CART baseline (depth 3) trained as benchmark.
            </p>
          </div>

          {/* Step 5 */}
          <div className="rounded-2xl border-2 border-[#7C3AED] bg-[#FAF8FE] p-4.5 space-y-2 shadow-sm">
            <div className="w-8 h-8 rounded-xl bg-[#7C3AED] text-white flex items-center justify-center mb-2 shadow-xs">
              <Zap className="w-4 h-4 fill-white" />
            </div>
            <span className="text-[10px] font-mono text-[#7C3AED] font-bold uppercase block">Step 05</span>
            <h4 className="font-heading font-bold text-sm text-[#181524]">AdaBoost Model</h4>
            <p className="text-xs text-[#6A6382] leading-relaxed">
              50 sequential stumps with adaptive weight updating.
            </p>
          </div>

          {/* Step 6 */}
          <div className="rounded-2xl border border-[#ECE6F8] bg-white p-4.5 space-y-2 shadow-xs hover:border-[#D8CCF6] hover:shadow-md transition-all">
            <div className="w-8 h-8 rounded-xl bg-[#EFF6FF] text-[#2563EB] flex items-center justify-center mb-2">
              <Sparkles className="w-4 h-4" />
            </div>
            <span className="text-[10px] font-mono text-[#8E86A8] uppercase block">Step 06</span>
            <h4 className="font-heading font-bold text-sm text-[#181524]">Inference</h4>
            <p className="text-xs text-[#6A6382] leading-relaxed">
              Interactive prediction with confidence meter & XAI factors.
            </p>
          </div>
        </div>
      </section>

      {/* 4. WHY ADABOOST & MODEL COMPARISON HIGHLIGHTS */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
        {/* Why AdaBoost Card */}
        <div className="lg:col-span-6 rounded-3xl border border-[#EBE4F7] bg-white p-7 sm:p-9 space-y-5 shadow-xs">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-[#EFEAFF] text-[#7C3AED] flex items-center justify-center">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <span className="text-xs font-mono font-semibold text-[#7C3AED] uppercase tracking-wider block">
                College Project Feature
              </span>
              <h3 className="text-xl font-heading font-bold text-[#181524]">
                Why AdaBoost?
              </h3>
            </div>
          </div>

          <p className="text-sm text-[#5B5573] leading-relaxed">
            A single shallow Decision Tree acts as a <strong>weak learner</strong>—it often makes
            systematic errors on boundary customer profiles and is sensitive to local greedy splits.
          </p>

          <p className="text-sm text-[#5B5573] leading-relaxed">
            <strong>AdaBoost (Adaptive Boosting)</strong> combines multiple weak learners into a
            powerful strong classifier. In each boosting iteration, it identifies instances that were
            misclassified by prior stumps and exponentially increases their sample weights ($w_i$).
          </p>

          <div className="space-y-2.5 pt-2">
            <div className="flex items-start gap-2.5 text-xs text-[#453F5C]">
              <CheckCircle2 className="w-4 h-4 text-[#10B981] shrink-0 mt-0.5" />
              <span>
                <strong>Sequential Learning:</strong> Successive decision stumps focus on hard-to-classify customers.
              </span>
            </div>
            <div className="flex items-start gap-2.5 text-xs text-[#453F5C]">
              <CheckCircle2 className="w-4 h-4 text-[#10B981] shrink-0 mt-0.5" />
              <span>
                <strong>Adaptive Weighting:</strong> High penalties for churn cases that slipped past earlier rounds.
              </span>
            </div>
            <div className="flex items-start gap-2.5 text-xs text-[#453F5C]">
              <CheckCircle2 className="w-4 h-4 text-[#10B981] shrink-0 mt-0.5" />
              <span>
                <strong>Weighted Majority Vote:</strong> Final prediction incorporates estimator weights ($\alpha_t$).
              </span>
            </div>
          </div>
        </div>

        {/* Model Comparison Card */}
        <div className="lg:col-span-6 rounded-3xl border border-[#EBE4F7] bg-white p-7 sm:p-9 space-y-6 shadow-xs flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-xs font-mono font-semibold text-[#7C3AED] uppercase tracking-wider block">
                  Empirical Results
                </span>
                <h3 className="text-xl font-heading font-bold text-[#181524]">
                  Model Comparison
                </h3>
              </div>
              <span className="text-xs font-bold px-3 py-1 rounded-full bg-[#ECFDF5] text-[#059669] border border-[#A7F3D0]">
                +{accuracyImprovement}% Accuracy Boost
              </span>
            </div>

            <div className="space-y-3">
              {/* Decision Tree Baseline Row */}
              <div className="p-4 rounded-2xl bg-[#F9F7FD] border border-[#ECE4F8] flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-xl bg-white border border-[#E2D8F6] text-[#64748B] flex items-center justify-center">
                    <TreeDeciduous className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-heading font-bold text-[#181524]">
                      Decision Tree (Baseline)
                    </h4>
                    <span className="text-xs text-[#736B8C]">Single CART Tree · Max Depth 3</span>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-base font-bold text-[#334155]">{decisionTree.accuracy}%</span>
                  <span className="text-[11px] text-[#78718E] block">Accuracy</span>
                </div>
              </div>

              {/* AdaBoost Row */}
              <div className="p-4 rounded-2xl bg-[#FAF7FE] border-2 border-[#7C3AED] flex items-center justify-between shadow-xs">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-xl bg-[#7C3AED] text-white flex items-center justify-center shadow-xs">
                    <Zap className="w-4 h-4 fill-white" />
                  </div>
                  <div>
                    <h4 className="text-sm font-heading font-bold text-[#181524]">
                      AdaBoost Classifier
                    </h4>
                    <span className="text-xs text-[#7C3AED] font-medium">50 Decision Stumps · Boosted</span>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-base font-bold text-[#7C3AED]">{adaBoost.accuracy}%</span>
                  <span className="text-[11px] text-[#059669] font-semibold block">
                    +{accuracyImprovement}% Gain
                  </span>
                </div>
              </div>
            </div>

            {/* Quick Metrics Bar Comparison */}
            <div className="space-y-2 pt-1">
              <div className="flex justify-between text-xs text-[#5B5573]">
                <span>F1-Score: <strong>{adaBoost.f1Score}%</strong> vs {decisionTree.f1Score}%</span>
                <span className="font-semibold text-[#7C3AED]">+{comparison.f1Improvement}%</span>
              </div>
              <div className="h-2 w-full bg-[#EFEAFF] rounded-full overflow-hidden flex">
                <div className="h-full bg-[#7C3AED] rounded-full" style={{ width: `${adaBoost.f1Score}%` }} />
              </div>
            </div>
          </div>

          <div className="pt-2 border-t border-[#F0EAF8] flex items-center justify-between">
            <span className="text-xs text-[#7B7494]">Tested on {testCount} holdout samples</span>
            <button
              onClick={() => setActiveTab('performance')}
              className="text-xs font-semibold text-[#7C3AED] hover:text-[#6D28D9] flex items-center gap-1 cursor-pointer"
            >
              <span>Full Analytics & Confusion Matrix</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
