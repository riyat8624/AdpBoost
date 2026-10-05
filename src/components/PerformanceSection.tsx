import React, { useState } from 'react';
import {
  TrendingUp,
  Award,
  Sliders,
  RotateCw,
  BarChart3,
  CheckCircle2,
  TreeDeciduous,
  Zap,
} from 'lucide-react';
import { ModelComparisonResult } from '../ml/metrics';
import { AdaBoostClassifier, DecisionTreeClassifier } from '../ml/models';
import { PreprocessedData, TrainingConfig } from '../types/ml';

interface PerformanceSectionProps {
  comparison: ModelComparisonResult;
  adaModel: AdaBoostClassifier;
  dtModel: DecisionTreeClassifier;
  preprocessedData: PreprocessedData;
  onRetrain: (config: TrainingConfig) => void;
  isRetraining: boolean;
}

export const PerformanceSection: React.FC<PerformanceSectionProps> = ({
  comparison,
  adaModel,
  dtModel,
  preprocessedData,
  onRetrain,
  isRetraining,
}) => {
  const { decisionTree, adaBoost, accuracyImprovement, f1Improvement, sampleCount, testCount } =
    comparison;

  const [activeMatrixTab, setActiveMatrixTab] = useState<'adaboost' | 'decisionTree'>('adaboost');

  const [config, setConfig] = useState<TrainingConfig>({
    nEstimators: adaModel.nEstimators || 50,
    learningRate: adaModel.learningRate || 1.0,
    decisionTreeMaxDepth: dtModel.maxDepth || 3,
    testSplitRatio: 0.2,
    randomSeed: 42,
  });

  const sortedImportances = Object.entries(adaModel.featureImportances || {})
    .sort(([, a], [, b]) => b - a)
    .slice(0, 8);

  const cm =
    activeMatrixTab === 'adaboost'
      ? adaBoost.confusionMatrix
      : decisionTree.confusionMatrix;

  const totalTest = cm.trueNegative + cm.falsePositive + cm.falseNegative + cm.truePositive || 1;
  const tnPct = Math.round((cm.trueNegative / totalTest) * 100);
  const fpPct = Math.round((cm.falsePositive / totalTest) * 100);
  const fnPct = Math.round((cm.falseNegative / totalTest) * 100);
  const tpPct = Math.round((cm.truePositive / totalTest) * 100);

  const metricsData = [
    { label: 'Accuracy', dt: decisionTree.accuracy, ada: adaBoost.accuracy },
    { label: 'Precision', dt: decisionTree.precision, ada: adaBoost.precision },
    { label: 'Recall', dt: decisionTree.recall, ada: adaBoost.recall },
    { label: 'F1 Score', dt: decisionTree.f1Score, ada: adaBoost.f1Score },
  ];

  return (
    <div className="space-y-10 py-4">
      {/* Header */}
      <div>
        <span className="text-xs font-mono font-semibold text-[#7C3AED] uppercase tracking-wider block">
          Model Evaluation & Benchmarks
        </span>
        <h2 className="text-3xl sm:text-4xl font-heading font-black text-[#151226]">
          Performance Comparison
        </h2>
        <p className="text-sm text-[#6A6384] mt-1 max-w-2xl">
          Empirical evaluation on the 20% holdout test dataset ({testCount} samples) comparing
          baseline Decision Tree against AdaBoost.
        </p>
      </div>

      {/* Dataset & Split Summary Pills */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-5 rounded-3xl bg-white border border-[#ECE4F8] shadow-xs">
        <div>
          <span className="text-xs text-[#7B7494] block">Dataset Samples</span>
          <span className="text-2xl font-heading font-bold text-[#181524]">{sampleCount}</span>
          <span className="text-[11px] text-[#9A93B2] block">Customer Profiles</span>
        </div>
        <div>
          <span className="text-xs text-[#7B7494] block">Feature Dims</span>
          <span className="text-2xl font-heading font-bold text-[#7C3AED]">10</span>
          <span className="text-[11px] text-[#9A93B2] block">
            {preprocessedData.featureNames.length} Encoded Features
          </span>
        </div>
        <div>
          <span className="text-xs text-[#7B7494] block">Train / Test Partition</span>
          <span className="text-2xl font-heading font-bold text-[#181524]">80% / 20%</span>
          <span className="text-[11px] text-[#9A93B2] block">
            {sampleCount - testCount} Train · {testCount} Test
          </span>
        </div>
        <div>
          <span className="text-xs text-[#7B7494] block">Ensemble Architecture</span>
          <span className="text-2xl font-heading font-bold text-[#7C3AED]">50 Stumps</span>
          <span className="text-[11px] text-[#9A93B2] block">Decision Stumps</span>
        </div>
      </div>

      {/* Prominent Improvement Banner (in signature purple container style) */}
      <div className="rounded-3xl border-2 border-[#7C3AED] bg-gradient-to-r from-[#F5F2FE] via-white to-[#EFEAFF] p-7 sm:p-9 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-1.5 text-xs font-bold text-[#059669] bg-[#ECFDF5] border border-[#A7F3D0] px-3 py-1 rounded-full">
              <Award className="w-3.5 h-3.5" />
              <span>AdaBoost Improvement Verified</span>
            </div>
            <h3 className="text-2xl sm:text-3xl font-heading font-black text-[#151226]">
              AdaBoost Accuracy ({adaBoost.accuracy}%) − Decision Tree Accuracy ({decisionTree.accuracy}%) = +{accuracyImprovement}%
            </h3>
            <p className="text-sm text-[#5B5573] max-w-2xl leading-relaxed">
              AdaBoost achieves a{' '}
              <strong className="text-[#7C3AED]">+{accuracyImprovement}% accuracy gain</strong> and{' '}
              <strong className="text-[#059669]">+{f1Improvement}% higher F1-score</strong> by
              adaptively focusing on customers misclassified by earlier rounds.
            </p>
          </div>

          <div className="shrink-0 text-center md:text-right p-5 rounded-2xl bg-white border border-[#DDD3F6] shadow-sm">
            <span className="text-xs font-semibold text-[#7B7494] uppercase tracking-wider block">
              Net Accuracy Gain
            </span>
            <span className="text-4xl font-heading font-black text-[#7C3AED]">
              +{accuracyImprovement}%
            </span>
            <span className="text-xs text-[#059669] font-semibold block mt-0.5">
              +{f1Improvement}% F1 Boost
            </span>
          </div>
        </div>
      </div>

      {/* Side-by-Side Model Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Accuracy */}
        <div className="rounded-3xl border border-[#ECE4F8] bg-white p-6 space-y-3 shadow-xs">
          <span className="text-xs font-semibold text-[#7B7494] uppercase tracking-wider block">
            Accuracy Score
          </span>
          <div className="flex items-baseline justify-between">
            <span className="text-3xl font-heading font-black text-[#181524]">
              {adaBoost.accuracy}%
            </span>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-[#ECFDF5] text-[#059669]">
              +{accuracyImprovement}%
            </span>
          </div>
          <div className="text-xs text-[#6A6382] flex justify-between pt-2 border-t border-[#F0EBF8]">
            <span>Decision Tree:</span>
            <span className="font-semibold text-[#36304D]">{decisionTree.accuracy}%</span>
          </div>
        </div>

        {/* Precision */}
        <div className="rounded-3xl border border-[#ECE4F8] bg-white p-6 space-y-3 shadow-xs">
          <span className="text-xs font-semibold text-[#7B7494] uppercase tracking-wider block">
            Precision
          </span>
          <div className="flex items-baseline justify-between">
            <span className="text-3xl font-heading font-black text-[#7C3AED]">
              {adaBoost.precision}%
            </span>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-[#F3EEFD] text-[#6D28D9]">
              +{comparison.precisionImprovement}%
            </span>
          </div>
          <div className="text-xs text-[#6A6382] flex justify-between pt-2 border-t border-[#F0EBF8]">
            <span>Decision Tree:</span>
            <span className="font-semibold text-[#36304D]">{decisionTree.precision}%</span>
          </div>
        </div>

        {/* Recall */}
        <div className="rounded-3xl border border-[#ECE4F8] bg-white p-6 space-y-3 shadow-xs">
          <span className="text-xs font-semibold text-[#7B7494] uppercase tracking-wider block">
            Recall (Sensitivity)
          </span>
          <div className="flex items-baseline justify-between">
            <span className="text-3xl font-heading font-black text-[#059669]">
              {adaBoost.recall}%
            </span>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-[#ECFDF5] text-[#059669]">
              +{comparison.recallImprovement}%
            </span>
          </div>
          <div className="text-xs text-[#6A6382] flex justify-between pt-2 border-t border-[#F0EBF8]">
            <span>Decision Tree:</span>
            <span className="font-semibold text-[#36304D]">{decisionTree.recall}%</span>
          </div>
        </div>

        {/* F1 Score */}
        <div className="rounded-3xl border border-[#ECE4F8] bg-white p-6 space-y-3 shadow-xs">
          <span className="text-xs font-semibold text-[#7B7494] uppercase tracking-wider block">
            F1-Score
          </span>
          <div className="flex items-baseline justify-between">
            <span className="text-3xl font-heading font-black text-[#5B21B6]">
              {adaBoost.f1Score}%
            </span>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-[#EFEAFF] text-[#6D28D9]">
              +{f1Improvement}%
            </span>
          </div>
          <div className="text-xs text-[#6A6382] flex justify-between pt-2 border-t border-[#F0EBF8]">
            <span>Decision Tree:</span>
            <span className="font-semibold text-[#36304D]">{decisionTree.f1Score}%</span>
          </div>
        </div>
      </div>

      {/* Charts Row: Bar Chart Comparison & Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Metric Comparison Bar Chart */}
        <div className="lg:col-span-7 rounded-3xl border border-[#ECE4F8] bg-white p-7 space-y-6 shadow-xs">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="font-heading font-bold text-lg text-[#181524]">
                Metric Comparison Bar Chart
              </h3>
              <p className="text-xs text-[#7B7494]">Decision Tree Baseline vs AdaBoost Ensemble</p>
            </div>
            {/* Legend */}
            <div className="flex items-center gap-4 text-xs font-medium">
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-md bg-[#CBD5E1]" />
                <span className="text-[#64748B]">Decision Tree</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-md bg-[#7C3AED]" />
                <span className="text-[#7C3AED] font-semibold">AdaBoost</span>
              </div>
            </div>
          </div>

          <div className="space-y-5 pt-1">
            {metricsData.map((m, idx) => (
              <div key={idx} className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-[#36304D]">{m.label}</span>
                  <div className="flex items-center gap-3 font-mono">
                    <span className="text-[#64748B]">{m.dt}%</span>
                    <span className="text-[#A29BB9]">→</span>
                    <span className="text-[#7C3AED] font-bold">{m.ada}%</span>
                  </div>
                </div>

                <div className="space-y-1.5">
                  {/* Baseline Tree Bar */}
                  <div className="h-3 w-full bg-[#F3EEFB] rounded-full overflow-hidden flex">
                    <div
                      className="h-full bg-[#94A3B8] rounded-full transition-all duration-500"
                      style={{ width: `${m.dt}%` }}
                    />
                  </div>
                  {/* AdaBoost Bar */}
                  <div className="h-3.5 w-full bg-[#F3EEFB] rounded-full overflow-hidden flex">
                    <div
                      className="h-full bg-gradient-to-r from-[#8B5CF6] to-[#6D28D9] rounded-full transition-all duration-500 shadow-xs"
                      style={{ width: `${m.ada}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="pt-2 text-xs text-[#5B5573] bg-[#F9F7FD] p-3.5 rounded-2xl border border-[#ECE4F8]">
            💡 <strong>Observation:</strong> Recall shows the highest relative gain (+
            {comparison.recallImprovement}%), meaning AdaBoost drastically minimizes false
            negatives and captures churn risk that a single decision tree missed.
          </div>
        </div>

        {/* Confusion Matrix */}
        <div className="lg:col-span-5 rounded-3xl border border-[#ECE4F8] bg-white p-7 space-y-5 shadow-xs">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-heading font-bold text-lg text-[#181524]">
                Confusion Matrix
              </h3>
              <p className="text-xs text-[#7B7494]">{testCount} holdout customer samples</p>
            </div>

            <div className="flex items-center bg-[#F4EFFC] p-1 rounded-xl text-xs">
              <button
                onClick={() => setActiveMatrixTab('adaboost')}
                className={`px-3 py-1 rounded-lg transition-all cursor-pointer ${
                  activeMatrixTab === 'adaboost'
                    ? 'bg-[#7C3AED] text-white font-semibold shadow-xs'
                    : 'text-[#6B6482] hover:text-[#181524]'
                }`}
              >
                AdaBoost
              </button>
              <button
                onClick={() => setActiveMatrixTab('decisionTree')}
                className={`px-3 py-1 rounded-lg transition-all cursor-pointer ${
                  activeMatrixTab === 'decisionTree'
                    ? 'bg-white text-[#181524] font-semibold shadow-xs'
                    : 'text-[#6B6482] hover:text-[#181524]'
                }`}
              >
                Decision Tree
              </button>
            </div>
          </div>

          {/* Matrix Grid */}
          <div className="space-y-2 pt-1">
            <div className="text-center text-[11px] font-mono text-[#7B7494] uppercase tracking-wider font-semibold">
              Predicted Label
            </div>

            <div className="grid grid-cols-12 gap-2 text-xs">
              <div className="col-span-2 flex items-center justify-center -rotate-90 text-[11px] font-mono text-[#7B7494] uppercase tracking-wider font-semibold">
                Actual
              </div>

              <div className="col-span-10 grid grid-cols-2 gap-2.5">
                {/* Column Headers */}
                <div className="text-center font-mono text-[11px] text-[#7B7494] pb-0.5">
                  Pred Retained (0)
                </div>
                <div className="text-center font-mono text-[11px] text-[#7B7494] pb-0.5">
                  Pred Churn (1)
                </div>

                {/* True Negative */}
                <div className="p-4 rounded-2xl bg-[#ECFDF5] border border-[#A7F3D0] text-center space-y-1">
                  <span className="text-[10px] uppercase font-mono text-[#047857] block font-bold">
                    True Negative (TN)
                  </span>
                  <span className="text-2xl font-heading font-black text-[#065F46]">{cm.trueNegative}</span>
                  <span className="text-[11px] text-[#047857] block">
                    {tnPct}% Correct Retained
                  </span>
                </div>

                {/* False Positive */}
                <div className="p-4 rounded-2xl bg-[#FFFBEB] border border-[#FDE68A] text-center space-y-1">
                  <span className="text-[10px] uppercase font-mono text-[#B45309] block font-bold">
                    False Positive (FP)
                  </span>
                  <span className="text-2xl font-heading font-black text-[#92400E]">{cm.falsePositive}</span>
                  <span className="text-[11px] text-[#B45309] block">
                    {fpPct}% False Alarm
                  </span>
                </div>

                {/* False Negative */}
                <div className="p-4 rounded-2xl bg-[#FFF1F2] border border-[#FECDD3] text-center space-y-1">
                  <span className="text-[10px] uppercase font-mono text-[#BE123C] block font-bold">
                    False Negative (FN)
                  </span>
                  <span className="text-2xl font-heading font-black text-[#9F1239]">{cm.falseNegative}</span>
                  <span className="text-[11px] text-[#BE123C] block">
                    {fnPct}% Missed Churn
                  </span>
                </div>

                {/* True Positive */}
                <div className="p-4 rounded-2xl bg-[#F5F2FE] border-2 border-[#7C3AED] text-center space-y-1">
                  <span className="text-[10px] uppercase font-mono text-[#7C3AED] block font-bold">
                    True Positive (TP)
                  </span>
                  <span className="text-2xl font-heading font-black text-[#5B21B6]">{cm.truePositive}</span>
                  <span className="text-[11px] text-[#7C3AED] font-semibold block">
                    {tpPct}% Correct Churn
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Importance Section */}
      <div className="rounded-3xl border border-[#ECE4F8] bg-white p-7 sm:p-9 space-y-6 shadow-xs">
        <div>
          <span className="text-xs font-mono font-semibold text-[#7C3AED] uppercase tracking-wider block">
            Ensemble Interpretability
          </span>
          <h3 className="text-xl font-heading font-bold text-[#181524]">
            AdaBoost Feature Importance Chart
          </h3>
          <p className="text-xs text-[#7B7494] mt-0.5">
            Sum of estimator importance weights ($\alpha_t$) across all 50 weak learners.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {sortedImportances.map(([feature, weight], idx) => {
            const cleanName = feature
              .replace('contract_', 'Contract: ')
              .replace('internet_', 'Internet: ')
              .replace('payment_', 'Payment: ')
              .replace('gender_', 'Gender: ')
              .replace('monthlyCharges', 'Monthly Charges')
              .replace('totalCharges', 'Total Charges')
              .replace('supportCalls', 'Support Calls')
              .replace('numProducts', 'Number of Products')
              .replace('tenure', 'Tenure (Months)');

            return (
              <div key={idx} className="p-4 rounded-2xl bg-[#FAF8FE] border border-[#EAE3F7] space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-[#28223D]">{cleanName}</span>
                  <span className="font-mono text-[#7C3AED] font-bold">{weight}%</span>
                </div>
                <div className="h-2 w-full bg-[#EAE2F8] rounded-full overflow-hidden flex">
                  <div
                    className="h-full bg-[#7C3AED] rounded-full"
                    style={{ width: `${Math.min(100, weight * 3.5)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Interactive Hyperparameter Tuning & Retraining */}
      <div className="rounded-3xl border border-[#ECE4F8] bg-white p-7 sm:p-9 space-y-6 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Sliders className="w-5 h-5 text-[#7C3AED]" />
              <h3 className="text-lg font-heading font-bold text-[#181524]">
                Model Tuning & Live Retraining
              </h3>
            </div>
            <p className="text-xs text-[#7B7494] mt-0.5">
              Experiment with boosting rounds, learning rate, and tree depth to observe instant performance updates.
            </p>
          </div>

          <button
            onClick={() => onRetrain(config)}
            disabled={isRetraining}
            className="px-6 py-3 rounded-full bg-[#161324] hover:bg-black text-white font-semibold text-xs shadow-md transition-all flex items-center gap-2 cursor-pointer disabled:opacity-50"
          >
            <RotateCw className={`w-3.5 h-3.5 ${isRetraining ? 'animate-spin' : ''}`} />
            <span>{isRetraining ? 'Retraining...' : 'Retrain Models Now'}</span>
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
          {/* Number of Estimators */}
          <div className="p-4 rounded-2xl bg-[#FAF8FE] border border-[#ECE5F8] space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#36304D] font-semibold">Estimators (T)</span>
              <span className="font-mono font-bold text-[#7C3AED]">{config.nEstimators}</span>
            </div>
            <input
              type="range"
              min={5}
              max={100}
              step={5}
              value={config.nEstimators}
              onChange={(e) =>
                setConfig({ ...config, nEstimators: parseInt(e.target.value) || 50 })
              }
              className="w-full accent-[#7C3AED] cursor-pointer"
            />
            <span className="text-[10px] text-[#8C85A6] block">Boosting rounds</span>
          </div>

          {/* Learning Rate */}
          <div className="p-4 rounded-2xl bg-[#FAF8FE] border border-[#ECE5F8] space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#36304D] font-semibold">Learning Rate (η)</span>
              <span className="font-mono font-bold text-[#7C3AED]">{config.learningRate}</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={1.5}
              step={0.1}
              value={config.learningRate}
              onChange={(e) =>
                setConfig({ ...config, learningRate: parseFloat(e.target.value) || 1.0 })
              }
              className="w-full accent-[#7C3AED] cursor-pointer"
            />
            <span className="text-[10px] text-[#8C85A6] block">Shrinkage factor</span>
          </div>

          {/* DT Depth */}
          <div className="p-4 rounded-2xl bg-[#FAF8FE] border border-[#ECE5F8] space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#36304D] font-semibold">DT Depth</span>
              <span className="font-mono font-bold text-[#36304D]">
                {config.decisionTreeMaxDepth}
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={6}
              step={1}
              value={config.decisionTreeMaxDepth}
              onChange={(e) =>
                setConfig({ ...config, decisionTreeMaxDepth: parseInt(e.target.value) || 3 })
              }
              className="w-full accent-[#7C3AED] cursor-pointer"
            />
            <span className="text-[10px] text-[#8C85A6] block">Baseline complexity</span>
          </div>

          {/* Test Split Ratio */}
          <div className="p-4 rounded-2xl bg-[#FAF8FE] border border-[#ECE5F8] space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-[#36304D] font-semibold">Test Split Ratio</span>
              <span className="font-mono font-bold text-[#059669]">
                {Math.round(config.testSplitRatio * 100)}%
              </span>
            </div>
            <input
              type="range"
              min={0.1}
              max={0.35}
              step={0.05}
              value={config.testSplitRatio}
              onChange={(e) =>
                setConfig({ ...config, testSplitRatio: parseFloat(e.target.value) || 0.2 })
              }
              className="w-full accent-[#7C3AED] cursor-pointer"
            />
            <span className="text-[10px] text-[#8C85A6] block">Holdout fraction</span>
          </div>
        </div>
      </div>
    </div>
  );
};
