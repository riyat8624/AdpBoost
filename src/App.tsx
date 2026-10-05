/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo, useCallback } from 'react';
import { Navbar } from './components/Navbar';
import { HomeSection } from './components/HomeSection';
import { PredictionSection } from './components/PredictionSection';
import { PerformanceSection } from './components/PerformanceSection';
import { AboutAdaBoostSection } from './components/AboutAdaBoostSection';
import { DatasetAndCodeSection } from './components/DatasetAndCodeSection';
import { generateCustomerDataset } from './ml/dataset';
import { preprocessDataset } from './ml/preprocessing';
import { AdaBoostClassifier, DecisionTreeClassifier } from './ml/models';
import { compareModels, ModelComparisonResult } from './ml/metrics';
import { CustomerRecord, PreprocessedData, TrainingConfig } from './types/ml';
import { Zap, Github, BookOpen } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<
    'home' | 'prediction' | 'performance' | 'about' | 'dataset'
  >('home');

  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    nEstimators: 50,
    learningRate: 1.0,
    decisionTreeMaxDepth: 3,
    testSplitRatio: 0.2,
    randomSeed: 42,
  });

  const [isRetraining, setIsRetraining] = useState(false);

  // Initialize dataset (1000 samples)
  const dataset: CustomerRecord[] = useMemo(() => {
    return generateCustomerDataset(1000, 42);
  }, []);

  // Preprocess data and train models
  const { preprocessedData, adaModel, dtModel, comparison } = useMemo(() => {
    const preprocessed = preprocessDataset(
      dataset,
      trainingConfig.testSplitRatio,
      trainingConfig.randomSeed
    );

    // 1. Fit Decision Tree Baseline
    const dt = new DecisionTreeClassifier(
      trainingConfig.decisionTreeMaxDepth,
      preprocessed.featureNames
    );
    dt.fit(preprocessed.XTrain, preprocessed.yTrain);
    const dtTestPreds = dt.predict(preprocessed.XTest);

    // 2. Fit AdaBoost Ensemble
    const ada = new AdaBoostClassifier(
      trainingConfig.nEstimators,
      trainingConfig.learningRate,
      preprocessed.featureNames
    );
    ada.fit(preprocessed.XTrain, preprocessed.yTrain);
    const adaTestPreds = ada.predict(preprocessed.XTest);

    // 3. Compute Metrics & Comparison
    const comp = compareModels(
      preprocessed.yTest,
      dtTestPreds,
      adaTestPreds,
      dataset.length,
      preprocessed.XTrain.length,
      preprocessed.featureNames.length
    );

    return {
      preprocessedData: preprocessed,
      adaModel: ada,
      dtModel: dt,
      comparison: comp,
    };
  }, [dataset, trainingConfig]);

  const handleRetrain = useCallback((newConfig: TrainingConfig) => {
    setIsRetraining(true);
    // Small timeout to allow UI to show spinning state
    setTimeout(() => {
      setTrainingConfig(newConfig);
      setIsRetraining(false);
    }, 200);
  }, []);

  return (
    <div className="min-h-screen bg-[#F8F6FD] text-[#1E1B2E] flex flex-col font-sans selection:bg-purple-200 selection:text-purple-900">
      {/* Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        adaAccuracy={comparison.adaBoost.accuracy}
        improvement={comparison.accuracyImprovement}
      />

      {/* Main Content Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'home' && (
          <HomeSection comparison={comparison} setActiveTab={setActiveTab} />
        )}

        {activeTab === 'prediction' && (
          <PredictionSection adaModel={adaModel} dtModel={dtModel} />
        )}

        {activeTab === 'performance' && (
          <PerformanceSection
            comparison={comparison}
            adaModel={adaModel}
            dtModel={dtModel}
            preprocessedData={preprocessedData}
            onRetrain={handleRetrain}
            isRetraining={isRetraining}
          />
        )}

        {activeTab === 'about' && <AboutAdaBoostSection />}

        {activeTab === 'dataset' && <DatasetAndCodeSection dataset={dataset} />}
      </main>

      {/* Footer */}
      <footer className="border-t border-[#ECE4F8] bg-white mt-12 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-[#7B7494]">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-md bg-[#7C3AED] text-white flex items-center justify-center">
              <Zap className="w-3 h-3 fill-white" />
            </div>
            <span className="font-heading font-bold text-[#181524]">
              Customer Classification using AdaBoost
            </span>
            <span>· College AI/ML Project</span>
          </div>

          <div className="flex items-center gap-4">
            <span className="font-mono text-[11px] text-[#8C85A6]">
              Scikit-Learn · AdaBoost.M1 (SAMME) · CART Decision Tree
            </span>
            <button
              onClick={() => setActiveTab('about')}
              className="text-[#7C3AED] font-semibold hover:text-[#6D28D9] transition-colors cursor-pointer"
            >
              Viva Defense Guide
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
