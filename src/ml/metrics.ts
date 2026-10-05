import { ModelMetrics } from '../types/ml';

export function calculateMetrics(yTrue: number[], yPred: (0 | 1)[]): ModelMetrics {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;

  for (let i = 0; i < yTrue.length; i++) {
    const trueVal = yTrue[i];
    const predVal = yPred[i];

    if (trueVal === 1 && predVal === 1) tp++;
    else if (trueVal === 0 && predVal === 1) fp++;
    else if (trueVal === 0 && predVal === 0) tn++;
    else if (trueVal === 1 && predVal === 0) fn++;
  }

  const total = tp + fp + tn + fn || 1;
  const accuracy = ((tp + tn) / total) * 100;

  const precision = tp + fp > 0 ? (tp / (tp + fp)) * 100 : 0;
  const recall = tp + fn > 0 ? (tp / (tp + fn)) * 100 : 0;

  const f1Score =
    precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return {
    accuracy: Math.round(accuracy * 10) / 10,
    precision: Math.round(precision * 10) / 10,
    recall: Math.round(recall * 10) / 10,
    f1Score: Math.round(f1Score * 10) / 10,
    confusionMatrix: {
      trueNegative: tn,
      falsePositive: fp,
      falseNegative: fn,
      truePositive: tp,
    },
  };
}

export interface ModelComparisonResult {
  decisionTree: ModelMetrics;
  adaBoost: ModelMetrics;
  accuracyImprovement: number; // AdaBoost - Decision Tree
  f1Improvement: number;
  precisionImprovement: number;
  recallImprovement: number;
  sampleCount: number;
  trainCount: number;
  testCount: number;
  featureCount: number;
}

export function compareModels(
  yTest: number[],
  dtPreds: (0 | 1)[],
  adaPreds: (0 | 1)[],
  totalSamples: number,
  trainCount: number,
  featureCount: number
): ModelComparisonResult {
  const dtMetrics = calculateMetrics(yTest, dtPreds);
  const adaMetrics = calculateMetrics(yTest, adaPreds);

  const accuracyImprovement =
    Math.round((adaMetrics.accuracy - dtMetrics.accuracy) * 10) / 10;
  const f1Improvement = Math.round((adaMetrics.f1Score - dtMetrics.f1Score) * 10) / 10;
  const precisionImprovement =
    Math.round((adaMetrics.precision - dtMetrics.precision) * 10) / 10;
  const recallImprovement =
    Math.round((adaMetrics.recall - dtMetrics.recall) * 10) / 10;

  return {
    decisionTree: dtMetrics,
    adaBoost: adaMetrics,
    accuracyImprovement,
    f1Improvement,
    precisionImprovement,
    recallImprovement,
    sampleCount: totalSamples,
    trainCount,
    testCount: yTest.length,
    featureCount,
  };
}
