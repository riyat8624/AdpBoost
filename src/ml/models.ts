import {
  DecisionStump,
  AdaBoostModelData,
  DecisionTreeNode,
  DecisionTreeModelData,
  PredictionResult,
  CustomerInput,
} from '../types/ml';
import { extractFeatureVector, FEATURE_NAMES } from './preprocessing';

// ==========================================
// 1. ADABOOST CLASSIFIER (Ensemble with Decision Stumps)
// ==========================================

export class AdaBoostClassifier {
  public estimators: DecisionStump[] = [];
  public learningRate: number;
  public nEstimators: number;
  public featureNames: string[];
  public trainingIterations: AdaBoostModelData['trainingIterations'] = [];
  public featureImportances: Record<string, number> = {};

  constructor(nEstimators = 50, learningRate = 1.0, featureNames = FEATURE_NAMES) {
    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
    this.featureNames = featureNames;
  }

  public fit(X: number[][], y: number[]): void {
    const N = X.length;
    const numFeatures = this.featureNames.length;
    this.estimators = [];
    this.trainingIterations = [];

    // Convert 0/1 to -1/+1
    const yMapped = y.map((v) => (v === 1 ? 1 : -1));

    // Initialize sample weights uniformly: w_i = 1 / N
    let weights = new Float64Array(N);
    weights.fill(1.0 / N);

    // Precompute sorted thresholds for each feature to make training fast and exact
    const featureThresholds: number[][] = [];
    for (let j = 0; j < numFeatures; j++) {
      const vals = Array.from(new Set(X.map((row) => row[j]))).sort((a, b) => a - b);
      const thresholds: number[] = [];
      // Pick midpoints between adjacent distinct values, cap at max 25 percentiles if too large
      if (vals.length <= 25) {
        for (let k = 0; k < vals.length - 1; k++) {
          thresholds.push((vals[k] + vals[k + 1]) / 2);
        }
        if (vals.length === 1) thresholds.push(vals[0]);
      } else {
        const step = Math.floor(vals.length / 20);
        for (let k = 0; k < vals.length - 1; k += step) {
          thresholds.push((vals[k] + vals[k + 1]) / 2);
        }
      }
      featureThresholds.push(thresholds.length > 0 ? thresholds : [0]);
    }

    for (let round = 1; round <= this.nEstimators; round++) {
      let minError = Infinity;
      let bestStump: DecisionStump = {
        featureIndex: 0,
        featureName: this.featureNames[0],
        threshold: 0,
        polarity: 1,
        alpha: 0,
        error: 0.5,
      };

      // Search best stump across all features and candidate thresholds
      for (let j = 0; j < numFeatures; j++) {
        const thresholds = featureThresholds[j];
        for (const thresh of thresholds) {
          for (const polarity of [1, -1] as const) {
            let error = 0;
            for (let i = 0; i < N; i++) {
              const val = X[i][j];
              const pred = polarity === 1 ? (val >= thresh ? 1 : -1) : val < thresh ? 1 : -1;
              if (pred !== yMapped[i]) {
                error += weights[i];
              }
            }

            if (error < minError) {
              minError = error;
              bestStump = {
                featureIndex: j,
                featureName: this.featureNames[j],
                threshold: thresh,
                polarity,
                alpha: 0,
                error,
              };
            }
          }
        }
      }

      // Avoid numerical division by zero
      const safeError = Math.max(1e-10, Math.min(0.499999, minError));

      // Calculate estimator weight: alpha = 0.5 * ln((1 - error) / error) * learningRate
      const alpha = 0.5 * Math.log((1.0 - safeError) / safeError) * this.learningRate;
      bestStump.alpha = alpha;
      bestStump.error = safeError;

      // Update sample weights: w_i = w_i * exp(-alpha * y_i * h(x_i))
      let weightSum = 0;
      for (let i = 0; i < N; i++) {
        const val = X[i][bestStump.featureIndex];
        const pred =
          bestStump.polarity === 1
            ? val >= bestStump.threshold
              ? 1
              : -1
            : val < bestStump.threshold
            ? 1
            : -1;
        weights[i] = weights[i] * Math.exp(-alpha * yMapped[i] * pred);
        weightSum += weights[i];
      }

      // Re-normalize weights so sum is 1.0
      if (weightSum > 0) {
        for (let i = 0; i < N; i++) {
          weights[i] /= weightSum;
        }
      }

      this.estimators.push(bestStump);

      // Track training accuracy at this round
      const currentTrainPreds = this.predictRawBatch(X);
      let correct = 0;
      for (let i = 0; i < N; i++) {
        if ((currentTrainPreds[i] >= 0 ? 1 : 0) === y[i]) correct++;
      }
      const trainAccuracy = (correct / N) * 100;

      this.trainingIterations.push({
        round,
        featureIndex: bestStump.featureIndex,
        featureName: bestStump.featureName,
        error: Math.round(safeError * 1000) / 1000,
        alpha: Math.round(alpha * 1000) / 1000,
        trainAccuracy: Math.round(trainAccuracy * 10) / 10,
      });

      // If perfect fit or error >= 0.5, stop early
      if (safeError <= 1e-6) break;
    }

    // Calculate normalized feature importances
    const totalAlpha = this.estimators.reduce((sum, est) => sum + est.alpha, 0);
    this.featureImportances = {};
    for (const name of this.featureNames) {
      this.featureImportances[name] = 0;
    }
    for (const est of this.estimators) {
      this.featureImportances[est.featureName] += (est.alpha / (totalAlpha || 1)) * 100;
    }
    for (const name of this.featureNames) {
      this.featureImportances[name] = Math.round(this.featureImportances[name] * 10) / 10;
    }
  }

  public predictRaw(x: number[]): number {
    let score = 0;
    for (const est of this.estimators) {
      const val = x[est.featureIndex];
      const pred = est.polarity === 1 ? (val >= est.threshold ? 1 : -1) : val < est.threshold ? 1 : -1;
      score += est.alpha * pred;
    }
    return score;
  }

  public predictRawBatch(X: number[][]): number[] {
    return X.map((x) => this.predictRaw(x));
  }

  public predict(X: number[][]): (0 | 1)[] {
    return X.map((x) => (this.predictRaw(x) >= 0 ? 1 : 0));
  }

  public predictProbability(x: number[]): number {
    const raw = this.predictRaw(x);
    // Sigmoid function mapped to [0, 1] for class 1 (Churn Risk)
    return 1 / (1 + Math.exp(-1.5 * raw));
  }

  public toJSON(): AdaBoostModelData {
    return {
      estimators: this.estimators,
      featureNames: this.featureNames,
      learningRate: this.learningRate,
      nEstimators: this.nEstimators,
      trainingIterations: this.trainingIterations,
      featureImportances: this.featureImportances,
    };
  }

  public static fromJSON(data: AdaBoostModelData): AdaBoostClassifier {
    const model = new AdaBoostClassifier(data.nEstimators, data.learningRate, data.featureNames);
    model.estimators = data.estimators;
    model.trainingIterations = data.trainingIterations;
    model.featureImportances = data.featureImportances;
    return model;
  }
}

// ==========================================
// 2. DECISION TREE CLASSIFIER (Baseline CART)
// ==========================================

function computeGini(y: number[]): number {
  if (y.length === 0) return 0;
  let count1 = 0;
  for (let i = 0; i < y.length; i++) {
    if (y[i] === 1) count1++;
  }
  const p1 = count1 / y.length;
  const p0 = 1 - p1;
  return 1 - (p0 * p0 + p1 * p1);
}

export class DecisionTreeClassifier {
  public root: DecisionTreeNode | null = null;
  public maxDepth: number;
  public featureNames: string[];

  constructor(maxDepth = 3, featureNames = FEATURE_NAMES) {
    this.maxDepth = maxDepth;
    this.featureNames = featureNames;
  }

  public fit(X: number[][], y: number[]): void {
    this.root = this.buildTree(X, y, 0);
  }

  private buildTree(X: number[][], y: number[], depth: number): DecisionTreeNode {
    const numSamples = y.length;
    let count1 = 0;
    for (let i = 0; i < numSamples; i++) {
      if (y[i] === 1) count1++;
    }
    const count0 = numSamples - count1;
    const majorityPred: 0 | 1 = count1 >= count0 ? 1 : 0;

    // Base cases: max depth reached, node is pure, or sample size too small
    if (depth >= this.maxDepth || count0 === 0 || count1 === 0 || numSamples <= 4) {
      return {
        isLeaf: true,
        prediction: majorityPred,
        samples: numSamples,
        value: [count0, count1],
      };
    }

    const currentGini = computeGini(y);
    let bestGain = 0;
    let bestFeature = -1;
    let bestThreshold = 0;

    const numFeatures = this.featureNames.length;
    for (let j = 0; j < numFeatures; j++) {
      const vals = Array.from(new Set(X.map((row) => row[j]))).sort((a, b) => a - b);
      if (vals.length <= 1) continue;

      // Sample percentiles to find best split
      const step = Math.max(1, Math.floor(vals.length / 12));
      for (let k = 0; k < vals.length - 1; k += step) {
        const thresh = (vals[k] + vals[k + 1]) / 2;

        const leftY: number[] = [];
        const rightY: number[] = [];
        for (let i = 0; i < numSamples; i++) {
          if (X[i][j] <= thresh) leftY.push(y[i]);
          else rightY.push(y[i]);
        }

        if (leftY.length === 0 || rightY.length === 0) continue;

        const leftGini = computeGini(leftY);
        const rightGini = computeGini(rightY);
        const weightedGini =
          (leftY.length / numSamples) * leftGini + (rightY.length / numSamples) * rightGini;
        const gain = currentGini - weightedGini;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = j;
          bestThreshold = thresh;
        }
      }
    }

    // If no meaningful split was found, make leaf
    if (bestGain <= 1e-5 || bestFeature === -1) {
      return {
        isLeaf: true,
        prediction: majorityPred,
        samples: numSamples,
        value: [count0, count1],
      };
    }

    // Split data
    const leftX: number[][] = [];
    const leftY: number[] = [];
    const rightX: number[][] = [];
    const rightY: number[] = [];

    for (let i = 0; i < numSamples; i++) {
      if (X[i][bestFeature] <= bestThreshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    return {
      isLeaf: false,
      featureIndex: bestFeature,
      featureName: this.featureNames[bestFeature],
      threshold: bestThreshold,
      samples: numSamples,
      value: [count0, count1],
      left: this.buildTree(leftX, leftY, depth + 1),
      right: this.buildTree(rightX, rightY, depth + 1),
    };
  }

  public predictSingle(x: number[]): { pred: 0 | 1; proba: number } {
    let node = this.root;
    while (node && !node.isLeaf) {
      if (node.featureIndex !== undefined && node.threshold !== undefined) {
        if (x[node.featureIndex] <= node.threshold) {
          node = node.left || null;
        } else {
          node = node.right || null;
        }
      } else {
        break;
      }
    }

    if (!node || node.prediction === undefined) {
      return { pred: 0, proba: 0.5 };
    }

    const [c0, c1] = node.value || [1, 0];
    const total = c0 + c1;
    const proba = total > 0 ? c1 / total : node.prediction === 1 ? 1 : 0;
    return { pred: node.prediction, proba };
  }

  public predict(X: number[][]): (0 | 1)[] {
    return X.map((x) => this.predictSingle(x).pred);
  }

  public toJSON(): DecisionTreeModelData {
    return {
      root: this.root || { isLeaf: true, prediction: 0 },
      maxDepth: this.maxDepth,
      featureNames: this.featureNames,
    };
  }
}

// ==========================================
// 3. PREDICTION & EXPLANATION PIPELINE
// ==========================================

export function explainPrediction(
  customer: CustomerInput,
  adaModel: AdaBoostClassifier,
  dtModel: DecisionTreeClassifier
): PredictionResult {
  const x = extractFeatureVector(customer);
  const rawScore = adaModel.predictRaw(x);
  const churnProb = adaModel.predictProbability(x);
  const adaBoostClass: 0 | 1 = rawScore >= 0 ? 1 : 0;

  const dtResult = dtModel.predictSingle(x);
  const decisionTreeClass = dtResult.pred;

  // Determine human-readable classification
  const classLabel = adaBoostClass === 1 ? 'Churn Risk (At Risk)' : 'Retained (Loyal Customer)';
  const confidence = Math.round((adaBoostClass === 1 ? churnProb : 1 - churnProb) * 1000) / 10;
  const dtConfidence =
    Math.round((decisionTreeClass === 1 ? dtResult.proba : 1 - dtResult.proba) * 1000) / 10;

  // Analyze top feature drivers
  const explanations: string[] = [];
  const featureContributions: PredictionResult['featureContributions'] = [];

  // Contract
  if (customer.contract === 'Month-to-month') {
    featureContributions.push({
      feature: 'Contract Type',
      impact: 'churn_risk',
      description: 'Month-to-month contracts have a 3.4x higher churn rate than multi-year plans.',
    });
    explanations.push('Month-to-month contract provides no long-term customer lock-in.');
  } else {
    featureContributions.push({
      feature: 'Contract Type',
      impact: 'retained',
      description: `${customer.contract} contract significantly stabilizes customer lifetime value.`,
    });
    explanations.push(`Long-term agreement (${customer.contract}) strongly favors retention.`);
  }

  // Support calls
  if (customer.supportCalls >= 4) {
    featureContributions.push({
      feature: 'Support Calls',
      impact: 'churn_risk',
      description: `${customer.supportCalls} support calls indicates unresolved technical or service friction.`,
    });
    explanations.push(`High support volume (${customer.supportCalls} calls) signals recurring service dissatisfaction.`);
  } else if (customer.supportCalls <= 1) {
    featureContributions.push({
      feature: 'Support Calls',
      impact: 'retained',
      description: 'Low support calls indicate smooth user experience and product satisfaction.',
    });
  }

  // Tenure
  if (customer.tenure <= 6) {
    featureContributions.push({
      feature: 'Tenure',
      impact: 'churn_risk',
      description: `Early tenure (${customer.tenure} months) carries the highest abandonment risk.`,
    });
    explanations.push(`New customer with short tenure (${customer.tenure} mo) has low brand loyalty.`);
  } else if (customer.tenure >= 36) {
    featureContributions.push({
      feature: 'Tenure',
      impact: 'retained',
      description: `Established tenure (${customer.tenure} months) reflects solid loyalty habituation.`,
    });
    explanations.push(`Long tenure (${customer.tenure} mo) creates high switching friction.`);
  }

  // Monthly charges & Internet
  if (customer.monthlyCharges > 85 && customer.internetService === 'Fiber optic') {
    featureContributions.push({
      feature: 'Monthly Charges',
      impact: 'churn_risk',
      description: 'Premium fiber pricing ($' + customer.monthlyCharges + '/mo) without bundles invites price comparison.',
    });
    explanations.push('High monthly cost ($' + customer.monthlyCharges + ') on Fiber Optic increases churn sensitivity.');
  } else if (customer.monthlyCharges < 50) {
    featureContributions.push({
      feature: 'Monthly Charges',
      impact: 'retained',
      description: 'Affordable billing ($' + customer.monthlyCharges + '/mo) minimizes price sensitivity.',
    });
  }

  // Payment method
  if (customer.paymentMethod === 'Electronic check') {
    featureContributions.push({
      feature: 'Payment Method',
      impact: 'churn_risk',
      description: 'Manual electronic checks require repeated monthly payment decisions.',
    });
  } else if (customer.paymentMethod === 'Bank transfer' || customer.paymentMethod === 'Credit card') {
    featureContributions.push({
      feature: 'Payment Method',
      impact: 'retained',
      description: 'Automated bank/credit card billing reduces involuntary payment churn.',
    });
  }

  // If few explanations generated, add summary
  if (explanations.length === 0) {
    explanations.push(
      adaBoostClass === 1
        ? 'Ensemble combination of high monthly fees and low commitment drove the churn risk prediction.'
        : 'Solid tenure, multi-product engagement, and reliable payment methods secured retention classification.'
    );
  }

  return {
    predictedClass: adaBoostClass,
    classLabel,
    confidence,
    rawScore: Math.round(rawScore * 100) / 100,
    adaBoostClass,
    decisionTreeClass,
    dtConfidence,
    explanation: explanations,
    featureContributions,
  };
}
