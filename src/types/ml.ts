export type Gender = 'Male' | 'Female';
export type ContractType = 'Month-to-month' | 'One year' | 'Two year';
export type InternetService = 'DSL' | 'Fiber optic' | 'No';
export type PaymentMethod = 'Electronic check' | 'Mailed check' | 'Bank transfer' | 'Credit card';

export interface CustomerRecord {
  id: string;
  age: number;
  gender: Gender;
  tenure: number; // months
  monthlyCharges: number; // $
  totalCharges: number; // $
  contract: ContractType;
  internetService: InternetService;
  supportCalls: number; // count
  paymentMethod: PaymentMethod;
  numProducts: number; // 1-6
  customerClass: 0 | 1; // 0 = Retained/Loyal, 1 = Churn Risk
}

export type CustomerInput = Omit<CustomerRecord, 'id' | 'customerClass'>;

export interface PreprocessedData {
  featureNames: string[];
  XTrain: number[][];
  yTrain: number[]; // -1 or 1 for AdaBoost, 0 or 1 for DT
  XTest: number[][];
  yTest: number[];
  trainOriginal: CustomerRecord[];
  testOriginal: CustomerRecord[];
  featureMins: number[];
  featureMaxs: number[];
  categoricalValues: Record<string, string[]>;
}

export interface DecisionStump {
  featureIndex: number;
  featureName: string;
  threshold: number;
  polarity: 1 | -1; // 1 means x[j] >= threshold -> +1, else -1; -1 means inverted
  alpha: number; // weight in ensemble
  error: number; // weighted error in training round
}

export interface AdaBoostModelData {
  estimators: DecisionStump[];
  featureNames: string[];
  learningRate: number;
  nEstimators: number;
  trainingIterations: {
    round: number;
    featureIndex: number;
    featureName: string;
    error: number;
    alpha: number;
    trainAccuracy: number;
  }[];
  featureImportances: Record<string, number>;
}

export interface DecisionTreeNode {
  featureIndex?: number;
  featureName?: string;
  threshold?: number;
  left?: DecisionTreeNode;
  right?: DecisionTreeNode;
  isLeaf: boolean;
  prediction?: 0 | 1;
  samples?: number;
  value?: [number, number]; // [count0, count1]
}

export interface DecisionTreeModelData {
  root: DecisionTreeNode;
  maxDepth: number;
  featureNames: string[];
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: {
    trueNegative: number;
    falsePositive: number;
    falseNegative: number;
    truePositive: number;
  };
}

export interface PredictionResult {
  predictedClass: 0 | 1;
  classLabel: string;
  confidence: number; // 0 - 100%
  rawScore: number;
  adaBoostClass: 0 | 1;
  decisionTreeClass: 0 | 1;
  dtConfidence: number;
  explanation: string[];
  featureContributions: {
    feature: string;
    impact: 'churn_risk' | 'retained' | 'neutral';
    description: string;
  }[];
}

export interface TrainingConfig {
  nEstimators: number;
  learningRate: number;
  decisionTreeMaxDepth: number;
  testSplitRatio: number;
  randomSeed: number;
}
