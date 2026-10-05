import {
  CustomerRecord,
  CustomerInput,
  PreprocessedData,
} from '../types/ml';

export const FEATURE_NAMES = [
  'age',
  'gender_Male',
  'tenure',
  'monthlyCharges',
  'totalCharges',
  'contract_MonthToMonth',
  'contract_OneYear',
  'contract_TwoYear',
  'internet_DSL',
  'internet_FiberOptic',
  'internet_No',
  'supportCalls',
  'payment_ElectronicCheck',
  'payment_MailedCheck',
  'payment_BankTransfer',
  'payment_CreditCard',
  'numProducts',
];

export function extractFeatureVector(
  input: CustomerInput,
  medians?: { totalCharges: number; monthlyCharges: number }
): number[] {
  const gender_Male = input.gender === 'Male' ? 1 : 0;
  const contract_MonthToMonth = input.contract === 'Month-to-month' ? 1 : 0;
  const contract_OneYear = input.contract === 'One year' ? 1 : 0;
  const contract_TwoYear = input.contract === 'Two year' ? 1 : 0;

  const internet_DSL = input.internetService === 'DSL' ? 1 : 0;
  const internet_FiberOptic = input.internetService === 'Fiber optic' ? 1 : 0;
  const internet_No = input.internetService === 'No' ? 1 : 0;

  const payment_ElectronicCheck = input.paymentMethod === 'Electronic check' ? 1 : 0;
  const payment_MailedCheck = input.paymentMethod === 'Mailed check' ? 1 : 0;
  const payment_BankTransfer = input.paymentMethod === 'Bank transfer' ? 1 : 0;
  const payment_CreditCard = input.paymentMethod === 'Credit card' ? 1 : 0;

  // Impute if null / undefined / NaN
  const safeAge = Number.isFinite(input.age) ? input.age : 40;
  const safeTenure = Number.isFinite(input.tenure) ? input.tenure : 24;
  const safeMonthly = Number.isFinite(input.monthlyCharges)
    ? input.monthlyCharges
    : medians?.monthlyCharges ?? 65;
  const safeTotal = Number.isFinite(input.totalCharges)
    ? input.totalCharges
    : (medians?.totalCharges ?? safeTenure * safeMonthly);
  const safeSupport = Number.isFinite(input.supportCalls) ? input.supportCalls : 1;
  const safeProducts = Number.isFinite(input.numProducts) ? input.numProducts : 2;

  return [
    safeAge,
    gender_Male,
    safeTenure,
    safeMonthly,
    safeTotal,
    contract_MonthToMonth,
    contract_OneYear,
    contract_TwoYear,
    internet_DSL,
    internet_FiberOptic,
    internet_No,
    safeSupport,
    payment_ElectronicCheck,
    payment_MailedCheck,
    payment_BankTransfer,
    payment_CreditCard,
    safeProducts,
  ];
}

export function preprocessDataset(
  dataset: CustomerRecord[],
  testRatio = 0.2,
  seed = 42
): PreprocessedData {
  // Stratified split into positive (class 1) and negative (class 0)
  const class0 = dataset.filter((d) => d.customerClass === 0);
  const class1 = dataset.filter((d) => d.customerClass === 1);

  // Deterministic shuffle
  const shuffle = <T>(array: T[], s: number): T[] => {
    const arr = [...array];
    let curSeed = s;
    for (let i = arr.length - 1; i > 0; i--) {
      curSeed = (curSeed * 9301 + 49297) % 233280;
      const j = Math.floor((curSeed / 233280) * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  };

  const shuf0 = shuffle(class0, seed);
  const shuf1 = shuffle(class1, seed + 1);

  const testCount0 = Math.floor(shuf0.length * testRatio);
  const testCount1 = Math.floor(shuf1.length * testRatio);

  const testOriginal = [...shuf0.slice(0, testCount0), ...shuf1.slice(0, testCount1)];
  const trainOriginal = [...shuf0.slice(testCount0), ...shuf1.slice(testCount1)];

  // Compute medians for imputation
  const totalChargesVals = trainOriginal.map((d) => d.totalCharges).sort((a, b) => a - b);
  const monthlyChargesVals = trainOriginal.map((d) => d.monthlyCharges).sort((a, b) => a - b);
  const medians = {
    totalCharges: totalChargesVals[Math.floor(totalChargesVals.length / 2)] || 1500,
    monthlyCharges: monthlyChargesVals[Math.floor(monthlyChargesVals.length / 2)] || 65,
  };

  const XTrain = trainOriginal.map((d) => extractFeatureVector(d, medians));
  const yTrain = trainOriginal.map((d) => d.customerClass);

  const XTest = testOriginal.map((d) => extractFeatureVector(d, medians));
  const yTest = testOriginal.map((d) => d.customerClass);

  // Calculate min and max per feature
  const numFeatures = FEATURE_NAMES.length;
  const featureMins = new Array(numFeatures).fill(Infinity);
  const featureMaxs = new Array(numFeatures).fill(-Infinity);

  for (let j = 0; j < numFeatures; j++) {
    for (let i = 0; i < XTrain.length; i++) {
      if (XTrain[i][j] < featureMins[j]) featureMins[j] = XTrain[i][j];
      if (XTrain[i][j] > featureMaxs[j]) featureMaxs[j] = XTrain[i][j];
    }
  }

  return {
    featureNames: FEATURE_NAMES,
    XTrain,
    yTrain,
    XTest,
    yTest,
    trainOriginal,
    testOriginal,
    featureMins,
    featureMaxs,
    categoricalValues: {
      gender: ['Male', 'Female'],
      contract: ['Month-to-month', 'One year', 'Two year'],
      internetService: ['DSL', 'Fiber optic', 'No'],
      paymentMethod: ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
    },
  };
}
