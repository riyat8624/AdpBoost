import { CustomerRecord, Gender, ContractType, InternetService, PaymentMethod } from '../types/ml';

// Seeded pseudorandom generator for deterministic reproducible data
function createRng(seed: number) {
  let s = seed;
  return function () {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

export function generateCustomerDataset(numSamples = 1000, seed = 42): CustomerRecord[] {
  const rng = createRng(seed);
  const data: CustomerRecord[] = [];

  const genders: Gender[] = ['Male', 'Female'];
  const contracts: ContractType[] = ['Month-to-month', 'One year', 'Two year'];
  const internetServices: InternetService[] = ['DSL', 'Fiber optic', 'No'];
  const paymentMethods: PaymentMethod[] = [
    'Electronic check',
    'Mailed check',
    'Bank transfer',
    'Credit card',
  ];

  for (let i = 1; i <= numSamples; i++) {
    const id = `CUST-${String(i).padStart(4, '0')}`;
    const gender = genders[Math.floor(rng() * genders.length)];
    const age = Math.floor(18 + rng() * 62); // 18 - 80

    // Contract distribution (55% month-to-month, 25% one-year, 20% two-year)
    const contractRoll = rng();
    const contract: ContractType =
      contractRoll < 0.55 ? 'Month-to-month' : contractRoll < 0.8 ? 'One year' : 'Two year';

    // Tenure (months): Month-to-month customers tend to have lower tenure
    let tenure: number;
    if (contract === 'Month-to-month') {
      tenure = Math.max(1, Math.floor(rng() * 24));
    } else if (contract === 'One year') {
      tenure = Math.floor(12 + rng() * 36);
    } else {
      tenure = Math.floor(24 + rng() * 48);
    }

    // Internet service
    const netRoll = rng();
    const internetService: InternetService =
      netRoll < 0.45 ? 'Fiber optic' : netRoll < 0.8 ? 'DSL' : 'No';

    // Products subscribed (1 - 6)
    const numProducts = Math.floor(1 + rng() * 5);

    // Monthly charges: Fiber optic has higher base rate
    let baseCharge = 20 + numProducts * 8;
    if (internetService === 'Fiber optic') baseCharge += 45;
    else if (internetService === 'DSL') baseCharge += 25;
    const monthlyCharges = Math.round((baseCharge + (rng() * 15 - 7.5)) * 10) / 10;

    // Total charges = tenure * monthlyCharges + small noise
    const totalCharges = Math.round(tenure * monthlyCharges * (0.95 + rng() * 0.1) * 10) / 10;

    // Payment method
    const payRoll = rng();
    const paymentMethod: PaymentMethod =
      payRoll < 0.35
        ? 'Electronic check'
        : payRoll < 0.6
        ? 'Credit card'
        : payRoll < 0.8
        ? 'Bank transfer'
        : 'Mailed check';

    // Customer support calls
    // Customers with fiber optic or month-to-month tend to call more often
    let lambda = 1.2;
    if (contract === 'Month-to-month') lambda += 1.0;
    if (internetService === 'Fiber optic') lambda += 0.8;
    if (paymentMethod === 'Electronic check') lambda += 0.4;
    const supportCalls = Math.min(9, Math.max(0, Math.floor(rng() * (lambda + 3))));

    // Determine Ground Truth Churn Risk (customerClass: 1 = Churn Risk, 0 = Retained)
    // Non-linear interaction score:
    let churnScore = 0;

    // Contract impact (heavy)
    if (contract === 'Month-to-month') churnScore += 2.2;
    else if (contract === 'One year') churnScore -= 1.0;
    else churnScore -= 2.5;

    // Tenure impact (exponential decay of churn probability)
    churnScore -= Math.min(3.0, (tenure / 12) * 0.7);

    // Support calls impact (very strong indicator of frustration)
    if (supportCalls >= 4) churnScore += 2.8;
    else if (supportCalls >= 2) churnScore += 1.2;
    else churnScore -= 0.6;

    // Monthly charges & Fiber optic combo
    if (internetService === 'Fiber optic' && monthlyCharges > 85) churnScore += 1.5;
    if (internetService === 'No') churnScore -= 1.2;

    // Payment method
    if (paymentMethod === 'Electronic check') churnScore += 0.8;
    else if (paymentMethod === 'Bank transfer' || paymentMethod === 'Credit card')
      churnScore -= 0.7;

    // Senior citizen / age effect
    if (age > 60) churnScore += 0.4;

    // Products / Service engagement
    if (numProducts >= 4) churnScore -= 0.9;
    else if (numProducts === 1) churnScore += 0.6;

    // Random noise to simulate real-world irreducible error (5-8%)
    const noise = (rng() - 0.5) * 1.4;
    const finalScore = churnScore + noise;

    // Threshold into binary class
    const customerClass: 0 | 1 = finalScore > 0.35 ? 1 : 0;

    data.push({
      id,
      age,
      gender,
      tenure,
      monthlyCharges,
      totalCharges,
      contract,
      internetService,
      supportCalls,
      paymentMethod,
      numProducts,
      customerClass,
    });
  }

  return data;
}

export function exportDatasetToCsv(dataset: CustomerRecord[]): string {
  const headers = [
    'customer_id',
    'age',
    'gender',
    'tenure_months',
    'monthly_charges',
    'total_charges',
    'contract_type',
    'internet_service',
    'support_calls',
    'payment_method',
    'num_products',
    'customer_class',
    'churn_status',
  ];

  const rows = dataset.map((d) => [
    d.id,
    d.age,
    d.gender,
    d.tenure,
    d.monthlyCharges,
    d.totalCharges,
    `"${d.contract}"`,
    `"${d.internetService}"`,
    d.supportCalls,
    `"${d.paymentMethod}"`,
    d.numProducts,
    d.customerClass,
    d.customerClass === 1 ? 'Churn Risk' : 'Retained',
  ]);

  return [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
}
