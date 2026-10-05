import React, { useState } from 'react';
import {
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  Sliders,
  TrendingDown,
  Zap,
  ArrowRight,
  ShieldAlert,
  Award,
} from 'lucide-react';
import {
  CustomerInput,
  Gender,
  ContractType,
  InternetService,
  PaymentMethod,
  PredictionResult,
} from '../types/ml';
import { AdaBoostClassifier, DecisionTreeClassifier, explainPrediction } from '../ml/models';

interface PredictionSectionProps {
  adaModel: AdaBoostClassifier;
  dtModel: DecisionTreeClassifier;
}

export const PredictionSection: React.FC<PredictionSectionProps> = ({ adaModel, dtModel }) => {
  const [input, setInput] = useState<CustomerInput>({
    age: 42,
    gender: 'Female',
    tenure: 8,
    monthlyCharges: 85.5,
    totalCharges: 684.0,
    contract: 'Month-to-month',
    internetService: 'Fiber optic',
    supportCalls: 4,
    paymentMethod: 'Electronic check',
    numProducts: 2,
  });

  const [autoSyncTotal, setAutoSyncTotal] = useState(true);
  const [prediction, setPrediction] = useState<PredictionResult | null>(() =>
    explainPrediction(input, adaModel, dtModel)
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const loadPreset = (presetKey: 'high_churn' | 'loyal' | 'new_customer' | 'boundary') => {
    let preset: CustomerInput;
    if (presetKey === 'high_churn') {
      preset = {
        age: 38,
        gender: 'Female',
        tenure: 4,
        monthlyCharges: 98.5,
        totalCharges: 394.0,
        contract: 'Month-to-month',
        internetService: 'Fiber optic',
        supportCalls: 5,
        paymentMethod: 'Electronic check',
        numProducts: 1,
      };
    } else if (presetKey === 'loyal') {
      preset = {
        age: 54,
        gender: 'Male',
        tenure: 52,
        monthlyCharges: 55.0,
        totalCharges: 2860.0,
        contract: 'Two year',
        internetService: 'DSL',
        supportCalls: 1,
        paymentMethod: 'Bank transfer',
        numProducts: 4,
      };
    } else if (presetKey === 'new_customer') {
      preset = {
        age: 26,
        gender: 'Male',
        tenure: 2,
        monthlyCharges: 65.0,
        totalCharges: 130.0,
        contract: 'Month-to-month',
        internetService: 'Fiber optic',
        supportCalls: 2,
        paymentMethod: 'Credit card',
        numProducts: 2,
      };
    } else {
      preset = {
        age: 45,
        gender: 'Female',
        tenure: 18,
        monthlyCharges: 79.0,
        totalCharges: 1422.0,
        contract: 'One year',
        internetService: 'Fiber optic',
        supportCalls: 3,
        paymentMethod: 'Electronic check',
        numProducts: 3,
      };
    }

    setInput(preset);
    setErrorMessage(null);
    setPrediction(explainPrediction(preset, adaModel, dtModel));
  };

  const handlePredict = (e: React.FormEvent) => {
    e.preventDefault();

    if (isNaN(input.age) || input.age < 18 || input.age > 100) {
      setErrorMessage('Age must be a valid number between 18 and 100.');
      return;
    }
    if (isNaN(input.tenure) || input.tenure < 1 || input.tenure > 72) {
      setErrorMessage('Tenure must be between 1 and 72 months.');
      return;
    }
    if (isNaN(input.monthlyCharges) || input.monthlyCharges < 10 || input.monthlyCharges > 300) {
      setErrorMessage('Monthly charges must be between $10 and $300.');
      return;
    }
    if (isNaN(input.totalCharges) || input.totalCharges < 0) {
      setErrorMessage('Total charges cannot be negative.');
      return;
    }
    if (isNaN(input.supportCalls) || input.supportCalls < 0 || input.supportCalls > 15) {
      setErrorMessage('Support calls must be between 0 and 15.');
      return;
    }

    setErrorMessage(null);
    const result = explainPrediction(input, adaModel, dtModel);
    setPrediction(result);
  };

  return (
    <div className="space-y-8 py-4">
      {/* Header */}
      <div>
        <span className="text-xs font-mono font-semibold text-[#7C3AED] uppercase tracking-wider block">
          Inference Dashboard
        </span>
        <h2 className="text-3xl sm:text-4xl font-heading font-black text-[#151226]">
          Customer Classification
        </h2>
        <p className="text-sm text-[#6A6384] mt-1 max-w-2xl">
          Enter customer metrics or select a scenario to evaluate real-time classification using the trained AdaBoost model.
        </p>
      </div>

      {/* Preset Pills (Clean pill cards in lavender palette) */}
      <div className="flex flex-wrap items-center gap-2 p-3 rounded-2xl bg-white border border-[#ECE5F8] shadow-xs">
        <span className="text-xs font-semibold text-[#575071] flex items-center gap-1.5 mr-2">
          <Sliders className="w-3.5 h-3.5 text-[#7C3AED]" />
          Presets:
        </span>
        <button
          type="button"
          onClick={() => loadPreset('high_churn')}
          className="text-xs px-3.5 py-1.5 rounded-full bg-[#FFF1F2] hover:bg-[#FFE4E6] text-[#BE123C] border border-[#FECDD3] font-medium transition-colors cursor-pointer"
        >
          ⚠️ High Churn Risk (Month-to-month, 5 Calls)
        </button>
        <button
          type="button"
          onClick={() => loadPreset('loyal')}
          className="text-xs px-3.5 py-1.5 rounded-full bg-[#ECFDF5] hover:bg-[#D1FAE5] text-[#047857] border border-[#A7F3D0] font-medium transition-colors cursor-pointer"
        >
          ✅ Loyal Retained (2-Year, 52 mo, Auto-pay)
        </button>
        <button
          type="button"
          onClick={() => loadPreset('new_customer')}
          className="text-xs px-3.5 py-1.5 rounded-full bg-[#F5F3FF] hover:bg-[#EDE9FE] text-[#6D28D9] border border-[#DDD6FE] font-medium transition-colors cursor-pointer"
        >
          🆕 New Customer (2 mo)
        </button>
        <button
          type="button"
          onClick={() => loadPreset('boundary')}
          className="text-xs px-3.5 py-1.5 rounded-full bg-[#F3EEFD] hover:bg-[#EAE2FB] text-[#7C3AED] border border-[#DDD0F8] font-medium transition-colors cursor-pointer"
        >
          ⚖️ Boundary Edge-Case
        </button>
      </div>

      {/* Main Grid: Form on Left, Result on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Form Container */}
        <form
          onSubmit={handlePredict}
          className="lg:col-span-7 rounded-3xl border border-[#ECE5F8] bg-white p-6 sm:p-8 space-y-6 shadow-xs"
        >
          <div className="flex items-center justify-between pb-3 border-b border-[#F0EAF8]">
            <h3 className="font-heading font-bold text-[#181524] text-base flex items-center gap-2">
              <div className="w-6 h-6 rounded-lg bg-[#EFEAFF] text-[#7C3AED] flex items-center justify-center">
                <Zap className="w-3.5 h-3.5 fill-[#7C3AED]" />
              </div>
              <span>Customer Information</span>
            </h3>
            <span className="text-[11px] text-[#7C3AED] font-mono font-semibold bg-[#F5F2FE] px-2.5 py-0.5 rounded-full">
              10 Model Features
            </span>
          </div>

          {errorMessage && (
            <div className="p-3.5 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0 text-rose-600" />
              <span>{errorMessage}</span>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Age */}
            <div>
              <label className="block text-xs font-semibold text-[#443D5C] mb-1.5">
                Age (Years)
              </label>
              <input
                type="number"
                min={18}
                max={100}
                value={input.age}
                onChange={(e) => setInput({ ...input, age: parseInt(e.target.value) || 0 })}
                className="w-full bg-[#FAF8FE] border border-[#DDD3F6] rounded-xl px-3.5 py-2.5 text-sm text-[#181524] focus:bg-white focus:outline-none focus:border-[#7C3AED] focus:ring-2 focus:ring-[#7C3AED]/20 transition-all"
              />
            </div>

            {/* Gender */}
            <div>
              <label className="block text-xs font-semibold text-[#443D5C] mb-1.5">Gender</label>
              <select
                value={input.gender}
                onChange={(e) => setInput({ ...input, gender: e.target.value as Gender })}
                className="w-full bg-[#FAF8FE] border border-[#DDD3F6] rounded-xl px-3.5 py-2.5 text-sm text-[#181524] focus:bg-white focus:outline-none focus:border-[#7C3AED] transition-all"
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
              </select>
            </div>

            {/* Tenure */}
            <div>
              <label className="block text-xs font-semibold text-[#443D5C] mb-1.5">
                Tenure (Months)
              </label>
              <input
                type="number"
                min={1}
                max={72}
                value={input.tenure}
                onChange={(e) => {
                  const t = parseInt(e.target.value) || 1;
                  setInput({
                    ...input,
                    tenure: t,
                    totalCharges: autoSyncTotal
                      ? Math.round(t * input.monthlyCharges * 10) / 10
                      : input.totalCharges,
                  });
                }}
                className="w-full bg-[#FAF8FE] border border-[#DDD3F6] rounded-xl px-3.5 py-2.5 text-sm text-[#181524] focus:bg-white focus:outline-none focus:border-[#7C3AED] transition-all"
              />
            </div>

            {/* Contract Type */}
            <div>
              <label className="block text-xs font-semibold text-[#443D5C] mb-1.5">
                Contract Type
              </label>
              <select
                value={input.contract}
                onChange={(e) =>
                  setInput({ ...input, contract: e.target.value as ContractType })
                }
                className="w-full bg-[#FAF8FE] border border-[#DDD3F6] rounded-xl px-3.5 py-2.5 text-sm text-[#181524] focus:bg-white focus:outline-none focus:border-[#7C3AED] transition-all"
              >
                <option value="Month-to-month">Month-to-month</option>
                <option value="One year">One year</option>
                <option value="Two year">Two year</option>
              </select>
            </div>

            {/* Internet Service */}
            <div>
              <label className="block text-xs font-semibold text-[#443D5C] mb-1.5">
                Internet Service
              </label>
              <select
                value={input.internetService}
                onChange={(e) =>
                  setInput({ ...input, internetService: e.target.value as InternetService })
                }
                className="w-full bg-[#FAF8FE] border border-[#DDD3F6] rounded-xl px-3.5 py-2.5 text-sm text-[#181524] focus:bg-white focus:outline-none focus:border-[#7C3AED] transition-all"
              >
                <option value="Fiber optic">Fiber optic</option>
                <option value="DSL">DSL</option>
                <option value="No">No Internet Service</option>
              </select>
            </div>

            {/* Support Calls */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-[#443D5C]">Support Calls (Past 6 mo)</label>
                <span className="text-xs font-mono font-bold text-[#7C3AED]">
                  {input.supportCalls} calls
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={9}
                value={input.supportCalls}
                onChange={(e) =>
                  setInput({ ...input, supportCalls: parseInt(e.target.value) || 0 })
                }
                className="w-full accent-[#7C3AED] cursor-pointer"
              />
            </div>

            {/* Monthly Charges */}
            <div>
              <label className="block text-xs font-semibold text-[#443D5C] mb-1.5">
                Monthly Charges ($)
              </label>
              <input
                type="number"
                step="0.5"
                min={15}
                max={200}
                value={input.monthlyCharges}
                onChange={(e) => {
                  const m = parseFloat(e.target.value) || 0;
                  setInput({
                    ...input,
                    monthlyCharges: m,
                    totalCharges: autoSyncTotal
                      ? Math.round(input.tenure * m * 10) / 10
                      : input.totalCharges,
                  });
                }}
                className="w-full bg-[#FAF8FE] border border-[#DDD3F6] rounded-xl px-3.5 py-2.5 text-sm text-[#181524] focus:bg-white focus:outline-none focus:border-[#7C3AED] transition-all"
              />
            </div>

            {/* Total Charges */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-[#443D5C]">Total Charges ($)</label>
                <button
                  type="button"
                  onClick={() => setAutoSyncTotal(!autoSyncTotal)}
                  className="text-[10px] text-[#7C3AED] font-medium"
                >
                  {autoSyncTotal ? '✓ Auto (tenure×rate)' : 'Custom'}
                </button>
              </div>
              <input
                type="number"
                step="1"
                min={0}
                max={10000}
                value={input.totalCharges}
                onChange={(e) => {
                  setAutoSyncTotal(false);
                  setInput({ ...input, totalCharges: parseFloat(e.target.value) || 0 });
                }}
                className="w-full bg-[#FAF8FE] border border-[#DDD3F6] rounded-xl px-3.5 py-2.5 text-sm text-[#181524] focus:bg-white focus:outline-none focus:border-[#7C3AED] transition-all"
              />
            </div>

            {/* Payment Method */}
            <div>
              <label className="block text-xs font-semibold text-[#443D5C] mb-1.5">
                Payment Method
              </label>
              <select
                value={input.paymentMethod}
                onChange={(e) =>
                  setInput({ ...input, paymentMethod: e.target.value as PaymentMethod })
                }
                className="w-full bg-[#FAF8FE] border border-[#DDD3F6] rounded-xl px-3.5 py-2.5 text-sm text-[#181524] focus:bg-white focus:outline-none focus:border-[#7C3AED] transition-all"
              >
                <option value="Electronic check">Electronic check</option>
                <option value="Credit card">Credit card (Automatic)</option>
                <option value="Bank transfer">Bank transfer (Automatic)</option>
                <option value="Mailed check">Mailed check</option>
              </select>
            </div>

            {/* Products Subscribed */}
            <div>
              <label className="block text-xs font-semibold text-[#443D5C] mb-1.5">
                Subscribed Products (1-6)
              </label>
              <select
                value={input.numProducts}
                onChange={(e) =>
                  setInput({ ...input, numProducts: parseInt(e.target.value) || 1 })
                }
                className="w-full bg-[#FAF8FE] border border-[#DDD3F6] rounded-xl px-3.5 py-2.5 text-sm text-[#181524] focus:bg-white focus:outline-none focus:border-[#7C3AED] transition-all"
              >
                {[1, 2, 3, 4, 5, 6].map((n) => (
                  <option key={n} value={n}>
                    {n} {n === 1 ? 'Product (Single service)' : 'Products (Bundled package)'}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="pt-2">
            <button
              type="submit"
              className="w-full py-4 rounded-full bg-[#7C3AED] hover:bg-[#6D28D9] text-white font-semibold text-sm shadow-lg shadow-purple-500/25 transition-all flex items-center justify-center gap-2 cursor-pointer hover:scale-[1.01]"
            >
              <Sparkles className="w-4 h-4" />
              <span>Predict Customer</span>
            </button>
          </div>
        </form>

        {/* Prediction Results Card */}
        <div className="lg:col-span-5 space-y-6">
          {prediction ? (
            <div
              className={`rounded-3xl border p-6 sm:p-8 space-y-6 transition-all shadow-xl bg-white ${
                prediction.predictedClass === 1
                  ? 'border-rose-300 shadow-rose-900/5'
                  : 'border-emerald-300 shadow-emerald-900/5'
              }`}
            >
              {/* Header Badge */}
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-semibold uppercase tracking-wider text-[#7C3AED]">
                  Prediction Output
                </span>
                <span
                  className={`text-xs font-semibold px-3 py-1 rounded-full border ${
                    prediction.predictedClass === 1
                      ? 'bg-rose-50 text-rose-700 border-rose-200'
                      : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  }`}
                >
                  AdaBoost Ensemble
                </span>
              </div>

              {/* Classification Heading */}
              <div>
                <span className="text-xs text-[#6B6484] block mb-1">Customer Classification:</span>
                <div className="flex items-center gap-3">
                  {prediction.predictedClass === 1 ? (
                    <div className="w-12 h-12 rounded-2xl bg-rose-100 text-rose-600 flex items-center justify-center shrink-0">
                      <AlertTriangle className="w-6 h-6" />
                    </div>
                  ) : (
                    <div className="w-12 h-12 rounded-2xl bg-emerald-100 text-emerald-600 flex items-center justify-center shrink-0">
                      <CheckCircle2 className="w-6 h-6" />
                    </div>
                  )}
                  <div>
                    <h3
                      className={`text-2xl sm:text-3xl font-heading font-black ${
                        prediction.predictedClass === 1 ? 'text-rose-600' : 'text-emerald-600'
                      }`}
                    >
                      {prediction.predictedClass === 1
                        ? 'Potential Churn Risk'
                        : 'Loyal / Retained'}
                    </h3>
                    <p className="text-xs text-[#6B6484] mt-0.5">
                      {prediction.predictedClass === 1
                        ? 'High probability of cancellation within 60 days.'
                        : 'Account indicates solid engagement and low churn likelihood.'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Confidence Meter */}
              <div className="space-y-2 p-4 rounded-2xl bg-[#FBF9FE] border border-[#EBE4F7]">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-[#36304D] font-semibold">Confidence Score</span>
                  <span className="font-mono text-base font-bold text-[#7C3AED]">
                    {prediction.confidence}%
                  </span>
                </div>
                <div className="h-3 w-full bg-[#EAE2F8] rounded-full overflow-hidden flex">
                  <div
                    className={`h-full transition-all duration-500 rounded-full ${
                      prediction.predictedClass === 1
                        ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                        : 'bg-gradient-to-r from-[#8B5CF6] to-[#10B981]'
                    }`}
                    style={{ width: `${prediction.confidence}%` }}
                  />
                </div>
                <div className="flex justify-between text-[11px] text-[#7E7796] font-mono">
                  <span>Ensemble Margin: {prediction.rawScore}</span>
                  <span>50 Decision Stumps</span>
                </div>
              </div>

              {/* Side-by-Side Model Comparison on this Sample */}
              <div className="p-4 rounded-2xl bg-[#FAF8FE] border border-[#E9E1F7] space-y-3">
                <span className="text-xs font-semibold text-[#36304D] block">
                  Model Decision Comparison:
                </span>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-white border border-[#E8E0F5]">
                    <span className="text-[11px] text-[#7C7594] block">Baseline Tree</span>
                    <span
                      className={`font-semibold ${
                        prediction.decisionTreeClass === 1 ? 'text-amber-600' : 'text-[#36304D]'
                      }`}
                    >
                      {prediction.decisionTreeClass === 1 ? 'Churn Risk' : 'Retained'}
                    </span>
                    <span className="text-[10px] text-[#9790AF] block">
                      ({prediction.dtConfidence}% conf)
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-[#EFEAFF] border border-[#D5C6F7]">
                    <span className="text-[11px] text-[#6D28D9] font-medium block">AdaBoost</span>
                    <span
                      className={`font-semibold ${
                        prediction.adaBoostClass === 1 ? 'text-rose-600' : 'text-emerald-700'
                      }`}
                    >
                      {prediction.adaBoostClass === 1 ? 'Churn Risk' : 'Retained'}
                    </span>
                    <span className="text-[10px] text-[#6D28D9] block">
                      ({prediction.confidence}% conf)
                    </span>
                  </div>
                </div>

                {prediction.adaBoostClass !== prediction.decisionTreeClass && (
                  <p className="text-[11px] text-[#5B21B6] bg-white p-2.5 rounded-xl border border-[#D5C6F7]">
                    💡 <strong>Ensemble Advantage:</strong> The baseline tree misclassified this customer profile.
                    AdaBoost corrected the error using weighted weak learners.
                  </p>
                )}
              </div>

              {/* Explainable AI Drivers */}
              <div className="space-y-2">
                <span className="text-xs font-semibold text-[#36304D] block">
                  Key Influencing Factors (Explainable AI):
                </span>
                <div className="space-y-1.5">
                  {prediction.featureContributions.map((fc, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-xl bg-white border border-[#ECE5F8] text-xs flex items-start gap-2.5"
                    >
                      {fc.impact === 'churn_risk' ? (
                        <TrendingDown className="w-3.5 h-3.5 text-rose-500 shrink-0 mt-0.5" />
                      ) : (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                      )}
                      <div>
                        <span className="font-semibold text-[#181524]">{fc.feature}: </span>
                        <span className="text-[#5F5877]">{fc.description}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-3xl border border-dashed border-[#DCD3F2] p-8 text-center bg-white text-[#7E7796]">
              <Sparkles className="w-8 h-8 mx-auto text-[#7C3AED] mb-2" />
              <p className="text-sm">Click "Predict Customer" to see classification details.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
