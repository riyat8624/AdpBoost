import React, { useState } from 'react';
import {
  BrainCircuit,
  ArrowRight,
  Layers,
  HelpCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Award,
  Zap,
  BookOpen,
} from 'lucide-react';

export const AboutAdaBoostSection: React.FC = () => {
  const [openFaq, setOpenFaq] = useState<number | null>(0);

  const vivaQuestions = [
    {
      q: '1. What is AdaBoost and who invented it?',
      a: 'AdaBoost stands for Adaptive Boosting. It was formulated in 1996 by Yoav Freund and Robert Schapire, who won the Gödel Prize for their breakthrough. It is an ensemble meta-algorithm that converts a collection of weak classifiers into a single strong classifier by sequentially focusing on previously misclassified samples.',
    },
    {
      q: '2. What is a "Weak Learner" in the context of AdaBoost?',
      a: 'A weak learner is any machine learning classifier that performs slightly better than random guessing (error rate ε < 0.5 for binary classification). In AdaBoost, the canonical weak learner is a "Decision Stump"—a 1-level decision tree that makes a split on only a single feature with a single threshold.',
    },
    {
      q: '3. How does AdaBoost differ fundamentally from Bagging (e.g., Random Forest)?',
      a: 'In Bagging (Bootstrap Aggregating / Random Forest), multiple deep, independent decision trees are trained simultaneously in parallel on random bootstrap subsets of data with equal weights. In AdaBoost, models are trained sequentially: each new learner explicitly depends on the errors of preceding learners by using updated sample weights.',
    },
    {
      q: '4. How is the estimator weight (alpha) calculated?',
      a: 'The voting weight α_t of estimator t is calculated as: α_t = 0.5 * ln((1 - ε_t) / ε_t), where ε_t is the weighted training error. If an estimator is very accurate (low ε_t), α_t is large and positive. If its error approaches 0.5 (random guess), α_t approaches 0.',
    },
    {
      q: '5. How are sample weights updated between boosting rounds?',
      a: 'After round t, each sample weight is updated via: w_i ← w_i * exp(-α_t * y_i * h_t(x_i)), followed by normalization so sum(w_i) = 1. If sample i was correctly classified (y_i * h_t(x_i) = +1), its weight decreases by exp(-α_t). If misclassified (y_i * h_t(x_i) = -1), its weight increases by exp(+α_t).',
    },
    {
      q: '6. What happens if a weak learner achieves an error rate ≥ 0.5?',
      a: 'If a weak learner has an error rate of exactly 0.5, it provides no predictive value and α_t becomes 0. If error > 0.5, inverting its predictions yields an error < 0.5. In standard AdaBoost, training halts if a stump cannot find a split with error < 0.5.',
    },
    {
      q: '7. Why does AdaBoost outperform a single baseline Decision Tree?',
      a: 'A single decision tree is prone to high variance or greedy, sub-optimal local splits. AdaBoost aggregates 50+ diverse decision stumps. Each stump focuses on different customer feature dimensions (tenure, contract type, support calls, charges) and combines their weighted outputs, drastically reducing both bias and variance.',
    },
    {
      q: '8. Is AdaBoost sensitive to outliers or label noise?',
      a: 'Yes. Because AdaBoost exponentially increases the weights of misclassified instances, outliers or mislabeled noisy data will repeatedly receive higher and higher weights, potentially forcing subsequent stumps to overfit the noise. This is why data preprocessing and outlier handling are crucial.',
    },
  ];

  return (
    <div className="space-y-12 py-4">
      {/* Page Header */}
      <div>
        <span className="text-xs font-mono font-semibold text-[#7C3AED] uppercase tracking-wider block">
          Theoretical Foundations & Viva Guide
        </span>
        <h2 className="text-3xl sm:text-4xl font-heading font-black text-[#151226]">
          Understanding AdaBoost
        </h2>
        <p className="text-sm text-[#6A6384] mt-1 max-w-2xl">
          Core concepts, mathematical formulas, and oral defense viva question bank.
        </p>
      </div>

      {/* Why AdaBoost Banner */}
      <section className="rounded-3xl border border-[#7C3AED]/40 bg-gradient-to-r from-[#8E78F4] via-[#8168EB] to-[#6E50E2] text-white p-8 sm:p-10 space-y-4 shadow-xl shadow-purple-900/10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-white/20 backdrop-blur-md flex items-center justify-center text-white">
            <Zap className="w-5 h-5 fill-white" />
          </div>
          <h3 className="text-2xl font-heading font-black text-white">Why AdaBoost?</h3>
        </div>

        <p className="text-base text-purple-100 leading-relaxed max-w-3xl">
          AdaBoost combines multiple <strong>weak learners</strong> (simple Decision Stumps with
          accuracy slightly better than random guessing) into an accurate, robust{' '}
          <strong>strong classifier</strong>. In customer classification, human churn behaviors
          are non-linear and nuanced: high support calls might cause churn for month-to-month users,
          while two-year contract users remain loyal despite calling support.
        </p>

        <p className="text-base text-purple-100 leading-relaxed max-w-3xl">
          A single shallow Decision Tree cannot capture all these nuanced rules without overfitting.
          AdaBoost sequentially adjusts sample weights so each consecutive stump tackles the customer
          segments that preceding stumps failed to classify correctly.
        </p>
      </section>

      {/* Visual Pipeline */}
      <section className="rounded-3xl border border-[#ECE4F8] bg-white p-7 sm:p-9 space-y-6 shadow-xs">
        <div className="flex items-center justify-between">
          <div>
            <span className="text-xs font-mono font-semibold text-[#7C3AED] uppercase tracking-wider block">
              Sequential Execution
            </span>
            <h3 className="text-xl font-heading font-bold text-[#181524]">
              Visual Boosting Pipeline
            </h3>
          </div>
          <span className="text-xs text-[#7B7494] font-medium hidden sm:inline-block">
            T = 50 Boosting Iterations
          </span>
        </div>

        {/* Visual Flow Blocks */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3.5">
          {/* Step 1 */}
          <div className="p-4 rounded-2xl bg-[#FAF8FE] border border-[#ECE5F8] space-y-2">
            <span className="text-[10px] font-mono text-[#8C85A6] uppercase block font-semibold">Round 01</span>
            <h4 className="font-heading font-bold text-sm text-[#181524]">Weak Learner 1</h4>
            <p className="text-xs text-[#6A6382]">
              Decision Stump fit on uniform weights ($w_i = 1/N$).
            </p>
            <div className="pt-1 text-[11px] font-mono text-[#7C3AED] font-semibold">
              Split: Contract
            </div>
          </div>

          {/* Step 2 */}
          <div className="p-4 rounded-2xl bg-[#FFFBEB] border border-[#FDE68A] space-y-2">
            <span className="text-[10px] font-mono text-[#B45309] uppercase block font-semibold">Round 02</span>
            <h4 className="font-heading font-bold text-sm text-[#181524]">Weak Learner 2</h4>
            <p className="text-xs text-[#6A6382]">
              Misclassified samples gain weight; stump focuses on hard cases.
            </p>
            <div className="pt-1 text-[11px] font-mono text-[#B45309] font-semibold">
              Split: Support Calls
            </div>
          </div>

          {/* Step 3 */}
          <div className="p-4 rounded-2xl bg-[#F0FDF4] border border-[#BBF7D0] space-y-2">
            <span className="text-[10px] font-mono text-[#16A34A] uppercase block font-semibold">Round 03 ... T</span>
            <h4 className="font-heading font-bold text-sm text-[#181524]">Weak Learner 3</h4>
            <p className="text-xs text-[#6A6382]">
              Sequential re-weighting continues over all 50 estimators.
            </p>
            <div className="pt-1 text-[11px] font-mono text-[#16A34A] font-semibold">
              Split: Tenure & Fees
            </div>
          </div>

          {/* Step 4 */}
          <div className="p-4 rounded-2xl bg-[#F3EEFD] border border-[#DDD0F8] space-y-2">
            <span className="text-[10px] font-mono text-[#7C3AED] uppercase block font-semibold">Aggregation</span>
            <h4 className="font-heading font-bold text-sm text-[#181524]">Weighted Vote</h4>
            <p className="text-xs text-[#6A6382]">
              Sum of estimator votes weighted by their importance ($\alpha_t$).
            </p>
            <div className="pt-1 text-[11px] font-mono text-[#7C3AED] font-semibold">
              ∑ α_t · h_t(x)
            </div>
          </div>

          {/* Step 5 */}
          <div className="p-4 rounded-2xl bg-[#7C3AED] text-white space-y-2 shadow-sm">
            <span className="text-[10px] font-mono text-purple-200 uppercase block font-semibold">Final Output</span>
            <h4 className="font-heading font-bold text-sm text-white">Strong Classifier</h4>
            <p className="text-xs text-purple-100">
              Binary sign generates final customer classification & probability.
            </p>
            <div className="pt-1 text-[11px] font-mono text-white font-bold">
              H(x) = sign(Margin)
            </div>
          </div>
        </div>

        {/* Textual Flow Banner */}
        <div className="p-3.5 rounded-2xl bg-[#FAF8FE] border border-[#ECE5F8] text-xs font-mono text-center text-[#554E6D] overflow-x-auto whitespace-nowrap">
          Weak Learner 1 → Reweight Misclassified → Weak Learner 2 → Reweight → Weak Learner 3 → Weighted Combination (Σ αt ht) → Final Prediction
        </div>
      </section>

      {/* Core Mathematical Formulations */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="rounded-3xl border border-[#ECE4F8] bg-white p-7 space-y-3 shadow-xs">
          <span className="text-xs font-mono font-semibold text-[#7C3AED] block">Equation 01</span>
          <h4 className="font-heading font-bold text-base text-[#181524]">Weighted Error</h4>
          <div className="p-3 rounded-xl bg-[#FAF8FE] font-mono text-sm text-[#7C3AED] border border-[#ECE5F8] font-bold">
            ε_t = ∑ w_i · [y_i ≠ h_t(x_i)]
          </div>
          <p className="text-xs text-[#6A6382] leading-relaxed">
            Sum of weights of customer instances misclassified by decision stump $h_t$ in iteration $t$.
          </p>
        </div>

        <div className="rounded-3xl border border-[#ECE4F8] bg-white p-7 space-y-3 shadow-xs">
          <span className="text-xs font-mono font-semibold text-[#7C3AED] block">Equation 02</span>
          <h4 className="font-heading font-bold text-base text-[#181524]">Estimator Weight (Alpha)</h4>
          <div className="p-3 rounded-xl bg-[#FAF8FE] font-mono text-sm text-[#059669] border border-[#ECE5F8] font-bold">
            α_t = ½ · ln( (1 - ε_t) / ε_t )
          </div>
          <p className="text-xs text-[#6A6382] leading-relaxed">
            Estimators with low error receive exponentially higher voting weight in the final ensemble.
          </p>
        </div>

        <div className="rounded-3xl border border-[#ECE4F8] bg-white p-7 space-y-3 shadow-xs">
          <span className="text-xs font-mono font-semibold text-[#7C3AED] block">Equation 03</span>
          <h4 className="font-heading font-bold text-base text-[#181524]">Sample Re-weighting</h4>
          <div className="p-3 rounded-xl bg-[#FAF8FE] font-mono text-sm text-[#B45309] border border-[#ECE5F8] font-bold">
            w_i ← w_i · exp( -α_t · y_i · h_t(x_i) )
          </div>
          <p className="text-xs text-[#6A6382] leading-relaxed">
            Weights are re-normalized so $\sum w_i = 1$. Misclassified instances receive multiplied weight.
          </p>
        </div>
      </section>

      {/* College Viva Defense Questions */}
      <section className="rounded-3xl border border-[#ECE4F8] bg-white p-7 sm:p-9 space-y-6 shadow-xs">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-[#EFEAFF] text-[#7C3AED] flex items-center justify-center">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-mono font-semibold text-[#7C3AED] uppercase tracking-wider block">
              College Viva & Exam Kit
            </span>
            <h3 className="text-xl font-heading font-bold text-[#181524]">
              Viva Questions & Model Answers
            </h3>
          </div>
        </div>

        <div className="space-y-3">
          {vivaQuestions.map((item, idx) => (
            <div
              key={idx}
              className="rounded-2xl border border-[#ECE5F8] bg-[#FAF8FE] overflow-hidden transition-all"
            >
              <button
                type="button"
                onClick={() => setOpenFaq(openFaq === idx ? null : idx)}
                className="w-full px-5 py-4 text-left font-medium text-sm text-[#181524] hover:text-[#7C3AED] flex items-center justify-between gap-4 cursor-pointer"
              >
                <span className="font-semibold">{item.q}</span>
                {openFaq === idx ? (
                  <ChevronUp className="w-4 h-4 text-[#7C3AED] shrink-0" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-[#8C85A6] shrink-0" />
                )}
              </button>

              {openFaq === idx && (
                <div className="px-5 pb-4 pt-1 text-xs text-[#5B5573] leading-relaxed border-t border-[#ECE5F8] bg-white">
                  {item.a}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};
