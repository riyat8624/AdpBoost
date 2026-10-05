import React, { useState } from 'react';
import {
  Database,
  Download,
  Code2,
  Search,
  Check,
  Copy,
  Terminal,
} from 'lucide-react';
import { CustomerRecord } from '../types/ml';
import { exportDatasetToCsv } from '../ml/dataset';
import {
  PYTHON_TRAIN_SCRIPT,
  PYTHON_APP_SCRIPT,
  PYTHON_REQUIREMENTS,
  triggerDownload,
} from '../ml/pythonExporter';

interface DatasetAndCodeSectionProps {
  dataset: CustomerRecord[];
}

export const DatasetAndCodeSection: React.FC<DatasetAndCodeSectionProps> = ({ dataset }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [activeCodeTab, setActiveCodeTab] = useState<'train' | 'app' | 'req' | 'readme'>('train');
  const [copied, setCopied] = useState(false);

  const itemsPerPage = 10;

  const filteredData = dataset.filter((d) => {
    const term = searchTerm.toLowerCase();
    return (
      d.id.toLowerCase().includes(term) ||
      d.contract.toLowerCase().includes(term) ||
      d.internetService.toLowerCase().includes(term) ||
      d.paymentMethod.toLowerCase().includes(term) ||
      d.gender.toLowerCase().includes(term)
    );
  });

  const totalPages = Math.ceil(filteredData.length / itemsPerPage) || 1;
  const paginatedData = filteredData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const handleDownloadCsv = () => {
    const csvContent = exportDatasetToCsv(dataset);
    triggerDownload('customer_churn_data.csv', csvContent, 'text/csv');
  };

  const getCodeContent = () => {
    switch (activeCodeTab) {
      case 'train':
        return PYTHON_TRAIN_SCRIPT;
      case 'app':
        return PYTHON_APP_SCRIPT;
      case 'req':
        return PYTHON_REQUIREMENTS;
      case 'readme':
        return `# Customer Classification using AdaBoost\nComplete College Project Kit\nSee customer-adaboost/README.md for full instructions.`;
      default:
        return '';
    }
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(getCodeContent());
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadCode = () => {
    const filenames: Record<string, string> = {
      train: 'train_model.py',
      app: 'app.py',
      req: 'requirements.txt',
      readme: 'README.md',
    };
    triggerDownload(filenames[activeCodeTab], getCodeContent(), 'text/plain');
  };

  return (
    <div className="space-y-10 py-4">
      {/* Header */}
      <div>
        <span className="text-xs font-mono font-semibold text-[#7C3AED] uppercase tracking-wider block">
          Data Explorer & Code Submission Kit
        </span>
        <h2 className="text-3xl sm:text-4xl font-heading font-black text-[#151226]">
          Dataset & Python Code
        </h2>
        <p className="text-sm text-[#6A6384] mt-1 max-w-2xl">
          Search the 1,000-sample benchmark customer dataset and download complete Scikit-Learn
          training scripts for your college submission.
        </p>
      </div>

      {/* Dataset Explorer Card */}
      <section className="rounded-3xl border border-[#ECE4F8] bg-white p-7 space-y-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-[#7C3AED]" />
              <h3 className="font-heading font-bold text-lg text-[#181524]">
                Customer Classification Dataset
              </h3>
            </div>
            <p className="text-xs text-[#7B7494] mt-0.5">
              1,000 synthetic records modeled on real-world telecom customer churn dynamics.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-[#8C85A6]" />
              <input
                type="text"
                placeholder="Search contract, payment..."
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setCurrentPage(1);
                }}
                className="bg-[#FAF8FE] border border-[#DDD3F6] rounded-full pl-8 pr-4 py-1.5 text-xs text-[#181524] placeholder-[#8C85A6] focus:bg-white focus:outline-none focus:border-[#7C3AED] w-48 sm:w-60 transition-all"
              />
            </div>

            <button
              onClick={handleDownloadCsv}
              className="px-4 py-2 rounded-full bg-[#161324] hover:bg-black text-white text-xs font-semibold shadow-xs transition-colors flex items-center gap-1.5 cursor-pointer whitespace-nowrap"
            >
              <Download className="w-3.5 h-3.5 text-purple-200" />
              <span>Download CSV</span>
            </button>
          </div>
        </div>

        {/* Dataset Table */}
        <div className="overflow-x-auto rounded-2xl border border-[#ECE4F8] bg-white">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#FAF8FE] border-b border-[#ECE4F8] text-[#554E6D] font-mono">
              <tr>
                <th className="py-3 px-3.5 font-semibold">ID</th>
                <th className="py-3 px-3.5 font-semibold">Age</th>
                <th className="py-3 px-3.5 font-semibold">Gender</th>
                <th className="py-3 px-3.5 font-semibold">Tenure</th>
                <th className="py-3 px-3.5 font-semibold">Contract</th>
                <th className="py-3 px-3.5 font-semibold">Internet</th>
                <th className="py-3 px-3.5 font-semibold">Calls</th>
                <th className="py-3 px-3.5 font-semibold">Monthly</th>
                <th className="py-3 px-3.5 font-semibold">Total</th>
                <th className="py-3 px-3.5 font-semibold">Payment</th>
                <th className="py-3 px-3.5 text-right font-semibold">Class</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#F0EAF8]">
              {paginatedData.map((row) => (
                <tr key={row.id} className="hover:bg-[#FAF8FE] text-[#36304D] transition-colors">
                  <td className="py-3 px-3.5 font-mono text-[#7C3AED] font-semibold">{row.id}</td>
                  <td className="py-3 px-3.5">{row.age}</td>
                  <td className="py-3 px-3.5">{row.gender}</td>
                  <td className="py-3 px-3.5">{row.tenure} mo</td>
                  <td className="py-3 px-3.5 font-medium">{row.contract}</td>
                  <td className="py-3 px-3.5">{row.internetService}</td>
                  <td className="py-3 px-3.5 font-mono">{row.supportCalls}</td>
                  <td className="py-3 px-3.5 font-mono">${row.monthlyCharges}</td>
                  <td className="py-3 px-3.5 font-mono">${row.totalCharges}</td>
                  <td className="py-3 px-3.5">{row.paymentMethod}</td>
                  <td className="py-3 px-3.5 text-right">
                    <span
                      className={`text-[11px] font-semibold px-2.5 py-1 rounded-full ${
                        row.customerClass === 1
                          ? 'bg-rose-50 text-rose-700 border border-rose-200'
                          : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      }`}
                    >
                      {row.customerClass === 1 ? 'Churn Risk' : 'Retained'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between text-xs text-[#7B7494] pt-1">
          <span>
            Showing {(currentPage - 1) * itemsPerPage + 1} to{' '}
            {Math.min(currentPage * itemsPerPage, filteredData.length)} of {filteredData.length} records
          </span>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1.5 rounded-full bg-[#FAF8FE] border border-[#ECE5F8] text-[#36304D] font-medium disabled:opacity-40 cursor-pointer"
            >
              Prev
            </button>
            <span className="font-mono text-[#36304D] font-semibold px-2">
              {currentPage} / {totalPages}
            </span>
            <button
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="px-3 py-1.5 rounded-full bg-[#FAF8FE] border border-[#ECE5F8] text-[#36304D] font-medium disabled:opacity-40 cursor-pointer"
            >
              Next
            </button>
          </div>
        </div>
      </section>

      {/* College Project Submission Code Viewer */}
      <section className="rounded-3xl border border-[#ECE4F8] bg-white p-7 space-y-6 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Code2 className="w-5 h-5 text-[#7C3AED]" />
              <h3 className="font-heading font-bold text-lg text-[#181524]">
                Standalone Python / Scikit-Learn Code
              </h3>
            </div>
            <p className="text-xs text-[#7B7494] mt-0.5">
              Exact scripts included in the repository for running on your local machine or terminal.
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={handleCopyCode}
              className="px-4 py-2 rounded-full bg-[#FAF8FE] hover:bg-[#F3EEFD] text-[#36304D] text-xs font-semibold border border-[#ECE5F8] transition-colors flex items-center gap-1.5 cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-[#10B981]" />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-[#7C3AED]" />
                  <span>Copy Code</span>
                </>
              )}
            </button>

            <button
              onClick={handleDownloadCode}
              className="px-4 py-2 rounded-full bg-[#7C3AED] hover:bg-[#6D28D9] text-white text-xs font-semibold shadow-xs transition-colors flex items-center gap-1.5 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download File</span>
            </button>
          </div>
        </div>

        {/* File Tabs */}
        <div className="flex items-center gap-2 border-b border-[#F0EAF8] pb-2">
          <button
            onClick={() => setActiveCodeTab('train')}
            className={`px-3.5 py-1.5 rounded-full text-xs font-mono transition-all cursor-pointer ${
              activeCodeTab === 'train'
                ? 'bg-[#7C3AED] text-white font-semibold shadow-xs'
                : 'text-[#6A6382] hover:bg-[#FAF8FE]'
            }`}
          >
            train_model.py
          </button>
          <button
            onClick={() => setActiveCodeTab('app')}
            className={`px-3.5 py-1.5 rounded-full text-xs font-mono transition-all cursor-pointer ${
              activeCodeTab === 'app'
                ? 'bg-[#7C3AED] text-white font-semibold shadow-xs'
                : 'text-[#6A6382] hover:bg-[#FAF8FE]'
            }`}
          >
            app.py (Streamlit)
          </button>
          <button
            onClick={() => setActiveCodeTab('req')}
            className={`px-3.5 py-1.5 rounded-full text-xs font-mono transition-all cursor-pointer ${
              activeCodeTab === 'req'
                ? 'bg-[#7C3AED] text-white font-semibold shadow-xs'
                : 'text-[#6A6382] hover:bg-[#FAF8FE]'
            }`}
          >
            requirements.txt
          </button>
        </div>

        {/* Code Content Box */}
        <div className="relative rounded-2xl border border-[#ECE5F8] bg-[#FAF8FE] p-5 overflow-x-auto max-h-[420px] font-mono text-xs text-[#28223D] leading-relaxed">
          <pre>{getCodeContent()}</pre>
        </div>

        {/* How to Run Locally Terminal Box */}
        <div className="p-5 rounded-2xl bg-[#151226] text-white space-y-2 shadow-sm">
          <span className="text-xs font-mono text-purple-300 uppercase tracking-wider block font-semibold">
            How to Run Locally on Your PC
          </span>
          <div className="bg-[#1F1B36] p-4 rounded-xl border border-purple-950 font-mono text-xs space-y-1.5">
            <p className="text-[#8E86A8]"># 1. Install dependencies</p>
            <p className="text-[#34D399]">pip install -r requirements.txt</p>
            <p className="text-[#8E86A8] pt-1"># 2. Train baseline and AdaBoost models</p>
            <p className="text-[#34D399]">python train_model.py</p>
            <p className="text-[#8E86A8] pt-1"># 3. Launch interactive web dashboard</p>
            <p className="text-[#34D399]">streamlit run app.py</p>
          </div>
        </div>
      </section>
    </div>
  );
};
