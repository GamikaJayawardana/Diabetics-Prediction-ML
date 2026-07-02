import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Heart, User, ActivitySquare, BrainCircuit, Loader2, ShieldAlert, ShieldCheck, Info, Moon, Sun, Download } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip as RechartsTooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, ReferenceLine, Cell } from 'recharts';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) { return { hasError: true, error }; }
  componentDidCatch(error, errorInfo) { console.error("UI Error:", error, errorInfo); }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ color: '#ef4444', padding: '2rem', background: '#0f172a', height: '100vh', fontFamily: 'monospace' }}>
          <h2>Application Crash Detected</h2>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{this.state.error?.toString()}</pre>
          <button onClick={() => window.location.reload()} style={{ padding: '0.5rem 1rem', marginTop: '1rem' }}>Reload App</button>
        </div>
      );
    }
    return this.props.children; 
  }
}

const InputField = ({ label, name, placeholder, value, onChange }) => (
  <div className="enhanced-input-group">
    <label>{label}</label>
    <div className="enhanced-input-control">
      <input 
        type="number" 
        className="enhanced-input-value" 
        name={name} 
        value={value} 
        onChange={onChange}
        placeholder={placeholder}
        step="any"
      />
    </div>
  </div>
);

function AppContent() {
  const [formData, setFormData] = useState({
    Pregnancies: '',
    Glucose: '',
    BloodPressure: '',
    SkinThickness: '',
    Insulin: '',
    BMI: '',
    DiabetesPedigreeFunction: '',
    Age: ''
  });

  const [loading, setLoading] = useState(false);
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [isDarkMode, setIsDarkMode] = useState(true);

  useEffect(() => {
    document.body.classList.toggle('light-mode', !isDarkMode);
  }, [isDarkMode]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value === '' ? '' : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const isComplete = Object.values(formData).every(val => val !== '');
    if (!isComplete) {
      setError("Please fill out all patient vitals before predicting.");
      return;
    }

    setLoading(true);
    setError('');
    
    const payload = {};
    for (const key in formData) {
      payload[key] = parseFloat(formData[key]);
    }

    const currentIP = window.location.hostname;
    const baseURL = `http://${currentIP}:8000`;

    try {
      const response = await axios.post(`${baseURL}/predict`, payload);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Server connection error.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const generatePDF = async () => {
    setIsGeneratingPdf(true);
    const dashboardElement = document.querySelector('.main-content');
    
    try {
      const canvas = await html2canvas(dashboardElement, {
        scale: 2, 
        useCORS: true,
        backgroundColor: isDarkMode ? '#0f172a' : '#f8fafc'
      });
      
      const imgData = canvas.toDataURL('image/png');
      
      const pdf = new jsPDF({
        orientation: 'landscape',
        unit: 'px',
        format: [canvas.width, canvas.height]
      });
      
      pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
      pdf.save(`Diabetes_Report_${new Date().toISOString().split('T')[0]}.pdf`);
    } catch (err) {
      console.error("Error generating PDF", err);
      alert("Failed to generate PDF report.");
    } finally {
      setIsGeneratingPdf(false);
    }
  };

  const renderRadarChart = () => {
    const data = [
      { subject: 'Glucose', A: parseFloat(formData.Glucose) || 0, B: 110, C: 142, fullMark: 200 },
      { subject: 'BP', A: parseFloat(formData.BloodPressure) || 0, B: 70, C: 75, fullMark: 140 },
      { subject: 'SkinThick', A: parseFloat(formData.SkinThickness) || 0, B: 27, C: 33, fullMark: 100 },
      { subject: 'Insulin', A: parseFloat(formData.Insulin) || 0, B: 80, C: 100, fullMark: 300 },
      { subject: 'BMI', A: parseFloat(formData.BMI) || 0, B: 30, C: 35, fullMark: 60 },
      { subject: 'Age', A: parseFloat(formData.Age) || 0, B: 27, C: 37, fullMark: 100 },
    ];

    const userColor = result?.is_high_risk ? '#ef4444' : '#10b981';
    const gridStroke = isDarkMode ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.15)";
    const tickFill = isDarkMode ? "#94a3b8" : "#64748b";
    const tooltipBg = isDarkMode ? "rgba(15,23,42,0.95)" : "rgba(255,255,255,0.95)";
    const tooltipBorder = isDarkMode ? "#334155" : "#cbd5e1";
    const tooltipColor = isDarkMode ? "#f8fafc" : "#1e293b";

    return (
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke={gridStroke} />
          <PolarAngleAxis dataKey="subject" tick={{ fill: tickFill, fontSize: 11 }} />
          <PolarRadiusAxis angle={30} domain={[0, 'auto']} tick={false} />
          {result && <Radar name="Current Patient" dataKey="A" stroke={userColor} fill={userColor} fillOpacity={0.4} strokeWidth={3} animationDuration={400} />}
          <Radar name="Avg Healthy" dataKey="B" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} />
          <Radar name="Avg Diabetic" dataKey="C" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.1} />
          <RechartsTooltip contentStyle={{ backgroundColor: tooltipBg, border: `1px solid ${tooltipBorder}`, borderRadius: '8px', color: tooltipColor }} />
        </RadarChart>
      </ResponsiveContainer>
    );
  };

  const renderShapBarChart = () => {
    if (!result?.shap_values) return (
      <div className="empty-state">
        <Info size={32} opacity={0.5} />
        <p>Run diagnosis to view key drivers.</p>
      </div>
    );

    const sortedShap = Object.entries(result.shap_values)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 5)
      .map(([feature, val]) => ({
        feature,
        impact: parseFloat(val.toFixed(3)),
        color: val > 0 ? (isDarkMode ? "#ef4444" : "#dc2626") : (isDarkMode ? "#10b981" : "#059669")
      }));

    const textColor = isDarkMode ? "#f8fafc" : "#1e293b";
    const gridColor = isDarkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)";
    const tooltipBg = isDarkMode ? "rgba(15,23,42,0.95)" : "rgba(255,255,255,0.95)";
    const tooltipBorder = isDarkMode ? "#334155" : "#cbd5e1";

    const CustomTooltip = ({ active, payload }) => {
      if (active && payload && payload.length) {
        const data = payload[0].payload;
        const isIncrease = data.impact > 0;
        return (
          <div style={{ backgroundColor: tooltipBg, border: `1px solid ${tooltipBorder}`, padding: '12px', borderRadius: '8px', color: textColor, fontSize: '0.9rem', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }}>
            <p style={{ margin: '0 0 6px 0', fontWeight: '600', borderBottom: `1px solid ${tooltipBorder}`, paddingBottom: '4px' }}>{data.feature}</p>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: data.color, fontWeight: '500' }}>
              {isIncrease ? "Increases Risk" : "Decreases Risk"}
            </div>
          </div>
        );
      }
      return null;
    };

    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={sortedShap}
          margin={{ top: 10, right: 20, left: 20, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke={gridColor} />
          <XAxis type="number" tick={{ fill: textColor, fontSize: 11 }} stroke={gridColor} />
          <YAxis dataKey="feature" type="category" tick={{ fill: textColor, fontSize: 11 }} width={80} stroke={gridColor} />
          <RechartsTooltip content={<CustomTooltip />} cursor={{fill: isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'}} />
          <ReferenceLine x={0} stroke={textColor} opacity={0.4} />
          <Bar dataKey="impact" radius={[0, 4, 4, 0]} barSize={24}>
            {sortedShap.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const isHighRisk = result?.is_high_risk;
  const probability = result?.probability || 0;
  const probPercent = (probability * 100).toFixed(1);

  return (
    <div className="app-container">
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <h1><BrainCircuit size={32} color="var(--accent-color)" /> Intelligent Diabetes Prediction</h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {result && (
            <button 
              onClick={generatePDF} 
              className="pdf-btn"
              disabled={isGeneratingPdf}
            >
              {isGeneratingPdf ? <Loader2 size={16} className="loader-spin" style={{animation: 'spin 1s linear infinite'}}/> : <Download size={16} />}
              Export Report
            </button>
          )}
          <p className="header-subtitle" style={{ margin: 0, color: 'var(--text-secondary)' }}>XGBoost Analysis Dashboard</p>
          <button 
            onClick={() => setIsDarkMode(!isDarkMode)} 
            className="theme-toggle-btn"
            aria-label="Toggle Dark Mode"
          >
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>
      </header>

      <main className="main-content">
        <section className="glass-panel input-panel-container">
          <div className="panel-title">
            <User size={22} color="var(--accent-color)" />
            Patient Vitals
          </div>
          
          <form className="input-panel" onSubmit={handleSubmit}>
            <div className="enhanced-input-grid">
              <InputField label="Pregnancies" name="Pregnancies" placeholder="e.g. 1" value={formData.Pregnancies} onChange={handleChange} />
              <InputField label="Glucose (mg/dL)" name="Glucose" placeholder="e.g. 110" value={formData.Glucose} onChange={handleChange} />
              <InputField label="Blood Pressure (mm Hg)" name="BloodPressure" placeholder="e.g. 72" value={formData.BloodPressure} onChange={handleChange} />
              <InputField label="Skin Thickness (mm)" name="SkinThickness" placeholder="e.g. 25" value={formData.SkinThickness} onChange={handleChange} />
              <InputField label="Insulin (μU/ml)" name="Insulin" placeholder="e.g. 80" value={formData.Insulin} onChange={handleChange} />
              <InputField label="BMI" name="BMI" placeholder="e.g. 32.0" value={formData.BMI} onChange={handleChange} />
              <InputField label="Diabetes Pedigree Function" name="DiabetesPedigreeFunction" placeholder="e.g. 0.372" value={formData.DiabetesPedigreeFunction} onChange={handleChange} />
              <InputField label="Age (years)" name="Age" placeholder="e.g. 29" value={formData.Age} onChange={handleChange} />
            </div>

            {error && <div style={{ color: '#ef4444', textAlign: 'center', fontSize: '0.95rem', margin: '0.5rem 0' }}>{error}</div>}

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? (
                <><Loader2 size={20} className="loader-spin" style={{animation: 'spin 1s linear infinite'}}/> Analyzing...</>
              ) : (
                <><BrainCircuit size={20} /> Run AI Diagnosis</>
              )}
            </button>
          </form>
        </section>

        <section className="dashboard-grid">
          <div className="top-metrics">
            <div className={`metric-card ${result ? (isHighRisk ? 'danger' : 'success') : 'neutral'}`}>
              <div className="metric-title">
                {result ? (isHighRisk ? <ShieldAlert size={20} /> : <ShieldCheck size={20} />) : <Activity size={20} />}
                Diagnosis Status
              </div>
              <div className="metric-value">
                {result ? (isHighRisk ? 'HIGH RISK' : 'LOW RISK') : 'AWAITING DATA'}
              </div>
            </div>

            <div className="metric-card glass-panel" style={{ padding: '1.5rem' }}>
              <div className="metric-title"><Activity size={20} color="var(--accent-color)" /> AI Confidence Score</div>
              <div className="metric-value" style={{ color: result ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                {result ? `${probPercent}%` : '--%'}
              </div>
              <div className="confidence-bar-container">
                <div className="confidence-header">
                  <span>Probability</span>
                </div>
                <div className="confidence-track">
                  <div 
                    className="confidence-fill" 
                    style={{ 
                      width: result ? `${probPercent}%` : '0%',
                      backgroundColor: isHighRisk ? 'var(--danger-color)' : 'var(--success-color)'
                    }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          <div className="visuals-grid">
            <div className="glass-panel chart-container">
              <div className="panel-title">
                <ActivitySquare size={20} color="var(--accent-color)" />
                Vital Distribution
              </div>
              <div className="chart-wrapper">
                {renderRadarChart()}
              </div>
            </div>

            <div className="glass-panel chart-container" style={{ paddingRight: '1rem' }}>
              <div className="panel-title">
                <Heart size={20} color="var(--accent-color)" />
                Key Drivers Analysis
              </div>
              <div className="chart-wrapper">
                {renderShapBarChart()}
              </div>
            </div>
          </div>
          
        </section>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppContent />
    </ErrorBoundary>
  );
}
