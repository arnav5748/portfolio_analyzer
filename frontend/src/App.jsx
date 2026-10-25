
import { useState } from "react";
import axios from "axios";
import {
  LayoutDashboard,
  BriefcaseBusiness,
  ChartNoAxesCombined,
  BrainCircuit,
  Plus,
  Trash2,
  LoaderCircle,
  TrendingUp,
  Activity,
  ShieldAlert,
  Gauge,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

import "./App.css";

const API_URL = "http://localhost:8000";

const initialPortfolio = [
  { ticker: "AAPL", weight: 40 },
  { ticker: "MSFT", weight: 35 },
  { ticker: "NVDA", weight: 25 },
];

function MetricCard({ title, value, subtitle, icon: Icon, color }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{title}</span>
        <div className={`metric-icon ${color}`}>
          <Icon size={19} />
        </div>
      </div>
      <h2>{value}</h2>
      <p>{subtitle}</p>
    </div>
  );
}

function App() {
  const [portfolio, setPortfolio] = useState(initialPortfolio);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const totalWeight = portfolio.reduce(
    (sum, item) => sum + Number(item.weight || 0),
    0
  );

  function updateAsset(index, field, value) {
    const updated = [...portfolio];
    updated[index][field] =
      field === "ticker" ? value.toUpperCase() : value;

    setPortfolio(updated);
  }

  function addAsset() {
    setPortfolio([...portfolio, { ticker: "", weight: 0 }]);
  }

  function removeAsset(index) {
    setPortfolio(portfolio.filter((_, i) => i !== index));
  }

  async function analyzePortfolio() {
    setError("");
    setResult(null);

    const weights = {};

    for (const item of portfolio) {
      const ticker = item.ticker.trim().toUpperCase();
      const weight = Number(item.weight);

      if (!ticker) {
        setError("Please enter a ticker for every asset.");
        return;
      }

      if (weight < 0 || !Number.isFinite(weight)) {
        setError("Weights must be valid non-negative numbers.");
        return;
      }

      if (weights[ticker] !== undefined) {
        setError(`Duplicate ticker: ${ticker}`);
        return;
      }

      weights[ticker] = weight / 100;
    }

    if (Math.abs(totalWeight - 100) > 0.01) {
      setError("Portfolio weights must add up to 100%.");
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/analyze`, {
        weights,
      });

      setResult({
        ...response.data.portfolio,
        ml_analysis: response.data.ml_analysis,
      });
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Unable to connect to the backend. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  }

  const performance = result?.performance;
  const risk = result?.risk;
  const ml = result?.ml_analysis;
  const marketRegime = ml?.market_regime;
  const riskForecast = ml?.risk_forecast;
  const shapExplanation = ml?.explanation;
  const shapFeatures = shapExplanation?.top_features || [];
  const maxShapImpact = Math.max(
    ...shapFeatures.map((item) => Math.abs(item.shap_value)),
    0.000001
  );

  const chartData = result
    ? [
        {
          date: result.start_date,
          value: 1,
        },
        {
          date: result.end_date,
          value: performance.cumulative_growth,
        },
      ]
    : [];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <ChartNoAxesCombined size={23} />
          </div>
          <span>PortfolioIQ</span>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav>
          <div className="nav-item active">
            <LayoutDashboard size={19} />
            Dashboard
          </div>

          <div className="nav-item">
            <BriefcaseBusiness size={19} />
            My Portfolio
          </div>

          <div className="nav-item">
            <Activity size={19} />
            Risk Analysis
          </div>

          <div className="nav-item">
            <BrainCircuit size={19} />
            AI Insights
            <span className="coming-soon">Soon</span>
          </div>
        </nav>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          Backend connected
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">Workspace / Dashboard</div>
            <h1>Portfolio Overview</h1>
          </div>

          <div className="profile">
            <div className="avatar">A</div>
            <span>Investor</span>
          </div>
        </header>

        <section className="welcome">
          <div>
            <h2>Analyze your investments.</h2>
            <p>
              Build a custom portfolio and understand its performance
              and risk using historical market data.
            </p>
          </div>
          <div className="welcome-icon">
            <TrendingUp size={36} />
          </div>
        </section>

        <section className="portfolio-section">
          <div className="section-heading">
            <div>
              <h2>Portfolio Builder</h2>
              <p>Choose assets and assign their portfolio weights.</p>
            </div>
            <span className="asset-count">
              {portfolio.length} assets
            </span>
          </div>

          <div className="portfolio-table">
            <div className="table-header">
              <span>ASSET TICKER</span>
              <span>WEIGHT (%)</span>
              <span>ACTION</span>
            </div>

            {portfolio.map((item, index) => (
              <div className="asset-row" key={index}>
                <input
                  value={item.ticker}
                  onChange={(e) =>
                    updateAsset(index, "ticker", e.target.value)
                  }
                  placeholder="e.g. AAPL"
                  maxLength={20}
                />

                <input
                  type="number"
                  min="0"
                  max="100"
                  step="0.01"
                  value={item.weight}
                  onChange={(e) =>
                    updateAsset(index, "weight", e.target.value)
                  }
                />

                <button
                  className="remove-button"
                  onClick={() => removeAsset(index)}
                  aria-label="Remove asset"
                >
                  <Trash2 size={17} />
                </button>
              </div>
            ))}
          </div>

          <div className="portfolio-actions">
            <button className="add-button" onClick={addAsset}>
              <Plus size={17} />
              Add Asset
            </button>

            <div
              className={`weight-total ${
                Math.abs(totalWeight - 100) > 0.01 ? "invalid" : ""
              }`}
            >
              Total: {totalWeight.toFixed(2)}%
            </div>
          </div>

          <button
            className="analyze-button"
            onClick={analyzePortfolio}
            disabled={loading || portfolio.length === 0}
          >
            {loading ? (
              <>
                <LoaderCircle className="spin" size={19} />
                Analyzing Portfolio...
              </>
            ) : (
              <>
                <ChartNoAxesCombined size={19} />
                Analyze Portfolio
              </>
            )}
          </button>

          {error && <div className="error-message">{error}</div>}
        </section>

        {result && (
          <>
            <div className="results-heading">
              <div>
                <h2>Analysis Results</h2>
                <p>
                  {result.start_date} to {result.end_date} ·{" "}
                  {result.observations.toLocaleString()} observations
                </p>
              </div>
              <span className="live-badge">Analysis complete</span>
            </div>

            <section className="metrics-grid">
              <MetricCard
                title="Total Return"
                value={`${(performance.total_return * 100).toFixed(2)}%`}
                subtitle="Historical cumulative return"
                icon={TrendingUp}
                color="green"
              />

              <MetricCard
                title="Annualized Return"
                value={`${(performance.annualized_return * 100).toFixed(2)}%`}
                subtitle="Compounded annual growth"
                icon={ArrowUpRight}
                color="blue"
              />

              <MetricCard
                title="Annualized Volatility"
                value={`${(risk.annualized_volatility * 100).toFixed(2)}%`}
                subtitle="Historical portfolio fluctuation"
                icon={Activity}
                color="orange"
              />

              <MetricCard
                title="Sharpe Ratio"
                value={risk.sharpe_ratio.toFixed(2)}
                subtitle="Risk-adjusted return"
                icon={Gauge}
                color="purple"
              />

              <MetricCard
                title="Maximum Drawdown"
                value={`${(risk.max_drawdown * 100).toFixed(2)}%`}
                subtitle="Largest historical peak-to-trough loss"
                icon={ArrowDownRight}
                color="red"
              />

              <MetricCard
                title="Market Beta"
                value={
                  risk.beta === null ? "N/A" : risk.beta.toFixed(2)
                }
                subtitle="Sensitivity to SPY"
                icon={ShieldAlert}
                color="blue"
              />
            </section>

            {marketRegime && riskForecast && (
              <section className="ai-insights-section">
                <div className="section-heading">
                  <div>
                    <h2>AI Risk Intelligence</h2>
                    <p>
                      Machine learning insights generated for your custom-weight portfolio.
                    </p>
                  </div>
                  <span className="live-badge">ML Analysis</span>
                </div>

                <section className="metrics-grid">
                  <MetricCard
                    title="Predicted Future Volatility"
                    value={`${(riskForecast.predicted_future_volatility * 100).toFixed(2)}%`}
                    subtitle="Model-estimated daily volatility"
                    icon={Activity}
                    color="orange"
                  />

                  <MetricCard
                    title="High-Risk Probability"
                    value={`${(riskForecast.high_risk_probability * 100).toFixed(2)}%`}
                    subtitle="Experimental classifier output"
                    icon={ShieldAlert}
                    color="red"
                  />
                </section>

                <section className="charts-grid">
                  <div className="chart-card">
                    <div className="chart-heading">
                      <div>
                        <h3>Gaussian Mixture Model (GMM)</h3>
                        <p>Current market regime</p>
                      </div>
                      <BrainCircuit size={22} />
                    </div>
                    <h2>{marketRegime.gmm_label}</h2>
                    <p className="chart-note">
                      State {marketRegime.gmm_state} · Identifies market conditions
                      from portfolio and macroeconomic features.
                    </p>
                  </div>

                  <div className="chart-card">
                    <div className="chart-heading">
                      <div>
                        <h3>Hidden Markov Model (HMM)</h3>
                        <p>Current market regime</p>
                      </div>
                      <Activity size={22} />
                    </div>
                    <h2>{marketRegime.hmm_label}</h2>
                    <p className="chart-note">
                      State {marketRegime.hmm_state} · Accounts for temporal
                      patterns in market conditions.
                    </p>
                  </div>
                </section>

                <div className="chart-card">
                  <h3>How to interpret these results</h3>
                  <p>
                    GMM and HMM provide complementary views of market conditions.
                    The forecasting models estimate future volatility and classify
                    elevated-risk conditions using historical patterns.
                  </p>
                  <p className="chart-note">
                    These are experimental model outputs for a hackathon prototype,
                    not calibrated probabilities or guaranteed forecasts.
                    Latest model data: {ml.latest_date}.
                  </p>
                </div>

                {shapFeatures.length > 0 && (
                  <div className="chart-card shap-card">
                    <div className="chart-heading">
                      <div>
                        <h3>AI Explainability · SHAP</h3>
                        <p>What influenced the predicted future volatility?</p>
                      </div>
                      <BrainCircuit size={22} />
                    </div>

                    <div className="shap-baseline">
                      <span>Model baseline</span>
                      <strong>
                        {(shapExplanation.baseline_prediction * 100).toFixed(3)}%
                      </strong>
                    </div>

                    <div className="shap-feature-list">
                      {shapFeatures.map((item) => {
                        const positive = item.shap_value > 0;
                        const width =
                          (Math.abs(item.shap_value) / maxShapImpact) * 100;

                        return (
                          <div className="shap-feature" key={item.feature}>
                            <div className="shap-feature-heading">
                              <span>{item.feature.replaceAll("_", " ")}</span>
                              <strong className={positive ? "shap-positive" : "shap-negative"}>
                                {positive ? "+" : ""}{item.shap_value.toFixed(6)}
                              </strong>
                            </div>
                            <div className="shap-track">
                              <div
                                className={`shap-bar ${positive ? "shap-bar-positive" : "shap-bar-negative"}`}
                                style={{ width: `${width}%` }}
                              />
                            </div>
                            <p className="chart-note">
                              {positive
                                ? "Increased the model's predicted volatility"
                                : item.shap_value < 0
                                  ? "Decreased the model's predicted volatility"
                                  : "Had no meaningful effect on the prediction"}
                            </p>
                          </div>
                        );
                      })}
                    </div>

                    <p className="chart-note shap-disclaimer">
                      SHAP values explain how each feature shifts this model's
                      prediction relative to its baseline. They show model
                      contributions, not causal effects.
                    </p>
                  </div>
                )}
              </section>
            )}

            <section className="charts-grid">
              <div className="chart-card">
                <div className="chart-heading">
                  <div>
                    <h3>Portfolio Growth</h3>
                    <p>Growth of $1 invested</p>
                  </div>
                  <span className="chart-value">
                    ${performance.cumulative_growth.toFixed(2)}
                  </span>
                </div>

                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient
                          id="growthGradient"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="0%"
                            stopColor="#14b8a6"
                            stopOpacity={0.3}
                          />
                          <stop
                            offset="95%"
                            stopColor="#14b8a6"
                            stopOpacity={0}
                          />
                        </linearGradient>
                      </defs>

                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#e9edf3"
                      />

                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 11 }}
                      />

                      <YAxis
                        tick={{ fontSize: 11 }}
                      />

                      <Tooltip />

                      <Area
                        type="monotone"
                        dataKey="value"
                        stroke="#0d9488"
                        strokeWidth={2.5}
                        fill="url(#growthGradient)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                <p className="chart-note">
                  Summary view. We will connect the full daily growth
                  history in the next iteration.
                </p>
              </div>

              <div className="chart-card risk-card">
                <h3>Risk Summary</h3>
                <p>Historical downside risk measures</p>

                <div className="risk-row">
                  <span>Value at Risk (95%)</span>
                  <strong className="negative">
                    {(risk.var_95 * 100).toFixed(2)}%
                  </strong>
                </div>

                <div className="risk-row">
                  <span>Expected Shortfall (95%)</span>
                  <strong className="negative">
                    {(risk.expected_shortfall_95 * 100).toFixed(2)}%
                  </strong>
                </div>

                <div className="risk-row">
                  <span>Maximum Drawdown</span>
                  <strong className="negative">
                    {(risk.max_drawdown * 100).toFixed(2)}%
                  </strong>
                </div>

                <div className="risk-row">
                  <span>Sharpe Ratio</span>
                  <strong>{risk.sharpe_ratio.toFixed(2)}</strong>
                </div>

                <div className="risk-row">
                  <span>Market Beta</span>
                  <strong>
                    {risk.beta === null ? "N/A" : risk.beta.toFixed(2)}
                  </strong>
                </div>
              </div>
            </section>

            <section className="correlation-card">
              <h3>Asset Correlation Matrix</h3>
              <p>
                Historical correlation between the assets in your portfolio.
              </p>

              <div className="correlation-table">
                <table>
                  <thead>
                    <tr>
                      <th>Asset</th>
                      {Object.keys(result.correlation_matrix).map(
                        (ticker) => (
                          <th key={ticker}>{ticker}</th>
                        )
                      )}
                    </tr>
                  </thead>

                  <tbody>
                    {Object.entries(result.correlation_matrix).map(
                      ([ticker, values]) => (
                        <tr key={ticker}>
                          <th>{ticker}</th>
                          {Object.values(values).map((value, index) => (
                            <td
                              key={index}
                              style={{
                                backgroundColor: `rgba(13, 148, 136, ${
                                  Math.abs(value) * 0.25
                                })`,
                              }}
                            >
                              {value.toFixed(2)}
                            </td>
                          ))}
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}

        {!result && (
          <div className="empty-state">
            <div className="empty-icon">
              <ChartNoAxesCombined size={30} />
            </div>
            <h3>Your analysis will appear here</h3>
            <p>
              Configure your portfolio above and click Analyze Portfolio
              to generate your results.
            </p>
          </div>
        )}

        <footer>
          PortfolioIQ · Historical analysis, not financial advice.
        </footer>
      </main>
    </div>
  );
}

export default App;