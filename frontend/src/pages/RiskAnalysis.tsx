import { TrendingUp, Loader2 } from 'lucide-react';
import { useState, useEffect } from 'react';

export default function RiskAnalysis() {
  const [riskData, setRiskData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch latest session and then its risk history
    fetch('http://localhost:8000/api/sessions')
      .then(res => res.json())
      .then(data => {
        const sessions = Array.isArray(data.sessions) ? data.sessions : [];
        if (sessions.length > 0) {
          const latest = sessions[sessions.length - 1]; // Assumes newest is last or first depending on backend ordering
          return fetch(`http://localhost:8000/api/session/${latest.session_id}/risk`);
        }
        throw new Error("No sessions found");
      })
      .then(res => res.json())
      .then(data => {
        setRiskData(data);
        setLoading(false);
      })
      .catch(e => {
        console.error(e);
        setLoading(false);
      });
  }, []);

  return (
    <div className="animate-fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px' }}>Temporal Risk Analysis</h2>
      
      <div className="glass-panel" style={{ padding: '32px', marginBottom: '24px' }}>
        <p style={{ color: 'var(--text-muted)', marginBottom: '32px', fontSize: '1.05rem', lineHeight: 1.6 }}>
          The temporal model evaluates the progression of tactic probabilities across conversation turns. It uses a causal state architecture (Causal LSTM V2), meaning it predicts current risk using only the available conversational history, mimicking real-time streaming constraints perfectly.
        </p>

        <div style={{ height: '300px', width: '100%', backgroundColor: 'var(--bg-dark)', borderRadius: '12px', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', position: 'relative' }}>
          
          {loading ? (
             <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
               <Loader2 className="animate-spin" style={{ margin: '0 auto 16px' }} />
               Loading latest risk data...
             </div>
          ) : !riskData || !riskData.risk_history || riskData.risk_history.length === 0 ? (
            <>
              <TrendingUp size={48} color="var(--accent-primary)" style={{ opacity: 0.5, marginBottom: '16px' }} />
              <p style={{ color: 'var(--text-muted)' }}>Historical risk graph will populate when a session is active.</p>
            </>
          ) : (
            <div style={{ width: '100%', height: '100%', padding: '20px', display: 'flex', alignItems: 'flex-end', gap: '8px', position: 'relative' }}>
              {/* Threshold Line */}
              <div style={{ position: 'absolute', left: '20px', right: '20px', borderTop: '2px dashed var(--accent-error)', zIndex: 0, bottom: `calc(20px + (100% - 40px) * 0.20)` }}>
                <span style={{ position: 'absolute', right: '0', top: '-20px', color: 'var(--accent-error)', fontSize: '0.75rem', fontWeight: 600 }}>Threshold (0.20)</span>
              </div>
              
              {/* Bars */}
              {riskData.risk_history.map((r: number, i: number) => {
                const heightPct = Math.max(r * 100, 2); // Minimum 2% height for visibility
                const isOver = r > 0.20;
                return (
                  <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', height: '100%', zIndex: 1 }}>
                    <div style={{ 
                      width: '100%', 
                      height: `${heightPct}%`, 
                      backgroundColor: isOver ? 'var(--accent-error)' : 'var(--accent-primary)',
                      borderTopLeftRadius: '4px',
                      borderTopRightRadius: '4px',
                      transition: 'height 0.3s ease'
                    }} title={`Turn ${i+1}: ${(r*100).toFixed(1)}%`} />
                    <div style={{ textAlign: 'center', fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: '4px' }}>{i+1}</div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
        <StatBox label="Average Latency" value="~32ms" />
        <StatBox label="Detection Threshold" value="0.20 (τ)" />
        <StatBox label="Latest Risk Peak" value={riskData ? `${(Math.max(...(riskData.risk_history || [0])) * 100).toFixed(1)}%` : "0.0%"} />
      </div>
    </div>
  );
}

function StatBox({ label, value }: any) {
  return (
    <div className="glass-panel" style={{ padding: '20px', textAlign: 'center' }}>
      <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '8px' }}>{label}</div>
      <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{value}</div>
    </div>
  );
}
