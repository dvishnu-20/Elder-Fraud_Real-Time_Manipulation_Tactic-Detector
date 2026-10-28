import { useState, useRef, useEffect } from 'react';
import { Activity, PhoneCall, AlertTriangle, MessageSquare } from 'lucide-react';
import { useWebsocket } from '../hooks/useWebsocket';

const TACTIC_LABELS: Record<string, string> = {
  urgency: 'Urgency',
  isolation: 'Isolation',
  authority: 'Authority',
  emotional_destabilization: 'Emotional Destabil.',
  channel_control: 'Channel Control'
};

const RISK_STATE_COLORS: Record<string, string> = {
  NORMAL: '#10b981',
  WATCH: '#f59e0b',
  ELEVATED: '#f97316',
  HIGH_RISK: '#ef4444',
  CRITICAL: '#dc2626'
};

export default function LiveMonitor() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState('00:00');

  const {
    transcripts, tactics, risk, intervention, isConnected, sessionStatus, tacticHistory
  } = useWebsocket(sessionId, 'demo');

  const transcriptEndRef = useRef<HTMLDivElement>(null);

  const startDemo = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/session/start', { method: 'POST' });
      const data = await res.json();
      setSessionId(data.session_id);
      setStartTime(Date.now());
    } catch (e) {
      console.error("Failed to start demo", e);
      alert("Backend connection unavailable. Please start the FastAPI server.");
    }
  };

  useEffect(() => {
    if (!startTime || sessionStatus === 'completed') return;
    const interval = setInterval(() => {
      const sec = Math.floor((Date.now() - startTime) / 1000);
      setElapsed(`${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(sec % 60).padStart(2, '0')}`);
    }, 1000);
    return () => clearInterval(interval);
  }, [startTime, sessionStatus]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcripts]);

  const riskState = risk?.risk_state || 'NORMAL';
  const riskColor = RISK_STATE_COLORS[riskState] || '#10b981';
  const temporalRisk = risk?.temporal_risk ?? 0;
  const riskPercent = Math.round(temporalRisk * 100);

  return (
    <div className="animate-fade-in" style={{ display: 'flex', gap: '24px', height: '100%' }}>
      
      {/* Left Column - Transcript & Actions */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div className="glass-panel" style={{ padding: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Active Session</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              {sessionId ? `ID: ${sessionId} | ${elapsed}` : 'No active session'}
            </p>
          </div>
          
          <div style={{ display: 'flex', gap: '12px' }}>
            {isConnected ? (
              <div className="status-badge live">
                <PhoneCall size={14} /> {sessionStatus === 'completed' ? 'COMPLETED' : 'LIVE'}
              </div>
            ) : (
              <button className="btn-primary" onClick={startDemo}>
                <Activity size={16} /> Start Demo
              </button>
            )}
          </div>
        </div>

        <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <MessageSquare size={18} color="var(--accent-primary)" />
            <h3 style={{ fontWeight: 600 }}>Live Transcript</h3>
          </div>
          
          <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {transcripts.length === 0 ? (
              <p style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: '40px' }}>
                Waiting for conversation to start...
              </p>
            ) : (
              transcripts.map((msg, i) => {
                const isLeft = ['SPEAKER_A', 'CALLER', 'CLINIC'].includes(msg.speaker);
                return (
                <div key={i} style={{ 
                  alignSelf: isLeft ? 'flex-start' : 'flex-end',
                  maxWidth: '80%',
                  backgroundColor: isLeft ? 'rgba(59,130,246,0.1)' : 'rgba(255,255,255,0.05)',
                  border: `1px solid ${isLeft ? 'rgba(59,130,246,0.2)' : 'var(--border-color)'}`,
                  padding: '12px 16px',
                  borderRadius: '12px',
                  borderBottomLeftRadius: isLeft ? 0 : '12px',
                  borderBottomRightRadius: !isLeft ? 0 : '12px'
                }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px', fontWeight: 600 }}>
                    {msg.speaker} <span style={{ marginLeft: '8px', opacity: 0.5 }}>Turn {msg.turn_id}</span>
                  </div>
                  <div style={{ fontSize: '0.95rem' }}>{msg.text}</div>
                  
                  {/* Inline Tactics */}
                  {tacticHistory[msg.turn_id - 1] && Object.values(tacticHistory[msg.turn_id - 1]).some(p => p > 0.1) && (
                    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '12px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '8px' }}>
                      {Object.entries(tacticHistory[msg.turn_id - 1])
                        .filter(([_, prob]) => prob > 0.1)
                        .map(([tactic, prob]) => (
                          <span key={tactic} style={{ 
                            fontSize: '0.7rem', padding: '2px 6px', 
                            backgroundColor: 'rgba(0,0,0,0.2)', 
                            borderRadius: '4px',
                            color: prob > 0.5 ? 'var(--accent-warning)' : 'var(--text-muted)'
                          }}>
                            {TACTIC_LABELS[tactic] || tactic}: {Math.round(prob * 100)}%
                          </span>
                      ))}
                    </div>
                  )}
                </div>
              );
              })
            )}
            <div ref={transcriptEndRef} />
          </div>
        </div>
      </div>

      {/* Right Column - Analysis */}
      <div style={{ width: '340px', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto' }}>
        
        {/* Risk Score */}
        <div className="glass-panel" style={{ padding: '24px', textAlign: 'center', borderTop: `4px solid ${riskColor}` }}>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.05em', marginBottom: '16px' }}>TEMPORAL RISK</h3>
          <div style={{ 
            width: '120px', height: '120px', borderRadius: '50%', border: `6px solid ${riskColor}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px',
            fontSize: '2.5rem', fontWeight: 800, color: riskColor
          }}>
            {riskPercent}%
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: riskColor }}>
            {riskState.replace('_', ' ')}
          </div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '8px' }}>
            Threshold: τ = {risk?.threshold ?? 0.20}
          </div>
        </div>

        {/* Detected Tactics */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <h3 style={{ fontSize: '0.9rem', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '0.05em', marginBottom: '16px' }}>DETECTED TACTICS</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {Object.entries(TACTIC_LABELS).map(([key, label]) => {
              const val = tactics[key] || 0;
              const pct = Math.round(val * 100);
              const isActive = pct > 15;
              return (
                <div key={key}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
                    <span style={{ color: isActive ? 'white' : 'var(--text-muted)', fontWeight: isActive ? 600 : 400 }}>{label}</span>
                    <span style={{ color: isActive ? riskColor : 'var(--text-muted)' }}>{pct}%</span>
                  </div>
                  <div style={{ width: '100%', height: '6px', backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ width: `${pct}%`, height: '100%', backgroundColor: isActive ? riskColor : '#4b5563', transition: 'width 0.3s ease' }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Intervention */}
        {intervention && (
          <div className="glass-panel" style={{ padding: '20px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
            <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
              <AlertTriangle color="var(--accent-danger)" size={24} style={{ flexShrink: 0 }} />
              <div>
                <h4 style={{ color: 'var(--accent-danger)', fontWeight: 700, marginBottom: '8px' }}>RECOMMENDED ACTION</h4>
                <p style={{ fontSize: '0.9rem', lineHeight: 1.4, marginBottom: '8px' }}>{intervention.message}</p>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Turn {intervention.turn_id}</div>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

