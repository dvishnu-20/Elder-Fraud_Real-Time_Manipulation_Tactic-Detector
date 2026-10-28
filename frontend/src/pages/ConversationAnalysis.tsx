import { Play, RefreshCw } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useWebsocket } from '../hooks/useWebsocket';

export default function ConversationAnalysis() {
  const [text, setText] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  
  const { riskEvents, sendTurn, resetState, isConnected } = useWebsocket(sessionId, 'live');

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    
    // Create new session
    const id = `offline_${Date.now()}`;
    setSessionId(id);
    resetState();
    setAnalyzing(true);
    
    // We wait slightly to ensure WS is connected before sending
  };

  // When WS connects and analyzing is true, process the text
  useEffect(() => {
    if (analyzing && isConnected) {
      const lines = text.split('\n').filter(l => l.trim().length > 0);
      let i = 0;
      
      const interval = setInterval(() => {
        if (i < lines.length) {
          const line = lines[i];
          const speakerMatch = line.match(/^([^:]+):(.*)$/);
          if (speakerMatch) {
            sendTurn(speakerMatch[2].trim(), speakerMatch[1].trim());
          } else {
            sendTurn(line.trim(), "UNKNOWN");
          }
          i++;
        } else {
          clearInterval(interval);
          setAnalyzing(false);
        }
      }, 1000); // 1 second per turn for visualization
      
      return () => clearInterval(interval);
    }
  }, [analyzing, isConnected, text, sendTurn]);

  return (
    <div className="animate-fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px' }}>Offline Conversation Analysis</h2>
      
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '16px' }}>Paste Transcript</h3>
        <textarea 
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="e.g.&#10;CALLER: You need to act now.&#10;RECIPIENT: Ok, what do I do?"
          style={{ width: '100%', height: '200px', backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '16px', color: 'white', resize: 'vertical', marginBottom: '16px', fontFamily: 'monospace' }}
        />
        <div style={{ display: 'flex', gap: '16px' }}>
          <button className="btn-primary" onClick={handleAnalyze} disabled={analyzing || text.length === 0}>
            {analyzing ? <RefreshCw size={18} className="animate-spin" /> : <Play size={18} />} 
            {analyzing ? 'Analyzing...' : 'Analyze Turn-by-Turn'}
          </button>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '32px' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '24px' }}>Analysis Results</h3>
        {riskEvents.length === 0 && !analyzing && (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
            Turn-by-turn analysis will appear here.
          </div>
        )}
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {riskEvents.map((event, idx) => (
            <div key={idx} style={{ padding: '16px', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: '8px', borderLeft: `4px solid ${event.risk_triggered ? 'var(--accent-error)' : 'var(--accent-primary)'}` }}>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
                Turn {event.turn_id} | Risk: {(event.temporal_risk * 100).toFixed(1)}% | Status: {event.risk_state}
              </div>
              <div style={{ fontWeight: 500, marginBottom: '12px' }}>
                <span style={{ color: 'var(--accent-primary)', marginRight: '8px' }}>{event.speaker}:</span>
                {event.transcript}
              </div>
              
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {Object.entries(event.tactics)
                  .filter(([_, prob]) => prob > 0.1)
                  .map(([tactic, prob]) => (
                    <span key={tactic} style={{ fontSize: '0.8rem', padding: '4px 8px', backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: '4px' }}>
                      {tactic}: {(prob * 100).toFixed(0)}%
                    </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
