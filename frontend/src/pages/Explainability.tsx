import { AlertTriangle } from 'lucide-react';

export default function Explainability() {
  return (
    <div className="animate-fade-in" style={{ maxWidth: '900px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px' }}>Explainability Engine</h2>
      
      <div className="glass-panel" style={{ padding: '32px', marginBottom: '24px', backgroundColor: 'rgba(239, 68, 68, 0.05)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
          <AlertTriangle color="var(--accent-danger)" size={32} />
          <div>
            <h3 style={{ fontSize: '1.25rem', color: 'var(--accent-danger)', fontWeight: 600, marginBottom: '8px' }}>ALERT REASON</h3>
            <p style={{ fontSize: '1.1rem', marginBottom: '16px' }}>"Repeated urgency language combined with isolation commands increased the estimated manipulation risk beyond the safety threshold."</p>
          </div>
        </div>

        <div style={{ marginTop: '24px', padding: '24px', backgroundColor: 'var(--bg-dark)', borderRadius: '12px' }}>
          <h4 style={{ fontSize: '1rem', color: 'var(--text-muted)', marginBottom: '16px' }}>SUPPORTING EVIDENCE</h4>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ paddingLeft: '16px', borderLeft: '2px solid var(--accent-primary)' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Turn 3 (Urgency)</div>
              <div>"Act immediately to secure your funds."</div>
            </div>
            <div style={{ paddingLeft: '16px', borderLeft: '2px solid var(--accent-warning)' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Turn 7 (Isolation)</div>
              <div>"Do not tell the bank teller why you are withdrawing cash."</div>
            </div>
          </div>
        </div>
        
        <div style={{ marginTop: '24px', fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'center' }}>
          * Model-generated explanation based on causal attribution of LSTM cell states.
        </div>
      </div>
    </div>
  );
}
