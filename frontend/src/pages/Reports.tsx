import { FileText, CheckCircle2, AlertTriangle } from 'lucide-react';

export default function Reports() {
  const reports = [
    { phase: 'Phase 5', title: 'Temporal Validation', status: 'PASS', verdict: 'MODEST TEMPORAL VALUE' },
    { phase: 'Phase 6', title: 'External Generalization', status: 'PASS', verdict: 'MODERATE GENERALIZATION' },
    { phase: 'Phase 7', title: 'Real-Time Integration', status: 'PASS', verdict: 'ENGINEERING INTEGRATION PASS' },
    { phase: 'Phase 8', title: 'Controlled Pilot', status: 'WARNING', verdict: 'LIMITED (Domain Shift Found)' },
    { phase: 'Phase 8.1', title: 'Root Cause Analysis', status: 'PASS', verdict: 'CLASSIFIER LIMITATION CONFIRMED' },
    { phase: 'Phase 9', title: 'Robustness Improvement', status: 'PASS', verdict: 'STRONG ROBUSTNESS IMPROVEMENT' },
  ];

  return (
    <div className="animate-fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px' }}>Scientific Reports</h2>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {reports.map((r, i) => (
          <div key={i} className="glass-panel" style={{ padding: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderLeft: `4px solid ${r.status === 'PASS' ? 'var(--accent-success)' : 'var(--accent-warning)'}` }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
              <div style={{ padding: '12px', backgroundColor: 'var(--bg-dark)', borderRadius: '12px' }}>
                <FileText color="var(--accent-primary)" />
              </div>
              <div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '4px' }}>{r.phase}</div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>{r.title}</h3>
              </div>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '8px' }}>
              {r.status === 'PASS' ? (
                <div className="status-badge live"><CheckCircle2 size={14} /> PASS</div>
              ) : (
                <div className="status-badge warning"><AlertTriangle size={14} /> WARNING</div>
              )}
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Verdict: {r.verdict}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
