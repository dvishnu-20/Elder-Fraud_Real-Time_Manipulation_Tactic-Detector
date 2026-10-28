export default function TacticAnalysis() {
  return (
    <div className="animate-fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px' }}>Tactic Analysis</h2>
      
      <div style={{ display: 'grid', gap: '24px' }}>
        <TacticRow name="Urgency" color="#ef4444" trend="Increasing" val="85%" desc="Creating artificial deadlines to bypass critical thinking." />
        <TacticRow name="Isolation" color="#f59e0b" trend="Stable" val="12%" desc="Demanding the victim lie to protective parties." />
        <TacticRow name="Authority" color="#3b82f6" trend="High" val="92%" desc="Impersonating trusted entities (IRS, FBI, Bank)." />
        <TacticRow name="Emotional Destabilization" color="#8b5cf6" trend="Low" val="4%" desc="Using fear, panic, or confusion." />
        <TacticRow name="Channel Control" color="#10b981" trend="Stable" val="0%" desc="Refusing to let the victim disconnect the call." />
      </div>
    </div>
  );
}

function TacticRow({ name, color, trend, val, desc }: any) {
  return (
    <div className="glass-panel" style={{ padding: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderLeft: `4px solid ${color}` }}>
      <div style={{ maxWidth: '60%' }}>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '8px' }}>{name}</h3>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>{desc}</p>
      </div>
      <div style={{ textAlign: 'right' }}>
        <div style={{ fontSize: '2rem', fontWeight: 700, color: color }}>{val}</div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{trend}</div>
      </div>
    </div>
  );
}
