import { Activity, CheckCircle2, Server, Database, Network } from 'lucide-react';

export default function SystemHealth() {
  return (
    <div className="animate-fade-in" style={{ maxWidth: '800px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '24px' }}>System Diagnostics</h2>
      
      <div style={{ display: 'grid', gap: '16px' }}>
        <HealthRow icon={Server} name="FastAPI API Server" status="CONNECTED" detail="Uptime: 24h 12m" />
        <HealthRow icon={Network} name="WebSocket Gateway" status="CONNECTED" detail="Latency: 32ms" />
        <HealthRow icon={Database} name="DistilBERT Tactic Classifier" status="READY" detail="Loaded in memory" />
        <HealthRow icon={Activity} name="Causal LSTM Risk Engine" status="READY" detail="Loaded in memory" />
        <HealthRow icon={Server} name="Session Manager" status="READY" detail="Active Sessions: 1" />
      </div>

      <div style={{ marginTop: '32px', textAlign: 'center' }}>
        <button className="btn-primary">Run Full System Check</button>
      </div>
    </div>
  );
}

function HealthRow({ icon: Icon, name, status, detail }: any) {
  return (
    <div className="glass-panel" style={{ padding: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div style={{ padding: '10px', backgroundColor: 'var(--bg-dark)', borderRadius: '8px' }}>
          <Icon color="var(--accent-primary)" />
        </div>
        <div>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 600 }}>{name}</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{detail}</p>
        </div>
      </div>
      <div className="status-badge live">
        <CheckCircle2 size={14} /> {status}
      </div>
    </div>
  );
}
