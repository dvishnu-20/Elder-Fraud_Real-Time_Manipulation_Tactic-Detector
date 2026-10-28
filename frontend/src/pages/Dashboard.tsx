import { Activity, ShieldAlert, Users, Zap, CheckCircle2 } from 'lucide-react';

export default function Dashboard() {
  return (
    <div className="animate-fade-in">
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '20px', marginBottom: '32px' }}>
        <KpiCard icon={Activity} title="Active Sessions" value="1" trend="+1" />
        <KpiCard icon={Users} title="Calls Analyzed" value="1,248" trend="+12%" />
        <KpiCard icon={ShieldAlert} title="Risk Alerts" value="84" trend="-5%" />
        <KpiCard icon={Zap} title="Avg Latency" value="32ms" trend="Stable" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '20px' }}>Recent Alerts</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                <th style={{ padding: '12px', fontWeight: 500, fontSize: '0.9rem' }}>Time</th>
                <th style={{ padding: '12px', fontWeight: 500, fontSize: '0.9rem' }}>Session</th>
                <th style={{ padding: '12px', fontWeight: 500, fontSize: '0.9rem' }}>Risk Level</th>
                <th style={{ padding: '12px', fontWeight: 500, fontSize: '0.9rem' }}>Dominant Tactic</th>
              </tr>
            </thead>
            <tbody>
              {[
                { time: '10:42 AM', session: 'S-4921', risk: 'CRITICAL', tactic: 'Authority' },
                { time: '09:15 AM', session: 'S-4918', risk: 'HIGH_RISK', tactic: 'Urgency' },
                { time: 'Yesterday', session: 'S-4890', risk: 'ELEVATED', tactic: 'Isolation' },
              ].map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '12px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>{row.time}</td>
                  <td style={{ padding: '12px', fontSize: '0.9rem' }}>{row.session}</td>
                  <td style={{ padding: '12px' }}>
                    <span style={{ 
                      padding: '4px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600,
                      backgroundColor: row.risk === 'CRITICAL' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                      color: row.risk === 'CRITICAL' ? '#ef4444' : '#f59e0b'
                    }}>
                      {row.risk}
                    </span>
                  </td>
                  <td style={{ padding: '12px', fontSize: '0.9rem' }}>{row.tactic}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '20px' }}>System Status</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <StatusRow label="FastAPI Backend" status="READY" />
            <StatusRow label="Tactic Classifier (DistilBERT)" status="READY" />
            <StatusRow label="Temporal Model (Causal LSTM)" status="READY" />
            <StatusRow label="WebSocket Gateway" status="CONNECTED" />
            <StatusRow label="Intervention Engine" status="READY" />
          </div>
        </div>
      </div>
    </div>
  );
}

function KpiCard({ icon: Icon, title, value, trend }: any) {
  return (
    <div className="glass-panel" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
        <div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', fontWeight: 500 }}>{title}</p>
          <h3 style={{ fontSize: '2rem', fontWeight: 700, marginTop: '4px' }}>{value}</h3>
        </div>
        <div style={{ padding: '10px', backgroundColor: 'rgba(59,130,246,0.1)', borderRadius: '12px' }}>
          <Icon size={24} color="var(--accent-primary)" />
        </div>
      </div>
      <div style={{ fontSize: '0.85rem', color: trend.startsWith('+') || trend === 'Stable' ? 'var(--accent-success)' : 'var(--accent-danger)' }}>
        {trend} from previous period
      </div>
    </div>
  );
}

function StatusRow({ label, status }: { label: string, status: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '12px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
      <span style={{ fontSize: '0.9rem' }}>{label}</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-success)' }}>
        <CheckCircle2 size={16} />
        <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{status}</span>
      </div>
    </div>
  );
}
