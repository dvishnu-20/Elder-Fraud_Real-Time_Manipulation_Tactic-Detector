import { Search, Filter, Loader2 } from 'lucide-react';
import { useState, useEffect } from 'react';

type SessionData = {
  session_id: string;
  started_at: string;
  status: string;
  turn_count: number;
  current_risk: number;
  risk_state: string;
};

export default function History() {
  const [sessions, setSessions] = useState<SessionData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/api/sessions')
      .then(res => res.json())
      .then(data => {
        // Handle if backend returns a list or a dict containing a list
        const s = Array.isArray(data.sessions) ? data.sessions : [];
        setSessions(s.reverse()); // Show newest first
        setLoading(false);
      })
      .catch(e => {
        console.error("Failed to load sessions", e);
        setLoading(false);
      });
  }, []);

  return (
    <div className="animate-fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700 }}>Session History</h2>
        <div style={{ fontSize: '0.85rem', color: 'var(--accent-warning)', padding: '6px 12px', backgroundColor: 'rgba(245, 158, 11, 0.1)', borderRadius: '6px' }}>
          Session history available for current runtime only.
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', gap: '16px', marginBottom: '24px' }}>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 16px', backgroundColor: 'var(--bg-dark)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <Search size={18} color="var(--text-muted)" />
            <input type="text" placeholder="Search sessions..." style={{ background: 'transparent', border: 'none', color: 'white', outline: 'none', width: '100%' }} />
          </div>
          <button className="btn-secondary">
            <Filter size={18} /> Filter
          </button>
        </div>

        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Loader2 className="animate-spin" style={{ margin: '0 auto 16px' }} />
            Loading sessions...
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                <th style={{ padding: '12px', fontWeight: 500 }}>Session ID</th>
                <th style={{ padding: '12px', fontWeight: 500 }}>Date</th>
                <th style={{ padding: '12px', fontWeight: 500 }}>Turns</th>
                <th style={{ padding: '12px', fontWeight: 500 }}>Risk Level</th>
                <th style={{ padding: '12px', fontWeight: 500 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {sessions.length === 0 ? (
                <tr>
                  <td colSpan={5} style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    No sessions recorded in this runtime yet.
                  </td>
                </tr>
              ) : (
                sessions.map((s, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <td style={{ padding: '12px', color: 'var(--accent-primary)', fontSize: '0.85rem' }}>{s.session_id}</td>
                    <td style={{ padding: '12px', color: 'var(--text-muted)' }}>{new Date(s.started_at).toLocaleString()}</td>
                    <td style={{ padding: '12px' }}>{s.turn_count}</td>
                    <td style={{ padding: '12px' }}>
                      <span className={`status-badge ${s.risk_state === 'CRITICAL' ? 'error' : 'live'}`}>
                        {s.risk_state} ({(s.current_risk * 100).toFixed(1)}%)
                      </span>
                    </td>
                    <td style={{ padding: '12px', color: s.status === 'active' ? 'var(--accent-primary)' : 'var(--text-muted)' }}>
                      {s.status.toUpperCase()}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
