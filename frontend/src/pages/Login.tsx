import { Link, useNavigate } from 'react-router-dom';
import { Shield } from 'lucide-react';
import { useState } from 'react';

export default function Login() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    // Simulate auth
    setTimeout(() => {
      navigate('/dashboard');
    }, 800);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-dark)' }}>
      <div className="glass-panel animate-fade-in" style={{ width: '100%', maxWidth: '420px', padding: '40px' }}>
        
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginBottom: '32px' }}>
          <Shield size={48} color="var(--accent-primary)" style={{ marginBottom: '16px' }} />
          <h2 style={{ fontSize: '1.5rem', fontWeight: 600 }}>System Login</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Enter your credentials to access the dashboard</p>
        </div>

        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '8px', color: 'var(--text-muted)' }}>Email / Researcher ID</label>
            <input 
              type="text" 
              defaultValue="demo@research.local"
              style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: 'white', outline: 'none' }} 
            />
          </div>
          
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', marginBottom: '8px', color: 'var(--text-muted)' }}>Password</label>
            <input 
              type="password" 
              defaultValue="password"
              style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)', background: 'var(--bg-dark)', color: 'white', outline: 'none' }} 
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input type="checkbox" defaultChecked />
              Remember me
            </label>
            <span style={{ cursor: 'pointer', color: 'var(--accent-primary)' }}>Forgot Password?</span>
          </div>

          <button type="submit" className="btn-primary" style={{ marginTop: '12px' }} disabled={loading}>
            {loading ? 'Authenticating...' : 'Secure Login'}
          </button>
        </form>

        <div style={{ marginTop: '24px', textAlign: 'center' }}>
          <Link to="/" style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>← Back to Cover</Link>
        </div>
      </div>
    </div>
  );
}
