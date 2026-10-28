import { Link, Outlet, useLocation } from 'react-router-dom';
import { 
  Shield, LayoutDashboard, Activity, MessageSquare, 
  Target, TrendingUp, Lightbulb, History, FileText, 
  Cpu, Activity as HeartPulse, Lock, Info, LogOut
} from 'lucide-react';
import { useState } from 'react';

export default function MainLayout() {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);

  const navItems = [
    { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/live-monitor', label: 'Live Monitor', icon: Activity },
    { path: '/conversation', label: 'Conversation Analysis', icon: MessageSquare },
    { path: '/tactic-analysis', label: 'Tactic Analysis', icon: Target },
    { path: '/risk-analysis', label: 'Risk Analysis', icon: TrendingUp },
    { path: '/explanation', label: 'Explainability', icon: Lightbulb },
    { path: '/history', label: 'History', icon: History },
    { path: '/reports', label: 'Reports', icon: FileText },
    { path: '/model', label: 'Model Info', icon: Cpu },
    { path: '/system', label: 'System Health', icon: HeartPulse },
    { path: '/privacy', label: 'Privacy', icon: Lock },
    { path: '/about', label: 'About', icon: Info },
  ];

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', backgroundColor: 'var(--bg-dark)' }}>
      {/* Sidebar */}
      <aside style={{
        width: collapsed ? '80px' : '260px',
        backgroundColor: 'var(--bg-panel)',
        borderRight: '1px solid var(--border-color)',
        transition: 'width 0.3s',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <div style={{ padding: '20px', display: 'flex', alignItems: 'center', gap: '12px', borderBottom: '1px solid var(--border-color)' }}>
          <Shield color="var(--accent-primary)" size={28} onClick={() => setCollapsed(!collapsed)} style={{cursor: 'pointer'}} />
          {!collapsed && <span style={{ fontWeight: 'bold', fontSize: '1.1rem', whiteSpace: 'nowrap' }}>Elder-Fraud</span>}
        </div>
        
        <nav style={{ flex: 1, overflowY: 'auto', padding: '16px 12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {navItems.map((item) => {
            const active = location.pathname === item.path;
            const Icon = item.icon;
            return (
              <Link key={item.path} to={item.path} style={{
                display: 'flex', alignItems: 'center', gap: '12px', padding: '10px 12px',
                borderRadius: '8px',
                backgroundColor: active ? 'var(--accent-primary)' : 'transparent',
                color: active ? '#fff' : 'var(--text-muted)',
                textDecoration: 'none',
                transition: 'all 0.2s'
              }}>
                <Icon size={20} />
                {!collapsed && <span style={{ fontSize: '0.9rem', fontWeight: active ? 600 : 400 }}>{item.label}</span>}
              </Link>
            );
          })}
        </nav>

        <div style={{ padding: '16px', borderTop: '1px solid var(--border-color)' }}>
          <Link to="/login" style={{ display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--text-muted)', textDecoration: 'none', padding: '8px' }}>
            <LogOut size={20} />
            {!collapsed && <span>Logout</span>}
          </Link>
        </div>
      </aside>

      {/* Main Content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Top bar */}
        <header style={{ 
          height: '64px', backgroundColor: 'var(--bg-panel)', borderBottom: '1px solid var(--border-color)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px'
        }}>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>
            {navItems.find(i => i.path === location.pathname)?.label || 'System'}
          </h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div className="status-badge live">API: CONNECTED</div>
            <div className="status-badge live">WS: CONNECTED</div>
          </div>
        </header>

        {/* Page Content */}
        <main style={{ flex: 1, overflowY: 'auto', padding: '24px' }}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
