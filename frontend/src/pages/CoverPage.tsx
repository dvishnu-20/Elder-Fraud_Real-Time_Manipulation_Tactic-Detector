import { Link } from 'react-router-dom';
import { Shield, AlertTriangle, EyeOff, Activity, ShieldAlert, PhoneCall } from 'lucide-react';

export default function CoverPage() {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '60px 20px', background: 'radial-gradient(circle at top, var(--bg-panel), var(--bg-dark))' }}>
      
      <div style={{ display: 'inline-flex', alignItems: 'center', gap: '12px', padding: '8px 16px', borderRadius: '999px', background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)', marginBottom: '32px' }}>
        <Shield size={16} color="var(--accent-primary)" />
        <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-primary)', letterSpacing: '0.05em' }}>VALIDATED RESEARCH PROTOTYPE</span>
      </div>

      <h1 style={{ fontSize: '4rem', fontWeight: 800, textAlign: 'center', marginBottom: '16px', lineHeight: 1.1 }} className="text-gradient">
        ELDER-FRAUD DETECTOR
      </h1>
      
      <h2 style={{ fontSize: '1.5rem', color: 'var(--text-muted)', textAlign: 'center', maxWidth: '800px', marginBottom: '48px', fontWeight: 400 }}>
        Real-Time Psychological Manipulation & Scam Tactic Detection
      </h2>

      <div style={{ display: 'flex', gap: '16px', marginBottom: '80px' }}>
        <Link to="/login" className="btn-primary" style={{ padding: '12px 24px', fontSize: '1.1rem' }}>
          ENTER SYSTEM
        </Link>
        <button className="btn-secondary" style={{ padding: '12px 24px', fontSize: '1.1rem' }}>
          View Project Details
        </button>
        <button className="btn-secondary" style={{ padding: '12px 24px', fontSize: '1.1rem' }}>
          System Architecture
        </button>
      </div>

      <div style={{ maxWidth: '900px', width: '100%', marginBottom: '80px' }}>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '24px', textAlign: 'center' }}>The 5 Detected Manipulation Tactics</h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
          <TacticCard icon={Activity} title="URGENCY" desc="Deadlines, countdowns, and forcing rapid action to prevent critical thinking." color="#ef4444" />
          <TacticCard icon={EyeOff} title="ISOLATION" desc="Telling the victim to hide the conversation from family, bank tellers, or police." color="#f59e0b" />
          <TacticCard icon={ShieldAlert} title="AUTHORITY" desc="Impersonating IRS, FBI, Bank Fraud Dept, or Tech Support." color="#3b82f6" />
          <TacticCard icon={AlertTriangle} title="EMOTIONAL DESTABILIZATION" desc="Inducing panic, confusion, or fear to override logical reasoning." color="#8b5cf6" />
          <TacticCard icon={PhoneCall} title="CHANNEL CONTROL" desc="Demanding the victim stay on the phone or go to a quiet room." color="#10b981" />
        </div>
      </div>

      <div className="glass-panel" style={{ maxWidth: '900px', width: '100%', padding: '40px', textAlign: 'center' }}>
        <h3 style={{ marginBottom: '16px' }}>System Overview</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', maxWidth: '700px', margin: '0 auto', lineHeight: 1.6 }}>
          The Elder-Fraud Detector is a real-time system designed to identify psychological manipulation tactics occurring during suspicious conversations. It analyzes conversational turns, estimates tactic probabilities, models temporal progression, and produces explainable risk signals.
        </p>
      </div>

    </div>
  );
}

function TacticCard({ icon: Icon, title, desc, color }: any) {
  return (
    <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '12px', borderTop: `3px solid ${color}` }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <Icon color={color} size={24} />
        <h4 style={{ fontWeight: 700, letterSpacing: '0.05em' }}>{title}</h4>
      </div>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>{desc}</p>
    </div>
  );
}
