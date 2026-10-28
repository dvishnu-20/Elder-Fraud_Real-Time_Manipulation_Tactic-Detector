import { Lock, ShieldCheck, EyeOff } from 'lucide-react';

export default function Privacy() {
  return (
    <div className="animate-fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      
      <div className="glass-panel" style={{ padding: '40px', textAlign: 'center', marginBottom: '24px' }}>
        <Lock size={48} color="var(--accent-primary)" style={{ margin: '0 auto 20px' }} />
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '16px' }}>Privacy & Security Architecture</h2>
        <p style={{ color: 'var(--text-muted)', maxWidth: '600px', margin: '0 auto', fontSize: '1.1rem' }}>
          Privacy validation was performed under the project's evaluated audit protocol. The system is designed to protect PII while performing real-time analysis.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <ShieldCheck size={28} color="var(--accent-success)" style={{ marginBottom: '16px' }} />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px' }}>Audio & Transcript Policy</h3>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
            No raw audio is stored on disk permanently. Transcripts are held in volatile memory during the active session. Once a session is terminated or risk analysis is complete, the transcript buffers are explicitly flushed.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '24px' }}>
          <EyeOff size={28} color="var(--accent-primary)" style={{ marginBottom: '16px' }} />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px' }}>PII Redaction Matrix</h3>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: '16px' }}>
            The following categories are intercepted and redacted before being passed to the Tactic Classifier:
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {['Name', 'Phone Number', 'Email', 'Credit Card', 'Bank Account', 'SSN', 'OTP'].map(cat => (
              <span key={cat} style={{ padding: '4px 10px', backgroundColor: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '999px', fontSize: '0.8rem' }}>
                {cat}
              </span>
            ))}
          </div>
        </div>
      </div>

    </div>
  );
}
