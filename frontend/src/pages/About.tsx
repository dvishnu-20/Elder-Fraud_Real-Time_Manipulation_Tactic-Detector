import { Info } from 'lucide-react';

export default function About() {
  return (
    <div className="animate-fade-in" style={{ maxWidth: '800px', margin: '0 auto' }}>
      <div className="glass-panel" style={{ padding: '40px' }}>
        <Info size={40} color="var(--accent-primary)" style={{ marginBottom: '20px' }} />
        <h2 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '24px' }}>About the Project</h2>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', color: 'var(--text-muted)', lineHeight: 1.7 }}>
          <section>
            <h3 style={{ color: 'white', fontSize: '1.2rem', marginBottom: '8px' }}>Problem Statement</h3>
            <p>Elder fraud is a billion-dollar problem exacerbated by psychological manipulation. Traditional detection systems rely on keyword blocking, which is easily bypassed. This project aims to detect the underlying manipulation tactics mathematically in real-time.</p>
          </section>

          <section>
            <h3 style={{ color: 'white', fontSize: '1.2rem', marginBottom: '8px' }}>Machine Learning Architecture</h3>
            <p>The system utilizes a decoupled architecture. A fine-tuned DistilBERT transformer acts as a Turn-Level Tactic Classifier. Its 5-dimensional probability outputs are fed into a Causal LSTM Temporal Risk Engine, which tracks the escalation of manipulation over the conversation.</p>
          </section>

          <section>
            <h3 style={{ color: 'white', fontSize: '1.2rem', marginBottom: '8px' }}>Phase 9 Validation</h3>
            <p>Following rigorous Phase 8 pilot testing which revealed a domain shift vulnerability, Phase 9 successfully trained the classifier on curated real-world conversational data (CFPB, FTC, DailyDialog) and ASR augmentations, resulting in a robust, real-world ready system.</p>
          </section>
        </div>
      </div>
    </div>
  );
}
