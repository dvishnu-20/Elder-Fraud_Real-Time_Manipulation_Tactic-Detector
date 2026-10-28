import { Shield, Cpu } from 'lucide-react';

export default function ModelInfo() {
  return (
    <div className="animate-fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      
      <div className="glass-panel" style={{ padding: '32px', marginBottom: '24px' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Cpu color="var(--accent-primary)" />
          Model Architecture
        </h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
          <div style={{ padding: '20px', backgroundColor: 'var(--bg-dark)', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: 'var(--accent-primary)' }}>1. Tactic Classifier (Phase 9)</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '16px' }}>
              A fine-tuned DistilBERT transformer that operates on individual conversational turns to extract a 5-dimensional probability vector.
            </p>
            <ul style={{ fontSize: '0.85rem', color: 'var(--text-muted)', paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <li><strong>Architecture:</strong> DistilBERT (distilbert-base-uncased)</li>
              <li><strong>Outputs:</strong> 5 Binary Classification Heads</li>
              <li><strong>Training Data:</strong> Synthetic + CFPB + FTC + ASR Augmentation</li>
              <li><strong>Status:</strong> FROZEN</li>
            </ul>
          </div>

          <div style={{ padding: '20px', backgroundColor: 'var(--bg-dark)', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: 'var(--accent-elevated)' }}>2. Temporal Risk Engine (Phase 5)</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '16px' }}>
              A Causal LSTM that aggregates the 5-dimensional turn probabilities over time to calculate the evolving manipulation risk.
            </p>
            <ul style={{ fontSize: '0.85rem', color: 'var(--text-muted)', paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <li><strong>Architecture:</strong> Causal LSTM</li>
              <li><strong>Input Dimension:</strong> 5</li>
              <li><strong>Hidden Dimension:</strong> 64</li>
              <li><strong>Threshold (τ):</strong> 0.20</li>
              <li><strong>Status:</strong> FROZEN</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '32px' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Shield color="var(--accent-primary)" />
          Model Integrity Hashes
        </h2>
        <div style={{ backgroundColor: 'var(--bg-dark)', padding: '16px', borderRadius: '8px', fontFamily: 'monospace', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          <div><span style={{color: 'white'}}>DistilBERT V2 SHA256:</span> 76d2d4e440f65cb1e9f275a6496a0dcd6751a35e1f85261882815ccdda59a684</div>
          <div style={{marginTop: '12px'}}><span style={{color: 'white'}}>Causal LSTM V2 SHA256:</span> c6dd5b03ba91a20677ed4c7a148d23cba26ff08fa1528d5603a0cb1bd2ae64a1</div>
        </div>
      </div>

    </div>
  );
}
