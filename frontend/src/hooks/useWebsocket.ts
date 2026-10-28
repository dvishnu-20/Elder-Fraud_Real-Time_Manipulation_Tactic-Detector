import { useState, useEffect, useRef, useCallback } from 'react';

export type TranscriptMsg = {
    type: 'transcript';
    turn_id: number;
    speaker: string;
    text: string;
    timestamp: string;
};



export type RiskMsg = {
    type: 'risk_update';
    turn_id: number;
    temporal_risk: number;
    threshold: number;
    risk_triggered: boolean;
    risk_state: string;
    first_trigger_turn: number | null;
    risk_history: number[];
};

export type RiskEventMsg = {
    type: 'risk_event';
    turn_id: number;
    transcript: string;
    speaker: string;
    tactics: Record<string, number>;
    temporal_risk: number;
    threshold: number;
    risk_triggered: boolean;
    risk_state: string;
    classification_latency_ms: number;
    temporal_latency_ms: number;
    total_latency_ms: number;
};

export type ExplanationMsg = {
    type: 'explanation';
    summary: string;
    active_tactics: Array<{
        tactic: string;
        display_name: string;
        probability: number;
        description: string;
    }>;
    risk_trend: string;
    alert_level: string;
    alert_message: string | null;
    temporal_risk: number;
    turn_id: number;
};

export type InterventionMsg = {
    type: 'intervention';
    severity: string;
    message: string;
    turn_id: number;
    temporal_risk: number;
};

export type SessionMsg = {
    type: 'session_start' | 'session_end';
    session_id?: string;
    mode?: string;
    reason?: string;
};

export function useWebsocket(sessionId: string | null, mode: 'live' | 'demo' = 'demo') {
    const [transcripts, setTranscripts] = useState<TranscriptMsg[]>([]);
    const [tactics, setTactics] = useState<Record<string, number>>({});
    const [risk, setRisk] = useState<RiskMsg | null>(null);
    const [riskEvents, setRiskEvents] = useState<RiskEventMsg[]>([]);
    const [explanation, setExplanation] = useState<ExplanationMsg | null>(null);
    const [intervention, setIntervention] = useState<InterventionMsg | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [sessionStatus, setSessionStatus] = useState<string>('idle');
    const [riskHistory, setRiskHistory] = useState<number[]>([]);
    const [tacticHistory, setTacticHistory] = useState<Record<string, number>[]>([]);
    const [latencies, setLatencies] = useState<number[]>([]);

    const wsRef = useRef<WebSocket | null>(null);

    const resetState = useCallback(() => {
        setTranscripts([]);
        setTactics({});
        setRisk(null);
        setRiskEvents([]);
        setExplanation(null);
        setIntervention(null);
        setRiskHistory([]);
        setTacticHistory([]);
        setLatencies([]);
        setSessionStatus('idle');
    }, []);

    useEffect(() => {
        if (!sessionId) return;

        resetState();

        const wsPath = mode === 'demo' ? `/ws/demo/${sessionId}` : `/ws/session/${sessionId}`;
        const wsUrl = `ws://localhost:8000${wsPath}`;
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
            setIsConnected(true);
            setSessionStatus('connected');
        };
        ws.onclose = () => {
            setIsConnected(false);
            setSessionStatus('disconnected');
        };
        ws.onerror = () => {
            setSessionStatus('error');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            switch (data.type) {
                case 'session_start':
                    setSessionStatus('active');
                    break;

                case 'transcript':
                    setTranscripts(prev => [...prev, data as TranscriptMsg]);
                    break;

                case 'tactic_detection':
                    setTactics(data.tactics);
                    setTacticHistory(prev => [...prev, data.tactics]);
                    break;

                case 'risk_update':
                    setRisk(data as RiskMsg);
                    if (data.risk_history) {
                        setRiskHistory(data.risk_history);
                    }
                    break;

                case 'risk_event':
                    setRiskEvents(prev => [...prev, data as RiskEventMsg]);
                    if (data.total_latency_ms) {
                        setLatencies(prev => [...prev, data.total_latency_ms]);
                    }
                    break;

                case 'explanation':
                    setExplanation(data as ExplanationMsg);
                    break;

                case 'intervention':
                    setIntervention(data as InterventionMsg);
                    break;

                case 'session_end':
                    setSessionStatus('completed');
                    break;

                case 'error':
                    console.error('Pipeline error:', data.message);
                    break;
            }
        };

        return () => {
            ws.close();
        };
    }, [sessionId, mode, resetState]);

    const sendTurn = useCallback((text: string, speaker: string = 'UNKNOWN') => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'turn',
                text,
                speaker
            }));
        }
    }, []);

    return {
        transcripts,
        tactics,
        risk,
        riskEvents,
        explanation,
        intervention,
        isConnected,
        sessionStatus,
        riskHistory,
        tacticHistory,
        latencies,
        sendTurn,
        resetState
    };
}
