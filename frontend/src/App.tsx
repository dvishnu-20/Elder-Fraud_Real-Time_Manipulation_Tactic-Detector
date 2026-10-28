import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './layouts/MainLayout';
import CoverPage from './pages/CoverPage';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import LiveMonitor from './pages/LiveMonitor';
import ConversationAnalysis from './pages/ConversationAnalysis';
import TacticAnalysis from './pages/TacticAnalysis';
import RiskAnalysis from './pages/RiskAnalysis';
import Explainability from './pages/Explainability';
import History from './pages/History';
import Reports from './pages/Reports';
import ModelInfo from './pages/ModelInfo';
import SystemHealth from './pages/SystemHealth';
import Privacy from './pages/Privacy';
import About from './pages/About';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<CoverPage />} />
        <Route path="/login" element={<Login />} />
        
        <Route element={<MainLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/live-monitor" element={<LiveMonitor />} />
          <Route path="/conversation" element={<ConversationAnalysis />} />
          <Route path="/tactic-analysis" element={<TacticAnalysis />} />
          <Route path="/risk-analysis" element={<RiskAnalysis />} />
          <Route path="/explanation" element={<Explainability />} />
          <Route path="/history" element={<History />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/model" element={<ModelInfo />} />
          <Route path="/system" element={<SystemHealth />} />
          <Route path="/privacy" element={<Privacy />} />
          <Route path="/about" element={<About />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
