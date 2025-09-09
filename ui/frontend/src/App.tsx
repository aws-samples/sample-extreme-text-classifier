import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AppLayout } from './components/layout/app-layout';
import { Dashboard } from './pages/dashboard';
import { Datasets } from './pages/datasets';
import { Attributes } from './pages/attributes';
import { Classify } from './pages/classify';
import { Wizard } from './pages/wizard';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="datasets" element={<Datasets />} />
          <Route path="attributes" element={<Attributes />} />
          <Route path="attributes/:datasetId" element={<Attributes />} />
          <Route path="classify" element={<Classify />} />
          <Route path="classify/:datasetId" element={<Classify />} />
          <Route path="wizard" element={<Wizard />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
