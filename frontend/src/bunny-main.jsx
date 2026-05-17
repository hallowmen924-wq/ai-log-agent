import React from 'react';
import { createRoot } from 'react-dom/client';
import './bunny.css';
import BunnyAssistant from './components/BunnyAssistant';

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BunnyAssistant />
  </React.StrictMode>
);
