const DEFAULT_BASE_URL = 'http://127.0.0.1:18000';

function getBaseUrl() {
  const configured = (import.meta.env.VITE_BACKEND_URL || '').trim();
  if (configured) {
    return configured.replace(/\/$/, '');
  }
  if (import.meta.env.DEV) {
    return '/api';
  }
  return DEFAULT_BASE_URL;
}

export const API_BASE_URL = getBaseUrl();

function buildUrl(path, query = {}) {
  const url = new URL(`${API_BASE_URL}${path}`, window.location.origin);
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(key, value);
    }
  });
  return url.toString();
}

async function request(path, options = {}) {
  const response = await fetch(buildUrl(path), options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }
  return response.text();
}

export function fetchStatus() {
  return request('/analysis/status');
}

export function fetchHealth() {
  return request('/health');
}

export function fetchCharts() {
  return request('/charts');
}

export function fetchProductSummary() {
  return request('/product-pattern-summary');
}

export function fetchOntologyState() {
  return request('/ontology/state');
}

export function saveOntologyState(payload) {
  return request('/ontology/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
}

export function fetchFaissEntries(limit = 120, storeName = '') {
  return request('/faiss/entries', {
    method: 'GET',
    headers: { Accept: 'application/json' },
  },);
}

export function fetchFaissEntriesByStore(limit = 120, storeName = '', type = '') {
  return request(buildPath('/faiss/entries', { limit, store_name: storeName, type }));
}

export function fetchFaissEntry(docId) {
  return request(buildPath('/faiss/entry', { doc_id: docId }));
}

export function fetchSimilarLogVectors(query, limit = 8) {
  return request(buildPath('/faiss/similar_logs', { query, limit }));
}

function buildPath(path, query) {
  const queryText = new URLSearchParams(
    Object.entries(query).filter(([, value]) => value !== undefined && value !== null && value !== ''),
  ).toString();
  return queryText ? `${path}?${queryText}` : path;
}

export function fetchFaissStats() {
  return request('/faiss/stats');
}

export function startCardloanDebate(question, reviewerPrompts, reviewerSettings = {}) {
  return request('/chat/cardloan-debate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, reviewer_prompts: reviewerPrompts, reviewer_settings: reviewerSettings }),
  });
}

export function chatStrategy(question) {
  return request('/chat/strategy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
}

export function uploadRegulationFiles(files) {
  const formData = new FormData();
  Array.from(files).forEach((file) => {
    formData.append('files', file);
  });
  return request('/regulation/upload', {
    method: 'POST',
    body: formData,
  });
}

export function createFaissWebSocket(onMessage) {
  const configured = (import.meta.env.VITE_BACKEND_URL || '').trim();
  let targetBase = configured || DEFAULT_BASE_URL;
  if (import.meta.env.DEV && !configured) {
    targetBase = `${window.location.protocol}//${window.location.host}`;
  }
  const wsBase = targetBase.replace(/^http/, 'ws').replace(/\/$/, '');
  const socket = new WebSocket(`${wsBase}/ws/faiss`);
  socket.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data));
    } catch {
      return;
    }
  };
  return socket;
}