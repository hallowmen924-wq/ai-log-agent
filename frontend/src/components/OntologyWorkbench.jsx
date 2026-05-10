import React, { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { fetchOntologyState, saveOntologyState } from '../api';

const PRODUCT_OPTIONS = ['C6', 'C9', 'C11', 'C12'];
const DIRECTION_OPTIONS = [
  { value: 'input_fields', label: 'Input' },
  { value: 'output_fields', label: 'Output' },
];

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value || {}));
}

function hashSeed(value) {
  return Array.from(String(value || '')).reduce((acc, char, index) => acc + char.charCodeAt(0) * (index + 1), 0);
}

function flattenMappings(ontology) {
  const records = [];
  const products = ontology?.products || {};
  Object.entries(products).forEach(([productCode, productPayload]) => {
    ['input_fields', 'output_fields'].forEach((direction) => {
      Object.entries(productPayload?.[direction] || {}).forEach(([fieldCode, mapping]) => {
        records.push({
          productCode,
          productName: productPayload?.product_name || productCode,
          direction,
          fieldCode,
          ...mapping,
        });
      });
    });
  });
  return records;
}

function detectDuplicates(ontology) {
  const duplicates = [];
  const products = ontology?.products || {};
  Object.entries(products).forEach(([productCode, productPayload]) => {
    ['input_fields', 'output_fields'].forEach((direction) => {
      const grouped = {};
      Object.entries(productPayload?.[direction] || {}).forEach(([fieldCode, mapping]) => {
        const featureId = String(mapping?.feature_id || '').trim();
        if (!featureId) {
          return;
        }
        grouped[featureId] = grouped[featureId] || [];
        grouped[featureId].push({ fieldCode, mapping });
      });
      Object.entries(grouped).forEach(([featureId, items]) => {
        if (items.length > 1) {
          duplicates.push({
            id: `${productCode}-${direction}-${featureId}`,
            productCode,
            direction,
            featureId,
            featureName: items[0]?.mapping?.feature_name || featureId,
            items,
          });
        }
      });
    });
  });
  return duplicates;
}

function buildClusterNodes(commonfeature, productFilter, searchText) {
  const features = (commonfeature?.common_features || [])
    .filter((item) => !productFilter || (item.products || []).includes(productFilter))
    .filter((item) => {
      const query = String(searchText || '').trim().toLowerCase();
      if (!query) {
        return true;
      }
      return [item.feature_id, item.feature_name, item.category, ...(item.aliases || [])]
        .join(' ')
        .toLowerCase()
        .includes(query);
    })
    .slice()
    .sort((left, right) => (right.coverage?.mapping_count || 0) - (left.coverage?.mapping_count || 0))
    .slice(0, 84);

  const categories = Array.from(new Set(features.map((item) => item.category || 'misc')));
  const columns = Math.max(2, Math.ceil(Math.sqrt(categories.length || 1)));
  const centers = Object.fromEntries(
    categories.map((category, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const x = 16 + column * (68 / Math.max(columns - 1, 1));
      const y = 18 + row * 28;
      return [category, { x, y }];
    }),
  );

  const seenPerCategory = {};
  return features.map((feature) => {
    const category = feature.category || 'misc';
    const ordinal = seenPerCategory[category] || 0;
    seenPerCategory[category] = ordinal + 1;
    const seed = hashSeed(feature.feature_id);
    const angle = ((seed % 360) * Math.PI) / 180;
    const ring = 10 + (ordinal % 6) * 3.2;
    const center = centers[category] || { x: 50, y: 50 };
    const size = 68 + Math.min(26, (feature.coverage?.mapping_count || 0) * 2);
    const radiusX = 7 + (size / 24);
    const radiusY = 6 + (size / 26);
    const anchoredX = Math.max(10, Math.min(90, center.x + Math.cos(angle) * ring));
    const anchoredY = Math.max(12, Math.min(88, center.y + Math.sin(angle) * ring));
    return {
      ...feature,
      x: `calc(${anchoredX}% - ${radiusX}rem)`,
      y: `calc(${anchoredY}% - ${radiusY}rem)`,
      delay: (seed % 12) * 0.08,
      size,
      categoryLabel: category,
    };
  });
}

function ensureFeature(commonfeature, mapping) {
  const next = cloneJson(commonfeature);
  next.common_features = Array.isArray(next.common_features) ? next.common_features : [];
  const featureId = String(mapping?.feature_id || '').trim();
  if (!featureId) {
    return next;
  }
  const existing = next.common_features.find((item) => item.feature_id === featureId);
  if (existing) {
    existing.feature_name = mapping.feature_name || existing.feature_name || featureId;
    existing.category = mapping.category || existing.category || 'unclassified';
    existing.description = mapping.description || existing.description || '';
    existing.directions = Array.from(new Set([...(existing.directions || []), mapping.directionHint].filter(Boolean)));
    existing.aliases = Array.from(new Set([...(existing.aliases || []), mapping.label].filter(Boolean)));
    return next;
  }
  next.common_features.push({
    feature_id: featureId,
    feature_name: mapping.feature_name || featureId,
    category: mapping.category || 'unclassified',
    description: mapping.description || '',
    directions: mapping.directionHint ? [mapping.directionHint] : [],
    aliases: mapping.label ? [mapping.label] : [],
    products: [],
    coverage: { product_count: 0, mapping_count: 0 },
    field_mappings: [],
    sample_values: [],
  });
  return next;
}

export default function OntologyWorkbench({ theme = 'midnight', reduceMotion = false, onToast, onError }) {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [ontology, setOntology] = useState({});
  const [commonfeature, setCommonfeature] = useState({});
  const [selectedProduct, setSelectedProduct] = useState('C12');
  const [selectedDirection, setSelectedDirection] = useState('input_fields');
  const [searchText, setSearchText] = useState('');
  const [selectedFieldCode, setSelectedFieldCode] = useState('');
  const [localError, setLocalError] = useState('');

  useEffect(() => {
    let ignore = false;
    async function load() {
      try {
        setLoading(true);
        const payload = await fetchOntologyState();
        if (ignore) {
          return;
        }
        setOntology(payload?.ontology || {});
        setCommonfeature(payload?.commonfeature || {});
        setLocalError('');
      } catch (error) {
        if (!ignore) {
          const message = String(error.message || error);
          setLocalError(message);
          onError?.(message);
        }
      } finally {
        if (!ignore) {
          setLoading(false);
        }
      }
    }
    load();
    return () => {
      ignore = true;
    };
  }, [onError]);

  const allMappings = useMemo(() => flattenMappings(ontology), [ontology]);
  const filteredMappings = useMemo(() => {
    const query = String(searchText || '').trim().toLowerCase();
    return allMappings.filter((item) => {
      if (selectedProduct && item.productCode !== selectedProduct) {
        return false;
      }
      if (selectedDirection && item.direction !== selectedDirection) {
        return false;
      }
      if (!query) {
        return true;
      }
      return [item.fieldCode, item.label, item.feature_id, item.feature_name, item.category]
        .join(' ')
        .toLowerCase()
        .includes(query);
    });
  }, [allMappings, searchText, selectedDirection, selectedProduct]);

  useEffect(() => {
    if (!filteredMappings.length) {
      setSelectedFieldCode('');
      return;
    }
    if (!filteredMappings.some((item) => item.fieldCode === selectedFieldCode)) {
      setSelectedFieldCode(filteredMappings[0].fieldCode);
    }
  }, [filteredMappings, selectedFieldCode]);

  const selectedRecord = filteredMappings.find((item) => item.fieldCode === selectedFieldCode) || null;
  const duplicateGroups = useMemo(() => detectDuplicates(ontology), [ontology]);
  const clusterNodes = useMemo(
    () => buildClusterNodes(commonfeature, selectedProduct, searchText),
    [commonfeature, searchText, selectedProduct],
  );
  const featureSummary = commonfeature?.statistics || {};

  function updateSelectedMapping(field, value) {
    if (!selectedRecord) {
      return;
    }
    setOntology((previous) => {
      const next = cloneJson(previous);
      next.products[selectedRecord.productCode][selectedRecord.direction][selectedRecord.fieldCode][field] = value;
      return next;
    });
    setCommonfeature((previous) => ensureFeature(previous, {
      feature_id: field === 'feature_id' ? value : selectedRecord.feature_id,
      feature_name: field === 'feature_name' ? value : selectedRecord.feature_name,
      category: field === 'category' ? value : selectedRecord.category,
      description: field === 'description' ? value : selectedRecord.description,
      label: selectedRecord.label,
      directionHint: selectedRecord.direction === 'input_fields' ? 'input' : 'output',
    }));
  }

  function updateCommonFeature(field, value) {
    if (!selectedRecord) {
      return;
    }
    setCommonfeature((previous) => {
      const next = cloneJson(previous);
      next.common_features = Array.isArray(next.common_features) ? next.common_features : [];
      const target = next.common_features.find((item) => item.feature_id === selectedRecord.feature_id);
      if (!target) {
        next.common_features.push({
          feature_id: selectedRecord.feature_id,
          feature_name: selectedRecord.feature_name,
          category: selectedRecord.category || 'unclassified',
          description: '',
          aliases: [selectedRecord.label].filter(Boolean),
          directions: [selectedRecord.direction === 'input_fields' ? 'input' : 'output'],
          field_mappings: [],
          coverage: { product_count: 0, mapping_count: 0 },
          sample_values: [],
        });
      }
      const finalTarget = next.common_features.find((item) => item.feature_id === selectedRecord.feature_id);
      finalTarget[field] = value;
      return next;
    });
  }

  async function handleSave() {
    try {
      setSaving(true);
      const payload = await saveOntologyState({ ontology, commonfeature });
      setOntology(payload?.ontology || {});
      setCommonfeature(payload?.commonfeature || {});
      setLocalError('');
      onToast?.({
        id: `ontology-save-${Date.now()}`,
        kicker: 'Ontology Saved',
        title: '온톨로지 저장 완료',
        meta: 'ontology.json / commonfeature.json 동기화',
        message: '공통 feature와 상품별 매핑이 파일에 반영되었습니다.',
        tone: theme === 'cute' ? 'cute' : 'cyan',
      });
    } catch (error) {
      const message = String(error.message || error);
      setLocalError(message);
      onError?.(message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <motion.section key="ontology" className="content-stack ontology-shell" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.28, ease: 'easeOut' }}>
      <section className="panel ontology-hero-panel">
        <div className="ontology-hero-copy">
          <div className="panel-kicker">Ontology Studio</div>
          <h2>공통 feature 실시간 군집화</h2>
          <p>ontology.json을 기준으로 공통 feature를 귀엽게 군집화하고, 잘못된 매핑이나 상품별 중복 매핑을 바로 편집할 수 있는 작업 화면입니다.</p>
        </div>
        <div className="ontology-pill-row">
          <span className="sample-pill">공통 feature {featureSummary.common_feature_count || 0}</span>
          <span className="sample-pill">규칙 매핑 {(featureSummary.classified_feature_count || 0).toLocaleString()}</span>
          <span className="sample-pill">fallback {(featureSummary.fallback_feature_count || 0).toLocaleString()}</span>
          <span className="sample-pill">중복 경고 {duplicateGroups.length}</span>
        </div>
      </section>

      {localError ? <div className="error-banner">온톨로지 로딩 오류: {localError}</div> : null}

      <div className="ontology-top-grid">
        <section className="panel ontology-cluster-panel">
          <div className="panel-head panel-head-spread">
            <div>
              <div className="panel-kicker">Live Cluster</div>
              <h2>고객처럼 모여있는 feature 군집</h2>
            </div>
            <div className="ontology-toolbar-row">
              <select className="store-select" value={selectedProduct} onChange={(event) => setSelectedProduct(event.target.value)}>
                {PRODUCT_OPTIONS.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
              <input className="vector-search-input ontology-search-input" value={searchText} onChange={(event) => setSearchText(event.target.value)} placeholder="feature, field code, category 검색" />
            </div>
          </div>
          <div className={`ontology-cluster-stage ${loading ? 'loading' : ''}`}>
            <div className="ontology-cluster-grid" />
            {clusterNodes.map((node) => (
              <motion.button
                key={node.feature_id}
                type="button"
                className={`ontology-bubble ${node.category === 'unclassified' ? 'is-fallback' : ''}`}
                style={{ left: node.x, top: node.y, width: node.size, height: node.size }}
                initial={reduceMotion ? false : { opacity: 0, scale: 0.8 }}
                animate={reduceMotion ? { opacity: 1 } : { opacity: 1, scale: [1, 1.04, 1], y: [0, -8, 0], x: [0, 5, 0] }}
                transition={{ duration: 4.2 + (node.delay || 0), delay: node.delay, repeat: reduceMotion ? 0 : Infinity, ease: 'easeInOut' }}
                onClick={() => {
                  const candidate = filteredMappings.find((item) => item.feature_id === node.feature_id);
                  if (candidate) {
                    setSelectedProduct(candidate.productCode);
                    setSelectedDirection(candidate.direction);
                    setSelectedFieldCode(candidate.fieldCode);
                  }
                }}
              >
                <span className="ontology-bubble-face">
                  <span className="eye" />
                  <span className="eye" />
                  <span className="smile" />
                </span>
                <span className="ontology-bubble-name">{node.feature_name}</span>
                <span className="ontology-bubble-meta">{node.coverage?.mapping_count || 0} maps</span>
              </motion.button>
            ))}
            {!clusterNodes.length ? <div className="empty-box ontology-empty-stage">표시할 군집 노드가 없습니다.</div> : null}
          </div>
          <div className="ontology-category-row">
            {Array.from(new Set(clusterNodes.map((item) => item.categoryLabel))).map((item) => (
              <span className="reason-chip" key={item}>{item}</span>
            ))}
          </div>
        </section>

        <section className="panel ontology-alert-panel">
          <div className="panel-head">
            <div>
              <div className="panel-kicker">Duplicate Watch</div>
              <h2>중복 매핑 경고</h2>
            </div>
          </div>
          <div className="ontology-duplicate-list">
            <AnimatePresence initial={false}>
              {duplicateGroups.slice(0, 18).map((item) => (
                <motion.button
                  key={item.id}
                  type="button"
                  className="ontology-duplicate-card"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  onClick={() => {
                    const first = item.items[0];
                    setSelectedProduct(item.productCode);
                    setSelectedDirection(item.direction);
                    setSelectedFieldCode(first.fieldCode);
                  }}
                >
                  <strong>{item.productCode} · {item.direction === 'input_fields' ? 'Input' : 'Output'}</strong>
                  <span>{item.featureName}</span>
                  <p>{item.items.map((entry) => entry.fieldCode).join(', ')}</p>
                </motion.button>
              ))}
            </AnimatePresence>
            {!duplicateGroups.length ? <div className="empty-box">현재 감지된 중복 매핑이 없습니다.</div> : null}
          </div>
        </section>
      </div>

      <div className="ontology-editor-grid">
        <section className="panel ontology-mapping-list-panel">
          <div className="panel-head panel-head-spread">
            <div>
              <div className="panel-kicker">Field Mapping</div>
              <h2>실제 결과물 편집</h2>
            </div>
            <div className="ontology-toolbar-row">
              <select className="store-select" value={selectedDirection} onChange={(event) => setSelectedDirection(event.target.value)}>
                {DIRECTION_OPTIONS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
              </select>
              <span className="sample-pill">{filteredMappings.length} fields</span>
            </div>
          </div>
          <div className="ontology-mapping-list">
            {filteredMappings.map((item) => (
              <button
                key={`${item.productCode}-${item.direction}-${item.fieldCode}`}
                type="button"
                className={`ontology-mapping-row ${selectedFieldCode === item.fieldCode ? 'active' : ''}`}
                onClick={() => setSelectedFieldCode(item.fieldCode)}
              >
                <div>
                  <strong>{item.fieldCode}</strong>
                  <p>{item.label || '라벨 없음'}</p>
                </div>
                <span>{item.feature_name || item.feature_id}</span>
              </button>
            ))}
            {!filteredMappings.length ? <div className="empty-box">조건에 맞는 매핑이 없습니다.</div> : null}
          </div>
        </section>

        <section className="panel ontology-editor-panel">
          <div className="panel-head panel-head-spread">
            <div>
              <div className="panel-kicker">Editor</div>
              <h2>{selectedRecord ? `${selectedRecord.productCode} ${selectedRecord.fieldCode}` : '매핑 선택'}</h2>
            </div>
            <button className="primary-button" type="button" disabled={saving || !selectedRecord} onClick={handleSave}>{saving ? '저장 중' : '저장'}</button>
          </div>
          {selectedRecord ? (
            <div className="ontology-editor-form">
              <label className="prompt-editor">
                <span>원본 라벨</span>
                <input className="store-select" value={selectedRecord.label || ''} readOnly />
              </label>
              <label className="prompt-editor">
                <span>feature_id</span>
                <input className="store-select" value={selectedRecord.feature_id || ''} onChange={(event) => updateSelectedMapping('feature_id', event.target.value)} />
              </label>
              <label className="prompt-editor">
                <span>feature_name</span>
                <input className="store-select" value={selectedRecord.feature_name || ''} onChange={(event) => updateSelectedMapping('feature_name', event.target.value)} />
              </label>
              <label className="prompt-editor">
                <span>category</span>
                <input className="store-select" value={selectedRecord.category || ''} onChange={(event) => updateSelectedMapping('category', event.target.value)} />
              </label>
              <label className="prompt-editor">
                <span>confidence</span>
                <input className="store-select" value={selectedRecord.confidence || ''} onChange={(event) => updateSelectedMapping('confidence', event.target.value)} />
              </label>
              <label className="prompt-editor">
                <span>match_basis</span>
                <input className="store-select" value={selectedRecord.match_basis || ''} onChange={(event) => updateSelectedMapping('match_basis', event.target.value)} />
              </label>
              <label className="prompt-editor prompt-editor-wide">
                <span>공통 feature 설명</span>
                <textarea
                  value={(commonfeature?.common_features || []).find((item) => item.feature_id === selectedRecord.feature_id)?.description || ''}
                  onChange={(event) => updateCommonFeature('description', event.target.value)}
                />
              </label>
              <div className="ontology-editor-meta">
                <span className="sample-pill">observed {selectedRecord.observed_count || 0}</span>
                <span className="sample-pill">direction {selectedRecord.direction === 'input_fields' ? 'input' : 'output'}</span>
                <span className="sample-pill">product {selectedRecord.productCode}</span>
              </div>
              <div className="summary-box ontology-sample-box">
                <div className="summary-box-title">샘플 값</div>
                {(selectedRecord.sample_values || []).length ? (
                  <div className="detail-meta-row">
                    {selectedRecord.sample_values.slice(0, 8).map((item) => (
                      <span className="sample-pill" key={`${item.value}-${item.count}`}>{item.value} · {item.count}</span>
                    ))}
                  </div>
                ) : <p>기록된 샘플 값이 없습니다.</p>}
              </div>
            </div>
          ) : <div className="empty-box">왼쪽에서 편집할 필드를 선택하세요.</div>}
        </section>
      </div>
    </motion.section>
  );
}