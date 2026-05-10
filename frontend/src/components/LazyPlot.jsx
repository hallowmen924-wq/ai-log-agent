import React, { useEffect, useState } from 'react';

function PlotLoading({ height = 240 }) {
  return (
    <div
      className="empty-box"
      style={{ minHeight: `${height}px`, width: '100%' }}
      aria-live="polite"
    >
      차트 로딩 중...
    </div>
  );
}

export default function LazyPlot({ loadingHeight, ...props }) {
  const [PlotComponent, setPlotComponent] = useState(null);
  const [loadError, setLoadError] = useState('');

  useEffect(() => {
    let active = true;

    async function loadPlot() {
      try {
        const [{ default: Plotly }, { default: createPlotlyComponent }] = await Promise.all([
          import('plotly.js-basic-dist-min'),
          import('react-plotly.js/factory'),
        ]);

        if (!active) {
          return;
        }

        setPlotComponent(() => createPlotlyComponent(Plotly));
      } catch (error) {
        if (!active) {
          return;
        }
        setLoadError(error instanceof Error ? error.message : 'plot load failed');
      }
    }

    loadPlot();

    return () => {
      active = false;
    };
  }, []);

  if (loadError) {
    return <div className="error-banner">차트 로딩 실패: {loadError}</div>;
  }

  if (!PlotComponent) {
    return <PlotLoading height={loadingHeight} />;
  }

  return <PlotComponent {...props} />;
}