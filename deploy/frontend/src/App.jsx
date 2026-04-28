import { useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function valueToColor(value) {
  const clamped = Math.max(0, Math.min(1, value))
  const hue = 205 - clamped * 180
  return `hsl(${hue}, 85%, ${18 + clamped * 48}%)`
}

function App() {
  const [mw, setMw] = useState(5.5)
  const [strk, setStrk] = useState(180)
  const [dip, setDip] = useState(45)
  const [rake, setRake] = useState(90)
  const [lat, setLat] = useState(0)
  const [lon, setLon] = useState(0)
  const [dep, setDep] = useState(10)
  const [nx, setNx] = useState(10)
  const [nz, setNz] = useState(10)
  const [dx, setDx] = useState(1)
  const [dz, setDz] = useState(1)
  const [applyDz, setApplyDz] = useState(true)
  const [randomSeed, setRandomSeed] = useState(42)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const slipMap = result?.slip_map_2d || []
  const rows = slipMap.length
  const cols = slipMap[0]?.length || 0

  const summaryCards = useMemo(() => {
    if (!result) return []
    return [
      { label: 'Inference time', value: `${(result.inference_duration_seconds * 1000).toFixed(0)} ms` },
      { label: 'Slip min', value: Number(result.slip_stats.min).toFixed(4) },
      { label: 'Slip mean', value: Number(result.slip_stats.mean).toFixed(4) },
      { label: 'Slip max', value: Number(result.slip_stats.max).toFixed(4) },
    ]
  }, [result])

  const handlePredict = async () => {
    setLoading(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mw,
          strk,
          dip,
          rake,
          lat,
          lon,
          dep,
          nx,
          nz,
          dx,
          dz,
          random_seed: randomSeed,
          apply_dz: applyDz,
        }),
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data?.detail || 'Prediction failed')
      }

      setResult(data)
    } catch (predictionError) {
      console.error('Error predicting:', predictionError)
      setError(predictionError.message || 'Failed to connect to backend')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Slipgen MLOps</p>
          <h1>Interactive slip forecast dashboard</h1>
          <p className="lede">Drive the torch-backed FastAPI model, inspect the predicted slip field, and watch the inference metrics that Grafana scrapes from Prometheus.</p>
        </div>
        <button className="primary-button" onClick={handlePredict} disabled={loading}>
          {loading ? 'Generating...' : 'Generate Slip Map'}
        </button>
      </header>

      <main className="layout">
        <section className="panel controls-panel">
          <h2>Inputs</h2>
          <div className="slider-group">
            <label><span>Mw</span><span>{mw.toFixed(1)}</span></label>
            <input type="range" min="1" max="10" step="0.1" value={mw} onChange={(e) => setMw(Number.parseFloat(e.target.value))} />
          </div>
          <div className="slider-group">
            <label><span>Strike</span><span>{strk.toFixed(0)}</span></label>
            <input type="range" min="0" max="360" step="1" value={strk} onChange={(e) => setStrk(Number.parseFloat(e.target.value))} />
          </div>
          <div className="slider-group">
            <label><span>Dip</span><span>{dip.toFixed(0)}</span></label>
            <input type="range" min="0" max="90" step="1" value={dip} onChange={(e) => setDip(Number.parseFloat(e.target.value))} />
          </div>
          <div className="slider-group">
            <label><span>Rake</span><span>{rake.toFixed(0)}</span></label>
            <input type="range" min="-180" max="180" step="1" value={rake} onChange={(e) => setRake(Number.parseFloat(e.target.value))} />
          </div>

          <div className="two-col-grid">
            {[
              ['Lat', lat, setLat],
              ['Lon', lon, setLon],
              ['Dep', dep, setDep],
              ['Nx', nx, setNx],
              ['Nz', nz, setNz],
              ['Dx', dx, setDx],
              ['Dz', dz, setDz],
              ['Seed', randomSeed, setRandomSeed],
            ].map(([label, value, setter]) => (
              <label key={label} className="number-input">
                <span>{label}</span>
                <input type="number" value={value} onChange={(e) => setter(Number(e.target.value))} />
              </label>
            ))}
          </div>

          <label className="checkbox-row">
            <input type="checkbox" checked={applyDz} onChange={(e) => setApplyDz(e.target.checked)} />
            <span>Apply Dz scaling to convert normalized pixels to slip values</span>
          </label>

          {error ? <p className="error-banner">{error}</p> : null}

          <button className="secondary-button" onClick={handlePredict} disabled={loading}>
            Run inference
          </button>
        </section>

        <section className="panel results-panel">
          <div className="results-header">
            <h2>Inference output</h2>
            <p>{rows && cols ? `Heatmap shape ${rows} x ${cols}` : 'Waiting for a prediction'}</p>
          </div>

          <div className="metric-grid">
            {summaryCards.map((card) => (
              <article key={card.label} className="metric-card">
                <span>{card.label}</span>
                <strong>{card.value}</strong>
              </article>
            ))}
          </div>

          <div className="heatmap-frame">
            {slipMap.length ? (
              <div
                className="heatmap"
                style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
              >
                {slipMap.flatMap((row, rowIndex) =>
                  row.map((value, colIndex) => (
                    <div
                      key={`${rowIndex}-${colIndex}`}
                      className="heatmap-cell"
                      title={`(${rowIndex}, ${colIndex}) = ${Number(value).toFixed(4)}`}
                      style={{ background: valueToColor(value) }}
                    />
                  ))
                )}
              </div>
            ) : (
              <div className="empty-state">
                <p>The prediction heatmap will appear here.</p>
              </div>
            )}
          </div>

          {result ? (
            <details className="json-panel">
              <summary>Model response</summary>
              <pre>{JSON.stringify({ model_info: result.model_info, image_stats: result.image_stats, slip_stats: result.slip_stats, computed_parameters: result.computed_parameters }, null, 2)}</pre>
            </details>
          ) : null}
        </section>
      </main>
    </div>
  )
}

export default App
