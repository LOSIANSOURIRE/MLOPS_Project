import { useEffect, useMemo, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const VIRIDIS_STOPS = [
  [0.0, [68, 1, 84]],
  [0.13, [72, 35, 116]],
  [0.25, [64, 67, 135]],
  [0.38, [52, 94, 141]],
  [0.5, [41, 120, 142]],
  [0.63, [32, 144, 140]],
  [0.75, [34, 167, 132]],
  [0.88, [68, 190, 112]],
  [1.0, [253, 231, 37]],
]

function clamp01(value) {
  return Math.max(0, Math.min(1, value))
}

function interpolateColor(a, b, t) {
  return a.map((channel, index) => Math.round(channel + (b[index] - channel) * t))
}

function viridisColor(value) {
  const clamped = clamp01(value)
  for (let index = 0; index < VIRIDIS_STOPS.length - 1; index += 1) {
    const [startPos, startColor] = VIRIDIS_STOPS[index]
    const [endPos, endColor] = VIRIDIS_STOPS[index + 1]
    if (clamped >= startPos && clamped <= endPos) {
      const span = endPos - startPos || 1
      const localT = (clamped - startPos) / span
      const [r, g, b] = interpolateColor(startColor, endColor, localT)
      return `rgb(${r}, ${g}, ${b})`
    }
  }
  const [r, g, b] = VIRIDIS_STOPS[VIRIDIS_STOPS.length - 1][1]
  return `rgb(${r}, ${g}, ${b})`
}

function buildTicks(maxValue) {
  if (!Number.isFinite(maxValue) || maxValue <= 0) return [0]
  const count = 5
  return Array.from({ length: count }, (_, index) => (maxValue * index) / (count - 1))
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
  const heatmapCanvasRef = useRef(null)
  const colorbarCanvasRef = useRef(null)

  const slipMap = result?.slip_plane_2d || result?.slip_map_2d || []
  const rows = slipMap.length
  const cols = slipMap[0]?.length || 0
  const minValue = Number(result?.slip_stats?.min ?? 0)
  const maxValue = Number(result?.slip_stats?.max ?? 1)
  const usePhysicalCoords = Boolean(result && applyDz)
  const axisLabelX = usePhysicalCoords ? 'Along-strike direction (km)' : 'Pixel columns'
  const axisLabelY = usePhysicalCoords ? 'Down-dip direction (km)' : 'Pixel rows'
  const colorbarLabel = usePhysicalCoords ? 'Slip (m)' : 'Normalized intensity'

  useEffect(() => {
    const canvas = heatmapCanvasRef.current
    if (!canvas || !rows || !cols) return

    const pixelRatio = window.devicePixelRatio || 1
    const displaySize = Math.max(550, Math.min(900, canvas.clientWidth || 800, canvas.clientHeight || 800))
    
    // Allocate space for axis labels
    const totalSize = displaySize + 60 // Extra 60px for labels
    canvas.width = Math.floor(totalSize * pixelRatio)
    canvas.height = Math.floor(totalSize * pixelRatio)
    canvas.style.width = '100%'
    canvas.style.height = '100%'

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
    ctx.clearRect(0, 0, totalSize, totalSize)
    
    // Light background
    ctx.fillStyle = 'rgba(20, 24, 36, 0.5)'
    ctx.fillRect(0, 0, totalSize, totalSize)

    const padding = { top: 20, right: 20, bottom: 40, left: 50 }
    const plotSize = displaySize - padding.left - padding.right
    const cellSize = plotSize / cols

    const sourceCanvas = document.createElement('canvas')
    sourceCanvas.width = cols
    sourceCanvas.height = rows
    const sourceCtx = sourceCanvas.getContext('2d')
    if (!sourceCtx) return

    const imageData = sourceCtx.createImageData(cols, rows)
    const denominator = maxValue - minValue || 1

    for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
      for (let colIndex = 0; colIndex < cols; colIndex += 1) {
        const value = Number(slipMap[rowIndex][colIndex])
        const normalized = clamp01((value - minValue) / denominator)
        const [red, green, blue] = viridisColor(normalized)
          .match(/\d+/g)
          .map((channel) => Number.parseInt(channel, 10))
        const offset = (rowIndex * cols + colIndex) * 4
        imageData.data[offset] = red
        imageData.data[offset + 1] = green
        imageData.data[offset + 2] = blue
        imageData.data[offset + 3] = 255
      }
    }

    sourceCtx.putImageData(imageData, 0, 0)
    ctx.imageSmoothingEnabled = true
    ctx.drawImage(sourceCanvas, padding.left, padding.top, plotSize, plotSize)

    // Draw grid lines
    ctx.strokeStyle = 'rgba(12, 16, 28, 0.15)'
    ctx.lineWidth = 1
    for (let colIndex = 1; colIndex < cols; colIndex += 1) {
      const x = padding.left + colIndex * cellSize
      ctx.beginPath()
      ctx.moveTo(x, padding.top)
      ctx.lineTo(x, padding.top + plotSize)
      ctx.stroke()
    }
    for (let rowIndex = 1; rowIndex < rows; rowIndex += 1) {
      const y = padding.top + rowIndex * cellSize
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(padding.left + plotSize, y)
      ctx.stroke()
    }

    // Draw plot border
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)'
    ctx.lineWidth = 2
    ctx.strokeRect(padding.left - 1, padding.top - 1, plotSize + 2, plotSize + 2)

    // Draw axis tick labels
    ctx.fillStyle = 'rgba(242, 246, 255, 1)'
    ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'

    // X-axis ticks at bottom
    const xTickCount = 6
    for (let i = 0; i < xTickCount; i += 1) {
      const colIndex = Math.round((i / (xTickCount - 1)) * (cols - 1))
      const x = padding.left + (colIndex * plotSize) / (cols - 1)
      const label = colIndex.toString()
      ctx.fillText(label, x, padding.top + plotSize + 8)
    }

    // Y-axis ticks at left
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    const yTickCount = 6
    for (let i = 0; i < yTickCount; i += 1) {
      const rowIndex = Math.round((i / (yTickCount - 1)) * (rows - 1))
      const y = padding.top + (rowIndex * plotSize) / (rows - 1)
      const label = rowIndex.toString()
      ctx.fillText(label, padding.left - 12, y)
    }
  }, [cols, maxValue, minValue, rows, slipMap, usePhysicalCoords])

  useEffect(() => {
    const canvas = colorbarCanvasRef.current
    if (!canvas) return

    const pixelRatio = window.devicePixelRatio || 1
    const barWidth = 48
    const height = 420
    const textWidth = 70
    const totalWidth = barWidth + textWidth
    
    canvas.width = Math.floor(totalWidth * pixelRatio)
    canvas.height = Math.floor(height * pixelRatio)
    canvas.style.width = `${totalWidth}px`
    canvas.style.height = `${height}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
    
    // Draw background panel for better visibility
    ctx.fillStyle = 'rgba(20, 24, 36, 0.4)'
    ctx.fillRect(-5, -5, totalWidth + 10, height + 10)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    ctx.strokeRect(-5, -5, totalWidth + 10, height + 10)
    
    // Draw gradient colorbar
    const gradient = ctx.createLinearGradient(0, height, 0, 0)
    for (let index = 0; index < VIRIDIS_STOPS.length; index += 1) {
      const [position, color] = VIRIDIS_STOPS[index]
      gradient.addColorStop(position, `rgb(${color[0]}, ${color[1]}, ${color[2]})`)
    }
    ctx.fillStyle = gradient
    ctx.fillRect(8, 8, barWidth, height - 16)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'
    ctx.lineWidth = 2
    ctx.strokeRect(8, 8, barWidth, height - 16)

    // Draw colorbar tick values with better visibility
    ctx.fillStyle = 'rgba(242, 246, 255, 1)'
    ctx.font = 'bold 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'

    const tickCount = 6
    for (let i = 0; i < tickCount; i += 1) {
      const y = 8 + (i / (tickCount - 1)) * (height - 16)
      const value = maxValue - (i / (tickCount - 1)) * (maxValue - minValue)
      const label = value.toFixed(1)
      ctx.fillText(label, barWidth + 16, y)
    }
  }, [result, minValue, maxValue])

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

          <div className="figure-card">
            <div className="figure-header">
              <div>
                <h3>Slip distribution</h3>
                <p>{rows && cols ? `Grid ${rows} x ${cols}` : 'Waiting for a prediction'}</p>
              </div>
            </div>

            <div className="figure-body">
              <div className="axis-label y-axis-label">{axisLabelY}</div>
              <div className="plot-column">
                <div className="plot-shell">
                  {slipMap.length ? (
                    <canvas ref={heatmapCanvasRef} className="heatmap-canvas" />
                  ) : (
                    <div className="empty-state figure-empty">
                      <p>The prediction heatmap will appear here.</p>
                    </div>
                  )}
                </div>
                <div className="axis-footer">
                  <span>{axisLabelX}</span>
                </div>
              </div>

              <aside className="colorbar-panel">
                <canvas ref={colorbarCanvasRef} className="colorbar-canvas" />
                <div className="colorbar-labels">
                  <span>{Number(maxValue).toFixed(2)}</span>
                  <span>{colorbarLabel}</span>
                  <span>{Number(minValue).toFixed(2)}</span>
                </div>
              </aside>
            </div>
          </div>

          {result ? (
            <details className="json-panel">
              <summary>Model response</summary>
              <pre>{JSON.stringify({ model_info: result.model_info, image_stats: result.image_stats, slip_stats: result.slip_stats, computed_parameters: result.computed_parameters, slip_plane_2d: result.slip_plane_2d }, null, 2)}</pre>
            </details>
          ) : null}
        </section>
      </main>
    </div>
  )
}

export default App
