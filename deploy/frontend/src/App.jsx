import { useState } from 'react'

function App() {
  const [mw, setMw] = useState(5.5)
  const [slipMap, setSlipMap] = useState(null)

  const handlePredict = async () => {
    // Call the FastAPI endpoint
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          mw, 
          strk: 180, 
          dip: 45, 
          rake: 90,
          lat: 0.0,
          lon: 0.0,
          dep: 10.0,
          nx: 10,
          nz: 10,
          dx: 1.0,
          dz: 1.0,
          random_seed: 42,
          apply_dz: false
        })
      });
      const data = await response.json();
      setSlipMap(data.slip_map_2d);
    } catch (error) {
      console.error("Error predicting:", error);
      alert("Failed to connect to backend");
    }
  }

  return (
    <div style={{ display: 'flex', padding: '20px', fontFamily: 'sans-serif' }}>
      <div style={{ width: '350px', paddingRight: '20px', borderRight: '1px solid #ccc' }}>
        <h2>Slipgen MLOps</h2>
        <h3>Hyperparameters</h3>
        <div style={{ marginBottom: '15px' }}>
          <label>Moment Magnitude (Mw): {mw.toFixed(1)}</label><br/>
          <input 
            type="range" 
            min="1.0" 
            max="10.0" 
            step="0.1" 
            value={mw} 
            onChange={(e) => setMw(parseFloat(e.target.value))} 
            style={{ width: '100%' }}
          />
        </div>
        <button 
          onClick={handlePredict}
          style={{ padding: '10px 15px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
        >
          Generate Slip Map
        </button>
      </div>
      <div style={{ paddingLeft: '20px', flex: 1 }}>
        <h3>Result Matrix</h3>
        {slipMap ? (
          <div>
            <p>Output shape: 50x50 generated successfully via inference endpoint.</p>
            <div style={{ background: '#f5f5f5', padding: '10px', height: '400px', overflow: 'auto' }}>
              <pre>{JSON.stringify(slipMap.slice(0,2), null, 2)} ... (truncated)</pre>
            </div>
          </div>
        ) : (
          <p style={{ color: '#666' }}>Waiting for prediction...</p>
        )}
      </div>
    </div>
  )
}

export default App
