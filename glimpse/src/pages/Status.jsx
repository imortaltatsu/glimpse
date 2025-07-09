import { useEffect, useState } from 'react'

function Status() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function fetchStatus() {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch('https://arfetch.adityaberry.me/status')
        if (!res.ok) throw new Error('Failed to fetch status')
        const data = await res.json()
        setStatus(data)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetchStatus()
  }, [])

  return (
    <div className="main-content">
      <h2 style={{ fontFamily: 'Orbitron, monospace', color: 'var(--primary-color)', marginBottom: '2rem' }}>System Status</h2>
      {loading && <div className="loading-text">Checking status...</div>}
      {error && <div style={{ color: 'var(--accent-color)' }}>Error: {error}</div>}
      {status && (
        <div className="results-grid">
          <div className="result-card">
            <div className="result-title">Backend Health</div>
            <div className="result-content">{status.healthy ? '✅ Online' : '❌ Offline'}</div>
          </div>
          <div className="result-card">
            <div className="result-title">Web Pages Indexed</div>
            <div className="result-content">{status.web_count}</div>
          </div>
          <div className="result-card">
            <div className="result-title">ARNS Domains Indexed</div>
            <div className="result-content">{status.arns_count}</div>
          </div>
          <div className="result-card">
            <div className="result-title">Last Index Update</div>
            <div className="result-content">{status.last_index_time || 'N/A'}</div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Status 