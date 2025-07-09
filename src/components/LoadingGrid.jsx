function LoadingGrid() {
  return (
    <div className="loading">
      <div className="loading-grid">
        {[...Array(9)].map((_, index) => (
          <div key={index} className="loading-cell"></div>
        ))}
      </div>
      <div className="loading-text">SEARCHING...</div>
    </div>
  )
}

export default LoadingGrid 