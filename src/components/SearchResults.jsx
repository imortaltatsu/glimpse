function SearchResults({ results }) {
  if (results.length === 0) {
    return (
      <div className="results-grid">
        <div className="result-card" style={{ textAlign: 'center', gridColumn: '1 / -1' }}>
          <div className="result-content">
            No results found. Try a different search query.
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="results-grid">
      {results.map((result, index) => (
        <a
          key={index}
          className="result-card"
          href={result.url}
          target="_blank"
          rel="noopener noreferrer"
          style={{ textDecoration: 'none', color: 'inherit', cursor: 'pointer' }}
        >
          <div className="result-score">
            {(result.score * 100).toFixed(1)}%
          </div>
          <div className="result-url">
            {result.url}
          </div>
          <div className="result-title">
            {result.txid || result.arns || 'Search Result'}
          </div>
          <div className="result-content">
            {result.score > 0.5 ? 'High relevance match' : 'Relevant content found'}
          </div>
        </a>
      ))}
    </div>
  )
}

export default SearchResults 