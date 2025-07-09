import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import SearchBox from './components/SearchBox'
import SearchResults from './components/SearchResults'
import LoadingGrid from './components/LoadingGrid'
import Status from './pages/Status'

function Home({ handleSearch, searchType, setSearchType, isLoading, searchResults }) {
  return (
    <main className="main-content">
      <SearchBox 
        onSearch={handleSearch}
        searchType={searchType}
        onSearchTypeChange={setSearchType}
      />
      <div className="results-container">
        {isLoading ? (
          <LoadingGrid />
        ) : (
          <SearchResults results={searchResults} />
        )}
      </div>
    </main>
  )
}

function App() {
  const [searchResults, setSearchResults] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [searchType, setSearchType] = useState('web')

  const handleSearch = async (query) => {
    if (!query.trim()) return
    setIsLoading(true)
    setSearchResults([])
    try {
      const endpoint = searchType === 'web' ? 'searchweb' : 'searcharns'
      const response = await fetch(`https://arfetch.adityaberry.me/${endpoint}?query=${encodeURIComponent(query)}&top_k=10`)
      if (!response.ok) {
        throw new Error('Search failed')
      }
      const data = await response.json()
      setSearchResults(data.results || [])
    } catch (error) {
      console.error('Search error:', error)
      setSearchResults([])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Router>
      <div className="container">
        <header className="header">
          <div className="logo">
            <span className="logo-text">GLIMPSE</span>
          </div>
          <nav style={{ marginTop: '1rem' }}>
            <Link to="/" style={{ color: 'var(--primary-color)', marginRight: '2rem', textDecoration: 'none', fontWeight: 700 }}>Home</Link>
            <Link to="/status" style={{ color: 'var(--primary-color)', textDecoration: 'none', fontWeight: 700 }}>Status</Link>
          </nav>
        </header>
        <Routes>
          <Route path="/" element={<Home handleSearch={handleSearch} searchType={searchType} setSearchType={setSearchType} isLoading={isLoading} searchResults={searchResults} />} />
          <Route path="/status" element={<Status />} />
        </Routes>
        <footer className="footer">
          <div className="footer-text">Powered by Arweave</div>
        </footer>
      </div>
    </Router>
  )
}

export default App 