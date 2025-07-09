import { useState, useRef } from 'react'
import './ToggleSlider.css'

const SECTIONS = ['Web', 'Images', 'Videos', 'Audios']

function SearchBox({ onSearch, searchType, onSearchTypeChange }) {
  const [query, setQuery] = useState('')
  const [showComingSoon, setShowComingSoon] = useState(false)
  const [comingSoonType, setComingSoonType] = useState('')
  const [section, setSection] = useState('Web')
  const [mode, setMode] = useState('all') // 'all' or 'arns'
  const lastRandomLetter = useRef('')

  const handleSubmit = (e) => {
    e.preventDefault()
    onSearch(query)
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit(e)
    }
  }

  const handleInputChange = (e) => {
    const val = e.target.value
    setQuery(val)
    // If a single letter (a-z or A-Z) is typed, trigger random retrieval
    if (
      section === 'Web' &&
      val.length === 1 &&
      /^[a-zA-Z]$/.test(val) &&
      lastRandomLetter.current !== val
    ) {
      lastRandomLetter.current = val
      onSearch(val)
    }
    if (val.length !== 1) {
      lastRandomLetter.current = ''
    }
  }

  const handleSectionClick = (sec) => {
    setSection(sec)
    if (sec !== 'Web') {
      setShowComingSoon(true)
      setComingSoonType(sec)
      setTimeout(() => {
        setShowComingSoon(false)
        setSection('Web')
      }, 1800)
    }
  }

  const handleModeToggle = (checked) => {
    setMode(checked ? 'arns' : 'all')
    onSearchTypeChange(checked ? 'arns' : 'web')
  }

  return (
    <div className="search-container">
      <div className="section-selector-group">
        {SECTIONS.map(sec => (
          <button
            key={sec}
            className={`section-selector-btn${section === sec ? ' active' : ''}`}
            onClick={() => handleSectionClick(sec)}
          >
            {sec}
          </button>
        ))}
      </div>
      {showComingSoon && section !== 'Web' && (
        <div className="coming-soon-toast">
          {comingSoonType} search coming soon!
        </div>
      )}
      <form onSubmit={handleSubmit} className="search-box">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          placeholder={`Enter your ${section.toLowerCase()} search query...`}
          className="search-input"
          disabled={section !== 'Web'}
        />
        <button type="submit" className="search-btn" disabled={section !== 'Web'}>
          <span className="btn-text">SEARCH</span>
        </button>
      </form>
      <div className="search-options" style={{ justifyContent: 'center', marginTop: '1.5rem' }}>
        <div className="toggle-slider-container">
          <span className={`toggle-label${mode === 'all' ? ' active' : ''}`}>All</span>
          <label className="toggle-slider">
            <input
              type="checkbox"
              checked={mode === 'arns'}
              onChange={e => handleModeToggle(e.target.checked)}
              disabled={section !== 'Web'}
            />
            <span className="slider"></span>
          </label>
          <span className={`toggle-label${mode === 'arns' ? ' active' : ''}`}>ARNS</span>
        </div>
      </div>
    </div>
  )
}

export default SearchBox 