# Glimpse - Retro Search Engine

A beautiful retro-styled search engine frontend built with React + Vite that connects to your Arweave indexing backend.

## Features

- 🎨 **Retro Grid Design** - Beautiful minimal retro aesthetic with animated grid background
- ⚡ **Fast Search** - Real-time search across Arweave web content and ARNS domains
- 🔍 **Dual Search Modes** - Search web content or ARNS domains
- 📱 **Responsive Design** - Works perfectly on desktop and mobile
- 🌟 **Smooth Animations** - Retro-style loading animations and hover effects

## Tech Stack

- **Frontend**: React 18 + Vite
- **Styling**: CSS with retro grid animations
- **Backend**: FastAPI (Python)
- **Search**: CLIP embeddings + FAISS indexing
- **Data**: Arweave blockchain content

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### 3. Backend Server

The FastAPI backend is deployed at `https://arfetch.adityaberry.me`. For local development, you can run:

```bash
python app.py
```

## Development

### Project Structure

```
glimpse/
├── src/
│   ├── components/
│   │   ├── SearchBox.jsx      # Search input and type selection
│   │   ├── SearchResults.jsx  # Results display grid
│   │   └── LoadingGrid.jsx    # Animated loading component
│   ├── App.jsx               # Main app component
│   ├── main.jsx              # React entry point
│   └── index.css             # Global retro styling
├── app.py                    # FastAPI backend
├── package.json              # Frontend dependencies
├── vite.config.js           # Vite configuration
└── index.html               # HTML template
```

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## API Endpoints

The frontend connects to these backend endpoints:

- `GET /api/searchweb?query={query}&top_k={k}` - Search web content
- `GET /api/searcharns?query={query}&top_k={k}` - Search ARNS domains

## Styling

The app features a retro aesthetic with:

- **Color Scheme**: Green terminal colors (#00ff41, #003b00)
- **Typography**: Orbitron (headings) + Share Tech Mono (body)
- **Animations**: Grid movement, glow effects, loading pulses
- **Layout**: CSS Grid for responsive results display

## Production Build

1. Build the frontend:
```bash
npm run build
```

2. The built files will be in the `dist/` directory

3. The FastAPI backend will serve the static files automatically

## Customization

### Colors
Edit the CSS variables in `src/index.css`:
```css
:root {
  --primary-color: #00ff41;
  --secondary-color: #003b00;
  --accent-color: #ff6b35;
  /* ... */
}
```

### Grid Animation
Modify the grid animation in `src/index.css`:
```css
@keyframes gridMove {
  0% { transform: translate(0, 0); }
  100% { transform: translate(50px, 50px); }
}
```

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Mobile browsers

## License

MIT License 