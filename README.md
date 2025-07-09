# Glimpse - Retro Search Engine

A beautiful retro-styled search engine frontend built with React + Vite that connects to your Arweave indexing backend.

## Features

- 🎨 **Retro Grid Design** - Beautiful minimal retro aesthetic with animated grid background
- ⚡ **Fast Search** - Real-time search across Arweave web content and ARNS domains
- 🔍 **Dual Search Modes** - Search web content or ARNS domains
- 📱 **Responsive Design** - Works perfectly on desktop and mobile
- 🌟 **Smooth Animations** - Retro-style loading animations and hover effects
- 🎯 **Section Selector** - Choose between Web, Images, Videos, and Audios (Web active)
- 🔄 **Random Retrieval** - Type a single letter for instant random search results
- 📊 **Status Page** - Monitor backend health and indexing statistics

## Tech Stack

- **Frontend**: React 19 + Vite 7
- **Styling**: CSS with retro grid animations
- **Backend**: FastAPI (Python) - deployed at arfetch.adityaberry.me
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
│   │   ├── LoadingGrid.jsx    # Animated loading component
│   │   └── ToggleSlider.css   # Retro toggle slider styles
│   ├── pages/
│   │   └── Status.jsx         # Status page component
│   ├── App.jsx               # Main app component
│   ├── main.jsx              # React entry point
│   └── index.css             # Global retro styling
├── public/                   # Static assets
├── index.html               # Vite HTML template
├── vite.config.js           # Vite configuration with API proxy
├── package.json             # Frontend dependencies
└── README.md               # This file
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
- `GET /api/status` - Get backend health and statistics

## Features

### Search Interface
- **Section Selector**: Choose between Web, Images, Videos, and Audios
- **Toggle Slider**: Switch between "All" and "ARNS" search modes
- **Random Retrieval**: Type a single letter for instant search results
- **Clickable Results**: Click any result card to open the URL in a new tab

### Retro Design
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

3. Deploy the `dist/` folder to any static hosting service

## Customization

### Colors
Edit the CSS variables in `src/index.css`:
```css
:root {
  --primary-color: #00ff41;
  --secondary-color: #23272e;
  --accent-color: #ff6b35;
  /* ... */
}
```

### Grid Animation
Modify the grid animation in `src/index.css`:
```css
@keyframes gridMove {
  0% { transform: translate(0, 0); }
  100% { transform: translate(48px, 48px); }
}
```

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Mobile browsers

## License

MIT License
