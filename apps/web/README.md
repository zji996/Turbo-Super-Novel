# I2V Studio (Web Frontend)

Image-to-Video (I2V) generation frontend for the Turbo Novel platform.

## Tech Stack

- React 18 + TypeScript
- Vite
- TailwindCSS v4
- React Router v7

## Getting Started

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Type check
pnpm typecheck

# Lint
pnpm lint

# Build for production
pnpm build
```

## Environment Variables

Copy `env.example` to `.env` and configure:

- `VITE_API_BASE_URL`: Backend API URL (default: `http://localhost:8000`)

## Routes

- `/tools/i2v` - I2V Studio (MVP)
- `/projects` - Coming Soon
- `/assets` - Coming Soon
