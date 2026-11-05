# Document QA Bot — React frontend

React + TypeScript + Vite UI for the FastAPI backend in `../backend`.

## Run

```bash
npm install
npm run dev        # http://localhost:3000
```

The backend must be running (default `http://localhost:8030`). Set its URL in `.env`:

```
VITE_API_BASE_URL=http://localhost:8030
```

Only `VITE_*` variables are bundled into the browser code, and anyone can read them.
Never put Groq, Cohere or other API keys here; they belong in `backend/.env`.

If you serve the frontend from another origin, add it to the backend's `CORS_ORIGINS`.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Type-check, then build to `dist/` |
| `npm run preview` | Serve the production build |
| `npm run typecheck` | TypeScript only |
| `npm test` | Vitest unit/component tests (API calls are mocked) |

## Layout

```
src/
  services/api.ts   all HTTP calls (Axios) and error-message handling
  types/            TypeScript copies of the backend's Pydantic schemas
  context/          shared state: document list + activity, chat history
  hooks/            evaluation report fetching
  components/       layout, ui (shared pieces), documents, chat, sources, evaluation
  pages/            Dashboard, Documents, Chat, Evaluation (one per route)
```
