# Tesis / Superlog

Two independent codebases at the top level:

- **`superlog/`** — Superlog monorepo (TS, pnpm workspace, Turborepo). The main project.
- **`scripts/`** — Python log analysis pipeline (separate virtualenv). See `scripts/AGENTS.md`.

## superlog/ — TypeScript monorepo

### Stack

| Property | Value |
|---|---|
| Package manager | `pnpm@9.12.0` |
| Runtime | Node >=20. **api/worker/proxy require >=22.12.0** |
| Build system | Turborepo v2 (`turbo.json`: typecheck depends on `^build`) |
| Linter | Biome 1.9.x (`pnpm lint` = `biome check .`); VCS integration enabled (respects `.gitignore`) |
| Formatter | Biome, 2-space, double quotes, semicolons always, trailing commas all, **lineWidth 100** |
| TypeScript | strict, `noUncheckedIndexedAccess`, `noImplicitOverride`, ESNext module, Bundler resolution |
| Module | ESM (`"type": "module"`) |

### Key commands (run from `superlog/`)

```bash
pnpm install                          # install deps
pnpm install --frozen-lockfile        # CI use
pnpm dev                              # turbo run dev (all apps in parallel)
pnpm build                            # turbo run build
pnpm typecheck                        # turbo run typecheck (requires build first)
pnpm lint                             # biome check .
pnpm format                           # biome format --write .
```

### Local dev stack (requires Docker)

```bash
docker compose up -d                    # postgres:5434, clickhouse:8123/9000, otel collector
pnpm --filter @superlog/db db:migrate   # run DB migrations
pnpm dev                                # start all apps
```

| Service | URL |
|---|---|
| Web app (Vite) | http://localhost:5173 |
| API | http://localhost:4100 |
| OTLP intake | http://localhost:4101 |

**Postgres port**: mapped to **5434** locally (not 5432).

Alternative Procfile-based start with `overmind` / `honcho`:

```bash
overmind start -f Procfile              # runs all 4 apps, logs to tmp/logs/
pnpm dev:portless                       # start via Procfile.portless
```

### Testing

Uses **Node built-in test runner** (`tsx --test`), not vitest/jest.

```bash
# Single package:
pnpm --filter @superlog/worker test

# Single test file:
pnpm --filter @superlog/api exec tsx --test src/index.test.ts

# CI order (verify → tests):
pnpm test:ci-workflows   # security check
pnpm typecheck
pnpm build
# then (postgres required):
pnpm -r --filter '!@superlog/db' --if-present test               # all packages except db
pnpm --filter @superlog/worker test:telemetry-ingest              # telemetry ingestion
pnpm --filter @superlog/db test                                   # db package tests
```

**Worker** has the largest test surface — 40+ named test scripts (see `apps/worker/package.json`).

**Web tests** inject a `navigator` polyfill via `tsx --import ./src/test-setup.ts` (handles `@pierre/diffs` browser dependency). **Web build** includes an SSR prerender step (`pnpm prerender`).

**`@superlog/db`** exports multiple subpaths: `.`, `./schema`, `./keys`, `./agent-pr-retry-domain`.

**Test env vars**: `DATABASE_URL=postgres://postgres:postgres@localhost:5434/superlog` + `BETTER_AUTH_SECRET`.

### OTel tracing bootstrap

Three apps (`api`, `proxy`, `worker`) share the same pattern:
- Entrypoint via `--import ./tracing.ts` (loaded before app code).
- Loads `.env.superlog` → `.env` → `.env.local` → `$SUPERLOG_ENV_FILE` (in that order) **before** OTel SDK reads env.
- Telemetry **disabled in non-production** (sets `OTEL_SDK_DISABLED=true` unless `SUPERLOG_ENV=production`). To dogfood: set `SUPERLOG_ENV=production` in `.env.local`.
- HTTP/protobuf exporters only (no gRPC).
- The proxy's `start` script uses `node --import tsx --import ./tracing.ts` (not plain `tsx`).
- The `web` app uses client-side OTel: `WebTracerProvider` + `ZoneContextManager` in `src/instrumentation.ts`.

### CI (`.github/workflows/ci.yaml`)

- Triggers on **PRs** to `main`.
- `verify` job: `test:ci-workflows` → `typecheck` → `build`.
- `tests` job (needs `verify`): starts `docker compose up -d postgres`, runs tests in 3 phases.
- Node 20, pnpm cache.

### Skills

12 OTel-style skills in `.agents/skills/`, `.claude/skills/`, `agent/skills/` (identical copies at root and inside `superlog/`). Sourced from `superloglabs/skills` (see `skills-lock.json`).

### Notable quirks

- **Biome** uses `vcs.useIgnoreFile: true` — lint/format respects `.gitignore`.
- **Root `package.json`** has dependency overrides (protobufjs, esbuild, postcss, etc.).
- **`apps/api/.env.example`** documents 50+ env vars (OAuth, integrations, secrets). A pre-populated `.env` is checked in alongside it.
- **Docker compose**: postgres 16, clickhouse 26.1, otel/opentelemetry-collector-contrib 0.150.1.
- **Pricing** defined in `autumn.config.ts` (metadata-only; source of truth is `packages/billing/src/pricing.ts`).

## scripts/ — Python log analysis pipeline

See `scripts/AGENTS.md` for full reference. Quick start:

```bash
cd scripts && source .venv/bin/activate
python main.py --mode generative --backend ollama --model-name qwen3.6:latest
```

Key facts: Python 3.13, ROCm GPU (not CUDA), JSONL output, no tests/CI/lint/config. `main.py` calls `init_observability()` at module level (before `main()`).
