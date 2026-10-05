# Enterprise AI Knowledge & Productivity Assistant

## What this project is
A learning-focused portfolio project: an AI assistant that answers questions
from company documents, with RAG, tool calling, agents, MCP, security,
evaluation and observability. Built over a 28-day plan.

## Architecture
- frontend/ : Next.js + React + TypeScript (chat UI)
- backend/  : Python + FastAPI, managed with uv
- Model: Claude via the Anthropic API (AWS Bedrock later, behind a provider interface)
- Database (from Week 2): PostgreSQL + pgvector

## How we work (important)
- The developer is learning. Before writing code for any non-trivial task,
  explain the plan in simple words and wait for approval.
- Do not make architecture or library decisions without asking first.
- Only do what was asked. No extra features, refactors or "improvements".
- Work in small steps. After each step, explain what changed and why.
- When a test or command fails, explain the cause before fixing it.

## Security rules (never break these)
- Secrets (API keys) live only in .env files. Never hardcode them.
- Never commit .env files. Keep .env.example with placeholder values.
- Authorization is decided by application code and database queries,
  never by the LLM.

## Commands
- Backend: run from backend/ using uv (e.g. uv run ...)
- Frontend: run from frontend/ using npm