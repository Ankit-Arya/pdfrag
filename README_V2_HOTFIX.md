# V2 frontend type hotfix

This package adds the missing replacement for:

- `frontend/src/services/api.ts`

The frontend build failed because `ChatPanel.vue` used `grounding_status`, while
the old TypeScript `AnswerResponse` interface did not declare that property.

No frontend dependency versions were changed by this hotfix.

Rebuild both services:

```powershell
docker compose down --remove-orphans
docker compose build --no-cache backend frontend
docker compose up -d --force-recreate
```
