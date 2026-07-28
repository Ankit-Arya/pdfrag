# pdfrag frontend UI patch v5

This add-on is designed to be extracted over the project after applying the v4 PostgreSQL/JWT patch.

## Included UI

- Login and logout with JWT access/refresh tokens.
- Automatic access-token refresh.
- Role-aware navigation.
- Admin PDF upload, processing, status, reprocessing, and deletion.
- Admin user account creation and activation/deactivation.
- User Q&A without a temporary `collection_id`.
- Saved chat list, chat reload, and chat deletion.
- Account/session screen with refresh-session revocation.
- Shared knowledge status showing ready PDFs and stored chunks.

## Included backend API additions

The UI requires a few endpoints not present in v4:

- `GET/POST /api/admin/users`
- `PATCH /api/admin/users/{user_id}`
- `GET /api/knowledge/status`
- `GET /api/chats`
- `GET/DELETE /api/chats/{chat_id}`
- `GET /api/auth/sessions`
- `DELETE /api/auth/sessions/{session_id}`

No new database migration is required because these changes use the existing v4 tables.

## Apply

Extract the ZIP at the repository root and allow replacements.

Then rebuild both backend and frontend:

```powershell
docker compose down --remove-orphans
docker compose build --no-cache backend frontend
docker compose up -d
docker compose logs -f backend frontend
```

Open:

```text
http://localhost:8081/
```

Sign in with the bootstrap administrator configured in `.env`:

```env
BOOTSTRAP_ADMIN_EMAIL=admin@example.com
BOOTSTRAP_ADMIN_PASSWORD=change-this-password
```

The administrator can create normal user accounts from **Admin console → Users**.

## JWT storage behavior

- Access token: `sessionStorage`
- Refresh token: `localStorage`
- Expired access tokens are refreshed automatically.
- Revoked or invalid refresh sessions return the user to the login screen.

For a production deployment, an HttpOnly secure-cookie refresh-token flow is preferable to browser storage.

## v5.1 build correction

This revision removes two unused `props` bindings that caused strict `vue-tsc`
build failures in `AccountPanel.vue` and `UploadPanel.vue`. Component behavior and
public props are unchanged.
