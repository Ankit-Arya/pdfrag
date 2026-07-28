# pdfrag multi-folder PDF selection patch v5.2

Apply this overlay after frontend patch v5.1.

## What changed

- PDF selections now accumulate instead of replacing the existing queue.
- You can click **Add PDFs** repeatedly and choose files from different folders.
- You can click **Add folder** repeatedly to add PDFs from multiple folders in Chromium-based browsers such as Microsoft Edge and Google Chrome.
- Non-PDF files are ignored.
- Duplicate selections are ignored using relative path, size, and modified time.
- Folder-relative paths are shown when available.
- Individual queued files can be removed before upload.
- The complete queue can be cleared without refreshing the page.

## Apply

Extract this ZIP at the project root and allow the two frontend files to be replaced.

Then run:

```powershell
docker compose build --no-cache frontend
docker compose up -d --force-recreate frontend nginx
docker compose logs -f frontend
```

Open `http://localhost:8081/`, sign in as admin, and go to **Admin console > Documents**.
