PDFrag backend build fix
========================

Replace backend/Dockerfile with the included file.

Before rebuilding on Docker Desktop/Windows:

1. Quit memory-heavy applications if Docker is under memory pressure.
2. From an elevated PowerShell or Command Prompt:

   wsl --update
   wsl --shutdown

3. Restart Docker Desktop.
4. Remove the old BuildKit package cache (this does not delete your PostgreSQL volume):

   docker builder prune -af

5. Rebuild only the backend first:

   docker compose build --no-cache backend

6. If that succeeds:

   docker compose up -d

Useful checks:

   docker info
   docker system df
   wsl --version
   wsl --status

If PyTorch still exits with code 139 during installation, increase the memory/swap
available to Docker Desktop/WSL2 and retry. Do not delete named volumes; your
postgres_data volume contains the application database.
