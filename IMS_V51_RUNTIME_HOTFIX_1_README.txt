IMS v5.1 Runtime Hotfix 1

Purpose:
1. Fix PostgreSQL error:
   SELECT DISTINCT ... ORDER BY lower(filename)
2. Raise AI interpretation output budget from 950 to 1400 tokens to reduce truncated JSON planning.
3. Replace the misleading generic "language model request failed" message for unexpected backend failures.

Apply:
  python .\apply_ims_v51_runtime_hotfix_1.py --repo .

Then rebuild/recreate backend:
  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml build backend
  docker compose -f docker-compose.yml -f docker-compose.smart-rag.yml -f docker-compose.v5.yml up -d --force-recreate backend

No PDF reprocessing is required.
