# Security policy

This reference app intentionally has no authentication. Do not expose it to an untrusted network without adding access control, TLS, rate limiting, and tenant isolation.

Uploaded PDF content is untrusted. The application places it only inside labeled source blocks and instructs the model to ignore embedded instructions. This reduces prompt-injection risk but does not eliminate all model-level attacks.

Never commit `.env` or API credentials. For production, use a secrets manager and redact sensitive payloads from logs and observability systems.
