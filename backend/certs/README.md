# Custom certificate authorities

When a company proxy, antivirus, VPN, or firewall performs TLS inspection, export
its trusted root certificate in PEM/CRT format and place it in this directory
before building the backend, for example:

```text
backend/certs/organization-root-ca.crt
```

The Dockerfile installs every `.crt` file here into Debian's trusted CA store.
Do not commit private client certificates or private keys.
