# Security Package Plan

This package will hold production security primitives before any public launch.

- Envelope encryption for broker tokens, API secrets, notification secrets, and sensitive settings.
- Password hashing with Argon2id or bcrypt using per-user salts.
- JWT/session signing and rotation.
- Request signing for internal jobs where needed.
- Secret redaction for logs and API responses.
- Audit-event helpers for login, broker connect, order placement, settings changes, and admin access.
- Key loading from environment or a managed KMS, never from frontend payloads.
- Helpers for secure random IDs, CSRF state, OAuth state, and nonce generation.

