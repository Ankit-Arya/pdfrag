-- Optional manual migration. Normal startup already creates this table through
-- SQLAlchemy Base.metadata.create_all(). Use this only when the application DB
-- role is not allowed to create tables at startup.

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    success BOOLEAN NOT NULL,
    user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL,
    actor_email VARCHAR(320) NULL,
    chat_session_id UUID NULL REFERENCES chat_sessions(id) ON DELETE SET NULL,
    question TEXT NULL,
    response TEXT NULL,
    error_message TEXT NULL,
    event_metadata JSONB NOT NULL,
    ip_address VARCHAR(64) NULL,
    user_agent VARCHAR(500) NULL,
    request_id VARCHAR(100) NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS ix_audit_logs_success ON audit_logs(success);
CREATE INDEX IF NOT EXISTS ix_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS ix_audit_logs_actor_email ON audit_logs(actor_email);
CREATE INDEX IF NOT EXISTS ix_audit_logs_chat_session_id ON audit_logs(chat_session_id);
CREATE INDEX IF NOT EXISTS ix_audit_logs_request_id ON audit_logs(request_id);
CREATE INDEX IF NOT EXISTS ix_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS ix_audit_logs_event_created ON audit_logs(event_type, created_at);
CREATE INDEX IF NOT EXISTS ix_audit_logs_user_created ON audit_logs(user_id, created_at);
