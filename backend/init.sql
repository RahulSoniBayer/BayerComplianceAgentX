-- Initialize database for Bayer Compliance Agent
-- This script runs when the PostgreSQL container starts for the first time

-- Create database if it doesn't exist (handled by POSTGRES_DB env var)
-- CREATE DATABASE bayer_agent;

-- Connect to the database
\c bayer_agent;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for better performance
-- These will be created automatically by SQLAlchemy, but we can add custom ones here

-- Custom indexes for better search performance
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_content_gin 
-- ON document_chunks USING gin(to_tsvector('english', content));

-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_metadata_gin 
-- ON document_chunks USING gin(metadata);

-- Set up database permissions
GRANT ALL PRIVILEGES ON DATABASE bayer_agent TO bayer_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO bayer_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO bayer_user;

-- Create a function to clean up old files (optional)
CREATE OR REPLACE FUNCTION cleanup_old_files()
RETURNS void AS $$
BEGIN
    -- This function can be used to clean up old files and records
    -- Example: Delete records older than 30 days
    -- DELETE FROM pdf_documents WHERE created_at < NOW() - INTERVAL '30 days';
    -- DELETE FROM template_tasks WHERE created_at < NOW() - INTERVAL '30 days';
    
    -- Log the cleanup
    INSERT INTO processing_logs (log_level, message, details) 
    VALUES ('INFO', 'Database cleanup completed', '{"timestamp": "' || NOW() || '"}');
END;
$$ LANGUAGE plpgsql;

-- Create a scheduled job (if pg_cron extension is available)
-- SELECT cron.schedule('cleanup-old-files', '0 2 * * *', 'SELECT cleanup_old_files();');

-- Insert initial configuration (optional)
INSERT INTO processing_logs (log_level, message, details) 
VALUES ('INFO', 'Database initialized', '{"version": "1.0.0", "timestamp": "' || NOW() || '"}')
ON CONFLICT DO NOTHING;
