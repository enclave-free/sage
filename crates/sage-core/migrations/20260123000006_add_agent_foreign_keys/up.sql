DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'user_preferences_agent_id_fkey'
    ) THEN
        ALTER TABLE user_preferences
            ADD CONSTRAINT user_preferences_agent_id_fkey
            FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE;
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'scheduled_tasks_agent_id_fkey'
    ) THEN
        ALTER TABLE scheduled_tasks
            ADD CONSTRAINT scheduled_tasks_agent_id_fkey
            FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE;
    END IF;
END
$$;
