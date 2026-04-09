ALTER TABLE IF EXISTS scheduled_tasks
    DROP CONSTRAINT IF EXISTS scheduled_tasks_agent_id_fkey;

ALTER TABLE IF EXISTS user_preferences
    DROP CONSTRAINT IF EXISTS user_preferences_agent_id_fkey;
