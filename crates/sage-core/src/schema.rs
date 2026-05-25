// @generated automatically by Diesel CLI.
// Note: Some types manually adjusted for pgvector and UUID support

diesel::table! {
    use diesel::sql_types::*;
    use pgvector::sql_types::Vector;

    agents (id) {
        id -> Uuid,
        name -> Varchar,
        system_prompt -> Text,
        message_ids -> Array<Uuid>,
        llm_config -> Jsonb,
        last_memory_update -> Nullable<Timestamptz>,
        max_context_tokens -> Int4,
        compaction_threshold -> Float4,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use pgvector::sql_types::Vector;

    blocks (id) {
        id -> Uuid,
        agent_id -> Text,
        label -> Varchar,
        description -> Nullable<Text>,
        value -> Text,
        char_limit -> Int4,
        read_only -> Bool,
        version -> Int4,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use pgvector::sql_types::Vector;

    messages (id) {
        id -> Uuid,
        agent_id -> Uuid,
        user_id -> Text,
        role -> Text,
        content -> Text,
        // embedding handled via raw SQL due to pgvector complexity
        sequence_id -> Int8,
        tool_calls -> Nullable<Jsonb>,
        tool_results -> Nullable<Jsonb>,
        created_at -> Timestamptz,
        attachment_text -> Nullable<Text>,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use pgvector::sql_types::Vector;

    passages (id) {
        id -> Uuid,
        agent_id -> Text,
        content -> Text,
        embedding -> Nullable<Vector>,
        tags -> Array<Text>,
        created_at -> Timestamptz,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use pgvector::sql_types::Vector;

    summaries (id) {
        id -> Uuid,
        agent_id -> Uuid,
        from_sequence_id -> Int8,
        to_sequence_id -> Int8,
        content -> Text,
        embedding -> Nullable<Vector>,
        previous_summary_id -> Nullable<Uuid>,
        created_at -> Timestamptz,
    }
}

diesel::table! {
    use diesel::sql_types::*;

    user_preferences (id) {
        id -> Uuid,
        agent_id -> Uuid,
        key -> Varchar,
        value -> Text,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
    }
}

diesel::table! {
    use diesel::sql_types::*;

    scheduled_tasks (id) {
        id -> Uuid,
        agent_id -> Uuid,
        task_type -> Varchar,
        payload -> Jsonb,
        next_run_at -> Timestamptz,
        cron_expression -> Nullable<Varchar>,
        timezone -> Varchar,
        status -> Varchar,
        last_run_at -> Nullable<Timestamptz>,
        run_count -> Int4,
        last_error -> Nullable<Text>,
        description -> Text,
        created_at -> Timestamptz,
    }
}

diesel::table! {
    use diesel::sql_types::*;

    ai_config (key) {
        key -> Varchar,
        value -> Text,
        value_type -> Varchar,
        category -> Varchar,
        description -> Nullable<Text>,
        updated_at -> Timestamptz,
    }
}

diesel::table! {
    use diesel::sql_types::*;

    ai_config_user_type_overrides (id) {
        id -> Uuid,
        ai_config_key -> Varchar,
        user_type_id -> Int4,
        value -> Text,
        updated_at -> Timestamptz,
    }
}

diesel::table! {
    use diesel::sql_types::*;

    chat_contexts (id) {
        id -> Uuid,
        signal_identifier -> Text,
        context_type -> Varchar,
        display_name -> Nullable<Text>,
        created_at -> Timestamptz,
        reply_context -> Nullable<Text>,
    }
}

diesel::table! {
    use diesel::sql_types::*;

    external_identities (id) {
        id -> Uuid,
        identity_type -> Varchar,
        external_id -> Varchar,
        display_name -> Nullable<Text>,
        user_type_id -> Nullable<Int4>,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
    }
}

diesel::joinable!(scheduled_tasks -> agents (agent_id));

diesel::table! {
    use diesel::sql_types::*;

    web_sessions (id) {
        id -> Uuid,
        agent_id -> Uuid,
        owner_type -> Varchar,
        owner_id -> Varchar,
        user_type_id -> Nullable<Int4>,
        last_question -> Nullable<Text>,
        title -> Nullable<Text>,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
    }
}

diesel::allow_tables_to_appear_in_same_query!(
    ai_config,
    ai_config_user_type_overrides,
    agents,
    blocks,
    chat_contexts,
    external_identities,
    messages,
    passages,
    summaries,
    user_preferences,
    scheduled_tasks,
    web_sessions,
);
