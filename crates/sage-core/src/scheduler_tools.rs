//! Scheduler Tools
//!
//! Tools for scheduling future messages and tool calls:
//! - schedule_task: Create a one-off or recurring scheduled task
//! - list_schedules: List scheduled tasks
//! - cancel_schedule: Cancel a pending scheduled task

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use uuid::Uuid;

use crate::sage_agent::{tool_string_arg, LegacyTool, ToolArgs, ToolResult};
use crate::scheduler::{
    is_cron_expression, next_cron_time, parse_cron, parse_datetime, MessagePayload, SchedulerDb,
    TaskPayload, TaskType, ToolCallPayload,
};

// ============================================================================
// Schedule Task Tool
// ============================================================================

pub struct ScheduleTaskTool {
    scheduler_db: Arc<SchedulerDb>,
    agent_id: Uuid,
    default_timezone: String,
}

impl ScheduleTaskTool {
    pub fn new(scheduler_db: Arc<SchedulerDb>, agent_id: Uuid, default_timezone: String) -> Self {
        Self {
            scheduler_db,
            agent_id,
            default_timezone,
        }
    }
}

#[async_trait]
impl LegacyTool for ScheduleTaskTool {
    fn name(&self) -> &str {
        "schedule_task"
    }

    fn description(&self) -> &str {
        "Schedule a future message or tool execution. Supports one-off (ISO datetime) or recurring (cron expression)."
    }

    fn args_schema(&self) -> &str {
        r#"{"task_type": "message|tool_call", "description": "human-readable description", "run_at": "ISO datetime (2026-01-26T15:30:00Z) or cron (0 9 * * MON-FRI)", "payload": "JSON: {\"message\": \"...\"} for message, {\"tool\": \"name\", \"args\": {...}} for tool_call", "timezone": "optional IANA timezone for cron (default: user preference or UTC)"}"#
    }

    async fn execute(&self, args: &ToolArgs) -> Result<ToolResult> {
        // Parse task_type
        let task_type_str = tool_string_arg(args, "task_type").ok_or_else(|| {
            anyhow::anyhow!("'task_type' argument required (message or tool_call)")
        })?;
        let task_type: TaskType = task_type_str
            .parse()
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Parse description
        let description = tool_string_arg(args, "description")
            .ok_or_else(|| anyhow::anyhow!("'description' argument required"))?
            .to_string();

        // Parse run_at (datetime or cron)
        let run_at = tool_string_arg(args, "run_at").ok_or_else(|| {
            anyhow::anyhow!("'run_at' argument required (ISO datetime or cron expression)")
        })?;

        // Get timezone (from args, or use default)
        let timezone = tool_string_arg(args, "timezone")
            .map(str::to_string)
            .unwrap_or_else(|| self.default_timezone.clone());

        // Determine if cron or one-off
        let (next_run_at, cron_expression): (DateTime<Utc>, Option<String>) = if is_cron_expression(
            run_at,
        ) {
            // Validate cron expression
            if let Err(e) = parse_cron(run_at) {
                return Ok(ToolResult::error(format!(
                        "Invalid cron expression: {}. Use standard cron format (e.g., '0 9 * * MON-FRI' for weekdays at 9am).",
                        e
                    )));
            }

            // Calculate next run time
            match next_cron_time(run_at, &timezone) {
                Ok(next) => (next, Some(run_at.to_string())),
                Err(e) => {
                    return Ok(ToolResult::error(format!(
                        "Failed to calculate next run time: {}",
                        e
                    )))
                }
            }
        } else {
            // Parse as datetime
            match parse_datetime(run_at) {
                Ok(dt) => {
                    if dt <= Utc::now() {
                        return Ok(ToolResult::error("Scheduled time must be in the future."));
                    }
                    (dt, None)
                }
                Err(e) => return Ok(ToolResult::error(format!("Invalid datetime: {}", e))),
            }
        };

        // Parse payload
        let payload_value = args
            .get("payload")
            .ok_or_else(|| anyhow::anyhow!("'payload' argument required"))?;
        let payload_value = match payload_value {
            serde_json::Value::String(payload) => serde_json::from_str(payload)
                .map_err(|error| anyhow::anyhow!("Invalid payload JSON: {}", error))?,
            payload => payload.clone(),
        };

        let payload: TaskPayload = match task_type {
            TaskType::Message => {
                // Try to parse as MessagePayload
                match serde_json::from_value::<MessagePayload>(payload_value.clone()) {
                    Ok(p) => TaskPayload::Message(p),
                    Err(_) => {
                        if let Some(msg) = payload_value.get("message").and_then(|m| m.as_str()) {
                            TaskPayload::Message(MessagePayload {
                                message: msg.to_string(),
                            })
                        } else {
                            return Ok(ToolResult::error(
                                "Message payload must have a 'message' field. Example: {\"message\": \"Your reminder text\"}"
                            ));
                        }
                    }
                }
            }
            TaskType::ToolCall => {
                match serde_json::from_value::<ToolCallPayload>(payload_value.clone()) {
                    Ok(p) => TaskPayload::ToolCall(p),
                    Err(_) => {
                        return Ok(ToolResult::error(
                            "Tool call payload must contain a tool name and string arguments. Example: {\"tool\": \"web_search\", \"args\": {\"query\": \"...\"}}"
                        ));
                    }
                }
            }
        };

        // Create the task
        match self.scheduler_db.create_task(
            self.agent_id,
            task_type.clone(),
            payload,
            next_run_at,
            cron_expression.clone(),
            timezone.clone(),
            description.clone(),
        ) {
            Ok(task) => {
                let schedule_type = if cron_expression.is_some() {
                    "recurring"
                } else {
                    "one-off"
                };

                Ok(ToolResult::success(format!(
                    "Scheduled {} {} task '{}' (id: {}). Next run: {}",
                    schedule_type,
                    task_type.as_str(),
                    description,
                    task.id,
                    next_run_at.format("%Y-%m-%d %H:%M:%S UTC")
                )))
            }
            Err(e) => Ok(ToolResult::error(format!("Failed to create task: {}", e))),
        }
    }
}

// ============================================================================
// List Schedules Tool
// ============================================================================

pub struct ListSchedulesTool {
    scheduler_db: Arc<SchedulerDb>,
    agent_id: Uuid,
}

impl ListSchedulesTool {
    pub fn new(scheduler_db: Arc<SchedulerDb>, agent_id: Uuid) -> Self {
        Self {
            scheduler_db,
            agent_id,
        }
    }
}

#[async_trait]
impl LegacyTool for ListSchedulesTool {
    fn name(&self) -> &str {
        "list_schedules"
    }

    fn description(&self) -> &str {
        "List scheduled tasks. By default shows pending tasks only."
    }

    fn args_schema(&self) -> &str {
        r#"{"status": "optional filter: pending, completed, failed, cancelled, or all (default: pending)"}"#
    }

    async fn execute(&self, args: &ToolArgs) -> Result<ToolResult> {
        let status_filter = tool_string_arg(args, "status");

        // Convert "all" to None for no filtering
        let status_filter = match status_filter {
            Some("all") => None,
            other => other,
        };

        match self
            .scheduler_db
            .get_tasks_by_agent(self.agent_id, status_filter)
        {
            Ok(tasks) => {
                if tasks.is_empty() {
                    return Ok(ToolResult::success("No scheduled tasks found."));
                }

                let mut output = format!("Found {} scheduled task(s):\n\n", tasks.len());

                for task in tasks {
                    let schedule_type = if let Some(cron) = &task.cron_expression {
                        format!("recurring ({})", cron)
                    } else {
                        "one-off".to_string()
                    };

                    output.push_str(&format!(
                        "- [{}] {} ({})\n  ID: {}\n  Type: {}\n  Next run: {}\n  Status: {:?}\n  Runs: {}\n\n",
                        task.status.as_str(),
                        task.description,
                        schedule_type,
                        task.id,
                        task.task_type.as_str(),
                        task.next_run_at.format("%Y-%m-%d %H:%M:%S UTC"),
                        task.status,
                        task.run_count,
                    ));
                }

                Ok(ToolResult::success(output))
            }
            Err(e) => Ok(ToolResult::error(format!("Failed to list tasks: {}", e))),
        }
    }
}

// ============================================================================
// Cancel Schedule Tool
// ============================================================================

pub struct CancelScheduleTool {
    scheduler_db: Arc<SchedulerDb>,
}

impl CancelScheduleTool {
    pub fn new(scheduler_db: Arc<SchedulerDb>) -> Self {
        Self { scheduler_db }
    }
}

#[async_trait]
impl LegacyTool for CancelScheduleTool {
    fn name(&self) -> &str {
        "cancel_schedule"
    }

    fn description(&self) -> &str {
        "Cancel a pending scheduled task by ID."
    }

    fn args_schema(&self) -> &str {
        r#"{"id": "UUID of the task to cancel"}"#
    }

    async fn execute(&self, args: &ToolArgs) -> Result<ToolResult> {
        let id_str =
            tool_string_arg(args, "id").ok_or_else(|| anyhow::anyhow!("'id' argument required"))?;

        let task_id: Uuid = id_str
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid UUID format: {}", id_str))?;

        match self.scheduler_db.cancel_task(task_id) {
            Ok(true) => Ok(ToolResult::success(format!(
                "Successfully cancelled task {}",
                task_id
            ))),
            Ok(false) => Ok(ToolResult::error(format!(
                "Task {} not found or not in pending status (only pending tasks can be cancelled)",
                task_id
            ))),
            Err(e) => Ok(ToolResult::error(format!("Failed to cancel task: {}", e))),
        }
    }
}
