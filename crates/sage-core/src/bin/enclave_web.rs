use anyhow::Result;
use tokio::net::TcpListener;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use sage_core::{
    config::Config, sage_agent::SageAgent, web_runtime, web_runtime::EnclaveWebConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "sage=info,enclave_web=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    dotenvy::dotenv().ok();

    let config = Config::from_env()?;
    let web_config = EnclaveWebConfig::from_env()?;
    let api_key = config
        .tinfoil_api_key
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("TINFOIL_API_KEY not set"))?;

    {
        use diesel::prelude::*;
        use diesel_migrations::{embed_migrations, EmbeddedMigrations, MigrationHarness};
        const MIGRATIONS: EmbeddedMigrations = embed_migrations!("migrations");
        let mut conn = diesel::PgConnection::establish(&config.database_url)?;
        conn.run_pending_migrations(MIGRATIONS)
            .map_err(|e| anyhow::anyhow!("Migration failed: {}", e))?;
    }

    SageAgent::configure_lm_with_temperature(
        &config.tinfoil_api_url,
        api_key,
        &config.tinfoil_model,
        0.1,
    )
    .await?;

    let app = web_runtime::build_router(config.clone(), web_config.clone())?;
    let listener = TcpListener::bind(("0.0.0.0", web_config.http_port)).await?;
    info!("enclave_web listening on 0.0.0.0:{}", web_config.http_port);
    axum::serve(listener, app).await?;
    Ok(())
}
