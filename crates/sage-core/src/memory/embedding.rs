//! Embedding Service
//!
//! Shared embedding generation for all memory tiers.
//! Uses a Tinfoil/OpenAI-compatible embeddings API with nomic-embed-text
//! output (768 dimensions).

#![allow(dead_code)]

use anyhow::{Context, Result};

/// Embedding dimension for nomic-embed-text
pub const EMBEDDING_DIM: usize = 768;

/// Shared embedding service for generating vector embeddings
#[derive(Clone)]
pub struct EmbeddingService {
    api_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl EmbeddingService {
    /// Create a new embedding service
    pub fn new(api_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            api_url: api_url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Generate an embedding for a single text
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let response = self
            .client
            .post(format!("{}/embeddings", self.api_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "model": &self.model,
                "input": text,
                "encoding_format": "float"  // Important: avoid base64 encoding issues
            }))
            .send()
            .await
            .context("Failed to call embeddings API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Embeddings API returned {}: {}", status, body);
        }

        let json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse embeddings API response")?;
        let embedding = json["data"][0]["embedding"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Embeddings API response missing data[0].embedding"))?;

        let vec: Vec<f32> = embedding
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| anyhow::anyhow!("Embeddings API returned non-float value"))
            })
            .collect::<Result<_>>()?;

        if vec.len() != EMBEDDING_DIM {
            anyhow::bail!(
                "Unexpected embedding dimension: {} (expected {})",
                vec.len(),
                EMBEDDING_DIM
            );
        }

        Ok(vec)
    }

    /// Generate embeddings for multiple texts (batched)
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let response = self
            .client
            .post(format!("{}/embeddings", self.api_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "model": &self.model,
                "input": texts,
                "encoding_format": "float"
            }))
            .send()
            .await
            .context("Failed to call batch embeddings API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Batch embeddings API returned {}: {}", status, body);
        }

        let json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse batch embeddings API response")?;
        let data = json["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Batch embeddings API response missing data array"))?;

        if data.len() != texts.len() {
            anyhow::bail!(
                "Batch embeddings API returned {} embeddings for {} texts",
                data.len(),
                texts.len()
            );
        }

        data.iter()
            .map(|item| {
                let embedding = item["embedding"].as_array().ok_or_else(|| {
                    anyhow::anyhow!("Batch embeddings API item missing embedding")
                })?;

                let vec: Vec<f32> = embedding
                    .iter()
                    .map(|v| {
                        v.as_f64().map(|f| f as f32).ok_or_else(|| {
                            anyhow::anyhow!("Batch embeddings API returned non-float value")
                        })
                    })
                    .collect::<Result<_>>()?;

                if vec.len() != EMBEDDING_DIM {
                    anyhow::bail!(
                        "Unexpected embedding dimension: {} (expected {})",
                        vec.len(),
                        EMBEDDING_DIM
                    );
                }

                Ok(vec)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_dimension_is_expected() {
        assert_eq!(EMBEDDING_DIM, 768);
    }
}
