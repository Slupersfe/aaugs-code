use std::path::PathBuf;

use anyhow::Context;

const CURRENT_RELEASE_FALLBACK: u32 = 1;

const RELEASE_URL: &str = "https://code.aaugs.com/release.txt";
const REMOTE_REPO: &str = "https://github.com/Slupersfe/aaugs-code.git";

fn release_config_path() -> anyhow::Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not find home directory"))?;
    Ok(home.join("vibe").join("release.config"))
}

pub fn current_release() -> anyhow::Result<u32> {
    let path = release_config_path()?;
    if !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, format!("{}\n", CURRENT_RELEASE_FALLBACK))?;
        return Ok(CURRENT_RELEASE_FALLBACK);
    }
    let content = std::fs::read_to_string(&path)?;
    let n: u32 = content.trim().parse()
        .with_context(|| format!("invalid release.config: {:?}", content.trim()))?;
    Ok(n)
}

fn set_release(ver: u32) -> anyhow::Result<()> {
    let path = release_config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, format!("{}\n", ver))?;
    Ok(())
}

pub async fn check_update() -> anyhow::Result<Option<u32>> {
    let current = current_release()?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let resp = client.get(RELEASE_URL).send().await?;
    let text = resp.text().await?;
    let latest: u32 = text.trim().parse()
        .with_context(|| format!("invalid release number: {:?}", text.trim()))?;

    if latest > current {
        Ok(Some(latest))
    } else {
        Ok(None)
    }
}

pub fn perform_update() -> anyhow::Result<()> {
    let repo_dir = std::env::current_dir()?;

    if !repo_dir.join(".git").exists() {
        anyhow::bail!("no .git directory found, cannot auto-update");
    }

    // Fetch the latest version before pulling so we can write it (spawns its own blocking client)
    let latest = {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .context("failed to build http client")?;
        let resp = client.get(RELEASE_URL)
            .send()
            .context("failed to fetch release info")?;
        let text = resp.text()?;
        text.trim().parse::<u32>()
            .with_context(|| format!("invalid release number: {:?}", text.trim()))?
    };

    let output = std::process::Command::new("git")
        .args(["pull", REMOTE_REPO])
        .current_dir(&repo_dir)
        .output()
        .context("failed to run git pull")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git pull failed: {}", stderr);
    }

    let output = std::process::Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&repo_dir)
        .output()
        .context("failed to run cargo build")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("cargo build failed: {}", stderr);
    }

    set_release(latest)?;

    Ok(())
}
