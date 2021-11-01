use std::collections::HashMap;

use anyhow::{Context, Result};
use roogle_engine::Index;

pub fn make_index() -> Result<Index> {
    let mut own_path = std::env::current_dir().unwrap();
    own_path.push("./index/doc");

    let crates = std::fs::read_dir(own_path)
        .context("failed to read index files")?
        .map(|entry| {
            let entry = entry?;
            let json = std::fs::read_to_string(entry.path())
                .with_context(|| format!("failed to read `{:?}`", entry.file_name()))?;
            let krate = serde_json::from_str(&json)
                .with_context(|| format!("failed to deserialize `{:?}`", entry.file_name()))?;
            let file_name = entry
                .path()
                .with_extension("")
                .file_name()
                .with_context(|| format!("failed to get file name from `{:?}`", entry.path()))?
                .to_str()
                .context("failed to get `&str` from `&OsStr`")?
                .to_owned();
            Ok((file_name, krate))
        })
        .filter_map(|res: Result<_, anyhow::Error>| res.ok())
        .collect::<HashMap<_, _>>();
    Ok(Index { crates })
}
