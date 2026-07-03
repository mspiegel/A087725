//! Checkpointing for the value and policy networks (safetensors via `VarMap`).
//!
//! Each checkpoint writes round-numbered files plus a stable `*_latest`
//! copy (so `eval_ml24` can always point at the newest without knowing the
//! round), and appends a one-line TSV metrics record to `metrics.tsv`.

use std::io::Write;
use std::path::{Path, PathBuf};

use candle_core::{Error, Result};
use candle_nn::VarMap;

fn io_err(ctx: &str, e: std::io::Error) -> Error {
    Error::Msg(format!("{ctx}: {e}"))
}

pub fn value_path(dir: &Path, round: u32) -> PathBuf {
    dir.join(format!("value_r{round}.safetensors"))
}

pub fn value_latest_path(dir: &Path) -> PathBuf {
    dir.join("value_latest.safetensors")
}

pub fn policy_latest_path(dir: &Path) -> PathBuf {
    dir.join("policy_latest.safetensors")
}

/// Save both networks under `dir`, tagged by `round`, plus `*_latest` copies.
pub fn save(dir: &Path, round: u32, value: &VarMap, policy: &VarMap) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| io_err(&format!("mkdir {}", dir.display()), e))?;
    value.save(value_path(dir, round))?;
    value.save(value_latest_path(dir))?;
    policy.save(dir.join(format!("policy_r{round}.safetensors")))?;
    policy.save(dir.join("policy_latest.safetensors"))?;
    Ok(())
}

/// Append a tab-separated metrics line (created with a header on first write).
pub fn append_metrics(dir: &Path, header: &str, line: &str) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| io_err(&format!("mkdir {}", dir.display()), e))?;
    let path = dir.join("metrics.tsv");
    let need_header = !path.exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| io_err(&format!("open {}", path.display()), e))?;
    if need_header {
        writeln!(f, "{header}").map_err(|e| io_err("write header", e))?;
    }
    writeln!(f, "{line}").map_err(|e| io_err("write line", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puzzle24::ml::value_net::ValueNet;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    #[test]
    fn save_writes_files_and_latest_loads() {
        let dev = Device::Cpu;
        let dir = std::env::temp_dir().join(format!("ml24_ckpt_{}", std::process::id()));
        let vm_v = VarMap::new();
        let _net = ValueNet::new(VarBuilder::from_varmap(&vm_v, DType::F32, &dev), 16, 2).unwrap();
        let vm_p = VarMap::new();
        // Reuse a ValueNet shape for the "policy" slot — this test only checks I/O.
        let _p = ValueNet::new(VarBuilder::from_varmap(&vm_p, DType::F32, &dev), 16, 2).unwrap();

        save(&dir, 3, &vm_v, &vm_p).unwrap();
        assert!(value_path(&dir, 3).exists());
        assert!(value_latest_path(&dir).exists());

        // A fresh varmap+net can load the latest checkpoint.
        let mut vm2 = VarMap::new();
        let _n2 = ValueNet::new(VarBuilder::from_varmap(&vm2, DType::F32, &dev), 16, 2).unwrap();
        vm2.load(value_latest_path(&dir)).unwrap();

        append_metrics(&dir, "round\tx", "3\t1.5").unwrap();
        assert!(dir.join("metrics.tsv").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
