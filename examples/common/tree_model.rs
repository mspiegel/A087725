//! Minimal evaluator for a dumped sklearn HistGradientBoostingClassifier
//! (binary). Format produced by dump_model.py:
//!   baseline <f64>
//!   ntrees <N>
//!   nfeat <F>
//!   tree <n_nodes>
//!   <is_leaf> <feature_idx> <num_threshold> <left> <right> <value>   (per node)
//!   ...
//! Split convention matches sklearn: go LEFT iff x[feature_idx] <= num_threshold.

use std::fs;
use std::path::Path;

struct Node {
    is_leaf: bool,
    feat: u32,
    thr: f64,
    left: u32,
    right: u32,
    val: f64,
}
struct Tree {
    nodes: Vec<Node>,
}

pub struct TreeModel {
    baseline: f64,
    trees: Vec<Tree>,
    pub nfeat: usize,
}

impl TreeModel {
    pub fn load(path: &Path) -> std::io::Result<TreeModel> {
        let text = fs::read_to_string(path)?;
        let mut lines = text.lines();
        let mut baseline = 0.0;
        let mut nfeat = 0usize;
        let mut trees: Vec<Tree> = Vec::new();
        let mut cur: Option<Tree> = None;
        let mut remaining = 0usize;
        for line in &mut lines {
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.is_empty() {
                continue;
            }
            match toks[0] {
                "baseline" => baseline = toks[1].parse().unwrap(),
                "ntrees" => {}
                "nfeat" => nfeat = toks[1].parse().unwrap(),
                "tree" => {
                    if let Some(t) = cur.take() {
                        trees.push(t);
                    }
                    remaining = toks[1].parse().unwrap();
                    cur = Some(Tree {
                        nodes: Vec::with_capacity(remaining),
                    });
                }
                _ => {
                    // node line
                    let n = Node {
                        is_leaf: toks[0] == "1",
                        feat: toks[1].parse().unwrap(),
                        thr: toks[2].parse().unwrap(),
                        left: toks[3].parse().unwrap(),
                        right: toks[4].parse().unwrap(),
                        val: toks[5].parse().unwrap(),
                    };
                    cur.as_mut().unwrap().nodes.push(n);
                    remaining -= 1;
                }
            }
        }
        if let Some(t) = cur.take() {
            trees.push(t);
        }
        Ok(TreeModel {
            baseline,
            trees,
            nfeat,
        })
    }

    /// Raw margin (sum of tree leaf values + baseline).
    #[inline]
    pub fn raw(&self, x: &[f64]) -> f64 {
        let mut acc = self.baseline;
        for t in &self.trees {
            let mut i = 0u32;
            loop {
                let nd = &t.nodes[i as usize];
                if nd.is_leaf {
                    acc += nd.val;
                    break;
                }
                i = if x[nd.feat as usize] <= nd.thr {
                    nd.left
                } else {
                    nd.right
                };
            }
        }
        acc
    }

    /// P(class 1) = sigmoid(raw).
    #[inline]
    pub fn score(&self, x: &[f64]) -> f64 {
        1.0 / (1.0 + (-self.raw(x)).exp())
    }
}
