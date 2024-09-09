// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

use std::io::Write;

use hashbrown::{HashMap, HashSet};

use crate::graph::*;
use itertools::Itertools;
use petgraph::stable_graph::StableUnGraph;

/// Convert potential bidirectional coupling map to undirected.
/// Coupling pairs (a, b) and (b, a) are merged into (a, b), where a < b.
pub(crate) fn to_undirected(connectivity: &[(usize, usize)]) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut undirected: Vec<(usize, usize)> = connectivity
        .iter()
        .map(|p| if p.0 < p.1 { (p.0, p.1) } else { (p.1, p.0) })
        .collect::<HashSet<_>>()
        .into_iter()
        .collect_vec();
    undirected.sort_unstable_by_key(|p| p.0);
    let mut unique_elms: Vec<_> = undirected
        .iter()
        .flat_map(|p| vec![p.0, p.1])
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    unique_elms.sort_unstable();
    (unique_elms, undirected)
}

/// Write dot script to feed the graphviz drawer for graph image generation.
pub(crate) fn ungraph_to_dot<N: WriteDot, E: WriteDot>(graph: &StableUnGraph<N, E>) -> Vec<u8> {
    let mut buf = Vec::<u8>::new();
    writeln!(&mut buf, "graph {{").unwrap();
    writeln!(
        &mut buf,
        "node [fontname=\"Consolas\", fontsize=8.0, height=0.7];"
    )
    .unwrap();
    writeln!(
        &mut buf,
        "edge [fontname=\"Consolas\", fontsize=8.0, penwidth=2.5];"
    )
    .unwrap();
    for node in graph.node_weights() {
        writeln!(&mut buf, "{}", node.to_dot()).unwrap();
    }
    for edge in graph.edge_weights() {
        writeln!(&mut buf, "{}", edge.to_dot()).unwrap();
    }
    writeln!(&mut buf, "}}").unwrap();
    buf
}

/// Decode magnetization of the circuit outcome.
pub(crate) fn decode_magnetization(counts: &HashMap<String, usize>) -> ((f64, f64), (f64, f64)) {
    let shots = counts.values().fold(0_usize, |sum, v| sum + *v);
    // Momentums
    let mut m1 = 0.0;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    let mut m8 = 0.0;
    let mut bitlen = None;
    for (bitstring, count_num) in counts.iter() {
        // Suming over the decoded Z expectation values of the individual qubit
        let mag = bitstring.chars().fold(0.0, |sum, bit| match bit {
            '0' => sum + 1.0,
            '1' => sum - 1.0,
            _ => panic!("Decoded bitstring is not binary value."),
        });
        let freq = *count_num as f64 / shots as f64;
        m1 += freq * mag;
        m2 += freq * mag.powf(2.0);
        m4 += freq * mag.powf(4.0);
        m8 += freq * mag.powf(8.0);
        if let Some(n) = bitlen {
            if n != bitstring.len() {
                panic!("Bit length is not uniform in this count dictionary.");
            }
        } else {
            bitlen = Some(bitstring.len());
        }
    }
    let n = bitlen.unwrap() as f64;
    let s = shots as f64;
    let f = 1.0 / n * (m2 - m1.powf(2.0));
    let g = 1.0 / n.powf(3.0) * (m4 - m2.powf(2.0));
    let fstd =
        1.0 / n * (m4 / s - (m2 - m1.powf(2.0)).powf(2.0) * (s - 3.0) / s / (s - 1.0)).powf(0.5);
    let gstd = 1.0 / n.powf(3.0) * (m8 / s - fstd.powf(4.0) * (s - 3.0) / s / (s - 1.0)).powf(0.5);
    ((f, fstd), (g, gstd))
}
