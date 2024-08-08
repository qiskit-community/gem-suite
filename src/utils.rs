// This code is part of Qiskit.
//
// (C) Copyright IBM 2024.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::io::Write;

use hashbrown::HashSet;

use itertools::Itertools;
use petgraph::stable_graph::StableUnGraph;
use crate::graph::*;


/// Convert potential bidirectional coupling map to undirected.
/// Coupling pairs (a, b) and (b, a) are merged into (a, b), where a < b.
pub(crate) fn to_undirected(
    connectivity: &Vec<(usize, usize)>
) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut undirected: Vec<(usize, usize)> = connectivity
        .iter()
        .map(|p| {
            if p.0 < p.1 {
                (p.0, p.1)
            } else {
                (p.1, p.0)
            }
        })
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
pub(crate) fn ungraph_to_dot<N: WriteDot, E: WriteDot>(
    graph: &StableUnGraph<N, E>,
) -> Vec<u8> {
    let mut buf = Vec::<u8>::new();
    writeln!(&mut buf, "graph {{").unwrap();
    writeln!(&mut buf, "node [fontname=\"Consolas\", fontsize=8.0, height=0.7];").unwrap();
    writeln!(&mut buf, "edge [fontname=\"Consolas\", fontsize=8.0, penwidth=2.5];").unwrap();
    for node in graph.node_weights() {
        writeln!(&mut buf, "{}", node.to_dot()).unwrap();
    }
    for edge in graph.edge_weights() {
        writeln!(&mut buf, "{}", edge.to_dot()).unwrap();
    }
    writeln!(&mut buf, "}}").unwrap();
    buf
}
