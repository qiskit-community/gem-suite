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

use std::str;

use hashbrown::HashMap;
use petgraph::stable_graph::StableUnGraph;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use pyo3::{prelude::*, types::PyString};

use super::graph_builder::build_plaquette_graph;
use crate::graph::*;
use crate::utils::ungraph_to_dot;


#[derive(Debug, PartialEq, Clone, Copy)]
pub struct NoisyPlaquetteNode {
    pub index: PlaquetteIndex,
    pub noise: f64,
}

impl WriteDot for NoisyPlaquetteNode {
    fn to_dot(&self) -> String {
        let normalized = self.noise.clamp(0.0, 1.0);
        let rgb_code = format!("#{:02x}{:02x}{:02x}", 0_u8, 0_u8, (255.0 * normalized) as u8);
        let options = vec![
            format!("label=\"P{}\\n({:.3})\"", self.index, self.noise),
            format!("shape=hexagon"),
            format!("width=0.81"),
            format!("fontcolor=ghostwhite"),
            format!("fillcolor=\"{}\"", rgb_code),
            format!("style=filled"),
        ];
        format!("{} [{}];", self.index, options.join(", "))
    }
}


/// Draw plaquette graph with noise input.
/// 
/// Args:
///     plaquette_qubits_map (dict[int, list[int]]): A mapping of plaquette index to including qubits.
///     noise_map: (dict[int, float]): A mapping of plaquette index to noise.
///         The noise value should stay in the range [0.0, 1.0].
/// 
/// Returns:
///     str: Dot script of the noisy plaquette coupling graph. 
///         Noise intensity is shown in the graph nodes as filled colors.
#[pyfunction]
pub fn visualize_plaquette_with_noise(
    py: Python,
    plaquette_qubits_map: std::collections::BTreeMap<PlaquetteIndex, Vec<QubitIndex>>,
    noise_map: HashMap<usize, f64>,
) -> PyResult<Option<PyObject>> {
    let plaquette_graph = build_plaquette_graph(&plaquette_qubits_map);
    let mut noisy_graph = StableUnGraph::<NoisyPlaquetteNode, PlaquetteEdge>::with_capacity(
        plaquette_graph.node_count(), 
        plaquette_graph.edge_count(),
    );
    for nw in plaquette_graph.node_weights() {
        let plaquette_noise = noise_map.get(&nw.index).unwrap_or(&0.0);
        noisy_graph.add_node(NoisyPlaquetteNode { index: nw.index, noise: *plaquette_noise });
    }
    for eref in plaquette_graph.edge_references() {
        noisy_graph.add_edge(eref.source(), eref.target(), *eref.weight());
    }
    let buf = ungraph_to_dot(&noisy_graph);
    Ok(Some(
        PyString::new_bound(py, str::from_utf8(&buf)?).to_object(py)
    ))
}
