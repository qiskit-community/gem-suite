// This code is part of Qiskit.
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

// Note::
//
// Johnson's algorithm for simple cycle search is partly ported from the rustworkx-core project at 
// 
//     https://github.com/Qiskit/rustworkx/blob/main/src/connectivity/johnson_simple_cycles.rs
//
// with modification for simplification and specialization for HHL plaquette search.
// This code doesn't implement Python iterator. 
// Instead, it returns all plaquettes at a single function call.

mod graph_builder;
mod simple_cycle;

use graph_builder::*;
use simple_cycle::heavyhex_cycle;

use core::num;
use std::str;

use hashbrown::{HashMap, HashSet};
use petgraph::stable_graph::StableUnGraph;
use lazy_static::lazy_static;

use pyo3::{prelude::*, types::PyString};

use crate::graph::*;
use crate::utils::{ungraph_to_dot, to_undirected};


lazy_static! {
    // Each schedule contains three elements since 
    // GEM circuit is constant depth of 3, regardless of qubit number.
    // For HHL, we can create 12 different scheduling patterns,
    // which may vary the outcome due to the impact of crosstalk.
    static ref GATE_ORDER: Vec<Vec<Vec<SchedulingGroup>>> = vec![
        vec![
            vec![SchedulingGroup::E1, SchedulingGroup::E3],
            vec![SchedulingGroup::E5, SchedulingGroup::E2],
            vec![SchedulingGroup::E4, SchedulingGroup::E6],
        ],
        vec![
            vec![SchedulingGroup::E1, SchedulingGroup::E3],
            vec![SchedulingGroup::E4, SchedulingGroup::E6],
            vec![SchedulingGroup::E5, SchedulingGroup::E2],
        ],
        vec![    
            vec![SchedulingGroup::E1, SchedulingGroup::E6],
            vec![SchedulingGroup::E5, SchedulingGroup::E3],
            vec![SchedulingGroup::E4, SchedulingGroup::E2],
        ],
        vec![    
            vec![SchedulingGroup::E1, SchedulingGroup::E6],
            vec![SchedulingGroup::E4, SchedulingGroup::E2],
            vec![SchedulingGroup::E5, SchedulingGroup::E3],
        ],
        vec![
            vec![SchedulingGroup::E5, SchedulingGroup::E3],
            vec![SchedulingGroup::E1, SchedulingGroup::E6],
            vec![SchedulingGroup::E4, SchedulingGroup::E2],
        ],
        vec![    
            vec![SchedulingGroup::E5, SchedulingGroup::E3],
            vec![SchedulingGroup::E4, SchedulingGroup::E2],
            vec![SchedulingGroup::E1, SchedulingGroup::E6],
        ],
        vec![
            vec![SchedulingGroup::E5, SchedulingGroup::E2],
            vec![SchedulingGroup::E1, SchedulingGroup::E3],
            vec![SchedulingGroup::E4, SchedulingGroup::E6],
        ],
        vec![    
            vec![SchedulingGroup::E5, SchedulingGroup::E2],
            vec![SchedulingGroup::E4, SchedulingGroup::E6],
            vec![SchedulingGroup::E1, SchedulingGroup::E3],
        ],
        vec![    
            vec![SchedulingGroup::E4, SchedulingGroup::E6],
            vec![SchedulingGroup::E1, SchedulingGroup::E3],
            vec![SchedulingGroup::E5, SchedulingGroup::E2],
        ],
        vec![   
            vec![SchedulingGroup::E4, SchedulingGroup::E6],
            vec![SchedulingGroup::E5, SchedulingGroup::E2],
            vec![SchedulingGroup::E1, SchedulingGroup::E3],
        ],
        vec![
            vec![SchedulingGroup::E4, SchedulingGroup::E2],
            vec![SchedulingGroup::E1, SchedulingGroup::E6],
            vec![SchedulingGroup::E5, SchedulingGroup::E3],
        ],
        vec![
            vec![SchedulingGroup::E4, SchedulingGroup::E2],
            vec![SchedulingGroup::E5, SchedulingGroup::E3],
            vec![SchedulingGroup::E1, SchedulingGroup::E6],
        ],
    ];
}


/// Annotated qubit dataclass to expose in Python domain.
/// 
/// # Attributes
/// 
/// * `index`: Qubit physical index.
/// * `role`: Qubit role in [Site, Bond].
/// * `group`: Two qubit gate parameterization grouping in [A, B].
/// * `neighbors`: Index of neighboring qubits.
#[pyclass]
#[derive(Debug, PartialEq)]
pub struct PyQubit {
    #[pyo3(get)]
    index: QubitIndex,
    #[pyo3(get)]
    role: String,
    #[pyo3(get)]
    group: String,
    #[pyo3(get)]
    coordinate: Option<(usize, usize)>,
    #[pyo3(get)]
    neighbors: Vec<QubitIndex>,
}

#[pymethods]
impl PyQubit {
    pub fn __repr__(&self) -> String {
        format!(
            "PyQubit(index={}, role=\"{}\", group=\"{}\", coordinate={}, neighbors={:?})",
            self.index,
            self.role,
            self.group,
            if let Some(c) = self.coordinate {format!("({}, {})", c.0, c.1)} else {format!("None")},
            self.neighbors,
        )
    }
}


/// Plaquette dataclass to expose in Python domain.
/// 
/// # Attributes
/// 
/// * `index`: Plaquette index.
/// * `qubits`: Physical index of component qubits.
/// * `neighbors`: Index of neighboring plaquettes.
#[pyclass]
#[derive(Debug, PartialEq)]
pub struct PyPlaquette {
    #[pyo3(get)]
    index: PlaquetteIndex,
    #[pyo3(get)]
    qubits: Vec<QubitIndex>,
    #[pyo3(get)]
    neighbors: Vec<PlaquetteIndex>,
}

#[pymethods]
impl PyPlaquette {
    pub fn __repr__(&self) -> String {
        format!(
            "PyPlaquette(index={}, qubits={:?}, neighbors={:?})",
            self.index,
            self.qubits,
            self.neighbors,
        )
    }
}


/// ScheduledGate dataclass to expose in Python domain.
/// 
/// # Attributes
/// 
/// * `index0`: First qubit where the entangling gate is applied to.
/// * `index1`: Second qubit where the entangling gate is applied to.
/// * `group`: Operation group in either A or B.
#[pyclass]
#[derive(Debug, PartialEq)]
pub struct PyScheduledGate {
    #[pyo3(get)]
    index0: QubitIndex,
    #[pyo3(get)]
    index1: QubitIndex,
    #[pyo3(get)]
    group: String,
}

#[pymethods]
impl PyScheduledGate {
    pub fn __repr__(&self) -> String {
        format!(
            "PyGate(index0={}, index1={:?}, group={:?})",
            self.index0,
            self.index1,
            self.group,
        )
    }
}


#[pyclass]
/// Plaquette representation of heavy hex lattice devices.
/// Graph node and edges are immediately annotated for GEM experiments.
/// Qubits are classified into either site or bond type,
/// and the site qubits are further classified into OpGroup A or B.
/// Edges (qubit coupling) are classified into one of 6 scheduling groups 
/// corresponding to different scheduling pattern of entangling instructions.
pub struct PyHeavyHexLattice {
    pub plaquette_qubits_map: HashMap<PlaquetteIndex, Vec<QubitIndex>>,
    pub qubit_graph: StableUnGraph<QubitNode, QubitEdge>,
    pub plaquette_graph: StableUnGraph<PlaquetteNode, PlaquetteEdge>,
    pub decode_graph: StableUnGraph<DecodeNode, DecodeEdge>,
}

#[pymethods]
impl PyHeavyHexLattice{

    /// Create new PyHeavyHexPlaquette object from the device coupling map.
    /// 
    /// # Arguments
    /// 
    /// * `coupling_map`: Coupling pairs, e.g. [(0, 1), (1, 0), (1, 2), ...],
    ///     which can be either uni or bi-directional.
    #[new]
    pub fn new(coupling_map: Vec<(usize, usize)>) -> Self {
        let (qubits, connectivity) = to_undirected(&coupling_map);
        let plaquettes = heavyhex_cycle(&qubits, &connectivity)
            .into_iter()
            .enumerate()
            .collect::<HashMap<_, _>>();
        PyHeavyHexLattice::from_plaquettes(plaquettes, connectivity)
    }

    /// Return dot script representing the annotated qubit lattice 
    /// to create image with the graphviz drawer.
    pub fn qubit_graph_dot(&self, py: Python) -> PyResult<Option<PyObject>> {
        let buf = ungraph_to_dot(&self.qubit_graph);
        Ok(Some(
            PyString::new_bound(py, str::from_utf8(&buf)?).to_object(py)
        ))
    }

    /// Return dot script representing the plaquette lattice 
    /// to create image with the graphviz drawer.
    pub fn plaquette_graph_dot(&self, py: Python) -> PyResult<Option<PyObject>> {
        let buf = ungraph_to_dot(&self.plaquette_graph);
        Ok(Some(
            PyString::new_bound(py, str::from_utf8(&buf)?).to_object(py)
        ))
    }

    /// Return dot script representing the annotated qubit graph for decoding
    /// to create image with the graphviz drawer.
    pub fn decode_graph_dot(&self, py: Python) -> PyResult<Option<PyObject>> {
        let buf = ungraph_to_dot(&self.decode_graph);
        Ok(Some(
            PyString::new_bound(py, str::from_utf8(&buf)?).to_object(py)
        ))
    }

    /// Return annotated qubit dataclasses in this lattice.
    pub fn qubits(&self) -> Vec<PyQubit> {
        let mut nodes: Vec<_> = self.qubit_graph
            .node_indices()
            .map(|n| {
                let neighbors: Vec<_> = self.qubit_graph
                    .neighbors(n)
                    .map(|m| self.qubit_graph.node_weight(m).unwrap().index)
                    .collect();
                let weight = self.qubit_graph.node_weight(n).unwrap();
                PyQubit {
                    index: weight.index,
                    role: match weight.role {
                        Some(QubitRole::Bond) => format!("Bond"),
                        Some(QubitRole::Site) => format!("Site"),
                        None => format!("None"),
                    },
                    group: match weight.group {
                        Some(OpGroup::A) => format!("A"),
                        Some(OpGroup::B) => format!("B"),
                        None => format!("None"),
                    },
                    coordinate: weight.coordinate,
                    neighbors,
                }
            })
            .collect();
        nodes.sort_unstable_by_key(|n| n.index);
        nodes
    }

    /// Return plaquette dataclasses in this lattice.
    pub fn plaquettes(&self) -> Vec<PyPlaquette> {
        let mut nodes: Vec<_> = self.plaquette_graph
            .node_indices()
            .map(|n| {
                let neighbors: Vec<_> = self.plaquette_graph
                    .neighbors(n)
                    .map(|m| self.plaquette_graph.node_weight(m).unwrap().index)
                    .collect();
                let weight = self.plaquette_graph.node_weight(n).unwrap();
                PyPlaquette {
                    index: weight.index,
                    qubits: self.plaquette_qubits_map[&weight.index].to_owned(),
                    neighbors: neighbors,
                }
            })
            .collect();
        nodes.sort_unstable_by_key(|n| n.index);
        nodes
    }

    /// Create new sublattice.
    pub fn filter(&self, includes: Vec<usize>) -> Self {
        // TODO: Add validation for disconnected index.
        let new_plaquettes = self.plaquette_qubits_map
            .iter()
            .filter_map(|item| {
                if includes.contains(item.0) {
                    Some((*item.0, item.1.clone()))
                } else {
                    None
                }
            })
            .collect::<HashMap<_, _>>();
        let connectivity = self.qubit_graph
            .edge_weights()
            .map(|e| (e.neighbor0, e.neighbor1))
            .collect::<Vec<_>>();
        PyHeavyHexLattice::from_plaquettes(new_plaquettes, connectivity)
    }

    /// Schedule entangling gates to build GEM circuit.
    pub fn build_gate_schedule(&self, index: usize) -> Vec<Vec<PyScheduledGate>> {        
        let reverse_node_map = self.qubit_graph
            .node_indices()
            .map(|n| (self.qubit_graph.node_weight(n).unwrap().index, n))
            .collect::<HashMap<_, _>>();
        GATE_ORDER[index]
            .iter()
            .map(|siml_group| {
                siml_group
                    .iter()
                    .flat_map(|group| {
                        self.qubit_graph
                            .edge_weights()
                            .filter_map(|ew| {
                                if ew.group == Some(*group) {
                                    let nw0 = self.qubit_graph.node_weight(reverse_node_map[&ew.neighbor0]).unwrap();
                                    let nw1 = self.qubit_graph.node_weight(reverse_node_map[&ew.neighbor1]).unwrap();
                                    let opgroup = match (nw0.role, nw1.role) {
                                        (Some(QubitRole::Bond), Some(QubitRole::Site)) => nw1.group,
                                        (Some(QubitRole::Site), Some(QubitRole::Bond)) => nw0.group,
                                        _ => panic!("Qubit role configuration is invalid.")
                                    };
                                    let opgroup_str = match  opgroup.unwrap() {
                                        OpGroup::A => format!("A"),
                                        OpGroup::B => format!("B"),                                        
                                    };
                                    Some(PyScheduledGate{index0: nw0.index, index1: nw1.index, group: opgroup_str})
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    /// Generate check matrix H of this plaquette lattice.
    /// Matrix is flattened and returned with dimension,
    /// where the first dimension is the size of syndrome (plaquettes)
    /// and the second dimension is the size of error mechanism (bond qubits).
    pub fn check_matrix(&self) -> (Vec<bool>, (usize, usize)) {
        let num_syndrome = self.plaquette_graph.node_count();
        let num_bonds = self.decode_graph.edge_count();
        let size = num_syndrome * num_bonds;     
        let mut hmat = vec![false; size];
        let mut plaquettes = self.plaquette_graph
            .node_weights()
            .map(|pw| pw.index)
            .collect::<Vec<_>>();
        plaquettes.sort_unstable();
        for (i, pi) in plaquettes.into_iter().enumerate() {
            let sub_qubits = &self.plaquette_qubits_map[&pi];
            self.decode_graph.edge_weights().for_each(|ew| {
                if sub_qubits.contains(&ew.index) {
                    let idx = i * num_bonds + ew.bit_index.unwrap();
                    hmat[idx] = true;
                }
            });
        }
        (hmat, (num_syndrome, num_bonds))
    }

    /// Return a list of bond bit index to be decoded.
    pub fn decoding_bonds(&self) -> Vec<usize> {
        let mut idxs = self.decode_graph
            .edge_weights()
            .filter_map(|ew| {
                if ew.is_decoding_edge {
                    Some(ew.bit_index.unwrap())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        idxs.sort_unstable();
        idxs
    }
}

impl PyHeavyHexLattice {

    pub fn from_plaquettes(
        plaquettes: HashMap<usize, Vec<usize>>,
        connectivity: Vec<(usize, usize)>,
    ) -> Self {
        let mut plq_qubits: Vec<usize> = plaquettes
            .values()
            .map(|qs| qs.to_owned())
            .flatten()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        plq_qubits.sort_unstable();

        let qubit_graph = build_qubit_graph(&plq_qubits, &connectivity, &plaquettes);
        let plaquette_graph = build_plaquette_graph(&plaquettes);
        let decode_graph = build_decode_graph(&qubit_graph, &plaquette_graph, &plaquettes);
        
        PyHeavyHexLattice {
            plaquette_qubits_map: plaquettes, 
            qubit_graph, 
            plaquette_graph,
            decode_graph,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::FALCON_CMAP;

    #[test]
    fn test_scheduling() {
        let coupling_map = FALCON_CMAP.lock().unwrap().to_owned();
        let plaquette_lattice = PyHeavyHexLattice::new(coupling_map);
        let gates = plaquette_lattice.build_gate_schedule(0);
        assert_eq!(
            gates[0],
            // Scheduling group E1 and E3
            vec![
                PyScheduledGate {index0: 1, index1: 4, group: format!("A")},
                PyScheduledGate {index0: 8, index1: 11, group: format!("A")},
                PyScheduledGate {index0: 12, index1: 15, group: format!("A")},
                PyScheduledGate {index0: 19, index1: 22, group: format!("A")},
                PyScheduledGate {index0: 3, index1: 5, group: format!("B")},
                PyScheduledGate {index0: 7, index1: 10, group: format!("B")},
                PyScheduledGate {index0: 14, index1: 16, group: format!("B")},
                PyScheduledGate {index0: 18, index1: 21, group: format!("B")},
            ]
        );
        assert_eq!(
            gates[1],
            // Scheduling group E5 and E2
            vec![
                PyScheduledGate {index0: 1, index1: 2, group: format!("A")},
                PyScheduledGate {index0: 12, index1: 13, group: format!("A")},
                PyScheduledGate {index0: 23, index1: 24, group: format!("A")},
                PyScheduledGate {index0: 4, index1: 7, group: format!("B")},
                PyScheduledGate {index0: 11, index1: 14, group: format!("B")},
                PyScheduledGate {index0: 15, index1: 18, group: format!("B")},
                PyScheduledGate {index0: 22, index1: 25, group: format!("B")},
            ]
        );
        assert_eq!(
            gates[2],
            // Scheduling group E4 and E6
            vec![
                PyScheduledGate {index0: 5, index1: 8, group: format!("A")},
                PyScheduledGate {index0: 10, index1: 12, group: format!("A")},
                PyScheduledGate {index0: 16, index1: 19, group: format!("A")},
                PyScheduledGate {index0: 21, index1: 23, group: format!("A")},
                PyScheduledGate {index0: 2, index1: 3, group: format!("B")},
                PyScheduledGate {index0: 13, index1: 14, group: format!("B")},
                PyScheduledGate {index0: 24, index1: 25, group: format!("B")},
            ]
        );
    }
}
