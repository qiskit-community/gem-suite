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

pub mod visualization;
mod graph_builder;
mod simple_cycle;
mod decoding;

use bitvec::vec::BitVec;
use graph_builder::*;
use decoding::{decode_outcomes_fb, check_matrix_csc};
use itertools::Itertools;
use simple_cycle::heavyhex_cycle;

use std::str;

use hashbrown::{HashMap, HashSet};
use petgraph::stable_graph::StableUnGraph;
use lazy_static::lazy_static;

use pyo3::{prelude::*, types::{PyString, PyType}};

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
#[derive(Debug, Clone)]
/// Plaquette representation of heavy hex lattice devices.
/// Graph node and edges are immediately annotated for GEM experiments.
/// Qubits are classified into either site or bond type,
/// and the site qubits are further classified into OpGroup A or B.
/// Edges (qubit coupling) are classified into one of 6 scheduling groups 
/// corresponding to different scheduling pattern of entangling instructions.
pub struct PyHeavyHexLattice {
    pub plaquette_qubits_map: std::collections::BTreeMap<PlaquetteIndex, Vec<QubitIndex>>,
    pub qubit_graph: StableUnGraph<QubitNode, QubitEdge>,
    pub plaquette_graph: StableUnGraph<PlaquetteNode, PlaquetteEdge>,
    pub decode_graph: StableUnGraph<DecodeNode, DecodeEdge>,
    pub bit_specifier: BitSpecifier,
}

#[pymethods]
impl PyHeavyHexLattice{

    /// Create new PyHeavyHexPlaquette from the device coupling map.
    /// 
    /// # Arguments
    /// * `coupling_map`: Coupling pairs, e.g. [(0, 1), (1, 0), (1, 2), ...],
    ///     which can be either uni or bi-directional.
    #[new]
    pub fn new(coupling_map: Vec<(usize, usize)>) -> Self {
        let (qubits, connectivity) = to_undirected(&coupling_map);
        let plaquette_qubits_map = heavyhex_cycle(&qubits, &connectivity)
            .into_iter()
            .enumerate()
            .collect::<std::collections::BTreeMap<_, _>>();
        PyHeavyHexLattice::with_plaquettes(plaquette_qubits_map, connectivity)
    }

    /// Create new PyHeavyHexPlaquette from already inspected topology data.
    /// 
    /// # Arguments
    /// * `plaquette_qubits_map`: Mapping from plaquette index to component physical qubits.
    /// * `connectivity`: Unidirectional coupling between physical qubits.
    #[classmethod]
    pub fn from_plaquettes(
        _: &Bound<'_, PyType>,
        plaquette_qubits_map: std::collections::BTreeMap<usize, Vec<usize>>,
        connectivity: Vec<(usize, usize)>,
    ) -> Self {
        PyHeavyHexLattice::with_plaquettes(plaquette_qubits_map, connectivity)
    }

    /// Return dot script representing the annotated qubit lattice 
    pub fn qubit_graph_dot(&self, py: Python) -> PyResult<Option<PyObject>> {
        let buf = ungraph_to_dot(&self.qubit_graph);
        Ok(Some(
            PyString::new_bound(py, str::from_utf8(&buf)?).to_object(py)
        ))
    }

    /// Return dot script representing the plaquette lattice 
    pub fn plaquette_graph_dot(&self, py: Python) -> PyResult<Option<PyObject>> {
        let buf = ungraph_to_dot(&self.plaquette_graph);
        Ok(Some(
            PyString::new_bound(py, str::from_utf8(&buf)?).to_object(py)
        ))
    }

    /// Return dot script representing the annotated qubit graph for decoding
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

    /// Return connectivity of the qubits in this lattice.
    pub fn connectivity(&self) -> Vec<(usize, usize)> {
        self.qubit_graph
            .edge_weights()
            .map(|ew| (ew.neighbor0, ew.neighbor1))
            .collect_vec()
    }

    /// Create new sublattice from subset of plaquettes.
    pub fn filter(&self, includes: Vec<usize>) -> Self {
        // TODO: Add validation for disconnected index.
        if self.plaquette_qubits_map.keys().all(|pi| includes.contains(pi)) {
            // Nothing filtered out
            return self.clone()
        }
        let new_plaquettes = self.plaquette_qubits_map
            .iter()
            .filter_map(|item| {
                if includes.contains(item.0) {
                    Some((*item.0, item.1.clone()))
                } else {
                    None
                }
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        let connectivity = self.qubit_graph
            .edge_weights()
            .map(|e| (e.neighbor0, e.neighbor1))
            .collect_vec();
        PyHeavyHexLattice::with_plaquettes(new_plaquettes, connectivity)
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
                            .collect_vec()
                    })
                    .collect_vec()
            })
            .collect_vec()
    }

    /// Decode raw circuit outcome with plaquette lattice information
    /// to compute quantities associated with prepared state magnetization 
    /// and other set of quantities associated with device error.
    /// 
    /// # Arguments
    /// * `lattice`: Plaquette lattice to provide lattice topology.
    /// * `counts`: Count dictionary keyed on measured bitstring in little endian format.
    /// * `return_counts`: Set true to return decoded count dictionary. 
    ///   The dict data is not used in the following analysis in Python domain 
    ///   while data size is large and thus increases the overhead in the FFI boundary.
    ///   When this function is called from Rust, this overhead doesn't matter since
    ///   data is just moved.
    /// 
    /// # Returns
    /// A tuple of decoded count dictionary, plaquette and ZXZ bond observables,
    /// and f and g values associated with decoded magnetization.
    pub fn decode_outcomes(
        &self, 
        counts: HashMap<String, usize>,
        return_counts: bool,
    ) -> (Option<HashMap<String, usize>>, Vec<f64>, Vec<f64>, (f64, f64), (f64, f64)) {
        let out = decode_outcomes_fb(&self, &counts);
        if return_counts {
            (Some(out.0), out.1, out.2, out.3, out.4)
        } else {
            (None, out.1, out.2, out.3, out.4)
        }
    }

    /// Generate check matrix H of this plaquette lattice in Scipy CSC matrix form.
    /// Index of element "1" is return in the first tuple,
    /// which is a tuple of row indices and column indices.
    /// The shape of the whole matrix is return in the second tuple.
    /// Number of row is the length of syndrome, and column is the length of variables (decoding bonds).
    pub fn check_matrix_csc(&self) -> ((Vec<usize>, Vec<usize>), (usize, usize)) {
        check_matrix_csc(&self)
    }
}

impl PyHeavyHexLattice {

    /// Create new plaquette lattice object for heavy hex topology.
    pub fn with_plaquettes(
        plaquette_qubits_map: std::collections::BTreeMap<usize, Vec<usize>>,
        connectivity: Vec<(usize, usize)>,
    ) -> Self {
        let mut plq_qubits: Vec<usize> = plaquette_qubits_map
            .values()
            .map(|qs| qs.to_owned())
            .flatten()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect_vec();
        plq_qubits.sort_unstable();

        let qubit_graph = build_qubit_graph(
            &plq_qubits, 
            &connectivity, 
            &plaquette_qubits_map,
        );
        let plaquette_graph = build_plaquette_graph(
            &plaquette_qubits_map,
        );
        let decode_graph = build_decode_graph(
            &qubit_graph, 
            &plaquette_graph,
            &plaquette_qubits_map,
        );
        let bit_specifier = BitSpecifier::new(&decode_graph, &plq_qubits);

        PyHeavyHexLattice {
            plaquette_qubits_map, 
            qubit_graph, 
            plaquette_graph,
            decode_graph,
            bit_specifier,
        }
    }

}


/// Helper object to manage mapping between qubit index and bit index.
#[derive(Debug, Clone)]
pub struct BitSpecifier {
    pub bond_qubits: Vec<QubitIndex>,
    pub site_qubits: Vec<QubitIndex>,
    pub qubit_clbit_map: HashMap<QubitIndex, BitIndex>,
    pub bond_correlation: HashMap<BitIndex, (BitIndex, BitIndex)>,
}

impl BitSpecifier {

    pub fn new(
        decode_graph: &StableUnGraph<DecodeNode, DecodeEdge>,
        qubits: &Vec<usize>,
    ) -> Self {
        let bond_qubits = decode_graph
            .edge_weights()
            .map(|ew| (ew.bit_index.unwrap(), ew.index))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .collect_vec();
        let site_qubits = decode_graph
            .node_weights()
            .map(|nw| (nw.bit_index.unwrap(), nw.index))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .collect_vec();
        let bond_correlation = decode_graph
            .edge_weights()
            .map(|ew| {
                let s0 = site_qubits.iter().position(|qi| *qi == ew.neighbor0);
                let s1 = site_qubits.iter().position(|qi| *qi == ew.neighbor1);
                (ew.bit_index.unwrap(), (s0.unwrap(), s1.unwrap()))
            })
            .collect::<HashMap<_, _>>();
        let qubit_clbit_map = qubits
            .iter()
            .enumerate()
            .map(|(c, q)| (*q, c))
            .collect::<HashMap<_, _>>();
        BitSpecifier {
            bond_qubits,
            site_qubits,
            qubit_clbit_map,
            bond_correlation,
        }
    }

    pub fn to_bond_string(&self, meas_bits: &Vec<char>) -> BitVec{
        self.bond_qubits
            .iter()
            .map(|qi| {
                let ci = self.qubit_clbit_map[qi];
                match meas_bits.get(meas_bits.len() - 1 - ci) {
                    Some('0') => false,
                    Some('1') => true,
                    _ => panic!("Measured outcome of qubit {} is not a bit.", qi),
                }
            })
            .collect::<BitVec>()
    }

    pub fn to_site_string(&self, meas_bits: &Vec<char>) -> BitVec{
        self.site_qubits
            .iter()
            .map(|qi| {
                let ci = self.qubit_clbit_map[qi];
                match meas_bits.get(meas_bits.len() - 1 - ci) {
                    Some('0') => false,
                    Some('1') => true,
                    _ => panic!("Measured outcome of qubit {} is not a bit.", qi),
                }
            })
            .collect::<BitVec>()
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
