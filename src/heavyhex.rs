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

pub type DecodePyOut = (
    Option<HashMap<String, usize>>,
    Vec<f64>,
    Vec<f64>,
    (f64, f64),
    (f64, f64),
);

mod decoding;
mod graph_builder;
mod simple_cycle;
pub mod visualization;

use bitvec::prelude::*;
use decoding::{check_matrix_csc, decode_outcomes_fb, decode_outcomes_pm};
use graph_builder::*;
use itertools::Itertools;
use simple_cycle::heavyhex_cycle;

use std::str;

use hashbrown::{HashMap, HashSet};
use lazy_static::lazy_static;
use petgraph::stable_graph::StableUnGraph;

use pyo3::{
    prelude::*,
    types::{PyString, PyType},
};

use crate::graph::*;
use crate::utils::{to_undirected, ungraph_to_dot};

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
/// Attributes:
///     index (int): Qubit physical index.
///     role (str): Qubit role either in Site or Bond.
///     group (str): Two qubit gate parameterization grouping either in A or B.
///     neighbors (list[int]): Index of neighboring qubits.
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
    coordinate: Option<(isize, isize)>,
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
            if let Some(c) = self.coordinate {
                format!("({}, {})", c.0, c.1)
            } else {
                "None".to_string()
            },
            self.neighbors,
        )
    }
}

/// Plaquette dataclass to expose in Python domain.
///
/// Attributes:
///     index (int): Plaquette index.
///     qubits (list[int]): Physical index of component qubits.
///     neighbors (list[int]): Index of neighboring plaquettes.
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
            self.index, self.qubits, self.neighbors,
        )
    }
}

/// ScheduledGate dataclass to expose in Python domain.
///
/// Attributes:
///     index0 (int): First qubit where the entangling gate is applied to.
///     index1 (int): Second qubit where the entangling gate is applied to.
///     group (str):  Two qubit gate parameterization grouping either in A or B.
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
            self.index0, self.index1, self.group,
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
///
/// Args:
///     coupling_map (list[tuple[int, int]]): Coupling pairs which can be
///         either uni or bi-directional, e.g. [(0, 1), (1, 0), (1, 2), ...].
pub struct PyHeavyHexLattice {
    pub plaquette_qubits_map: std::collections::BTreeMap<PlaquetteIndex, Vec<QubitIndex>>,
    pub qubit_graph: StableUnGraph<QubitNode, QubitEdge>,
    pub plaquette_graph: StableUnGraph<PlaquetteNode, PlaquetteEdge>,
    pub decode_graph: StableUnGraph<DecodeNode, DecodeEdge>,
    pub bit_specifier: BitSpecifier,
}

#[pymethods]
impl PyHeavyHexLattice {
    /// Create new PyHeavyHexPlaquette from the device coupling map.
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
    /// Args:
    ///     plaquette_qubits_map (dict[int, list[int]]): Mapping from plaquette index to component physical qubits.
    ///     connectivity (list[tuple[int, int]]): Unidirectional coupling between physical qubits.
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
            PyString::new_bound(py, str::from_utf8(&buf)?).to_object(py),
        ))
    }

    /// Return dot script representing the plaquette lattice
    pub fn plaquette_graph_dot(&self, py: Python) -> PyResult<Option<PyObject>> {
        let buf = ungraph_to_dot(&self.plaquette_graph);
        Ok(Some(
            PyString::new_bound(py, str::from_utf8(&buf)?).to_object(py),
        ))
    }

    /// Return dot script representing the annotated qubit graph for decoding
    pub fn decode_graph_dot(&self, py: Python) -> PyResult<Option<PyObject>> {
        let buf = ungraph_to_dot(&self.decode_graph);
        Ok(Some(
            PyString::new_bound(py, str::from_utf8(&buf)?).to_object(py),
        ))
    }

    /// Return annotated qubit dataclasses in this lattice.
    pub fn qubits(&self) -> Vec<PyQubit> {
        let mut nodes: Vec<_> = self
            .qubit_graph
            .node_indices()
            .map(|n| {
                let neighbors: Vec<_> = self
                    .qubit_graph
                    .neighbors(n)
                    .map(|m| self.qubit_graph.node_weight(m).unwrap().index)
                    .collect();
                let weight = self.qubit_graph.node_weight(n).unwrap();
                PyQubit {
                    index: weight.index,
                    role: match weight.role {
                        Some(QubitRole::Bond) => "Bond".to_string(),
                        Some(QubitRole::Site) => "Site".to_string(),
                        None => "None".to_string(),
                    },
                    group: match weight.group {
                        Some(OpGroup::A) => "A".to_string(),
                        Some(OpGroup::B) => "B".to_string(),
                        None => "None".to_string(),
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
        let mut nodes: Vec<_> = self
            .plaquette_graph
            .node_indices()
            .map(|n| {
                let neighbors: Vec<_> = self
                    .plaquette_graph
                    .neighbors(n)
                    .map(|m| self.plaquette_graph.node_weight(m).unwrap().index)
                    .collect();
                let weight = self.plaquette_graph.node_weight(n).unwrap();
                PyPlaquette {
                    index: weight.index,
                    qubits: self.plaquette_qubits_map[&weight.index].to_owned(),
                    neighbors,
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
        if self
            .plaquette_qubits_map
            .keys()
            .all(|pi| includes.contains(pi))
        {
            // Nothing filtered out
            return self.clone();
        }
        let new_plaquettes = self
            .plaquette_qubits_map
            .iter()
            .filter_map(|item| {
                if includes.contains(item.0) {
                    Some((*item.0, item.1.clone()))
                } else {
                    None
                }
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        let connectivity = self
            .qubit_graph
            .edge_weights()
            .map(|e| (e.neighbor0, e.neighbor1))
            .collect_vec();
        PyHeavyHexLattice::with_plaquettes(new_plaquettes, connectivity)
    }

    /// Schedule entangling gates to build GEM circuit.
    pub fn build_gate_schedule(&self, index: usize) -> Vec<Vec<PyScheduledGate>> {
        let reverse_node_map = self
            .qubit_graph
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
                                    let nw0 = self
                                        .qubit_graph
                                        .node_weight(reverse_node_map[&ew.neighbor0])
                                        .unwrap();
                                    let nw1 = self
                                        .qubit_graph
                                        .node_weight(reverse_node_map[&ew.neighbor1])
                                        .unwrap();
                                    let opgroup = match (nw0.role, nw1.role) {
                                        (Some(QubitRole::Bond), Some(QubitRole::Site)) => nw1.group,
                                        (Some(QubitRole::Site), Some(QubitRole::Bond)) => nw0.group,
                                        _ => panic!("Qubit role configuration is invalid."),
                                    };
                                    let opgroup_str = match opgroup.unwrap() {
                                        OpGroup::A => "A".to_string(),
                                        OpGroup::B => "B".to_string(),
                                    };
                                    Some(PyScheduledGate {
                                        index0: nw0.index,
                                        index1: nw1.index,
                                        group: opgroup_str,
                                    })
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
    /// Args:
    ///     counts (dict[str, int]): Count dictionary keyed on measured bitstring in little endian format.
    ///     return_counts (bool): Set true to return decoded count dictionary.
    ///         The dict data is not used in the following analysis in Python domain
    ///         while data size is large and thus increases the overhead in the FFI boundary.
    ///         When this function is called from Rust, this overhead doesn't matter since
    ///         data is just moved.
    ///
    /// Returns:
    ///     tuple[dict[str, any] | None, list[float], list[float], tuple[float, float], tuple[float, float]]:
    ///         A tuple of an optional decoded count dictionary, plaquette and ZXZ bond observables,
    ///         and f and g values with uncertainty associated with decoded magnetization.
    pub fn decode_outcomes_fb(
        &self,
        counts: HashMap<String, usize>,
        return_counts: bool,
    ) -> DecodePyOut {
        let out = decode_outcomes_fb(self, &counts);
        if return_counts {
            (Some(out.0), out.1, out.2, out.3, out.4)
        } else {
            (None, out.1, out.2, out.3, out.4)
        }
    }

    /// Decode raw circuit outcome with plaquette lattice information
    /// to compute quantities associated with prepared state magnetization
    /// and other set of quantities associated with device error.
    ///
    /// Args:
    ///     solver (Callable): Call to the pymatching Matching decoder in the batch mode.
    ///     counts (dict[str, int]): Count dictionary keyed on measured bitstring in little endian format.
    ///     return_counts (bool): Set true to return decoded count dictionary.
    ///         The dict data is not used in the following analysis in Python domain
    ///         while data size is large and thus increases the overhead in the FFI boundary.
    ///         When this function is called from Rust, this overhead doesn't matter since
    ///         data is just moved.
    ///
    /// Returns:
    ///     tuple[dict[str, any] | None, list[float], list[float], tuple[float, float], tuple[float, float]]:
    ///         A tuple of an optional decoded count dictionary, plaquette and ZXZ bond observables,
    ///         and f and g values with uncertainty associated with decoded magnetization.
    pub fn decode_outcomes_pm(
        &self,
        py: Python,
        solver: PyObject,
        counts: HashMap<String, usize>,
        return_counts: bool,
    ) -> DecodePyOut {
        let out = decode_outcomes_pm(py, solver, self, &counts);
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
        check_matrix_csc(self)
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
            .flat_map(|qs| qs.to_owned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect_vec();
        plq_qubits.sort_unstable();

        let qubit_graph = build_qubit_graph(&plq_qubits, &connectivity, &plaquette_qubits_map);
        let plaquette_graph = build_plaquette_graph(&plaquette_qubits_map);
        let decode_graph =
            build_decode_graph(&qubit_graph, &plaquette_graph, &plaquette_qubits_map);
        let bit_specifier = BitSpecifier::new(&decode_graph, &plq_qubits, &plaquette_qubits_map);

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
/// Because bit mapping is computed repeateadly to process each count key,
/// this object provides significant speedup by precaching the mapping.
#[derive(Debug, Clone)]
pub struct BitSpecifier {
    pub bond_cidxs: Vec<QubitIndex>,
    pub site_cidxs: Vec<QubitIndex>,
    pub correlated_bits: Vec<(BitIndex, BitIndex, BitIndex)>,
    pub syndrome_bonds: Vec<Vec<BitIndex>>,
    pub n_bonds: usize,
    pub n_sites: usize,
}

impl BitSpecifier {
    pub fn new(
        decode_graph: &StableUnGraph<DecodeNode, DecodeEdge>,
        qubits: &[usize],
        plaquette_qubits_map: &std::collections::BTreeMap<PlaquetteIndex, Vec<QubitIndex>>,
    ) -> Self {
        let bond_qubits = decode_graph
            .edge_weights()
            .map(|ew| (ew.index, ew.bit_index.unwrap()))
            .collect::<HashMap<_, _>>();
        let site_qubits = decode_graph
            .node_weights()
            .map(|nw| (nw.index, nw.bit_index.unwrap()))
            .collect::<HashMap<_, _>>();
        let qubit_clbit_map = qubits
            .iter()
            .enumerate()
            .map(|(c, q)| (*q, c))
            .collect::<HashMap<_, _>>();
        let bond_cidxs = bond_qubits
            .iter()
            .map(|(qi, bi)| (bi, *qi))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .map(|qi| qubit_clbit_map.len() - 1 - qubit_clbit_map[&qi])
            .collect_vec();
        let site_cidxs = site_qubits
            .iter()
            .map(|(qi, bi)| (bi, *qi))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .map(|qi| qubit_clbit_map.len() - 1 - qubit_clbit_map[&qi])
            .collect_vec();
        let correlated_bits = decode_graph
            .edge_weights()
            .map(|ew| {
                (
                    ew.bit_index.unwrap(),
                    site_qubits[&ew.neighbor0],
                    site_qubits[&ew.neighbor1],
                )
            })
            .collect_vec();
        let syndrome_bonds = plaquette_qubits_map
            .values()
            .map(|sub_qubits| {
                sub_qubits
                    .iter()
                    .filter_map(|qi| bond_qubits.get(qi).copied())
                    .collect_vec()
            })
            .collect_vec();
        BitSpecifier {
            bond_cidxs,
            site_cidxs,
            correlated_bits,
            syndrome_bonds,
            n_bonds: bond_qubits.len(),
            n_sites: site_qubits.len(),
        }
    }

    pub fn to_bond_string(&self, meas_bits: &[char]) -> BitVec {
        self.bond_cidxs
            .iter()
            .map(|ci| match meas_bits[*ci] {
                '1' => true,
                '0' => false,
                _ => panic!("Measurement outcome is not bits."),
            })
            .collect::<BitVec>()
    }

    pub fn to_site_string(&self, meas_bits: &[char]) -> BitVec {
        self.site_cidxs
            .iter()
            .map(|ci| match meas_bits[*ci] {
                '1' => true,
                '0' => false,
                _ => panic!("Measurement outcome is not bits."),
            })
            .collect::<BitVec>()
    }

    pub fn calculate_syndrome(&self, bond_bits: &BitVec) -> BitVec {
        self.syndrome_bonds
            .iter()
            .map(|bis| {
                let sum =
                    bis.iter().fold(
                        0_usize,
                        |sum, bi| {
                            if bond_bits[*bi] {
                                sum + 1
                            } else {
                                sum
                            }
                        },
                    );
                sum % 2 == 1
            })
            .collect::<BitVec>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{EAGLE_CMAP, FALCON_CMAP};

    #[test]
    fn test_scheduling() {
        let coupling_map = FALCON_CMAP.lock().unwrap().to_owned();
        let plaquette_lattice = PyHeavyHexLattice::new(coupling_map);
        let gates = plaquette_lattice.build_gate_schedule(0);
        assert_eq!(
            gates[0],
            // Scheduling group E1 and E3
            vec![
                PyScheduledGate {
                    index0: 1,
                    index1: 4,
                    group: format!("A")
                },
                PyScheduledGate {
                    index0: 8,
                    index1: 11,
                    group: format!("A")
                },
                PyScheduledGate {
                    index0: 12,
                    index1: 15,
                    group: format!("A")
                },
                PyScheduledGate {
                    index0: 19,
                    index1: 22,
                    group: format!("A")
                },
                PyScheduledGate {
                    index0: 3,
                    index1: 5,
                    group: format!("B")
                },
                PyScheduledGate {
                    index0: 7,
                    index1: 10,
                    group: format!("B")
                },
                PyScheduledGate {
                    index0: 14,
                    index1: 16,
                    group: format!("B")
                },
                PyScheduledGate {
                    index0: 18,
                    index1: 21,
                    group: format!("B")
                },
            ]
        );
        assert_eq!(
            gates[1],
            // Scheduling group E5 and E2
            vec![
                PyScheduledGate {
                    index0: 1,
                    index1: 2,
                    group: format!("A")
                },
                PyScheduledGate {
                    index0: 12,
                    index1: 13,
                    group: format!("A")
                },
                PyScheduledGate {
                    index0: 23,
                    index1: 24,
                    group: format!("A")
                },
                PyScheduledGate {
                    index0: 4,
                    index1: 7,
                    group: format!("B")
                },
                PyScheduledGate {
                    index0: 11,
                    index1: 14,
                    group: format!("B")
                },
                PyScheduledGate {
                    index0: 15,
                    index1: 18,
                    group: format!("B")
                },
                PyScheduledGate {
                    index0: 22,
                    index1: 25,
                    group: format!("B")
                },
            ]
        );
        assert_eq!(
            gates[2],
            // Scheduling group E4 and E6
            vec![
                PyScheduledGate {
                    index0: 5,
                    index1: 8,
                    group: format!("A")
                },
                PyScheduledGate {
                    index0: 10,
                    index1: 12,
                    group: format!("A")
                },
                PyScheduledGate {
                    index0: 16,
                    index1: 19,
                    group: format!("A")
                },
                PyScheduledGate {
                    index0: 21,
                    index1: 23,
                    group: format!("A")
                },
                PyScheduledGate {
                    index0: 2,
                    index1: 3,
                    group: format!("B")
                },
                PyScheduledGate {
                    index0: 13,
                    index1: 14,
                    group: format!("B")
                },
                PyScheduledGate {
                    index0: 24,
                    index1: 25,
                    group: format!("B")
                },
            ]
        );
    }

    #[test]
    fn test_filter() {
        let coupling_map = EAGLE_CMAP.lock().unwrap().to_owned();
        let plaquette_lattice = PyHeavyHexLattice::new(coupling_map);
        let small_lattice = plaquette_lattice.filter(vec![1, 3]);
        let weights = small_lattice
            .qubit_graph
            .node_weights()
            .cloned()
            .collect_vec();
        assert_eq!(
            weights,
            vec![
                QubitNode {
                    index: 4,
                    role: Some(QubitRole::Site),
                    group: Some(OpGroup::A),
                    coordinate: Some((0, 0))
                },
                QubitNode {
                    index: 5,
                    role: Some(QubitRole::Bond),
                    group: None,
                    coordinate: Some((1, 0))
                },
                QubitNode {
                    index: 6,
                    role: Some(QubitRole::Site),
                    group: Some(OpGroup::B),
                    coordinate: Some((2, 0))
                },
                QubitNode {
                    index: 7,
                    role: Some(QubitRole::Bond),
                    group: None,
                    coordinate: Some((3, 0))
                },
                QubitNode {
                    index: 8,
                    role: Some(QubitRole::Site),
                    group: Some(OpGroup::A),
                    coordinate: Some((4, 0))
                },
                QubitNode {
                    index: 15,
                    role: Some(QubitRole::Bond),
                    group: None,
                    coordinate: Some((0, 1))
                },
                QubitNode {
                    index: 16,
                    role: Some(QubitRole::Bond),
                    group: None,
                    coordinate: Some((4, 1))
                },
                QubitNode {
                    index: 20,
                    role: Some(QubitRole::Site),
                    group: Some(OpGroup::A),
                    coordinate: Some((-2, 2))
                },
                QubitNode {
                    index: 21,
                    role: Some(QubitRole::Bond),
                    group: None,
                    coordinate: Some((-1, 2))
                },
                QubitNode {
                    index: 22,
                    role: Some(QubitRole::Site),
                    group: Some(OpGroup::B),
                    coordinate: Some((0, 2))
                },
                QubitNode {
                    index: 23,
                    role: Some(QubitRole::Bond),
                    group: None,
                    coordinate: Some((1, 2))
                },
                QubitNode {
                    index: 24,
                    role: Some(QubitRole::Site),
                    group: Some(OpGroup::A),
                    coordinate: Some((2, 2))
                },
                QubitNode {
                    index: 25,
                    role: Some(QubitRole::Bond),
                    group: None,
                    coordinate: Some((3, 2))
                },
                QubitNode {
                    index: 26,
                    role: Some(QubitRole::Site),
                    group: Some(OpGroup::B),
                    coordinate: Some((4, 2))
                },
                QubitNode {
                    index: 33,
                    role: Some(QubitRole::Bond),
                    group: None,
                    coordinate: Some((-2, 3))
                },
                QubitNode {
                    index: 34,
                    role: Some(QubitRole::Bond),
                    group: None,
                    coordinate: Some((2, 3))
                },
                QubitNode {
                    index: 39,
                    role: Some(QubitRole::Site),
                    group: Some(OpGroup::B),
                    coordinate: Some((-2, 4))
                },
                QubitNode {
                    index: 40,
                    role: Some(QubitRole::Bond),
                    group: None,
                    coordinate: Some((-1, 4))
                },
                QubitNode {
                    index: 41,
                    role: Some(QubitRole::Site),
                    group: Some(OpGroup::A),
                    coordinate: Some((0, 4))
                },
                QubitNode {
                    index: 42,
                    role: Some(QubitRole::Bond),
                    group: None,
                    coordinate: Some((1, 4))
                },
                QubitNode {
                    index: 43,
                    role: Some(QubitRole::Site),
                    group: Some(OpGroup::B),
                    coordinate: Some((2, 4))
                },
            ]
        );
    }
}
