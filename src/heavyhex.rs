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

use graph_builder::*;
use itertools::Itertools;
use simple_cycle::heavyhex_cycle;

use std::str;

use bitvec::prelude::*;
use hashbrown::{HashMap, HashSet};
use petgraph::stable_graph::StableUnGraph;
use fusion_blossom::mwpm_solver::{PrimalDualSolver, SolverSerial};
use fusion_blossom::util::{SolverInitializer, SyndromePattern};
use lazy_static::lazy_static;

use pyo3::{prelude::*, types::{PyString, PyType}};

use crate::graph::*;
use crate::utils::{ungraph_to_dot, to_undirected, decode_magnetization};


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
    pub plaquette_qubits_map: HashMap<PlaquetteIndex, Vec<QubitIndex>>,
    pub qubit_graph: StableUnGraph<QubitNode, QubitEdge>,
    pub plaquette_graph: StableUnGraph<PlaquetteNode, PlaquetteEdge>,
    pub decode_graph: StableUnGraph<DecodeNode, DecodeEdge>,
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
            .collect::<HashMap<_, _>>();
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
        plaquette_qubits_map: HashMap<usize, Vec<usize>>,
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
            .collect::<HashMap<_, _>>();
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

    /// Generate check matrix H of this plaquette lattice.
    /// Matrix is flattened and returned with dimension,
    /// where the first dimension is the size of parity check (plaquettes)
    /// and the second dimension is the size of variable (bond qubits).
    pub fn check_matrix(&self) -> (Vec<bool>, (usize, usize)) {
        let num_syndrome = self.plaquette_graph.node_count();
        let num_bonds = self.decode_graph.edge_count();
        let size = num_syndrome * num_bonds;     
        let mut hmat = vec![false; size];
        let mut plaquettes = self.plaquette_graph
            .node_weights()
            .map(|pw| pw.index)
            .collect_vec();
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

    /// Decode raw circuit outcome with plaquette lattice information
    /// to compute quantities associated with prepared state magnetization 
    /// and other set of quantities associated with device error.
    /// 
    /// # Arguments
    /// * `counts`: Count dictionary keyed on measured bitstring in little endian format.
    /// * `return_counts`: Set true to return decoded count dictionary. 
    ///   The dict data is not used in the following analysis in Python domain 
    ///   while data size is large and thus increases the overhead in the FFI boundary.
    ///   When this function is called from Rust, this overhead doesn't matter since
    ///   data is just moved.
    /// 
    /// # Returns
    /// A tuple of decoded dictionary, plaquette and ZXZ bond observables,
    /// and f and g values associated with decoded magnetization.
    pub fn decode_outcomes(
        &self,
        counts: HashMap<String, usize>,
        return_counts: bool,
    ) -> (Option<HashMap<String, usize>>, Vec<f64>, Vec<f64>, (f64, f64), (f64, f64)) {
        let mut solver = self.build_mwpm_solver();
        let mut decoding_bonds = self.decode_graph
            .edge_weights()
            .filter_map(|ew| {
                if ew.is_decode_variable {
                    Some(ew.bit_index)
                } else {
                    None
                }
            })
            .collect_vec();
        decoding_bonds.sort_unstable();
        let bond_qubits = self.decode_graph
            .edge_weights()
            .map(|ew| (ew.bit_index.unwrap(), ew.index))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .collect_vec();
        let site_qubits = self.decode_graph
            .node_weights()
            .map(|nw| (nw.bit_index.unwrap(), nw.index))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .collect_vec();
        let qubit_clbit_map = self.qubits()
            .into_iter()
            .enumerate()
            .map(|(ci, qi)| (qi.index, ci))
            .collect::<HashMap<_, _>>();
        let syndrome_qubits = self.plaquette_qubits_map
            .iter()
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_values()
            .collect_vec();
        let correlation_indices = self.decode_graph
            .edge_weights()
            .map(|ew| {
                let s0 = site_qubits.iter().position(|qi| *qi == ew.neighbor0);
                let s1 = site_qubits.iter().position(|qi| *qi == ew.neighbor1);
                (ew.bit_index.unwrap(), s0.unwrap(), s1.unwrap())
            })
            .collect_vec();
        let snake_line = traverse_snake(&self.decode_graph);
        let n_bonds = bond_qubits.len();
        let n_sites = site_qubits.len();
        let n_syndrome = self.plaquette_qubits_map.len();
        let mut decoded_counts = HashMap::<String, usize>::with_capacity(counts.len());
        let mut syndrome_sum = vec![0_usize; n_syndrome];
        let mut bond_correlation_sum = vec![0_isize; n_bonds];
        let mut total_shots = 0_usize;
        for (key, count_num) in counts.iter() {
            solver.clear();
            let bitstring = key.chars().collect_vec();
            let get_bit = |qi: &QubitIndex| -> bool {
                let ci = if let Some(index) = qubit_clbit_map.get(qi) {
                    index
                } else {
                    panic!("Qubit {} doesn't exist in the measured bitstring.", qi);
                };
                match bitstring.get(bitstring.len() - 1 - *ci) {
                    Some('0') => false,
                    Some('1') => true,
                    _ => panic!("Measured outcome of qubit {} is not a bit.", qi),
                }
            };
            let site_bits = site_qubits
                .iter()
                .map(|qi| get_bit(qi))
                .collect::<BitVec>();
            let bond_bits = bond_qubits
                .iter()
                .map(|qi| get_bit(qi))
                .collect::<BitVec>();
            // Compute bond correlation
            for idxs in correlation_indices.iter() {
                if bond_bits[idxs.0] ^ site_bits[idxs.1] ^ site_bits[idxs.2] {
                    bond_correlation_sum[idxs.0] += *count_num as isize;
                } else {
                    bond_correlation_sum[idxs.0] -= *count_num as isize;
                }
            }
            // Compute frustration of plaquette. i.e. syndromes.
            let syndrome = syndrome_qubits
                .iter()
                .map(|sub_qubits| {
                    // Compute total number of '1' in the plaquette bonds.
                    let sum = bond_qubits
                        .iter()
                        .zip(bond_bits.iter())
                        .fold(0_usize, |sum, (qi, bit)| {
                            if sub_qubits.contains(qi) & (bit == true) {
                                sum + 1
                            } else {
                                sum
                            }
                        });
                    // Compute parity; even or odd. 
                    // Odd parity must be an error and syndrome is set.
                    sum % 2 == 1
                })
                .collect::<BitVec>();
            syndrome
                .iter()
                .enumerate()
                .for_each(|(i, s)| if s == true { syndrome_sum[i] += count_num });
            // Decode syndrome string (errors) with minimum weight perfect matching.
            // The subgraph bond index is the index of decoding bonds, 
            // i.e. index of fusion-blossom decoding graph edge.
            // If return is [0, 2] and the decoding bond indices are [2, 4, 6, 7, 8],
            // the actual bond index to flip is [2, 6].
            solver.solve(&build_syndrome_pattern(&syndrome));
            let mut bond_flips = bitvec![0; n_bonds];
            for ei in solver.subgraph().iter() {
                let bond_index = decoding_bonds[*ei].unwrap();
                *bond_flips.get_mut(bond_index).unwrap() = true;
            }
            // Combination of bond outcomes + decoder output.
            // Value of 0 (1) indicates AFM (FM) bond.
            let loop_string = !(bond_bits ^ bond_flips);
            // Compute gauge string and decode site string as new dict key.
            let mut gauge = bitvec![0; n_sites];
            for gsb in snake_line.iter() {
                let gauge_bit = gauge[gsb.1] ^ loop_string[gsb.2];
                *gauge.get_mut(gsb.0).unwrap() = gauge_bit;
            }
            // Decode and convert into little endian.
            let site_key = (site_bits ^ gauge)
                .iter()
                .rev()
                .map(|bit| if *bit { '1' } else { '0' })
                .collect::<String>();
            *decoded_counts.entry_ref(&site_key).or_insert(0) += count_num;
            total_shots += count_num;
        }
        let w_ops = syndrome_sum
            .into_iter()
            .map(|v| 1.0 - 2.0 * v as f64 / total_shots as f64)
            .collect_vec();
        let zxz_ops = bond_correlation_sum
            .into_iter()
            .map(|v| v as f64 / total_shots as f64)
            .collect_vec();
        let (f, g) = decode_magnetization(&decoded_counts);
        if return_counts {
            (Some(decoded_counts), w_ops, zxz_ops, f, g)
        } else {
            (None, w_ops, zxz_ops, f, g)
        }
    }
}

impl PyHeavyHexLattice {

    /// Create new plaquette lattice object for heavy hex topology.
    pub fn with_plaquettes(
        plaquette_qubits_map: HashMap<usize, Vec<usize>>,
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

        PyHeavyHexLattice {
            plaquette_qubits_map, 
            qubit_graph, 
            plaquette_graph,
            decode_graph,
        }
    }

    /// Build minimum weight perfect matching solver from the decode graph.
    /// Variables are bond qubits with having is_decode_variable flag set.
    /// Syndrome should have the same size with the plaquettes.
    fn build_mwpm_solver(
        &self,
    ) -> SolverSerial {
        // We have one boundary (virtual) node, where unpaired syndromes are paired with.
        // We don't need multiple virtual nodes because boundary condition is uniform 
        // and coupling graph (plaquette lattice) is a single SCC.
        let vertex_num = self.plaquette_qubits_map.len() + 1;
        let mut reverse_pq_map = HashMap::<QubitIndex, Vec<usize>>::new();
        self.plaquette_qubits_map
            .iter()
            // Sort by plaquette index.
            // Syndrome string is ordered by the corresponding plaquette index.
            .collect::<std::collections::BTreeMap<_, _>>()
            .values()
            .enumerate()
            .for_each(|(si, qis)| {
                for q in qis.iter() {
                    if let Some(pis) = reverse_pq_map.get_mut(q) {
                        pis.push(si);
                    } else {
                        reverse_pq_map.insert(*q, vec![si]);
                    }
                }
            });
        let decoding_edges = self.decode_graph
            .edge_weights()
            .filter_map(|ew| {
                if ew.is_decode_variable {
                    let syndromes = &reverse_pq_map[&ew.index];
                    let edge = match syndromes.len() {
                        // Boundary bond. Connect with the virtual vertex.
                        1 => (syndromes[0], vertex_num - 1, 2),
                        // Make real edge.
                        2 => (syndromes[0], syndromes[1], 2),
                        _ => panic!("Bond {} belongs to more than two or zero syndromes.", ew.index),
                    };
                    Some((ew.bit_index.unwrap(), edge))
                } else {
                    None
                }
            })
            // Sort by bond (bit) index
            .collect::<std::collections::BTreeMap<usize, (usize, usize, isize)>>()
            .into_values()
            .collect_vec();
        let ini = SolverInitializer::new(vertex_num, decoding_edges, vec![vertex_num - 1]);
        SolverSerial::new(&ini)
    }

}


/// Build syndrome pattern to feed MWPM solver.
fn build_syndrome_pattern(
    syndrome: &BitVec
) -> SyndromePattern {
    let defect_vertices = syndrome
        .iter()
        .enumerate()
        .filter_map(|(i, b)| if b == true { Some(i) } else { None })
        .collect();
    SyndromePattern::new(defect_vertices, vec![])
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{FALCON_CMAP, EAGLE_CMAP};

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

    #[test]
    fn test_traverse_snake() {
        let coupling_map = FALCON_CMAP.lock().unwrap().to_owned();
        let lattice = PyHeavyHexLattice::new(coupling_map);
        let snake = traverse_snake(&lattice.decode_graph);
        assert_eq!(
            snake,
            vec![
                (6, 8, 8),
                (4, 6, 6),
                (2, 4, 3),
                (0, 2, 1),
                (1, 0, 0),
                (3, 1, 2),
                (5, 3, 4),
                (7, 5, 7),
                (9, 7, 9),
            ]
        )
    }

    #[test]
    fn test_mwpm_18_syndromes() {
        // Test against pymatching decoder initialized by a check matrix H.
        let coupling_map = EAGLE_CMAP.lock().unwrap().to_owned();
        let lattice = PyHeavyHexLattice::new(coupling_map);
        let mut solver = lattice.build_mwpm_solver();
        // # test 1
        let syndrome = build_syndrome_pattern(&bitvec![0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0]);
        solver.clear();
        solver.solve(&syndrome);
        assert_eq!(solver.subgraph(), vec![10, 11, 28, 35, 40]);
        // # test 2
        let syndrome = build_syndrome_pattern(&bitvec![1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]);
        solver.clear();
        solver.solve(&syndrome);
        assert_eq!(solver.subgraph(), vec![1, 5, 24, 29, 36, 43]);
        // # test 3
        let syndrome = build_syndrome_pattern(&bitvec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
        solver.clear();
        solver.solve(&syndrome);
        assert_eq!(solver.subgraph(), vec![24, 31]);
    }

    #[test]
    fn test_decode_outcome() {
        // Validation
        //
        // Eagle coupling map
        // [q01]--<q04>--[q07]--<q10>--[q12]--<q15>--[q18]--<q21>--[q23]
        //   |                           |                           |
        // <q02>                       <q13>                       <q24>
        //   |                           |                           |
        // [q03]--<q05>--[q08]--<q11>--[q14]--<q16>--[q19]--<q22>--[q25]
        //
        // test string
        // 100111000111010001010
        //
        // mapping
        //       0      1      2      3      4      5      6      7      8      9      10
        // site: 0[q01] 0[q03] 0[q07] 0[q08] 1[q12] 1[q14] 0[q18] 1[q19] 0[q23] 1[q25]
        // bond: 1<q02> 1<q04> 0<q05> 1<q10> 0<q11> 1<q13> 0<q15> 0<q16> 1<q21> 1<q22> 0<q24> 
        //
        // bond string
        // 11010100110
        //
        // site string
        // 0000110101
        //
        // syndrome 0 (plaquette 0)
        // 110101 (0: even parity)
        // 
        // syndrome 1 (plaquette 1)
        // 100110 (1: odd parity)
        //
        // syndrome 
        // 01
        //
        // mwpm solution for decoding <q04> <q13> <q15> (select bond that flips only syndrome 1)
        // q15
        // 
        // loop (bond with q15 flipped, and invert bits)
        // 00101001001
        //
        // compute gauge
        // snake line
        //          : gauge[ 8] = 0
        // (6, 8, 8): gauge[ 6] = gauge[ 8] ^ loop[ 8] = 0 ^ 0 = 0
        // (4, 6, 6): gauge[ 4] = gauge[ 6] ^ loop[ 6] = 0 ^ 0 = 0
        // (2, 4, 3): gauge[ 2] = gauge[ 4] ^ loop[ 3] = 0 ^ 0 = 0
        // (0, 2, 1): gauge[ 0] = gauge[ 2] ^ loop[ 1] = 0 ^ 0 = 0
        // (1, 0, 0): gauge[ 1] = gauge[ 0] ^ loop[ 0] = 0 ^ 0 = 0
        // (3, 1, 2): gauge[ 3] = gauge[ 1] ^ loop[ 2] = 0 ^ 1 = 1
        // (5, 3, 4): gauge[ 5] = gauge[ 3] ^ loop[ 4] = 1 ^ 1 = 0
        // (7, 5, 7): gauge[ 7] = gauge[ 5] ^ loop[ 7] = 0 ^ 1 = 1
        // (9, 7, 9): gauge[ 9] = gauge[ 7] ^ loop[ 9] = 1 ^ 0 = 1
        // -------------------------------------------------------
        // gauge
        // 0001000101
        //
        // site ^ gauge
        // 0001110000
        //
        // to little endian
        // 0000111000
        //
        // bond correlation
        // bond[ 0] ^ site[ 0] ^ site[ 1] = 1 ^ 0 ^ 0 = 1
        // bond[ 1] ^ site[ 0] ^ site[ 2] = 1 ^ 0 ^ 0 = 1
        // bond[ 2] ^ site[ 1] ^ site[ 3] = 0 ^ 0 ^ 0 = 0
        // bond[ 3] ^ site[ 2] ^ site[ 4] = 1 ^ 0 ^ 1 = 0
        // bond[ 4] ^ site[ 3] ^ site[ 5] = 0 ^ 0 ^ 1 = 1
        // bond[ 5] ^ site[ 4] ^ site[ 5] = 1 ^ 1 ^ 1 = 1
        // bond[ 6] ^ site[ 4] ^ site[ 6] = 0 ^ 1 ^ 0 = 1
        // bond[ 7] ^ site[ 5] ^ site[ 7] = 0 ^ 1 ^ 1 = 0
        // bond[ 8] ^ site[ 6] ^ site[ 8] = 1 ^ 0 ^ 0 = 1
        // bond[ 9] ^ site[ 7] ^ site[ 9] = 1 ^ 1 ^ 1 = 1
        // bond[10] ^ site[ 8] ^ site[ 9] = 0 ^ 0 ^ 1 = 1
        //
        let coupling_map = FALCON_CMAP.lock().unwrap().to_owned();
        let lattice = PyHeavyHexLattice::new(coupling_map);
        let outcomes = HashMap::<String, usize>::from([(format!("100111000111010001010"), 123)]);
        let tmp = lattice.decode_outcomes(outcomes, true);
        let count = tmp.0.unwrap();
        let w_ops = tmp.1;
        let zxz_ops = tmp.2;
        let f = tmp.3.0;
        let g = tmp.4.0;
        assert_eq!(
            count,
            HashMap::<String, usize>::from([(format!("0000111000"), 123_usize)])
        );
        assert_eq!(
            w_ops,
            vec![1.0, -1.0]
        );
        assert_eq!(
            zxz_ops,
            vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(f, 0.0);
        assert_eq!(g, 0.0);
    }
}
