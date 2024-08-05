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

use std::{str, io::Write};

use ahash::RandomState;
use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;
use itertools::Itertools;

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableUnGraph;
use lazy_static::lazy_static;

use pyo3::{prelude::*, types::PyString};

use crate::graph::*;


lazy_static! {
    static ref SCHEDULING_PATTERN: Vec<SchedulingGroup> = vec![
        SchedulingGroup::E1,
        SchedulingGroup::E2,
        SchedulingGroup::E3,
        SchedulingGroup::E4,
        SchedulingGroup::E5,
        SchedulingGroup::E6,
        SchedulingGroup::E2,
        SchedulingGroup::E1,
        SchedulingGroup::E4,
        SchedulingGroup::E3,
        SchedulingGroup::E6,
        SchedulingGroup::E5,
    ];
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
    index: usize,
    #[pyo3(get)]
    role: String,
    #[pyo3(get)]
    group: String,
    #[pyo3(get)]
    coordinate: Option<(usize, usize)>,
    #[pyo3(get)]
    neighbors: Vec<usize>,
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
    index: usize,
    #[pyo3(get)]
    qubits: Vec<usize>,
    #[pyo3(get)]
    neighbors: Vec<usize>,
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
    index0: usize,
    #[pyo3(get)]
    index1: usize,
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
    pub plaquette_qubits_map: HashMap<usize, Vec<usize>>,
    pub qubit_graph: StableUnGraph<QubitNode, QubitEdge>,
    pub plaquette_graph: StableUnGraph<PlaquetteNode, PlaquetteEdge>,
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
        let plaquettes = build_plaquette(&qubits, &connectivity)
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

    /// Return dot script representing the site qubit graph 
    /// to create image with the graphviz drawer.
    pub fn site_graph_dot(&self, py: Python) -> PyResult<Option<PyObject>> {
        let site_graph = self.site_graph();
        let buf = ungraph_to_dot(&site_graph);
        Ok(Some(
            PyString::new_bound(py, str::from_utf8(&buf)?).to_object(py)
        ))
    }

    /// Return dot script representing the snake site qubit graph 
    /// to create image with the graphviz drawer.
    pub fn snake_graph_dot(&self, py: Python) -> PyResult<Option<PyObject>> {
        let snake_graph = self.snake_graph();
        let buf = ungraph_to_dot(&snake_graph);
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
            .map(|e| (e.q0, e.q1))
            .collect::<Vec<_>>();
        PyHeavyHexLattice::from_plaquettes(new_plaquettes, connectivity)
    }

    /// Schedule entangling gates to build GEM circuit.
    fn build_gate_schedule(&self, index: usize) -> Vec<Vec<PyScheduledGate>> {        
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
                                    let nw0 = self.qubit_graph.node_weight(reverse_node_map[&ew.q0]).unwrap();
                                    let nw1 = self.qubit_graph.node_weight(reverse_node_map[&ew.q1]).unwrap();
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
}

impl PyHeavyHexLattice {

    pub fn from_plaquettes(
        plaquettes: HashMap<usize, Vec<usize>>,
        connectivity: Vec<(usize, usize)>,
    ) -> Self {
        // Consider qubits in plaquettes
        let mut plq_qubits: Vec<usize> = plaquettes
            .values()
            .map(|qs| qs.to_owned())
            .flatten()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        plq_qubits.sort();
        // Build graphs
        let qubit_graph = build_qubit_graph(&plq_qubits, &connectivity);
        let plaquette_graph = build_plaquette_graph(&plaquettes);
        // Create new lattice struct
        let mut ret = PyHeavyHexLattice {
            plaquette_qubits_map: plaquettes, 
            qubit_graph, 
            plaquette_graph,
        };
        ret.annotate_nodes();
        ret.annotate_edges();
        ret
    }

    pub(crate) fn site_graph(&self) -> StableUnGraph<QubitNode, SiteEdge> {
        build_site_graph(&self.qubit_graph)
    }

    pub(crate) fn snake_graph(&self) -> StableUnGraph<QubitNode, SiteEdge> {
        let site_graph = build_site_graph(&self.qubit_graph);
        build_snake_graph(&site_graph)
    }

    fn annotate_nodes(&mut self) -> () {
        // In HHL, degree 3 nodes become site qubits.
        let mut deg3nodes: Vec<_> = self.qubit_graph
            .node_indices()
            .filter_map(|n| {
                let neighbors: Vec<_> = self.qubit_graph.neighbors(n).collect();
                if neighbors.len() == 3 {
                    Some(n)
                } else {
                    None
                }
            })
            .collect();
        if deg3nodes.len() == 0 {
            // When there is only one plaquette no degree 3 node exists.
            // In this case use top left.
            let min_node = self.qubit_graph
                .node_indices()
                .min_by_key(|n| {
                    let xy = self.qubit_graph.node_weight(*n).unwrap().coordinate.unwrap();
                    xy.0 + xy.1
                })
                .unwrap();
            deg3nodes.push(min_node);
        }
        for n in deg3nodes {
            let weight = self.qubit_graph.node_weight_mut(n).unwrap();
            weight.role = Some(QubitRole::Site);
        }
        let mut unassigned: Vec<_> = self.qubit_graph
            .node_indices()
            .filter_map(|n| {
                if self.qubit_graph.node_weight(n).unwrap().role.is_none() {
                    Some(n)
                } else {
                    None
                }
            })
            .collect();
        while let Some(n) = unassigned.pop() {
            let neighbor_roles: Vec<_> = self.qubit_graph
                .neighbors(n)
                .filter_map(|n: NodeIndex| self.qubit_graph.node_weight(n).unwrap().role)
                .collect();
            if neighbor_roles.len() == 0 {
                unassigned.insert(0, n);
                continue;
            }
            let weight_mut = self.qubit_graph.node_weight_mut(n).unwrap();
            if neighbor_roles.contains(&QubitRole::Bond) & neighbor_roles.contains(&QubitRole::Site) {
                panic!("Cannot resolve qubit role of Q{}.", weight_mut.index);
            } else if neighbor_roles.contains(&QubitRole::Bond) {
                weight_mut.role = Some(QubitRole::Site);
            } else if neighbor_roles.contains(&QubitRole::Site) {
                weight_mut.role = Some(QubitRole::Bond);
            }
        }
        // Assign OpGroup to site qubits
        let site_nodes: Vec<_> = self.qubit_graph
            .node_indices()
            .filter_map(|n| {
                if self.qubit_graph.node_weight(n).unwrap().role == Some(QubitRole::Site) {
                    Some(n)                
                } else {
                    None
                }
            })
            .collect();
    
        // Min qubit index node becomes Group A as a starting point
        let min_node = *site_nodes
            .iter()
            .min_by_key(|n| self.qubit_graph.node_weight(**n).unwrap().index)
            .unwrap();
        
        fn assign_recursive(
            node: NodeIndex, 
            graph: &mut StableUnGraph<QubitNode, QubitEdge>,
            group: OpGroup,
        ) -> () {
            let weight = graph.node_weight_mut(node).unwrap();
            if weight.group.is_some() {
                return;
            }
            weight.group = Some(group);
            let neighbor_sites: Vec<_> = graph
                .neighbors(node)
                .flat_map(|n| graph.neighbors(n).collect::<Vec<_>>())
                .collect();
            let next_group = match group {
                OpGroup::A => OpGroup::B,
                OpGroup::B => OpGroup::A,
            };
            for n_site in neighbor_sites {
                assign_recursive(n_site, graph, next_group)
            }
        }
    
        assign_recursive(min_node, &mut self.qubit_graph, OpGroup::A);
    }

    fn annotate_edges(&mut self) -> () {
        let node_map = self.qubit_graph
            .node_indices()
            .map(|n| (self.qubit_graph.node_weight(n).unwrap().index, n))
            .collect::<HashMap<_, _>>();
        for plaquette_qubits in self.plaquette_qubits_map.values() {
            let plq_nodes = plaquette_qubits
                .iter()
                .map(|qi| node_map[qi])
                .collect::<Vec<_>>();
            let get_xy = |n: &NodeIndex| -> (usize, usize) {
                self.qubit_graph.node_weight(*n).unwrap().coordinate.unwrap()
            };
            // Sort by distance from the minimum qubit node
            let topleft = *plq_nodes
                .iter()
                .min_by_key(|n| {
                    let xy = get_xy(*n);
                    xy.0 + xy.1
                })
                .unwrap();
            // Find previous node (counter-clock)
            let this_y = get_xy(&topleft).1;
            let mut any_prev = None;
            for nj in self.qubit_graph.neighbors(topleft) {
                if !plq_nodes.contains(&nj) {
                    continue;
                }
                if this_y != get_xy(&nj).1 {
                    any_prev = Some(nj);
                    break;
                }
            }
            let mut this = topleft.clone();
            let mut prev = if let Some(prev) = any_prev {
                prev
            } else {
                panic!("Plaquette is not closed loop.");
            };
            // Go clockwise from found edge
            let mut clkwise_edges = Vec::<EdgeIndex>::new();
            let coloring_iter = SCHEDULING_PATTERN
                .to_owned()
                .into_iter()
                .cycle();
            loop {
                for ni in self.qubit_graph.neighbors(this) {
                    if (ni == prev) | !plq_nodes.contains(&ni) {
                        continue;
                    }
                    let this_edge = self.qubit_graph.find_edge(this, ni).unwrap();
                    clkwise_edges.push(this_edge);
                    prev = this;
                    this = ni;
                    break;
                }
                if this == topleft {
                    break;
                }
            }
            for (edge, color) in clkwise_edges.into_iter().zip(coloring_iter) {
                let weight = self.qubit_graph.edge_weight_mut(edge).unwrap();
                if let Some(group) = weight.group {
                    if group != color {
                        panic!(
                            "Coupling edge from {} to {} has conflicting colors.",
                            weight.q0,
                            weight.q1,
                        )
                    } else {
                        continue;
                    }
                }
                weight.group = Some(color);
            }
        }

    }
}


pub(crate) fn build_qubit_graph(
    qubits: &Vec<usize>, 
    connectivity: &Vec<(usize, usize)>,
) -> StableUnGraph<QubitNode, QubitEdge> {
    let mut graph: StableUnGraph<QubitNode, QubitEdge> = StableUnGraph::with_capacity(qubits.len(), connectivity.len());
    // Build graph
    let mut qubit_node_map = HashMap::<usize, NodeIndex>::with_capacity(qubits.len());
    for qidx in qubits.iter() {
        let nidx = graph.add_node(QubitNode {index: *qidx, role: None, group: None, coordinate: None});
        qubit_node_map.insert(*qidx, nidx);
    }
    for (qi, qj) in connectivity.iter() {
        match (qubit_node_map.get(qi), qubit_node_map.get(qj)) {
            (Some(ni), Some(nj)) => {
                let edge = QubitEdge {q0: *qi, q1: *qj, group: None};
                graph.add_edge(*ni, *nj, edge);
            },
            // Node is not a part of plaquette
            _ => continue,
        }
    }
    // Assign qubit coordinate
    // Minimum qubit locates at (0, 0)
    let min_node = graph
        .node_indices()
        .min_by_key(|n| graph.node_weight(*n).unwrap().index)
        .unwrap();
    let node_weight = graph.node_weight_mut(min_node).unwrap();
    node_weight.coordinate = Some((0_usize, 0_usize));
    assign_coordinate_recursive(&min_node, &mut graph);
    // Check orientation.
    // Old IBM device may have different qubit index convention.
    let all_coords: Vec<_> = graph
        .node_weights()
        .map(|w| w.coordinate.unwrap())
        .collect();
    if !all_coords.contains(&(4_usize, 0_usize)) {
        for weight in graph.node_weights_mut() {
            let c0 = weight.coordinate.unwrap();
            weight.coordinate = Some((c0.1, c0.0));
        }
    }
    graph
}


pub(crate) fn build_plaquette_graph(
    plaquettes: &HashMap<usize, Vec<usize>>
) -> StableUnGraph<PlaquetteNode, PlaquetteEdge> {
    let nodes = plaquettes
        .keys()
        .map(|i| PlaquetteNode {index: *i})
        .collect::<Vec<_>>();
    let edges = plaquettes
        .iter()
        .tuple_combinations::<(_, _)>()
        .filter_map(
            |plqs| {
                let qs0 = plqs.0.1.iter().collect::<HashSet<_>>();
                let qs1 = plqs.1.1.iter().collect::<HashSet<_>>();
                if !qs0.is_disjoint(&qs1) {
                    Some(PlaquetteEdge {p0: *plqs.0.0, p1: *plqs.1.0})
                } else {
                    None
                }
            }
        )
        .collect::<Vec<_>>();
    let mut new_graph = StableUnGraph::<PlaquetteNode, PlaquetteEdge>::with_capacity(nodes.len(), edges.len());
    let mut node_map = HashMap::<usize, NodeIndex>::new();
    for node in nodes {
        let p_index = node.index;
        let n_index = new_graph.add_node(node);
        node_map.insert(p_index, n_index);
    }
    for edge in edges {
        let n1 = node_map[&edge.p0];
        let n2 = node_map[&edge.p1];
        new_graph.add_edge(n1, n2, edge);
    }
    new_graph
}


pub(crate) fn build_site_graph(
    graph: &StableUnGraph<QubitNode, QubitEdge>
) -> StableUnGraph<QubitNode, SiteEdge> {
    let site_nodes = graph
        .node_weights()
        .filter_map(|w| {
            if w.role == Some(QubitRole::Site) {
                Some(w.to_owned())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut node_map = HashMap::<usize, NodeIndex>::with_capacity(site_nodes.len());
    let mut site_graph = StableUnGraph::<QubitNode, SiteEdge>::with_capacity(site_nodes.len(), graph.node_count() - site_nodes.len());
    for node in site_nodes {
        let qindex = node.index;
        let nindex = site_graph.add_node(node);
        node_map.insert(qindex, nindex);
    }
    for ni in graph.node_indices() {
        let weight = graph.node_weight(ni).unwrap();
        if weight.role != Some(QubitRole::Bond) {
            continue;
        }
        let neighbors = graph
            .neighbors(ni)
            .map(|n| graph.node_weight(n).unwrap().index)
            .collect::<Vec<_>>();
        if neighbors.len() != 2 {
            panic!("Bond qubit doesn't have two site neighbors. Check if the lattice is heavy hex.")
        }
        let n0 = node_map[&neighbors[0]];
        let n1 = node_map[&neighbors[1]];
        site_graph.add_edge(n0, n1, SiteEdge { s1: neighbors[0], s2: neighbors[1], bond: weight.index});
    }
    site_graph
}


pub(crate) fn build_snake_graph(
    graph: &StableUnGraph<QubitNode, SiteEdge>
) -> StableUnGraph<QubitNode, SiteEdge> {
    let mut snake_graph = graph.clone();
    // Determine first edge (left-most or right-most) to keep
    let mut offset = 0_usize;
    let mut bond_by_row = std::collections::BTreeMap::<usize, Vec<(usize, EdgeIndex)>>::new();
    for ei in graph.edge_indices() {
        if let Some(sites) = graph.edge_endpoints(ei) {
            let xy0 = graph.node_weight(sites.0).unwrap().coordinate.unwrap();
            let xy1 = graph.node_weight(sites.1).unwrap().coordinate.unwrap();
            if (xy0.0 == xy1.0) & (xy0.1 != xy1.1) {
                // Edge connecting different rows
                let row_y = std::cmp::min(xy0.1, xy1.1);
                if let Some(edges) = bond_by_row.get_mut(&row_y) {
                    edges.push((xy0.0, ei));
                } else {
                    bond_by_row.insert(row_y, vec![(xy0.0, ei)]);
                }
            }
        }
    }
    let pos_eidx_tup = bond_by_row.values().collect::<Vec<_>>();
    if pos_eidx_tup.len() > 1 {
        let center_row0 = pos_eidx_tup[0].iter().fold(0, |acc, v| acc + v.0) as f64 / pos_eidx_tup[0].len() as f64;
        let center_row1 = pos_eidx_tup[1].iter().fold(0, |acc, v| acc + v.0) as f64 / pos_eidx_tup[1].len() as f64;
        if center_row0 > center_row1 {
            offset = 1;
        }
    }
    // Remove vertical edges to make snake
    for (ri, row) in pos_eidx_tup.into_iter().enumerate() {
        let keep = if (ri + offset) % 2 == 0 {
            row.iter().min_by_key(|e| e.0).unwrap()
        } else {
            row.iter().max_by_key(|e| e.0).unwrap()
        };
        for edge in row {
            if edge.1 != keep.1 {
                snake_graph.remove_edge(edge.1);
            }
        }
    }
    snake_graph
}


fn assign_coordinate_recursive(
    node: &NodeIndex,
    graph: &mut StableUnGraph<QubitNode, QubitEdge>,
) -> () {
    let neighbors: Vec<_> = graph
        .neighbors(*node)
        .map(|n| (n, *graph.node_weight(n).unwrap()))
        .collect();
    let qi = *graph.node_weight(*node).unwrap();
    let xy = qi.coordinate.unwrap();

    for (nj, qj) in neighbors {
        if let Some(_) = qj.coordinate {
            continue;
        }
        if let Some(qj_mut) = graph.node_weight_mut(nj) {
            let new_xy = if qj.index.abs_diff(qi.index) == 1 {
                // Move horizontally
                if qi.index < qj.index {
                    Some((xy.0 + 1, xy.1))
                } else {
                    Some((xy.0 - 1, xy.1))
                }
            } else {
                // Move vertically
                if qi.index < qj.index {
                    Some((xy.0, xy.1 + 1))
                } else {
                    Some((xy.0, xy.1 - 1))
                }
            };
            qj_mut.coordinate = new_xy;
            assign_coordinate_recursive(&nj, graph);
        }
    }
}


/// Lightweight Johnson's algorithm to find plaquettes.
/// HHL plaquette is a cycle of 12 qubits and the qubit connectivity is always a single connected components.
/// No self cycle exists and no isolated qubit exists.
/// These are valid assumptions for the coupling map of production backends.
/// 
/// # Arguments
/// 
/// * `qubits`: All qubits in the device coupling map.
/// * `connectivity`: Non-duplicated coupling pairs, e.g. [(0, 1), (1, 2), ...].
/// 
/// # Return
/// 
/// This function returns a vector of qubit list comprising a plaquette. 
/// The returned vector is sorted by the minimum qubit index in plaquettes.
pub fn build_plaquette(
    qubits: &Vec<usize>,
    connectivity: &Vec<(usize, usize)>,
) -> Vec<Vec<usize>> {
    let mut plaquettes = Vec::<Vec<usize>>::new();

    let mut scc = qubits.to_owned();
    let mut node_map: HashMap<usize, NodeIndex> = HashMap::with_capacity(scc.len());
    let mut scc_graph = StableUnGraph::<(), ()>::with_capacity(scc.len(), connectivity.len());
    for qubit in scc.iter() {
        let new_node = scc_graph.add_node(());
        node_map.insert(*qubit, new_node);
    }
    for qubits in connectivity.iter() {
        match (node_map.get(&qubits.0), node_map.get(&qubits.1)) {
            (Some(n0), Some(n1)) => {
                scc_graph.add_edge(*n0, *n1, ());
            },
            _ => continue
        }
    }
    let reverse_node_map = node_map.iter().map(|(k, v)| (*v, *k)).collect::<HashMap<_, _>>();

    // Reverse the order to pop from small qubit index
    scc.reverse();
    while let Some(next_qubit) = scc.pop() {
        let start_node = node_map[&next_qubit];
        let mut path: Vec<NodeIndex> = vec![start_node];
        let mut blocked: HashSet<NodeIndex> = path.iter().copied().collect();
        let mut closed: HashSet<NodeIndex> = HashSet::new();
        let mut block: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();
        let mut stack: Vec<(NodeIndex, IndexSet<NodeIndex, ahash::RandomState>)> = vec![(
            start_node,
            scc_graph.neighbors(start_node).collect::<IndexSet<NodeIndex, ahash::RandomState>>()
        )];
        while let Some(res) = process_stack(
            start_node, 
            &mut stack, 
            &mut path, 
            &mut closed, 
            &mut blocked, 
            &mut block, 
            &scc_graph, 
            &reverse_node_map,
        ) {
            if !plaquettes.contains(&res) {
                // Because the coupling graph is undirected,
                // the result can be either clockwise and counter-clockwise cycle.
                // Take only one since they are the identical plaquette.
                plaquettes.push(res);
            }
        }
        scc_graph.remove_node(start_node);
    }
    plaquettes.sort_unstable_by_key(|plq| plq[0]);
    plaquettes
}


fn to_undirected(
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
        .collect::<Vec<_>>();
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


fn unblock(
    node: NodeIndex,
    blocked: &mut HashSet<NodeIndex>,
    block: &mut HashMap<NodeIndex, HashSet<NodeIndex>>,
) {
    let mut stack: IndexSet<NodeIndex, RandomState> = IndexSet::with_hasher(RandomState::default());
    stack.insert(node);
    while let Some(stack_node) = stack.pop() {
        if blocked.remove(&stack_node) {
            match block.get_mut(&stack_node) {
                Some(block_set) => {
                    block_set.drain().for_each(|n| {
                        stack.insert(n);
                    });
                }
                None => {
                    block.insert(stack_node, HashSet::new());
                }
            }
            blocked.remove(&stack_node);
        }
    }
}


fn process_stack(
    start_node: NodeIndex,
    stack: &mut Vec<(NodeIndex, IndexSet<NodeIndex, ahash::RandomState>)>,
    path: &mut Vec<NodeIndex>,
    closed: &mut HashSet<NodeIndex>,
    blocked: &mut HashSet<NodeIndex>,
    block: &mut HashMap<NodeIndex, HashSet<NodeIndex>>,
    subgraph: &StableUnGraph<(), ()>,
    reverse_node_map: &HashMap<NodeIndex, usize>,
) -> Option<Vec<usize>> {
    if subgraph.node_count() < 12 {
        // No hope to find HHL plaquette
        return None;
    }
    while let Some((this_node, neighbors)) = stack.last_mut() {

        if let Some(next_node) = neighbors.pop() {
            if next_node == start_node {
                if path.len() == 12 {
                    // Out path in input graph basis
                    let mut out_path: Vec<usize> = Vec::with_capacity(path.len());
                    for n in path {
                        out_path.push(reverse_node_map[n]);
                        closed.insert(*n);
                    }
                    out_path.sort();
                    return Some(out_path);
                }
            } else if blocked.insert(next_node) {
                if path.len() < 12 {
                    path.push(next_node);
                    stack.push((
                        next_node,
                        subgraph.neighbors(next_node).collect::<IndexSet<NodeIndex, ahash::RandomState>>(),
                    ));
                    closed.remove(&next_node);
                    blocked.insert(next_node);
                    continue;
                } else {
                    // Undo insert because this node is not added to the path
                    // because of the length limit.
                    blocked.remove(&next_node);
                }
            }
        }
        if neighbors.is_empty() {
            if closed.contains(this_node) {
                unblock(*this_node, blocked, block);
            } else {
                for neighbor in subgraph.neighbors(*this_node) {
                    let block_neighbor = block.entry(neighbor).or_insert_with(HashSet::new);
                    block_neighbor.insert(*this_node);
                }
            }
            stack.pop();
            path.pop();
        }
    }
    None
}


fn ungraph_to_dot<N: WriteDot, E: WriteDot>(
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{FALCON_CMAP, EAGLE_CMAP};

    #[test]
    fn test_falcon_map() {
        let coupling_map = FALCON_CMAP.lock().unwrap();
        let (qubits, connectivity) = to_undirected(&coupling_map);

        let plaquettes = build_plaquette(&qubits, &connectivity);
        assert_eq!(plaquettes.len(), 2);
        assert_eq!(
            plaquettes[0],
            vec![1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14],
        );
        assert_eq!(
            plaquettes[1],
            vec![12, 13, 14, 15, 16, 18, 19, 21, 22, 23, 24, 25],
        );
    }

    #[test]
    fn test_eagle_map() {
        let coupling_map = EAGLE_CMAP.lock().unwrap();
        let (qubits, connectivity) = to_undirected(&coupling_map);

        let plaquettes = build_plaquette(&qubits, &connectivity);
        assert_eq!(plaquettes.len(), 18);
        assert_eq!(
            plaquettes[0],
            vec![0, 1, 2, 3, 4, 14, 15, 18, 19, 20, 21, 22],
        );
        assert_eq!(
            plaquettes[1],
            vec![4, 5, 6, 7, 8, 15, 16, 22, 23, 24, 25, 26],
        );
        assert_eq!(
            plaquettes[2],
            vec![8, 9, 10, 11, 12, 16, 17, 26, 27, 28, 29, 30],
        );
        assert_eq!(
            plaquettes[3],
            vec![20, 21, 22, 23, 24, 33, 34, 39, 40, 41, 42, 43],
        );
        assert_eq!(
            plaquettes[4],
            vec![24, 25, 26, 27, 28, 34, 35, 43, 44, 45, 46, 47],
        );
        assert_eq!(
            plaquettes[5],
            vec![28, 29, 30, 31, 32, 35, 36, 47, 48, 49, 50, 51],
        );
        assert_eq!(
            plaquettes[6],
            vec![37, 38, 39, 40, 41, 52, 53, 56, 57, 58, 59, 60],
        );
        assert_eq!(
            plaquettes[7],
            vec![41, 42, 43, 44, 45, 53, 54, 60, 61, 62, 63, 64],
        );
        assert_eq!(
            plaquettes[8],
            vec![45, 46, 47, 48, 49, 54, 55, 64, 65, 66, 67, 68],
        );
        assert_eq!(
            plaquettes[9],
            vec![58, 59, 60, 61, 62, 71, 72, 77, 78, 79, 80, 81],
        );
        assert_eq!(
            plaquettes[10],
            vec![62, 63, 64, 65, 66, 72, 73, 81, 82, 83, 84, 85],
        );
        assert_eq!(
            plaquettes[11],
            vec![66, 67, 68, 69, 70, 73, 74, 85, 86, 87, 88, 89],
        );
        assert_eq!(
            plaquettes[12],
            vec![75, 76, 77, 78, 79, 90, 91, 94, 95, 96, 97, 98],
        );
        assert_eq!(
            plaquettes[13],
            vec![79, 80, 81, 82, 83, 91, 92, 98, 99, 100, 101, 102],
        );
        assert_eq!(
            plaquettes[14],
            vec![83, 84, 85, 86, 87, 92, 93, 102, 103, 104, 105, 106],
        );
        assert_eq!(
            plaquettes[15],
            vec![96, 97, 98, 99, 100, 109, 110, 114, 115, 116, 117, 118],
        );
        assert_eq!(
            plaquettes[16],
            vec![100, 101, 102, 103, 104, 110, 111, 118, 119, 120, 121, 122],
        );
        assert_eq!(
            plaquettes[17],
            vec![104, 105, 106, 107, 108, 111, 112, 122, 123, 124, 125, 126],
        );
    }

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
