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

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableUnGraph;
use lazy_static::lazy_static;

use pyo3::{prelude::*, types::PyString};

use crate::plaquette_model::{CouplingEdge, OpGroup, QubitNode, QubitRole, SchedulingGroup};


lazy_static! {
    static ref SCHEDULING_ORDER: Vec<SchedulingGroup> = vec![
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
}


#[pyclass]
pub struct PyHeavyHexPlaquette {
    #[pyo3(get)]
    pub plaquettes: Vec<Vec<usize>>,
    pub graph: StableUnGraph<QubitNode, CouplingEdge>,    
}

#[pymethods]
impl PyHeavyHexPlaquette{
    #[new]
    pub fn new(coupling_map: Vec<(usize, usize)>) -> Self {
        let (qubits, connectivity) = to_undirected(&coupling_map);
        let plaquettes = build_plaquette(&qubits, &connectivity);
        // Consider qubits in plaquettes
        let mut plq_qubits: Vec<usize> = plaquettes
            .clone()
            .into_iter()
            .flatten()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        plq_qubits.sort();
        // plq_qubits = qubits.clone();
        let mut coupling_graph: StableUnGraph<QubitNode, CouplingEdge> = StableUnGraph::with_capacity(plq_qubits.len(), connectivity.len());
        // Build graph
        let mut qubit_node_map = HashMap::<usize, NodeIndex>::with_capacity(plq_qubits.len());
        for qidx in plq_qubits.iter() {
            let nidx = coupling_graph.add_node(QubitNode {index: *qidx, role: None, group: None, coordinate: None});
            qubit_node_map.insert(*qidx, nidx);
        }
        for (qi, qj) in connectivity.iter() {
            let edge = CouplingEdge {q0: *qi, q1: *qj, group: None};
            match (qubit_node_map.get(qi), qubit_node_map.get(qj)) {
                (Some(ni), Some(nj)) => {
                    coupling_graph.add_edge(*ni, *nj, edge);
                },
                // Node is not a part of plaquette
                _ => continue,
            }
        }
        // Assign qubit coordinate
        // Minimum qubit locates at (0, 0)
        let min_node = coupling_graph
            .node_indices()
            .min_by_key(|n| coupling_graph.node_weight(*n).unwrap().index)
            .unwrap();
        let node_weight = coupling_graph.node_weight_mut(min_node).unwrap();
        node_weight.coordinate = Some((0_usize, 0_usize));
        assign_coordinate_recursive(&min_node, &mut coupling_graph);
        // Check orientation.
        // Old IBM device may have different qubit index convention.
        let all_coords: Vec<_> = coupling_graph
            .node_weights()
            .map(|w| w.coordinate.unwrap())
            .collect();
        if !all_coords.contains(&(4_usize, 0_usize)) {
            for weight in coupling_graph.node_weights_mut() {
                let c0 = weight.coordinate.unwrap();
                weight.coordinate = Some((c0.1, c0.0));
            }
        }
        let mut ret = PyHeavyHexPlaquette {plaquettes, graph: coupling_graph};
        ret.annotate_nodes();
        ret.annotate_edges();

        ret
    }

    pub fn to_dot(&self, py: Python) -> PyResult<Option<PyObject>> {
        let mut buf = Vec::<u8>::new();
        writeln!(&mut buf, "graph {{").unwrap();
        writeln!(&mut buf, "node [fontname=\"Consolas\", fontsize=8.0, height=0.7];")?;
        writeln!(&mut buf, "edge [fontname=\"Consolas\", penwidth=2.5];")?;
        writeln!(&mut buf, "layout=fdp;")?;
        for node in self.graph.node_weights() {
            writeln!(&mut buf, "{}", node.to_dot()).unwrap();
        }
        for edge in self.graph.edge_weights() {
            writeln!(&mut buf, "{}", edge.to_dot()).unwrap();
        }
        writeln!(&mut buf, "}}").unwrap();
        Ok(Some(
            PyString::new_bound(py, str::from_utf8(&buf)?).to_object(py)
        ))
    }
}

impl PyHeavyHexPlaquette {

    fn annotate_nodes(&mut self) -> () {
        // In HHL, degree 3 nodes become site qubits.
        let mut deg3nodes: Vec<_> = self.graph
            .node_indices()
            .filter_map(|n| {
                let neighbors: Vec<_> = self.graph.neighbors(n).collect();
                if neighbors.len() == 3 {
                    Some(n)
                } else {
                    None
                }
            })
            .collect();
        if deg3nodes.len() == 0 {
            // When there is only one plaquette no degree 3 node exists.
            // deg3nodes.push();
        }
        for n in deg3nodes {
            let weight = self.graph.node_weight_mut(n).unwrap();
            weight.role = Some(QubitRole::Site);
        }
        let mut unassigned: Vec<_> = self.graph
            .node_indices()
            .filter_map(|n| {
                if self.graph.node_weight(n).unwrap().role.is_none() {
                    Some(n)
                } else {
                    None
                }
            })
            .collect();
        while let Some(n) = unassigned.pop() {
            let neighbor_roles: Vec<_> = self.graph
                .neighbors(n)
                .filter_map(|n: NodeIndex| self.graph.node_weight(n).unwrap().role)
                .collect();
            if neighbor_roles.len() == 0 {
                unassigned.insert(0, n);
                continue;
            }
            let weight_mut = self.graph.node_weight_mut(n).unwrap();
            if neighbor_roles.contains(&QubitRole::Bond) & neighbor_roles.contains(&QubitRole::Site) {
                panic!("Cannot resolve qubit role of Q{}.", weight_mut.index);
            } else if neighbor_roles.contains(&QubitRole::Bond) {
                weight_mut.role = Some(QubitRole::Site);
            } else if neighbor_roles.contains(&QubitRole::Site) {
                weight_mut.role = Some(QubitRole::Bond);
            }
        }
        // Assign OpGroup to site qubits
        let site_nodes: Vec<_> = self.graph
            .node_indices()
            .filter_map(|n| {
                if self.graph.node_weight(n).unwrap().role == Some(QubitRole::Site) {
                    Some(n)                
                } else {
                    None
                }
            })
            .collect();
    
        // Min qubit index node becomes Group A as a starting point
        let min_node = *site_nodes
            .iter()
            .min_by_key(|n| self.graph.node_weight(**n).unwrap().index)
            .unwrap();
        
        fn assign_recursive(
            node: NodeIndex, 
            graph: &mut StableUnGraph<QubitNode, CouplingEdge>,
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
    
        assign_recursive(min_node, &mut self.graph, OpGroup::A);
    }

    fn annotate_edges(&mut self) -> () {
        let node_map = self.graph
            .node_indices()
            .map(|n| (self.graph.node_weight(n).unwrap().index, n))
            .collect::<HashMap<_, _>>();
        for plaquette_qubits in self.plaquettes.iter() {
            let plq_nodes = plaquette_qubits
                .iter()
                .map(|qi| node_map[qi])
                .collect::<Vec<_>>();
            let get_xy = |n: &NodeIndex| -> (usize, usize) {
                self.graph.node_weight(*n).unwrap().coordinate.unwrap()
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
            for nj in self.graph.neighbors(topleft) {
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
            let coloring_iter = SCHEDULING_ORDER
                .to_owned()
                .into_iter()
                .cycle();
            loop {
                for ni in self.graph.neighbors(this) {
                    if (ni == prev) | !plq_nodes.contains(&ni) {
                        continue;
                    }
                    let this_edge = self.graph.find_edge(this, ni).unwrap();
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
                let weight = self.graph.edge_weight_mut(edge).unwrap();
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


fn assign_coordinate_recursive(
    node: &NodeIndex,
    graph: &mut StableUnGraph<QubitNode, CouplingEdge>,
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
}
