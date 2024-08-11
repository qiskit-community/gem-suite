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

use hashbrown::{HashMap, HashSet};
use itertools::Itertools;

use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableUnGraph;
use lazy_static::lazy_static;

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
}


/// Build basic qubit coupling graph 
/// with annotation for GEM circuit generation.
/// A node in the graph is a qubit, and an edge is physical coupling between qubits.
pub(super) fn build_qubit_graph(
    qubits: &Vec<usize>, 
    connectivity: &Vec<(usize, usize)>,
    plaquette_qubits_map: &HashMap<PlaquetteIndex, Vec<QubitIndex>>,
) -> StableUnGraph<QubitNode, QubitEdge> {
    let mut graph: StableUnGraph<QubitNode, QubitEdge> = StableUnGraph::with_capacity(qubits.len(), connectivity.len());
    // Build graph
    let mut node_map = HashMap::<usize, NodeIndex>::with_capacity(qubits.len());
    for qidx in qubits.iter() {
        let nidx = graph.add_node(QubitNode {index: *qidx, role: None, group: None, coordinate: None});
        node_map.insert(*qidx, nidx);
    }
    for (qi, qj) in connectivity.iter() {
        match (node_map.get(qi), node_map.get(qj)) {
            (Some(ni), Some(nj)) => {
                let edge = QubitEdge {neighbor0: *qi, neighbor1: *qj, group: None};
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
    assign_qubit_coordinate_recursive(&min_node, &mut graph);
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
    annotate_nodes(&mut graph);
    annotate_edges(&mut graph, plaquette_qubits_map);
    graph
}


/// Build plaquette graph.
/// A graph node corresponds to a single heavy hex lattice,
/// and an edge between nodes represents some shared qubits between plaquettes.
pub(super) fn build_plaquette_graph(
    plaquette_qubits_map: &HashMap<PlaquetteIndex, Vec<QubitIndex>>,
) -> StableUnGraph<PlaquetteNode, PlaquetteEdge> {
    let nodes = plaquette_qubits_map
        .iter()
        .collect::<std::collections::BTreeMap<_, _>>()
        .into_keys()
        .enumerate()
        .map(|(si, pi)| PlaquetteNode { index: *pi, syndrome_index: si })
        .collect_vec();
    let edges = plaquette_qubits_map
        .iter()
        .tuple_combinations::<(_, _)>()
        .filter_map(
            |plqs| {
                let qs0 = plqs.0.1.iter().collect::<HashSet<_>>();
                let qs1 = plqs.1.1.iter().collect::<HashSet<_>>();
                if !qs0.is_disjoint(&qs1) {
                    Some(PlaquetteEdge {neighbor0: *plqs.0.0, neighbor1: *plqs.1.0})
                } else {
                    None
                }
            }
        )
        .collect_vec();
    let mut graph = StableUnGraph::<PlaquetteNode, PlaquetteEdge>::with_capacity(nodes.len(), edges.len());
    let mut node_map = HashMap::<usize, NodeIndex>::new();
    for node in nodes {
        let p_index = node.index;
        let n_index = graph.add_node(node);
        node_map.insert(p_index, n_index);
    }
    for edge in edges {
        let n1 = node_map[&edge.neighbor0];
        let n2 = node_map[&edge.neighbor1];
        graph.add_edge(n1, n2, edge);
    }
    graph
}


/// Build graph with code annotation.
/// In this graph, nodes are site qubits and edges are bond qubits.
/// Each plaquette represents syndrome bit, i.e. W-operator
/// and bond qubits correspond to noise mechanisms.
/// Site and bond qubits gain new index in this graph.
pub(super) fn build_decode_graph(
    qubit_graph: &StableUnGraph<QubitNode, QubitEdge>,
    plaquette_graph: &StableUnGraph<PlaquetteNode, PlaquetteEdge>,
    plaquette_qubits_map: &HashMap<PlaquetteIndex, Vec<QubitIndex>>,
) -> StableUnGraph<DecodeNode, DecodeEdge> {
    // Build unannotated site graph.
    let mut site_qubits = qubit_graph
        .node_weights()
        .filter_map(|nw| {
            if nw.role == Some(QubitRole::Site) {
                Some(DecodeNode {
                    index: nw.index,
                    bit_index: None,
                    coordinate: nw.coordinate.unwrap(),
                })
            } else {
                None
            }
        })
        .collect_vec();
    site_qubits.sort_unstable_by_key(|q| q.index);
    for (bi, q) in site_qubits.iter_mut().enumerate() {
        q.bit_index = Some(bi);
    }
    let mut bond_qubits = qubit_graph
        .node_indices()
        .filter_map(|ni| {
            let nw = qubit_graph.node_weight(ni).unwrap();
            if nw.role == Some(QubitRole::Bond) {
                let neighbors = qubit_graph
                    .neighbors(ni)
                    .map(|mi| qubit_graph.node_weight(mi).unwrap().index)
                    .collect_vec();
                if neighbors.len() != 2 {
                    panic!("Bond qubit doesn't have two site neighbors. Check if the lattice is heavy hex.")
                }
                Some(DecodeEdge {
                    index: nw.index,
                    neighbor0: neighbors[0],
                    neighbor1: neighbors[1],
                    bit_index: None,
                    variable_index: None,
                    keep_in_snake: true,
                })
            } else {
                None
            }
        })
        .collect_vec();
    bond_qubits.sort_unstable_by_key(|q| q.index);
    for (bi, q) in bond_qubits.iter_mut().enumerate() {
        q.bit_index = Some(bi);
    }
    let mut node_map = HashMap::<QubitIndex, NodeIndex>::with_capacity(site_qubits.len());
    let mut decode_graph = StableUnGraph::<DecodeNode, DecodeEdge>::with_capacity(site_qubits.len(), bond_qubits.len());
    for site in site_qubits {
        let qi = site.index;
        let ni = decode_graph.add_node(site);
        node_map.insert(qi, ni);
    }
    for bond in bond_qubits {
        let ni0 = node_map[&bond.neighbor0];
        let ni1 = node_map[&bond.neighbor1];
        decode_graph.add_edge(ni0, ni1, bond);
    }
    // Compute snake attribute to form zig-zag pattern
    // Determine first edge (left-most or right-most) to keep
    let mut offset = 0_usize;
    let mut bond_by_row = std::collections::BTreeMap::<usize, Vec<(usize, EdgeIndex)>>::new();
    for ei in decode_graph.edge_indices() {
        if let Some(sites) = decode_graph.edge_endpoints(ei) {
            let xy0 = decode_graph.node_weight(sites.0).unwrap().coordinate;
            let xy1 = decode_graph.node_weight(sites.1).unwrap().coordinate;
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
    let pos_eidx_tup = bond_by_row.values().collect_vec();
    if pos_eidx_tup.len() > 1 {
        let center_row0 = pos_eidx_tup[0].iter().fold(0, |acc, v| acc + v.0) as f64 / pos_eidx_tup[0].len() as f64;
        let center_row1 = pos_eidx_tup[1].iter().fold(0, |acc, v| acc + v.0) as f64 / pos_eidx_tup[1].len() as f64;
        if center_row0 > center_row1 {
            offset = 1;
        }
    }
    // Set snake removal flag
    for (ri, row) in pos_eidx_tup.into_iter().enumerate() {
        let keep = if (ri + offset) % 2 == 0 {
            row.iter().min_by_key(|e| e.0).unwrap()
        } else {
            row.iter().max_by_key(|e| e.0).unwrap()
        };
        for edge in row {
            if edge.1 != keep.1 {
                let remove_edge = decode_graph.edge_weight_mut(edge.1).unwrap();
                remove_edge.keep_in_snake = false;
            }
        }
    }
    // Compute decoding bonds
    // Decoding bonds are bond qubits shared by multiple plaquettes,
    // or a neighbor of a site qubit with degree-2 connectivity, i.e. boundary.
    let edge_map = decode_graph
        .edge_indices()
        .map(|e| {
            let ew = decode_graph.edge_weight(e).unwrap();
            (ew.index, (ew.bit_index.unwrap(), e))
        })
        .collect::<HashMap<_, _>>();
    let mut decoding_edges = std::collections::BTreeMap::<usize, EdgeIndex>::new();
    // Find shared bonds
    for plq_edge in plaquette_graph.edge_weights() {
        let p0_qubits = plaquette_qubits_map[&plq_edge.neighbor0].iter().collect::<HashSet<_>>();
        let p1_qubits = plaquette_qubits_map[&plq_edge.neighbor1].iter().collect::<HashSet<_>>();
        for qi in p0_qubits.intersection(&p1_qubits) {
            if let Some(decode_edge) = edge_map.get(*qi) {
                decoding_edges.insert(decode_edge.0, decode_edge.1);
            }
        }
    }
    // Find boundary bond
    for plq_node in plaquette_graph.node_weights() {
        let plq_qubits = &plaquette_qubits_map[&plq_node.index];
        for ni in decode_graph.node_indices() {
            let nw = decode_graph.node_weight(ni).unwrap();
            if !plq_qubits.contains(&nw.index) {
                continue;
            }
            let neighbors = decode_graph.neighbors(ni).collect_vec();
            if neighbors.len() == 2 {
                let boundary_edge = decode_graph.find_edge(ni, neighbors[0]).unwrap();
                let bit_idx = decode_graph.edge_weight(boundary_edge).unwrap().bit_index.unwrap();
                decoding_edges.insert(bit_idx, boundary_edge);
                // Move to next plaquette
                break;
            }
        }
    }
    // Set decoding flag
    decoding_edges
        .into_values()
        .enumerate()
        .for_each(|(i, ei)| {
            let ew = decode_graph.edge_weight_mut(ei).unwrap();
            ew.variable_index = Some(i);
        });
    decode_graph
}


fn assign_qubit_coordinate_recursive(
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
            assign_qubit_coordinate_recursive(&nj, graph);
        }
    }
}


fn annotate_nodes(qubit_graph: &mut StableUnGraph<QubitNode, QubitEdge>) -> () {
    // In HHL, degree 3 nodes become site qubits.
    let mut deg3nodes: Vec<_> = qubit_graph
        .node_indices()
        .filter_map(|n| {
            let neighbors: Vec<_> = qubit_graph.neighbors(n).collect();
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
        let min_node = qubit_graph
            .node_indices()
            .min_by_key(|n| {
                let xy = qubit_graph.node_weight(*n).unwrap().coordinate.unwrap();
                xy.0 + xy.1
            })
            .unwrap();
        deg3nodes.push(min_node);
    }
    for n in deg3nodes {
        let weight = qubit_graph.node_weight_mut(n).unwrap();
        weight.role = Some(QubitRole::Site);
    }
    let mut unassigned: Vec<_> = qubit_graph
        .node_indices()
        .filter_map(|n| {
            if qubit_graph.node_weight(n).unwrap().role.is_none() {
                Some(n)
            } else {
                None
            }
        })
        .collect();
    while let Some(n) = unassigned.pop() {
        let neighbor_roles: Vec<_> = qubit_graph
            .neighbors(n)
            .filter_map(|n: NodeIndex| qubit_graph.node_weight(n).unwrap().role)
            .collect();
        if neighbor_roles.len() == 0 {
            unassigned.insert(0, n);
            continue;
        }
        let weight_mut = qubit_graph.node_weight_mut(n).unwrap();
        if neighbor_roles.contains(&QubitRole::Bond) & neighbor_roles.contains(&QubitRole::Site) {
            panic!("Cannot resolve qubit role of Q{}.", weight_mut.index);
        } else if neighbor_roles.contains(&QubitRole::Bond) {
            weight_mut.role = Some(QubitRole::Site);
        } else if neighbor_roles.contains(&QubitRole::Site) {
            weight_mut.role = Some(QubitRole::Bond);
        }
    }
    // Assign OpGroup to site qubits
    let site_nodes: Vec<_> = qubit_graph
        .node_indices()
        .filter_map(|n| {
            if qubit_graph.node_weight(n).unwrap().role == Some(QubitRole::Site) {
                Some(n)                
            } else {
                None
            }
        })
        .collect();

    // Min qubit index node becomes Group A as a starting point
    let min_node = *site_nodes
        .iter()
        .min_by_key(|n| qubit_graph.node_weight(**n).unwrap().index)
        .unwrap();

    assign_opgroup_recursive(min_node, qubit_graph, OpGroup::A);
}


fn annotate_edges(
    qubit_graph: &mut StableUnGraph<QubitNode, QubitEdge>,
    plaquette_qubits_map: &HashMap<PlaquetteIndex, Vec<QubitIndex>>,
) -> () {
    let node_map = qubit_graph
        .node_indices()
        .map(|n| (qubit_graph.node_weight(n).unwrap().index, n))
        .collect::<HashMap<_, _>>();
    for plaquette_qubits in plaquette_qubits_map.values() {
        let plq_nodes = plaquette_qubits
            .iter()
            .map(|qi| node_map[qi])
            .collect_vec();
        let get_xy = |n: &NodeIndex| -> (usize, usize) {
            qubit_graph.node_weight(*n).unwrap().coordinate.unwrap()
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
        for nj in qubit_graph.neighbors(topleft) {
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
            for ni in qubit_graph.neighbors(this) {
                if (ni == prev) | !plq_nodes.contains(&ni) {
                    continue;
                }
                let this_edge = qubit_graph.find_edge(this, ni).unwrap();
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
            let weight = qubit_graph.edge_weight_mut(edge).unwrap();
            if let Some(group) = weight.group {
                if group != color {
                    panic!(
                        "Coupling edge from {} to {} has conflicting colors.",
                        weight.neighbor0,
                        weight.neighbor1,
                    )
                } else {
                    continue;
                }
            }
            weight.group = Some(color);
        }
    }
}


fn assign_opgroup_recursive(
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
        .flat_map(|n| graph.neighbors(n).collect_vec())
        .collect();
    let next_group = match group {
        OpGroup::A => OpGroup::B,
        OpGroup::B => OpGroup::A,
    };
    for n_site in neighbor_sites {
        assign_opgroup_recursive(n_site, graph, next_group)
    }
}


/// Traverse snake graph and returns a vector of three bit index tuple
/// (gauge, site, bond) that draw snake pattern in one stroke.
pub(super) fn traverse_snake(
    graph: &StableUnGraph<DecodeNode, DecodeEdge>,
) -> Vec<(usize, usize, usize)> {
    let snake_ends = graph
        .node_indices()
        .filter_map(|ni| {
            let n_neighbors = graph
                .neighbors(ni)
                .fold(0_usize, |sum, mi| {
                    let ei = graph.find_edge(ni, mi).unwrap();
                    if graph.edge_weight(ei).unwrap().keep_in_snake {
                        sum + 1
                    } else {
                        sum
                    }
                });
            if n_neighbors == 1 {
                Some(ni)
            } else {
                None
            }
        })
        .collect_vec();
    if snake_ends.len() != 2 {
        panic!(
            "The snake graph has more than two end nodes. Likely invalid plaquette sublattice."
        )
    }
    let start_node = std::cmp::min_by_key(
        snake_ends[0], 
        snake_ends[1], 
        |ni| graph.node_weight(*ni).unwrap().bit_index.unwrap()
    );
    let mut visited = Vec::<NodeIndex>::with_capacity(graph.node_count());
    let mut this = start_node;
    let mut snake_edge = Vec::<(usize, usize, usize)>::new();
    loop {
        let neighbors = graph
            .neighbors(this)
            .filter_map(|ni| {
                let ei = graph.find_edge(this, ni).unwrap();
                if !visited.contains(&ni) & graph.edge_weight(ei).unwrap().keep_in_snake {
                    Some((ni, ei))
                } else {
                    None
                }
            })
            .collect_vec();
        let next = match neighbors.len() {
            1 => neighbors[0],
            0 => break,
            _ => panic!("Snake graph is not in one stroke."),
        };
        let gauge = graph.node_weight(next.0).unwrap().bit_index.unwrap();
        let site = graph.node_weight(this).unwrap().bit_index.unwrap();
        let bond = graph.edge_weight(next.1).unwrap().bit_index.unwrap();
        visited.push(this);
        this = next.0;
        snake_edge.push((gauge, site, bond));
    }
    snake_edge
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::heavyhex::PyHeavyHexLattice;
    use crate::mock::FALCON_CMAP;

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
}
