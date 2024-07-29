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

use ahash::RandomState;
use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;

use petgraph::algo::kosaraju_scc;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableUnGraph;


/// Lightweight Johnson's algorithm to find plaquettes.
/// HHL plaquette is a cycle of 12 qubits and the qubit connectivity is always a single connected components.
/// No self cycle exists and no isolated qubit exists.
/// Qubit index always starts from 0.
/// These are valid assumptions for the coupling map of production backends.
/// 
/// # Arguments
/// 
/// * `coupling_map`: A device coupling map in a vector of two usize tuple, e.g. [(0, 1), (2, 3), ...].
/// 
/// # Return
/// 
/// This function returns ...
/// 
pub fn build_plaquette(coupling_map: Vec<(usize, usize)>) -> Vec<Vec<usize>> {
    let mut plaquettes = Vec::<Vec<usize>>::new();

    // Format coupling map to build undirectional subgraph of initial SCC.
    // Coupling map is defined by instructions and it can be either uni-directional or bi-directional.
    // For example, in the ECR architecture the coupling map may be single direction but
    // the tunable coupler architecture has bidirectional coupling since CZ gate is symmetric.
    let connectivity: HashSet<(NodeIndex, NodeIndex)> = coupling_map
        .iter()
        .map(|p| {
            if p.0 < p.1 {
                (NodeIndex::new(p.0), NodeIndex::new(p.1))
            } else {
                (NodeIndex::new(p.1), NodeIndex::new(p.0))
            }
        })
        .collect();
    let mut nodes: Vec<NodeIndex> = connectivity
        .iter()
        .flat_map(|p| vec![p.0, p.1])
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    nodes.sort_unstable();

    let mut sccs: Vec<Vec<NodeIndex>> = vec![nodes];
    while let Some(scc) = sccs.pop() {
        let temp = build_subgraph(&connectivity, &scc);
        let mut subgraph = temp.0;
        let node_map = temp.1;
        let reverse_node_map = node_map.iter().map(|(k, v)| (*v, *k)).collect();
        let start_node = node_map[&scc[0]];

        let mut path: Vec<NodeIndex> = vec![start_node];
        let mut blocked: HashSet<NodeIndex> = path.iter().copied().collect();
        let mut closed: HashSet<NodeIndex> = HashSet::new();
        let mut block: HashMap<NodeIndex, HashSet<NodeIndex>> = HashMap::new();
        let mut stack: Vec<(NodeIndex, IndexSet<NodeIndex, ahash::RandomState>)> = vec![(
            start_node,
            subgraph.neighbors(start_node).collect::<IndexSet<NodeIndex, ahash::RandomState>>()
        )];
        while let Some(res) = process_stack(
            start_node, 
            &mut stack, 
            &mut path, 
            &mut closed, 
            &mut blocked, 
            &mut block, 
            &subgraph, 
            &reverse_node_map,
        ) {
            if !plaquettes.contains(&res) {
                // Because the coupling graph is undirected,
                // the result can be either clockwise and counter-clockwise cycle.
                // Take only one since they are the identical plaquette.
                plaquettes.push(res);
            }
        }
        subgraph.remove_node(start_node);
        sccs.extend(kosaraju_scc(&subgraph).into_iter().filter_map(|scc| {
            // HHL plaquette must be 12 qubit cycle.
            // Ignore smaller subgraph because they don't have any chance to form a plaquette.
            if scc.len() >= 12 {
                let mut res: Vec<NodeIndex> = scc
                    .iter()
                    .map(|n| reverse_node_map[n])
                    .collect();
                res.sort_unstable();
                Some(res)
            } else {
                None
            }
        }));
    }
    plaquettes.sort_unstable_by_key(|plq| plq[0]);
    plaquettes
}


fn build_subgraph(
    connectivity: &HashSet<(NodeIndex, NodeIndex)>,
    scc: &[NodeIndex],
) -> (StableUnGraph<(), ()>, HashMap<NodeIndex, NodeIndex>) {
    let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(scc.len());
    let mut out_graph = StableUnGraph::<(), ()>::with_capacity(scc.len(), connectivity.len());
    for node in scc {
        let new_node = out_graph.add_node(());
        node_map.insert(*node, new_node);
    }
    for pair in connectivity.iter() {
        if scc.contains(&pair.0) & scc.contains(&pair.1) {
            let n0 = *node_map.get(&pair.0).unwrap();
            let n1 = *node_map.get(&pair.1).unwrap();
            out_graph.add_edge(n0, n1, ());
        }
    }
    (out_graph, node_map)
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
                // If block doesn't have stack_node treat it as an empty set
                // (so no updates to stack) and populate it with an empty
                // set.
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
    reverse_node_map: &HashMap<NodeIndex, NodeIndex>,
) -> Option<Vec<usize>> {
    while let Some((this_node, neighbors)) = stack.last_mut() {
        if let Some(next_node) = neighbors.pop() {
            if next_node == start_node {
                if path.len() == 12 {
                    // Out path in input graph basis
                    let mut out_path: Vec<usize> = Vec::with_capacity(path.len());
                    for n in path {
                        out_path.push(reverse_node_map[n].index());
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

    #[test]
    fn test_falcon_map() {
        let coupling_map: Vec<(usize, usize)> = vec![
            (0, 1),
            (1, 0),
            (1, 2),
            (1, 4),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 5),
            (4, 1),
            (4, 7),
            (5, 3),
            (5, 8),
            (6, 7),
            (7, 4),
            (7, 6),
            (7, 10),
            (8, 5),
            (8, 9),
            (8, 11),
            (9, 8),
            (10, 7),
            (10, 12),
            (11, 8),
            (11, 14),
            (12, 10),
            (12, 13),
            (12, 15),
            (13, 12),
            (13, 14),
            (14, 11),
            (14, 13),
            (14, 16),
            (15, 12),
            (15, 18),
            (16, 14),
            (16, 19),
            (17, 18),
            (18, 15),
            (18, 17),
            (18, 21),
            (19, 16),
            (19, 20),
            (19, 22),
            (20, 19),
            (21, 18),
            (21, 23),
            (22, 19),
            (22, 25),
            (23, 21),
            (23, 24),
            (24, 23),
            (24, 25),
            (25, 22),
            (25, 24),
            (25, 26),
            (26, 25),
        ];
        let plaquettes = build_plaquette(coupling_map);
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

}
