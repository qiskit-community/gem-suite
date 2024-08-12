// GEM experiment suite
//
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

use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableUnGraph;


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
pub(super) fn heavyhex_cycle(
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
    use crate::utils::to_undirected;

    #[test]
    fn test_falcon_map() {
        let coupling_map = FALCON_CMAP.lock().unwrap();
        let (qubits, connectivity) = to_undirected(&coupling_map);

        let plaquettes = heavyhex_cycle(&qubits, &connectivity);
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

        let plaquettes = heavyhex_cycle(&qubits, &connectivity);
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
