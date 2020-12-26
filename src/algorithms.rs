use std::collections::{HashSet, HashMap};
use std::hash::{Hash, Hasher};
use std::fmt::Write as FmtWrite;

use rand::{thread_rng, Rng, seq::SliceRandom};

use howlong::*;

/// A State for the travelling salesman problem (TSP), containing a `path` (partial or complete tour)
/// of an undirected (and, ideally, complete) graph represented by a square adjacency matrix of `u32`s
///
/// The first element of the `path` is the start node, and the last is the current node (or, in the
/// case of a goal State, the final node in a complete tour).
///
/// A goal State for the TSP is one containing a Hamiltonian cycle, a path from the start node that
/// visits every other node exactly once and ends at the start node.
/// E.g., for a graph with four nodes `0`, `1`, `2`, `3`, a possible goal State would be
/// `0 -> 1 -> 2 -> 3 -> 0`.
/// Note that non-complete graphs may not have any Hamiltonian cycle, hence the need for complete
/// graphs.
///
/// The `cost` of a State is the sum of the edges between its nodes. E.g., for a State containing the
/// `path` `0 -> 2 -> 1`, the `cost` would be `adj_matrix[0][2] + adj_matrix[2][1]`.
#[derive(Default, Debug, Clone)]
struct State {
    path: Vec<u32>,
    cost: u32,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl Eq for State {}

impl Hash for State {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state);
    }
}

impl State {
    /// Construct a new State with the given path and its computed cost based on the edges in the
    /// given adjacency matrix
    fn new(path: Vec<u32>, adj_matrix: &Vec<Vec<u32>>) -> std::result::Result<Self, &str> {
        // all states must have at least one node in their path
        if path.is_empty() {
            return Err("States can't have an empty path!");
        }

        let mut new_state = State { path: vec![], cost: 0 };

        for (i, n) in path.iter().enumerate() {
            // check for invalid node (beyond the matrix's bounds)
            if *n as usize >= adj_matrix.len() {
                return Err("Invalid node, out of matrix's bounds!");
            }
            if let Some(last_node) = new_state.path.last() { // if path isn't empty,
                // check for invalid cycle
                if (!new_state.path.contains(n)) ||
                    (
                        // only the first element is permitted to repeat...
                        (n == new_state.path.first().unwrap()) &&
                        // ...provided we're seeing it as the FINAL element left to add
                        (i == path.len() - 1)
                    )
                {
                    // update cost
                    new_state.cost += adj_matrix[*last_node as usize][*n as usize];
                }
                else {
                    return Err("Node already in State path and not a valid Hamiltonian cycle!");
                }
            }
            new_state.path.push(*n);
        }

        Ok(new_state)
    }

    /// Get the State's path
    fn path(&self) -> Vec<u32> {
        self.path.to_vec()
    }

    /// Get the State's cost
    fn cost(&self) -> u32 {
        self.cost
    }

    /// Returns whether this State is a goal State (a Hamiltonian cycle over the given adjacency
    /// matrix)
    fn is_goal(&self, adj_matrix: &Vec<Vec<u32>>) -> bool {
        // all nodes (as indices of adj_matrix) must be in the path
        for i in 0..adj_matrix.len() {
            if !self.path.contains(&(i as u32)) {
                return false;
            }
        }
        // and the path's last element must also be its first
        return self.path.first() == self.path.last();
    }

    /// Return the set of States containing the neighbors of this State's last ("current") node
    /// that are not already in the path (with the exception of the path's first node)
    ///
    /// The sole exception of the first node is so that the the only cyclical paths returned are
    /// Hamiltonian cycles -- i.e., goal States -- which are the only valid cyclical States).
    fn successors(&self, adj_matrix: &Vec<Vec<u32>>) -> Vec<State> {
        let mut successors: Vec<State> = vec![];

        // stop if this state is a goal state
        if self.is_goal(adj_matrix) {
            return successors;
        }

        let current_node = self.path.last().unwrap();
        let first_node = self.path.first().unwrap();

        for neighbor_index in (0..adj_matrix.len()).filter(|&x| x != *current_node as usize) {
            // only non-zero entries represent an edge (and thus a neighbor)
            if adj_matrix[*current_node as usize][neighbor_index] > 0 {
                if (!self.path.contains(&(neighbor_index as u32))) || // only permit acyclic paths...
                    (neighbor_index == (*first_node) as usize) // ...except for apparent Hamiltonian paths
                {
                    // the successor state will be the current state's path + a valid neighbor
                    let mut succ_path = self.path();
                    succ_path.push(neighbor_index as u32);
                    let succ_state = State::new(succ_path, &adj_matrix).unwrap();

                    // reject false Hamiltonian paths (those that can't be valid goal States)
                    if (neighbor_index == (*first_node) as usize) && (!succ_state.is_goal(&adj_matrix)) {
                        continue;
                    }
                    else {
                        successors.push(succ_state);
                    }
                }
            }
        }

        return successors;
    }
}

/// Prints the best path found, its cost, and the number of expansions it made
///
/// # Arguments
///
/// * `adj_matrix` - Adjacency matrix of graph
pub fn hill_climbing(adj_matrix: &Vec<Vec<u32>>) -> String {
    unimplemented!();
}

pub fn simulated_annealing(adj_matrix: &Vec<Vec<u32>>) -> String {
    unimplemented!();
}

pub fn genetic(adj_matrix: &Vec<Vec<u32>>) -> String {
    unimplemented!();
}

/// Runs and times the given algorithm (in both CPU and wall clock time, in seconds) with the
/// given adjacency matrix
///
/// # Arguments
///
/// * `algo` - Algorithm to run
/// * `adj_matrix` - Adjacency matrix of graph
pub fn time_algo(
    algo: fn(&Vec<Vec<u32>>) -> String,
    adj_matrix: &Vec<Vec<u32>>
) -> String {
    // start timers
    let wall_clock = howlong::clock::HighResolutionClock::now();
    let cpu_clock = howlong::clock::ProcessCPUClock::now();

    // run algo!
    let algo_results = algo(&adj_matrix);

    // end timers
    let total_wall_time = (howlong::HighResolutionClock::now() - wall_clock).as_secs();
    let elapsed_cpu_time = howlong::ProcessCPUClock::now() - cpu_clock;
    // colloquially, "cpu time" = user + system time:
    let total_cpu_time = elapsed_cpu_time.user.as_secs() + elapsed_cpu_time.system.as_secs();

    // output results
    let mut output = String::new();

    write!(
        output,
        "{}\nwall time (seconds): {}\ncpu time (seconds):{}",
        algo_results, total_wall_time, total_cpu_time
    ).unwrap();

    return output;
}

#[cfg(test)]
mod state_tests {
    use super::*;

    /// Tests State hash and equality: k1 == k2 -> hash(k1) == hash(k2)
    #[test]
    fn state_hash_eq() {
        use std::collections::hash_map::DefaultHasher;

        // comparison and hashing is path-based; these shouldn't be equal
        let diff_path_0 = State { path: vec![0, 1, 2], cost: 3 };
        let diff_path_1 = State { path: vec![0, 1, 3], cost: 3 };
        let diff_path_2 = State { path: vec![0, 2, 1], cost: 3 };

        assert_ne!(diff_path_0, diff_path_1);
        assert_ne!(diff_path_0, diff_path_2);
        assert_ne!(diff_path_1, diff_path_2);

        let mut hasher = DefaultHasher::new();
        diff_path_0.hash(&mut hasher);
        let hash_0 = hasher.finish();
        let mut hasher = DefaultHasher::new();
        diff_path_1.hash(&mut hasher);
        let hash_1 = hasher.finish();
        let mut hasher = DefaultHasher::new();
        diff_path_2.hash(&mut hasher);
        let hash_2 = hasher.finish();

        assert_ne!(hash_0, hash_1);
        assert_ne!(hash_0, hash_2);
        assert_ne!(hash_1, hash_2);

        // but these should be equal, despite different costs
        let diff_cost_0 = State { path: vec![3, 4, 5], cost: 3 };
        let diff_cost_1 = State { path: vec![3, 4, 5], cost: 4 };

        assert_eq!(diff_cost_0, diff_cost_1);

        let mut hasher = DefaultHasher::new();
        diff_cost_0.hash(&mut hasher);
        let hash_0 = hasher.finish();
        let mut hasher = DefaultHasher::new();
        diff_cost_1.hash(&mut hasher);
        let hash_1 = hasher.finish();

        assert_eq!(hash_0, hash_1);

        // test existence in containers
        let diff_cost_0_copy = State { path: vec![3, 4, 5], cost: 3 };
        let diff_cost_1_copy = State { path: vec![3, 4, 5], cost: 4 };
        assert_eq!(diff_cost_0, diff_cost_0_copy);
        assert_eq!(diff_cost_1, diff_cost_1_copy);

        let mut test_set: HashSet<State> = HashSet::new();

        test_set.insert(diff_cost_0);
        assert!(test_set.contains(&diff_cost_0_copy));
        assert!(test_set.contains(&diff_cost_1));
        assert!(!test_set.contains(&diff_path_0));

        let mut test_vec: Vec<State> = vec![];

        test_vec.push(diff_cost_1);
        assert!(test_vec.contains(&diff_cost_1_copy));
        assert!(test_vec.contains(&diff_cost_0_copy));
        assert!(!test_vec.contains(&diff_path_0));
    }

    /// Tests State constructor from paths
    #[test]
    fn state_ctor() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0] //  3
        ];
        let empty_path_err: std::result::Result<State, &'static str> = Err("States can't have an empty path!");
        let oob_path_err: std::result::Result<State, &'static str> = Err("Invalid node, out of matrix's bounds!");
        let invalid_path_err: std::result::Result<State, &'static str> = Err("Node already in State path and not a valid Hamiltonian cycle!");

        let path_0: Vec<u32> = vec![0, 2, 1];
        let state_0 = State::new(path_0, &adj_matrix).unwrap();
        assert_eq!(state_0.cost(), 72); // cost: 42 + 30 = 72
        assert_eq!(state_0.path(), vec![0, 2, 1]);

        let path_1: Vec<u32> = vec![3];
        let state_1 = State::new(path_1, &adj_matrix).unwrap();
        assert_eq!(state_1.cost(), 0); // cost: 0
        assert_eq!(state_1.path(), vec![3]);

        let path_2: Vec<u32> = vec![];
        assert_eq!(State::new(path_2, &adj_matrix), empty_path_err);

        let path_3: Vec<u32> = vec![1, 1];
        let state_3 = State::new(path_3, &adj_matrix).unwrap();
        assert_eq!(state_3.cost(), 0); // cost: 0
        assert_eq!(state_3.path(), vec![1, 1]);

        let path_4: Vec<u32> = vec![1, 0, 1];
        let state_4 = State::new(path_4, &adj_matrix).unwrap();
        assert_eq!(state_4.cost(), 40); // cost: 20 + 20 = 40
        assert_eq!(state_4.path(), vec![1, 0, 1]);

        let path_5: Vec<u32> = vec![0, 1, 2, 3];
        assert!(State::new(path_5, &adj_matrix).is_ok());

        let path_6: Vec<u32> = vec![0, 1, 2, 3, 0];
        assert!(State::new(path_6, &adj_matrix).is_ok());

        let invalid_path_0: Vec<u32> = vec![1, 1, 1]; // invalid cycle with 1
        assert_eq!(State::new(invalid_path_0, &adj_matrix), invalid_path_err);

        let invalid_path_1: Vec<u32> = vec![0, 0, 0, 0, 0, 0]; // invalid cycle with 0
        assert_eq!(State::new(invalid_path_1, &adj_matrix), invalid_path_err);

        let invalid_path_2: Vec<u32> = vec![0, 1, 2, 3, 1]; // invalid cycle with 1
        assert_eq!(State::new(invalid_path_2, &adj_matrix), invalid_path_err);

        let invalid_path_3: Vec<u32> = vec![0, 1, 2, 3, 1, 0]; // invalid cycle with 1
        assert_eq!(State::new(invalid_path_3, &adj_matrix), invalid_path_err);

        let invalid_path_4: Vec<u32> = vec![0, 1, 2, adj_matrix.len() as u32]; // out of bounds node
        assert_eq!(State::new(invalid_path_4, &adj_matrix), oob_path_err);
    }

    /// Tests State.is_goal()
    #[test]
    fn state_is_goal() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0] //  3
        ];

        // not goals
        let not_a_goal_1 = State::new(vec![1], &adj_matrix).unwrap();
        assert!(!not_a_goal_1.is_goal(&adj_matrix));

        let not_a_goal_2 = State::new(vec![1, 2], &adj_matrix).unwrap();
        assert!(!not_a_goal_2.is_goal(&adj_matrix));

        let not_a_goal_3 = State::new(vec![1, 2, 1], &adj_matrix).unwrap();
        assert!(!not_a_goal_3.is_goal(&adj_matrix));

        let not_a_goal_4 = State::new(vec![1, 2, 3, 1], &adj_matrix).unwrap();
        assert!(!not_a_goal_4.is_goal(&adj_matrix));

        let not_a_goal_5 = State::new(vec![0, 1, 2, 3], &adj_matrix).unwrap();
        assert!(!not_a_goal_5.is_goal(&adj_matrix));

        // goals
        let goal_0 = State::new(vec![1, 2, 3, 0, 1], &adj_matrix).unwrap();
        assert!(goal_0.is_goal(&adj_matrix));

        let goal_1 = State::new(vec![2, 0, 3, 1, 2], &adj_matrix).unwrap();
        assert!(goal_1.is_goal(&adj_matrix));
    }

    /// Tests State.successors()
    #[test]
    fn state_successors_incomplete() {
        let adj_matrix: Vec<Vec<u32>> = vec![ // incomplete graph!
            //   0  1  2   3                  0────────1
            vec![0, 8, 11, 8], // 0           │╲  ╭────╯
            vec![8, 0, 8, 0],  // 1           │ ╰─┼────╮
            vec![11, 8, 0, 8], // 2           │╭──╯    │
            vec![8, 0, 8, 0]   // 3           2────────3
        ];                                    // only for testing; always use complete graphs!

        // 3 successors
        let state = State::new(vec![0], &adj_matrix).unwrap();
        let state_succs = state.successors(&adj_matrix);
        assert_eq!(state_succs.len(), 3);
        assert!(state_succs.contains(&State::new(vec![0, 1], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![0, 2], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![0, 3], &adj_matrix).unwrap()));

        // 2 successors from single node
        let state = State::new(vec![3], &adj_matrix).unwrap();
        let state_succs = state.successors(&adj_matrix);
        assert_eq!(state_succs.len(), 2);
        assert!(state_succs.contains(&State::new(vec![3, 0], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![3, 2], &adj_matrix).unwrap()));

        // 2 successors from two nodes
        let state = State::new(vec![0, 2], &adj_matrix).unwrap();
        let state_succs = state.successors(&adj_matrix);
        assert_eq!(state_succs.len(), 2);
        assert!(state_succs.contains(&State::new(vec![0, 2, 1], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![0, 2, 3], &adj_matrix).unwrap()));

        // 1 successor
        let state = State::new(vec![0, 1, 2], &adj_matrix).unwrap();
        let state_succs = state.successors(&adj_matrix);
        assert_eq!(state_succs.len(), 1);
        assert!(state_succs.contains(&State::new(vec![0, 1, 2, 3], &adj_matrix).unwrap()));

        // 1 successor which is a Hamiltonian path
        let state = State::new(vec![3, 0, 1, 2], &adj_matrix).unwrap();
        let state_succs = state.successors(&adj_matrix);
        assert_eq!(state_succs.len(), 1);
        assert!(state_succs.contains(&State::new(vec![3, 0, 1, 2, 3], &adj_matrix).unwrap()));
        assert!(state_succs[0].is_goal(&adj_matrix));

        // no successors (goal)
        let state = State::new(vec![3, 0, 1, 2, 3], &adj_matrix).unwrap();
        let state_succs = state.successors(&adj_matrix);
        assert_eq!(state_succs.len(), 0);
        assert!(state_succs.is_empty());
    }

    #[test]
    fn state_succesors_complete() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0] //  3
        ];

        // 3 successors
        let state = State::new(vec![2], &adj_matrix).unwrap();
        let state_succs = state.successors(&adj_matrix);
        assert_eq!(state_succs.len(), 3);
        assert!(state_succs.contains(&State::new(vec![2, 0], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![2, 1], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![2, 3], &adj_matrix).unwrap()));

        // 2 successors from two nodes
        let state = State::new(vec![1, 0], &adj_matrix).unwrap();
        let state_succs = state.successors(&adj_matrix);
        assert_eq!(state_succs.len(), 2);
        assert!(state_succs.contains(&State::new(vec![1, 0, 2], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![1, 0, 3], &adj_matrix).unwrap()));

        // 1 successor
        let state = State::new(vec![1, 2, 3], &adj_matrix).unwrap();
        let state_succs = state.successors(&adj_matrix);
        assert_eq!(state_succs.len(), 1);
        assert!(state_succs.contains(&State::new(vec![1, 2, 3, 0], &adj_matrix).unwrap()));

        // 1 successor which is a Hamiltonian path
        let state = State::new(vec![1, 2, 3, 0], &adj_matrix).unwrap();
        let state_succs = state.successors(&adj_matrix);
        assert_eq!(state_succs.len(), 1);
        assert!(state_succs.contains(&State::new(vec![1, 2, 3, 0, 1], &adj_matrix).unwrap()));
        assert!(state_succs[0].is_goal(&adj_matrix));

        // no successors (goal)
        let state = State::new(vec![2, 3, 0, 1, 2], &adj_matrix).unwrap();
        let state_succs = state.successors(&adj_matrix);
        assert_eq!(state_succs.len(), 0);
        assert!(state_succs.is_empty());
    }
}
