use std::hash::{Hash, Hasher};
#[allow(unused_imports)]
use std::collections::{HashSet, HashMap};
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use rand::{
    thread_rng, Rng,
    seq::SliceRandom,
    distributions::{WeightedIndex, Distribution},
    rngs::ThreadRng
};

use howlong::*;
use priority_queue::PriorityQueue;

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

impl Display for State {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "path: {:?}", self.path).unwrap();
        write!(f, "cost: {}", self.cost).unwrap();

        Ok(())
    }
}

impl State {
    /// Construct a new State with the given path and its computed cost based on the edges in the
    /// given adjacency matrix
    fn new(path: Vec<u32>, adj_matrix: &[Vec<u32>]) -> std::result::Result<Self, &str> {
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

    /// Get a copy of the State's path
    fn get_path(&self) -> Vec<u32> {
        self.path.to_vec()
    }

    /// Get the State's cost
    fn get_cost(&self) -> u32 {
        self.cost
    }

    /// Returns whether this State is a goal State (a Hamiltonian cycle over the given adjacency
    /// matrix)
    fn is_goal(&self, adj_matrix: &[Vec<u32>]) -> bool {
        // all nodes (as indices of adj_matrix) must be in the path
        for i in 0..adj_matrix.len() {
            if !self.path.contains(&(i as u32)) {
                return false
            }
        }
        // and the path's last element must also be its first
        self.path.first() == self.path.last()
    }

    /// Return the set of States containing the neighbors of this State's last ("current") node
    /// that are not already in the path (with the exception of the path's first node)
    ///
    /// The sole exception of the first node is so that the the only cyclical paths returned are
    /// Hamiltonian cycles -- i.e., goal States -- which are the only valid cyclical States).
    fn get_successors(&self, adj_matrix: &[Vec<u32>]) -> Vec<State> {
        let mut successors: Vec<State> = vec![];

        // stop if this state is a goal state
        if self.is_goal(adj_matrix) {
            return successors
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
                    let mut succ_path = self.get_path();
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

        successors
    }

    /// Measures the cost of the State's path were the given swap of its elements (by index)
    /// applied and returns the difference between the cost of the new path and the cost of the
    /// original path
    ///
    /// E.g., if the original path has a cost of 5 and the proposed swap makes the path cost 3, the
    /// returned value is -2.
    ///
    /// Returns an Err if the swap is invalid (attempting to swap the first or last node, swapping
    /// the same element with itself, or swapping invalid elements)
    ///
    /// # Arguments
    ///
    /// * `i` - Index of element to swap with element at index `j`
    /// * `j` - Index of element to swap with element at index `i`
    /// * `adj_matrix` - Adjacency matrix of the graph
    fn weigh_swap(&self, (i, j): (usize, usize), adj_matrix: &[Vec<u32>]) -> std::result::Result<i32, &str> {
        if (i > 0) &&
           (i < (self.path.len() - 1)) &&
           (j > 0) &&
           (j < (self.path.len() - 1)) &&
           (i != j) &&
           (self.path.len() > 1) {
               let mut swapped_cost = 0;
               let mut swapped_path = self.get_path();

               swapped_path.swap(i, j);

               for k in 0..(swapped_path.len() - 1) {
                   swapped_cost += adj_matrix[swapped_path[k] as usize][swapped_path[k + 1] as usize];
               }

               Ok(swapped_cost as i32 - self.cost as i32)
        }
        else {
            Err("Invalid swap!")
        }
    }

    /// Carries out the given swap of elements (by index) in this State's path
    ///
    /// Returns an Err if the swap is invalid (attempting to swap the first or last node, swapping
    /// the same element with itself, or swapping invalid elements)
    ///
    /// # Arguments
    ///
    /// * `i` - Index of element to swap with element at index `j`
    /// * `j` - Index of element to swap with element at index `i`
    /// * `adj_matrix` - Adjacency matrix of the graph
    fn do_swap(&mut self, (i, j): (usize, usize), adj_matrix: &[Vec<u32>]) -> std::result::Result<(), &str> {
        if (i > 0) &&
           (i < (self.path.len() - 1)) &&
           (j > 0) &&
           (j < (self.path.len() - 1)) &&
           (i != j) &&
           (self.path.len() > 1) {
            self.path.swap(i, j);

            // recompute path cost
            self.cost = 0;
            for k in 0..(self.path.len() - 1) {
                self.cost += adj_matrix[self.path[k] as usize][self.path[k + 1] as usize];
            }

            Ok(())
        }
        else {
            Err("Invalid swap!")
        }
    }

    /// Returns the index tuple of the two elements in this State's path whose swap reduces the
    /// path's cost the most and None if no swaps reduce the cost
    ///
    /// # Arguments
    ///
    /// * `adj_matrix` - Adjacency matrix of the graph
    fn find_best_swap(&self, adj_matrix: &[Vec<u32>]) -> Option<(usize, usize)> {
        let mut max_reduction: i32 = 0;
        let mut swap_pair: (usize, usize) = (0, 0);

        for i in 1..(self.path.len() - 1) {           // go from left to right across the path to
            for j in (i + 1)..(self.path.len() - 1) { // avoid checking (3,1) after checking (1,3)
                let reduction = self.weigh_swap((i, j), adj_matrix).unwrap();

                if (reduction < 0) && (reduction < max_reduction) {
                    max_reduction = reduction;
                    swap_pair = (i, j);
                }
            }
        }

        if max_reduction < 0 {
            Some(swap_pair)
        }
        else {
            None
        }
    }
}

/// Returns a random Hamiltonian cycle (a tour beginning and ending at some node in
/// the graph and visiting every other node exactly once) as a State, assuming the graph is complete
///
/// # Arguments
///
/// * `adj_matrix` - Adjacency matrix of the graph
fn random_hamiltonian_cycle(adj_matrix: &[Vec<u32>]) -> std::result::Result<State, &str> {
    if adj_matrix.len() < 2 {
        return Err("Adjacency matrix too small!")
    }
    else {
        for i in 0..adj_matrix.len() {
            if adj_matrix[i].len() != adj_matrix.len() {
                return Err("Adjacency matrix not square!")
            }
        }
    }

    let n = adj_matrix.len();
    let mut cycle = vec![0; n];
    let mut rng = rand::thread_rng();

    for (i, elt) in cycle.iter_mut().enumerate() {
        *elt = i as u32;
    }
    cycle.shuffle(&mut rng);
    cycle.push(cycle[0]); // return to start node

    State::new(cycle, adj_matrix)
}

/// Returns the best Hamiltonian cycle, and its cost, found via hill climbing
///
/// In this implementation, the algorithm starts with a random Hamiltonian cycle in the graph and
/// examines each possible swap of two elements among the elements between the first and last in
/// the cycle. Among the swaps that decrease the cycle cost, it carries out the swap that decreases
/// it the most. It then repeats this until no swaps decrease the cost anymore, at which point it
/// stops.
///
/// # Arguments
///
/// * `adj_matrix` - Adjacency matrix of graph
pub fn hill_climbing(adj_matrix: &[Vec<u32>]) -> String {
    // start timers
    let wall_clock = howlong::clock::HighResolutionClock::now();
    let cpu_clock = howlong::clock::ProcessCPUClock::now();

    // run algo
    let mut s: State = random_hamiltonian_cycle(adj_matrix).unwrap();

    while let Some(best_candidate) = s.find_best_swap(adj_matrix) {
        s.do_swap(best_candidate, adj_matrix).unwrap();
    }

    // end timers
    let total_wall_time = (howlong::HighResolutionClock::now() - wall_clock).as_secs();
    let elapsed_cpu_time = howlong::ProcessCPUClock::now() - cpu_clock;
    // colloquially, "cpu time" = user + system time:
    let total_cpu_time = elapsed_cpu_time.user.as_secs() + elapsed_cpu_time.system.as_secs();

    // output end state
    format!(
        "{}\nwall time (seconds): {}\ncpu time (seconds): {}",
        s, total_wall_time, total_cpu_time
    )
}

/// Returns the best Hamiltonian cycle, and its cost, found via simulated annealing
///
/// The algorithm starts with a random Hamiltonian cycle in the graph and a preset temperature and
/// cooling rate. The cooling rate determines how fast the temperature "cools down". Until the
/// temperature cools to a certain point, random swaps of the elements in the cycle between the
/// first and last element are attempted. When a swap decreases the cost, it's carried out. When
/// it increases the cost, it's accepted with a certain probability related to the magnitude of the
/// increase and the current temperature (less likely if the increase is a lot, yet more likely if
/// the temperature is high).
///
/// Smaller cooling rates increase run time and accuracy significantly, with eventual diminishing returns.
///
/// Larger temperatures increase run time significantly and accuracy by relatively little (although
/// ones too low begin to degrade accuracy significantly).
///
/// # Arguments
///
/// * `adj_matrix` - Adjacency matrix of graph
/// * `cooling_rate` - How fast `temperature` decreases per iteration
/// * `temperature` - Starting temperature
pub fn simulated_annealing(
    adj_matrix: &[Vec<u32>],
    cooling_rate: f64,
    mut temperature: f64
) -> String {
    // start timers
    let wall_clock = howlong::clock::HighResolutionClock::now();
    let cpu_clock = howlong::clock::ProcessCPUClock::now();

    // run algo
    let mut s: State = random_hamiltonian_cycle(adj_matrix).unwrap();
    let n: usize = adj_matrix.len();

    let mut rng = thread_rng();
    let mut delta: i32;

    loop {
        temperature *= 1.0 - cooling_rate; // lower temperature
        // stop if cooled down sufficiently
        if temperature < 1.0 {
            // end timers
            let total_wall_time = (howlong::HighResolutionClock::now() - wall_clock).as_secs();
            let elapsed_cpu_time = howlong::ProcessCPUClock::now() - cpu_clock;
            let total_cpu_time = elapsed_cpu_time.user.as_secs() + elapsed_cpu_time.system.as_secs();

            // output end state
            return format!(
                "{}\nwall time (seconds): {}\ncpu time (seconds): {}",
                s, total_wall_time, total_cpu_time
            )
        }

        let i = rng.gen_range(1..(n - 1));
        let mut j;
        loop {
            j = rng.gen_range(1..(n - 1));

            if j != i { break; }
        }

        delta = s.weigh_swap((i, j), adj_matrix).unwrap(); // delta = next - current
        if delta < 0 { // if the swap is an improvement (a descent in cost, i.e. delta negative),
            // accept it
            s.do_swap((i, j), adj_matrix).unwrap();
        }
        else { // otherwise,
            // accept it with probability p = e^(-delta / temperature)
            let p = std::f64::consts::E.powf(-delta as f64 / temperature as f64);
            // delta, being positive here, needs to be negated so that the exponential is in [0, 1)
            // this uses the Boltzmann distribution to gradually settle on the final (and hopefully
            // global) minimum

            if rng.gen_bool(p) {
                s.do_swap((i, j), adj_matrix).unwrap();
            }
        }
    }
}

/// Populates the `fitnesses` vector with the "fitness values" in the range [0.0, 1.0] of the States
/// in the `population` vector
///
/// The lower a State's path cost, the higher its "fitness" (the closer to 1.0).
///
/// This is done via inverse normalization, or "inverse softmax". For each cost `i`, compute `1 / i`,
/// then divide each `1 / i` by the sum of the `1 / i`s. (Thanks, Keeks! 💙)
///
/// E.g., the following four states have the accompanying fitness values:
///
/// ```
/// state    cost    fitness
/// a        31      0.052
/// b        56      0.029
/// c        2       0.811
/// d        15      0.108
/// ```
///
/// # Arguments
///
/// * `population` - States whose fitness to calculate
/// * `fitnesses` - Output vector in which to save fitness values
fn calculate_fitnesses(population: &Vec<State>, fitnesses: &mut Vec<f32>) {
    *fitnesses = population.iter().map(|x| 1.0 / x.get_cost() as f32).collect();
    let fitness_sum: f32 = fitnesses.iter().fold(0.0, |acc, x| acc + x);
    *fitnesses = fitnesses.iter().map(|&x| x / fitness_sum).collect();
}

/// Returns references to two States from `population` selected for mating
///
/// The two States are never the same State, and each is selected based on its corresponding fitness
/// in `fitnesses`
///
/// # Arguments
///
/// * `population` - States to select for mating
/// * `fitnesses` - Fitness values for States in `population` affecting their chance of mate selection
/// * `rng` - Random number generator, instantiated via `thread_rng()`
fn select_mating_pair<'a>(
    population: &'a Vec<State>,
    fitnesses: &Vec<f32>,
    rng: &mut ThreadRng
) -> (&'a State, &'a State) {
    let dist = WeightedIndex::new(fitnesses).unwrap();

    let parent_a = &population[dist.sample(rng)];
    let mut parent_b;

    loop {
        parent_b = &population[dist.sample(rng)];

        if parent_b != parent_a { break; }
    }

    (parent_a, parent_b)
}

/// Returns the "child" State of States `parent_a` and `parent_b`
///
/// Child path starts as a copy of one parent's path and is then modified by an element swap to
/// more closely match the other parent's.
///
/// E.g., if `parent_b`'s path was chosen as the starting path and the parents differ in terminal
/// node, the child's path is `parent_b`'s swapped so that it has `parent_a`'s terminal node:
///
/// ```
/// parent_a: [0, 3, 2, 1, 0]
/// parent_b: [2, 3, 1, 0, 2]
///            ^        &  ^
///    child: [0, 3, 1, 2, 0]
///            &        ^  &
/// ```
///
/// If, however, the parents have the same terminal node, the first differing element in
/// `parent_b`'s path, call it "first_diff", is swapped with whatever element is in `parent_b`'s
/// path at the index of "first_diff" in `parent_a`:
///
/// ```
///    index:  0  1  2  3  4  5  6
/// parent_a: [1, 2, 3, 4, 5, 6, 1]
/// parent_b: [1, 2, 6, 4, 3, 5, 1]
///                  ^        &
///    child: [1, 2, 5, 4, 3, 6, 1]
///                  &        ^
/// ```
///
/// Since `6` was "first_diff", and since `6` appears at index `5` in `parent_a`, `6` is swapped with
/// the element at index `5` in `parent_b`, which happens to be `5`. In a slight way, the child
/// is now a combination of `parent_a` and `parent_b`. Kind of.
///
/// # Arguments
///
/// * `parent_a` - First parent
/// * `parent_b` - Second parent
/// * `rng` - Random number generator, instantiated via `thread_rng()`
/// * `adj_matrix` - Adjacency matrix of graph
fn mate(
    parent_a: &State,
    parent_b: &State,
    rng: &mut ThreadRng,
    adj_matrix: &[Vec<u32>]
) -> State {
    let path_a: Vec<u32>;
    let path_b: Vec<u32>;

    // coin flip to randomly determine starting parent
    if rng.gen_bool(0.5) {
        path_a = parent_a.get_path();
        path_b = parent_b.get_path();
    }
    else {
        path_a = parent_b.get_path();
        path_b = parent_a.get_path();
    }

    let n = path_a.len();
    let mut path_child: Vec<u32> = path_b.to_vec(); // use path_b as the basic template

    // lame (but much faster!) reproduction logic xD
    // same terminal nodes
    if path_a[0] == path_b[0] {
        for i in 1..(n - 1) {
            if path_a[i] != path_b[i] {
                path_child.swap(i, path_a.iter().position(|&x| x == path_b[i]).unwrap());
                break;
            }
        }
    }
    // different terminal nodes
    else {
        path_child.swap(0, path_b.iter().position(|&x| x == path_a[0]).unwrap()); // first terminal node
        path_child[n - 1] = path_child[0];                                        // last terminal node
    }

    State::new(path_child, adj_matrix).unwrap()
}

// A common mating logic is the one below, where half of one parent's "DNA" is retained,
// and the remaining "nucleotides" (here, element nodes) are filled in.
// This takes much longer than the mating logic above since it fully traverses both parent paths,
// and maybe even returns slightly worse results (it's close)! D:
// fn mate(
//     parent_a: &State,
//     parent_b: &State,
//     rng: &mut ThreadRng,
//     adj_matrix: &[Vec<u32>]
// ) -> State {
//     let path_a: Vec<u32>;
//     let path_b: Vec<u32>;
//     // coin flip to randomly determine starting parent
//     if rng.gen_bool(0.5) {
//         path_a = parent_a.get_path();
//         path_b = parent_b.get_path();
//     }
//     else {
//         path_a = parent_b.get_path();
//         path_b = parent_a.get_path();
//     }
//     let n = path_a.len();
//     let mut path_child: Vec<u32> = Vec::with_capacity(n);
//
//     // fill with half of first parent
//     for i in 0..(n / 2) {
//         path_child.push(path_a[i]);
//     }
//
//     // fill rest with in-order, non-present elements of the other parent
//     for n in path_b.iter() {
//         if !path_child.contains(n) {
//             path_child.push(*n);
//         }
//     }
//     path_child.push(path_child[0]); // cap with terminal node
//
//     State::new(path_child, adj_matrix).unwrap()
// }

/// Mutates an individual State via a random swap of its non-terminal elements
///
/// # Arguments
///
/// * `individual` - The State to mutate
/// * `rng` - Random number generator, instantiated via `thread_rng()`
/// * `adj_matrix` - Adjacency matrix of graph
fn mutate(
    individual: &mut State,
    rng: &mut ThreadRng,
    adj_matrix: &[Vec<u32>]
) {
    let i: usize = rng.gen_range(1..(individual.get_path().len() - 1));
    let mut j: usize;

    loop {
        j = rng.gen_range(1..(individual.get_path().len() - 1));

        if j != i { break; }
    }
    individual.do_swap((i, j), adj_matrix).unwrap();
}

/// Returns a reference to the fittest (cheapest path) State in `population`
///
/// # Arguments
///
/// * `population` - The States to select from
fn fittest_individual(population: &Vec<State>) -> &State {
    let mut cheapest: &State = &population[0];

    for i in 1..population.len() {
        if population[i].get_cost() < cheapest.get_cost() {
            cheapest = &population[i];
        }
    }

    cheapest
}

/// Returns the best Hamiltonian cycle, and its cost, found via genetic algorithm
///
/// # Arguments
///
/// * `adj_matrix` - Adjacency matrix of graph
/// * `population_size` - Number of individuals in each population (an even number that's at least 2)
/// * `max_generations` - How many generations to simulate before stopping
/// * `mutation_rate` - A probability in [0.0, 1.0] of an offspring having a random mutation
///
/// # Panics
///
/// Panics if:
///
/// * `population_size` is not an even number that's at least 2
/// * `max_generations` is not greater than 0
/// * `mutation_rate` is not in [0.0, 1.0]
pub fn genetic(
    adj_matrix: &[Vec<u32>],
    population_size: usize,
    max_generations: u32,
    mutation_rate: f32
) -> String {
    if population_size % 2 != 0 || population_size < 2 {
        panic!("population_size must be an even number that's at least 2!");
    }
    else if max_generations <= 0 {
        panic!("max_generations must be greater than 0!");
    }
    else if mutation_rate < 0.0 || mutation_rate > 1.0 {
        panic!("mutation_rate must be between 0.0 and 1.0!");
    }

    // start timers
    let wall_clock = howlong::clock::HighResolutionClock::now();
    let cpu_clock = howlong::clock::ProcessCPUClock::now();

    // run algo
    let mut rng = thread_rng();
    let mut population: Vec<State> = Vec::with_capacity(population_size); // candidate cycles are individuals
    let mut fitnesses: Vec<f32> = Vec::with_capacity(population_size); // based on individuals' costs

    // randomly-generate starting population
    for _ in 0..population_size {
        population.push(random_hamiltonian_cycle(adj_matrix).unwrap());
    }
    // simulate all generations
    for _ in 0..max_generations {
        let mut next_generation: Vec<State> = Vec::with_capacity(population_size);

        calculate_fitnesses(&population, &mut fitnesses);
        for _ in 0..population_size {
            // select two different individuals to reproduce, based on their fitness
            let (parent_a, parent_b): (&State, &State) = select_mating_pair(&population, &fitnesses, &mut rng);
            let mut child: State = mate(parent_a, parent_b, &mut rng, adj_matrix);

            if rng.gen_bool(mutation_rate as f64) { // mutate probabilistically
                mutate(&mut child, &mut rng, adj_matrix);
            }
            next_generation.push(child); // add to new generation
        }
        population = next_generation; // replace old population with new generation
    }
    // recalculate fitnesses to determine final generation's fittest individual
    calculate_fitnesses(&population, &mut fitnesses);

    // end timers
    let total_wall_time = (howlong::HighResolutionClock::now() - wall_clock).as_secs();
    let elapsed_cpu_time = howlong::ProcessCPUClock::now() - cpu_clock;
    let total_cpu_time = elapsed_cpu_time.user.as_secs() + elapsed_cpu_time.system.as_secs();

    // output end state
    return format!(
        "{}\nwall time (seconds): {}\ncpu time (seconds): {}",
        fittest_individual(&population), total_wall_time, total_cpu_time
    )
}

/// Returns the best Hamiltonian cycle, and its cost, found via A*
///
/// # Arguments
///
/// * `adj_matrix` - Adjacency matrix of graph
/// * `heuristic` - Heuristic, `h(s)`, for a State `s` to use in computing `f(s) = g(s) + h(s)`
fn a_star(
    adj_matrix: &[Vec<u32>],
    heuristic: fn(&State, &[Vec<u32>], &HashSet<Rc<State>>) -> i32
) -> String {
    // start timers
    let wall_clock = howlong::clock::HighResolutionClock::now();
    let cpu_clock = howlong::clock::ProcessCPUClock::now();

    // run algo
    let initial_state: Rc<State>;
    let mut frontier = PriorityQueue::new();
    let mut explored = HashSet::new();
    let mut in_frontier = HashMap::new(); // since lookup isn't a thing in priority queues
    let f = |s: &State, adj: &[Vec<u32>], exp: &HashSet<Rc<State>>| { // f(s) = g(s) + h(s)
        // invert result to get PriorityQueue to act as a min heap
        -(s.get_cost() as i32 + heuristic(s, adj, exp))
    };
    let mut solution: Option<State> = None;

    // initial state is a single, random node path
    let rand_node = thread_rng().gen_range(0..adj_matrix.len()) as u32;

    initial_state = Rc::new(State::new(vec![rand_node], adj_matrix).unwrap());
    let f_eval = f(&initial_state, &adj_matrix, &explored);

    frontier.push(Rc::clone(&initial_state), f_eval);
    in_frontier.insert(initial_state, f_eval);

    while !frontier.is_empty() {
        let (curr, _) = frontier.pop().unwrap();
        in_frontier.remove(&curr).unwrap();

        // stop if we've expanded a goal State
        if curr.is_goal(adj_matrix) {
            solution = Some(Rc::try_unwrap(curr).unwrap());
            break;
        }

        explored.insert(curr.clone());
        for succ in curr.get_successors(adj_matrix) {
            let succ_r = Rc::new(succ);

            if !explored.contains(&succ_r) && !in_frontier.contains_key(&succ_r) {
                let f_eval = f(&succ_r, &adj_matrix, &explored);
                frontier.push(Rc::clone(&succ_r), f_eval);
                in_frontier.insert(succ_r, f_eval);
            }
            else if in_frontier.contains_key(&succ_r) {
                let f_eval = f(&succ_r, &adj_matrix, &explored);

                if f_eval > *in_frontier.get(&succ_r).unwrap() {
                    // reinsert to update f_eval priority in frontier and value in in_frontier
                    frontier.push(Rc::clone(&succ_r), f_eval).unwrap();
                    in_frontier.insert(succ_r, f_eval).unwrap();
                }
            }
        }
    }

    // end timers
    let total_wall_time = (howlong::HighResolutionClock::now() - wall_clock).as_secs();
    let elapsed_cpu_time = howlong::ProcessCPUClock::now() - cpu_clock;
    let total_cpu_time = elapsed_cpu_time.user.as_secs() + elapsed_cpu_time.system.as_secs();

    // output goal state
    return format!(
        "{}\nwall time (seconds): {}\ncpu time (seconds): {}",
        solution.unwrap(), total_wall_time, total_cpu_time
    )
}

/// Run uniform cost search (as A* with the zero heuristic) and return the best Hamiltonian cycle
/// and its cost
///
/// This heuristic is trivially admissable (it values everything as 0, so it's in no danger of
/// overestimating their true cost), so A* will return the optimal solution when using it. However,
/// it's a *very bad* admissable heuristic because it provides no guidance ("Everything's 0;
/// go anywhere, A*! 🤷"), so the runtime will be very slow.
pub fn a_star_uniform_cost(adj_matrix: &[Vec<u32>]) -> String {
    let zero_heuristic = |
        _s: &State,
        _adj_matrix: &[Vec<u32>],
        _explored: &HashSet<Rc<State>>
    | -> i32 { 0 };

    return a_star(adj_matrix, zero_heuristic);
}

/// Run A* with a "grab random edges" heuristic and return the best Hamiltonian cycle and its cost,
/// found via genetic algorithm
///
/// The heuristic picks a random edge from each node that's still unvisited, sums up all those
/// random edges, and returns that sum.
///
/// (It's a poor attempt to guess the remaining distance in the path of State `s` which can easily
/// be wrong and overestimate, so this heuristic is not admissable. A* is not guaranteed to return
/// an optimal solution using this heuristic.)
pub fn a_star_random_edge(adj_matrix: &[Vec<u32>]) -> String {
    let random_heuristic = |
        s: &State,
        adj_matrix: &[Vec<u32>],
        _explored: &HashSet<Rc<State>>
    | -> i32 {
        let mut rng = thread_rng();
        let mut sum: i32 = 0;

        for unvisited in (0..adj_matrix.len()).filter(|&x| !s.get_path().contains(&(x as u32))) {
            let rand_edge = rng.gen_range(0..adj_matrix.len());

            sum += adj_matrix[unvisited][rand_edge] as i32;
        }

        sum
    };

    return a_star(adj_matrix, random_heuristic);
}

/// Run A* with a "grab cheapest remaining edges" heuristic and return the best Hamiltonian cycle
/// and its cost
///
/// The heuristic picks the cheapest remaining edge of each node that's still unvisited, sums up
/// all those cheapest edges, and returns that sum. It is admissable, so A* will return the optimal
/// solution.
pub fn a_star_cheapest_edge(adj_matrix: &[Vec<u32>]) -> String {
    let cheapest_heuristic = |
        s: &State,
        adj_matrix: &[Vec<u32>],
        _explored: &HashSet<Rc<State>>
    | -> i32 {
        (0..adj_matrix.len()).filter(|&x| !s.get_path().contains(&(x as u32)))
                             .fold(0, |sum, i| sum + *adj_matrix[i].iter()
                                                                  .filter(|&&x| x > 0) // avoid non-edges
                                                                  .min()
                                                                  .unwrap() as i32)
    };

    return a_star(adj_matrix, cheapest_heuristic);
}

#[cfg(test)]
mod state_tests {
    use super::*;

    /// Tests State hash and equality: k1 == k2 -> hash(k1) == hash(k2)
    #[test]
    fn hash_eq() {
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
    fn ctor() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0]  // 3
        ];
        let empty_path_err: std::result::Result<State, &'static str> = Err("States can't have an empty path!");
        let oob_path_err: std::result::Result<State, &'static str> = Err("Invalid node, out of matrix's bounds!");
        let invalid_path_err: std::result::Result<State, &'static str> = Err("Node already in State path and not a valid Hamiltonian cycle!");

        let path_0: Vec<u32> = vec![0, 2, 1];
        let state_0 = State::new(path_0, &adj_matrix).unwrap();
        assert_eq!(state_0.get_cost(), 72); // cost: 42 + 30 = 72
        assert_eq!(state_0.get_path(), vec![0, 2, 1]);

        let path_1: Vec<u32> = vec![3];
        let state_1 = State::new(path_1, &adj_matrix).unwrap();
        assert_eq!(state_1.get_cost(), 0); // cost: 0
        assert_eq!(state_1.get_path(), vec![3]);

        let path_2: Vec<u32> = vec![];
        assert_eq!(State::new(path_2, &adj_matrix), empty_path_err);

        let path_3: Vec<u32> = vec![1, 1];
        let state_3 = State::new(path_3, &adj_matrix).unwrap();
        assert_eq!(state_3.get_cost(), 0); // cost: 0
        assert_eq!(state_3.get_path(), vec![1, 1]);

        let path_4: Vec<u32> = vec![1, 0, 1];
        let state_4 = State::new(path_4, &adj_matrix).unwrap();
        assert_eq!(state_4.get_cost(), 40); // cost: 20 + 20 = 40
        assert_eq!(state_4.get_path(), vec![1, 0, 1]);

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
    fn is_goal() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0]  // 3
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

    /// Tests State.get_successors()
    #[test]
    fn successors_incomplete() {
        let adj_matrix: Vec<Vec<u32>> = vec![ // incomplete graph!
            //   0  1  2   3                  0────────1
            vec![0, 8, 11, 8], // 0           │╲  ╭────╯
            vec![8, 0, 8, 0],  // 1           │ ╰─┼────╮
            vec![11, 8, 0, 8], // 2           │╭──╯    │
            vec![8, 0, 8, 0]   // 3           2────────3
        ];                                    // only for testing; always use complete graphs!

        // 3 successors
        let state = State::new(vec![0], &adj_matrix).unwrap();
        let state_succs = state.get_successors(&adj_matrix);
        assert_eq!(state_succs.len(), 3);
        assert!(state_succs.contains(&State::new(vec![0, 1], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![0, 2], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![0, 3], &adj_matrix).unwrap()));

        // 2 successors from single node
        let state = State::new(vec![3], &adj_matrix).unwrap();
        let state_succs = state.get_successors(&adj_matrix);
        assert_eq!(state_succs.len(), 2);
        assert!(state_succs.contains(&State::new(vec![3, 0], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![3, 2], &adj_matrix).unwrap()));

        // 2 successors from two nodes
        let state = State::new(vec![0, 2], &adj_matrix).unwrap();
        let state_succs = state.get_successors(&adj_matrix);
        assert_eq!(state_succs.len(), 2);
        assert!(state_succs.contains(&State::new(vec![0, 2, 1], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![0, 2, 3], &adj_matrix).unwrap()));

        // 1 successor
        let state = State::new(vec![0, 1, 2], &adj_matrix).unwrap();
        let state_succs = state.get_successors(&adj_matrix);
        assert_eq!(state_succs.len(), 1);
        assert!(state_succs.contains(&State::new(vec![0, 1, 2, 3], &adj_matrix).unwrap()));

        // 1 successor which is a Hamiltonian path
        let state = State::new(vec![3, 0, 1, 2], &adj_matrix).unwrap();
        let state_succs = state.get_successors(&adj_matrix);
        assert_eq!(state_succs.len(), 1);
        assert!(state_succs.contains(&State::new(vec![3, 0, 1, 2, 3], &adj_matrix).unwrap()));
        assert!(state_succs[0].is_goal(&adj_matrix));

        // no successors (goal)
        let state = State::new(vec![3, 0, 1, 2, 3], &adj_matrix).unwrap();
        let state_succs = state.get_successors(&adj_matrix);
        assert_eq!(state_succs.len(), 0);
        assert!(state_succs.is_empty());
    }

    #[test]
    fn succesors_complete() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0]  // 3
        ];

        // 3 successors
        let state = State::new(vec![2], &adj_matrix).unwrap();
        let state_succs = state.get_successors(&adj_matrix);
        assert_eq!(state_succs.len(), 3);
        assert!(state_succs.contains(&State::new(vec![2, 0], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![2, 1], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![2, 3], &adj_matrix).unwrap()));

        // 2 successors from two nodes
        let state = State::new(vec![1, 0], &adj_matrix).unwrap();
        let state_succs = state.get_successors(&adj_matrix);
        assert_eq!(state_succs.len(), 2);
        assert!(state_succs.contains(&State::new(vec![1, 0, 2], &adj_matrix).unwrap()));
        assert!(state_succs.contains(&State::new(vec![1, 0, 3], &adj_matrix).unwrap()));

        // 1 successor
        let state = State::new(vec![1, 2, 3], &adj_matrix).unwrap();
        let state_succs = state.get_successors(&adj_matrix);
        assert_eq!(state_succs.len(), 1);
        assert!(state_succs.contains(&State::new(vec![1, 2, 3, 0], &adj_matrix).unwrap()));

        // 1 successor which is a Hamiltonian path
        let state = State::new(vec![1, 2, 3, 0], &adj_matrix).unwrap();
        let state_succs = state.get_successors(&adj_matrix);
        assert_eq!(state_succs.len(), 1);
        assert!(state_succs.contains(&State::new(vec![1, 2, 3, 0, 1], &adj_matrix).unwrap()));
        assert!(state_succs[0].is_goal(&adj_matrix));

        // no successors (goal)
        let state = State::new(vec![2, 3, 0, 1, 2], &adj_matrix).unwrap();
        let state_succs = state.get_successors(&adj_matrix);
        assert_eq!(state_succs.len(), 0);
        assert!(state_succs.is_empty());
    }

    #[test]
    fn weigh_swap() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0]  // 3
        ];
                                           // i: 0  1  2  3  4
        let s_goal_path: State = State::new(vec![0, 1, 3, 2, 0], &adj_matrix).unwrap();
        assert_eq!(s_goal_path.get_cost(), 108);

        // swapping 1 and 2 yields a path costing 141, which is a delta of 141 - 108 = 33
        assert_eq!(s_goal_path.weigh_swap((1, 2), &adj_matrix).unwrap(), 33);
        assert_eq!(s_goal_path.weigh_swap((2, 1), &adj_matrix).unwrap(), 33);

        // swapping 2 and 3 yields 97, which is a delta of -11
        assert_eq!(s_goal_path.weigh_swap((2, 3), &adj_matrix).unwrap(), -11);
        assert_eq!(s_goal_path.weigh_swap((3, 2), &adj_matrix).unwrap(), -11);

        // swapping 1 and 3 yields the same cost as before, 108, for a delta of 0
        assert_eq!(s_goal_path.weigh_swap((1, 3), &adj_matrix).unwrap(), 0);
        assert_eq!(s_goal_path.weigh_swap((3, 1), &adj_matrix).unwrap(), 0);

                                      // i: 0  1  2  3
        let s_path: State = State::new(vec![0, 2, 3, 0], &adj_matrix).unwrap();
        assert_eq!(s_path.get_cost(), 89);

        assert_eq!(s_path.weigh_swap((1, 2), &adj_matrix).unwrap(), 0);
        assert_eq!(s_path.weigh_swap((2, 1), &adj_matrix).unwrap(), 0);
    }

    #[test]
    fn weigh_swap_bad() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0]  // 3
        ];
        let s: State = State::new(vec![0, 1, 3, 2, 0], &adj_matrix).unwrap();

        // invalid elts (terminal nodes)
        assert!(s.weigh_swap((0, 1), &adj_matrix).is_err());
        assert!(s.weigh_swap((3, 4), &adj_matrix).is_err());
        assert!(s.weigh_swap((4, 3), &adj_matrix).is_err());
        assert!(s.weigh_swap((4, 0), &adj_matrix).is_err());

        // out of bounds
        assert!(s.weigh_swap((1, 5), &adj_matrix).is_err());
        assert!(s.weigh_swap((6, 2), &adj_matrix).is_err());

        // same elt
        assert!(s.weigh_swap((2, 2), &adj_matrix).is_err());
        assert!(s.weigh_swap((0, 0), &adj_matrix).is_err());

        // 1-elt state path
        let s: State = State::new(vec![0], &adj_matrix).unwrap();

        assert!(s.weigh_swap((0, 0), &adj_matrix).is_err());
        assert!(s.weigh_swap((1, 2), &adj_matrix).is_err());

        // 2-elt state path
        let s: State = State::new(vec![0, 1], &adj_matrix).unwrap();

        assert!(s.weigh_swap((0, 1), &adj_matrix).is_err());
        assert!(s.weigh_swap((1, 0), &adj_matrix).is_err());
    }

    #[test]
    fn do_swap() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0]  // 3
        ];
        let mut s: State = State::new(vec![0, 1, 3, 2, 0], &adj_matrix).unwrap();

        assert_eq!(s.get_cost(), 108);
        assert_eq!(s.get_path(), vec![0, 1, 3, 2, 0]);

        s.do_swap((1, 2), &adj_matrix).unwrap();

        assert_eq!(s.get_cost(), 141);
        assert_eq!(s.get_path(), vec![0, 3, 1, 2, 0]);
    }

    #[test]
    fn do_swap_bad() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0]  // 3
        ];
        let mut s: State = State::new(vec![0, 1, 3, 2, 0], &adj_matrix).unwrap();

        // invalid elts (terminal nodes)
        assert!(s.do_swap((0, 1), &adj_matrix).is_err());
        assert!(s.do_swap((3, 4), &adj_matrix).is_err());
        assert!(s.do_swap((4, 3), &adj_matrix).is_err());
        assert!(s.do_swap((4, 0), &adj_matrix).is_err());

        // out of bounds
        assert!(s.do_swap((1, 5), &adj_matrix).is_err());
        assert!(s.do_swap((6, 2), &adj_matrix).is_err());

        // same elt
        assert!(s.do_swap((2, 2), &adj_matrix).is_err());
        assert!(s.do_swap((0, 0), &adj_matrix).is_err());

        // 1-elt state path
        let mut s: State = State::new(vec![0], &adj_matrix).unwrap();

        assert!(s.do_swap((0, 0), &adj_matrix).is_err());
        assert!(s.do_swap((1, 2), &adj_matrix).is_err());

        // 2-elt state path
        let mut s: State = State::new(vec![0, 1], &adj_matrix).unwrap();

        assert!(s.do_swap((0, 1), &adj_matrix).is_err());
        assert!(s.do_swap((1, 0), &adj_matrix).is_err());
    }

    #[test]
    fn find_best_swap() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0]  // 3
        ];
        let s: State = State::new(vec![0, 1, 3, 2, 0], &adj_matrix).unwrap();

        assert_eq!(s.get_cost(), 108);

        // the ideal config is [0, 1, 2, 3, 0], which costs 97
        assert!(s.find_best_swap(&adj_matrix).is_some());
        assert_eq!(s.find_best_swap(&adj_matrix).unwrap(), (2, 3));
    }

    #[test]
    fn find_best_swap_none() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0]  // 3
        ];
        let s: State = State::new(vec![0, 1, 2, 3, 0], &adj_matrix).unwrap();

        assert_eq!(s.get_cost(), 97);

        // already in ideal config; no profitable swap exists
        assert!(s.find_best_swap(&adj_matrix).is_none());
    }
}

#[cfg(test)]
mod random_hamiltonian_cycle_tests {
    use super::*;

    #[test]
    fn args_too_small() {
        let err: std::result::Result<State, &'static str> = Err("Adjacency matrix too small!");

        let empty_adj_matrix = vec![];
        assert_eq!(empty_adj_matrix.len(), 0);
        assert_eq!(random_hamiltonian_cycle(&empty_adj_matrix), err);

        let singleton_adj_matrix = vec![vec![1]];
        assert_eq!(singleton_adj_matrix.len(), 1);
        assert_eq!(random_hamiltonian_cycle(&singleton_adj_matrix), err);
    }

    #[test]
    fn args_not_square() {
        let err: std::result::Result<State, &'static str> = Err("Adjacency matrix not square!");

        let non_square_adj_matrix_0 = vec![vec![0, 1], vec![]];
        assert_eq!(random_hamiltonian_cycle(&non_square_adj_matrix_0), err);

        let non_square_adj_matrix_1 = vec![vec![0, 1], vec![0]];
        assert_eq!(random_hamiltonian_cycle(&non_square_adj_matrix_1), err);

        let non_square_adj_matrix_2 = vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8, 9]];
        assert_eq!(random_hamiltonian_cycle(&non_square_adj_matrix_2), err);
    }

    #[test]
    fn args_good() {
        let adj_matrix = vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3]
        ];

        assert!(random_hamiltonian_cycle(&adj_matrix).is_ok());

        let s: State = random_hamiltonian_cycle(&adj_matrix).unwrap();

        assert!(s.is_goal(&adj_matrix));
        assert_eq!(s.get_path().len(), adj_matrix.len() + 1);
    }
}

#[cfg(test)]
mod genetic_tests {
    use super::*;

    #[test]
    fn calculate_fitnesses_normal() {
        let adj_matrix: Vec<Vec<u32>> = vec![
            //   0  1   2   3
            vec![0, 20, 42, 35], // 0
            vec![20, 0, 30, 34], // 1
            vec![42, 30, 0, 12], // 2
            vec![35, 34, 12, 0]  // 3
        ];
        let pop: Vec<State> = vec![
            State::new(vec![0, 3, 2, 1, 0], &adj_matrix).unwrap(),
            State::new(vec![2, 1, 3, 0, 2], &adj_matrix).unwrap(),
            State::new(vec![2, 0, 1, 3, 2], &adj_matrix).unwrap(),
            State::new(vec![2, 1, 0, 3, 2], &adj_matrix).unwrap()
        ];
        let mut fit: Vec<f32> = vec![9.99, 9.99, 9.99, 9.99];
        let correct_fit: Vec<f32> = vec![
            0.27885514,
            0.19183652,
            0.25045323,
            0.27885514
        ];

        calculate_fitnesses(&pop, &mut fit);

        assert_eq!(fit, correct_fit);
    }
}
