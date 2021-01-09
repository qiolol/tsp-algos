#[allow(dead_code)]
mod graph_generation;
use graph_generation::*;
#[allow(dead_code)]
mod algorithms;
use algorithms::*;

fn main() {
    // ./graphs/square_x.txt
    //     20
    //   0─────1      0 20 42 35
    //   │╲35 ╱│      20 0 30 34
    //   │ ╲ ╱ │      42 30 0 12
    // 42│  ╳  │34    35 34 12 0
    //   │ ╱ ╲ │
    //   │╱30 ╲│
    //   2─────3
    //     12
    // shortest tour from 0: 0 -> 1 -> 2 -> 3 -> 0 (total cost: 97)
    // let graph: Vec<Vec<u32>> = read_adj_matrix("./graphs/square_x.txt");

    // write_graph(&generate_complete_random_graph(100), "./graphs/big.txt").unwrap(); // generate new graph
    println!("Using a ./graphs/big.txt, a 100-node graph...");
    let mut graph: Vec<Vec<u32>> = read_adj_matrix("./graphs/big.txt");

    println!(
        "\
┌─────────────┐\n\
│hill climbing│\n\
└─────────────┘\
        \n{}\n",
        hill_climbing(&graph)
    );

    println!(
        "\
┌───────────────────┐\n\
│simulated annealing│\n\
└───────────────────┘\
        \n{}\n",
        simulated_annealing(
            &graph,
            0.000015, // cooling_rate (keeping it small is crucial!)
            100.0 // temperature
        )
    );

    println!(
        "\
┌───────┐\n\
│genetic│\n\
└───────┘\
        \n{}\n",
        genetic(
            &graph,
            100, // population size (# per generation)
            500, // # of generations to simulate
            0.5 // mutation rate
        )
    );

    println!(
        "... A* explodes RAM very easily, so it needs a smaller graph. xD\n\
Using ./graphs/a_star_sized.txt, which is a puny 11 nodes..."
    );
    graph = read_adj_matrix("./graphs/a_star_sized.txt");

    println!(
        "\
┌──┐\n\
│A*│\n\
└──┘\
        \n{}",
        // Can run A* with several heuristics:
        // - a_star_uniform_cost (slow but optimal)
        // - a_star_random_edge (fast but non-optimal)
        // - a_star_cheapest_edge (fast-ish And optimal)
        a_star_cheapest_edge(&graph)
    );
}
