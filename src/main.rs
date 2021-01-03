mod graph_generation;
use graph_generation::*;
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

    // write_graph(&generate_complete_random_graph(100), "./graphs/big.txt"); // generate new graph
    let graph: Vec<Vec<u32>> = read_adj_matrix("./graphs/big.txt");

    println!(
        "\
┌────────────────┐\n\
│⛰️ hill climbing│\n\
└────────────────┘\
        \n{}",
        time_algo(hill_climbing, &graph)
    );

    println!(
        "\
┌─────────────────────┐\n\
│🌡️ simulated annealing│\n\
└─────────────────────┘\
        \n{}",
        time_algo(simulated_annealing, &graph)
    );
}
