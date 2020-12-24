mod graph_generation;
use graph_generation::*;
mod algorithms;
use algorithms::*;

fn main() {
    let graph: Vec<Vec<u32>> = read_adj_matrix("./graphs/square_x.txt");
    // ./graphs/square_x.txt
    //     20
    //   0─────1      35 34 12 0
    //   │╲35 ╱│      20 0 30 34
    //   │ ╲ ╱ │      42 30 0 12
    // 42│  ╳  │34    35 34 12 0
    //   │ ╱ ╲ │
    //   │╱30 ╲│
    //   2─────3
    //     12
    // shortest tour from 0: 0 -> 1 -> 2 -> 3 -> 0 (total cost: 97)

    println!("{:?}", graph);
    
}
