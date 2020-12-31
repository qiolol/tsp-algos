# tsp-algos
Traveling salesperson problem (TSP) algorithms

Specifically, [Euclidean](https://en.wikipedia.org/wiki/Travelling_salesman_problem#Euclidean) TSP algorithms seeking minimum-cost [Hamiltonian](https://en.wikipedia.org/wiki/Hamiltonian_path) cycles over graphs that are:

* random
* undirected
* weighted
* completely connected
* made of nodes whose edges follow the triangle inequality (going directly from A to B is cheaper than going from A to C to B)

Includes a generator of such graphs in `src/graph_generation.rs`.

An example, hand-made graph is `./graphs/square_x.txt`:

```
    20         adjacency matrix:
  0─────1      0 20 42 35
  │╲35 ╱│      20 0 30 34
  │ ╲ ╱ │      42 30 0 12
42│  ╳  │34    35 34 12 0
  │ ╱ ╲ │
  │╱30 ╲│
  2─────3
    12
```

The Hamiltonian cycle (a path beginning and ending with the same node that goes through every other node exactly once) of least cost from node `0` is `0 -> 1 -> 2 -> 3 -> 0` (with total cost 97).

Because the graph is complete, this path can be traveled starting from any node: `3 -> 2 -> 1 -> 0 -> 3` and `2 -> 3 -> 0 -> 1 -> 2` are equivalent ways of tracing the same, 97-cost, hourglass-shaped cycle over the nodes.

## ⛰️ hill climbing
[Hill climbing](https://en.wikipedia.org/wiki/Hill_climbing) is a simple, dumb, greedy search for the global maximum. At each step, it
looks at each direction, and it steps in the direction of *steepest ascent* -- the neighboring
state that increases the value the most. If no direction improves upon the value of the current
state, it stops.

Naively, this parks it at the global maximum. Often, this only leads to the *local* maximum
because, without more complex techniques, there's no way for the algorithm to know whether the
global maximum is just beyond the next "valley". Starting from a random position (like this
implementation does), choosing the next step with some randomness, and restarting randomly a
number of times after finishing are ways to make hill climbing more effective at reaching the
global maximum.

In the context of the TSP defined here, the "global maximum" is the global *minimum* of the
path cost of a Hamiltonian cycle over the graph. The graphs used are complete, which *does*
mean there can't be any local minima/maxima to get stuck at! Every node is connected to every
other, so the optimal Hamiltonian cycle is reachable starting from any node. So our naive
hill climbing algo here has nothing to worry about, right? Wrong! The problem is this algo
operates on Hamiltonian cycles, not on the graph itself. It generates a cycle at random, and
then it swaps pairs of elements around in hopes of reaching the optimal cycle, choosing the
pair resulting in "steepest descent" each time.

Two assumptions are made:

1. For any node, there's an optimal Hamiltonian cycle starting and ending with that node
(put another way, the *one* optimal cycle is reachable from any node, and there's always a
representation of this cycle that starts and ends with any given node)

2. For any Hamiltonian cycle, the optimal Hamiltonian cycle is reachable via element swaps

The actual graph that a cycle runs over may be complete, but the *graph of all possible cycles*
is *incomplete*! Imagining its nodes as possible Hamiltonian cycles and its edges as swaps,
there's no guarantee that the optimal cycle is only one swap away from the cycle the algo
starts with, so it's incomplete. Incomplete means the danger of local minima/maxima returns!
Some series of swaps *is* going to get us to the optimal cycle, but this poor algo is limited
to a single, narrow swap strategy: steepest descent. If the optimal cycle is separated from
the current cycle by a swap that results in an *increase* in cost (or even a decrease less
significant than the decrease of some other swap), the algo will never arrive at the optimal
cycle -- it'll settle into some local minimum.

Deviously, this only apparent for graphs of 5 nodes or more (5 may take a few tries to see a
deviation in the returned cost; 6 or more is more obvious). A mathematically-inclined friend
patiently explained this is because, for sufficiently small cycles, whatever contrived
"rotations" of nodes in the cycle's path necessary to get to the optimum are equivalent to
steepest-descent swaps.

## 🌡️ simulated annealing
// TODO

## 🧬 genetic
// TODO
