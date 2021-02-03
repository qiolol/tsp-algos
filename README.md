# tsp-algos
Travelling salesperson problem (TSP) algorithms

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

2. Any Hamiltonian cycle can be steadily transformed into the optimal cycle via element swaps

The actual graph that a cycle runs over may be complete, but the **graph of all possible cycles**
is *incomplete*! Imagining the nodes as possible Hamiltonian cycles and its edges as swaps,
there's no guarantee that the optimal cycle is only one swap away from the cycle the algo
starts with, so it must be incomplete. Incomplete means the danger of local minima/maxima returns!
Some series of swaps *can* get us to the optimal cycle, but this poor algo is limited to a
single, narrow swap strategy: steepest descent. If the optimal cycle is separated from
the current cycle by a swap that results in an *increase* in cost (or even a decrease less
significant than the decrease of some other swap), the algo will never arrive at the optimal
cycle -- it'll settle into some local minimum.

Deviously, this only apparent for graphs of 5 nodes or more -- i.e., graphs that are a pain
to draw and solve by hand. Running it on 4-node ones like the hand-made graph above may fool
you into thinking it finds the global minimum every time! (And 5 may take a few tries to see
a deviation in the returned cost; 6 or more is more obvious). A mathematically-inclined friend
patiently explained this is because, for sufficiently small cycles, whatever contrived "rotations"
of nodes in the cycle's path necessary to get to the optimum are equivalent to steepest-descent swaps.

## 🌡️ simulated annealing
[Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) is a cool (xDDD)
metallurgical metaphor that combines hill climbing's greed with some randomness.
While hill climbing's greed guarantees it getting trapped in some local minimum,
a random algorithm that blindly goes all over the place is guaranteed to find
the *global* minimum (albeit extremely slowly, barring extreme luck).
Simulated annealing seeks the best of both worlds! Like hill climbing, it starts
with a random Hamiltonian cycle. Its first difference is that it has two key variables:

* `temperature`
* `cooling_rate`

Next, instead of searching for and executing the best swap, it makes a *random* swap.
If that swap is a cost decrease, it keeps it. If it's a cost *increase*, it keeps it with
a probability. It does this over and over until `temperature` drops below some value,
at which point it returns the cycle it's left with.

The [Boltzmann distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution)
is "the probability that a system will be in a certain state as a function of
that state's energy and the temperature of the system". This models nature's
"preference" for *lower-energy* states, which makes it convenient to borrow for an algorithm
seeking the global *minimum* of something! In simulated annealing, the "energy" is the
*cost increase* of the swap. A higher increase is worse and decreases the probability it will be kept.
Temperature, represented in the `temperature` variable, also affects this probability.
At high `temperature`, an increase is *more likely* to be kept. So, even if a large increase
is unlikely to be accepted, a sufficiently high temperature may very well shift the probability
in its favor.

[Metallurgical annealing](https://en.wikipedia.org/wiki/Annealing_(metallurgy)) works
"by heating the material (generally until glowing) for a while and then slowly
letting it cool to room temperature in still air." The heat loosens the metal atoms
up, letting them re-adjust to lower-energy (presumably more comfortable) configurations
before recrystallizing as the metal cools into more uniformly-aligned grains.
The higher uniformity of the crystals grains increases the metal's ductility.

True to its name, simulated annealing works the same way! `temperature` begins
at some high value. So, in early iterations, all kinds of swaps -- even very bad
ones -- are tolerated. Imagining it as a cursor on a spiky line graph, the algo
jumps around everywhere in a mad rollick! With each iteration, `temperature`
decreases according to `cooling_rate`. As `temperature` cools, the algo becomes
more conservative about what swaps it accepts and rejects increases more and more.
Ideally, `cooling_rate` is *very small* so that `temperature` decreases *very slowly*.
The more time spent dancing madly around
the solution space, the likelier that the global minimum was approached; simultaneously,
as `temperature` cools, the more likely the algo is to linger around the minimum.
By the time `temperature` gets low, the algo essentially becomes plain, greedy hill climbing,
only accepting swaps that decrease cost, and it sinks into whatever minimum it's near.
The hope is that the wild, high-`temperature` romp in the beginning stumbles near global minimum, and
that the increasing reluctance to accept bad swaps traps it there.

It's all probabilistic.

And it works pretty well! On the pre-generated, 100-node graph (`./graphs/big.txt`),
it reliably beats hill climbing in the lowest cost it can get to, and it's much
faster to boot. The preset values, `temperature = 100.0` and `cooling_rate = 0.00001`,
increase its run time to around that of hill climbing, and it gets to a minimum cost
3,000 to 6,000 less. The longer it runs, the more accurate its result. On the same
graph, after a 5 hour run with `temperature = 100.0` and `cooling_rate = 0.0000001`,
it got down to a path cost of 8,565! Of course, I have no idea if that's the actual
global minimum of the graph, but it's probably close...

## 🧬 genetic
[Genetic algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm), my favorite,
are biological metaphors where many solutions compete with each other to pass on their
"genes". The higher a solution's *fitness*, the more likely it gets to mate. In a
very crude approximation of biological evolution, the best solutions thus reproduce
and combine, and then their children (which may also experience random mutations)
repeat the process for some number of generations until either some limit of
fitness or a limit of generation number is hit, and the fittest individual of the
final generation is chosen as **the** solution.

In this TSP setup, solutions are Hamiltonian cycles, and their fitness is how
cheap their path cost is. The "DNA strands" are the arrays of nodes, and the nodes
are the "nucleotides" (the A, C, T, G). Sadly, this isn't the best problem format
for this kind of genetic algorithm. The reproduction logic is pretty limited due
to the narrow criteria for solution validity. Remember, a Hamiltonian cycle must
contain all nodes and have no repeated nodes (except for the first and last node).
That means that, given two parents `a` and `b` (two Hamiltonian cycles), using half
of `a`'s path and half of `b`'s path for their child `c`'s path (which could be
perfectly appropriate for other kinds of problems) would very likely yield an
invalid Hamiltonian cycle! The ways to get around this aren't very satisfying.
A common solution seems to be to take half of one parent's path, and then fill
the remaining half with the missing elements from the other parent's path, in the
order they appear in it. E.g.,

```
parent_a: [0, 3, 2, 1, 0]
           ^  ^
parent_b: [2, 3, 1, 0, 2]
           &     &
child:    [_, _, _, _, _]
                          fill with half (5 / 2 ~= 2) of parent_a,
          [0, 3, _, _, 0] capping with first node
           ^  ^        ^  
          [0, 3, 2, 1, 0] fill with remaining elements in order from parent_b
                 &  &
```

This preserves half of one parent's DNA, and then it peppers in segments of the
other's, which is kind of apt to a combination of DNA, right? Unfortunately, it's
very slow (since it's traversing two entire paths per mating), and it doesn't
even seem to work well (the solutions it returns are very bad).

What I tried is even lamer, but much faster: if the two parents differ in terminal
(first and last) node, take one parent and switch its terminal node to the other
parent's. If they don't differ in terminal nodes, take one parent and swap one
of its elements to more closely resemble the other parent. Surprisingly, the
solutions this produces seem slightly better, or at worst *as* bad, but take
much less to arrive at since there's no full path traversals happening. They're
still really bad solutions though. 😋 Adjusting the runtime to around 7 seconds
(already longer than hill climbing and simulated annealing), the solutions are
around 47,000 (3 to 4 times worse!)... ¬w¬

## 🅰️⭐
[A*](https://en.wikipedia.org/wiki/A*_search_algorithm) is a classic search algorithm
(there's [an old Red Blob page](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
with a great walkthrough of it). So far, all the algorithms (genetic, simulated
annealing, and hill climbing) were "stochastic", going somewhere, figuring out
how badly they need to go somewhere else, and doing that until they run out of
generations/temperature units/locally-obvious places to go. A* is a systematic
algorithm. Instead of starting with some random goal state and iteratively
mutating it to be better, it starts its journey from one, humble node, and builds up
a path. It keeps track of where it's been and where it hasn't, and,
unlike the algorithms above, is *guaranteed* to find the optimal solution (if
it uses an *admissible* heuristic, anyway).

Heuristics are problem-specific functions that give A* numeric intuition
about how good its options are. Fundamentally, A* measures an option `n` with:

```
f(n) = `g(n) + h(n)`
```

`g(n)` is the cost of the path so far from the start to `n`, and `h(n)` is the
*estimated optimal* cost from `n` to a goal. `h` is the heuristic function!

A heuristic is *admissible* if it never overestimates the cost to a goal (it's "optimistic").
This means A* will not be too discouraged to eventually reach a goal, thereby guaranteeing that
it will. More optimism isn't always better though: with zero discouragement, no options are taken
off the table, and A* will take all of them. This means it'll waste more time, going through suboptimal
detours before getting to the optimal ones. Sometimes, when an non-optimal but fast solution is wanted,
inadmissible heuristics can be used because these take *so* many options off the table that A* wastes
very little time before arriving at a solution, if not the best solution! The sweet spot
is a "minimally-optimistic" heuristic -- one that's *just* optimistic enough to be
admissible and no more.

Three basic heuristics are used in this A* implementation:

- Uniform cost, so-called because `h(n) = 0` for any `n`
    - ✅ Makes A* optimal! It's very admissible since it grossly *underestimates*
    the cost to any goal (everything's `0`!).
    - ❌ Makes A* *very* slow... being *extremely* optimistic, it eliminates few
    options, causing A* to unselectively run all over them, consuming far more cycles.
- Random edge, which estimates the cost to the goal by summing up random edges between
all nodes not yet visited
    - ❌ Non-optimal! Using random edges is a pretty lazy way to measure the cost
    to the goal, and it can easily overestimate. This makes it it inadmissible,
    so it's no surprise it hurts optimality.
    - ✅ Fastest one! 🏁 Again, no surprise; inadmissible heuristics hem in the
    options A* explores, so it explores much fewer before finishing.
- Cheapest edge, which estimates the cost to the goal by summing up the cheapest
edges between unvisited nodes
    - ✅ Optimal! This heuristic is admissible since it assumes the path to the goal
    will always be taken via the cheapest remaining edges, which is optimistic.
    - ✅ Fast! Not as fast as an inadmissible heuristic, but much faster than a WAY
    MORE admissible one like uniform cost! It cuts A*'s search time by a lot since
    it eliminates some search paths but doesn't eliminate enough of them to stop
    arrival at an optimal solution.
    - 🏆 Of the three, this is the sweet spot heuristic.

Sadly, there's a price to pay for A\*'s systematicness and optimality. All the
memorization it does about where it's been means it's quick to gobble up memory.
While stochastic algorithms like hill climbing and simulated annealing can work
on a 100-node graph no sweat, A* chokes and begins 💥 exploding my RAM on even
a ***15***-node graph! This depends on the heuristic. Less admissible ones can
handle larger sizes since they decrease the number of paths A* explores, but
using inadmissible ones sacrifices optimality, at which point you might as well
use the stochastic algorithms.

This could very well be due to my implementation, which is *not* guaranteed to
be written as efficiently as possible by any means. A* implementations can get
really complex, using memoization and disjoint-set data structures and whatnot.
It might also be that TSP is especially challenging for A\*. A grid in a
roguelike game, with a branching factor of 8 (a tile's neighbors), might be more
manageable than a complete graph, where all `n` nodes connect to every other
for a branching factor of `n`, particularly when A* contemplates cost estimations
to its solution, sampling `n` edges each step along the way, and when the solution
can be based on the proximity to some entity instead of all unvisited nodes!
