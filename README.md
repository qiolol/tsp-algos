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
it got down to a path cost of `8565`! Of course, I have no idea if that's the actual
global minimum of the graph, but it's probably close...

## 🧬 genetic
// TODO
