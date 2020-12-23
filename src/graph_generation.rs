//! Generates random, undirected, weighted, and completely connected graphs and writes them to a
//! file as adjacency matrices
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::prelude::*;

use rand::{thread_rng, Rng};

/// A 2D point
#[derive(Default, Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct Point {
    pub id: u32,
    pub x: u32,
    pub y: u32,
}

/// An undirected graph using an adjacency list
#[derive(Default, Debug)]
pub struct Graph {
    pub size: usize,
    nodes: HashSet<Point>,
    adj_list: HashMap<u32, HashSet<u32>>,
    adj_matrix: Vec<Vec<u32>>,
}

/// An undirected graph
///
/// When printed, a Graph is represented as an adjacency matrix. The first line
/// of the output string contains the number of nodes, and every subsequent line
/// is a row of the adjacency matrix. A `0` indicates absence of an edge, and any greater number
/// represents an edge weighted with the Euclidean distance between its nodes.
impl Graph {
    fn new() -> Self {
        Default::default()
    }

    /// Returns the number of nodes in the Graph
    #[allow(dead_code)]
    fn size(&self) -> usize {
        self.size
    }

    /// Returns the Graph's nodes in a Vec
    fn get_nodes(&self) -> Vec<Point> {
        return self.nodes.iter().map(|&x| x).collect();
    }

    /// Returns the Point with the specified id in an Option
    #[allow(dead_code)]
    fn get_node(&self, id: u32) -> Option<Point> {
        if self.adj_list.contains_key(&id) {
            if let Some(node_ref) = self.get_nodes().iter().find(|&&x| x.id == id) {
                return Some(*node_ref);
            }
        }

        return None;
    }

    /// Returns a Point p's neighbors in a Result-wrapped Vec
    #[allow(dead_code)]
    fn get_neighbors(&self, p: &Point) -> Result<Vec<Point>, &'static str> {
        if !self.nodes.contains(p) {
            return Err("Point not in graph!");
        }

        let neighbor_ids = self.adj_list.get(&p.id).unwrap();
        let mut neighbor_points: Vec<Point> = vec![];

        for id in neighbor_ids {
            for node in &self.nodes {
                if node.id == *id {
                    neighbor_points.push(*node);
                }
            }
        }

        return Ok(neighbor_points);
    }

    /// Returns whether the Graph contains Point p
    fn has_node(&self, p: &Point) -> bool {
        return self.nodes.contains(p) && self.adj_list.contains_key(&p.id);
    }

    /// Returns whether the Graph contains a Point with the given id
    /// (riskier convenience function; Points are not equality tested by id alone)
    #[allow(dead_code)]
    fn has_node_by_id(&self, id: u32) -> bool {
        if let Some(_) = self.get_node(id) {
            return true;
        }
        else {
            return false;
        }
    }

    /// Returns whether the Graph has an edge between Points p1 and p2
    fn has_edge(&self, p1: &Point, p2: &Point) -> bool {
        if self.has_node(p1) && self.has_node(p2) {
            return self.adj_list.get(&p1.id).unwrap().contains(&p2.id) &&
                self.adj_list.get(&p2.id).unwrap().contains(&p1.id);
        }

        return false;
    }

    /// Returns whether the Graph has an edge between two Points with ids id1 and id2
    /// (riskier convenience function; Points are not equality tested by id alone)
    #[allow(dead_code)]
    fn has_edge_by_id(&self, id1: u32, id2: u32) -> bool {
        if self.has_node_by_id(id1) && self.has_node_by_id(id2) {
            return self.has_edge(&self.get_node(id1).unwrap(), &self.get_node(id2).unwrap());
        }

        return false;
    }

    /// Adds a Point as a node to the Graph
    ///
    /// Returns an Err if a Point with the same coordinates or id was already
    /// in the Graph
    ///
    /// # Arguments
    ///
    /// * `p` - Point to add to the Graph
    fn add_node(&mut self, p: Point) -> Result<(), &'static str> {
        if self.nodes.contains(&p) {
            return Err("Point already in Graph!");
        }
        else {
            // p wasn't in the Graph, but another Point with the same coordinates or id might be.
            // Since Point's PartialEq defaults to checking even its id (which is nice and simple),
            // two points with the SAME coordinates would be distinguished as DIFFERENT points (which
            // is not nice, especially when calculating square roots).
            // Similarly, a point with the same id but different coordinates might be in here.
            // Check for redundant id
            if self.adj_list.contains_key(&p.id) {
                return Err("Point ID already in Graph with different coordinates!");
            }
            // Check for redundant coords
            for point in &self.nodes {
                if (p.x == point.x) && (p.y == point.y) {
                    return Err("Point coordinates already in Graph with different ID!");
                }
            }

            self.nodes.insert(p);
            self.adj_list.insert(p.id, HashSet::<u32>::new());
            self.size += 1;

            return Ok(());
        }
    }

    /// Connects two nodes in the Graph
    ///
    /// Returns an Err if either Point argument is not in the Graph or if the edge already exists
    ///
    /// # Arguments
    ///
    /// * `p1` - Point to connect to `p2`
    /// * `p2` - Point to connect to `p1`
    fn add_edge(&mut self, p1: &Point, p2: &Point) -> Result<(), &'static str> {
        if !self.has_node(p1) || !self.has_node(p2) {
            return Err("Point(s) not in Graph!");
        }
        else if self.has_edge(p1, p2) {
            return Err("Edge already in Graph!");
        }

        let p1_neighbors = self.adj_list.get_mut(&p1.id).unwrap();
        p1_neighbors.insert(p2.id);

        let p2_neighbors = self.adj_list.get_mut(&p2.id).unwrap();
        p2_neighbors.insert(p1.id);

        return Ok(());
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.size;

        write!(f, "{}\n", n).unwrap();
        for i in 0..n {
            for j in 0..n {
                write!(f, "{}", self.adj_matrix[i][j]).unwrap();
                if j != (n - 1) { // don't write space at end of row
                    write!(f, " ").unwrap();
                }
            }
            write!(f, "\n").unwrap();
        }
        Ok(())
    }
}

/// Generates and returns a completely connected, random Graph
///
/// Nodes are generated as random Points on a 1,000 by 1,000 square.
///
/// # Arguments
///
/// * `n` - Number of nodes in the Graph
///
/// # Panics
///
/// - if `n` is 0
pub fn generate_complete_random_graph<'a>(n: usize) -> Graph {
    if n == 0 {
        panic!("Graph size must be greater than zero!");
    }

    let mut graph = Graph::new();
    let mut rng = thread_rng();

    // generate nodes
    let mut i = 0;

    while i < n {
        let p = Point { id: i as u32, x: rng.gen_range(0, 1000), y: rng.gen_range(0, 1000) };

        match graph.add_node(p) {
            Ok(_) => i += 1,
            Err(e) => {
                eprintln!("{}", e);
                i += 0 // repeat iteration cuz we likely got same coords by chance
            }
        };
    }

    let keys = graph.get_nodes();

    // add to each node edges to every other node
    for node in &keys {
        let other_nodes: Vec<Point> = keys.iter().filter(|&&x| x != *node).map(|&x| x).collect();

        for other_node in other_nodes.iter() {
            if !graph.has_edge(node, other_node) {
                graph.add_edge(node, &other_node).unwrap();
            }
        }
    }

    // make adjacency matrix with weighted edge values
    let mut adj_matrix = vec![vec![0; n]; n];

    for node in graph.get_nodes() {
        for neighbor in graph.get_neighbors(&node).unwrap() {
            if adj_matrix[node.id as usize][neighbor.id as usize] == 0 ||
                adj_matrix[neighbor.id as usize][node.id as usize] == 0 {
                let x_0 = node.x as i32;
                let y_0 = node.y as i32;

                let x_1 = neighbor.x as i32;
                let y_1 = neighbor.y as i32;

                // sqrt((x0 - x1)^2 + (y0 - y1)^2), as a rounded int
                let euclidean_distance = (
                    (
                        (x_0 - x_1).pow(2)
                        +
                        (y_0 - y_1).pow(2)
                    ) as f64
                ).sqrt().round() as u32;

                adj_matrix[node.id as usize][neighbor.id as usize] = euclidean_distance;
                adj_matrix[neighbor.id as usize][node.id as usize] = euclidean_distance;
            }
        }
    }
    graph.adj_matrix = adj_matrix;

    return graph;
}

/// Writes a Graph to a file
///
/// # Arguments
///
/// * `graph` - Graph to write to file
/// * `filename` - Path of the file to write to
pub fn write_graph(graph: &Graph, filename: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(filename)?;

    write!(file, "{}", graph)
}

/// Reads and returns a square adjacency matrix from a text file
///
/// The expected format is for the first line of the file to contain a number which indicates the
/// number of rows and columns of the matrix and for the subsequent lines to represent the matrix.
///
/// E.g.,
///
/// 3
///
/// 0 5 5
///
/// 5 0 3
///
/// 5 3 0
pub fn read_adj_matrix(filename: &str) -> Vec<Vec<u32>> {
    let mut adj_matrix = vec![];
    let mut file = File::open(filename).expect("Unable to open file!");
    let mut content = String::new();

    file.read_to_string(&mut content).expect("Unable to read file!");

    for (i, line) in content.split('\n').enumerate() {
        if (!line.is_empty()) && (i > 0) {
            let mut row: Vec<u32> = vec![];

            for (j, edge) in line.split(' ').enumerate() {
                row.push(edge.parse().unwrap());
            }
            adj_matrix.push(row);
        }
    }

    return adj_matrix;
}

#[cfg(test)]
mod graph_tests {
    use super::*;

    // Returns a square-shaped Graph of four nodes and four edges and its Points
    fn square_graph() -> (Graph, Point, Point, Point, Point) {
        let mut g = Graph::new();

        let p0 = Point { id: 0, x: 0, y: 0 };
        let p1 = Point { id: 1, x: 1, y: 1 };
        let p2 = Point { id: 2, x: 2, y: 2 };
        let p3 = Point { id: 3, x: 3, y: 3 };

        g.add_node(p0).unwrap();
        g.add_node(p1).unwrap();
        g.add_node(p2).unwrap();
        g.add_node(p3).unwrap();

        g.add_edge(&p0, &p1).unwrap();
        g.add_edge(&p1, &p2).unwrap();
        g.add_edge(&p2, &p3).unwrap();
        g.add_edge(&p3, &p0).unwrap();

        return (g, p0, p1, p2, p3);
    }

    // Test empty graph
    #[test]
    fn empty_graph() {
        let mut g = Graph::new();

        assert_eq!(g.size(), 0);

        let p0 = Point { id: 0, x: 0, y: 0 };
        let p1 = Point { id: 1, x: 1, y: 1 };

        assert!(!g.has_node(&p0));
        assert!(!g.has_node_by_id(0));

        assert!(!g.has_edge(&p0, &p1));
        assert!(!g.has_edge_by_id(0, 1));

        assert!(g.add_node(p0).is_ok());
        assert_eq!(g.size(), 1);

        assert!(g.add_node(p1).is_ok());
        assert_eq!(g.size(), 2);

        assert!(g.add_edge(&p0, &p1).is_ok());
        assert_eq!(g.size(), 2);
    }

    // Test normal graph operations
    #[test]
    fn square_graph_normal() {
        let (g, p0, p1, p2, p3) = square_graph();

        assert_eq!(g.size(), 4);

        assert!(g.has_node(&p0));
        assert!(g.has_node(&p1));
        assert!(g.has_node(&p2));
        assert!(g.has_node(&p3));
        assert!(g.has_node_by_id(0));
        assert!(g.has_node_by_id(1));
        assert!(g.has_node_by_id(2));
        assert!(g.has_node_by_id(3));

        assert!(g.has_edge(&p0, &p1));
        assert!(g.has_edge(&p1, &p2));
        assert!(g.has_edge(&p2, &p3));
        assert!(g.has_edge(&p3, &p0));
        assert!(g.has_edge_by_id(0, 1));
        assert!(g.has_edge_by_id(1, 2));
        assert!(g.has_edge_by_id(2, 3));
        assert!(g.has_edge_by_id(3, 0));

        assert!(g.get_node(0).is_some());
        assert_eq!(g.get_node(0).unwrap(), p0);

        assert!(g.get_neighbors(&p1).is_ok());

        let p1_neighbors = g.get_neighbors(&p1).unwrap();

        assert_eq!(p1_neighbors.len(), 2);
        assert!(p1_neighbors.iter().any(|&p| p.id == 0));
        assert!(p1_neighbors.iter().any(|&p| p.id == 2));
    }

    // Test operations with Points not in the Graph
    #[test]
    fn square_graph_missing_points() {
        let (mut g, p0, p1, p2, _p3) = square_graph();
        let p4 = Point { id: 4, x: 4, y: 4 };
        let p5 = Point { id: 5, x: 5, y: 5 };

        let expected_ok: Result<(), &'static str> = Ok(());
        let expected_add_edge_err: Result<(), &'static str> = Err("Point(s) not in Graph!");
        let expected_get_neigh_err: Result<Vec<Point>, &'static str> = Err("Point not in graph!");

        assert_eq!(g.has_node(&p0), true);
        assert_eq!(g.has_node_by_id(0), true);
        assert_eq!(g.has_node(&p4), false);
        assert_eq!(g.has_node_by_id(4), false);

        assert_eq!(g.has_edge(&p0, &p2), false);
        assert_eq!(g.has_edge_by_id(0, 2), false);
        assert_eq!(g.has_edge(&p1, &p2), true);
        assert_eq!(g.has_edge_by_id(1, 2), true);
        assert_eq!(g.has_edge(&p4, &p5), false);
        assert_eq!(g.has_edge_by_id(4, 5), false);
        assert_eq!(g.has_edge(&p0, &p5), false);
        assert_eq!(g.has_edge_by_id(0, 5), false);

        assert_eq!(g.add_edge(&p0, &p2), expected_ok);
        assert!(g.add_edge(&p4, &p5).is_err());
        assert_eq!(g.add_edge(&p4, &p5), expected_add_edge_err);
        assert_eq!(g.add_edge(&p0, &p5), expected_add_edge_err);

        assert_eq!(g.get_node(0).unwrap(), p0);
        assert!(g.get_node(0).is_some());
        assert!(g.get_node(4).is_none());

        assert!(g.get_neighbors(&p1).is_ok());
        assert!(g.get_neighbors(&p4).is_err());
        assert_eq!(g.get_neighbors(&p4), expected_get_neigh_err);
    }

    // Test operations with Points already in the Graph
    #[test]
    fn square_graph_redundant_points() {
        let (mut g, p0, p1, _p2, _p3) = square_graph();
        let p1_clone = Point { id: 1, x: 1, y: 1 };

        assert_eq!(p1, p1_clone);

        let expected_add_node_same_err: Result<(), &'static str> = Err("Point already in Graph!");
        let expected_add_edge_already_err: Result<(), &'static str> = Err("Edge already in Graph!");

        assert!(g.add_node(p1_clone).is_err());
        assert_eq!(g.add_node(p1_clone), expected_add_node_same_err);
        assert_eq!(g.add_node(p1), expected_add_node_same_err);

        assert!(g.add_edge(&p0, &p1).is_err());
        assert_eq!(g.add_edge(&p0, &p1), expected_add_edge_already_err);
        assert_eq!(g.add_edge(&p0, &p1_clone), expected_add_edge_already_err);
    }

    // Test operations with Points that have the same coordinates as Points in the Graph but
    // different id
    #[test]
    fn square_graph_dangerous_id() {
        let (mut g, _p0, p1, _p2, _p3) = square_graph();
        let p1_diff_id = Point { id: 10, x: 1, y: 1 };

        assert_ne!(p1, p1_diff_id);

        let expected_add_node_diff_id_err: Result<(), &'static str> = Err("Point coordinates already in Graph with different ID!");
        let expected_add_edge_err: Result<(), &'static str> = Err("Point(s) not in Graph!");

        assert_eq!(g.add_node(p1_diff_id), expected_add_node_diff_id_err);
        assert_eq!(g.add_edge(&p1, &p1_diff_id), expected_add_edge_err);
    }

    // Test operations with Points that have the same id as Points in the Graph but different
    // coordinates
    #[test]
    fn square_graph_dangerous_coords() {
        let (mut g, _p0, p1, _p2, _p3) = square_graph();
        let p1_diff_coords = Point { id: 1, x: 10, y: 10 };

        assert_ne!(p1, p1_diff_coords);

        let expected_add_node_diff_id_err: Result<(), &'static str> = Err("Point ID already in Graph with different coordinates!");
        let expected_add_edge_err: Result<(), &'static str> = Err("Point(s) not in Graph!");

        assert_eq!(g.add_node(p1_diff_coords), expected_add_node_diff_id_err);
        assert_eq!(g.add_edge(&p1, &p1_diff_coords), expected_add_edge_err);
    }
}
