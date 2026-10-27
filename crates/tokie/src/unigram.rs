//! Unigram. The segmentation is the one with the highest sum of scores.
//!
//! [`Lattice`] is the one the model runs. It follows tokenizers v1. Pieces
//! start on character boundaries. A tie keeps the path written first. A
//! character with no one-character piece costs `min(score) - 10`. Consecutive
//! unknowns collapse to one id.

/// One vocabulary piece. `bytes` is the raw span it covers.
#[derive(Clone, Copy, Debug)]
pub struct Piece<'a> {
    pub bytes: &'a [u8],
    pub id: u32,
    pub score: f64,
}

/// Append the ids for `text`. A byte no piece covers becomes `unk` on its own.
pub fn unigram(text: &[u8], pieces: &[Piece<'_>], unk: u32, out: &mut Vec<u32>) {
    let n = text.len();
    let mut best = vec![f64::NEG_INFINITY; n + 1];
    let mut from = vec![0usize; n + 1];
    let mut id_at = vec![unk; n + 1];
    best[0] = 0.0;
    for i in 0..n {
        if best[i] == f64::NEG_INFINITY {
            continue;
        }
        let mut covered = false;
        for piece in pieces {
            let end = i + piece.bytes.len();
            if end <= n && &text[i..end] == piece.bytes {
                let score = best[i] + piece.score;
                if score > best[end] {
                    best[end] = score;
                    from[end] = i;
                    id_at[end] = piece.id;
                }
                covered = true;
            }
        }
        if !covered {
            let end = i + 1;
            let score = best[i];
            if score > best[end] {
                best[end] = score;
                from[end] = i;
                id_at[end] = unk;
            }
        }
    }
    if best[n] == f64::NEG_INFINITY {
        out.push(unk);
        return;
    }
    let mut ends = Vec::new();
    let mut at = n;
    while at > 0 {
        ends.push(at);
        at = from[at];
    }
    for end in ends.into_iter().rev() {
        out.push(id_at[end]);
    }
}

const NO_ID: u32 = u32::MAX;
const UNK_PENALTY: f64 = 10.0;

struct BuildNode {
    kids: std::collections::BTreeMap<u8, u32>,
    id: u32,
    score: f64,
}

struct Node {
    edge_at: u32,
    edge_n: u16,
    id: u32,
    score: f64,
}

struct Edge {
    byte: u8,
    to: u32,
}

#[derive(Clone, Copy)]
struct Slot {
    start: u32,
    id: u32,
    score: f64,
    set: bool,
}

/// Vocabulary trie used by [`crate::model::Model`].
pub(crate) struct Lattice {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    unk: u32,
    unk_score: f64,
}

impl Lattice {
    pub(crate) fn build(pieces: Vec<(String, f64)>, unk: u32) -> Self {
        let mut min_score = f64::INFINITY;
        let mut built = vec![BuildNode {
            kids: std::collections::BTreeMap::new(),
            id: NO_ID,
            score: 0.0,
        }];
        for (id, (token, score)) in pieces.into_iter().enumerate() {
            if score < min_score {
                min_score = score;
            }
            if token.is_empty() {
                continue;
            }
            let mut at = 0u32;
            let bytes = token.into_bytes();
            let len = bytes.len();
            for (i, byte) in bytes.into_iter().enumerate() {
                let last = i + 1 == len;
                if let Some(&child) = built[at as usize].kids.get(&byte) {
                    at = child;
                    if last && built[at as usize].id == NO_ID {
                        built[at as usize].id = id as u32;
                        built[at as usize].score = score;
                    }
                } else {
                    let child = built.len() as u32;
                    built[at as usize].kids.insert(byte, child);
                    built.push(BuildNode {
                        kids: std::collections::BTreeMap::new(),
                        id: if last { id as u32 } else { NO_ID },
                        score: if last { score } else { 0.0 },
                    });
                    at = child;
                }
            }
        }
        if min_score == f64::INFINITY {
            min_score = 0.0;
        }
        let mut edges = Vec::new();
        let mut nodes = Vec::with_capacity(built.len());
        for node in &built {
            let edge_at = edges.len() as u32;
            for (&byte, &to) in &node.kids {
                edges.push(Edge { byte, to });
            }
            nodes.push(Node {
                edge_at,
                edge_n: u16::try_from(node.kids.len()).expect("a byte has one edge"),
                id: node.id,
                score: node.score,
            });
        }
        Self {
            nodes,
            edges,
            unk,
            unk_score: min_score - UNK_PENALTY,
        }
    }

    pub(crate) fn unk(&self) -> u32 {
        self.unk
    }

    pub(crate) fn unk_score(&self) -> f64 {
        self.unk_score
    }

    pub(crate) fn nodes(&self) -> impl Iterator<Item = (u32, u16, u32, f64)> + '_ {
        self.nodes
            .iter()
            .map(|node| (node.edge_at, node.edge_n, node.id, node.score))
    }

    pub(crate) fn edges(&self) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.edges.iter().map(|edge| (edge.byte, edge.to))
    }

    pub(crate) fn from_parts(
        nodes: Vec<(u32, u16, u32, f64)>,
        edges: Vec<(u8, u32)>,
        unk: u32,
        unk_score: f64,
    ) -> Self {
        Self {
            nodes: nodes
                .into_iter()
                .map(|(edge_at, edge_n, id, score)| Node {
                    edge_at,
                    edge_n,
                    id,
                    score,
                })
                .collect(),
            edges: edges
                .into_iter()
                .map(|(byte, to)| Edge { byte, to })
                .collect(),
            unk,
            unk_score,
        }
    }

    /// Append the ids for `text`.
    pub(crate) fn segment(&self, text: &[u8], out: &mut Vec<u32>) {
        let n = text.len();
        if n == 0 {
            return;
        }
        let mut ends = vec![
            Slot {
                start: 0,
                id: 0,
                score: 0.0,
                set: false,
            };
            n + 1
        ];
        let mut starts = 0usize;
        while starts < n {
            let till = ends[starts].score;
            let mblen = char_len(text[starts]);
            let mut has_single = false;
            let mut node = 0u32;
            let mut len = 0usize;
            while starts + len < n {
                let Some(next) = self.step(node, text[starts + len]) else {
                    break;
                };
                node = next;
                len += 1;
                let leaf = &self.nodes[node as usize];
                if leaf.id == NO_ID {
                    continue;
                }
                let end = starts + len;
                let candidate = till + leaf.score;
                let slot = &mut ends[end];
                if !slot.set || candidate > slot.score {
                    *slot = Slot {
                        start: starts as u32,
                        id: leaf.id,
                        score: candidate,
                        set: true,
                    };
                }
                if len == mblen {
                    has_single = true;
                }
            }
            if !has_single {
                let end = starts + mblen;
                let candidate = till + self.unk_score;
                let slot = &mut ends[end];
                if !slot.set || candidate > slot.score {
                    *slot = Slot {
                        start: starts as u32,
                        id: self.unk,
                        score: candidate,
                        set: true,
                    };
                }
            }
            starts += mblen;
        }
        let mut path = Vec::new();
        let mut at = n;
        while at > 0 {
            let slot = &ends[at];
            debug_assert!(slot.set);
            let start = slot.start as usize;
            path.push((start, at, slot.id));
            at = start;
        }
        path.reverse();
        let mut i = 0;
        while i < path.len() {
            if path[i].2 != self.unk {
                out.push(path[i].2);
                i += 1;
                continue;
            }
            let mut j = i + 1;
            while j < path.len() && path[j].2 == self.unk {
                j += 1;
            }
            let bytes = &text[path[i].0..path[j - 1].1];
            out.push(self.exact(bytes).unwrap_or(self.unk));
            i = j;
        }
    }

    fn step(&self, node: u32, byte: u8) -> Option<u32> {
        let node = &self.nodes[node as usize];
        let edges = &self.edges[node.edge_at as usize..node.edge_at as usize + node.edge_n as usize];
        let at = edges.binary_search_by_key(&byte, |edge| edge.byte).ok()?;
        Some(edges[at].to)
    }

    fn exact(&self, text: &[u8]) -> Option<u32> {
        let mut node = 0u32;
        for &byte in text {
            node = self.step(node, byte)?;
        }
        let id = self.nodes[node as usize].id;
        (id != NO_ID).then_some(id)
    }
}

fn char_len(byte: u8) -> usize {
    if byte < 0x80 {
        1
    } else if byte < 0xE0 {
        2
    } else if byte < 0xF0 {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::Lattice;

    #[test]
    fn the_best_path_matches_the_tokenizers_example() {
        let pieces = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), 0.0),
            ("b".to_string(), 0.0),
            ("c".to_string(), 0.0),
            ("d".to_string(), 0.0),
            ("cd".to_string(), 1.0),
            ("ab".to_string(), 2.0),
            ("abc".to_string(), 5.0),
            ("abcd".to_string(), 10.0),
        ];
        let lattice = Lattice::build(pieces, 0);
        let mut ids = Vec::new();
        lattice.segment(b"abcdacdxx", &mut ids);
        assert_eq!(ids, vec![8, 1, 5, 0]);
    }

    #[test]
    fn an_equal_total_keeps_the_earlier_path() {
        let pieces = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), 0.0),
            ("b".to_string(), 1.0),
            ("ab".to_string(), 1.0),
        ];
        let lattice = Lattice::build(pieces, 0);
        let mut ids = Vec::new();
        lattice.segment(b"ab", &mut ids);
        assert_eq!(ids, vec![3]);
    }
}
