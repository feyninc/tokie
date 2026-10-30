//! BERT WordPiece. The longest vocabulary prefix wins. A miss on the first
//! character makes the whole word `unk`.

/// Append the ids for `word`. `prefix` is the continuation mark, `##` for BERT.
/// `scratch` holds one continuation candidate at a time.
pub fn wordpiece(
    word: &str,
    unk: u32,
    max_chars: usize,
    prefix: &str,
    scratch: &mut String,
    mut vocab: impl FnMut(&str) -> Option<u32>,
    out: &mut Vec<u32>,
) {
    if word.chars().count() > max_chars {
        out.push(unk);
        return;
    }
    let bytes = word.as_bytes();
    let mut ids = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        let mut end = bytes.len();
        let mut found = None;
        while end > start {
            if !word.is_char_boundary(start) || !word.is_char_boundary(end) {
                end -= 1;
                continue;
            }
            let piece = if start == 0 {
                &word[start..end]
            } else {
                scratch.clear();
                scratch.push_str(prefix);
                scratch.push_str(&word[start..end]);
                scratch.as_str()
            };
            if let Some(id) = vocab(piece) {
                found = Some((end, id));
                break;
            }
            end -= 1;
        }
        let Some((next, id)) = found else {
            out.push(unk);
            return;
        };
        ids.push(id);
        start = next;
    }
    out.extend_from_slice(&ids);
}

/// Byte trie of the WordPiece vocab. Continuation tokens already contain the
/// prefix (`##ing`), so a match walks that prefix once and then the suffix.
/// The longest char-aligned path wins, which is the same search as probing
/// the string map from the end of the word.
pub struct VocabTrie {
    nodes: Vec<TrieNode>,
    edges: Vec<TrieEdge>,
    /// `wide_at[node]` is the start of a 256-wide child row, or [`NO_ID`]
    /// when the node is narrow enough that a binary search of `edges` is
    /// shorter. Wide rows are the nodes a walk actually revisits: the root
    /// and the continuation prefix, whose fanout is the whole alphabet.
    wide_at: Vec<u32>,
    wide: Vec<u32>,
    rows: Vec<(String, u32)>,
}

struct TrieNode {
    edge_at: u32,
    edge_n: u16,
    id: u32,
}

struct TrieEdge {
    byte: u8,
    to: u32,
}

const NO_ID: u32 = u32::MAX;

impl VocabTrie {
    pub fn from_rows(rows: Vec<(String, u32)>) -> Self {
        let mut built = vec![BuildNode {
            kids: std::collections::BTreeMap::new(),
            id: NO_ID,
        }];
        for (token, id) in &rows {
            let mut at = 0u32;
            let bytes = token.as_bytes();
            for (i, &byte) in bytes.iter().enumerate() {
                let last = i + 1 == bytes.len();
                if let Some(&child) = built[at as usize].kids.get(&byte) {
                    at = child;
                    if last {
                        built[at as usize].id = *id;
                    }
                } else {
                    let child = built.len() as u32;
                    built[at as usize].kids.insert(byte, child);
                    built.push(BuildNode {
                        kids: std::collections::BTreeMap::new(),
                        id: if last { *id } else { NO_ID },
                    });
                    at = child;
                }
            }
        }
        let mut edges = Vec::new();
        let mut nodes = Vec::with_capacity(built.len());
        for node in &built {
            let edge_at = edges.len() as u32;
            for (&byte, &to) in &node.kids {
                edges.push(TrieEdge { byte, to });
            }
            nodes.push(TrieNode {
                edge_at,
                edge_n: u16::try_from(node.kids.len()).expect("a byte has one edge"),
                id: node.id,
            });
        }
        // A binary search of 8 edges is four dependent loads. One indexed
        // load replaces it. Narrow nodes stay packed: a 256-row for every
        // node would be larger than the vocab and colder than the edges.
        const WIDE_AT: usize = 8;
        let mut wide_at = vec![NO_ID; nodes.len()];
        let mut wide = Vec::new();
        for (index, node) in nodes.iter().enumerate() {
            if node.edge_n as usize >= WIDE_AT {
                let at = wide.len() as u32;
                wide_at[index] = at;
                let start = wide.len();
                wide.resize(start + 256, NO_ID);
                let slice = &edges[node.edge_at as usize..node.edge_at as usize + node.edge_n as usize];
                for edge in slice {
                    wide[start + edge.byte as usize] = edge.to;
                }
            }
        }
        Self {
            nodes,
            edges,
            wide_at,
            wide,
            rows,
        }
    }

    pub fn rows(&self) -> &[(String, u32)] {
        &self.rows
    }

    /// Append the ids for `word`. A miss on any piece makes the whole word `unk`.
    pub fn encode(&self, word: &str, unk: u32, max_chars: usize, prefix: &str, out: &mut Vec<u32>) {
        // A word of at most `max_chars` bytes has at most that many characters.
        if word.len() > max_chars && word.chars().count() > max_chars {
            out.push(unk);
            return;
        }
        let bytes = word.as_bytes();
        let cont = if prefix.is_empty() {
            0
        } else {
            let Some(node) = self.walk(0, prefix.as_bytes()) else {
                out.push(unk);
                return;
            };
            node
        };
        let mark = out.len();
        let mut start = 0usize;
        while start < bytes.len() {
            let node = if start == 0 { 0 } else { cont };
            let Some((end, id)) = self.longest(node, bytes, start) else {
                out.truncate(mark);
                out.push(unk);
                return;
            };
            out.push(id);
            start = end;
        }
    }

    fn walk(&self, mut node: u32, bytes: &[u8]) -> Option<u32> {
        for &byte in bytes {
            node = self.step(node, byte)?;
        }
        Some(node)
    }

    fn longest(&self, mut node: u32, bytes: &[u8], start: usize) -> Option<(usize, u32)> {
        let mut best = None;
        let mut i = start;
        while i < bytes.len() {
            let Some(next) = self.step(node, bytes[i]) else {
                break;
            };
            node = next;
            i += 1;
            if is_boundary(bytes, i) && self.nodes[node as usize].id != NO_ID {
                best = Some((i, self.nodes[node as usize].id));
            }
        }
        best
    }

    fn step(&self, node: u32, byte: u8) -> Option<u32> {
        let wide = self.wide_at[node as usize];
        if wide != NO_ID {
            let to = self.wide[wide as usize + byte as usize];
            return (to != NO_ID).then_some(to);
        }
        let node = &self.nodes[node as usize];
        let edges = &self.edges[node.edge_at as usize..node.edge_at as usize + node.edge_n as usize];
        let at = edges.binary_search_by_key(&byte, |edge| edge.byte).ok()?;
        Some(edges[at].to)
    }
}

struct BuildNode {
    kids: std::collections::BTreeMap<u8, u32>,
    id: u32,
}

fn is_boundary(bytes: &[u8], index: usize) -> bool {
    index == bytes.len() || bytes[index] & 0xC0 != 0x80
}

#[cfg(test)]
mod tests {
    use super::{VocabTrie, wordpiece};

    #[test]
    fn the_trie_matches_the_string_probe_on_a_multibyte_word() {
        let rows = vec![
            ("你".to_string(), 1),
            ("##好".to_string(), 2),
            ("你好".to_string(), 3),
            ("hello".to_string(), 4),
            ("hel".to_string(), 5),
            ("##lo".to_string(), 6),
            ("cat".to_string(), 7),
            ("##s".to_string(), 8),
            ("a".to_string(), 10),
            ("b".to_string(), 11),
            ("d".to_string(), 12),
            ("e".to_string(), 13),
            ("f".to_string(), 14),
            ("g".to_string(), 15),
            ("i".to_string(), 16),
            ("j".to_string(), 17),
        ];
        let trie = VocabTrie::from_rows(rows.clone());
        let mut map = std::collections::HashMap::new();
        for (token, id) in &rows {
            map.insert(token.clone(), *id);
        }
        for word in ["你好", "hello", "你", "xyz", "cats", "a", "j"] {
            let mut slow = Vec::new();
            let mut scratch = String::new();
            wordpiece(word, 99, 100, "##", &mut scratch, |token| map.get(token).copied(), &mut slow);
            let mut fast = Vec::new();
            trie.encode(word, 99, 100, "##", &mut fast);
            assert_eq!(fast, slow, "{word}");
        }
    }
}
