use pretokie::mask::Gpt2;
fn main() {
    let tests = &[
        " />\n", ". \n", "]] s", "]]s", " \n", "\n ", "  ", " a", " 1",
        " .", ".\n", "\n\n", "a1", "1a", "a.", ".a", "a b", "a\nb",
        "don't", "it's", "we'll", "a'b", "'s", "a's", "a'sb",
        " \n  \nHello", "abc123", "  hello",
    ];
    for t in tests {
        let pieces: Vec<&str> = Gpt2::new(t).collect();
        println!("{:>20?} → {:?}", t, pieces);
    }
}
