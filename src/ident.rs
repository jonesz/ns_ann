use super::lsh::IdentifierMethod;

/// Given a usize bin, return another bin that is 1-hamming distance away.
pub(super) fn similar_bin<const N: usize>(bin: usize, ident: &IdentifierMethod) -> usize {
    todo!();
}

/// Given a `usize` bin, return an iterator containing the nearest
/// `I` identifiers.
pub trait Identifier<'a, I> {
    fn ann(x: usize) -> dyn Iterator<Item = &'a I>;
}
