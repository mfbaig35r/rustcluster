//! Top-k selection over a slice of scalar scores.
//!
//! Two flavors:
//! - `topk_smallest` for L2 distance (smaller is better)
//! - `topk_largest` for inner product (larger is better)
//!
//! The implementation uses `select_nth_unstable_by` to do the partition in
//! O(n) average time, then sorts the resulting prefix in O(k log k). For
//! the typical k ≪ n case this is faster than a streaming heap because we
//! already have the whole distance vector materialized after the GEMM.
//!
//! `f32` NaN inputs panic (via `partial_cmp().unwrap()`). The PyO3 boundary
//! is responsible for rejecting NaN before it reaches here.

use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Keep the k smallest values (L2 distance).
    Smallest,
    /// Keep the k largest values (inner product).
    Largest,
}

impl Direction {
    /// Sentinel value used to pad the result when there are fewer than k
    /// candidates. `+inf` for Smallest (so it sorts last), `-inf` for Largest.
    pub fn sentinel(self) -> f32 {
        match self {
            Direction::Smallest => f32::INFINITY,
            Direction::Largest => f32::NEG_INFINITY,
        }
    }

    fn cmp(self, a: f32, b: f32) -> Ordering {
        match self {
            Direction::Smallest => a.partial_cmp(&b).expect("NaN in score input"),
            Direction::Largest => b.partial_cmp(&a).expect("NaN in score input"),
        }
    }
}

/// Select the top-k entries from `scores`, returning `(values, positions)`
/// of length exactly `k`. Slots beyond `scores.len()` are padded with the
/// direction sentinel and position `-1`.
///
/// `skip` is an optional internal position to exclude from the result
/// (used to implement `exclude_self` when the query equals an indexed point).
pub fn topk(
    scores: &[f32],
    k: usize,
    direction: Direction,
    skip: Option<usize>,
) -> (Vec<f32>, Vec<i64>) {
    let n = scores.len();
    let effective_n = if skip.is_some_and(|s| s < n) {
        n - 1
    } else {
        n
    };
    let take = k.min(effective_n);

    // Build (score, position) pairs, optionally dropping the skipped position.
    let mut indexed: Vec<(f32, i64)> = if let Some(s) = skip {
        scores
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if i == s { None } else { Some((v, i as i64)) })
            .collect()
    } else {
        scores
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as i64))
            .collect()
    };

    let mut values = vec![direction.sentinel(); k];
    let mut positions = vec![-1i64; k];

    if take == 0 {
        return (values, positions);
    }

    let cmp = |a: &(f32, i64), b: &(f32, i64)| direction.cmp(a.0, b.0);

    if take < indexed.len() {
        indexed.select_nth_unstable_by(take, cmp);
        indexed[..take].sort_by(cmp);
    } else {
        indexed.sort_by(cmp);
    }

    for (i, (v, p)) in indexed.iter().take(take).enumerate() {
        values[i] = *v;
        positions[i] = *p;
    }

    (values, positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smallest_basic() {
        let scores = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let (v, p) = topk(&scores, 3, Direction::Smallest, None);
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
        assert_eq!(p, vec![1, 3, 2]);
    }

    #[test]
    fn largest_basic() {
        let scores = vec![0.1, 0.9, 0.5, 0.8, 0.3];
        let (v, p) = topk(&scores, 2, Direction::Largest, None);
        assert_eq!(v, vec![0.9, 0.8]);
        assert_eq!(p, vec![1, 3]);
    }

    #[test]
    fn k_larger_than_n() {
        let scores = vec![3.0, 1.0];
        let (v, p) = topk(&scores, 5, Direction::Smallest, None);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 3.0);
        assert_eq!(v[2], f32::INFINITY);
        assert_eq!(p[0], 1);
        assert_eq!(p[1], 0);
        assert_eq!(p[2], -1);
    }

    #[test]
    fn skip_position() {
        let scores = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let (v, p) = topk(&scores, 3, Direction::Smallest, Some(1));
        assert_eq!(v, vec![2.0, 3.0, 4.0]);
        assert_eq!(p, vec![3, 2, 4]);
    }

    #[test]
    fn k_zero_returns_empty_padded() {
        let scores = vec![1.0, 2.0];
        let (v, p) = topk(&scores, 0, Direction::Smallest, None);
        assert!(v.is_empty());
        assert!(p.is_empty());
    }

    #[test]
    fn empty_scores() {
        let (v, p) = topk(&[], 3, Direction::Smallest, None);
        assert_eq!(v, vec![f32::INFINITY; 3]);
        assert_eq!(p, vec![-1; 3]);
    }
}
