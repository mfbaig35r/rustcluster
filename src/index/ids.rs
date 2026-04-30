//! External u64 id ↔ internal usize position bijection for flat indexes.
//!
//! Two modes:
//! - **Sequential** (default): id `i` lives at internal position `i`. No
//!   storage overhead.
//! - **Explicit**: ids supplied via `add_with_ids` are stored in a `Vec<u64>`
//!   with a parallel `HashMap<u64, usize>` for `id → position` lookup.
//!
//! An index commits to one mode on first insert. Mixing modes is an error.

use std::collections::HashMap;

use crate::error::ClusterError;

#[derive(Debug, Clone)]
pub enum IdMap {
    /// No vectors have been added yet — caller can pick either mode.
    Empty,
    /// Sequential mode: external id `i` == internal position `i`.
    Sequential { ntotal: usize },
    /// Explicit mode: `positions[id]` gives internal position; `ids[i]`
    /// gives external id at position `i`.
    Explicit {
        ids: Vec<u64>,
        positions: HashMap<u64, usize>,
    },
}

impl IdMap {
    pub fn new() -> Self {
        IdMap::Empty
    }

    pub fn ntotal(&self) -> usize {
        match self {
            IdMap::Empty => 0,
            IdMap::Sequential { ntotal } => *ntotal,
            IdMap::Explicit { ids, .. } => ids.len(),
        }
    }

    /// External id for an internal position. Always returns a valid u64.
    pub fn external(&self, position: usize) -> u64 {
        match self {
            IdMap::Empty => panic!("external() called on empty IdMap"),
            IdMap::Sequential { .. } => position as u64,
            IdMap::Explicit { ids, .. } => ids[position],
        }
    }

    /// Append `n` sequential ids. Errors if the map is in Explicit mode.
    pub fn extend_sequential(&mut self, n: usize) -> Result<(), ClusterError> {
        match self {
            IdMap::Empty => {
                *self = IdMap::Sequential { ntotal: n };
                Ok(())
            }
            IdMap::Sequential { ntotal } => {
                *ntotal += n;
                Ok(())
            }
            IdMap::Explicit { .. } => Err(ClusterError::IndexIdModeConflict(
                "index is in explicit-id mode; use add_with_ids".to_string(),
            )),
        }
    }

    /// Append explicit u64 ids. Errors if the map is in Sequential mode or
    /// any id already exists.
    pub fn extend_explicit(&mut self, new_ids: &[u64]) -> Result<(), ClusterError> {
        match self {
            IdMap::Empty => {
                let mut positions = HashMap::with_capacity(new_ids.len());
                for (pos, &id) in new_ids.iter().enumerate() {
                    if positions.insert(id, pos).is_some() {
                        return Err(ClusterError::IndexDuplicateId(id));
                    }
                }
                *self = IdMap::Explicit {
                    ids: new_ids.to_vec(),
                    positions,
                };
                Ok(())
            }
            IdMap::Explicit { ids, positions } => {
                let base = ids.len();
                ids.reserve(new_ids.len());
                positions.reserve(new_ids.len());
                for (offset, &id) in new_ids.iter().enumerate() {
                    if positions.insert(id, base + offset).is_some() {
                        // Roll back the partial insert before erroring out.
                        for (rb_offset, &rb_id) in new_ids.iter().enumerate().take(offset) {
                            positions.remove(&rb_id);
                            // Restore overwritten mappings on collision is
                            // unnecessary because we error on first dup.
                            let _ = rb_offset;
                        }
                        return Err(ClusterError::IndexDuplicateId(id));
                    }
                    ids.push(id);
                }
                Ok(())
            }
            IdMap::Sequential { .. } => Err(ClusterError::IndexIdModeConflict(
                "index is in sequential-id mode; use add() not add_with_ids()".to_string(),
            )),
        }
    }
}

impl Default for IdMap {
    fn default() -> Self {
        IdMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_extends() {
        let mut m = IdMap::new();
        m.extend_sequential(3).unwrap();
        m.extend_sequential(2).unwrap();
        assert_eq!(m.ntotal(), 5);
        assert_eq!(m.external(0), 0);
        assert_eq!(m.external(4), 4);
    }

    #[test]
    fn explicit_extends() {
        let mut m = IdMap::new();
        m.extend_explicit(&[100, 200, 300]).unwrap();
        m.extend_explicit(&[400]).unwrap();
        assert_eq!(m.ntotal(), 4);
        assert_eq!(m.external(0), 100);
        assert_eq!(m.external(3), 400);
    }

    #[test]
    fn cannot_mix_modes() {
        let mut m = IdMap::new();
        m.extend_sequential(2).unwrap();
        assert!(m.extend_explicit(&[5]).is_err());

        let mut m = IdMap::new();
        m.extend_explicit(&[10, 20]).unwrap();
        assert!(m.extend_sequential(1).is_err());
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut m = IdMap::new();
        m.extend_explicit(&[1, 2, 3]).unwrap();
        assert!(m.extend_explicit(&[4, 2]).is_err());
    }
}
