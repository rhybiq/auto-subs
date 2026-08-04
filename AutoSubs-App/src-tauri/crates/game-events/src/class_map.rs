//! Maps YAMNet's 521 AudioSet output classes (vendored in
//! `assets/yamnet_labels.txt`, one label per line in model output order) to
//! the small set of [`GameEventKind`]s this crate cares about.
//!
//! Matches on the class *label string* rather than a hardcoded index, so a
//! future model re-export with reordered/added classes doesn't silently
//! mismap. AudioSet has no class for CS2's bomb-plant/defuse beep — that's
//! handled separately by [`crate::beep_heuristic`].

use crate::GameEventKind;

const LABELS: &str = include_str!("../assets/yamnet_labels.txt");

const GUNFIRE_LABELS: &[&str] = &[
    "Gunshot, gunfire",
    "Machine gun",
    "Fusillade",
    "Artillery fire",
    "Cap gun",
];

const EXPLOSION_LABELS: &[&str] = &["Explosion", "Boom"];

/// Index `i` of the returned vec corresponds to model output class `i`.
pub fn class_kind_map() -> Vec<Option<GameEventKind>> {
    LABELS
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|label| {
            if GUNFIRE_LABELS.contains(&label) {
                Some(GameEventKind::Gunfire)
            } else if EXPLOSION_LABELS.contains(&label) {
                Some(GameEventKind::Explosion)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_all_521_classes_with_expected_counts() {
        let map = class_kind_map();
        assert_eq!(map.len(), 521);
        assert_eq!(
            map.iter().filter(|k| **k == Some(GameEventKind::Gunfire)).count(),
            GUNFIRE_LABELS.len()
        );
        assert_eq!(
            map.iter().filter(|k| **k == Some(GameEventKind::Explosion)).count(),
            EXPLOSION_LABELS.len()
        );
    }
}
