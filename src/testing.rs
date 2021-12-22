/// Check if input is sorted. Used for testing.
///
/// From https://stackoverflow.com/questions/51272571/how-do-i-check-if-a-slice-is-sorted.
#[allow(dead_code)]
pub fn is_sorted<T>(data: &[T]) -> bool
where
    T: std::cmp::PartialOrd,
{
    data.windows(2).all(|w| w[0] <= w[1])
}
