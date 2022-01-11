pub trait Criterion: CloneCriterion {
    fn proxy_gain(
        &self,
        sum_left: f64,
        sum_right: f64,
        n_left: usize,
        n_right: usize,
    ) -> f64;
    fn gain(&self, proxy_gain: f64, sum: f64, n: usize) -> f64;
}

// https://users.rust-lang.org/t/solved-is-it-possible-to-clone-a-boxed-trait-object/1714/7
trait CloneCriterion {
    fn clone_criterion<'a>(&self) -> Box<dyn Criterion>;
}

impl<T> CloneCriterion for T
where
    T: Criterion + Clone + 'static,
{
    fn clone_criterion(&self) -> Box<dyn Criterion> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Criterion> {
    fn clone(&self) -> Self {
        self.clone_criterion()
    }
}


#[derive(Clone)]
pub struct GiniCriterion {}

impl GiniCriterion {
    pub fn new() -> Self {
        GiniCriterion {}
    }
}

impl Criterion for GiniCriterion {
    // The Gini impurity for two class classification at a node is 1 - p_0 * p_0 - p_1 * p_1
    // = 1 - 2 * (sum / n) (sum / n - 1). The Gini improvement at a node is 
    // Gini(node) - (n_left / n) * Gini(left) - (n_right / n) * Gini(right) (yes, weighted)
    // = - 2 / n * [ sum (sum / n - 1) - sum_left (sum_left / n_left) - sum_right (sum_right / n_left)
    // = - 2 / n [ sum * sum / n - sum_left * sum_left / n_left - sum_right * sum_right / n_right ]
    // We can use sum_left * sum_left / n_left + sum_rigth * sum_right / n_right as a proxy for
    // the improvement of the gini of the node. This is the same as for MSE.
    fn proxy_gain(
        &self,
        sum_left: f64,
        sum_right: f64,
        n_left: usize,
        n_right: usize,
    ) -> f64 {
        sum_left * sum_left / n_left as f64 + sum_right * sum_right / n_right as f64
    }

    // This is also equal to MSE up to the constant factor of 2.
    fn gain(&self, proxy_gain: f64, sum: f64, n: usize) -> f64 {
        2. * (-(sum / n as f64).powi(2) + proxy_gain / n as f64)
    }
}

#[derive(Clone)]
pub struct MSECriterion {}

impl MSECriterion {
    pub fn new() -> Self {
        MSECriterion {}
    }
}
impl Criterion for MSECriterion {
    // Inspired by https://github.com/scikit-learn/scikit-learn/blob/cb4688ad15f052d7c55b1d3f09ee65bc3d5bb24b/sklearn/tree/_criterion.pyx#L900
    // The RSS after fitting a mean to (u, v] is L(u, v) = sum_{i=u+1}^v (y_i - mean)^2.
    // Here mean = 1 / (v - u) * sum_{i=u+1}^v y_i.
    // Then L(u, v) = \sum_{i=u+1}^v y_i^2 - 1 / (v - u) (sum_{i=u+1} y_i)^2.
    // The node impurity splitting at s is
    // L(start, s) + L(s, stop) = \sum_{i=start+1}^stop y_i^2 - 1 / (s - start) (sum_{i=start+1}^v y_i)^2 - 1 / (stop - s) (sum_{i=s+1}^stop y_i)^2.
    // The first term is independent of s, so does not need to be calculated to find the best split.
    // We find the maximum of the negative of the second term, which is the proxy gain.
    fn proxy_gain(
        &self,
        sum_left: f64,
        sum_right: f64,
        n_left: usize,
        n_right: usize,
    ) -> f64 {
        sum_left * sum_left / n_left as f64 + sum_right * sum_right / n_right as f64
    }

    fn gain(&self, proxy_gain: f64, sum: f64, n: usize) -> f64 {
        -(sum / n as f64).powi(2) + proxy_gain / n as f64
    }
}
