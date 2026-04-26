//! Tests with example functions.

use nalgebra::storage::Owned;
use nalgebra::*;

use crate::{LeastSquaresProblem, LevenbergMarquardt, SparseJacobian};

#[derive(Clone)]
pub struct LinearFullRank {
    pub params: OVector<f64, U5>,
    pub m: usize,
}

impl LinearFullRank {
    fn new(params: OVector<f64, U5>, m: usize) -> Self {
        Self { params, m }
    }
}

impl LeastSquaresProblem<f64, Dyn, U5> for LinearFullRank {
    type ParameterStorage = Owned<f64, U5>;
    type ResidualStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, params: &OVector<f64, U5>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U5> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, Dyn>> {
        let m = Dyn::from_usize(self.m);
        let u1 = Dim::from_usize(1);
        let mut residuals = OVector::<f64, Dyn>::from_element_generic(
            m,
            u1,
            -2. * self.params.sum() / self.m as f64 - 1.,
        );
        for (el, p) in residuals
            .rows_range_mut(..5)
            .iter_mut()
            .zip(self.params.iter())
        {
            *el += p;
        }
        Some(residuals)
    }

    fn jacobian(&self) -> Option<SparseJacobian<f64>> {
        let m = Dyn::from_usize(self.m);
        let u5 = U5;
        let mut jacobian = OMatrix::from_element_generic(m, u5, -2. / self.m as f64);
        for i in 0..5 {
            jacobian[(i, i)] += 1.;
        }
        Some(SparseJacobian::from_dense(jacobian))
    }
}

#[test]
fn test_sparse_solver_smoke() {
    let problem = LinearFullRank::new(OVector::<f64, U5>::from_element(1.0), 200);
    let (_problem, report) = LevenbergMarquardt::new()
        .with_patience(500)
        .minimize(problem);
    assert!(!report.termination.was_usage_issue());
    assert!(report.objective_function.is_finite());
}
