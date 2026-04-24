use crate::{LeastSquaresProblem, SparseJacobian};
use crate::qr::{LinearLeastSquaresDiagonalProblem, PivotedQR};
use crate::trust_region::{LMParameter, determine_lambda_and_parameter_update};
use crate::utils::{enorm, epsmch};
use alloc::{collections::BTreeMap, vec::Vec};
use nalgebra::{
    DefaultAllocator, Dim, DimMax, DimMaximum, DimMin, OVector, RealField, Vector,
    allocator::{Allocator, Reallocator},
    convert,
};
use num_traits::Float;

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::redundant_clone
)]
pub(crate) mod test_examples;
#[cfg(test)]
mod test_helpers;
#[cfg(test)]
mod test_init_step;
#[cfg(test)]
#[allow(clippy::float_cmp, clippy::clone_on_copy, clippy::redundant_clone)]
mod test_update_diag;

#[derive(PartialEq, Eq, Debug)]
/// Reasons for terminating the minimization.
pub enum TerminationReason {
    /// The residual or Jacobian computation was not successful, it returned `None`.
    User(&'static str),
    /// Encountered `NaN` or `$\pm\infty$`.
    Numerical(&'static str),
    /// The residuals are literally zero.
    ResidualsZero,
    /// The residuals vector and the Jacobian columns are almost orthogonal.
    ///
    /// This is the `gtol` termination criterion.
    Orthogonal,
    /// The `ftol` or `xtol` criterion was fulfilled.
    Converged { ftol: bool, xtol: bool },
    /// The bound for `ftol`, `xtol` or `gtol` was set so low that the
    /// test passed with the machine epsilon but not with the actual
    /// bound. This means you must increase the bound.
    NoImprovementPossible(&'static str),
    /// Maximum number of function evaluations was hit.
    LostPatience,
    /// The number of parameters `$n$` is zero.
    NoParameters,
    /// The number of residuals `$m$` is zero.
    NoResiduals,
    /// The shape of the computed residuals or Jacobian is not correct.
    WrongDimensions(&'static str),
}

impl TerminationReason {
    /// Compute whether the outcome is considered successful.
    ///
    /// This does not necessarily mean we have a minimizer.
    /// Some termination criteria are approximations for necessary
    /// optimality conditions or check limitations due to
    /// floating point arithmetic.
    pub fn was_successful(&self) -> bool {
        matches!(
            self,
            TerminationReason::ResidualsZero
                | TerminationReason::Orthogonal
                | TerminationReason::Converged { .. }
        )
    }

    /// A fundamental assumptions was not met.
    ///
    /// For example if the number of residuals changed.
    pub fn was_usage_issue(&self) -> bool {
        matches!(
            self,
            TerminationReason::NoParameters
                | TerminationReason::NoResiduals
                | TerminationReason::NoImprovementPossible(_)
                | TerminationReason::WrongDimensions(_)
        )
    }
}

#[derive(Debug)]
/// Information about the minimization.
///
/// Use this to inspect the minimization process. Most importantly
/// you may want to check if there was a failure.
pub struct MinimizationReport<F: RealField> {
    pub termination: TerminationReason,
    /// Number of residuals which were computed.
    pub number_of_evaluations: usize,
    /// Contains the value of `$f(\vec{x})$`.
    pub objective_function: F,
}

/// Levenberg-Marquardt optimization algorithm.
///
/// See the [module documentation](index.html) for a usage example.
///
/// The runtime and termination behavior can be controlled by various hyperparameters.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LevenbergMarquardt<F> {
    ftol: F,
    xtol: F,
    gtol: F,
    stepbound: F,
    patience: usize,
    scale_diag: bool,
    use_sparse_solver: bool,
    sparse_schur_camera_variables: Option<usize>,
}

impl<F: RealField + Float> Default for LevenbergMarquardt<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: RealField + Float> LevenbergMarquardt<F> {
    pub fn new() -> Self {
        if cfg!(feature = "minpack-compat") {
            let user_tol = convert(1.49012e-08);
            Self {
                ftol: user_tol,
                xtol: user_tol,
                gtol: F::zero(),
                stepbound: convert(100.0),
                patience: 100,
                scale_diag: true,
                use_sparse_solver: false,
                sparse_schur_camera_variables: None,
            }
        } else {
            let user_tol = F::default_epsilon() * convert(30.0);
            Self {
                ftol: user_tol,
                xtol: user_tol,
                gtol: user_tol,
                stepbound: convert(100.0),
                patience: 100,
                scale_diag: true,
                use_sparse_solver: false,
                sparse_schur_camera_variables: None,
            }
        }
    }

    /// Set the relative error desired in the objective function `$f$`.
    ///
    /// Termination occurs when both the actual and
    /// predicted relative reductions for `$f$` are at most `ftol`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{ftol} < 0$`.
    #[must_use]
    pub fn with_ftol(self, ftol: F) -> Self {
        assert!(!ftol.is_negative(), "ftol must be >= 0");
        Self { ftol, ..self }
    }

    /// Set relative error between last two approximations.
    ///
    /// Termination occurs when the relative error between
    /// two consecutive iterates is at most `xtol`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{xtol} < 0$`.
    #[must_use]
    pub fn with_xtol(self, xtol: F) -> Self {
        assert!(!xtol.is_negative(), "xtol must be >= 0");
        Self { xtol, ..self }
    }

    /// Set orthogonality desired between the residual vector and its derivative.
    ///
    /// Termination occurs when the cosine of the angle
    /// between the residual vector `$\vec{r}$` and any column of the Jacobian `$\mathbf{J}$` is at
    /// most `gtol` in absolute value.
    ///
    /// With other words, the algorithm will terminate if
    /// ```math
    ///   \cos\bigl(\sphericalangle (\mathbf{J}\vec{e}_i, \vec{r})\bigr) =
    ///   \frac{|(\mathbf{J}^\top \vec{r})_i|}{\|\mathbf{J}\vec{e}_i\|\|\vec{r}\|} \leq \texttt{gtol}
    ///   \quad\text{for all }i=1,\ldots,n.
    /// ```
    ///
    /// This is based on the fact that those vectors are orthogonal near the optimum (gradient is zero).
    /// The angle check is scale invariant, whereas checking that
    /// `$\nabla f(\vec{x})\approx \vec{0}$` is not.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{gtol} < 0$`.
    #[must_use]
    pub fn with_gtol(self, gtol: F) -> Self {
        assert!(!gtol.is_negative(), "gtol must be >= 0");
        Self { gtol, ..self }
    }

    /// Shortcut to set `tol` as in MINPACK `LMDER1`.
    ///
    /// Sets `ftol = xtol = tol` and `gtol = 0`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{tol} \leq 0$`.
    #[must_use]
    pub fn with_tol(self, tol: F) -> Self {
        assert!(tol.is_positive(), "tol must > 0");
        Self {
            ftol: tol,
            xtol: tol,
            gtol: F::zero(),
            ..self
        }
    }

    /// Set factor for the initial step bound.
    ///
    /// This bound is set to `$\mathtt{stepbound}\cdot\|\mathbf{D}\vec{x}\|$`
    /// if nonzero, or else to `stepbound` itself. In most cases `stepbound` should lie
    /// in the interval `$[0.1,100]$`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{stepbound} \leq 0$`.
    #[must_use]
    pub fn with_stepbound(self, stepbound: F) -> Self {
        assert!(stepbound.is_positive(), "stepbound must be > 0");
        Self { stepbound, ..self }
    }

    /// Set factor for the maximal number of function evaluations.
    ///
    /// The maximal number of function evaluations is set to
    /// `$\texttt{patience}\cdot(n + 1)$`.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{patience} \leq 0$`.
    #[must_use]
    pub fn with_patience(self, patience: usize) -> Self {
        assert!(patience > 0, "patience must be > 0");
        Self { patience, ..self }
    }

    /// Enable or disable whether the variables will be rescaled internally.
    #[must_use]
    pub fn with_scale_diag(self, scale_diag: bool) -> Self {
        Self { scale_diag, ..self }
    }

    /// Enable the experimental sparse solver path.
    ///
    /// This path avoids materializing a dense `m x n` Jacobian matrix and instead
    /// uses sparse Jacobian products for damped normal-equation solves.
    ///
    /// The default remains `false` to preserve the classic MINPACK-like behavior.
    #[must_use]
    pub fn with_sparse_solver(self, use_sparse_solver: bool) -> Self {
        Self {
            use_sparse_solver,
            ..self
        }
    }

    /// Configure Schur-style elimination for the sparse solver.
    ///
    /// This splits the parameter vector into two blocks:
    /// - camera/state variables in `[0, sparse_schur_camera_variables)`
    /// - point/landmark variables in `[sparse_schur_camera_variables, n)`
    ///
    /// When enabled together with `with_sparse_solver(true)`, the sparse path
    /// uses a Schur-complement solve tailored to bundle-adjustment-like structure.
    #[must_use]
    pub fn with_sparse_schur_camera_variables(self, sparse_schur_camera_variables: usize) -> Self {
        Self {
            sparse_schur_camera_variables: Some(sparse_schur_camera_variables),
            ..self
        }
    }

    /// Try to solve the given least squares problem.
    ///
    /// The parameters of the problem which are set when this function is called
    /// are used as the initial guess for `$\vec{x}$`.
    pub fn minimize<N, M, O>(&self, target: O) -> (O, MinimizationReport<F>)
    where
        N: Dim,
        M: DimMin<N> + DimMax<N>,
        O: LeastSquaresProblem<F, M, N>,
        DefaultAllocator:
            Allocator<N> + Allocator<M, N> + Reallocator<F, M, N, DimMaximum<M, N>, N>,
    {
        if self.use_sparse_solver {
            return self.minimize_sparse(target);
        }

        let (mut lm, mut residuals) = match LM::new(self, target) {
            Err(report) => return report,
            Ok(res) => res,
        };
        let n = lm.x.nrows();
        loop {
            // Build linear least squares problem used for the trust-region subproblem
            let mut lls = {
                let jacobian = match lm.jacobian() {
                    Err(reason) => return lm.into_report(reason),
                    Ok(jacobian) => jacobian,
                };
                if jacobian.cols != n || jacobian.rows != lm.m {
                    return lm.into_report(TerminationReason::WrongDimensions("jacobian"));
                }

                let qr = PivotedQR::new(jacobian.to_dense::<M, N>());
                qr.into_least_squares_diagonal_problem(residuals)
            };

            // Update the diagonal, initialize "delta" in first call
            if let Err(reason) = lm.update_diag(&mut lls) {
                return lm.into_report(reason);
            };

            residuals = loop {
                let param =
                    determine_lambda_and_parameter_update(&mut lls, &lm.diag, lm.delta, lm.lambda);
                let tr_iteration = lm.trust_region_iteration(&mut lls, param);
                match tr_iteration {
                    // successful parameter update, break and recompute Jacobian
                    Ok(Some(residuals)) => break residuals,
                    // terminate (either success or failure)
                    Err(reason) => return lm.into_report(reason),
                    // need another iteration
                    Ok(None) => (),
                }
            };
        }
    }

    fn minimize_sparse<N, M, O>(&self, target: O) -> (O, MinimizationReport<F>)
    where
        N: Dim,
        M: DimMin<N> + DimMax<N>,
        O: LeastSquaresProblem<F, M, N>,
        DefaultAllocator:
            Allocator<N> + Allocator<M, N> + Reallocator<F, M, N, DimMaximum<M, N>, N>,
    {
        let (mut lm, mut residuals) = match LM::new(self, target) {
            Err(report) => return report,
            Ok(res) => res,
        };
        let n = lm.x.nrows();
        let mut lambda = Float::max(F::default_epsilon(), convert(1.0e-6f64));

        loop {
            let jacobian = match lm.jacobian() {
                Err(reason) => return lm.into_report(reason),
                Ok(jacobian) => jacobian,
            };
            if jacobian.cols != n || jacobian.rows != lm.m {
                return lm.into_report(TerminationReason::WrongDimensions("jacobian"));
            }

            let col_norms = sparse_column_norms::<F, N>(&jacobian, n);
            let jt_r = sparse_jt_mul::<F, N>(&jacobian, residuals.as_slice(), n);

            if lm.first_update {
                lm.xnorm = if lm.config.scale_diag {
                    for (d, col_norm) in lm.diag.iter_mut().zip(col_norms.iter()) {
                        *d = if col_norm.is_zero() {
                            F::one()
                        } else {
                            *col_norm
                        };
                    }
                    lm.tmp.cmpy(F::one(), &lm.diag, &lm.x, F::zero());
                    enorm(&lm.tmp)
                } else {
                    enorm(&lm.x)
                };

                lm.delta = if lm.xnorm.is_zero() {
                    lm.config.stepbound
                } else {
                    lm.config.stepbound * lm.xnorm
                };
                lm.first_update = false;
            } else if lm.config.scale_diag {
                for (d, norm) in lm.diag.iter_mut().zip(col_norms.iter()) {
                    *d = Float::max(*norm, *d);
                }
            }

            lm.gnorm = max_scaled_gradient(&jt_r, &col_norms, lm.residuals_norm);
            if lm.gnorm <= lm.config.gtol {
                return lm.into_report(TerminationReason::Orthogonal);
            }

            let mut accepted = false;
            let mut accepted_residuals = None;
            let max_inner = 12usize;

            for _ in 0..max_inner {
                let step = if let Some(n_camera) = self.sparse_schur_camera_variables {
                    if n_camera > 0 && n_camera < n {
                        solve_damped_normal_equations_schur::<F, N>(
                            &jacobian,
                            &jt_r,
                            &lm.diag,
                            lambda,
                            n,
                            n_camera,
                        )
                    } else {
                        solve_damped_normal_equations::<F, N>(
                            &jacobian,
                            &jt_r,
                            &lm.diag,
                            lambda,
                            n,
                        )
                    }
                } else {
                    solve_damped_normal_equations::<F, N>(&jacobian, &jt_r, &lm.diag, lambda, n)
                };

                let pnorm = scaled_norm(&step, &lm.diag);
                if !pnorm.is_finite() && !cfg!(feature = "minpack-compat") {
                    return lm.into_report(TerminationReason::Numerical("subproblem ||Dp||"));
                }

                lm.tmp.copy_from(&lm.x);
                lm.tmp.axpy(-F::one(), &step, F::one());

                lm.target.set_params(&lm.tmp);
                lm.report.number_of_evaluations += 1;

                let new_objective_function;
                let (new_residuals, new_residuals_norm) = if let Some(res) = lm.target.residuals() {
                    if res.nrows() != lm.m {
                        return lm.into_report(TerminationReason::WrongDimensions("residuals"));
                    }
                    let norm = enorm(&res);
                    new_objective_function = norm * norm * convert(0.5);
                    (res, norm)
                } else {
                    return lm.into_report(TerminationReason::User("residuals"));
                };

                let actual_reduction = if new_residuals_norm * convert(0.1f64) < lm.residuals_norm {
                    F::one() - Float::powi(new_residuals_norm / lm.residuals_norm, 2)
                } else {
                    -F::one()
                };

                let j_step = sparse_j_mul::<F, N>(&jacobian, &step);
                let predicted_reduction = predicted_reduction_model(
                    residuals.as_slice(),
                    j_step.as_slice(),
                    lm.residuals_norm,
                );
                let ratio = if predicted_reduction <= F::zero() {
                    F::zero()
                } else {
                    actual_reduction / predicted_reduction
                };

                if ratio > convert(0.0001f64) {
                    accepted = true;
                    accepted_residuals = Some((new_residuals, new_residuals_norm, new_objective_function, step, pnorm));
                    lambda = Float::max(lambda * convert(0.333333333333f64), F::default_epsilon());
                    break;
                }

                lambda *= convert(2.0f64);
                lm.target.set_params(&lm.x);
            }

            if !accepted {
                return lm.into_report(TerminationReason::LostPatience);
            }

            let (new_residuals, new_residuals_norm, new_objective_function, _step, pnorm) =
                accepted_residuals.expect("accepted step must exist");

            core::mem::swap(&mut lm.x, &mut lm.tmp);
            lm.residuals_norm = new_residuals_norm;
            lm.report.objective_function = new_objective_function;
            residuals = new_residuals;

            if lm.config.scale_diag {
                lm.tmp.cmpy(F::one(), &lm.diag, &lm.x, F::zero());
                lm.xnorm = enorm(&lm.tmp);
            } else {
                lm.xnorm = enorm(&lm.x);
            }

            if !cfg!(feature = "minpack-compat") && lm.residuals_norm <= F::min_positive_value() {
                return lm.into_report(TerminationReason::ResidualsZero);
            }

            let xtol_check = lm.delta <= lm.config.xtol * lm.xnorm || pnorm <= lm.config.xtol * (lm.xnorm + lm.config.xtol);
            let ftol_check = lm.report.objective_function <= lm.config.ftol;
            if ftol_check || xtol_check {
                return lm.into_report(TerminationReason::Converged {
                    ftol: ftol_check,
                    xtol: xtol_check,
                });
            }

            if lm.report.number_of_evaluations >= lm.max_fev {
                return lm.into_report(TerminationReason::LostPatience);
            }
        }
    }
}

fn sparse_column_norms<F, N>(jacobian: &SparseJacobian<F>, n: usize) -> OVector<F, N>
where
    F: RealField + Float + Copy,
    N: Dim,
    DefaultAllocator: Allocator<N>,
{
    let mut out = OVector::<F, N>::zeros_generic(Dim::from_usize(n), Dim::from_usize(1));
    for &(_, j, v) in jacobian.entries.iter() {
        if j < n {
            out[j] += v * v;
        }
    }
    out.apply(|x| *x = Float::sqrt(*x));
    out
}

fn sparse_jt_mul<F, N>(jacobian: &SparseJacobian<F>, y: &[F], n: usize) -> OVector<F, N>
where
    F: RealField + Float + Copy,
    N: Dim,
    DefaultAllocator: Allocator<N>,
{
    let mut out = OVector::<F, N>::zeros_generic(Dim::from_usize(n), Dim::from_usize(1));
    for &(i, j, v) in jacobian.entries.iter() {
        if i < y.len() && j < n {
            out[j] += v * y[i];
        }
    }
    out
}

fn sparse_j_mul<F, N>(jacobian: &SparseJacobian<F>, x: &OVector<F, N>) -> alloc::vec::Vec<F>
where
    F: RealField + Float + Copy,
    N: Dim,
    DefaultAllocator: Allocator<N>,
{
    let mut out = alloc::vec![F::zero(); jacobian.rows];
    for &(i, j, v) in jacobian.entries.iter() {
        if i < out.len() && j < x.nrows() {
            out[i] += v * x[j];
        }
    }
    out
}

fn max_scaled_gradient<F, N>(jt_r: &OVector<F, N>, col_norms: &OVector<F, N>, rnorm: F) -> F
where
    F: RealField + Float + Copy,
    N: Dim,
    DefaultAllocator: Allocator<N>,
{
    let mut out = F::zero();
    if rnorm.is_zero() {
        return out;
    }
    for i in 0..jt_r.nrows() {
        let denom = col_norms[i] * rnorm;
        if denom.is_positive() {
            out = Float::max(out, Float::abs(jt_r[i] / denom));
        }
    }
    out
}

fn scaled_norm<F, N>(x: &OVector<F, N>, diag: &OVector<F, N>) -> F
where
    F: RealField + Float + Copy,
    N: Dim,
    DefaultAllocator: Allocator<N>,
{
    let mut s = F::zero();
    for i in 0..x.nrows() {
        let v = x[i] * diag[i];
        s += v * v;
    }
    Float::sqrt(s)
}

fn predicted_reduction_model<F>(residuals: &[F], j_step: &[F], residual_norm: F) -> F
where
    F: RealField + Float + Copy,
{
    if residual_norm.is_zero() {
        return F::zero();
    }
    let mut model_sq = F::zero();
    let m = core::cmp::min(residuals.len(), j_step.len());
    for i in 0..m {
        let r = residuals[i] - j_step[i];
        model_sq += r * r;
    }
    let model_norm = Float::sqrt(model_sq);
    F::one() - Float::powi(model_norm / residual_norm, 2)
}

fn solve_damped_normal_equations<F, N>(
    jacobian: &SparseJacobian<F>,
    jt_r: &OVector<F, N>,
    diag: &OVector<F, N>,
    lambda: F,
    n: usize,
) -> OVector<F, N>
where
    F: RealField + Float + Copy,
    N: Dim,
    DefaultAllocator: Allocator<N>,
{
    let b = jt_r.clone_owned();
    let mut x = OVector::<F, N>::zeros_generic(Dim::from_usize(n), Dim::from_usize(1));
    let mut r = b.clone_owned();
    let mut p = r.clone_owned();
    let bnorm = Float::sqrt(Float::max(r.dot(&r), F::default_epsilon()));
    let tol = Float::max(convert(1.0e-10f64), F::default_epsilon() * convert(10.0f64));
    let mut rr_old = r.dot(&r);

    let max_iter = 2 * n + 20;
    for _ in 0..max_iter {
        if rr_old <= tol * tol * bnorm * bnorm {
            break;
        }

        let ap = sparse_normal_op_mul(jacobian, &p, diag, lambda, n);
        let denom = p.dot(&ap);
        if !denom.is_finite() || Float::abs(denom) <= F::default_epsilon() {
            break;
        }
        let alpha = rr_old / denom;

        x.axpy(alpha, &p, F::one());
        r.axpy(-alpha, &ap, F::one());

        let rr_new = r.dot(&r);
        if rr_new <= tol * tol * bnorm * bnorm {
            break;
        }
        let beta = rr_new / rr_old;
        p *= beta;
        p += &r;
        rr_old = rr_new;
    }

    x
}

fn solve_damped_normal_equations_schur<F, N>(
    jacobian: &SparseJacobian<F>,
    jt_r: &OVector<F, N>,
    diag: &OVector<F, N>,
    lambda: F,
    n: usize,
    n_camera: usize,
) -> OVector<F, N>
where
    F: RealField + Float + Copy,
    N: Dim,
    DefaultAllocator: Allocator<N>,
{
    let n_point = n - n_camera;
    if n_point == 0 {
        return solve_damped_normal_equations(jacobian, jt_r, diag, lambda, n);
    }

    let mut rows: Vec<Vec<(usize, F)>> = alloc::vec![Vec::new(); jacobian.rows];
    for &(r, c, v) in jacobian.entries.iter() {
        if r < rows.len() && c < n {
            rows[r].push((c, v));
        }
    }

    let mut b = alloc::vec![F::zero(); n_camera * n_camera];
    let mut c_diag = alloc::vec![F::zero(); n_point];
    let mut e_by_point: Vec<BTreeMap<usize, F>> = (0..n_point).map(|_| BTreeMap::new()).collect();

    for row in rows.iter() {
        let mut cams: Vec<(usize, F)> = Vec::new();
        let mut pts: Vec<(usize, F)> = Vec::new();
        for &(col, val) in row.iter() {
            if col < n_camera {
                cams.push((col, val));
            } else {
                pts.push((col - n_camera, val));
            }
        }

        for &(ci, vi) in cams.iter() {
            for &(cj, vj) in cams.iter() {
                b[ci * n_camera + cj] += vi * vj;
            }
        }

        for &(pj, vj) in pts.iter() {
            c_diag[pj] += vj * vj;
        }

        for &(ci, vi) in cams.iter() {
            for &(pj, vj) in pts.iter() {
                let entry = e_by_point[pj].entry(ci).or_insert(F::zero());
                *entry += vi * vj;
            }
        }
    }

    let mut rhs_c = alloc::vec![F::zero(); n_camera];
    let mut rhs_p = alloc::vec![F::zero(); n_point];
    for i in 0..n_camera {
        rhs_c[i] = jt_r[i];
    }
    for i in 0..n_point {
        rhs_p[i] = jt_r[n_camera + i];
    }

    for i in 0..n_camera {
        b[i * n_camera + i] += lambda * diag[i] * diag[i];
    }
    for i in 0..n_point {
        c_diag[i] += lambda * diag[n_camera + i] * diag[n_camera + i];
        if c_diag[i] <= F::default_epsilon() {
            c_diag[i] = F::default_epsilon();
        }
    }

    let mut schur = b.clone();
    let mut schur_rhs = rhs_c;
    for p in 0..n_point {
        if e_by_point[p].is_empty() {
            continue;
        }
        let inv_c = Float::recip(c_diag[p]);
        let entries: Vec<(usize, F)> = e_by_point[p].iter().map(|(k, v)| (*k, *v)).collect();
        for &(ai, eai) in entries.iter() {
            schur_rhs[ai] -= eai * rhs_p[p] * inv_c;
            for &(aj, eaj) in entries.iter() {
                schur[ai * n_camera + aj] -= eai * eaj * inv_c;
            }
        }
    }

    let dc = cg_dense_symmetric(&schur, &schur_rhs, n_camera);

    let mut out = OVector::<F, N>::zeros_generic(Dim::from_usize(n), Dim::from_usize(1));
    for i in 0..n_camera {
        out[i] = dc[i];
    }

    for p in 0..n_point {
        let mut accum = rhs_p[p];
        for (&ci, &e_val) in e_by_point[p].iter() {
            accum -= e_val * dc[ci];
        }
        out[n_camera + p] = accum / c_diag[p];
    }

    out
}

fn cg_dense_symmetric<F>(a: &[F], b: &[F], n: usize) -> Vec<F>
where
    F: RealField + Float + Copy,
{
    let mut x = alloc::vec![F::zero(); n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let bnorm_sq = Float::max(dot_slice(&r, &r), F::default_epsilon());
    let tol_sq = Float::powi(Float::max(convert(1.0e-10f64), F::default_epsilon() * convert(10.0f64)), 2);
    let mut rr_old = dot_slice(&r, &r);

    let max_iter = 2 * n + 20;
    let mut ap = alloc::vec![F::zero(); n];
    for _ in 0..max_iter {
        if rr_old <= tol_sq * bnorm_sq {
            break;
        }

        for i in 0..n {
            let mut sum = F::zero();
            for j in 0..n {
                sum += a[i * n + j] * p[j];
            }
            ap[i] = sum;
        }

        let denom = dot_slice(&p, &ap);
        if Float::abs(denom) <= F::default_epsilon() || !denom.is_finite() {
            break;
        }
        let alpha = rr_old / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rr_new = dot_slice(&r, &r);
        if rr_new <= tol_sq * bnorm_sq {
            break;
        }

        let beta = rr_new / rr_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rr_old = rr_new;
    }
    x
}

fn dot_slice<F>(a: &[F], b: &[F]) -> F
where
    F: RealField + Float + Copy,
{
    let mut s = F::zero();
    let n = core::cmp::min(a.len(), b.len());
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

fn sparse_normal_op_mul<F, N>(
    jacobian: &SparseJacobian<F>,
    x: &OVector<F, N>,
    diag: &OVector<F, N>,
    lambda: F,
    n: usize,
) -> OVector<F, N>
where
    F: RealField + Float + Copy,
    N: Dim,
    DefaultAllocator: Allocator<N>,
{
    let jx = sparse_j_mul(jacobian, x);
    let mut out = sparse_jt_mul(jacobian, jx.as_slice(), n);
    if !lambda.is_zero() {
        for i in 0..out.nrows() {
            out[i] += lambda * diag[i] * diag[i] * x[i];
        }
    }
    out
}

/// Struct which holds the state of the LM algorithm and which implements its individual steps.
struct LM<'a, F, N, M, O>
where
    F: RealField + Copy,
    N: Dim,
    M: DimMin<N> + DimMax<N>,
    O: LeastSquaresProblem<F, M, N>,
    DefaultAllocator: Allocator<N> + Allocator<M, N> + Allocator<DimMaximum<M, N>, N>,
{
    config: &'a LevenbergMarquardt<F>,
    /// Current parameters `$\vec{x}$`
    x: Vector<F, N, O::ParameterStorage>,
    tmp: Vector<F, N, O::ParameterStorage>,
    /// The implementation of `LeastSquaresProblem`
    target: O,
    /// Statistics and termination reasons, used for return value
    report: MinimizationReport<F>,
    /// The delta from the trust-region algorithm
    delta: F,
    lambda: F,
    /// `$\|\mathbf{D}\vec{x}\|`
    xnorm: F,
    gnorm: F,
    residuals_norm: F,
    /// The diagonal of `$\mathbf{D}$`
    diag: OVector<F, N>,
    /// Flag to check if it is the first trust region iteration
    first_trust_region_iteration: bool,
    /// Flag to check if it is the first diagonal update
    first_update: bool,
    max_fev: usize,
    m: usize,
}

impl<'a, F, N, M, O> LM<'a, F, N, M, O>
where
    F: RealField + Float + Copy,
    N: Dim,
    M: DimMin<N> + DimMax<N>,
    O: LeastSquaresProblem<F, M, N>,
    DefaultAllocator: Allocator<N> + Allocator<M, N> + Allocator<DimMaximum<M, N>, N>,
{
    #[allow(clippy::type_complexity)]
    fn new(
        config: &'a LevenbergMarquardt<F>,
        target: O,
    ) -> Result<(Self, Vector<F, M, O::ResidualStorage>), (O, MinimizationReport<F>)> {
        let mut report = MinimizationReport {
            termination: TerminationReason::ResidualsZero,
            number_of_evaluations: 1,
            objective_function: <F as Float>::nan(),
        };

        // Evaluate at start point
        let x = target.params();
        let (residuals, residuals_norm) = if let Some(residuals) = target.residuals() {
            let norm = enorm(&residuals);
            report.objective_function = norm * norm * convert(0.5);
            (residuals, norm)
        } else {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::User("residuals"),
                    ..report
                },
            ));
        };

        // Initialize diagonal
        let n = x.shape_generic().0;
        let diag = OVector::<F, N>::from_element_generic(n, Dim::from_usize(1), F::one());
        // Check n > 0
        if diag.nrows() == 0 {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::NoParameters,
                    ..report
                },
            ));
        }

        let m = residuals.nrows();
        if m == 0 {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::NoResiduals,
                    ..report
                },
            ));
        }

        if !residuals_norm.is_finite() && !cfg!(feature = "minpack-compat") {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::Numerical("residuals norm"),
                    ..report
                },
            ));
        }

        if residuals_norm <= Float::min_positive_value() && !cfg!(feature = "minpack-compat") {
            // Already zero, nothing to do
            return Err((target, report));
        }

        Ok((
            Self {
                config,
                target,
                report,
                tmp: x.clone(),
                x,
                diag,
                delta: F::zero(),
                lambda: F::zero(),
                xnorm: F::zero(),
                gnorm: F::zero(),
                residuals_norm,
                first_trust_region_iteration: true,
                first_update: true,
                max_fev: config.patience * (n.value() + 1),
                m,
            },
            residuals,
        ))
    }

    fn into_report(self, termination: TerminationReason) -> (O, MinimizationReport<F>) {
        (
            self.target,
            MinimizationReport {
                termination,
                ..self.report
            },
        )
    }

    fn jacobian(&self) -> Result<SparseJacobian<F>, TerminationReason> {
        match self.target.jacobian() {
            Some(jacobian) => Ok(jacobian),
            None => Err(TerminationReason::User("jacobian")),
        }
    }

    fn update_diag(
        &mut self,
        lls: &mut LinearLeastSquaresDiagonalProblem<F, M, N>,
    ) -> Result<(), TerminationReason>
    where
        DefaultAllocator: Allocator<N>,
    {
        // Compute norm of scaled gradient and detect degeneracy
        self.gnorm = match lls.max_a_t_b_scaled(self.residuals_norm) {
            Some(max_at_b) => max_at_b,
            None if !cfg!(feature = "minpack-compat") => {
                return Err(TerminationReason::Numerical("jacobian"));
            }
            None => F::zero(),
        };
        if self.gnorm <= self.config.gtol {
            return Err(TerminationReason::Orthogonal);
        }

        if self.first_update {
            // Initialize diag and xnorm
            self.xnorm = if self.config.scale_diag {
                for (d, col_norm) in self.diag.iter_mut().zip(lls.column_norms.iter()) {
                    *d = if col_norm.is_zero() {
                        F::one()
                    } else {
                        *col_norm
                    };
                }
                self.tmp.cmpy(F::one(), &self.diag, &self.x, F::zero());
                enorm(&self.tmp)
            } else {
                enorm(&self.x)
            };
            if !self.xnorm.is_finite() && !cfg!(feature = "minpack-compat") {
                return Err(TerminationReason::Numerical("subproblem x"));
            }
            // Initialize delta
            self.delta = if self.xnorm.is_zero() {
                self.config.stepbound
            } else {
                self.config.stepbound * self.xnorm
            };
            self.first_update = false;
        } else if self.config.scale_diag {
            // Update diag
            for (d, norm) in self.diag.iter_mut().zip(lls.column_norms.iter()) {
                *d = Float::max(*norm, *d);
            }
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn trust_region_iteration(
        &mut self,
        lls: &mut LinearLeastSquaresDiagonalProblem<F, M, N>,
        param: LMParameter<F, N>,
    ) -> Result<Option<Vector<F, M, O::ResidualStorage>>, TerminationReason>
    where
        DefaultAllocator: Allocator<N>,
    {
        const P1: f64 = 0.1;
        const P0001: f64 = 1.0e-4;

        self.lambda = param.lambda;
        let pnorm = param.dp_norm;
        if !pnorm.is_finite() && !cfg!(feature = "minpack-compat") {
            return Err(TerminationReason::Numerical("subproblem ||Dp||"));
        }

        let predicted_reduction;
        let dir_der;
        {
            let temp1 = Float::powi(lls.a_x_norm(&param.step) / self.residuals_norm, 2);
            if !temp1.is_finite() && !cfg!(feature = "minpack-compat") {
                return Err(TerminationReason::Numerical("trust-region reduction"));
            }
            let temp2 = Float::powi((Float::sqrt(self.lambda) * pnorm) / self.residuals_norm, 2);
            if !temp2.is_finite() && !cfg!(feature = "minpack-compat") {
                return Err(TerminationReason::Numerical("trust-region reduction"));
            }
            predicted_reduction = temp1 + temp2 / convert(0.5);
            dir_der = -(temp1 + temp2);
        }

        if self.first_trust_region_iteration && pnorm < self.delta {
            self.delta = pnorm;
        }
        self.first_trust_region_iteration = false;

        // Compute new parameters: x - p
        self.tmp.copy_from(&self.x);
        self.tmp.axpy(-F::one(), &param.step, F::one());

        // Evaluate
        self.target.set_params(&self.tmp);
        self.report.number_of_evaluations += 1;
        let new_objective_function;
        let (residuals, new_residuals_norm) = if let Some(residuals) = self.target.residuals() {
            if residuals.nrows() != self.m {
                return Err(TerminationReason::WrongDimensions("residuals"));
            }
            let norm = enorm(&residuals);
            new_objective_function = norm * norm * convert(0.5);
            (residuals, norm)
        } else {
            return Err(TerminationReason::User("residuals"));
        };

        // Compute predicted and actual reduction
        let actual_reduction = if new_residuals_norm * convert(P1) < self.residuals_norm {
            F::one() - Float::powi(new_residuals_norm / self.residuals_norm, 2)
        } else {
            -F::one()
        };

        let ratio = if predicted_reduction.is_zero() {
            F::zero()
        } else {
            actual_reduction / predicted_reduction
        };
        let half: F = convert(0.5);
        if ratio <= convert(0.25) {
            let mut temp = if !actual_reduction.is_negative() {
                half
            } else {
                half * dir_der / (dir_der + half * actual_reduction)
            };
            if new_residuals_norm * convert(P1) >= self.residuals_norm || temp < convert(P1) {
                temp = convert(P1);
            };
            self.delta = temp * Float::min(self.delta, pnorm * convert(10.));
            self.lambda /= temp;
        } else if self.lambda.is_zero() || ratio >= convert(0.75) {
            self.delta = pnorm / convert(0.5);
            self.lambda *= half;
        }

        let update_considered_good = ratio >= convert(P0001);
        if update_considered_good {
            // update x, residuals and their norms
            core::mem::swap(&mut self.x, &mut self.tmp);
            self.xnorm = if self.config.scale_diag {
                self.tmp.cmpy(F::one(), &self.diag, &self.x, F::zero());
                enorm(&self.tmp)
            } else {
                enorm(&self.x)
            };
            if !self.xnorm.is_finite() && !cfg!(feature = "minpack-compat") {
                return Err(TerminationReason::Numerical("new x"));
            }
            self.residuals_norm = new_residuals_norm;
            self.report.objective_function = new_objective_function;
        }

        // convergence tests
        if !cfg!(feature = "minpack-compat") && self.residuals_norm <= F::min_positive_value() {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::ResidualsZero);
        }
        let ftol_check = Float::abs(actual_reduction) <= self.config.ftol
            && predicted_reduction <= self.config.ftol
            && ratio * convert(0.5) <= F::one();
        let xtol_check = self.delta <= self.config.xtol * self.xnorm;
        if ftol_check || xtol_check {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::Converged {
                ftol: ftol_check,
                xtol: xtol_check,
            });
        }

        // termination tests
        if self.report.number_of_evaluations >= self.max_fev {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::LostPatience);
        }

        // We now check if one of the ftol, xtol or gtol criteria
        // is fulfilld with the machine epsilon.
        if Float::abs(actual_reduction) <= epsmch()
            && predicted_reduction <= epsmch()
            && ratio * convert(0.5) <= F::one()
        {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::NoImprovementPossible("ftol"));
        }
        if self.delta <= epsmch::<F>() * self.xnorm {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::NoImprovementPossible("xtol"));
        }
        if self.gnorm <= epsmch() {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::NoImprovementPossible("gtol"));
        }

        if update_considered_good {
            Ok(Some(residuals))
        } else {
            // Need another iteration, did not change the parameters
            Ok(None)
        }
    }

    #[inline]
    fn reset_params_if(&mut self, reset: bool) {
        if reset {
            self.target.set_params(&self.x);
        }
    }
}
