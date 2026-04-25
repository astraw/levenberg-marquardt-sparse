use crate::utils::enorm;
use crate::{LeastSquaresProblem, SparseJacobian};
use nalgebra::{
    DefaultAllocator, Dim, OVector, RealField, Vector,
    allocator::Allocator,
    convert,
};
use num_traits::Float;
#[cfg(feature = "tracing")]
use tracing::debug;

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
}

impl<F: RealField + Float> Default for LevenbergMarquardt<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: RealField + Float> LevenbergMarquardt<F> {
    pub fn new() -> Self {
        let user_tol = F::default_epsilon() * convert(30.0);
        Self {
            ftol: user_tol,
            xtol: user_tol,
            gtol: user_tol,
            stepbound: convert(100.0),
            patience: 100,
            scale_diag: true,
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

    /// Try to solve the given least squares problem.
    ///
    /// The parameters of the problem which are set when this function is called
    /// are used as the initial guess for `$\vec{x}$`.
    pub fn minimize<N, M, O>(&self, target: O) -> (O, MinimizationReport<F>)
    where
        N: Dim,
        M: Dim,
        O: LeastSquaresProblem<F, M, N>,
        DefaultAllocator: Allocator<N>,
    {
        let (mut lm, mut residuals) = match LM::new(self, target) {
            Err(report) => return report,
            Ok(res) => res,
        };
        let n = lm.x.nrows();
        let mut lambda = Float::max(F::default_epsilon(), convert(1.0e-6f64));
        let mut exhausted_outer_loops = 0usize;
        let exhausted_outer_limit = 8usize;

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

            #[cfg(feature = "tracing")]
            debug!(
                evals = lm.report.number_of_evaluations,
                obj = format_args!("{:?}", lm.report.objective_function),
                gnorm = format_args!("{:?}", lm.gnorm),
                lambda = format_args!("{:?}", lambda),
                "sparse outer iteration start"
            );

            let mut accepted = false;
            let mut accepted_residuals = None;
            let max_inner = 12usize;
            #[cfg(feature = "tracing")]
            let mut inner_tries = 0usize;

            for _ in 0..max_inner {
                #[cfg(feature = "tracing")]
                {
                    inner_tries += 1;
                }
                let step = solve_damped_normal_equations::<F, N>(
                    &jacobian, &jt_r, &lm.diag, &col_norms, lambda, n,
                );

                let pnorm = scaled_norm(&step, &lm.diag);
                if !pnorm.is_finite() {
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
                // Stable predicted reduction: (2 r·Jp - ||Jp||²) / ||r||²
                // Avoids catastrophic cancellation in 1 - ||r - Jp||² / ||r||²
                let predicted_reduction = {
                    let rn_sq = lm.residuals_norm * lm.residuals_norm;
                    if rn_sq.is_zero() {
                        F::zero()
                    } else {
                        let r_dot_jp = residuals
                            .as_slice()
                            .iter()
                            .zip(j_step.iter())
                            .fold(F::zero(), |acc, (&a, &b)| acc + a * b);
                        let jp_sq = j_step.iter().fold(F::zero(), |acc, &v| acc + v * v);
                        (convert::<f64, F>(2.0) * r_dot_jp - jp_sq) / rn_sq
                    }
                };
                let ratio = if predicted_reduction <= F::zero() {
                    F::zero()
                } else {
                    actual_reduction / predicted_reduction
                };

                if ratio > convert(0.0001f64) {
                    accepted = true;
                    accepted_residuals = Some((
                        new_residuals,
                        new_residuals_norm,
                        new_objective_function,
                        step,
                        pnorm,
                    ));
                    #[cfg(feature = "tracing")]
                    debug!(
                        evals = lm.report.number_of_evaluations,
                        inner_tries,
                        obj = format_args!("{:?}", lm.report.objective_function),
                        "sparse step accepted"
                    );
                    lambda = Float::max(lambda * convert(0.333333333333f64), F::default_epsilon());
                    break;
                }

                #[cfg(feature = "tracing")]
                debug!(
                    evals = lm.report.number_of_evaluations,
                    inner_try = inner_tries,
                    ratio = format_args!("{:?}", ratio),
                    "sparse step rejected, increasing lambda"
                );
                lambda *= convert(10.0f64);
                lm.target.set_params(&lm.x);
            }

            if !accepted {
                // All inner retries failed; lambda is now very large.
                // Continue to the next outer iteration — the recomputed Jacobian
                // at the same point with the elevated lambda will produce a tiny,
                // conservative step that is almost certainly acceptable.
                exhausted_outer_loops += 1;
                #[cfg(feature = "tracing")]
                debug!(
                    evals = lm.report.number_of_evaluations,
                    exhausted_outer_loops,
                    max_inner,
                    "all inner tries exhausted, retrying outer iteration with larger lambda"
                );
                if lm.report.number_of_evaluations >= lm.max_fev {
                    return lm.into_report(TerminationReason::LostPatience);
                }
                if exhausted_outer_loops >= exhausted_outer_limit {
                    // We repeatedly failed to find an acceptable step while staying
                    // at the same parameters/objective value. Treat this as practical
                    // convergence (stagnation) instead of burning evaluations.
                    return lm.into_report(TerminationReason::Converged {
                        ftol: true,
                        xtol: true,
                    });
                }
                continue;
            }

            exhausted_outer_loops = 0;

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

            if lm.residuals_norm <= F::min_positive_value() {
                return lm.into_report(TerminationReason::ResidualsZero);
            }

            let xtol_check = lm.delta <= lm.config.xtol * lm.xnorm
                || pnorm <= lm.config.xtol * (lm.xnorm + lm.config.xtol);
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

fn solve_damped_normal_equations<F, N>(
    jacobian: &SparseJacobian<F>,
    jt_r: &OVector<F, N>,
    diag: &OVector<F, N>,
    col_norms: &OVector<F, N>,
    lambda: F,
    n: usize,
) -> OVector<F, N>
where
    F: RealField + Float + Copy,
    N: Dim,
    DefaultAllocator: Allocator<N>,
{
    // Jacobi preconditioner: M_inv[i] = 1 / (col_norms[i]^2 + lambda * diag[i]^2)
    // col_norms[i]^2 = diag(J^T J)[i], so M approximates the diagonal of the system matrix.
    let mut m_inv = OVector::<F, N>::zeros_generic(Dim::from_usize(n), Dim::from_usize(1));
    for i in 0..n {
        let d = col_norms[i] * col_norms[i] + lambda * diag[i] * diag[i];
        m_inv[i] = if d > F::default_epsilon() {
            F::one() / d
        } else {
            F::one()
        };
    }

    let mut x = OVector::<F, N>::zeros_generic(Dim::from_usize(n), Dim::from_usize(1));
    let mut r = jt_r.clone_owned();

    // z = M^{-1} r
    let mut z = OVector::<F, N>::zeros_generic(Dim::from_usize(n), Dim::from_usize(1));
    for i in 0..n {
        z[i] = r[i] * m_inv[i];
    }

    let mut p = z.clone_owned();
    let bz0 = Float::max(jt_r.dot(&z), F::default_epsilon()); // ||b||^2 in M^{-1} norm
    let tol_sq = Float::powi(
        Float::max(convert(1.0e-10f64), F::default_epsilon() * convert(10.0f64)),
        2,
    );
    let mut rz_old = r.dot(&z); // r^T M^{-1} r

    let max_iter = 2 * n + 20;
    for _ in 0..max_iter {
        if rz_old <= tol_sq * bz0 {
            break;
        }

        let ap = sparse_normal_op_mul(jacobian, &p, diag, lambda, n);
        let denom = p.dot(&ap);
        if !denom.is_finite() || Float::abs(denom) <= F::default_epsilon() {
            break;
        }
        let alpha = rz_old / denom;

        x.axpy(alpha, &p, F::one());
        r.axpy(-alpha, &ap, F::one());
        for i in 0..n {
            z[i] = r[i] * m_inv[i];
        }

        let rz_new = r.dot(&z);
        if rz_new <= tol_sq * bz0 {
            break;
        }
        let beta = rz_new / rz_old;
        p *= beta;
        p += &z;
        rz_old = rz_new;
    }

    x
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
    M: Dim,
    O: LeastSquaresProblem<F, M, N>,
    DefaultAllocator: Allocator<N>,
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
    /// `$\|\mathbf{D}\vec{x}\|`
    xnorm: F,
    gnorm: F,
    residuals_norm: F,
    /// The diagonal of `$\mathbf{D}$`
    diag: OVector<F, N>,
    /// Flag to check if it is the first diagonal update
    first_update: bool,
    max_fev: usize,
    m: usize,
}

impl<'a, F, N, M, O> LM<'a, F, N, M, O>
where
    F: RealField + Float + Copy,
    N: Dim,
    M: Dim,
    O: LeastSquaresProblem<F, M, N>,
    DefaultAllocator: Allocator<N>,
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

        if !residuals_norm.is_finite() {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::Numerical("residuals norm"),
                    ..report
                },
            ));
        }

        if residuals_norm <= Float::min_positive_value() {
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
                xnorm: F::zero(),
                gnorm: F::zero(),
                residuals_norm,
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
}

