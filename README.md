# levenberg-marquardt-sparse

[![Crates.io][ci]][cl] ![MIT][li] [![docs.rs][di]][dl]

[ci]: https://img.shields.io/crates/v/levenberg-marquardt-sparse.svg
[cl]: https://crates.io/crates/levenberg-marquardt-sparse/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/levenberg-marquardt-sparse/badge.svg
[dl]: https://docs.rs/levenberg-marquardt-sparse/

Sparse Levenberg-Marquardt solver for nonlinear least squares problems.

This crate is a fork of [`levenberg-marquardt`](https://crates.io/crates/levenberg-marquardt) that
replaces the dense QR-based trust-region solver with a sparse normal-equations solver based on
Jacobi-preconditioned Conjugate Gradient (PCG). It is designed for problems where the Jacobian
`$\mathbf{J} \in \mathbb{R}^{m \times n}$` is sparse (e.g. bundle adjustment, SLAM), allowing
`$m$` and `$n$` to be large without materializing a dense matrix.

The migration from dense to sparse was performed via agentic coding using GitHub Copilot.

## Key differences from `levenberg-marquardt`

- **Always sparse:** the solver always uses the sparse PCG path; there is no dense fallback.
- **Sparse Jacobian API:** `jacobian()` returns `Option<SparseJacobian<F>>` (COO format).
  Dense problems can use `SparseJacobian::from_dense(matrix)` as a bridge.
- **No `minpack-compat` feature:** MINPACK-compatibility mode has been removed.
- **No QR / trust-region code:** `src/qr.rs` and `src/trust_region.rs` have been removed.
- **Lambda management:** uses 10× lambda increase on step rejection with a stagnation guard
  (8 consecutive failed outer iterations → treat as converged).

## Usage

```rust
use levenberg_marquardt_sparse::{LeastSquaresProblem, LevenbergMarquardt, SparseJacobian};

impl LeastSquaresProblem<f64, M, N> for MyProblem {
    type ParameterStorage = ...;
    type ResidualStorage = ...;

    fn set_params(&mut self, p: &OVector<f64, N>) { ... }
    fn params(&self) -> OVector<f64, N> { ... }
    fn residuals(&self) -> Option<OVector<f64, M>> { ... }

    fn jacobian(&self) -> Option<SparseJacobian<f64>> {
        // Build a sparse Jacobian in COO (triplet) format:
        //   entries is a Vec of (row, col, value)
        Some(SparseJacobian::from_triplets(rows, cols, entries))
        // Or, for small/dense problems, use the bridge method:
        // Some(SparseJacobian::from_dense(matrix))
    }
}

let problem = MyProblem::new(initial_params);
let (problem, report) = LevenbergMarquardt::new().minimize(problem);
assert!(report.termination.was_successful());
```

## Hyperparameters

`LevenbergMarquardt` exposes builder methods for the standard LM tolerances:

| Method | Default | Description |
|---|---|---|
| `.with_ftol(f)` | `30 * ε` | Relative reduction in objective required to converge |
| `.with_xtol(f)` | `30 * ε` | Relative change in parameters required to converge |
| `.with_gtol(f)` | `30 * ε` | Orthogonality of residual and Jacobian columns |
| `.with_stepbound(f)` | `100.0` | Initial trust-region step-bound factor |
| `.with_patience(n)` | `100` | Max evaluations = `patience * (n_params + 1)` |
| `.with_scale_diag(b)` | `true` | Whether to rescale parameters by column norms |

where `ε` is the floating-point machine epsilon for `F`.

## `no_std` support

The crate is `#![no_std]` (requires `alloc`).

## References

- The [MINPACK](https://www.netlib.org/minpack/) Fortran implementation (original LM algorithm).
- Moré, J.J. (1978) "The Levenberg-Marquardt algorithm: Implementation and theory."
  In: Watson G.A. (eds) *Numerical Analysis*, Lecture Notes in Mathematics vol 630. Springer.
- Nocedal & Wright, *Numerical Optimization* (2nd ed.), chapters 4 and 10.
