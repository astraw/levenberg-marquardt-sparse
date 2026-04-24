use alloc::vec::Vec;
use nalgebra::{
    ComplexField, DefaultAllocator, Dim, Matrix, OMatrix, Vector,
    allocator::Allocator,
    storage::{IsContiguous, RawStorage, RawStorageMut, Storage},
};

/// Sparse Jacobian in triplet (COO) format.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseJacobian<F> {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, F)>,
}

impl<F: ComplexField + Copy> SparseJacobian<F> {
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: Vec::new(),
        }
    }

    #[must_use]
    pub fn from_triplets(rows: usize, cols: usize, entries: Vec<(usize, usize, F)>) -> Self {
        Self { rows, cols, entries }
    }

    #[must_use]
    pub fn from_dense<M, N, S>(matrix: Matrix<F, M, N, S>) -> Self
    where
        M: Dim,
        N: Dim,
        S: RawStorage<F, M, N>,
    {
        let mut entries = Vec::new();
        let rows = matrix.nrows();
        let cols = matrix.ncols();
        for i in 0..rows {
            for j in 0..cols {
                let value = matrix[(i, j)];
                if !value.is_zero() {
                    entries.push((i, j, value));
                }
            }
        }
        Self { rows, cols, entries }
    }

    #[must_use]
    pub fn to_dense<M, N>(&self) -> OMatrix<F, M, N>
    where
        M: Dim,
        N: Dim,
        DefaultAllocator: Allocator<M, N>,
    {
        let mut out = OMatrix::<F, M, N>::zeros_generic(
            Dim::from_usize(self.rows),
            Dim::from_usize(self.cols),
        );
        for &(i, j, value) in &self.entries {
            out[(i, j)] += value;
        }
        out
    }
}

/// A least squares minimization problem.
///
/// This is what [`LevenbergMarquardt`](struct.LevenbergMarquardt.html) needs
/// to compute the residuals and the Jacobian. See the [module documentation](index.html)
/// for a usage example.
pub trait LeastSquaresProblem<F, M, N>
where
    F: ComplexField + Copy,
    N: Dim,
    M: Dim,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage: RawStorageMut<F, M> + Storage<F, M> + IsContiguous;
    type ParameterStorage: RawStorageMut<F, N> + Storage<F, N> + IsContiguous + Clone;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: &Vector<F, N, Self::ParameterStorage>);

    /// Get the current parameter vector `$\vec{x}$`.
    fn params(&self) -> Vector<F, N, Self::ParameterStorage>;

    /// Compute the residual vector.
    fn residuals(&self) -> Option<Vector<F, M, Self::ResidualStorage>>;

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<SparseJacobian<F>>;
}
