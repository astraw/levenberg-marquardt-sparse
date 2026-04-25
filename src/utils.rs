#![allow(unexpected_cfgs)]
use crate::LeastSquaresProblem;
use alloc::{format, string::String};
use core::cell::RefCell;
use nalgebra::{
    Complex, ComplexField, DefaultAllocator, Dim, OMatrix, RealField, U1, Vector,
    allocator::Allocator, convert, storage::RawStorage, storage::Storage,
};
use num_traits::float::Float;

// mod derivest;
mod finite_difference;

#[cfg(feature = "RUSTC_IS_NIGHTLY")]
pub use core::intrinsics::{likely, unlikely};

#[cfg(not(feature = "RUSTC_IS_NIGHTLY"))]
#[inline]
pub fn likely(b: bool) -> bool {
    b
}

#[cfg(not(feature = "RUSTC_IS_NIGHTLY"))]
#[inline]
pub fn unlikely(b: bool) -> bool {
    b
}

/// Compute a numerical approximation of the Jacobian.
///
/// The residuals function is called approximately `$30\cdot nm$` times which
/// can make this slow in debug builds and for larger problems.
///
/// The function is intended to be used for debugging or testing.
/// You can try to check your derivative implementation of an
/// [`LeastSquaresProblem`](trait.LeastSquaresProblem.html) with this.
///
/// Computing the derivatives numerically is numerically unstable: You can construct
/// functions where the computed result is far off. If you
/// observe large differences between the derivative computed by this function
/// and your implementation the reason _might_ be due to instability.
///
/// The achieved precision by this function
/// is lower than the floating point precision in general. So the error is bigger
/// than `$10^{-15}$` for `f64` and bigger than `$10^{-7}$` for `f32`. See the example
/// below for what that means in your tests. If possible use `f64` for the testing.
///
/// A much more precise alternative is provided by
/// [`differentiate_holomorphic_numerically`](fn.differentiate_holomorphic_numerically.html)
/// but it requires your residuals to be holomorphic and `LeastSquaresProblem` to be implemented
/// for complex numbers.
///
/// # Example
///
/// You can use this function to check your derivative implementation in a unit test.
/// For example:
///
/// ```rust
/// # use levenberg_marquardt_sparse::{LeastSquaresProblem, SparseJacobian, differentiate_numerically};
/// # use approx::assert_relative_eq;
/// # use nalgebra::{convert, ComplexField, storage::Owned, Matrix2, Vector2, OVector, U2};
/// #
/// # struct ExampleProblem<F: ComplexField> {
/// #     p: Vector2<F>,
/// # }
/// #
/// # impl<F: ComplexField + Copy> LeastSquaresProblem<F, U2, U2> for ExampleProblem<F> {
/// #     type ParameterStorage = Owned<F, U2>;
/// #     type ResidualStorage = Owned<F, U2>;
/// #
/// #     fn set_params(&mut self, p: &OVector<F, U2>) {
/// #         self.p.copy_from(p);
/// #     }
/// #
/// #     fn params(&self) -> OVector<F, U2> { self.p }
/// #
/// #     fn residuals(&self) -> Option<Vector2<F>> {
/// #         Some(Vector2::new(
/// #             self.p.x * self.p.x + self.p.y - convert(11.0),
/// #             self.p.x + self.p.y * self.p.y - convert(7.0),
/// #         ))
/// #     }
/// #
/// #     fn jacobian(&self) -> Option<SparseJacobian<F>> {
/// #         let two: F = convert(2.);
/// #         Some(SparseJacobian::from_dense(Matrix2::new(
/// #             two * self.p.x,
/// #             F::one(),
/// #             F::one(),
/// #             two * self.p.y,
/// #         )))
/// #     }
/// # }
/// // Let `problem` be an instance of `LeastSquaresProblem`
/// # let mut problem = ExampleProblem::<f64> { p: Vector2::new(6., -10.), };
/// let jacobian_numerical = differentiate_numerically(&mut problem).unwrap();
/// let jacobian_trait = problem.jacobian().unwrap().to_dense::<_, _>();
/// assert_relative_eq!(jacobian_numerical, jacobian_trait, epsilon = 1e-13);
/// ```
///
/// The `assert_relative_eq!` macro is from the `approx` crate.
pub fn differentiate_numerically<F, N, M, O>(problem: &mut O) -> Option<OMatrix<F, M, N>>
where
    F: RealField + Float + Copy,
    N: Dim,
    M: Dim,
    O: LeastSquaresProblem<F, M, N>,
    DefaultAllocator: Allocator<M, N>,
{
    let params = problem.params();
    let n = params.data.shape().0;
    let m = problem.residuals()?.data.shape().0;
    let params = RefCell::new(params);
    let problem = RefCell::new(problem);
    let mut jacobian = OMatrix::<F, M, N>::zeros_generic(m, n);
    for j in 0..n.value() {
        let x = params.borrow()[j];
        for i in 0..m.value() {
            let f = |x| {
                params.borrow_mut()[j] = x;
                let mut problem = problem.borrow_mut();
                problem.set_params(&params.borrow());
                problem.residuals().map(|v| v[i])
            };
            jacobian[(i, j)] = finite_difference::derivative(x, f)?;
        }
        params.borrow_mut()[j] = x;
    }
    // reset the initial params
    problem.borrow_mut().set_params(&params.borrow());
    Some(jacobian)
}

/// Compute a numerical approximation of the Jacobian for _holomorphic_ residuals.
///
/// This method is _much_ more precise than
/// [`differentiate_numerically`](fn.differentiate_numerically.html) but
/// it requires that your residuals are holomorphic on a neighborhood of the real line.
/// You also must provide an implementation of
/// [`LeastSquaresProblem`](trait.LeastSquaresProblem.html) for complex numbers.
///
/// This method is mainly intended for testing your derivative implementation.
///
/// # Panics
///
/// The function panics if the parameters which are set when the function is
/// called are not real.
///
/// # Example
///
/// ```rust
/// # use levenberg_marquardt_sparse::{LeastSquaresProblem, SparseJacobian, differentiate_holomorphic_numerically};
/// # use approx::assert_relative_eq;
/// # use nalgebra::{storage::Owned, Complex, Matrix2, Vector2, OVector, U2};
/// use nalgebra::{ComplexField, convert};
///
/// struct ExampleProblem<F: ComplexField> {
///     params: Vector2<F>,
/// }
///
/// // Implement LeastSquaresProblem to be usable with complex numbers
/// impl<F: ComplexField + Copy> LeastSquaresProblem<F, U2, U2> for ExampleProblem<F> {
///     // ... omitted ...
/// #     type ParameterStorage = Owned<F, U2>;
/// #     type ResidualStorage = Owned<F, U2>;
/// #
/// #     fn set_params(&mut self, params: &OVector<F, U2>) {
/// #         self.params.copy_from(params);
/// #     }
/// #
/// #     fn params(&self) -> OVector<F, U2> { self.params }
/// #
/// #     fn residuals(&self) -> Option<Vector2<F>> {
/// #         Some(Vector2::new(
/// #             self.params.x * self.params.x + self.params.y - convert(11.0),
/// #             self.params.x + self.params.y * self.params.y - convert(7.0),
/// #         ))
/// #     }
/// #
/// #     fn jacobian(&self) -> Option<SparseJacobian<F>> {
/// #         let two: F = convert(2.);
/// #         Some(SparseJacobian::from_dense(Matrix2::new(
/// #             two * self.params.x,
/// #             F::one(),
/// #             F::one(),
/// #             two * self.params.y,
/// #         )))
/// #     }
/// }
///
/// // parameters for which you want to test your derivative
/// let x = Vector2::new(0.03877264483558185, -0.7734472300384164);
///
/// // instantiate f64 variant to compute the derivative we want to check
/// let jacobian_from_trait = (ExampleProblem::<f64> { params: x })
///     .jacobian()
///     .unwrap()
///     .to_dense::<_, _>();
///
/// // then use Complex<f64> and compute the numerical derivative
/// let jacobian_numerically = {
///     let mut problem = ExampleProblem::<Complex<f64>> {
///         params: convert(x),
///     };
///     differentiate_holomorphic_numerically(&mut problem).unwrap()
/// };
///
/// assert_relative_eq!(jacobian_from_trait, jacobian_numerically, epsilon = 1e-15);
/// ```
pub fn differentiate_holomorphic_numerically<F, N, M, O>(
    problem: &mut O,
) -> Option<OMatrix<F, M, N>>
where
    F: RealField + Copy,
    N: Dim,
    M: Dim,
    O: LeastSquaresProblem<Complex<F>, M, N>,
    DefaultAllocator:
        Allocator<N, Buffer<Complex<F>> = O::ParameterStorage> + Allocator<N> + Allocator<M, N>,
{
    let mut params = problem.params();
    assert!(params.iter().all(|x| x.im.is_zero()), "params must be real");
    let n = params.data.shape().0;
    let m = problem.residuals()?.data.shape().0;
    let mut jacobian = OMatrix::<F, M, N>::zeros_generic(m, n);
    for i in 0..n.value() {
        let xi = params[i];
        let h = Complex::<F>::from_real(F::default_epsilon()) * xi.abs();
        params[i] = xi + Complex::<F>::i() * h;
        problem.set_params(&params);
        let mut residuals = problem.residuals()?;
        residuals /= h;
        for (dst, src) in jacobian.column_mut(i).iter_mut().zip(residuals.iter()) {
            *dst = src.imaginary();
        }
        params[i] = xi;
    }
    problem.set_params(&params);
    Some(jacobian)
}

#[inline]
pub(crate) fn giant<F: Float>() -> F {
    F::max_value()
}

#[inline]
pub(crate) fn dwarf<F: Float>() -> F {
    F::min_positive_value()
}

#[inline]
pub(crate) fn enorm<F, N, VS>(v: &Vector<F, N, VS>) -> F
where
    F: nalgebra::RealField + Float + Copy,
    N: Dim,
    VS: Storage<F, N, U1>,
{
    let mut s1 = F::zero();
    let mut s2 = F::zero();
    let mut s3 = F::zero();
    let mut x1max = F::zero();
    let mut x3max = F::zero();
    let agiant = Float::sqrt(giant::<F>()) / convert(v.nrows() as f64);
    let rdwarf = Float::sqrt(dwarf());
    for xi in v.iter() {
        let xabs = xi.abs();
        if unlikely(xabs.is_nan()) {
            return xabs;
        }
        if unlikely(xabs >= agiant || xabs <= rdwarf) {
            if xabs > rdwarf {
                // sum for large components
                if xabs > x1max {
                    s1 = F::one() + s1 * Float::powi(x1max / xabs, 2);
                    x1max = xabs;
                } else {
                    s1 += Float::powi(xabs / x1max, 2);
                }
            } else {
                // sum for small components
                if xabs > x3max {
                    s3 = F::one() + s3 * Float::powi(x3max / xabs, 2);
                    x3max = xabs;
                } else if xabs != F::zero() {
                    s3 += Float::powi(xabs / x3max, 2);
                }
            }
        } else {
            s2 += xabs * xabs;
        }
    }

    if unlikely(!s1.is_zero()) {
        x1max * Float::sqrt(s1 + (s2 / x1max) / x1max)
    } else if likely(!s2.is_zero()) {
        Float::sqrt(if likely(s2 >= x3max) {
            s2 * (F::one() + (x3max / s2) * (x3max * s3))
        } else {
            x3max * ((s2 / x3max) + (x3max * s3))
        })
    } else {
        x3max * Float::sqrt(s3)
    }
}

#[allow(dead_code)]
/// Debug helper to inspect the binary representation of  a `f64` or `f32`.
pub(crate) fn float_repr<F: Float>(f: F) -> alloc::string::String {
    assert!(F::one() / (F::one() + F::one()) != F::zero());
    let bytes = core::mem::size_of::<F>();
    let mut out;
    if bytes == 8 {
        out = String::with_capacity((8 * 2 + 8 - 1) + 27 + 3);
        let f = unsafe { *(&f as *const F as *const f64) };
        let as_int: u64 = f.to_bits();
        for i in (0..bytes).rev() {
            out += &format!(
                "{:02x}{}",
                as_int >> (8 * i) & 0xFF,
                if i == 0 { "" } else { ":" }
            );
        }
        out += &format!(" ({f:+.20E})");
    } else if bytes == 4 {
        out = String::with_capacity((4 * 2 + 4 - 1) + 17 + 3);
        let f = unsafe { *(&f as *const F as *const f32) };
        let as_int: u32 = f.to_bits();
        for i in (0..bytes).rev() {
            out += &format!(
                "{:02x}{}",
                as_int >> (8 * i) & 0xFF,
                if i == 0 { "" } else { ":" }
            );
        }
        out += &format!(" ({f:.10E})");
    } else {
        unimplemented!()
    }
    out
}

#[test]
fn test_linear_case() {
    use crate::lm::test_examples::LinearFullRank;
    use approx::assert_relative_eq;
    use nalgebra::{OVector, U5};
    let mut x = OVector::<f64, U5>::from_element(1.);
    x[2] = -10.;
    let mut problem = LinearFullRank { params: x, m: 6 };
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap().to_dense::<nalgebra::Dyn, U5>();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-12);
}

#[test]
fn test_reset_parameters() {
    use approx::assert_relative_eq;
    use nalgebra::{Matrix2, OVector, U2, Vector2, storage::Owned};
    #[derive(Clone)]
    struct AllButOne {
        params: OVector<f64, U2>,
    }
    impl LeastSquaresProblem<f64, U2, U2> for AllButOne {
        type ParameterStorage = Owned<f64, U2>;
        type ResidualStorage = Owned<f64, U2>;

        fn set_params(&mut self, params: &OVector<f64, U2>) {
            self.params.copy_from(params);
        }

        fn params(&self) -> OVector<f64, U2> {
            self.params
        }

        fn residuals(&self) -> Option<OVector<f64, U2>> {
            Some(Vector2::new(0.0, -100. * self.params[1].powi(2)))
        }

        #[rustfmt::skip]
        fn jacobian(&self) -> Option<crate::SparseJacobian<f64>> {
            Some(crate::SparseJacobian::from_dense(Matrix2::new(
                0.,0.,
                0.,-200. * self.params[1],
            )))
        }
    }
    let mut problem = AllButOne {
        params: Vector2::<f64>::new(0., 1. / 3.),
    };
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    let jac_trait = problem.jacobian().unwrap().to_dense::<U2, U2>();
    assert_relative_eq!(jac_num, jac_trait, epsilon = 1e-12);
}
