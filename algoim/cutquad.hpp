#ifndef ALGOIM_CUTQUAD_HPP
#define ALGOIM_CUTQUAD_HPP

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/array.hpp"
#include "jlcxx/functions.hpp"

/// Compute quadrature weights and points for a hypercube cut by a level set.
/// \tparam N - dimension of the hypercube 
/// \tparam T - floating point type used for the quadrature rule
/// \param[in] phi - function defining the level set
/// \param[in] q - number of Gaussian quadrature points in one-dimension
/// \param[in] bernstein_degree - the polynomial degree of the Bernstein fit
/// \param[in] x_min - minimum value of hypercube coordinates
/// \param[in] x_max - maximum value of hypercube coordinates
/// \param[in] x_work - work array that is passed to `phi`
/// \param[inout] wts - quadrature weights for the volume domain 
/// \param[inout] pts - quadrature points for the volume domain 
/// \param[inout] surf_wts - quadrature weights for the surface `phi(x) = 0`
/// \param[inout] surf_pts - quadrature points for the surface `phi(x) = 0`
/// \note The `wts`, `pts`, `surf_wts` and `surf_pts` arrays should be empty on 
/// entry.  On exit, they will have the weights and points.
template<int N, typename T = double>
void calc_cut_quad(T (*phi)(jlcxx::ArrayRef<T>), int q, int bernstein_degree,
                   const jlcxx::ArrayRef<T> x_min,
                   const jlcxx::ArrayRef<T> x_max, jlcxx::ArrayRef<T> x_work,
                   jlcxx::ArrayRef<T> wts, jlcxx::ArrayRef<T> pts,
                   jlcxx::ArrayRef<T> surf_wts, jlcxx::ArrayRef<T> surf_pts);

/// Compute derivative of quadrature weights and points.
/// \tparam N - dimension of the hypercube 
/// \tparam T - floating point type used for the quadrature rule derivatives
/// \param[in] phi - function defining the level set
/// \param[in] dphi - derivative of `phi` with respect to some user parameter
/// \param[in] q - number of Gaussian quadrature points in one-dimension
/// \param[in] bernstein_degree - the polynomial degree of the Bernstein fit
/// \param[in] x_min - minimum value of hypercube coordinates
/// \param[in] x_max - maximum value of hypercube coordinates
/// \param[in] x_work - work array that is passed to `phi`
/// \param[inout] wts_dot - derivative of volume quadrature weights
/// \param[inout] pts_dot - derivative of volume quadrature points 
/// \param[inout] surf_wts_dot - derivative of surface quadrature weights
/// \param[inout] surf_pts_dot - derivative of surface quadrature points
/// \note The `wts`, `pts`, `surf_wts` and `surf_pts` arrays should be empty on 
/// entry.  On exit, they will have the weights and points.
template<int N, typename T = double>
void diff_cut_quad(T (*phi)(jlcxx::ArrayRef<T>), T (*dphi)(jlcxx::ArrayRef<T>),
                   int q, int bernstein_degree, const jlcxx::ArrayRef<T> x_min,
                   const jlcxx::ArrayRef<T> xmax, jlcxx::ArrayRef<T> x_work,
                   jlcxx::ArrayRef<T> wts_dot, jlcxx::ArrayRef<T> pts_dot,
                   jlcxx::ArrayRef<T> surf_wts_dot,
                   jlcxx::ArrayRef<T> surf_pts_dot);

/// Compute quadrature weights and points for a hypercube cut by a level set.
/// \tparam N - dimension of the hypercube 
/// \tparam T - floating point type used for the quadrature rule
/// \param[in] phi - function defining the level set
/// \param[in] q - number of Gaussian quadrature points in one-dimension
/// \param[in] bernstein_degree - the polynomial degree of the Bernstein fit
/// \param[in] x_min - minimum value of hypercube coordinates
/// \param[in] x_max - maximum value of hypercube coordinates
/// \param[in] x_work - work array that is passed to `phi`
/// \param[inout] wts - quadrature weights for the volume domain 
/// \param[inout] pts - quadrature points for the volume domain 
/// \note The `wts` and `pts` arrays should be empty on entry.  On exit, they
/// will have the weights and points.
template<int N, typename T = double>
void cut_cell_quad(T (*phi)(jlcxx::ArrayRef<T>), int q, int bernstein_degree,
                   const jlcxx::ArrayRef<T> x_min,
                   const jlcxx::ArrayRef<T> x_max, jlcxx::ArrayRef<T> x_work,
                   jlcxx::ArrayRef<T> wts, jlcxx::ArrayRef<T> pts);

/// Compute derivative of quadrature weights and points.
/// \tparam N - dimension of the hypercube 
/// \tparam T - floating point type used for the quadrature rule derivatives
/// \param[in] phi - function defining the level set
/// \param[in] dphi - derivative of `phi` with respect to some user parameter
/// \param[in] q - number of Gaussian quadrature points in one-dimension
/// \param[in] bernstein_degree - the polynomial degree of the Bernstein fit
/// \param[in] x_min - minimum value of hypercube coordinates
/// \param[in] x_max - maximum value of hypercube coordinates
/// \param[in] x_work - work array that is passed to `phi`
/// \param[inout] wts_dot - derivative of volume quadrature weights
/// \param[inout] pts_dot - derivative of volume quadrature points 
/// \note The `wts_dot` and `pts_dot` arrays should be empty on entry.  On 
/// exit, they will have the weights and points.
template<int N, typename T = double>
void diff_cell_quad(T (*phi)(jlcxx::ArrayRef<T>), T (*dphi)(jlcxx::ArrayRef<T>),
                    int q, int bernstein_degree, const jlcxx::ArrayRef<T> x_min,
                    const jlcxx::ArrayRef<T> xmax, jlcxx::ArrayRef<T> x_work,
                    jlcxx::ArrayRef<T> wts_dot, jlcxx::ArrayRef<T> pts_dot);

/// Compute quadrature weights and points for a surface inside a hypercube.
/// \tparam N - dimension of the hypercube 
/// \tparam T - floating point type used for the quadrature rule
/// \param[in] phi - function defining the surface via a level set
/// \param[in] q - number of Gaussian quadrature points in one-dimension
/// \param[in] bernstein_degree - the polynomial degree of the Bernstein fit
/// \param[in] x_min - minimum value of hypercube coordinates
/// \param[in] x_max - maximum value of hypercube coordinates
/// \param[in] x_work - work array that is passed to `phi`
/// \param[inout] surf_wts - quadrature weights for the surface `phi(x) = 0`
/// \param[inout] surf_pts - quadrature points for the surface `phi(x) = 0`
/// \note The `surf_wts` and `surf_pts` arrays should be empty on entry.  On 
/// exit, they will have the weights and points.
template<int N, typename T = double>
void cut_surf_quad(T (*phi)(jlcxx::ArrayRef<T>), int q, int bernstein_degree,
                   const jlcxx::ArrayRef<T> x_min,
                   const jlcxx::ArrayRef<T> x_max, jlcxx::ArrayRef<T> x_work,
                   jlcxx::ArrayRef<T> surf_wts, jlcxx::ArrayRef<T> surf_pts);

/// Compute derivative of surface quadrature weights and points.
/// \tparam N - dimension of the hypercube 
/// \tparam T - floating point type used for the quadrature rule derivatives
/// \param[in] phi - function defining the level set
/// \param[in] dphi - derivative of `phi` with respect to some user parameter
/// \param[in] q - number of Gaussian quadrature points in one-dimension
/// \param[in] bernstein_degree - the polynomial degree of the Bernstein fit
/// \param[in] x_min - minimum value of hypercube coordinates
/// \param[in] x_max - maximum value of hypercube coordinates
/// \param[in] x_work - work array that is passed to `phi`
/// \param[inout] surf_wts_dot - derivative of surface quadrature weights
/// \param[inout] surf_pts_dot - derivative of surface quadrature points
/// \note The `surf_wts_dot` and `surf_pts_dot` arrays should be empty on 
/// entry.  On exit, they will have the weights and points.
template<int N, typename T = double>
void diff_surf_quad(T (*phi)(jlcxx::ArrayRef<T>), T (*dphi)(jlcxx::ArrayRef<T>),
                    int q, int bernstein_degree, const jlcxx::ArrayRef<T> x_min,
                    const jlcxx::ArrayRef<T> xmax, jlcxx::ArrayRef<T> x_work,
                    jlcxx::ArrayRef<T> surf_wts_dot,
                    jlcxx::ArrayRef<T> surf_pts_dot);


JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    mod.method("calc_cut_quad1d", &calc_cut_quad<1,double>);
    mod.method("calc_cut_quad2d", &calc_cut_quad<2,double>);
    mod.method("calc_cut_quad3d", &calc_cut_quad<3,double>);
    mod.method("diff_cut_quad1d", &diff_cut_quad<1,double>);
    mod.method("diff_cut_quad2d", &diff_cut_quad<2,double>);
    mod.method("diff_cut_quad3d", &diff_cut_quad<3,double>);
    mod.method("cut_cell_quad1d", &cut_cell_quad<1,double>);
    mod.method("cut_cell_quad2d", &cut_cell_quad<2,double>);
    mod.method("cut_cell_quad3d", &cut_cell_quad<3,double>);
    mod.method("diff_cell_quad1d", &diff_cell_quad<1,double>);
    mod.method("diff_cell_quad2d", &diff_cell_quad<2,double>);
    mod.method("diff_cell_quad3d", &diff_cell_quad<3,double>);
    mod.method("cut_surf_quad1d", &cut_surf_quad<1,double>);
    mod.method("cut_surf_quad2d", &cut_surf_quad<2,double>);
    mod.method("cut_surf_quad3d", &cut_surf_quad<3,double>);
    mod.method("diff_surf_quad1d", &diff_surf_quad<1,double>);
    mod.method("diff_surf_quad2d", &diff_surf_quad<2,double>);
    mod.method("diff_surf_quad3d", &diff_surf_quad<3,double>);
}

#endif