#include "quadrature_multipoly.hpp"
#include "dual.hpp"

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/array.hpp"
#include "jlcxx/functions.hpp"

#include "cutquad.hpp"

using namespace algoim;
using namespace duals;

template<int N, typename T>
void calc_cut_quad(T (*phi)(jlcxx::ArrayRef<T>), int q, int bernstein_degree,
                   const jlcxx::ArrayRef<T> x_min,
                   const jlcxx::ArrayRef<T> x_max, jlcxx::ArrayRef<T> x_work,
                   jlcxx::ArrayRef<T> wts, jlcxx::ArrayRef<T> pts,
                   jlcxx::ArrayRef<T> surf_wts, jlcxx::ArrayRef<T> surf_pts)
{
    // Find the Bernstein polynomial approximation of phi
    xarray<T,N> phipoly(nullptr, bernstein_degree+1);
    algoim_spark_alloc(T, phipoly);
    bernstein::bernsteinInterpolate<N>(
        [&](const uvector<T,N>& x) 
        {                  
            for (int i = 0; i < N; ++i)
            {                
                x_work[i] = x_min[i] + x(i)*(x_max[i] - x_min[i]);
            }
            return phi(x_work);
        }, 
        phipoly);
    // Get quadrature object
    ImplicitPolyQuadrature<N,T> ipquad(phipoly);

    // compute scaling for quadrature weights
    T wts_fac = 1;
    std::array<T,N> nrm_fac;
    for (size_t di = 0; di < N; ++di)
    {
        nrm_fac[di] = 1;
        wts_fac *= (x_max[di] - x_min[di]);
        for (size_t j = 0; j < N-1; ++j)
        {
            size_t it = (di + j + 1) % N;
            nrm_fac[di] *= (x_max[it] - x_min[it]);
        }
    }

    // find the quadrature weights and points 
    ipquad.integrate(AutoMixed, q,
        [&](const uvector<T,N>& x, T w)
        {
            if (bernstein::evalBernsteinPoly(phipoly, x) > 0)
            {
                wts.push_back(w*wts_fac);
                for (size_t di = 0; di < N; ++di)
                {
                    pts.push_back(x_min[di] + x(di)*(x_max[di] - x_min[di]));
                }
            }
        });

    if constexpr(N==1)
    {
        // no need to integrate over a surface in 1D, but we need to find where phi's roots are
        xarray<T,1> roots(nullptr, bernstein_degree);
        algoim_spark_alloc(T, roots);

        int count = bernstein::bernsteinUnitIntervalRealRoots(phipoly.data(), bernstein_degree+1, roots.data());
        for (int i = 0; i < count; ++i)
        {
            // evaluate the poly derivative to determine sign of weight 
            real x = roots[i];
            real poly, dpoly;
            bernstein::bernsteinValueAndDerivative(phipoly.data(), bernstein_degree+1, Binomial::row(bernstein_degree), x, poly, dpoly);
            //std::cout << "poly (root test) = " << poly << std::endl;
            assert( fabs(poly) < 1e-12 );
            dpoly > 0 ? surf_wts.push_back(-nrm_fac[0]) : surf_wts.push_back(nrm_fac[0]);
            surf_pts.push_back(x_min[0] + x*(x_max[0] - x_min[0]));
        }
    }
    else
    {
        ipquad.integrate_surf(AutoMixed, q, 
            [&](const uvector<T,N>& x, T w, const uvector<T,N>& wn)
            {
                for (size_t di = 0; di < N; ++di)
                {
                    surf_wts.push_back(wn(di)*nrm_fac[di]);
                    surf_pts.push_back(x_min[di] + 
                                       x(di)*(x_max[di] - x_min[di]));
                }
            });
    }
};

template<int N, typename T>
void diff_cut_quad(T (*phi)(jlcxx::ArrayRef<T>), T (*dphi)(jlcxx::ArrayRef<T>), 
                   int q, int bernstein_degree, const jlcxx::ArrayRef<T> x_min, 
                   const jlcxx::ArrayRef<T> x_max, jlcxx::ArrayRef<T> x_work,
                   jlcxx::ArrayRef<T> wts_dot, jlcxx::ArrayRef<T> pts_dot,
                   jlcxx::ArrayRef<T> surf_wts_dot,
                   jlcxx::ArrayRef<T> surf_pts_dot)
{
    // Find the Bernstein polynomial approximation of phi
    xarray<dual<T>,N> phipoly_dual(nullptr, bernstein_degree+1);
    algoim_spark_alloc(dual<T>, phipoly_dual);
    bernstein::bernsteinInterpolate<N>(
        [&](const uvector<T,N>& x) 
        {                  
            for (int i = 0; i < N; ++i)
                x_work[i] = x_min[i] + x(i)*(x_max[i] - x_min[i]);
            return dual(phi(x_work), dphi(x_work));
        }, 
        phipoly_dual);
    // Get quadrature object
    ImplicitPolyQuadrature<N,dual<T>> ipquad_dual(phipoly_dual);

    // compute scaling for quadrature weights
    T wts_fac = 1;
    std::array<T,N> nrm_fac;
    for (size_t di = 0; di < N; ++di)
    {
        nrm_fac[di] = 1;
        wts_fac *= (x_max[di] - x_min[di]);
        for (size_t j = 0; j < N-1; ++j)
        {
            size_t it = (di + j + 1) % N;
            nrm_fac[di] *= (x_max[it] - x_min[it]);
        }
    }

    // find the derivatives of the quadrature weights and points 
    ipquad_dual.integrate(AutoMixed, q,
        [&](const uvector<dual<T>,N>& x_dual, dual<T> w_dual)
        {
            if (bernstein::evalBernsteinPoly(phipoly_dual, x_dual) > 0)
            {
                wts_dot.push_back(w_dual.dpart()*wts_fac);
                for (size_t di = 0; di < N; ++di)
                {
                    pts_dot.push_back(
                        x_dual(di).dpart()*(x_max[di] - x_min[di]));
                }
            }
        });

    if constexpr(N==1)
    {
        xarray<dual<T>,1> roots(nullptr, bernstein_degree);
        algoim_spark_alloc(dual<T>, roots);
        int count = bernstein::bernsteinUnitIntervalRealRoots(phipoly_dual.data(), bernstein_degree+1, roots.data());
        for (int i = 0; i < count; ++i)
        {
            dual<T> x_dual = roots[i];
            surf_wts_dot.push_back(0.0);
            surf_pts_dot.push_back(x_dual.dpart()*(x_max[0] - x_min[0]));
        }
    }
    else
    {
        ipquad_dual.integrate_surf(AutoMixed, q, 
            [&](const uvector<dual<T>,N>& x_dual, dual<T> w_dual,
                const uvector<dual<T>,N>& wn_dual)
            {
                for (size_t di = 0; di < N; ++di)
                {
                    surf_wts_dot.push_back(wn_dual(di).dpart()*nrm_fac[di]);
                    surf_pts_dot.push_back(
                        x_dual(di).dpart()*(x_max[di] - x_min[di]));
                }
            });
    }
};

template<int N, typename T>
void cut_cell_quad(T (*phi)(jlcxx::ArrayRef<T>), int q, int bernstein_degree,
                   const jlcxx::ArrayRef<T> x_min,
                   const jlcxx::ArrayRef<T> x_max, jlcxx::ArrayRef<T> x_work,
                   jlcxx::ArrayRef<T> wts, jlcxx::ArrayRef<T> pts)
{
    // Find the Bernstein polynomial approximation of phi
    xarray<T,N> phipoly(nullptr, bernstein_degree+1);
    algoim_spark_alloc(T, phipoly);
    bernstein::bernsteinInterpolate<N>(
        [&](const uvector<T,N>& x) 
        {                  
            for (int i = 0; i < N; ++i)
            {                
                x_work[i] = x_min[i] + x(i)*(x_max[i] - x_min[i]);
            }
            return phi(x_work);
        }, 
        phipoly);
    // Get quadrature object
    ImplicitPolyQuadrature<N,T> ipquad(phipoly);

    // compute scaling for quadrature weights
    T wts_fac = 1;
    std::array<T,N> nrm_fac;
    for (size_t di = 0; di < N; ++di)
    {
        nrm_fac[di] = 1;
        wts_fac *= (x_max[di] - x_min[di]);
        for (size_t j = 0; j < N-1; ++j)
        {
            size_t it = (di + j + 1) % N;
            nrm_fac[di] *= (x_max[it] - x_min[it]);
        }
    }

    // find the quadrature weights and points 
    ipquad.integrate(AutoMixed, q,
        [&](const uvector<T,N>& x, T w)
        {
            if (bernstein::evalBernsteinPoly(phipoly, x) > 0)
            {
                wts.push_back(w*wts_fac);
                for (size_t di = 0; di < N; ++di)
                {
                    pts.push_back(x_min[di] + x(di)*(x_max[di] - x_min[di]));
                }
            }
        });
};

template<int N, typename T>
void diff_cell_quad(T (*phi)(jlcxx::ArrayRef<T>), T (*dphi)(jlcxx::ArrayRef<T>), 
                    int q, int bernstein_degree, const jlcxx::ArrayRef<T> x_min, 
                    const jlcxx::ArrayRef<T> x_max, jlcxx::ArrayRef<T> x_work,
                    jlcxx::ArrayRef<T> wts_dot, jlcxx::ArrayRef<T> pts_dot)
{
    // Find the Bernstein polynomial approximation of phi
    xarray<dual<T>,N> phipoly_dual(nullptr, bernstein_degree+1);
    algoim_spark_alloc(dual<T>, phipoly_dual);
    bernstein::bernsteinInterpolate<N>(
        [&](const uvector<T,N>& x) 
        {                  
            for (int i = 0; i < N; ++i)
                x_work[i] = x_min[i] + x(i)*(x_max[i] - x_min[i]);
            return dual(phi(x_work), dphi(x_work));
        }, 
        phipoly_dual);
    // Get quadrature object
    ImplicitPolyQuadrature<N,dual<T>> ipquad_dual(phipoly_dual);

    // compute scaling for quadrature weights
    T wts_fac = 1;
    std::array<T,N> nrm_fac;
    for (size_t di = 0; di < N; ++di)
    {
        nrm_fac[di] = 1;
        wts_fac *= (x_max[di] - x_min[di]);
        for (size_t j = 0; j < N-1; ++j)
        {
            size_t it = (di + j + 1) % N;
            nrm_fac[di] *= (x_max[it] - x_min[it]);
        }
    }

    // find the derivatives of the quadrature weights and points 
    ipquad_dual.integrate(AutoMixed, q,
        [&](const uvector<dual<T>,N>& x_dual, dual<T> w_dual)
        {
            if (bernstein::evalBernsteinPoly(phipoly_dual, x_dual) > 0)
            {
                wts_dot.push_back(w_dual.dpart()*wts_fac);
                for (size_t di = 0; di < N; ++di)
                {
                    pts_dot.push_back(
                        x_dual(di).dpart()*(x_max[di] - x_min[di]));
                }
            }
        });
};

template<int N, typename T>
void cut_surf_quad(T (*phi)(jlcxx::ArrayRef<T>), int q, int bernstein_degree,
                   const jlcxx::ArrayRef<T> x_min,
                   const jlcxx::ArrayRef<T> x_max, jlcxx::ArrayRef<T> x_work,
                   jlcxx::ArrayRef<T> surf_wts, jlcxx::ArrayRef<T> surf_pts)
{
    // Find the Bernstein polynomial approximation of phi
    xarray<T,N> phipoly(nullptr, bernstein_degree+1);
    algoim_spark_alloc(T, phipoly);
    bernstein::bernsteinInterpolate<N>(
        [&](const uvector<T,N>& x) 
        {                  
            for (int i = 0; i < N; ++i)
            {                
                x_work[i] = x_min[i] + x(i)*(x_max[i] - x_min[i]);
            }
            return phi(x_work);
        }, 
        phipoly);
    // Get quadrature object
    ImplicitPolyQuadrature<N,T> ipquad(phipoly);

    // compute scaling for quadrature weights
    T wts_fac = 1;
    std::array<T,N> nrm_fac;
    for (size_t di = 0; di < N; ++di)
    {
        nrm_fac[di] = 1;
        wts_fac *= (x_max[di] - x_min[di]);
        for (size_t j = 0; j < N-1; ++j)
        {
            size_t it = (di + j + 1) % N;
            nrm_fac[di] *= (x_max[it] - x_min[it]);
        }
    }

    if constexpr(N==1)
    {
        // no need to integrate over a surface in 1D, but we need to find where phi's roots are
        xarray<T,1> roots(nullptr, bernstein_degree);
        algoim_spark_alloc(T, roots);

        int count = bernstein::bernsteinUnitIntervalRealRoots(phipoly.data(), bernstein_degree+1, roots.data());
        for (int i = 0; i < count; ++i)
        {
            // evaluate the poly derivative to determine sign of weight 
            real x = roots[i];
            real poly, dpoly;
            bernstein::bernsteinValueAndDerivative(phipoly.data(), bernstein_degree+1, Binomial::row(bernstein_degree), x, poly, dpoly);            
            //assert( fabs(poly) < 1e-12 );
            dpoly > 0 ? surf_wts.push_back(-nrm_fac[0]) : surf_wts.push_back(nrm_fac[0]);
            surf_pts.push_back(x_min[0] + x*(x_max[0] - x_min[0]));
        }
    }
    else
    {
        ipquad.integrate_surf(AutoMixed, q, 
            [&](const uvector<T,N>& x, T w, const uvector<T,N>& wn)
            {
                for (size_t di = 0; di < N; ++di)
                {
                    surf_wts.push_back(wn(di)*nrm_fac[di]);
                    surf_pts.push_back(x_min[di] + 
                                       x(di)*(x_max[di] - x_min[di]));
                }
            });
    }
};

template<int N, typename T>
void diff_surf_quad(T (*phi)(jlcxx::ArrayRef<T>), T (*dphi)(jlcxx::ArrayRef<T>), 
                   int q, int bernstein_degree, const jlcxx::ArrayRef<T> x_min, 
                   const jlcxx::ArrayRef<T> x_max, jlcxx::ArrayRef<T> x_work,
                   jlcxx::ArrayRef<T> surf_wts_dot,
                   jlcxx::ArrayRef<T> surf_pts_dot)
{
    // Find the Bernstein polynomial approximation of phi
    xarray<dual<T>,N> phipoly_dual(nullptr, bernstein_degree+1);
    algoim_spark_alloc(dual<T>, phipoly_dual);
    bernstein::bernsteinInterpolate<N>(
        [&](const uvector<T,N>& x) 
        {                  
            for (int i = 0; i < N; ++i)
                x_work[i] = x_min[i] + x(i)*(x_max[i] - x_min[i]);
            return dual(phi(x_work), dphi(x_work));
        }, 
        phipoly_dual);
    // Get quadrature object
    ImplicitPolyQuadrature<N,dual<T>> ipquad_dual(phipoly_dual);

    // compute scaling for quadrature weights
    T wts_fac = 1;
    std::array<T,N> nrm_fac;
    for (size_t di = 0; di < N; ++di)
    {
        nrm_fac[di] = 1;
        wts_fac *= (x_max[di] - x_min[di]);
        for (size_t j = 0; j < N-1; ++j)
        {
            size_t it = (di + j + 1) % N;
            nrm_fac[di] *= (x_max[it] - x_min[it]);
        }
    }

    if constexpr(N==1)
    {
        xarray<dual<T>,1> roots(nullptr, bernstein_degree);
        algoim_spark_alloc(dual<T>, roots);
        int count = bernstein::bernsteinUnitIntervalRealRoots(phipoly_dual.data(), bernstein_degree+1, roots.data());
        for (int i = 0; i < count; ++i)
        {
            dual<T> x_dual = roots[i];
            surf_wts_dot.push_back(0.0);
            surf_pts_dot.push_back(x_dual.dpart()*(x_max[0] - x_min[0]));
        }
    }
    else
    {
        ipquad_dual.integrate_surf(AutoMixed, q, 
            [&](const uvector<dual<T>,N>& x_dual, dual<T> w_dual,
                const uvector<dual<T>,N>& wn_dual)
            {
                for (size_t di = 0; di < N; ++di)
                {
                    surf_wts_dot.push_back(wn_dual(di).dpart()*nrm_fac[di]);
                    surf_pts_dot.push_back(
                        x_dual(di).dpart()*(x_max[di] - x_min[di]));
                }
            });
    }
};