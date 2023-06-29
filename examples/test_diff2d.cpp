// Simple test of differentiation of two-dimensional quadrature rules

#include <iostream>
#include <iomanip>
#include <fstream>
#include "quadrature_multipoly.hpp"
#include "dual.hpp"

using namespace algoim;
using namespace duals;

template <int N, typename T>
void getQuadrature(const xarray<T,N>& phi, int q, std::vector<uvector<T,N>>& pts0, std::vector<T>& wts0,
                   std::vector<uvector<T,N>>& pts1, std::vector<T>& wts1,
                   std::vector<uvector<T,N>>& pts_surf, std::vector<uvector<T,N>>& wts_surf)
{
    ImplicitPolyQuadrature<N,T> ipquad(phi);
    ipquad.integrate(AutoMixed, q, [&](const uvector<T,2>& x, T w)
    {
        if (bernstein::evalBernsteinPoly(phi, x) < 0)
        {
            pts0.push_back(x);
            wts0.push_back(w);
        }
        else
        {
            pts1.push_back(x);
            wts1.push_back(w);
        }
    });
    ipquad.integrate_surf(AutoMixed, q, [&](const uvector<T,2>& x, T w, const uvector<T,2>& wn)
    {
        pts_surf.push_back(x);
        wts_surf.push_back(wn);
    });
}

int main(int argc, char* argv[])
{

    // For the LSF phi, we consider an ellipse centered at (-1,0.5) with 
    // semi-major axis of 1.5 (horizontal) and semi-minor axis of 0.5 
    // (vertical).  This has the ellipse peirce the box [0,1]^2 along the left
    // side.
    auto phi = [](const uvector<real,2>& x)
    {
        return 4*pow(x(0) + 1, 2) + 36*pow(x(1) - 0.5, 2) - 9;
    };

    int q = 4; // number of quadrature points
    real xmin = 0.0;
    real xmax = 1.0;
    const real eps_fd = 1e-6; //pow(std::numeric_limits<real>::epsilon(), 1/3);
    const real tol = 1.0e6 * std::numeric_limits<real>::epsilon();

    // Construct Bernstein polynomial representation of phi
    int P = 3;
    xarray<real,2> phipoly(nullptr, P);
    algoim_spark_alloc(real, phipoly);
    bernstein::bernsteinInterpolate<2>([&](const uvector<real,2>& x) { return phi(xmin + x * (xmax - xmin)); }, phipoly);

    // Build quadrature "hierarchy"
    ImplicitPolyQuadrature<2> ipquad(phipoly);

    // Compute quadrature scheme and record the nodes & weights; phase0 corresponds to
    // {phi < 0}, phase1 corresponds to {phi > 0}, and surf corresponds to {phi == 0}.
    std::vector<uvector<real,2>> phase0_pts, phase1_pts, surf_pts, surf_wts;
    std::vector<real> phase0_wts, phase1_wts;
    getQuadrature(phipoly, q, phase0_pts, phase0_wts, phase1_pts, phase1_wts, surf_pts, surf_wts);

    // display the quadrature points to screen 
    std::cout << "Phase 0 quadrature points: " << std::endl;
    for (auto & node : phase0_pts) {
        std::cout << node << std::endl;
    }
    std::cout << "Phase 1 quadrature points: " << std::endl;
    for (auto & node : phase1_pts) {
        std::cout << node << std::endl;
    }
    std::cout << "Surf quadrature points: " << std::endl;
    for (auto & node : surf_pts) {
        std::cout << node << std::endl;
    }
    std::cout << std::endl;

    // Differentiate w.r.t. phipoly coefficients
    std::vector<uvector<real,2>> phase0_pts_m, phase1_pts_m, surf_pts_m, surf_wts_m;
    std::vector<real> phase0_wts_m, phase1_wts_m;

    // Store a dual number version of the Bernstein coefficients
    xarray<duald,2> phipoly_dot(nullptr, P);
    algoim_spark_alloc(duald, phipoly_dot);
    for (int i = 0; i < phipoly.size(); ++i)
    {
        phipoly_dot[i].rpart(phipoly[i]);
        phipoly_dot[i].dpart(0.0);
    }
    std::vector<uvector<duald,2>> phase0_pts_d, phase1_pts_d, surf_pts_d, surf_wts_d;
    std::vector<duald>  phase0_wts_d, phase1_wts_d;
    
    for (int p = 0; p < phipoly.size(); ++p)
    {

        // Compute the derivatives using central difference method 
        phipoly[p] += eps_fd;
        phase0_pts.clear(); phase0_wts.clear(); phase1_pts.clear(); phase1_wts.clear(); surf_pts.clear(); surf_wts.clear();
        getQuadrature(phipoly, q, phase0_pts, phase0_wts, phase1_pts, phase1_wts, surf_pts, surf_wts);
        phipoly[p] -= 2*eps_fd;
        phase0_pts_m.clear(); phase0_wts_m.clear(); phase1_pts_m.clear(); phase1_wts_m.clear(); surf_pts_m.clear(); surf_wts_m.clear();
        getQuadrature(phipoly, q, phase0_pts_m, phase0_wts_m, phase1_pts_m, phase1_wts_m, surf_pts_m, surf_wts_m);
        phipoly[p] += eps_fd;        

        // Compute the derivatives using dual numbers
        phipoly_dot[p].dpart(1.0);
        phase0_pts_d.clear(); phase0_wts_d.clear(); phase1_pts_d.clear(); phase1_wts_d.clear(); surf_pts_d.clear(); surf_wts_d.clear();
        getQuadrature(phipoly_dot, q, phase0_pts_d, phase0_wts_d, phase1_pts_d, phase1_wts_d, surf_pts_d, surf_wts_d);
        phipoly_dot[p].dpart(0.0);

        // compare against FD values 
        std::cout << "==========================================================================================" << std::endl;
        std::cout << "Derivatives w.r.t. Bernstein coefficient " << p << std::endl;
        std::cout << "Phase 0 quadrature point derivatives and errors:" << std::endl;        
        for (int j = 0; j < phase0_pts_d.size(); ++j)
        {
            for (int i = 0; i < 2; ++i)
            {
                real diff_fd = (phase0_pts[j](i) - phase0_pts_m[j](i))/(2*eps_fd);
                real error = abs(diff_fd - phase0_pts_d[j](i).dpart());
                assert(error < tol && "AD derivative does not match analytical value.");
                std::cout << "-------------------------------" << std::endl;
                std::cout << "point and coord. indices (j,i) = (" << j << "," << i << "):" << std::endl;
                std::cout << "\tFD    = " << diff_fd << std::endl;
                std::cout << "\tAD    = " << phase0_pts_d[j](i).dpart() << std::endl;
                std::cout << "\tdiff. = " << error << std::endl;
            }
        }
        std::cout << std::endl;
        std::cout << "Phase 0 quadrature weight derivatives and errors:" << std::endl;        
        for (int j = 0; j < phase0_pts_d.size(); ++j)
        {
            real diff_fd = (phase0_wts[j] - phase0_wts_m[j])/(2*eps_fd);
            real error = abs(diff_fd - phase0_wts_d[j].dpart());
            assert(error < tol && "AD derivative does not match analytical value.");
            std::cout << "-------------------------------" << std::endl;
            std::cout << "point index (j) = (" << j << "):" << std::endl;
            std::cout << "\tFD    = " << diff_fd << std::endl;
            std::cout << "\tAD    = " << phase0_wts_d[j].dpart() << std::endl;
            std::cout << "\tdiff. = " << error << std::endl;
        }
        std::cout << std::endl;
        std::cout << "Phase 1 quadrature point derivatives and errors:" << std::endl;        
        for (int j = 0; j < phase1_pts_d.size(); ++j)
        {
            for (int i = 0; i < 2; ++i)
            {
                real diff_fd = (phase1_pts[j](i) - phase1_pts_m[j](i))/(2*eps_fd);
                real error = abs(diff_fd - phase1_pts_d[j](i).dpart());
                assert(error < tol && "AD derivative does not match analytical value.");
                std::cout << "-------------------------------" << std::endl;
                std::cout << "point and coord. indices (j,i) = (" << j << "," << i << "):" << std::endl;
                std::cout << "\tFD    = " << diff_fd << std::endl;
                std::cout << "\tAD    = " << phase1_pts_d[j](i).dpart() << std::endl;
                std::cout << "\tdiff. = " << error << std::endl;
            }
        }
        std::cout << std::endl;
        std::cout << "Phase 1 quadrature weight derivatives and errors:" << std::endl;        
        for (int j = 0; j < phase1_pts_d.size(); ++j)
        {
            real diff_fd = (phase1_wts[j] - phase1_wts_m[j])/(2*eps_fd);
            real error = abs(diff_fd - phase1_wts_d[j].dpart());
            assert(error < tol && "AD derivative does not match analytical value.");
            std::cout << "-------------------------------" << std::endl;
            std::cout << "point index (j) = (" << j << "):" << std::endl;
            std::cout << "\tFD    = " << diff_fd << std::endl;
            std::cout << "\tAD    = " << phase1_wts_d[j].dpart() << std::endl;
            std::cout << "\tdiff. = " << error << std::endl;
        }
        std::cout << std::endl;
        std::cout << "Surface quadrature point derivatives and errors:" << std::endl;        
        for (int j = 0; j < surf_pts_d.size(); ++j)
        {
            for (int i = 0; i < 2; ++i)
            {
                real diff_fd = (surf_pts[j](i) - surf_pts_m[j](i))/(2*eps_fd);
                real error = abs(diff_fd - surf_pts_d[j](i).dpart());
                assert(error < tol && "AD derivative does not match analytical value.");
                std::cout << "-------------------------------" << std::endl;
                std::cout << "point and coord. indices (j,i) = (" << j << "," << i << "):" << std::endl;
                std::cout << "\tFD    = " << diff_fd << std::endl;
                std::cout << "\tAD    = " << surf_pts_d[j](i).dpart() << std::endl;
                std::cout << "\tdiff. = " << error << std::endl;
            }
        }
        std::cout << std::endl;
        std::cout << "Surface quadrature weight derivatives and errors:" << std::endl;        
        for (int j = 0; j < surf_pts_d.size(); ++j)
        {
            for (int i = 0; i < 2; ++i)
            {
                real diff_fd = (surf_wts[j](i) - surf_wts_m[j](i))/(2*eps_fd);
                real error = abs(diff_fd - surf_wts_d[j](i).dpart());
                assert(error < tol && "AD derivative does not match analytical value.");
                std::cout << "-------------------------------" << std::endl;
                std::cout << "point and coord. indices (j,i) = (" << j << "," << i << "):" << std::endl;
                std::cout << "\tFD    = " << diff_fd << std::endl;
                std::cout << "\tAD    = " << surf_wts_d[j](i).dpart() << std::endl;
                std::cout << "\tdiff. = " << error << std::endl;
            }
        }
        std::cout << std::endl;

    }
}