// Simple test of differentiation of three-dimensional quadrature rules

#include <iostream>
#include <iomanip>
#include <fstream>
#include "quadrature_multipoly.hpp"
#include "dual.hpp"

using namespace algoim;
using namespace duals;

int main(int argc, char* argv[])
{

    real radius = 0.25;
    int q = 5; // number of quadrature points
    real xmin = 0.0;
    real xmax = 1.0;
    const real eps_fd = 1e-6; //pow(std::numeric_limits<real>::epsilon(), 1/3);
    const real tol = 1.0e6 * std::numeric_limits<real>::epsilon();

    // For the LSF phi, we consider an ellipse centered at (-1,0.5) with 
    // semi-major axis of 1.5 (horizontal) and semi-minor axis of 0.5 
    // (vertical).  This has the ellipse peirce the box [0,1]^2 along the left
    // side.
    auto phi = [&](const uvector<real,3>& x)
    {
        return pow(x(0), 2) + pow(x(1) - 0.5, 2) + pow(x(2) - 0.5, 2) - radius*radius;
    };

    // Construct Bernstein polynomial representation of phi
    int P = 3;
    xarray<real,3> phipoly(nullptr, P);
    algoim_spark_alloc(real, phipoly);
    bernstein::bernsteinInterpolate<3>([&](const uvector<real,3>& x) { return phi(xmin + x * (xmax - xmin)); }, phipoly);

    duald radius_dot(radius, 1.0); 
    auto phi_diff = [&](const uvector<real,3>& x)
    {
        return pow(x(0), 2) + pow(x(1) - 0.5, 2) + pow(x(2) - 0.5, 2) - radius_dot*radius_dot;
    };
    xarray<duald,3> phipoly_dot(nullptr, P);
    algoim_spark_alloc(duald, phipoly_dot);
    bernstein::bernsteinInterpolate<3>([&](const uvector<real,3>& x) { return phi_diff(xmin + x * (xmax - xmin)); }, phipoly_dot);

    // Get quadrature object
    ImplicitPolyQuadrature<3,duald> ipquad_dot(phipoly_dot);

    // compute the derivative of the volume and surface integrals
    duald volume_dot = 0.0;
    duald surf_dot = 0.0;
    ipquad_dot.integrate(AutoMixed, q, [&](const uvector<duald,3>& x, duald w)
    {
        //std::cout << "w.dpart() = " << w.dpart() << std::endl;
        if (bernstein::evalBernsteinPoly(phipoly_dot, x) > 0)
            volume_dot += w;
    });
    ipquad_dot.integrate_surf(AutoMixed, q, [&](const uvector<duald,3>& x, duald w, const uvector<duald,3>& wn)
    {
        std::cout << w.dpart() << std::endl;
        for (int i = 0; i < 3; ++i)
            std::cout << wn(i).dpart() << ", ";
        std::cout << std::endl;
        //surf_dot += sqrnorm(wn);
        surf_dot += w;
    });
    // scale appropriately
    volume_dot *= pow(xmax - xmin, 3);
    surf_dot *= pow(xmax - xmin, 2);

    std::cout << "Volume (exact)  = " << 1 - (2.0/3.0)*util::pi*pow(radius, 3) << std::endl;
    std::cout << "Volume (quad.)  = " << volume_dot.rpart() << std::endl;
    std::cout << std::endl;
    std::cout << "Volume derivative (exact)  = " << -2*util::pi*pow(radius, 2) << std::endl;
    std::cout << "Volume derivative (AD)     = " << volume_dot.dpart() << std::endl;
    std::cout << std::endl;

    std::cout << "Surface (exact)  = " << 2.0*util::pi*pow(radius, 2) << std::endl;
    std::cout << "Surface (quad.)  = " << surf_dot.rpart() << std::endl;
    std::cout << std::endl;
    std::cout << "Surface derivative (exact) = " << 4*util::pi*radius << std::endl;
    std::cout << "Surface derivative (AD)    = " << surf_dot.dpart() << std::endl;

}