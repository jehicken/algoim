// Simple test of differentiation of one-dimensional quadrature rules

#include <iostream>
#include <iomanip>
#include <fstream>
#include "quadrature_multipoly.hpp"
#include "gaussquad.hpp"
#include "dual.hpp"

using namespace algoim;

int main(int argc, char* argv[])
{
    using duals::duald;

    std::cout << "1D-Quadrature Differentiation Test.\n\n";
    std::cout << std::scientific << std::setprecision(10);

    // For the LSF phi, we consider a quadratic with one root at x = 0.5 within the range [xmin,xmax]
    auto phi = [](const uvector<real,1>& x)
    {
        return x(0)*x(0) - 0.25; //  (x(0) - 0.25)*(x(0) - 0.75);
    };
    int q = 10; // number of quadrature points
    real xmin = 0.0;
    real xmax = 1.0;
    real root = 0.5; // analytical location of the root in [0,1]

    const real tol = 1.0e1 * std::numeric_limits<real>::epsilon();

    // Construct Bernstein polynomial by mapping [0,1] onto bounding box [xmin,xmax]
    int P = 3;
    xarray<real,1> phipoly(nullptr, P);
    algoim_spark_alloc(real, phipoly);
    bernstein::bernsteinInterpolate<1>([&](const uvector<real,1>& x) { return phi(xmin + x * (xmax - xmin)); }, phipoly);

    // Build quadrature "hierarchy"
    ImplicitPolyQuadrature<1> ipquad(phipoly);

    // Compute quadrature scheme and record the nodes & weights; phase0 corresponds to
    // {phi < 0}, phase1 corresponds to {phi > 0}.
    std::vector<uvector<real,2>> phase0, phase1;
    ipquad.integrate(AutoMixed, q, [&](const uvector<real,1>& x, real w)
    {
        if (bernstein::evalBernsteinPoly(phipoly, x) < 0)
            phase0.push_back(add_component(x, 1, w));
        else
            phase1.push_back(add_component(x, 1, w));
    });

    // display the quadrature points to screen 
    std::cout << "Phase 0 quadrature: " << std::endl;
    for (auto & node : phase0) {
        std::cout << node << std::endl;
    }
    std::cout << "Phase 1 quadrature: " << std::endl;
    for (auto & node : phase1) {
        std::cout << node << std::endl;
    }
    std::cout << std::endl;

    // Differentiate w.r.t. phipoly coefficients
    for (int p = 0; p < phipoly.size(); ++p)
    {
        // hard coded values of d(root)/d\alpha, where \alpha are the Bernstein coefficents
        real drootdalpha = (p == 1) ? -0.5 : -0.25;

        // Compute the analytical values of the derivatives 
        std::vector<double> dwts0(q), dwts1(q), dpts0(q), dpts1(q);
        for (int j = 0; j < q; ++j)
        {
            //pts0[j] = xmin + GaussQuad::x(q, j)*(root - xmin);
            dpts0[j] = algoim::GaussQuad::x(q, j) * drootdalpha;
            //wts0[j] = GaussQuad::w(q, j)*(root - xmin);
            dwts0[j] = algoim::GaussQuad::w(q, j) * drootdalpha;
            //pts1[j] = root + Gaussquad::x(q, j)*(xmax - root);
            dpts1[j] = (1.0 -algoim::GaussQuad::x(q, j)) * drootdalpha;
            //wts1[j] = GaussQuad::w(q,j)*(xmax - root);
            dwts1[j] = -algoim::GaussQuad::w(q,j) * drootdalpha;
        }

        // Prepare to compute the derivatives using dual numbers
        xarray<duald,1> phipoly_dot(nullptr, P);
        algoim_spark_alloc(duald, phipoly_dot);
        for (int i = 0; i < phipoly.size(); ++i)
        {
            phipoly_dot[i].rpart(phipoly[i]);
            phipoly_dot[i].dpart(0.0);
            //std::cout << phipoly[i] << ", ";
        }
        phipoly_dot[p].dpart(1.0);

        // Build quadrature "hierarchy"
        ImplicitPolyQuadrature<1,duald> ipquad_dot(phipoly_dot);

        // Compute quadrature derivatives and record the nodes & weights; phase0 corresponds to
        // {phi < 0}, phase1 corresponds to {phi > 0}.
        std::vector<duald> phase0_pts, phase0_wts, phase1_pts, phase1_wts;
        ipquad_dot.integrate(AutoMixed, q, [&](const uvector<duald,1>& x, duald w)
        {
            if (bernstein::evalBernsteinPoly(phipoly_dot, x) < 0)
            {
                //phase0_dot.push_back(add_component(x, 1, w));
                phase0_pts.push_back(x(0));
                phase0_wts.push_back(w);
            }
            else
            {
                //phase1_dot.push_back(add_component(x, 1, w));
                phase1_pts.push_back(x(0));
                phase1_wts.push_back(w);
            }
        });

        // compare against analytical values 
        std::cout << "==========================================================================================" << std::endl;
        std::cout << "Derivatives w.r.t. Bernstein coefficient " << p << std::endl;
        std::cout << "Phase 0 quadrature point derivatives and errors:" << std::endl;        
        for (int j = 0; j < q; ++j)
        {
            real error = abs(dpts0[j] - phase0_pts[j].dpart());
            assert(error < tol && "AD derivative does not match analytical value.");
            std::cout << "j = " << j << ": exact = " << dpts0[j] << ": AD = " << phase0_pts[j].dpart() << ": diff. = " << error << std::endl;
            
        }
        std::cout << "Phase 0 quadrature weight derivatives and errors:" << std::endl;        
        for (int j = 0; j < q; ++j)
        {
            real error = abs(dwts0[j] - phase0_wts[j].dpart());
            assert(error < tol && "AD derivative does not match analytical value.");
            std::cout << "j = " << j << ": exact = " << dwts0[j] << ": AD = " << phase0_wts[j].dpart() << ": diff. = " << error << std::endl;
        }
        std::cout << "Phase 1 quadrature point derivatives and errors:" << std::endl;        
        for (int j = 0; j < q; ++j)
        {
            real error = abs(dpts1[j] - phase1_pts[j].dpart());
            assert(error < tol && "AD derivative does not match analytical value.");
            std::cout << "j = " << j << ": exact = " << dpts1[j] << ": AD = " << phase1_pts[j].dpart() << ": diff. = " << error << std::endl;
        }
        std::cout << "Phase 1 quadrature weight derivatives and errors:" << std::endl;        
        for (int j = 0; j < q; ++j)
        {
            real error = abs(dwts1[j] - phase1_wts[j].dpart());
            assert(error < tol && "AD derivative does not match analytical value.");
            std::cout << "j = " << j << ": exact = " << dwts1[j] << ": AD = " << phase1_wts[j].dpart() << ": diff. = " << error << std::endl;
        }
    }

    return 0;
}
