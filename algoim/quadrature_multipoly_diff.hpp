#ifndef ALGOIM_QUADRATURE_MULTIPOLY_DIFF_HPP
#define ALGOIM_QUADRATURE_MULTIPOLY_DIFF_HPP

#include "quadrature_multipoly.hpp"

namespace algoim
{
    namespace detail
    {

        // Using the polynomials in phi, eliminate the axis k by restricting to faces, computing
        // discriminants and resultants, and storing the computed polynomials in psi.  Simultaneously,
        // find the corresponding derivatives for psi.
        template<int N>
        void eliminate_axis_diff(PolySet<N,ALGOIM_M>& phi, PolySet<N,ALGOIM_M>& phi_dot, int k,
                                 PolySet<N-1,ALGOIM_M>& psi, PolySet<N-1,ALGOIM_M>& psi_dot)
        {
            static_assert(N >= 2, "N >= 2 required to eliminate axis");
            assert(0 <= k && k < N);
            assert(psi.count() == 0);

            // For every phi(i) ...
            for (int i = 0; i < phi.count(); ++i)
            {
                const auto& p = phi.poly(i);
                const auto& mask = phi.mask(i);

                // Examine bottom and top faces in the k'th dimension
                for (int side = 0; side <= 1; ++side)
                {
                    xarray<real,N-1> p_face(nullptr, remove_component(p.ext(), k));
                    algoim_spark_alloc(real, p_face);
                    restrictToFace(p, k, side, p_face);
                    auto p_face_mask = nonzeroMask(p_face, restrictToFace(mask, k, side));
                    if (!maskEmpty(p_face_mask))
                    {
                        bernstein::autoReduction(p_face);
                        bernstein::normalise(p_face);
                        psi.push_back(p_face, p_face_mask);
                    }
                }

                // Consider discriminant
                xarray<real,N> p_k(nullptr, p.ext());
                algoim_spark_alloc(real, p_k);
                bernstein::elevatedDerivative(p, k, p_k);
                auto disc_mask = intersectionMask(p, mask, p_k, mask);
                if (!maskEmpty(disc_mask))
                {
                    // note: computed disc might have lower degree than the following
                    uvector<int,N-1> R = discriminantExtent(p.ext(), k);
                    xarray<real,N-1> disc(nullptr, R);
                    algoim_spark_alloc(real, disc);
                    if (discriminant(p, k, disc))
                    {
                        bernstein::normalise(disc);
                        psi.push_back(disc, collapseMask(disc_mask, k));
                    }
                }
            }

            // Consider every pairwise combination of resultants ...
            for (int i = 0; i < phi.count(); ++i) for (int j = i + 1; j < phi.count(); ++j)
            {
                const auto& p = phi.poly(i);
                const auto& pmask = phi.mask(i);
                const auto& q = phi.poly(j);
                const auto& qmask = phi.mask(j);
                auto mask = intersectionMask(p, pmask, q, qmask);
                if (!maskEmpty(mask))
                {
                    // note: computed resultant might have lower degree than the following
                    uvector<int,N-1> R = resultantExtent(p.ext(), q.ext(), k);
                    xarray<real,N-1> res(nullptr, R);
                    algoim_spark_alloc(real, res);
                    if (resultant(p, q, k, res))
                    {
                        bernstein::normalise(res);
                        psi.push_back(res, collapseMask(mask, k));
                    }
                }
            };
        }

    } // namespace detail


    template<int N>
    struct ImplicitPolyQuadratureDiff : public ImplicitPolyQuadrature<N>
    {
        PolySet<N,ALGOIM_M> phi_dot;                                            // Derivatives of phi w.r.t. coefficients
        ImplicitPolyQuadratureDiff<N-1> base_dot;                               // Derivative of base polynomials corresponding to removal of axis k
        std::array<std::tuple<int,ImplicitPolyQuadratureDiff<N-1>>,N-1> base_other_dot; // Stores derivatives of other base cases, besides k, when in aggregate mode    

        // Default ctor sets to an uninitialised state
        ImplicitPolyQuadratureDiff() : ImplicitPolyQuadrature() {}

        ImplicitPolyQuadratureDiff(const xarray<real,N>&) = delete;

        ImplicitPolyQuadratureDiff(const ImplicitPolyQuadrature&) = delete;

        // Build differentiated quadrature hierarchy for a domain implicitly defined by a single polynomial
        // Note: we could call base-class ctor, but we prefer p_dot to be pushed back at the same time as p
        ImplicitPolyQuadratureDiff(const xarray<real,N>& p, const xarray<real,N>& p_dot)
        {
            auto mask = detail::nonzeroMask(p, booluarray<N,ALGOIM_M>(true));
            if (!detail::maskEmpty(mask))
            {
                phi.push_back(p, mask);
                phi_dot.push_back(p_dot, mask);
            }
            build_diff(true, false);
        }

        // Build differentiated quadrature hierarchy for a domain implicitly defined by two polynomials
        // Note: we could call base-class ctor, but we prefer p_dot and q_dot to be pushed back at the same time as p and q
        ImplicitPolyQuadratureDiff(const xarray<real,N>& p, const xarray<real,N>& p_dot, const xarray<real,N>& q, const xarray<real,N>& q_dot)
        {
            {
                auto mask = detail::nonzeroMask(p, booluarray<N,ALGOIM_M>(true));
                if (!detail::maskEmpty(mask))
                {
                    phi.push_back(p, mask);
                    phi_dot.push_back(p_dot, mask);
                }
            }
            {
                auto mask = detail::nonzeroMask(q, booluarray<N,ALGOIM_M>(true));
                if (!detail::maskEmpty(mask))
                {
                    phi.push_back(q, mask);
                    phi_dot.push_back(q_dot, mask);
                }
            }
            build_diff(true, false);
        }

        // Build differentiated quadrature hierarchy for a given domain implicitly defined by two polynomials with user-defined masks
        // Note: we could call base-class ctor, but we prefer p_dot and q_dot to be pushed back at the same time as p and q
        ImplicitPolyQuadratureDiff(const xarray<real,N>& p, const booluarray<N,ALGOIM_M>& pmask, const xarray<real,N>& p_dot, const xarray<real,N>& q, const booluarray<N,ALGOIM_M>& qmask, const xarray<real,N>& q_dot)
        {
            {
                auto mask = detail::nonzeroMask(p, pmask);
                if (!maskEmpty(mask)) {
                    phi.push_back(p, mask);
                    phi_dot.push_back(p_dot, mask);
                }
            }
            {
                auto mask = detail::nonzeroMask(q, qmask);
                if (!maskEmpty(mask))
                {
                    phi.push_back(q, mask);
                    phi_dot.push_back(q_dot, mask);
                }
            }
            build_diff(true, false);
        }


        // Assuming phi has been instantiated, determine elimination axis and build base
        void build_diff(bool outer, bool auto_apply_TS)
        {
            std::cout << "Inside build_diff..." << std::endl;
            type = outer ? OuterSingle : Inner;
            this->auto_apply_TS = auto_apply_TS;

            // If phi is empty, apply a tensor-product Gaussian quadrature
            if (phi.count() == 0)
            {
                k = N;
                this->auto_apply_TS = false;
                return;
            }

            if constexpr (N == 1)
            {
                // If in one dimension, there is only one choice of height direction and
                // the recursive process halts
                k = 0;
                return;
            }
            else
            {
                // Compute score; penalise any directions which likely contain vertical tangents
                uvector<bool,N> has_disc;
                uvector<real,N> score = detail::score_estimate(phi, has_disc);
                assert(max(abs(score)) > 0);
                score /= 2 * max(abs(score));
                for (int i = 0; i < N; ++i)
                    if (!has_disc(i))
                        score(i) += 1.0;

                // Choose height direction and form base polynomials; if tanh-sinh is being used at this
                // level, suggest the same all the way down; moreover, suggest tanh-sinh if a non-empty
                // discriminant mask has been found
                k = argmax(score);
                detail::eliminate_axis(phi, phi_dot, k, base.phi, base.phi_dot);
                base.build_diff(false, this->auto_apply_TS || has_disc(k));

                // If this is the outer integral, and surface quadrature schemes are required, apply
                // the dimension-aggregated scheme when necessary
                if (outer && has_disc(k))
                {
                    type = OuterAggregate;
                    for (int i = 0; i < N; ++i) if (i != k)
                    {
                        auto& [kother, base] = base_other[i < k ? i : i - 1];
                        kother = i;
                        detail::eliminate_axis(phi, kother, base.phi);
                        // In aggregate mode, triggered by non-empty discriminant mask,
                        // base integrals always have T-S suggested
                        base.build_diff(false, this->auto_apply_TS || true);
                    }
                }
            }
        }
    
    private:
        using ImplicitPolyQuadrature::ImplicitPolyQuadrature;
    };
    
    template<> struct ImplicitPolyQuadratureDiff<0> {};
} // namespace algoim

#endif