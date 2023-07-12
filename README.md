# Algoim
### Algorithms for implicitly defined geometry, level set methods, and Voronoi implicit interface methods.

For a description, examples, and install instructions, see the [Algoim GitHub page](https://algoim.github.io/).

## Modifications

This folk is a slightly modified of version of Saye's original library.  The two main additions are as follows:

1. We differentiated `algoim/quadrature_multipoly.hpp` and its dependencies using [cppduals](https://tesch1.gitlab.io/cppduals/index.html), except for those
   parts that rely on LAPACk routines.  For the LAPACK routines we had to do some old-fashioned "hand-differentiation."  cppduals
   is a header-only library, and it is included directly as `algoim/dual.hpp`.
3. We added `algoim/cutquad.hpp` and `algoim/cutquad.cpp` in order to faciliate wrapping the quadrature routines
   in Julia.

These additions/changes were only applied to the quadrature routines for domains implicitly-defined by multivariate polynomials.

## Notice

Algoim Copyright (c) 2022, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy). All rights reserved.

This software was developed under funding from the
U.S. Department of Energy and the U.S. Government consequently retains
certain rights. As such, the U.S. Government has been granted for
itself and others acting on its behalf a paid-up, nonexclusive,
irrevocable, worldwide license in the Software to reproduce,
distribute copies to the public, prepare derivative works, and perform
publicly and display publicly, and to permit others to do so.

License for Algoim can be found at [LICENSE](LICENSE).

License for cppduals can be found at [LICENSE](CPPDUALS_LICENSE)

## Citations 

To cite Algoim, please follow the guide on the [Algoim GitHub page](https://algoim.github.io/).

To cite cppduals, please consider [![DOI](https://joss.theoj.org/papers/10.21105/joss.01487/status.svg)](https://doi.org/10.21105/joss.01487)
