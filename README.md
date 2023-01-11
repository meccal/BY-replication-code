# BY_replication_code

This project replicates Bansal-Yaron (2004) using different methods:
- Value function iteration (VFI) with Tauchen (1986) discretization of the state space.
- Value function iteration with Chebyshev polynomials.
- Projection Method (collocation).

Folders: BY_VFI, BY_Projection, BY_VFI_Tauchen\
Each folder contains the main file (e.g. BY_VFI.jl) and files with the functions that I created for this project. Note that I tried to avoid built-in packages as much as possible.
In particular I created functions to generate Chebyshev bases and polynomials (chebyshev_func.jl) and to compute the Gauss-Hermite quadrature (Quadrature.jl).
Each folder contains the 1 state variable (no stochastic volatility) and the 2 state variables alternatives.

Folder: Simulations_analysis\
Contains the files that simulate the models and compute moments of the consumption and dividend growth, as well as of financial returns and valuation ratios.

For any question or comment, please contact me at lmecca@london.edu. 
