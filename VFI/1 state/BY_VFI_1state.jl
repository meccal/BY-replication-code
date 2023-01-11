"""
LUCA MECCA
lmecca@london.edu
Replicate the results of Bansal, Yaron (2004) using value function iteration (VFI)
1 state variable version
September 2022
Written and tested in julia 1.8
"""

using Distributions, Statistics, DataFrames, Plots, StatsBase, Random

include("Log-Linear_1state.jl")
include("chebyshev_func.jl")
include("Quadrature.jl")

#################################################################
########################## CALIBRATION ##########################
#################################################################
#Take the parameters for the monthly BY04 calibration

#parameters defining the consumption, long run growth
μ_c=0.0015 #mean consumption growth
ρ=0.979 #LRR persistence
ϕ_e=0.044 #LRR volatility multiple
μ_d=0.0015 #mean dividend growth
ϕ=3 #dividend leverage
ϕ_d=4.5 #dividend volatility multiple
σ=0.0079 #deterministic volatility


#preference parameters
γ=10 #risk aversion
ψ=1.5 #EIS
β=0.998 #time discount factor

#compute the theta parameter of EZ preferences
θ=(1-γ)/(1-1/ψ)


#Chebyshev parameters
n_x=3 #degree of approximation of the Chebyshev polynomials

#Gauss-Hermite parameters
n=5 #degree of approximation of Gauss-Hermite parameters
#From Judd (1998)
nodes = Dict(2 => [-0.7071067811, 0.7071067811], 3 => [-1.224744871, 0, 1.224744871], 4 => [-1.650680123, -0.5246476232, 0.5246476232, 1.650680123], 5=>[-2.02018287, -0.9585724646, 0, 0.9585724646, 2.02018287])
weights = Dict(2 => [0.8862269254, 0.8862269254], 3 => [0.2954089751, 0, 0.2954089751], 4 => [0.08131283544, 0.8049140900, 0.8049140900, 0.08131283544], 5=>[0.01995324204, 0.3936193231, 0.9453087204, 0.3936193231, 0.01995324204])


#Simulation parameters
T=1100000 #simulation periods
brns=100000 #burn-in period

#################################################################
######################### PREPARATION ###########################
#################################################################
#Simulate T values for the long run growth
Random.seed!(1234)
x_shocks=rand(Normal(), T) #draw standard Normally distributed shocks
x_range=Array{Float64}(undef,T,1)
x_range[1]= ϕ_e*σ*x_shocks[1] #set x[0]=0
for i in 2:T
    x_range[i]=ρ*x_range[i-1]+ϕ_e*σ*x_shocks[i] 
end
x_range=x_range[brns+1:T] #discard burn-in draws
#Define the boundaries of the grid
x_max=maximum(x_range)
x_min=minimum(x_range)
#Convert the grid to the [-1, 1] space
x_range_Ch=[X_to_Ch(x_min, x_max, x) for x in x_range]

#compute Chebyshev zeros (i.e. Chebyshev grid)
zero_nodes=[Ch_zero(i,n_x+1) for i in 1:(n_x+1)]

#and transform them to the X space
zero_nodes_X=[Ch_to_X(x_min, x_max, k) for k in zero_nodes]


#################################################################################
########################## Wealth to consumption ratio ##########################
#################################################################################
#We parameterize the policy functions using n_x degree polynomials over an n_x+1 Chebyshev grid
#Compute the log-linear parameters
A0, A1, z_pc, k1, k0=log_pc(γ,ψ,β, μ_c, ρ, σ, θ, ϕ_e)
#Now compute the policy function for the pc ratio
pc_log=log_pf(sort(zero_nodes_X), A0, A1)

#compute the part of the equation that does not include parameters
known_part=[β^θ*exp((1-γ)*(μ_c+x_t)+0.5*(1-γ)^2*σ^2) for x_t in zero_nodes_X]

#Set the starting point for the coefficients using log-linear solutions
α=zeros(n_x+1)
α[1]=exp(A0)
#Initialize the policy function
z_t=[Ch_pol(α, x_t, n_x) for x_t in zero_nodes]

#define initial value for the difference
difference=1
#Set a count to record the number of iterations before convergence
counter_WC=0
#create a list that stores the progression of the error
error_WC=ones(0)

@time begin
    while difference>0.0001 && counter_WC<20000
        global counter_WC+=1
        if mod(counter_WC,100)==0
            print("Iteration number " * string(counter_WC) * " for WC\n")
        end

        #compute the expectations part that includes time t+1 wealth-to-consumption
        #We use Guass-Hermite quadrature
        z_t1=[GH_WC(α, n, n_x, x_min, x_max, σ*ϕ_e, ρ*x_t, nodes, weights, θ) for x_t in zero_nodes_X]

        #compute the next W/C ratio
        global z_t_1=(known_part.*z_t1).^(1/θ).+1

        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(z_t_1-z_t))
        append!(error_WC, difference)
        #If the difference is too large, set PC=PC_1 and restart:
        global z_t=z_t_1

        #Regress the new guess of the value function on the Chebyshev basis to update the coefficients
        basis_functions=Matrix{Float64}(undef,n_x+1,n_x+1)
        for i in 1:(n_x+1)
            basis_functions[i,:]=Ch_basis(zero_nodes[i], n_x)
        end
        global α=inv(basis_functions' * basis_functions)*basis_functions'*z_t
    end
end #time recording

#policy function for the wealth-to-consumption ratio (in levels) at Chebyshev nodes
wc_pf=[Ch_pol(α, x, n_x) for x in sort(zero_nodes)]

#Plot
plot(log.(wc_pf),title="Projection and log-linear PC/WC ratio", label="Projection", legend=:topright)
plot!(pc_log, label="Log-linear") 



#################################################################################
############################ Price to dividend ratio ############################
#################################################################################
#Compute the log-linear parameters
A0m, A1m, z_pd=log_pd(γ,ψ,β, μ_c, μ_d, ρ, σ, θ, ϕ_e, ϕ_d, ϕ, A1, A0, k1, k0)
#Now compute the policy function for the pc ratio at the collocation nodes
pd_log=log_pf(sort(zero_nodes_X), A0m, A1m)

#compute the part of the equation that does not include parameters
known_part=[β^θ*exp(-γ*μ_c+μ_d+x_t*(ϕ-γ)+0.5*σ^2*(γ^2+ϕ_d^2))*
(Ch_pol(α,X_to_Ch(x_min, x_max, x_t),n_x)-1)^(1-θ)
for x_t in zero_nodes_X]

#Set the starting point for the coefficients using log-linear solutions
ζ=zeros(n_x+1)
ζ[1]=exp(A0m)
#Initialize the policy function
z_t=[(Ch_pol(ζ, x_t,n_x)) for x_t in zero_nodes]

#define initial value for the difference
difference=1
#Set a count to record the number of iterations before convergence
counter_PD=0
#create a list that stores the progression of the error
error_PD=ones(0)

@time begin
    while difference>0.0001 && counter_PD<20000
        global counter_PD+=1
        if mod(counter_PD,100)==0
            print("Iteration number " * string(counter_PD) * " for PD\n")
        end

        #compute the expectations part that includes time t+1 pd ratio
          #We use Guass-Hermite quadrature
        z_t1=[GH_PD(α, ζ, n, n_x, x_min, x_max, σ*ϕ_e, ρ*x_t, nodes, weights, θ) for x_t in zero_nodes_X]

        #compute the 
        global z_t_1=known_part.*z_t1

        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(z_t_1-z_t))
        append!(error_PD, difference)
        #If the difference is too large, set PC=PC_1 and restart:
        global z_t=z_t_1

        #Regress the new guess of the value function on the Chebyshev basis to update the coefficients
        basis_functions=Matrix{Float64}(undef,n_x+1,n_x+1)
        for i in 1:(n_x+1)
            basis_functions[i,:]=Ch_basis(zero_nodes[i], n_x)
        end
        global ζ=inv(basis_functions' * basis_functions)*basis_functions'*z_t
    end
end #time recording

#policy function for the price-to-dividend ratio (in levels)
pd_pf=[Ch_pol(ζ, x, n_x) for x in sort(zero_nodes)]
