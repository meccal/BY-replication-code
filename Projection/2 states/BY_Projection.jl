"""
LUCA MECCA
lmecca@london.edu
Replicate the results of Bansal, Yaron (2004) using the Projection method with Chebyshev polynomials (Collocation method)
2 state variables version
November 2022
Written and tested in julia 1.8
"""

using Distributions, Statistics, DataFrames, Plots, StatsBase, NLsolve, Random

include("chebyshev_func.jl")
include("Log-Linear.jl")
include("Quadrature.jl")


################################################################
########################## PARAMETERS ##########################
################################################################
#Take the parameters for the monthly BY04 calibration

#parameters defining the consumption, long run growth and stochastic volatility processes
μ_c=0.0015 #mean consumption growth
ρ=0.979 #LRR persistence
ϕ_e=0.044 #LRR volatility multiple
μ_d=0.0015 #mean dividend growth
ϕ=3 #dividend leverage
ϕ_d=4.5 #dividend volatility multiple
σ=0.0079 #baseline volatility
σ_ω=0.0000023 #volatility of volatility
ν=0.987 #persistence of volatility

#preference parameters
γ=10 #risk aversion
ψ=1.5 #EIS
β=0.998 #time discount factor

#compute the theta parameter of EZ preferences
θ=(1-γ)/(1-1/ψ)

#Chebyshev parameters
n_x=3 #degree of approximation of the Chebyshev polynomials for long-rung growth
n_σ=3 #degree of approximation of the Chebyshev polynomials for stochastic volatility
product="tensor" #choose "tensor" if you want to use the n-folder tensor product as basis functions
                 #choose "complete" if you want to use the complete set of polynomials up to a certain degree


#Gauss-Hermite parameters
n=5 #degree of approximation of Gauss-Hermite parameters
#From Judd (1998)
nodes = Dict(2 => [-0.7071067811, 0.7071067811], 3 => [-1.224744871, 0, 1.224744871], 4 => [-1.650680123, -0.5246476232, 0.5246476232, 1.650680123], 5=>[-2.02018287, -0.9585724646, 0, 0.9585724646, 2.02018287])
weights = Dict(2 => [0.8862269254, 0.8862269254], 3 => [0.2954089751, 0, 0.2954089751], 4 => [0.08131283544, 0.8049140900, 0.8049140900, 0.08131283544], 5=>[0.01995324204, 0.3936193231, 0.9453087204, 0.3936193231, 0.01995324204])

#Simulation parameters
T=11000000 #simulation periods
brns=1000000 #burn-in period


#################################################################
######################### PREPARATION ###########################
#################################################################
#Simulate T values for the long run growth
Random.seed!(1234)
x_shocks=rand(Normal(), T) #draw standard Normally distributed shocks
σ_shocks=rand(Normal(), T) 

#Stochastic volatility
σ_range=Array{Float64}(undef,T,1)
σ_range[1]= max(σ^2+σ_ω*σ_shocks[1], 10^(-8)) #set σ[0]=σ
for i in 2:T
    σ_range[i]=max(σ^2+ν*(σ_range[i-1]-σ^2)+σ_ω*σ_shocks[i], 10^(-8))
end


#Long-Run Growth
x_range=Array{Float64}(undef,T,1)
x_range[1]= ϕ_e*sqrt(σ_range[1])*x_shocks[1] #set x[0]=0
for i in 2:T
    x_range[i]=ρ*x_range[i-1]+ϕ_e*sqrt(σ_range[i])*x_shocks[i] 
end

σ_range=σ_range[brns+1:T] #discard burn-in draws
x_range=x_range[brns+1:T] #discard burn-in draws

#Define the boundaries of the grids
σ_max=maximum(σ_range)
σ_min=minimum(σ_range)
x_max=maximum(x_range)
x_min=minimum(x_range)

#Convert the grids to the [-1, 1] space
σ_range_Ch=[X_to_Ch(σ_min, σ_max, y) for y in σ_range]
x_range_Ch=[X_to_Ch(x_min, x_max, x) for x in x_range]

#compute Chebyshev zeros (i.e. collocation nodes)
#These are the points at which the residual function must be equal to zero in the collocation method 
zero_nodes_x=[Ch_zero(i,n_x+1) for i in 1:(n_x+1)]
zero_nodes_σ=[Ch_zero(i,n_x+1) for i in 1:(n_σ+1)]

#and transform them to the X space
zero_nodes_x_X=[Ch_to_X(x_min, x_max, k) for k in zero_nodes_x]
zero_nodes_σ_X=[Ch_to_X(σ_min, σ_max, k) for k in zero_nodes_σ]

#Now create all the nodes pairs in one column vector
#The outcomes are all the nodes pairs defined in the [-1, 1] interval
#Each pair is made of 2 elements: the first is the node value for σ and the second is the node value for x
zero_nodes_pairs=Ch_pairs(zero_nodes_σ, zero_nodes_x, n_σ, n_x, product)
zero_nodes_pairs_X=Ch_pairs(zero_nodes_σ_X, zero_nodes_x_X, n_σ, n_x, product)


#################################################################
####################### POLICY FUNCTIONS ########################
#################################################################

#####################################################################################
############################ Wealth to consumption ratio ############################
#####################################################################################
print("Projection - Now working on the W/C ratio\n")
#Compute the log-linear parameters
A0, A1, A2, z_pc, k1, k0=log_pc(γ, ψ, β, μ_c, ρ, σ, ν, θ, σ_ω, ϕ_e)
#Now compute the policy function for the pc ratio at the collocation nodes
pc_log=log_pf(sort(zero_nodes_σ_X), sort(zero_nodes_x_X), A0, A1, A2)

#This function defines all the equations that should be equal to zero in the collocation method
function f!(F, α)
    #compute the part of the equation that does not include parameters
    known_part=[β^θ*exp((1-γ)*(μ_c+node_pair[2])+0.5*(1-γ)^2*node_pair[1]) for node_pair in zero_nodes_pairs_X]

    #compute the part that includes time t wealth-to-consumption
    z_t=[(Ch_pol_2(α, node_pair[1], node_pair[2], n_σ, n_x, product)-1)^(-θ) for node_pair in zero_nodes_pairs]

    #compute the expectations part that includes time t+1 wealth-to-consumption
    #We use Guass-Hermite quadrature with product rule for multidimensional integral (product rule)
    z_t1=[GH_WC_2(α, n, n_σ, n_x, σ_min, σ_max, x_min, x_max, σ_ω, σ^2+ν*(node_pair[1]-σ^2),sqrt(node_pair[1])*ϕ_e, ρ*node_pair[2], nodes, weights, θ, product) for node_pair in zero_nodes_pairs_X]
    
    for i in 1:lastindex(z_t)
        F[i] = known_part[i]*z_t[i]*z_t1[i]-1
    end
end
#Find the solution
α=zeros(lastindex(zero_nodes_pairs))
α[1]=exp(A0)
α[2]=A1

sol=nlsolve(f!, α, method = :newton)
α_sol=sol.zero


#Now build the policy function at the collocaiton nodes
wc_pf=Array{Float64}(undef,n_σ+1, n_x+1)
for i in 1:(n_σ+1), j in 1:(n_x+1)
    wc_pf[i,j]=Ch_pol_2(α_sol, sort(zero_nodes_σ)[i], sort(zero_nodes_x)[j], n_σ, n_x, product)
end

#plot
surface(1:(n_σ+1), 1:(n_x+1), log.(wc_pf)', xaxis="σ^2_t", yaxis="x_t")
savefig("Proj_WC.png")
surface(1:(n_σ+1), 1:(n_x+1), pc_log', xaxis="σ^2_t", yaxis="x_t")


#####################################################################################
############################## Price to dividend ratio ##############################
#####################################################################################
print("Projection - Now working on the PD ratio")
#Compute the log-linear parameters
A0m, A1m, A2m, z_pd=log_pd(γ,ψ,β, μ_c, μ_d, ρ, σ, ν, θ, σ_ω, ϕ_e, ϕ_d, ϕ, A1, A2, k1, k0)
#Now compute the policy function for the pc ratio at the collocation nodes
pd_log=log_pf(sort(zero_nodes_σ_X), sort(zero_nodes_x_X), A0m, A1m, A2m)

#This function should define all the equations that should be equal to zero in the collocation method
function g!(G, ζ)
    #compute the part of the equation that does not include parameters
    known_part=[β^θ*exp(-γ*μ_c+μ_d+node_pair[2]*(ϕ-γ)+0.5*(γ^2+ϕ_d^2)*node_pair[1]) for node_pair in zero_nodes_pairs_X].*
    [(Ch_pol_2(α_sol, node_pair[1], node_pair[2], n_σ, n_x, product)-1)^(1-θ) for node_pair in zero_nodes_pairs]

    #compute the part that includes time t wealth-to-consumption
    z_t=[(Ch_pol_2(ζ, node_pair[1], node_pair[2], n_σ, n_x, product))^(-1) for node_pair in zero_nodes_pairs]

    #compute the expectations part that includes time t+1 wealth-to-consumption
    #We use Guass-Hermite quadrature
    z_t1=[GH_PD_2(α_sol, ζ, n, n_σ, n_x, σ_min, σ_max, x_min, x_max, σ_ω, σ^2+ν*(node_pair[1]-σ^2),sqrt(node_pair[1])*ϕ_e, ρ*node_pair[2], nodes, weights, θ, product) for node_pair in zero_nodes_pairs_X]

    for i in 1:lastindex(z_t)
        G[i] = known_part[i]*z_t[i]*z_t1[i]-1
    end
end

#Find the solution
ζ=zeros(lastindex(zero_nodes_pairs))
ζ[1]=exp(A0m)
ζ[2]=A1m
ζ[5]=-20

sol=nlsolve(g!, ζ, method = :newton)
ζ_sol=sol.zero

#Now build the policy function at the collocaiton nodes
pd_pf=Array{Float64}(undef,n_σ+1, n_x+1)
for i in 1:(n_σ+1), j in 1:(n_x+1)
    pd_pf[i,j]=Ch_pol_2(ζ_sol, sort(zero_nodes_σ)[i], sort(zero_nodes_x)[j], n_σ, n_x, product)
end

#plot
surface(1:(n_σ+1), 1:(n_x+1), log.(pd_pf)', xaxis="σ^2_t", yaxis="x_t")
savefig("Proj_PD.png")
surface(1:(n_σ+1), 1:(n_x+1), pd_log', xaxis="σ^2_t", yaxis="x_t")

#needed for simulation code
α=α_sol
ζ=ζ_sol

