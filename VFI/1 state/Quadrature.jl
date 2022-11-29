#This function computes the Gauss-Hermite quadrature approximation of a function of a Normally distributed random variable
#This function is specific to the application of replicating Bansal, Yaron (2004) with one state variable.
include("chebyshev_func.jl")
#n is the number of integration nodes
#n_x is the order of approximation of the Chebyshev polynomials
#x_min is the lower bound of the domain for x_t
#x_max is the upper bound of the domain for x_t

#Gauss-Hermite for wealth to consumption
function GH_WC(α::Any, n::Int64, n_x::Int64, x_min::Number, x_max::Number, σ_bar::Float64, μ::Float64, nodes::Any, weights::Any, θ::Float64)
    if n>5 || n<2
        return error("This functions does not support number of integration points greater than 5 or lower than 2")
    else
        #compute the approximate solution
        approx_sol=π^(-0.5)*sum([weights[n][i] * (Ch_pol(α, X_to_Ch(x_min, x_max, sqrt(2)*σ_bar*nodes[n][i]+μ), n_x))^θ for i in 1:n])

        return approx_sol
    end
end    


#Gauss-Hermite for PD ratio
function GH_PD(α_sol::Any, ζ::Any, n::Int64, n_x::Int64, x_min::Number, x_max::Number, σ_bar::Float64, μ::Float64, nodes::Any, weights::Any, θ::Float64)
    if n>5 || n<2
        return error("This function does not support number of integration points greater than 5 or lower than 2")
    else
        #compute the approximate solution
        approx_sol=π^(-0.5)*sum([weights[n][i]*(Ch_pol(α_sol, X_to_Ch(x_min, x_max, sqrt(2)*σ_bar*nodes[n][i]+μ), n_x))^(θ-1)*(1+Ch_pol(ζ, X_to_Ch(x_min, x_max, sqrt(2)*σ_bar*nodes[n][i]+μ), n_x)) for i in 1:n])

        return approx_sol
    end
end    


#Gauss-Hermite for wealth to consumption with 2 state variables
#product allows to choose two options: choose "tensor" to have the n-fold tensor product as basis functions
#choose "complete" to have the complete set of polynomials of degree p.
function GH_WC_2(α::Any, n::Int64, n_σ::Int64, n_x::Int64, σ_min::Number, σ_max::Number, x_min::Number, x_max::Number, σ_1::Float64, μ_1::Float64, σ_2::Float64, μ_2::Float64, nodes::Any, weights::Any, θ::Float64, product::String="tensor")
    if n>5 || n<2
        return error("This function does not support number of integration points greater than 5 or lower than 2")
    else
        #create the nodes and weights combinations
        weights_pairs=Array{Any}(undef,n^2)
        nodes_pairs=Array{Any}(undef,n^2)

        for i in 1:n, j in 1:n
            weights_pairs[n*(i-1)+j]=[weights[n][i], weights[n][j]]
            nodes_pairs[n*(i-1)+j]=[nodes[n][i], nodes[n][j]]
        end
        
        #compute the approximate solution
        approx_sol=π^(-1)*sum([weights_pairs[i][1]*weights_pairs[i][2]*
        (Ch_pol_2(α, X_to_Ch(σ_min, σ_max, sqrt(2)*σ_1*nodes_pairs[i][1]+μ_1), X_to_Ch(x_min, x_max, sqrt(2)*σ_2*nodes_pairs[i][2]+μ_2), n_σ, n_x, product))^θ for i in 1:n^2])
        
        return approx_sol
    end
end    


#Gauss-Hermite for price to dividend with 2 state variables
function GH_PD_2(α_sol::Any, ζ::Any, n::Int64, n_σ::Int64, n_x::Int64, σ_min::Number, σ_max::Number, x_min::Number, x_max::Number, σ_1::Float64, μ_1::Float64, σ_2::Float64, μ_2::Float64, nodes::Any, weights::Any, θ::Float64, product::String="tensor")
    if n>5 || n<2
        return error("This function does not support number of integration points greater than 5 or lower than 2")
    else
        #create the nodes and weights combinations
        weights_pairs=Array{Any}(undef,n^2)
        nodes_pairs=Array{Any}(undef,n^2)

        for i in 1:n, j in 1:n
            weights_pairs[n*(i-1)+j]=[weights[n][i], weights[n][j]]
            nodes_pairs[n*(i-1)+j]=[nodes[n][i], nodes[n][j]]
        end
        
        #compute the approximate solution
        approx_sol=π^(-1)*sum([weights_pairs[i][1]*weights_pairs[i][2]*
        (Ch_pol_2(α_sol, X_to_Ch(σ_min, σ_max, sqrt(2)*σ_1*nodes_pairs[i][1]+μ_1), X_to_Ch(x_min, x_max, sqrt(2)*σ_2*nodes_pairs[i][2]+μ_2), n_σ, n_x, product))^(θ-1) *
        (Ch_pol_2(ζ, X_to_Ch(σ_min, σ_max, sqrt(2)*σ_1*nodes_pairs[i][1]+μ_1), X_to_Ch(x_min, x_max, sqrt(2)*σ_2*nodes_pairs[i][2]+μ_2), n_σ, n_x, product)+1) for i in 1:n^2])
       
        return approx_sol
    end
end    