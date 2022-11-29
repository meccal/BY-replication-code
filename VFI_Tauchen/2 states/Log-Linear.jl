"""
LUCA MECCA
lmecca@london.edu
September 2022
"""

#Functions that we use to compute the log-linear policy functions with 2 state variables

#Parameters of the log pc ratio
function A_0(k0::Float64, k1::Float64, β::Float64, ψ::Number, μ_c::Float64, A2::Float64, ν::Float64, σ::Float64, θ::Float64, σ_ω::Float64)
    A0=1/(1-k1)*(log(β)+k0+(1-1/ψ)*μ_c+k1*A2*(1-ν)*σ^2+θ/2*(k1*A2*σ_ω)^2)
    return A0
end

function A_1(ψ::Number, k1::Float64, ρ::Float64)
    A1=(1-1/ψ)/(1-k1*ρ)
    return A1
end

function A_2(γ::Number,ψ::Number, k1::Float64, ν::Float64, ϕ_e::Float64, ρ::Float64)
    A2=-(γ-1)*(1-1/ψ)/(2*(1-k1*ν))*(1+(k1*ϕ_e/(1-k1*ρ))^2)
    return A2
end

function k_1(z_hat::Number)
    k1=exp(z_hat)/(1+exp(z_hat))
    return k1
end

function k_0(k1, z_hat)
    k0=log(1+exp(z_hat))-k1*z_hat
    return k0
end

#Now create a function that given the parameters of the BY model, finds the log linearizing parameters of the pc ratio
function log_pc(γ::Number,ψ::Number,β::Float64, μ_c::Float64, ρ::Float64, σ::Float64, ν::Float64, θ::Float64, σ_ω::Float64, ϕ_e::Float64)
    #first we need to find k1 by iteration
    #start with a guess for z_hat (the pc/pd ratio)
    global z_old=1
    #Then keep substituting in until you find a fixed poin in z_1
    difference=1
    while difference>1.0e-10
        k1=k_1(z_old)
        k0=k_0(k1, z_old)
        A2=A_2(γ, ψ, k1, ν, ϕ_e, ρ)
        A0=A_0(k0, k1, β, ψ, μ_c, A2, ν, σ, θ, σ_ω)
        #Now we can find the next guess for z (pc/pd ratio)
        global z_new=A0+A2*σ^2
        difference=abs(z_new-z_old)
        #if the difference is too large, keep iterating
        global z_old=z_new
    end
    #Once the system converged, compute A0, A1, A2
    k1=k_1(z_new)
    k0=k_0(k1, z_new)
    A1=A_1(ψ, k1, ρ)
    A2=A_2(γ, ψ, k1, ν, ϕ_e, ρ)
    A0=A_0(k0, k1, β, ψ, μ_c, A2, ν, σ, θ, σ_ω)

    return A0, A1, A2, z_new, k1, k0
end


#parameters of the log pd ratio
function A_0m(k0m::Float64, k1m::Float64, k0::Float64, k1::Float64, β::Float64, ψ::Number, μ_d::Float64, μ_c::Float64, A2m::Float64, ν::Float64, σ::Float64, θ::Float64, σ_ω::Float64, A2::Float64)
    A0m=1/(1-k1m)*(k0m+μ_d+ θ*log(β)+(θ-1)*(k0+A0*(k1-1))+(k1m*A2m+(θ-1)*A2*k1)*σ^2*(1-ν)+μ_c*(-θ/ψ+(θ-1))+0.5*(k1m*A2m+(θ-1)*k1*A2)^2*σ_ω^2)
    return A0m
end

function A_1m(ϕ::Number, ψ::Number, k1m::Float64, ρ::Float64)
    A1m=(ϕ-1/ψ)/(1-k1m*ρ)
    return A1m
end

function A_2m(γ::Number, θ::Float64, k1::Float64, k1m::Float64, ν::Float64, ϕ_e::Float64,ϕ_d::Float64, A1::Float64, A1m::Float64, A2::Float64)
    A2m=((1-θ)*A2*(1-k1*ν)+0.5*(γ^2+((θ-1)*k1*A1*ϕ_e+k1m*A1m*ϕ_e)^2+ϕ_d^2))/(1-k1m*ν)
    return A2m
end

#Now create a function that given the parameters of the BY model, finds the log linearizing parameters of the pd ratio
function log_pd(γ::Number,ψ::Number,β::Float64, μ_c::Float64, μ_d::Float64, ρ::Float64, σ::Float64, ν::Float64, θ::Float64, σ_ω::Float64, ϕ_e::Float64, ϕ_d::Float64, ϕ::Number,  A1::Float64, A2::Float64, k1::Float64, k0::Float64)
    #first we need to find k1 by iteration
    #start with a guess for z_hat (the pc/pd ratio)
    global z_old=1
    #Then keep substituting in until you find a fixed poin in z_1
    difference=1
    while difference>1.0e-10
        k1m=k_1(z_old)
        k0m=k_0(k1m, z_old)
        A1m=A_1m(ϕ,ψ,k1m, ρ)
        A2m=A_2m(γ, θ, k1,k1m, ν, ϕ_e, ϕ_d, A1, A1m, A2)
        A0m=A_0m(k0m, k1m, k0, k1, β, ψ, μ_d, μ_c, A2m, ν, σ, θ, σ_ω, A2)
        #Now we can find the next guess for z (pc/pd ratio)
        global z_new=A0m+A2m*σ^2
        difference=abs(z_new-z_old)
        #if the difference is too large, keep iterating
        global z_old=z_new
    end
    #Once the system converged, compute A0, A1, A2
    k1m=k_1(z_new)
    k0m=k_0(k1m, z_new)
    A1m=A_1m(ϕ,ψ,k1m, ρ)
    A2m=A_2m(γ, θ, k1,k1m, ν, ϕ_e, ϕ_d, A1, A1m, A2)
    A0m=A_0m(k0m, k1m, k0, k1, β, ψ, μ_d, μ_c, A2m, ν, σ, θ, σ_ω, A2)

    return A0m, A1m, A2m, z_new
end


#Finally write a function that, given a gird for two states variables and the log linearization parameters, 
#finds the policy function
function log_pf(σ_grid::Union{Vector{Float64},Matrix{Float64}}, x_grid::Union{Vector{Float64},Matrix{Float64}}, A0::Float64, A1::Float64, A2::Float64)
    M=length(σ_grid)
    J=length(x_grid)
    #create matrix that contains the policy objective (usally pc/pd ratio)
    z=Matrix{Float64}(undef,M,J) 
    for i in 1:M
        σ_temp=σ_grid[i]
        for j=1:J
            x_temp=x_grid[j]
            z[i,j]=A0+A1*x_temp+A2*σ_temp
        end
    end
    return z
end


#Function to compute the policy function of the risk free Replicate
function rf_pf(σ_grid::Matrix{Float64}, x_grid::Matrix{Float64}, β::Float64, θ::Float64, ψ::Number, μ_c::Float64, k0::Float64, k1::Float64, A0::Float64, A1::Float64, A2::Float64, σ::Float64, ν::Float64, σ_ω::Float64, ϕ_e::Float64)
    M=length(σ_grid)
    J=length(x_grid[1,:])
    #create matrix that contains the risk-free rate
    Rf=Matrix{Float64}(undef,M,J) 
    for i in 1:M
        σ_temp=σ_grid[i]
        for j=1:J
            x_temp=x_grid[i,j]
            Rf[i,j]=-θ*log(β)-(θ-1-θ/ψ)*μ_c-(θ-1)*(k0+(k1-1)*A0+k1*A2*σ^2*(1-ν))-0.5*(k1*A2*σ_ω*(θ-1))^2+x_temp/ψ-σ_temp*((θ-1)*A2*(k1*ν-1)+0.5*(θ-1-θ/ψ)^2+0.5*(θ-1)^2*(k1*A1*ϕ_e)^2)
        end
    end
    return Rf
end



