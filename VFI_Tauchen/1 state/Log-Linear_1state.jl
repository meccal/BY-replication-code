"""
LUCA MECCA
lmecca@london.edu
September 2022
"""

#Functions that we use to compute the log-linear policy functions with 1 state variable

###########################################################################################
#Functions that we use to compute the log-linear policy function
#Parameters of the log pc ratio
function A_0(k0::Float64, k1::Float64, β::Float64, ψ::Number, μ_c::Float64, σ::Float64, θ::Float64, A1::Float64)
    A0=(log(β)+μ_c*(1-1/ψ)+k0+0.5*θ*σ^2*((1-1/ψ)^2+(k1*A1*ϕ_e)^2))/(1-k1)
    return A0
end

function A_1(ψ::Number, k1::Float64, ρ::Float64)
    A1=(1-1/ψ)/(1-k1*ρ)
    return A1
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
function log_pc(γ::Number,ψ::Number,β::Float64, μ_c::Float64, ρ::Float64, σ::Float64, θ::Float64, ϕ_e::Float64)
    #first we need to find k1 by iteration
    #start with a guess for z_hat (the pc/pd ratio)
    global z_old=1
    #Then keep substituting in until you find a fixed poin in z_1
    difference=1
    while difference>1.0e-10
        k1=k_1(z_old)
        k0=k_0(k1, z_old)
        A1=A_1(ψ, k1, ρ)
        A0=A_0(k0, k1, β, ψ, μ_c, σ, θ, A1)
        #Now we can find the next guess for z (pc/pd ratio)
        global z_new=A0
        difference=abs(z_new-z_old)
        #if the difference is too large, keep iterating
        global z_old=z_new
    end
    #Once the system converged, compute A0, A1, A2
    k1=k_1(z_new)
    k0=k_0(k1, z_new)
    A1=A_1(ψ, k1, ρ)
    A0=A_0(k0, k1, β, ψ, μ_c, σ, θ, A1)

    return A0, A1, z_new, k1, k0
end


#parameters of the log pd ratio
function A_0m(k0m::Float64, k1m::Float64, k0::Float64, k1::Float64, β::Float64, ψ::Number, μ_d::Float64, μ_c::Float64, σ::Float64, θ::Float64, A0::Float64, A1m::Float64, ϕ_d::Float64, ϕ_e::Float64)
    A0m=1/(1-k1m)*((θ-1-θ/ψ)*μ_c+μ_d+θ*log(β)+(θ-1)*(A0*(k1-1)+k0)+k0m+0.5*σ^2*((θ-1-θ/ψ)^2+ϕ_e^2*((θ-1)*k1*A1+k1m*A1m)^2+ϕ_d^2))
    return A0m
end

function A_1m(ϕ::Number, ψ::Number, k1m::Float64, ρ::Float64)
    A1m=(ϕ-1/ψ)/(1-k1m*ρ)
    return A1m
end



#Now create a function that given the parameters of the BY model, finds the log linearizing parameters of the pd ratio
function log_pd(γ::Number,ψ::Number,β::Float64, μ_c::Float64, μ_d::Float64, ρ::Float64, σ::Float64, θ::Float64, ϕ_e::Float64, ϕ_d::Float64, ϕ::Number,  A1::Float64, A0::Float64, k1::Float64, k0::Float64)
    #first we need to find k1 by iteration
    #start with a guess for z_hat (the pc/pd ratio)
    global z_old=1
    #Then keep substituting in until you find a fixed poin in z_1
    difference=1
    while difference>1.0e-10
        k1m=k_1(z_old)
        k0m=k_0(k1m, z_old)
        A1m=A_1m(ϕ,ψ,k1m, ρ)
        A0m=A_0m(k0m, k1m, k0, k1, β, ψ, μ_d, μ_c, σ, θ, A0, A1m, ϕ_d, ϕ_e)
        #Now we can find the next guess for z (pc/pd ratio)
        global z_new=A0m
        difference=abs(z_new-z_old)
        #if the difference is too large, keep iterating
        global z_old=z_new
    end
    #Once the system converged, compute A0, A1, A2
    k1m=k_1(z_new)
    k0m=k_0(k1m, z_new)
    A1m=A_1m(ϕ,ψ,k1m, ρ)
    A0m=A_0m(k0m, k1m, k0, k1, β, ψ, μ_d, μ_c, σ, θ, A0, A1m, ϕ_d, ϕ_e)

    return A0m, A1m, z_new
end


#Finally write a function that, given a gird for one state variable and the log linearization parameters, 
#finds the policy function
function log_pf(x_grid::Matrix{Float64}, A0::Float64, A1::Float64)
    J=length(x_grid)
    #create matrix that contains the policy objective (usally pc/pd ratio)
    z=Matrix{Float64}(undef,J,1) 
    for j=1:J
        x_t=x_grid[j]
        z[j]=A0+A1*x_t
    end
    return z
end


#Function to compute the policy function of the risk free rate
function rf_pf(σ_grid::Matrix{Float64}, x_grid::Matrix{Float64}, β::Float64, θ::Float64, ψ::Number, μ_c::Float64, k0::Float64, k1::Float64, A0::Float64, A1::Float64, A2::Float64, σ_hat::Float64, ν::Float64, σ_ω::Float64, ϕ_e::Float64)
    M=length(σ_grid)
    J=length(x_grid[1,:])
    #create matrix that contains the risk-free rate
    Rf=Matrix{Float64}(undef,M,J) 
    for i in 1:M
        σ_temp=σ_grid[i]
        for j=1:J
            x_temp=x_grid[i,j]
            Rf[i,j]=-θ*log(β)-(θ-1-θ/ψ)*μ_c-(θ-1)*(k0+(k1-1)*A0+k1*A2*σ_hat^2*(1-ν))-0.5*(k1*A2*σ_ω*(θ-1))^2+x_temp/ψ-σ_temp*((θ-1)*A2*(k1*ν-1)+0.5*(θ-1-θ/ψ)^2+0.5*(θ-1)^2*(k1*A1*ϕ_e)^2)
        end
    end
    return Rf
end

