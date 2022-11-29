"""
LUCA MECCA
lmecca@london.edu
Replicate the results of Bansal, Yaron (2004) using value function iteration (VFI)
1 state variable version
September 2022
"""

using Distributions, Statistics, DataFrames, Plots, StatsBase, Interpolations

include("Discretize.jl")
include("Log-Linear_1state.jl")

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

#discretization parameters
J=30 #grid points for long run growth



#################################################################
######################## DISCRETIZATION #########################
#################################################################
#Now we need to discretize the state variable, as well as log consumption growth
#We have a grid of J elements for x_t
# We set up the grid for this AR(1) variable and the transition matrix using the Markov Chain Approximation
#See algorithm 9.2.1 of Heer and Mausner book for details


#Long-term growth grid
#for each point of the volatility grid, create a grid for x_t
#and compute the associated transition matrix
x_grid, T_matrix_x = Tau(ρ, σ*ϕ_e, J)

#################################################################
######################## VALUE FUNCTION #########################
#################################################################
#We are iterating on the Utility/Consumption ratio

#This is a function of the state variable x_t
#Therefore, the value function is a Jx1 vector
#Initialize the value function at 1
Vf=ones(J,1)

#constant
constant=Matrix{Float64}(undef,J,1)
for j in 1:J
    x_t=x_grid[j]
    constant[j]=β*exp(1/θ*((1-γ)*(μ_c+x_t)+0.5*(1-γ)^2*σ^2))
end


#The value function iteration continues until the difference between two consecutive
#value functions is less than 0.0001 or the number of iterations exceeds 10000.
#define initial value for the difference
difference=1
#Set a count to record the number of iterations before convergence
counter_VF=0
#create a list that stores the progression of the error
error_VF=ones(0)

@time begin
    while difference>0.0001 && counter_VF<10000
        global counter_VF+=1
        if mod(counter_VF,100)==0
            print("Iteration number " * string(counter_VF) * " for VF\n")
        end
        UC=Matrix{Float64}(undef,J,1)
        for j=1:J
            UC[j]=sum(T_matrix_x[:,j].*Vf.^(1-γ))^(1/θ)
        end
        
        global Vf_1=((1-β).+constant.*UC).^(θ/(1-γ))
        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(Vf-Vf_1))
        append!(error_VF, difference)
        #If the difference is too large, set Vf=Vf_1 and restart:
        global Vf=Vf_1
    end
end #time recording

#plot
plot(1:J, Vf, title="Utility/Consumption")


#################################################################
####################### POLICY FUNCTIONS ########################
#################################################################

############################ Price to consumption ratio ############################
#In this case, we use SDF in terms of UC ratio
constant=Matrix{Float64}(undef,J,1)
for j in 1:J
    x_t=x_grid[j]
    constant[j]=β*exp((1-1/ψ)*(μ_c+x_t+0.5*(1-γ)*σ^2))/(sum(T_matrix_x[:,j].*(Vf.^(1-γ))))^((1/ψ-γ)/(1-γ))
end


#Initialize at 1
pc_pf=ones(J,1)

#define initial value for the difference
difference=1
#Set a count to record the number of iterations before convergence
counter_PC=0
#create a list that stores the progression of the error
error_PC=ones(0)

@time begin
    while difference>0.0001 && counter_PC<20000
        global counter_PC+=1
        if mod(counter_PC,100)==0
            print("Iteration number " * string(counter_PC) * " for PC with UC SDF\n")
        end
        UC_PC=Matrix{Float64}(undef,J,1)
        for j=1:J
            UC_PC[j]=sum(T_matrix_x[:,j].*(Vf.^(1/ψ-γ).*(1 .+ pc_pf)))
        end

        global pc_pf_1=constant.*UC_PC
        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(pc_pf_1-pc_pf))
        append!(error_PC, difference)
        #If the difference is too large, set PC=PC_1 and restart:
        global pc_pf=pc_pf_1
    end
end #time recording

#plot
plot(1:J, log.(pc_pf), title="VFI PC")

#Plot the log-linear policy function for comparison
#First compute the A0, A1 elements of the log linear solution
A0, A1, z_pc, k1, k0=log_pc(γ,ψ,β, μ_c, ρ, σ, θ, ϕ_e)
#Now compute the policy function for the pc ratio
pc_log=log_pf(x_grid, A0, A1)
#Plot
plot(1:J, pc_log, title="Log-linear PF for log PC ratio")





#################################################################################
############################ Price to dividend ratio ############################
#We use SDF in terms of UC ratio
constant=Matrix{Float64}(undef,J,1)
for j in 1:J
    x_t=x_grid[j]
    constant[j]=β*exp(-1/ψ*μ_c+μ_d+x_t*(ϕ-1/ψ)+0.5*σ^2*((γ-1)/ψ+γ+ϕ_d^2))/(sum(T_matrix_x[:,j].*(Vf.^(1-γ))))^((1/ψ-γ)/(1-γ))
end


#Initialize at 1
pd_pf=ones(J,1)

#define initial value for the difference
difference=1
#Set a count to record the number of iterations before convergence
counter_PD=0
#create a list that stores the progression of the error
error_PD=ones(0)

@time begin
    while difference>0.001 && counter_PD<20000
        global counter_PD+=1
        if mod(counter_PD,100)==0
            print("Iteration number " * string(counter_PD) * " for PD with UC SDF\n")
        end
        UC_PD=Matrix{Float64}(undef,J,1)
        for j=1:J
            UC_PD[j]=sum(T_matrix_x[:,j].*(Vf.^(1/ψ-γ).*(1 .+ pd_pf)))
        end

        global pd_pf_1=constant.*UC_PD
        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(pd_pf_1-pd_pf))
        append!(error_PD, difference)
        #If the difference is too large, set PC=PC_1 and restart:
        global pd_pf=pd_pf_1
    end
end #time recording

#plot
plot(1:J, log.(pd_pf), title="VFI PF for log PD ratio with UC SDF")


#Plot the log-linear policy function for camparison
#First compute the A0, A1, A2 elements of the log linear SOLUTION
A0m, A1m, z_pd=log_pd(γ,ψ,β, μ_c, μ_d, ρ, σ, θ, ϕ_e, ϕ_d, ϕ,  A1, A0, k1, k0)
#Now compute the policy function for the pc ratio
pd_log=log_pf(x_grid, A0m, A1m)
#Plot
plot(1:J, pd_log, title="Log-linear PF for log PD ratio")


#print("Total number of iterations for value function: " * string(counter_VF))
#print("Total number of iterations for pc ratio: " * string(counter_PC))
#print("Total number of iterations for pd ratio: " * string(counter_PD))

#print the pd_ratios
plot(log.(pd_pf),title="VFI and log-linear PD ratio with ρ="*string(ρ), label="VFI (LHS)", legend=:topright)
plot!(pd_log, label="Log-linear (LHS)") 
plot!(twinx(), log.(pd_pf)-pd_log, color=:green, label="Difference (RHS)", legend=:bottomright) 
#savefig("PD_1state_"*string(ρ)*".png")

########################################################################
############################ Risk-free rate ############################
Rf=Matrix{Float64}(undef,J,1)
for i in 1:J
    Rf[i]=(β^θ*exp(-γ*(μ_c + x_grid[i]) + 0.5*γ^2*σ^2)*(pc_pf[i]-1)^(1-θ)*sum(T_matrix_x[:,i].*pc_pf.^(θ-1)))^(-1)-1
end

