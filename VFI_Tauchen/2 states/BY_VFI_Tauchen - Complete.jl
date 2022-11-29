"""
LUCA MECCA
lmecca@london.edu
Replicate the results of Bansal, Yaron (2004) using value function iteration (VFI)
2 state variables version
September 2022
"""

using Distributions, Statistics, DataFrames, Plots, StatsBase, Interpolations
include("Discretize.jl")


#################################################################
########################## CALIBRATION ##########################
#################################################################
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

#discretization parameters
M= 30 #grid points for the volatility
J= 30 #grid points for long run growth



#################################################################
######################## DISCRETIZATION #########################
#################################################################
#Now we need to discretize the two state variables
#We have a grid of M points for σ^2_t and, for each of those, we have a grid of J elements for x_t
# We set up the grid for these AR(1) variables and the transition matrix using the Markov Chain Approximation
#See algorithm 9.2.1 of Heer and Mausner book for details

#Volatility grid and associated transition matrix
#see Discretize.jl for Tau (Markov Chain) function
σ_grid, T_matrix_σ=Tau(ν, σ_ω, M)

#add unconditional volatility
σ_grid=σ_grid.+σ^2
#Now correct for the fact that volatility cannot be negative
σ_grid=σ_grid.-minimum(σ_grid).+10^(-7)

#Long-term growth grid
#for each point of the volatility grid, create a grid for x_t
#and compute the associated transition matrix
x_grid=Matrix{Float64}(undef,M,J) 
T_matrix_x=Matrix{Float64}(undef,J*M,J)
for i in 1:M
    x_grid[i, :], T_matrix_x[(i-1)*J+1:(i-1)*J+J, :] = Tau(ρ, sqrt(σ_grid[i]) * ϕ_e, J)
end

#Now compute the transtion matrices that include the probabilities of contemporaneous changes in both σ_t and x_t
#for each (σ_t, x_t) pair we create a (MxJ) matrix that include the proability of moving to the pair (σ_{t+1}, x_{t+1})
T_matrix_x_σ=Matrix{Any}(undef,M,J)
for i in 1:M
    #select the transition probabilities for σ_t from state i to all other states
    prob_σ=T_matrix_σ[:,i]
    #select the corresponding transition matrix for x_t
    T_matrix_x_temp=T_matrix_x[(i-1)*J+1:(i-1)*J+J,:]
    for j=1:J
        #select the transition probabilities for x from state i to all other states
        prob_x=T_matrix_x_temp[:,j]
        #Now compute the combined probabilities of moving across states
        prob_σ_x=Matrix{Float64}(undef,M,J) 
        for k in 1:lastindex(prob_σ)
            for h in 1:lastindex(prob_x)
                prob_σ_x[k,h]=prob_σ[k]*prob_x[h]
            end
        end
        #store it in the Transition matrix
        T_matrix_x_σ[i,j]=prob_σ_x
    end
end

#################################################################
######################## VALUE FUNCTION #########################
#################################################################
#We are iterating on the Utility/Consumption ratio

#This is a function of the state variables x_t and σ_t
#Therefore, the value function is a MxJ matrix

#1. Constant part
constant=Matrix{Float64}(undef,M,J)
for i in 1:M
    σ_t=σ_grid[i]
    for j in 1:J
        x_t=x_grid[i,j]
        constant[i,j]=β*exp(1/θ*((1-γ)*(μ_c+x_t)+0.5*(1-γ)^2*σ_t))
    end
end


#2. Initialization
Vf=ones(M,J) #Initialize the value function at 1
difference=1 #define initial value for the difference
counter_VF=0 #Set a count to record the number of iterations before convergence
error_VF=ones(0) #create a list that stores the progression of the error

#3. Iteration
#The value function iteration continues until the difference between two consecutive
#value functions is less than 0.0001 or the number of iterations exceeds 10000.
@time begin
    while difference>0.0001 && counter_VF<10000
        global counter_VF+=1
        if mod(counter_VF,100)==0
            print("Iteration number " * string(counter_VF) * " for VF\n")
        end
        UC=Matrix{Float64}(undef,M,J)
        for i in 1:M
            for j=1:J
                UC[i,j]=sum(T_matrix_x_σ[i,j].*Vf.^(1-γ))^(1/θ)
            end
        end

        global Vf_1=((1-β).+UC.*constant).^(θ/(1-γ))
        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(Vf-Vf_1))
        append!(error_VF, difference)
        #If the difference is too large, set Vf=Vf_1 and restart:
        global Vf=Vf_1
    end
end #time recording

#surface plot
surface(1:M, 1:J, Vf', title="Utility/Consumption", xaxis="σ^2_t", yaxis="x_t")
savefig("Value_Function_VFI.png")

#################################################################
####################### POLICY FUNCTIONS ########################
#################################################################

#####################################################################################
############################ Wealth to consumption ratio ############################
#####################################################################################
#We used SDF expressed in terms of W/C ratio
#1. Constant part
constant=Matrix{Float64}(undef,M,J)
for i in 1:M
    σ_t=σ_grid[i]
    for j in 1:J
        x_t=x_grid[i,j]
        constant[i,j]=β^θ*exp((1-γ)*(μ_c+x_t)+0.5*(1-γ)^2*σ_t)
    end
end

#2.Initialization
wc_pf=ones(M,J) #Initialize at 1
difference=1 #define initial value for the difference
counter_WC=0 #Set a count to record the number of iterations before convergence
error_WC=ones(0) #create a list that stores the progression of the error

#3. Iteration
@time begin
    while difference>0.0001 && counter_WC<10000
        global counter_WC+=1
        if mod(counter_WC,100)==0
            print("Iteration number " * string(counter_WC) * " for WC\n")
        end
        WC1=Matrix{Float64}(undef,M,J)
        for i in 1:M
            for j=1:J
                WC1[i,j]=sum(T_matrix_x_σ[i,j].*(wc_pf).^θ)
            end
        end

        global wc_pf_1=(constant.*WC1).^(1/θ) .+ 1
        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(wc_pf_1-wc_pf))
        append!(error_WC, difference)
        #If the difference is too large, set PC=PC_1 and restart:
        global wc_pf=wc_pf_1
    end
end #time recording

#surface plot
surface(1:M, 1:J, log.(wc_pf'), title="VFI PF for log W/C ratio", xaxis="σ^2_t", yaxis="x_t")
#savefig("VFI_WC.png")


####################################################################################
############################ Price to consumption ratio ############################
####################################################################################

####################################################################################
#Approach 1: We use SDF expressed in terms of U/C ratio
####################################################################################
#1. Constant part
constant=Matrix{Float64}(undef,M,J)
for i in 1:M
    σ_t=σ_grid[i]
    for j in 1:J
        x_t=x_grid[i,j]
        constant[i,j]=β*exp((1-1/ψ)*(μ_c+x_t+0.5*(1-γ)*σ_t))/(sum(T_matrix_x_σ[i,j].*(Vf.^(1-γ))))^((1/ψ-γ)/(1-γ))
    end
end

#2.Initialization
pc_pf=ones(M,J) #Initialize at 1
difference=1 #define initial value for the difference
counter_PC_uc=0 #Set a count to record the number of iterations before convergence
error_PC_uc=ones(0) #create a list that stores the progression of the error

#3. Iteration
@time begin
    while difference>0.0001 && counter_PC_uc<10000
        global counter_PC_uc+=1
        if mod(counter_PC_uc,100)==0
            print("Iteration number " * string(counter_PC_uc) * " for PC with UC SDF\n")
        end
        UC_PC=Matrix{Float64}(undef,M,J)
        for i in 1:M
            for j=1:J
                UC_PC[i,j]=sum(T_matrix_x_σ[i,j].*(Vf.^(1/ψ-γ).*(1 .+ pc_pf)))
            end
        end

        global pc_pf_1=constant.*UC_PC
        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(pc_pf_1-pc_pf))
        append!(error_PC_uc, difference)
        #If the difference is too large, set PC=PC_1 and restart:
        global pc_pf=pc_pf_1
    end
end #time recording

#surface plot
surface(1:M, 1:J, log.(pc_pf'), title="VFI PF for log PC ratio with UC SDF", xaxis="σ^2_t", yaxis="x_t")
savefig("VFI_PC.png")

####################################################################################
#Approach 2: We use SDF expressed in terms of PC ratio
####################################################################################
#1. Constant part
constant=Matrix{Float64}(undef,M,J)
for i in 1:M
    σ_t=σ_grid[i]
    for j in 1:J
        x_t=x_grid[i,j]
        constant[i,j]=β^θ*exp((1-γ)*(μ_c+x_t)+0.5*(1-γ)^2*σ_t)
    end
end

#2.Initialization
pc_pf_pc=ones(M,J) #Initialize at 1
difference=1 #define initial value for the difference
counter_PC_pc=0 #Set a count to record the number of iterations before convergence
error_PC_pc=ones(0) #create a list that stores the progression of the error

#3. Iteration
@time begin
    while difference>0.0001 && counter_PC_pc<10000
        global counter_PC_pc+=1
        if mod(counter_PC_pc,100)==0
            print("Iteration number " * string(counter_PC_pc) * " for PC with PC SDF\n")
        end

        PC1=Matrix{Float64}(undef,M,J)
        for i in 1:M
            for j=1:J
                PC1[i,j]=sum(T_matrix_x_σ[i,j].*((1 .+ pc_pf_pc).^θ))
            end
        end

        global pc_pf_pc_1=(constant.*PC1).^(1/θ)
        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(pc_pf_pc_1-pc_pf_pc))
        append!(error_PC_pc, difference)
        #If the difference is too large, set PC=PC_1 and restart:
        global pc_pf_pc=pc_pf_pc_1
    end
end #time recording

#surface plot
surface(1:M, 1:J, log.(pc_pf_pc'), title="VFI PF for log PC ratio with PC SDF", xaxis="σ^2_t", yaxis="x_t")
#savefig("VFI_PC_PC.png")

#################################################################################
############################ Price to dividend ratio ############################
#################################################################################

####################################################################################
#Approach 1: We use SDF expressed in terms of U/C ratio
####################################################################################
#1. Constant part
constant=Matrix{Float64}(undef,M,J)
for i in 1:M
    σ_t=σ_grid[i]
    for j in 1:J
        x_t=x_grid[i,j]
        constant[i,j]=β*exp(-1/ψ*μ_c+μ_d+x_t*(ϕ-1/ψ)+0.5*σ_t*((γ-1)/ψ+γ+ϕ_d^2))/(sum(T_matrix_x_σ[i,j].*(Vf.^(1-γ))))^((1/ψ-γ)/(1-γ))
    end
end

#2. Initialization
pd_pf=ones(M,J) #Initialize at 1
difference=1 #define initial value for the difference
counter_PD_uc=0 #Set a count to record the number of iterations before convergence
error_PD_uc=ones(0) #create a list that stores the progression of the error

#3. Iteration
@time begin
    while difference>0.001 && counter_PD_uc<30000
        global counter_PD_uc+=1
        if mod(counter_PD_uc,100)==0
            print("Iteration number " * string(counter_PD_uc) * " for PD with UC SDF\n")
        end
        UC_PD=Matrix{Float64}(undef,M,J)
        for i in 1:M
            for j=1:J
                UC_PD[i,j]=sum(T_matrix_x_σ[i,j].*(Vf.^(1/ψ-γ).*(1 .+ pd_pf)))
            end
        end

        global pd_pf_1=constant.*UC_PD
        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(pd_pf_1-pd_pf))
        append!(error_PD_uc, difference)
        #If the difference is too large, set PC=PC_1 and restart:
        global pd_pf=pd_pf_1
    end
end #time recording

#surface plot
surface(1:M, 1:J, log.(pd_pf'), title="VFI PF for log PD ratio with UC SDF",  xaxis="σ^2_t", yaxis="x_t")
savefig("VFI_PD.png")

####################################################################################
#Approach 2: We use SDF expressed in terms of PC ratio
####################################################################################
#1. Constant part
constant=Matrix{Float64}(undef,M,J)
for i in 1:M
    σ_t=σ_grid[i]
    for j in 1:J
        x_t=x_grid[i,j]
        constant[i,j]=β^θ*exp(-γ*μ_c+μ_d+x_t*(ϕ-γ)+0.5*σ_t*(γ^2+ϕ_d^2))
    end
end

constant=constant.*pc_pf_pc.^(1-θ)

#2. Initialization
pd_pf_pc=ones(M,J) #Initialize at 1
difference=1 #define initial value for the difference
counter_PD_pc=0 #Set a count to record the number of iterations before convergence
error_PD_pc=ones(0) #create a list that stores the progression of the error

#3. Iteration
@time begin
    while difference>0.001 && counter_PD_pc<30000
        global counter_PD_pc+=1
        if mod(counter_PD_pc,100)==0
            print("Iteration number " * string(counter_PD_pc) * " for PD with PC SDF\n")
        end
        PC_PD=Matrix{Float64}(undef,M,J)
        for i in 1:M
            for j=1:J
                PC_PD[i,j]=sum(T_matrix_x_σ[i,j].*((1 .+ pc_pf_pc).^(θ-1).*(1 .+ pd_pf_pc)))
            end
        end

        global pd_pf_pc_1=constant.*PC_PD
        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(pd_pf_pc_1-pd_pf_pc))
        append!(error_PD_pc, difference)
        #If the difference is too large, set PC=PC_1 and restart:
        global pd_pf_pc=pd_pf_pc_1
    end
end #time recording

#surface plot
surface(1:M, 1:J, log.(pd_pf_pc'), title="VFI PF for log PD ratio with PC SDF",  xaxis="σ^2_t", yaxis="x_t")
#savefig("VFI_PD_PC.png")

####################################################################################
#Approach 3: We use SDF expressed in terms of W/C ratio
####################################################################################
#1. Constant part
constant=Matrix{Float64}(undef,M,J)
for i in 1:M
    σ_t=σ_grid[i]
    for j in 1:J
        x_t=x_grid[i,j]
        constant[i,j]=β^θ*exp(-γ*μ_c+μ_d+x_t*(ϕ-γ)+0.5*σ_t*(γ^2+ϕ_d^2))
    end
end

constant=constant.*(wc_pf .- 1).^(1-θ)

#2. Initialization
pd_pf_wc=ones(M,J) #Initialize at 1
difference=1 #define initial value for the difference
counter_PD_wc=0 #Set a count to record the number of iterations before convergence
error_PD_wc=ones(0) #create a list that stores the progression of the error

#3. Iteration
@time begin
    while difference>0.001 && counter_PD_wc<30000
        global counter_PD_wc+=1
        if mod(counter_PD_wc,100)==0
            print("Iteration number " * string(counter_PD_wc) * " for PD with W/C SDF\n")
        end
        WC_PD=Matrix{Float64}(undef,M,J)
        for i in 1:M
            for j=1:J
                WC_PD[i,j]=sum(T_matrix_x_σ[i,j].*(wc_pf.^(θ-1).*(1 .+ pd_pf_wc)))
            end
        end

        global pd_pf_wc_1=constant.*WC_PD
        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(pd_pf_wc_1-pd_pf_wc))
        append!(error_PD_wc, difference)
        #If the difference is too large, set PC=PC_1 and restart:
        global pd_pf_wc=pd_pf_wc_1
    end
end #time recording

#surface plot
surface(1:M, 1:J, log.(pd_pf_wc'), title="VFI PF for log PD ratio with W/C SDF",  xaxis="σ^2_t", yaxis="x_t")
#savefig("VFI_PD_WC.png")




"""
########################################################################
############################ Risk-free rate ############################
########################################################################
#We use SDF expressed in terms of U/C ratio
rf_pf=Matrix{Float64}(undef,M,J)
for i in 1:M
    σ_t=σ_grid[i]
    for j=1:J    
        x_t=x_grid[i,j]
        rf_pf[i,j]=1/β/sum(T_matrix_x_σ[i,j].*Vf.^(1/ψ-γ))*exp(1/ψ*(μ_c+x_t)-0.5*σ_t*(1/ψ*(γ-1)+γ))*(sum(T_matrix_x_σ[i,j].*Vf.^(1-γ)))^((1/ψ-γ)/(1-γ))
    end
end
surface(1:M, 1:J, rf_pf', title="VFI PF for Riskfree rate with UC SDF",  xaxis="σ^2_t", yaxis="x_t")

#Log-linear solution
Rf_log=rf_pf(σ_grid, x_grid, β, θ, ψ, μ_c, k0, k1, A0, A1, A2, σ, ν, σ_ω, ϕ_e)
#Plot using a surface plot
surface(1:M, 1:J, exp.(Rf_log)', title="Log-linear PF for Riskfree rate",  xaxis="σ^2_t", yaxis="x_t")
"""



print("Total number of iterations for value function: " * string(counter_VF) *"\n")
print("Total number of iterations for pc ratio with UC ratio: " * string(counter_PC_uc)*"\n")
print("Total number of iterations for pc ratio with PC ratio: " * string(counter_PC_pc)*"\n")
print("Total number of iterations for W/C ratio: " * string(counter_WC)*"\n")
print("Total number of iterations for pd ratio with UC ratio: " * string(counter_PD_uc)*"\n")
print("Total number of iterations for pd ratio with W/C ratio: " * string(counter_PD_wc)*"\n")
print("Total number of iterations for pd ratio with PD ratio: " * string(counter_PD_pc)*"\n")
print("All done!")