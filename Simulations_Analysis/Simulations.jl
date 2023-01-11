"""
LUCA MECCA
lmecca@london.edu
Simulate Bansal, Yaron (2004)
2 state variables version
November 2022
"""

include("Generic.jl")

using CSV

#Folder where the .jl fles replicating the model are stored
path="..."

################################################################
########################## PARAMETERS ##########################
################################################################
#Select the method
#possible inputs are "VFI", "Projection"
method="VFI"

#Choose the simulation type
#choose "population" to simulate 1 path of 1M years
#choose "historical" to simulate 10000 paths of 70 years
simulation_type="population"

if simulation_type == "population"
    yrs=1000000 #number of years for the simulation
    yrs_brn=100000 #number of burn-in years
    K=1 #number of simulation
elseif simulation_type == "historical"
    yrs=70 #number of years for the simulation
    yrs_brn=10 #number of burn-in years
    K=10000 #number of simulation
else
    return(error("simulation_type should be equal to either historical or population"))
end

#choose how to aggregate monthly returns into annual
aggregation="sum" #choose "sum" to sum log monthly returns into annual
                       #choose "multiply" to multiply log returns by 12

#choose whether to have returns in logs or in levels
form="log" #choose "log" to display log returns results
              #choose "levels" to level returns results

#choose whether to obtain only the median of the distribution or more percentiles
p=[50]

##################################################################
########################### SIMULATION ###########################
##################################################################
#First run the one of the models
include(path * "/" * method * "/2 states" * "/" * "BY_" * method *".jl")

#Create containers for stastics of interest
Δc_mean=Array{Float64}(undef,K,1)
Δc_std=Array{Float64}(undef,K,1)
Δc_AC=Matrix{Float64}(undef,K,5)

Δd_mean=Array{Float64}(undef,K,1)
Δd_std=Array{Float64}(undef,K,1)
Δd_AC1=Matrix{Float64}(undef,K,1)

Δc_Δd_corr=Array{Float64}(undef,K,1)

rm_mean_num=Array{Float64}(undef,K,1)
rm_std_num=Array{Float64}(undef,K,1)
pd_mean_num=Array{Float64}(undef,K,1)
pd_std_num=Array{Float64}(undef,K,1)
pd_AC1_num=Matrix{Float64}(undef,K,1)
rm_mean_log=Array{Float64}(undef,K,1)
rm_std_log=Array{Float64}(undef,K,1)
pd_mean_log=Array{Float64}(undef,K,1)
pd_std_log=Array{Float64}(undef,K,1)
pd_AC1_log=Matrix{Float64}(undef,K,1)
rp_mean_num=Array{Float64}(undef,K,1)
rp_std_num=Array{Float64}(undef,K,1)
rp_mean_log=Array{Float64}(undef,K,1)
rp_std_log=Array{Float64}(undef,K,1)

rf_mean_num=Array{Float64}(undef,K,1)
rf_std_num=Array{Float64}(undef,K,1)
rf_mean_log=Array{Float64}(undef,K,1)
rf_std_log=Array{Float64}(undef,K,1)

for k in 1:K
    if mod(k,100)==0
        print("Simulation number " * string(k) *".\n")
    end

    Random.seed!(1233+k)
    x_shocks=rand(Normal(), (yrs+yrs_brn)*12+1) #draw standard Normally distributed shocks
    σ_shocks=rand(Normal(), (yrs+yrs_brn)*12+1)
    c_shocks=rand(Normal(), (yrs+yrs_brn)*12+1)
    d_shocks=rand(Normal(), (yrs+yrs_brn)*12+1)

    #simulate stochastic variables
    σ_range=Array{Float64}(undef,(yrs+yrs_brn)*12+1,1)
    x_range=Array{Float64}(undef,(yrs+yrs_brn)*12+1,1)
    Δc=Array{Float64}(undef,(yrs+yrs_brn)*12+1,1)
    Δd=Array{Float64}(undef,(yrs+yrs_brn)*12+1,1)

    σ_range[1]= max(σ^2+σ_ω*σ_shocks[1], 10^(-8)) #set σ[0]=σ
    x_range[1]= ϕ_e*sqrt(σ_range[1])*x_shocks[1] #set x[0]=0
    Δc[1]=μ_c+σ*c_shocks[1]
    Δd[1]=μ_d+ϕ_d*σ*d_shocks[1]

    for i in 2:(yrs+yrs_brn)*12+1
        σ_range[i]=max(σ^2+ν*(σ_range[i-1]-σ^2)+σ_ω*σ_shocks[i], 10^(-8))
        x_range[i]=ρ*x_range[i-1]+ϕ_e*sqrt(σ_range[i])*x_shocks[i] 
        Δc[i]=μ_c + x_range[i-1] + sqrt(σ_range[i-1])*c_shocks[i]
        Δd[i]=μ_d + ϕ*x_range[i-1] + ϕ_d * sqrt(σ_range[i-1])*d_shocks[i]
    end
     
    Δc=Δc[2:lastindex(Δc)] #drop NA observation
    Δd=Δd[2:lastindex(Δd)]

    x_range=x_range[(yrs_brn)*12+1: lastindex(x_range)] #drop burn-in observations
    σ_range=σ_range[(yrs_brn)*12+1: lastindex(σ_range)]

    #Convert the grids to the [-1, 1] space
    σ_range_Ch=[X_to_Ch(σ_min, σ_max, y) for y in σ_range]
    x_range_Ch=[X_to_Ch(x_min, x_max, x) for x in x_range]

    #PD ratio in levels
    pd_num=[Ch_pol_2(ζ,σ_range_Ch[i], x_range_Ch[i], n_σ, n_x, product) for i in 1:lastindex(x_range)]
    pd_log=exp.([A0m+A1m*x_range[i]+ A2m*σ_range[i] for i in 1:lastindex(x_range)])

    #Final consumption and dividend growth in levels
    ΔD_year=Array{Float64}(undef,yrs,1)
    ΔC_year=Array{Float64}(undef,yrs,1)

    #Final PD ratio in levels compute as pd ratio in the last month of the year, 
    #multiplied by dividends in the last month of the year, 
    #divided by the sum of the dividends of the corresponding year
    pd_num_year=Array{Float64}(undef,yrs,1)
    pd_log_year=Array{Float64}(undef,yrs,1)
    
    #In levels
    for i in 1:yrs
        Annual_ΔD=exp(sum(Δd[yrs_brn*12+(i-1)*12+1:yrs_brn*12+(i-1)*12+12])) #Annual D_{t+1}/D_t
        Annual_ΔC=exp(sum(Δc[yrs_brn*12+(i-1)*12+1:yrs_brn*12+(i-1)*12+12])) #Annual C_{t+1}/C_t

        Cumulative_ΔD=sum(exp.(cumsum((Δd[yrs_brn*12+(i-1)*12+1:yrs_brn*12+(i-1)*12+12]))))
        Cumulative_ΔC=sum(exp.(cumsum((Δc[yrs_brn*12+(i-1)*12+1:yrs_brn*12+(i-1)*12+12]))))

        ΔD_year[i]=Annual_ΔD-1
        ΔC_year[i]=Annual_ΔC-1

        pd_num_year[i]=pd_num[1+(i-1)*12+12]*Annual_ΔD/Cumulative_ΔD
        pd_log_year[i]=pd_log[1+(i-1)*12+12]*Annual_ΔD/Cumulative_ΔD

    end


    #return of the market
    Rm_num=(log.(exp.(Δd[(yrs_brn*12):lastindex(Δd)]).*(1 .+ pd_num)./lag(pd_num,1)))[2:lastindex(pd_num)]
    Rm_log=(log.(exp.(Δd[(yrs_brn*12):lastindex(Δd)]).*(1 .+ pd_log)./lag(pd_log,1)))[2:lastindex(pd_log)]

    #risk free rate
    Rf_num=log.([(β^θ*exp(-γ*(μ_c+x_range[i])+0.5*γ^2*σ_range[i])*(Ch_pol_2(α, σ_range_Ch[i], x_range_Ch[i], n_σ, n_x, product)-1)^(1-θ)*
    GH_WC_2(α, n, n_σ, n_x, σ_min, σ_max, x_min, x_max, σ_ω, σ^2+ν*(σ_range[i]-σ^2),sqrt(σ_range[i])*ϕ_e, ρ*x_range[i], nodes, weights, θ-1, product))^(-1) for i in 2:lastindex(x_range)])
    Rf_log=[-θ*log(β)-(θ-1-θ/ψ)*μ_c-(θ-1)*(k0+(k1-1)*A0+k1*A2*σ^2*(1-ν))-0.5*(k1*A2*σ_ω*(θ-1))^2+x_range[i]/ψ-σ_range[i]*((θ-1)*A2*(k1*ν-1)+0.5*(θ-1-θ/ψ)^2+0.5*(θ-1)^2*(k1*A1*ϕ_e)^2) for  i in 2:lastindex(x_range)]

    #annualize returns by summing log returns
    Rm_num_year=Array{Float64}(undef,(yrs,1))
    Rm_log_year=Array{Float64}(undef,(yrs,1))
    Rf_num_year=Array{Float64}(undef,(yrs,1))
    Rf_log_year=Array{Float64}(undef,(yrs,1))

    for i in 1:yrs
        Rm_num_year[i]=sum(Rm_num[(i-1)*12+1:(i-1)*12+12])
        Rm_log_year[i]=sum(Rm_log[(i-1)*12+1:(i-1)*12+12])
        Rf_num_year[i]=sum(Rf_num[(i-1)*12+1:(i-1)*12+12])
        Rf_log_year[i]=sum(Rf_log[(i-1)*12+1:(i-1)*12+12])
    end

    if form == "levels"
        Rm_num_year=exp.(Rm_num_year).-1
        Rm_log_year=exp.(Rm_log_year).-1
        Rf_num_year=exp.(Rf_num_year).-1
        Rf_log_year=exp.(Rf_log_year).-1
        Rm_num=exp.(Rm_num).-1
        Rm_log=exp.(Rm_log).-1
        Rf_num=exp.(Rf_num).-1
        Rf_log=exp.(Rf_log).-1
    else
        pd_num_year=log.(pd_num_year)
        pd_log_year=log.(pd_log_year)
    end


    #if aggregation == "sum", monthly log returns are summed into yearly log returns
    #if aggregation == "multiply", the mean and std of monthly log returns are multiplied by 12 and sqrt(12) respectively
    if aggregation == "sum"
        rm_mean_num[k], rm_std_num[k]=stats(Rm_num_year)
        rm_mean_log[k], rm_std_log[k]=stats(Rm_log_year)
        rf_mean_num[k], rf_std_num[k]=stats(Rf_num_year)
        rf_mean_log[k], rf_std_log[k]=stats(Rf_log_year)
    else
        rm_mean_num[k], rm_std_num[k]=stats(Rm_num, false, 12)
        rm_mean_log[k], rm_std_log[k]=stats(Rm_log, false, 12)
        rf_mean_num[k], rf_std_num[k]=stats(Rf_num, false, 12)
        rf_mean_log[k], rf_std_log[k]=stats(Rf_log, false, 12)
    end

    #compute risk-premium
    RP_num_year=Rm_num_year.-Rf_num_year
    RP_log_year=Rm_log_year.-Rf_log_year

    #compute Statistics
    Δc_mean[k],Δc_std[k], Δc_AC[k,:]=stats(ΔC_year, 5)
    Δd_mean[k],Δd_std[k], Δd_AC1[k,:]=stats(ΔD_year, 1)
    Δc_Δd_corr[k]=cor(ΔC_year, ΔD_year)[1,1]
    
    pd_mean_num[k], pd_std_num[k], pd_AC1_num[k,:]= stats(pd_num_year,1)
    pd_mean_log[k], pd_std_log[k], pd_AC1_log[k,:]= stats(pd_log_year,1)

    rp_mean_num[k], rp_std_num[k]=stats(RP_num_year)
    rp_mean_log[k], rp_std_log[k]=stats(RP_log_year)

end

#compute percentiles
Δc_mean_perc=perc(Δc_mean[:,1], p)
Δc_std_perc=perc(Δc_std[:,1],p)
Δc_AC1_perc=perc(Δc_AC[:,1], p)
Δc_AC2_perc=perc(Δc_AC[:,2], p)
Δc_AC3_perc=perc(Δc_AC[:,3],p)
Δc_AC4_perc=perc(Δc_AC[:,4], p)
Δc_AC5_perc=perc(Δc_AC[:,5],p)
Δd_mean_perc=perc(Δd_mean[:,1], p)
Δd_std_perc=perc(Δd_std[:,1], p)
Δd_AC1_perc=perc(Δd_AC1[:,1],p)

pd_mean_num_perc=perc(pd_mean_num[:,1], p)
pd_std_num_perc=perc(pd_std_num[:,1], p)
pd_AC1_num_perc=perc(pd_AC1_num[:,1], p)
pd_mean_log_perc=perc(pd_mean_log[:,1], p)
pd_std_log_perc=perc(pd_std_log[:,1], p)
pd_AC1_log_perc=perc(pd_AC1_log[:,1], p)

rm_mean_num_perc=perc(rm_mean_num[:,1], p)
rm_std_num_perc=perc(rm_std_num[:,1],p)
rm_mean_log_perc=perc(rm_mean_log[:,1], p)'
rm_std_log_perc=perc(rm_std_log[:,1], p)

rp_mean_num_perc=perc(rp_mean_num[:,1], p)
rp_std_num_perc=perc(rp_std_num[:,1], p)
rp_mean_log_perc=perc(rp_mean_log[:,1], p)
rp_std_log_perc=perc(rp_std_log[:,1], p)

rf_mean_num_perc=perc(rf_mean_num[:,1], p)
rf_mean_log_perc=perc(rf_mean_log[:,1],p)
rf_std_num_perc=perc(rf_std_num[:,1], p)
rf_std_log_perc=perc(rf_std_log[:,1], p)

if simulation_type=="population"

    #create and save DataFrames with results
    #consumption and dividend growth
    row1=(C_growth_mean=Δc_mean_perc[1,1], C_growth_std=Δc_std_perc[1,1], C_growth_AC1=Δc_AC1_perc[1,1], C_growth_AC2=Δc_AC2_perc[1,1], D_growth_mean=Δd_mean_perc[1,1], D_growth_std=Δd_std_perc[1,1], D_growth_AC1=Δd_AC1_perc[1,1])
    C_D_df=DataFrame([row1])
    CSV.write("C_D.csv",C_D_df) #save in a CSV file

    #financial moments
    row1=(Rm_mean=rm_mean_num[1,1], Rm_std=rm_std_num[1,1], Rf_mean=rf_mean_num[1,1], Rf_std=rf_std_num[1,1], 
    Rp_mean=rp_mean_num[1,1], Rp_std=rp_std_num[1,1], PD_mean=pd_mean_num[1,1], PD_std=pd_std_num[1,1], PD_AC1=pd_AC1_num[1,1])
    row2=(Rm_mean=rm_mean_log[1,1], Rm_std=rm_std_log[1,1], Rf_mean=rf_mean_log[1,1], Rf_std=rf_std_log[1,1], 
    Rp_mean=rp_mean_log[1,1], Rp_std=rp_std_log[1,1], PD_mean=pd_mean_log[1,1], PD_std=pd_std_log[1,1], PD_AC1=pd_AC1_log[1,1])
    returns_df=DataFrame([row1, row2])
    CSV.write("returns_" * method * ".csv",returns_df) #save in a CSV file
else
end
