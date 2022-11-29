#lag simply lags the time series by a k number of lags
function lag(t_series::Union{Vector{Float64},Matrix{Float64}}, k::Int64)
    t_series=t_series[:,:]
    T,K=size(t_series)
    t_series_lagged=[[NaN for i=1:k, j=1:K]; t_series[1:T-k,:]]
    return t_series_lagged
end

#stats takes as input an array of Numbers and returns its mean, standard deviation and, evenutally, first order autocorrelation
#set AC=0 if you do not want the function to return the autocorrelation
#factor: choose 12 if you want to annualize by multiply the mean by 12 and std by sqrt(12)
function stats(t_series::Union{Vector{Float64},Matrix{Float64}}, AC=false, factor::Int64=1)
    if AC==false
        avg=mean(t_series)*factor
        sdev=std(t_series)*sqrt(factor)
        return avg, sdev
    else
        avg=mean(t_series)*factor
        sdev=std(t_series)*sqrt(factor)
        acf=autocor(t_series, collect(LinRange{Int}(1,AC,AC)))
        return avg, sdev, acf
    end
end

#perc computes the pth percentiles of the input series (for each element of p)
function perc(series::Vector{Float64}, p)
    percentiles=[percentile(series, i) for i in p]
    return percentiles
end


