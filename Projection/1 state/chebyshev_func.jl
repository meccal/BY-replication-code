#This file contains the functions to apply Chebyshev polynomials function approximation when the state space is one-dimensional

#This functions finds the zeros of the Chebyshev polynomials
#n-1 is the degree of approximation of the Chebyshev polynomials
function Ch_zero(j::Int64=1,n::Int64=2)
    if j in 1:n+1
        ch_zero=cos(π*((2*j-1)/(2*n)))
        return ch_zero
    else
        #if j is not an integer between 0 and n, there is an error and the function does not run
        return error("The first input should be an integer between 1 and "*string(n+1))
    end
end


#This function transforms arguments defined over the Chebyshev domain [-1, 1] into outputs belonging to the X domain, defined by a min and a max
#x_min is the lower bound of the space of the variable
#x_max is the upper bound of the space of the variable
#csi is the Chebyshev input in the [-1, 1] domain
function Ch_to_X(x_min::Number, x_max::Number, k::Number)
    if x_min>=x_max
    #if x is outside its boundaries, return an error
        return error("The upper bound should be greater than the lower bound")
    #if k does not belong to the [-1, 1] interval, there is an error and the function does not run
    elseif k>1
        k=1
        x=x_min+(0.5*(x_max-x_min))*(1+k)
        return x

    elseif k<-1
        k=-1
        x=x_min+(0.5*(x_max-x_min))*(1+k)
        return x
        

    else
        x=x_min+(0.5*(x_max-x_min))*(1+k)
        return x
    end
end


#This function transforms arguments defined over the X space (deinfed by boundearies x_min and x_max) into an argument of the Chebyshev domain [-1, 1]
function X_to_Ch(x_min::Number, x_max::Number, x::Number)
    #if x is outside its boundaries, return an error
    if x_min>=x_max
        return error("The upper bound should be greater than the lower bound")
    elseif x>x_max
        x=x_max
        k=(2*x-(x_min+x_max))/(x_max-x_min)
        return k

    elseif x<x_min
        x=x_min
        k=(2*x-(x_min+x_max))/(x_max-x_min)
        return k
    else
        k=(2*x-(x_min+x_max))/(x_max-x_min)
        return k
    end
end


#This function computes the function approximation using Chebyshev polynomials at a given point x, for a given value of the parameters
#and a specified degree of approximation
#This works for 1-state variable polynomials
#coeff: vector of coefficients of the Chebyshev polynomials
#x: point at which the function is evaluated (must be between -1 and 1)
#n:degree of approximation
function Ch_pol(coeff::Array{Float64}, x::Number, n::Int64=2)
    #Number of coefficients should be equal to the degree of approximation +1 (intercept)
    if length(coeff)!=(n+1)
        return error("The number of coefficients should be the degree of approximation +1")
    elseif x>1 || x<-1
        return error("x should be between -1 and 1")
    else
        #create a vector that contains Chebyshev polynomials
        ch_pol=Array{Float64}(undef,n+1,1)
        #compute the first two Chebychev polynomials from
        ch_pol[1]=1
        ch_pol[2]=x
        #use the recursive property of Chebyshev polynomials to compute the remainders
        for i in 3:n+1
            ch_pol[i]=2*x*ch_pol[i-1]-ch_pol[i-2]
        end

        #compute the linear combination of Chebyshev polynomials with the input coefficients
        approx_value=sum(ch_pol.*coeff)

        return approx_value

    end
end


#2 state variables version of the above function
#coeff: vector of coefficients of the Chebyshev polynomials
#x: point at which the function is evaluated (must be between -1 and 1) for the first state variable
#y: point at which the function is evaluated (must be between -1 and 1) for the second state variable
#n_x:degree of approximation for the first state variable
#n_y:degree of approximation for the second state variable
#product allows to choose two options: choose "tensor" to have the n-fold tensor product as basis functions
#choose "complete" to have the complete set of polynomials of degree p.

function Ch_pol_2(coeff::Array{Float64}, y::Number, x::Number, n_y::Int64=2, n_x::Int64=2, product::String="tensor")
    #Number of coefficients should be equal to the degree of approximation +1 (intercept)
    if length(coeff)!=(n_y+1)*(n_x+1) && product =="tensor"
        return error("When product=tensor, the number of coefficients should be " * string((n_x+1)*(n_y+1)))
    elseif product =="complete" && n_x != n_y
        return error("When product=complete, n_x should be equal to n_y")
    elseif length(coeff)!=(n_x+1)*(n_x+2)/2 && product =="complete"
        return error("When product=complete, the number of coefficients should be " * string((n_x+1)*(n_x+2)/2))
    elseif y>1 || y<-1
        return error("y should be between -1 and 1")
    elseif x>1 || x<-1
        return error("x should be between -1 and 1")
    elseif product != "tensor" && product != "complete"
        return error("product should be equal to either tensor or complete")
    else
        #Create the Chebyshev polynomials for the two state variables
        #create a vector that contains Chebyshev polynomials
        ch_pol_y=Array{Float64}(undef,(n_y+1),1)
        ch_pol_x=Array{Float64}(undef,(n_x+1),1)
        #compute the first two Chebyshev polynomials from
        ch_pol_y[1]=1
        ch_pol_y[2]=y
        ch_pol_x[1]=1
        ch_pol_x[2]=x
        #use the recursvie property of Chebyshev polynomials to compute the remainders
        for i in 3:n_x+1
            ch_pol_y[i]=2*y*ch_pol_y[i-1]-ch_pol_y[i-2]
            ch_pol_x[i]=2*x*ch_pol_x[i-1]-ch_pol_x[i-2]
        end

        if product == "tensor"
            #compute all possible combinations (tensor product)
            ch_pol_pairs=Array{Float64}(undef,(n_y+1)*(n_x+1),1)
            for i in 1:(n_y+1), j in 1:(n_x+1)
                ch_pol_pairs[(n_y+1)*(i-1)+j]=ch_pol_y[i]*ch_pol_x[j]
            end
            #compute the linear combination of Chebyshev polynomials with the input coefficients
            approx_value=sum(ch_pol_pairs.*coeff)


        else # if product == "complete"
            p=n_x #order of approximation
            #compute the complete set of polynomials of degree p in 2 variables
            ch_pol_pairs=zeros(0)
            for i in 0:p, j in 0:p
                if i+j<=p #only up to degree pair
                    append!(ch_pol_pairs,ch_pol_y[i+1]*ch_pol_x[j+1])
                else
                end
            end
            approx_value=sum(ch_pol_pairs.*coeff)
        end

        return approx_value

    end
end


#This function returns the Chebyshev basis at a given point x for a specified degree of approximation
#This works for 1-state variable polynomials
#x: point at which the function is evaluated (must be between -1 and 1)
#n:degree of approximation
function Ch_basis(x::Number, n::Int64=2)
    if x>1 || x<-1
        return error("x should be between -1 and 1")
    else
        #create a vector that contains Chebyshev polynomials
        ch_basis=Array{Float64}(undef,n+1,1)
        #compute the first two Chebychev polynomials from
        ch_basis[1]=1
        ch_basis[2]=x
        #use the recursive property of Chebyshev polynomials to compute the remainders
        for i in 3:n+1
            ch_basis[i]=2*x*ch_basis[i-1]-ch_basis[i-2]
        end
        return ch_basis
    end
end


#This function returns the Chebyshev basis at a given point (x,y) for a specified degree of approximation
#This works for 2-state variables polynomials
#n:degree of approximation
#product allows to choose two options: choose "tensor" to have the n-fold tensor product as basis functions
#choose "complete" to have the complete set of polynomials of degree p.
function Ch_basis_2(y::Number, x::Number, n_y::Int64=2, n_x::Int64=2, product::String="tensor")
    if x>1 || x<-1 || y>1 || y<-1
        return error("x and y should be between -1 and 1")
    elseif product =="complete" && n_x != n_y
        return error("When product=complete, n_x should be equal to n_y")    
    else
        #create a vector that contains Chebyshev polynomials
        ch_basis_y=Array{Float64}(undef,n_y+1,1)
        ch_basis_x=Array{Float64}(undef,n_x+1,1)
        #compute the first two Chebychev polynomials from
        ch_basis_y[1]=1
        ch_basis_y[2]=y
        ch_basis_x[1]=1
        ch_basis_x[2]=x

        #use the recursive property of Chebyshev polynomials to compute the remainders
        for i in 3:n_y+1
            ch_basis_y[i]=2*y*ch_basis_y[i-1]-ch_basis_y[i-2]
        end

        for i in 3:n_x+1
            ch_basis_x[i]=2*x*ch_basis_x[i-1]-ch_basis_x[i-2]
        end

        if product == "tensor" #compute all the possible combinations of the n-fold tensor product
            ch_basis_x_y=Array{Float64}(undef,(n_y+1)*(n_x+1),1)
            for i in 1:(n_y+1), j in 1:(n_x+1)
                ch_basis_x_y[(n_y+1)*(i-1)+j]=ch_basis_y[i]*ch_basis_x[j]
            end
        else #compute the complete set of polynomials of degree p
            p=n_x
            ch_basis_x_y=zeros(0)
            for i in 0:p, j in 0:p
                if i+j<=p #only up to degree p
                    append!(ch_basis_x_y,ch_basis_y[i+1]*ch_basis_x[j+1])
                else
                end
            end
        end
        return ch_basis_x_y
    end
end


#This function takes as inputs the Chebyshev nodes for two state variables and combines them into pairs
#product allows to choose two options: choose "tensor" to have the n-fold tensor product as basis functions
#choose "complete" to have the complete set of polynomials of degree p.
function Ch_pairs(zero_nodes_y::Vector{Float64}, zero_nodes_x::Vector{Float64}, n_y::Int64=2, n_x::Int64=2, product::String="tensor")
    if length(zero_nodes_y) != n_y+1
        return error("The legnth of zero_nodes_y should be" * string(n_y+1))
    elseif length(zero_nodes_x) != n_x+1
        return error("The legnth of zero_nodes_x should be" * string(n_x+1))
    elseif product =="complete" && n_x != n_y
        return error("When product=complete, n_x should be equal to n_y")

    else
        if product == "tensor" #compute all the possible combinations of the n-fold tensor product
            zero_nodes_pairs=Array{Any}(undef,(1+n_x)*(1+n_y))
            for i in 1:(n_y+1), j in 1:(n_x+1)
                zero_nodes_pairs[(n_y+1)*(i-1)+j]=[zero_nodes_y[i], zero_nodes_x[j]]
            end

        else #compute the complete set of polynomials of degree p
            p=n_x
            zero_nodes_pairs_temp=Array{Float64}(undef, 0, 2)
            for i in 0:p, j in 0:p
                if i+j<=p #only up to degree p
                    zero_nodes_pairs_temp=[zero_nodes_pairs_temp;[zero_nodes_y[i+1], zero_nodes_x[j+1]]']
                else
                end
            end
            zero_nodes_pairs=Array{Any}(undef, size(zero_nodes_pairs_temp)[1])
            for i in 1:lastindex(zero_nodes_pairs)
                zero_nodes_pairs[i]=[zero_nodes_pairs_temp[i,1], zero_nodes_pairs_temp[i,2]]
            end
        end
        return zero_nodes_pairs
    end
end