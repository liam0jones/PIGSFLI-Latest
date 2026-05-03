"""Truncated exponential sampling from "Reduction of Autocorrelation Times in Lattice Path Integral Quantum Monte Carlo via Direct Sampling of the Truncated Exponential Distribution", Emanuel Casiano-Diaza, Kipton Barrosc, Ying Wai Lib, Adrian Del Maestro, arXiv:2302.04240v1

In the paper 
    a = a + DELTA_TAU
    b = b - DELTA_TAU
    c = c
"""
@inline function sample_truncated_exponential_from_a(rng::AbstractRNG, a::Float64, b::Float64, c::Float64)::Tuple{Float64, Float64, Bool} 
    b -= DELTA_TAU
    a += DELTA_TAU
    if abs(c) < MINDIFF 
        Z = b - a
        tau = a + rand(rng)*Z
        return tau, Z, a < tau < b
    end

    Zc = Z_times_c_single(a, b, c)
    tau = a - log(1.0-Zc*rand(rng))  / c

    is_plausible = a < tau < b

    return tau, Zc/c, is_plausible
end

@inline function sample_truncated_exponential_from_b(rng::AbstractRNG, a::Float64, b::Float64, c::Float64)::Tuple{Float64, Float64, Bool} 
    b -= DELTA_TAU
    a += DELTA_TAU
    if abs(c) < MINDIFF 
        Z = b - a
        tau = a + rand(rng)*Z
        return tau, Z, a < tau < b
    end

    Zc = Z_times_c_single(a, b, c)
    tau = b + log(1.0-Zc*rand(rng))  / c

    is_plausible = a < tau < b

    return tau, Zc/c, is_plausible
end

@inline function sample_joint_truncated_exponential_from_a(rng::AbstractRNG, a::Float64, b::Float64, c::Float64)::Tuple{Float64, Float64, Float64, Bool}
    
    b -= DELTA_TAU
    a += DELTA_TAU

    if abs(c) < MINDIFF 
        Zj = (a-b)^2/2
        tau1 = b - (b-a)*sqrt(1-rand(rng))
        tau2 = a + rand(rng)*(b-tau1-DELTA_TAU) + DELTA_TAU
        return tau1, tau2, Zj, a < tau1 < tau2 < b
    end

    if c*(b-a) < -600 
        # This would result in an infinite u (or even NaN here) and hence invalid tau1.
        # However, the limit can be computed analytically:
        # y = rand()
        # u = exp(-c(b-a)) (y - 1)  - r + r(b-a) c + ca ≈ exp(-c(b-a)) (y - 1)
        # from the ((-c*b + u) < -600.0) case below, we found the limit 
        # tau1 = b + log(c*b - u)/c ≈ a + log(1-y)/c 
        tau1 = a + log(1-rand(rng))/c
        # In the same limit, we can evaluate sampling of the second tau:
        # tau2:  a - log(1-y*(1-exp(-c(b-a))))/c ≈ b - log(y)/c 
        tau2 = b - log(rand(rng))/c

        # Z will be Inf 
        return tau1, tau2, Inf, a < tau1 < tau2 < b
    end 

    y = rand(rng)
    Zjc2 = Z_joint_times_c2(a, b, c) 
    u = y * Zjc2 - exp(c*(a-b)) + c*a 
    
    if c >= 0 
        arg = max(-1/ℯ, -exp(-c*b + u))
        tau1 = 1/c * (u-lambertw0_fast( arg)) 
    else  
        if (-c*b + u < -600) # Too close to zero exponent on branch -1. Here LambertW would not converge. We use the limiting value W(x,-1) ≈ ln(-x) - ln(-ln(-x)) for x → 0.
            tau1 = b + log(c*b - u)/c
        else    
            arg = max(-1/ℯ, -exp(-c*b + u))
            tau1 = 1/c * (u-lambertwm1_fast( arg))
        end
    end 

    Zj = Zjc2/c^2
    is_plausible1 =  (a < tau1 < b)    

    if !is_plausible1 
        return -1.0, -1.0, -1.0, false
    end

    tau2, _, is_plausible2 = sample_truncated_exponential_from_a(rng, tau1, b+DELTA_TAU, c)
    return tau1, tau2, Zj, is_plausible2
end 

@inline @fastmath function Z_joint(a::Float64, b::Float64, c::Float64; delta_min::Float64=DELTA_TAU)
    b -= delta_min
    a += delta_min 
    if abs(c) < MINDIFF  
        return (a-b)^2/2 
    end
    return (exp(-c*(b-a))-1)/c^2 + (b-a)/c
end

@inline @fastmath function Z_joint_times_c2(a::Float64, b::Float64, c::Float64) 
    return (exp(-c*(b-a))-1) + (b-a)*c
end

@inline @fastmath function Z_single(a::Float64, b::Float64, c::Float64; delta_min::Float64=DELTA_TAU)
    b -= delta_min
    a += delta_min
    if abs(c) < MINDIFF  
        return b - a
    end
    return (1.0 - exp(-c*(b-a))) / c
end

@inline @fastmath function Z_times_c_single(a::Float64, b::Float64, c::Float64) 
    return (1.0 - exp(-c*(b-a)))  
end

@inline @fastmath function F(x, Zj, a, b, c) 
    return 1/(c^2*Zj)*( exp(-c*(b-a)) - exp(-c*(b-x)) - c*(a-x) )
end
