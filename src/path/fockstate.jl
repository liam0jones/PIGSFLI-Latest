struct FockState <: AbstractVectorData 
    data::Vector{Int64} 
end 
FockState(length::Int64) = FockState(zeros(Int64,length))

"""Create a random fockstate for N bosons on M=L^D sites."""
function random_boson_config(M::Int64, N::Int64, rng::AbstractRNG, restart::Bool)::FockState
    state = FockState(M) 
    for _ in 1:N
        # if restart, put all bosons on the first site (first site not zeroth?)
        # else, put N bosons randomly on the lattice
        src = restart ? 1 : rand(rng, 0:M-1) 
        state[src+begin] += 1
    end 
    return state
end 
 