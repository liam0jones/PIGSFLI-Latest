const global rng_lookup = Base.ImmutableDict( 
    "Xoshiro256pp"      =>  Xoshiro,
    "MersenneTwister"   =>  MersenneTwister
)

function init_rng(rng_type::String, seed::Int64)::AbstractRNG
    rng = rng_lookup[rng_type](seed)
    return rng  
end 

@inline function random_bool(rng::AbstractRNG)
    return rand(rng, Bool) 
end
 