const DELTA_TAU = 1e-12
const MINDIFF = 1e-10

mutable struct Kink
    tau::Float64
    n::Int64
    src::Int64
    dest::Int64 
    prev::Int64 
    next::Int64
    src_replica::Int64
    dest_replica::Int64
    partner::Int64
end

# default the partner to index -1, head and tail have partner -1 instead of themselves
Kink(tau::Float64,n::Int64,src::Int64,dest::Int64,prev::Int64,next::Int64,src_replica::Int64,dest_replica::Int64) = Kink(tau,n,src,dest,prev,next,src_replica,dest_replica,-1)

function print(kink::Kink)
    println(@sprintf "%.17f %d %d %d %d %d %d %d" kink.tau kink.n kink.src kink.dest kink.prev kink.next kink.src_replica kink.dest_replica)
end

function Base.:(==)(kink1::Kink, kink2::Kink)
    return abs(kink1.tau - kink2.tau) < DELTA_TAU/2 && kink1.n == kink2.n && kink1.src == kink2.src && kink1.dest == kink2.dest && kink1.prev == kink2.prev && kink1.next == kink2.next && kink1.src_replica == kink2.src_replica && kink1.dest_replica == kink2.dest_replica
end 

# For serialization when saving the state
JLD2.writeas(::Type{Kink}) = Tuple{Float64,Vararg{Int64, 8}}
JLD2.wconvert(::Type{Tuple{Float64,Vararg{Int64, 8}}}, k::Kink) = (k.tau,k.n,k.src,k.dest,k.prev,k.next,k.src_replica,k.dest_replica,k.partner)
JLD2.rconvert(::Type{Kink}, t::Tuple{Float64,Vararg{Int64, 8}}) = Kink(t...)