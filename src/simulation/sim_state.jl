mutable struct SimState  
    const paths::Vector{Path}
    num_kinks::Vector{Int64} 
    head_idx::Vector{Int64} 
    tail_idx::Vector{Int64} 
    const last_kinks::Vector{Vector{Int64}} 
    const initial_fock_state::FockState
    const fock_state_at_slice::FockState
    fock_state_at_half_plus::Vector{FockState}
    const adjacency_matrix::Adjacency_Matrix
    const sub_sites::Sites 
    const L::Int64
    const N::Int64
    const M::Int64
    const D::Int64
    const m_A::Int64
    const t::Float64
    const U::Float64
    const beta::Float64
    num_replicas::Int64
    canonical::Bool
    eta::Float64
    mu::Float64
    num_swaps::Int64
    const rng_type::String
    const seed::Int64
    const trial_state::TrialState
    const model::Type{T} where {T<:ModelSystem}
end

function SimState( 
    initial_fock_state::FockState,
    adjacency_matrix::Adjacency_Matrix,
    sub_sites::Sites,
    L::Int64,
    N::Int64,
    M::Int64,
    D::Int64,
    m_A::Int64,
    t::Float64,
    U::Float64,
    beta::Float64,
    num_replicas::Int64,
    canonical::Bool,
    mu::Float64,
    eta::Float64,
    rng_type::String,
    seed::Int64,
    trial_state::TrialState,
    model::Type{T} where {T<:ModelSystem}
) 
    paths = [create_paths(initial_fock_state, M, replica_idx) for replica_idx in 1:num_replicas]
    num_kinks = repeat([M], num_replicas)  
    head_idx = repeat([-1], num_replicas)
    tail_idx = repeat([-1], num_replicas)
    last_kinks =  [collect(0:M-1) for _ in 1:num_replicas]  
    fock_state_at_slice = FockState(repeat([0], M))
    fock_state_at_half_plus = [FockState(repeat([0], M)) for _ in 1:num_replicas]
    num_swaps = 0

    sim_state = SimState(
        #rng,
        paths,
        num_kinks,
        head_idx,
        tail_idx, 
        last_kinks, 
        initial_fock_state,
        fock_state_at_slice,
        fock_state_at_half_plus,
        adjacency_matrix,
        sub_sites,
        L,
        N,
        M,
        D,
        m_A,
        t,
        U,
        beta,
        num_replicas,
        canonical,
        eta,
        mu,
        num_swaps,
        rng_type,
        seed,
        trial_state,
        model
    )

    reset!(sim_state; num_replicas=num_replicas)
    return sim_state
end

function reset!(state::SimState; num_replicas::Int64=0)
    if num_replicas == 0
        num_replicas = state.num_replicas
    else 
         state.num_replicas = num_replicas 
    end  
    state.num_kinks = repeat([state.M], num_replicas)  
    state.head_idx = repeat([-1], num_replicas)
    state.tail_idx = repeat([-1], num_replicas)
    state.last_kinks .= [collect(0:state.M-1) for _ in 1:num_replicas] 
    state.paths .= [create_paths(state.initial_fock_state, state.M, replica_idx) for replica_idx in 1:num_replicas]

    state.fock_state_at_half_plus = [FockState(repeat([0], state.M)) for _ in 1:num_replicas]
end

function init_fock_state_at_slice(state::SimState)
    return FockState(repeat([0], state.M))
end
 

