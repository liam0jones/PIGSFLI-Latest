@enum SimPhase pre_equilibrate_phase monte_carlo_phase
 
mutable struct MCStepCounters
    #1
    insert_worm_accepts::Int64
    insert_worm_attempts::Int64
    #2
    delete_worm_accepts::Int64
    delete_worm_attempts::Int64
    #3
    insert_anti_accepts::Int64
    insert_anti_attempts::Int64
    #4
    delete_anti_accepts::Int64
    delete_anti_attempts::Int64
    #5
    insertZero_worm_accepts::Int64
    insertZero_worm_attempts::Int64
    #6
    deleteZero_worm_accepts::Int64
    deleteZero_worm_attempts::Int64
    #7
    insertZero_anti_accepts::Int64
    insertZero_anti_attempts::Int64
    #8
    deleteZero_anti_accepts::Int64
    deleteZero_anti_attempts::Int64
    #9
    insertBeta_worm_accepts::Int64
    insertBeta_worm_attempts::Int64
    #10
    deleteBeta_worm_accepts::Int64
    deleteBeta_worm_attempts::Int64
    #11
    insertBeta_anti_accepts::Int64
    insertBeta_anti_attempts::Int64 
    #12
    deleteBeta_anti_accepts::Int64
    deleteBeta_anti_attempts::Int64
    #13
    advance_head_accepts::Int64
    advance_head_attempts::Int64
    #14
    recede_head_accepts::Int64
    recede_head_attempts::Int64
    #15
    advance_tail_accepts::Int64
    advance_tail_attempts::Int64
    #16
    recede_tail_accepts::Int64
    recede_tail_attempts::Int64
    #17
    ikbh_accepts::Int64
    ikbh_attempts::Int64
    #18
    dkbh_accepts::Int64
    dkbh_attempts::Int64
    #19
    ikah_accepts::Int64
    ikah_attempts::Int64
    #20
    dkah_accepts::Int64
    dkah_attempts::Int64
    #21
    ikbt_accepts::Int64
    ikbt_attempts::Int64
    #22
    dkbt_accepts::Int64
    dkbt_attempts::Int64
    #23
    ikat_accepts::Int64
    ikat_attempts::Int64
    #24
    dkat_accepts::Int64
    dkat_attempts::Int64
    #25
    insert_swap_kink_accepts::Int64
    insert_swap_kink_attempts::Int64
    #26
    delete_swap_kink_accepts::Int64
    delete_swap_kink_attempts::Int64
    #27
    swap_advance_head_accepts::Int64
    swap_advance_head_attempts::Int64
    #28
    swap_recede_head_accepts::Int64
    swap_recede_head_attempts::Int64
    #29
    swap_advance_tail_accepts::Int64
    swap_advance_tail_attempts::Int64
    #30
    swap_recede_tail_accepts::Int64
    swap_recede_tail_attempts::Int64
end

function reset!(cntr::MCStepCounters)
    for name in fieldnames(MCStepCounters)
        setproperty!(cntr,name, 0)  
    end
end

mutable struct SimTracker
    sim_phase::SimPhase 
    Ns::Vector{Float64}
    N_data::Vector{Int64}
    N_zero::Vector{Int64}
    N_beta::Vector{Int64}
    measurement_centers::Vector{Float64}
    measurement_center::Float64
    measurement_plus_minus::Float64
    N_sum::Vector{Float64}
    Z_ctr::Vector{Int64}
    kinetic_energy::Float64 
    diagonal_energy::Float64
    tr_kinetic_energy::Vector{Vector{Float64}}
    tr_diagonal_energy::Vector{Vector{Float64}}
    n_A::Vector{Vector{Int64}}
    n_i::Vector{Vector{Int64}}
    SWAP_histogram::Vector{Int64}
    SWAPn_histograms::Vector{Vector{Int64}}
    Pn::Vector{Vector{Int64}}
    Pn_squared::Vector{Vector{Int64}}
    n_A_accum::Vector{Float64}
    n_A_squared_accum::Vector{Float64}

    density::Vector{Int64}
    density_squared::Vector{Int64}
    corr_mat::Matrix{Float64}

    corr_accum::Vector{Float64}
    sigma2_accum::Vector{Float64}
    step_counters::MCStepCounters
    # for debugging
    num_preq_steps::Int64
    num_preq_updates::Int64
end


function SimTracker(state::SimState; num_replicas::Int64=0)  
    measurement_centers = get_measurement_centers(state.beta)
    n_measurement_centers = length(measurement_centers)
    measurement_center = state.beta/2.0
    measurement_plus_minus = 0.10*state.beta
    N_sum = repeat([0], state.num_replicas)
    Z_ctr = repeat([0], state.num_replicas)
    kinetic_energy = 0.0
    diagonal_energy = 0.0
    tr_kinetic_energy = init_tr_kinetic_energy(state.num_replicas, n_measurement_centers)
    tr_diagonal_energy = init_tr_diagonal_energy(state.num_replicas, n_measurement_centers)
    n_A = [repeat([0], state.m_A) for _ in 1:state.num_replicas] 
    n_i = [repeat([0], state.L) for _ in 1:state.num_replicas] 
    SWAP_histogram = init_SWAP_histogram(state.m_A)
    SWAPn_histograms = init_SWAPn_histograms(state.m_A, state.N)
    Pn = init_Pn(state.m_A, state.N)
    Pn_squared = init_Pn_squared(state.m_A, state.N)
    n_A_accum = init_n_A_accum(state.m_A)
    n_A_squared_accum = init_n_A_squared_accum(state.m_A)

    density = zeros(Int64,state.M)
    density_squared = zeros(Int64,state.M)
    # One-body correlation matrix: C_ij = <b_i^dagger b_j> (MxM structure)
    corr_mat = zeros(Float64, state.M, state.M)

    corr_accum = init_corr_accum(state.L)
    sigma2_accum = init_sigma2_accum(state.m_A)
    num_preq_steps = 0
    num_preq_updates = 0
    step_counters = MCStepCounters(repeat([0],60)...)

    tracker = SimTracker(pre_equilibrate_phase, Vector{Int64}(), Vector{Int64}(), Vector{Int64}(), Vector{Int64}(), measurement_centers, measurement_center, measurement_plus_minus, N_sum, Z_ctr, kinetic_energy, diagonal_energy, tr_kinetic_energy,tr_diagonal_energy,n_A,n_i,SWAP_histogram,SWAPn_histograms,Pn,Pn_squared, n_A_accum, n_A_squared_accum, density , density_squared , corr_mat , corr_accum,sigma2_accum,step_counters,num_preq_steps,num_preq_updates)
    reset!(tracker, state; num_replicas=num_replicas)

    return tracker
end
 


function reset!(tracker::SimTracker, state::SimState; num_replicas::Int64=0)
    if num_replicas == 0
        num_replicas = state.num_replicas 
    end  
    tracker.Ns = repeat([state.N], num_replicas)  
    tracker.N_zero = repeat([state.N], num_replicas)  
    tracker.N_beta = repeat([state.N], num_replicas)  
    tracker.N_data = Vector{Int64}()
end


function init_tr_kinetic_energy(num_replicas::Int64, n_measurement_centers::Int64)
    return [repeat([0.0], n_measurement_centers) for _ in 1:num_replicas]
end

function init_tr_diagonal_energy(num_replicas::Int64, n_measurement_centers::Int64)
    return [repeat([0.0], n_measurement_centers) for _ in 1:num_replicas]
end

function init_SWAP_histogram(m_A::Int64)
    # TODO: Check why this is m_A+1 and not m_A?
    return repeat([0], m_A+1)
end

function init_SWAPn_histograms(m_A::Int64, N::Int64)::Vector{Vector{Int64}}
    return [repeat([0], N+1) for _ in 1:m_A]
end

function init_Pn(m_A::Int64, N::Int64)
    return [repeat([0], N+1) for _ in 1:m_A]
end

function init_Pn_squared(m_A::Int64, N::Int64)
    return [repeat([0], N+1) for _ in 1:m_A]
end

function init_n_A_accum(m_A::Int64)
    return repeat([0.0], m_A)
end

function init_n_A_squared_accum(m_A::Int64)
    return repeat([0.0], m_A)
end

function init_corr_accum(L::Int64)
    return repeat([0.0], floor(Int64,(L+1)/2+1))
end

function init_sigma2_accum(m_A::Int64)
    return repeat([0.0], m_A)
end
 
function print_mc_update_stats(sim_tracker::SimTracker)
    step_counters = sim_tracker.step_counters

    println("----------- Detailed Balance -----------")
    prnt = (name::String, n_accepts::Int64, n_attempts::Int64) -> println(@sprintf "%-20s: %d / %d = %.3f" name n_accepts n_attempts n_accepts/n_attempts)

    prnt("insert_worm", step_counters.insert_worm_accepts, step_counters.insert_worm_attempts)
    prnt("delete_worm", step_counters.delete_worm_accepts, step_counters.delete_worm_attempts)
    prnt("insert_anti", step_counters.insert_anti_accepts, step_counters.insert_anti_attempts)
    prnt("delete_anti", step_counters.delete_anti_accepts, step_counters.delete_anti_attempts)
    prnt("insertZero_worm", step_counters.insertZero_worm_accepts, step_counters.insertZero_worm_attempts)
    prnt("deleteZero_worm", step_counters.deleteZero_worm_accepts, step_counters.deleteZero_worm_attempts)
    prnt("insertZero_anti", step_counters.insertZero_anti_accepts, step_counters.insertZero_anti_attempts)
    prnt("deleteZero_anti", step_counters.deleteZero_anti_accepts, step_counters.deleteZero_anti_attempts)
    prnt("insertBeta_worm", step_counters.insertBeta_worm_accepts, step_counters.insertBeta_worm_attempts)
    prnt("deleteBeta_worm", step_counters.deleteBeta_worm_accepts, step_counters.deleteBeta_worm_attempts)
    prnt("insertBeta_anti", step_counters.insertBeta_anti_accepts, step_counters.insertBeta_anti_attempts)
    prnt("deleteBeta_anti", step_counters.deleteBeta_anti_accepts, step_counters.deleteBeta_anti_attempts)
    prnt("advance_head", step_counters.advance_head_accepts, step_counters.advance_head_attempts)
    prnt("recede_head", step_counters.recede_head_accepts, step_counters.recede_head_attempts)
    prnt("advance_tail", step_counters.advance_tail_accepts, step_counters.advance_tail_attempts)
    prnt("recede_tail", step_counters.recede_tail_accepts, step_counters.recede_tail_attempts)
    prnt("ikbh", step_counters.ikbh_accepts, step_counters.ikbh_attempts)
    prnt("dkbh", step_counters.dkbh_accepts, step_counters.dkbh_attempts)
    prnt("ikah", step_counters.ikah_accepts, step_counters.ikah_attempts)
    prnt("dkah", step_counters.dkah_accepts, step_counters.dkah_attempts)
    prnt("ikbt", step_counters.ikbt_accepts, step_counters.ikbt_attempts)
    prnt("dkbt", step_counters.dkbt_accepts, step_counters.dkbt_attempts)
    prnt("ikat", step_counters.ikat_accepts, step_counters.ikat_attempts)
    prnt("dkat", step_counters.dkat_accepts, step_counters.dkat_attempts)
    prnt("insert_swap_kink", step_counters.insert_swap_kink_accepts, step_counters.insert_swap_kink_attempts)
    prnt("delete_swap_kink", step_counters.delete_swap_kink_accepts, step_counters.delete_swap_kink_attempts)
    prnt("swap_advance_head", step_counters.swap_advance_head_accepts, step_counters.swap_advance_head_attempts)
    prnt("swap_recede_head", step_counters.swap_recede_head_accepts, step_counters.swap_recede_head_attempts)
    prnt("swap_advance_tail", step_counters.swap_advance_tail_accepts, step_counters.swap_advance_tail_attempts)
    prnt("swap_recede_tail", step_counters.swap_recede_tail_accepts, step_counters.swap_recede_tail_attempts)
end
