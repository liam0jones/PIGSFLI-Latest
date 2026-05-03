function is_configuration_valid(sim_state::SimState; silent::Bool=false)
    is_valid = true
    for r = 1:sim_state.num_replicas
        partner_missing = kink_partner_missing(sim_state, r; silent=true) 
        sorted_kinks = create_sorted_kink_vectors(sim_state, r)
        _, found_a_duplicate_link = find_duplicate_links(sim_state, r, sorted_kinks)
        _, found_a_broken_link = find_broken_links(sim_state, r, sorted_kinks)
        a_last_kink_not_in_last_kinks = find_last_kink_not_in_last_kinks(sim_state, r, sorted_kinks)
        a_tail_index_no_tail = find_tail_index_no_tail(sim_state, r)
        a_head_index_no_head = find_head_index_no_head(sim_state, r)

        if partner_missing || found_a_duplicate_link || found_a_broken_link || a_last_kink_not_in_last_kinks || a_tail_index_no_tail || a_head_index_no_head
            is_valid = false
        end

        if !silent && !is_valid
            println("Replica $r is invalid.")
            println("Partner missing: $partner_missing")
            println("Duplicate link: $found_a_duplicate_link")
            println("Broken link: $found_a_broken_link")
            println("Last kink not in last kinks: $a_last_kink_not_in_last_kinks")
            println("Tail index not a tail: $a_tail_index_no_tail")
            println("Head index not a head: $a_head_index_no_head") 
        end
    end
    return is_valid
end
 

function find_partner(kink::Kink, sim_state::SimState, r::Int64; silent=false)
    path = sim_state.paths[r][1:sim_state.num_kinks[r]]
    for k in path 
        if k.src == kink.dest && k.dest == kink.src && k.tau ≈ kink.tau
            return k
        end
    end
    if !silent
        println("No partner found for kink $(kink.src) -> $(kink.dest).")
    end
    return nothing
end

function kink_partner_missing(sim_state::SimState, r::Int64; silent=false) 
    for kink in sim_state.paths[r].data[1:sim_state.num_kinks[r]]
        if isnothing(find_partner(kink, sim_state, r;silent=silent)) 
            return true
        end
    end
    return false 
end

function find_last_kink_not_in_last_kinks(sim_state::SimState, r::Int64, sorted_kinks::Vector{Vector{Kink}})
    paths = sim_state.paths[r][1:sim_state.num_kinks[r]]
    a_last_kink_not_in_last_kinks = false
    for i = 0:sim_state.M-1
        if !(sorted_kinks[i+begin][end] == paths[sim_state.last_kinks[r][i+begin]+begin])
            a_last_kink_not_in_last_kinks = true
            break
        end
    end
    return a_last_kink_not_in_last_kinks 
end

function find_tail_index_no_tail(sim_state::SimState, r::Int64)
    if sim_state.tail_idx[r] == -1 
        return false
    end 
    # check that tail inside active kinks 
    if sim_state.tail_idx[r] >  sim_state.num_kinks[r] - 1 
        return true
    end
    # check that src == dest 
    tail_kink = sim_state.paths[r][sim_state.tail_idx[r]+begin] 
    if tail_kink.src != tail_kink.dest 
        return true
    end

    return false
end

function find_head_index_no_head(sim_state::SimState, r::Int64)
    if sim_state.head_idx[r] == -1 
        return false
    end
    # check that head inside active kinks 
    if sim_state.head_idx[r] >  sim_state.num_kinks[r] - 1 
        return true
    end
    # check that src == dest 
    head_kink = sim_state.paths[r][sim_state.head_idx[r]+begin] 
    if head_kink.src != head_kink.dest 
        return true
    end

    return false
end

function find_broken_links(sim_state::SimState, r::Int64, sorted_kinks::Vector{Vector{Kink}})
    broken_links = Vector{Tuple{Int64, Float64, Float64}}() # (src, tau_start, tau_end) 
    # check for broken links 
    found_a_broken_link = false
    for i = 1:sim_state.M 
        for (j, kink) in enumerate(sorted_kinks[i])
            if j<length(sorted_kinks[i]) && kink.next != -1 && (sim_state.paths[r][kink.next+begin].tau - sorted_kinks[i][j+1].tau) > DELTA_TAU/2 
                push!(broken_links, (i-1, kink.tau, sorted_kinks[i][j+1].tau))
                found_a_broken_link = true
            elseif j>1 && !(sim_state.paths[r][kink.prev+begin].tau ≈ sorted_kinks[i][j-1].tau)
                push!(broken_links, (i-1, sorted_kinks[i][j-1].tau, kink.tau))
                found_a_broken_link = true
            end 
        end 
    end
    return broken_links, found_a_broken_link
end

function find_duplicate_links(sim_state::SimState, r::Int64, sorted_kinks::Vector{Vector{Kink}})
    duplicate_links = Vector{Tuple{Int64, Float64}}() # (src, tau)
    # check for broken links 
    found_a_duplicate_link = false
    for i = 1:sim_state.M 
        for (j, kink) in enumerate(sorted_kinks[i])
            #if j<length(sorted_kinks[i])
            #println(i-1,":", j-1," --- ",sorted_kinks[i][j].tau - sorted_kinks[i][j+1].tau)
            #end
            if j<length(sorted_kinks[i]) && kink.next != -1 && (abs(sorted_kinks[i][j].tau - sorted_kinks[i][j+1].tau) < DELTA_TAU/2)
                push!(duplicate_links, (i-1, kink.tau))
                found_a_duplicate_link = true 
            end 
        end 
    end
    return duplicate_links, found_a_duplicate_link
end

function create_sorted_kink_vectors(sim_state::SimState, r::Int64)
    sorted_kinks = [Vector{Kink}() for _ in 1:sim_state.M]
    # create the sorted list
    for kink in sim_state.paths[r].data[1:sim_state.num_kinks[r]]  
        append!(sorted_kinks[kink.src+begin], [kink])  
    end
    for i = 1:sim_state.M 
        taus = Vector{Float64}()
        for kink in sorted_kinks[i]
            append!(taus,kink.tau) 
        end
        idx = sortperm(taus)
        sorted_kinks[i] = sorted_kinks[i][idx]
    end
    return sorted_kinks
end