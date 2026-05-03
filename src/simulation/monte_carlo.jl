using ProgressBars

const CHECK_CONFIG::Bool = false
const MAX_BIN_ITERS::Int64 = 500_000_000

function run_monte_carlo!(
    rng::AbstractRNG,
    sim_state::SimState,
    sim_options::SimOptions,
    sim_tracker::SimTracker
    ;
    output_fh::WriteOutputHandler,
    state_fh::SnapshotHandler,
    jld2_path::String=""
)
    sim_tracker.sim_phase = monte_carlo_phase
    num_replicas::Int64 = sim_state.num_replicas
    sweeps::Int64 = sim_options.sweeps
    sweeps_between_measurement::Int64 = sim_options.sweep * sim_options.measurement_frequency
    perform_swap_update::Bool = (num_replicas > 1)
    bins_wanted::Int64 = sim_options.bins_wanted
    bin_size::Int64 = sim_options.bin_size

    # Stage 2: Equilibration
    # When measure_corr_mat is active, interleave measurement with all equilibration
    # sweeps.  Whenever bin_size valid antiworm samples accumulate, flush the bin
    # directly to JLD2 and reset accumulators — turning stage 2 wall time into
    # real corr_mat bins.  Only Z_ctr and corr_mat are touched; energy/density are
    # intentionally skipped so stage 3's energy estimator starts from a clean slate.
    # Any partial accumulation at the end carries into stage 3's first bin.
    stage2_corr_ctr::Int64 = 0
    stage2_bin_count::Int64 = 0
    if !sim_options.restart
        println("Stage (2/3): Equilibrating...\n")
        if sim_options.measure_corr_mat && sweeps > 0
            C_ij_buf = similar(sim_tracker.corr_mat)
            sweeps_done::Int64 = 0
            jld2_fh = jld2_path != "" ? jldopen(jld2_path, "a+") : nothing
            try
                while sweeps_done < sweeps
                    n = min(sweeps_between_measurement, sweeps - sweeps_done)
                    run_sweeps!(sim_state, sim_tracker, rng, n, num_replicas, perform_swap_update)
                    sweeps_done += n
                    if stage2_bin_count < bins_wanted
                        if sim_state.head_idx[1] == -1 && sim_state.tail_idx[1] == -1
                            sim_tracker.Z_ctr[1] += 1
                        else
                            stage2_corr_ctr = measure_corr_mat!(sim_state, sim_tracker, 1, stage2_corr_ctr)
                        end
                        if stage2_corr_ctr >= bin_size
                            if sim_tracker.Z_ctr[1] > 0
                                stage2_bin_count += 1
                                if jld2_fh !== nothing
                                    z = sim_tracker.Z_ctr[1]
                                    @. C_ij_buf = sim_tracker.corr_mat / (sim_state.eta * z)
                                    C_ij_buf .= (C_ij_buf .+ C_ij_buf') ./ 2
                                    jld2_fh[@sprintf("C_%04d", stage2_bin_count)] = C_ij_buf
                                end
                            end
                            fill!(sim_tracker.corr_mat, 0.0)
                            fill!(sim_tracker.Z_ctr, 0)
                            stage2_corr_ctr = 0
                        end
                    end
                end
            finally
                jld2_fh !== nothing && close(jld2_fh)
            end
            corr_mat_bins_still_needed_after_stage2::Int64 = max(0, bins_wanted - stage2_bin_count)
            println("  Stage 2 wrote $(stage2_bin_count) full corr_mat bins.")
            if corr_mat_bins_still_needed_after_stage2 > 0
                println("  Still need $(corr_mat_bins_still_needed_after_stage2) bins in Stage 3.\n")
            else
                println("  All $(bins_wanted) correlation matrix bins collected in Stage 2.\n")
            end
        else
            run_sweeps!(sim_state, sim_tracker, rng, sweeps, num_replicas, perform_swap_update)
        end
    else
        println("Stage (2/3): RESTARTED SIMULATION: Equilibration not needed\n")
    end

    println("Stage (3/3): Main Monte Carlo loop...")
    # Energy/density always run for all bins_wanted iterations.
    # Corr_mat collection is capped at bins_wanted total across stage 2 and stage 3.
    corr_mat_bins_still_needed::Int64 = max(0, bins_wanted - stage2_bin_count)
    stage3_corr_bin_count::Int64 = 0
    # Discard any partial stage 2 accumulation so stage 3 always starts clean.
    if sim_options.measure_corr_mat
        fill!(sim_tracker.corr_mat, 0.0)
        fill!(sim_tracker.Z_ctr, 0)
    end
    for m_count = tqdm(1:bins_wanted)
        writing_ctr = [0, 0]
        bin_iter_ctr::Int64 = 0
        collect_corr_mat::Bool = sim_options.measure_corr_mat && (stage3_corr_bin_count < corr_mat_bins_still_needed)
        # Diagonal configurations (writing_ctr[1]) always clock the bin regardless of
        # whether corr_mat is active.  Antiworm samples (writing_ctr[2]) accumulate
        # opportunistically during the same sweeps and are flushed at bin end if
        # nonzero — their estimator is normalised by eta*Z_ctr, not by bin_size, so
        # any positive count yields a valid unbiased per-bin estimate.
        while writing_ctr[1] < bin_size
            run_sweeps!(sim_state, sim_tracker, rng, sweeps_between_measurement, num_replicas, perform_swap_update)
            if CHECK_CONFIG && !is_configuration_valid(sim_state)
                error("Invalid configuration found!")
            end
            if (!perform_swap_update)
                writing_ctr = conventional_measurement!(sim_tracker, sim_state, sim_options, writing_ctr, collect_corr_mat)
            else
                writing_ctr[1] = swap_measurement!(sim_tracker, sim_state, sim_options, writing_ctr[1])
            end
            bin_iter_ctr += 1
            if bin_iter_ctr >= MAX_BIN_ITERS
                @warn "Bin $m_count: inner loop reached MAX_BIN_ITERS=$MAX_BIN_ITERS without collecting $bin_size diagonal samples (got $(writing_ctr[1])). N_zero/N_beta may never match N=$(sim_state.N). Breaking to prevent infinite loop."
                break
            end
        end
        if (!perform_swap_update)
            corr_bin_for_write::Int64 = collect_corr_mat ? (stage2_bin_count + stage3_corr_bin_count + 1) : 0
            z_ctr_before_write::Int64 = collect_corr_mat ? sim_tracker.Z_ctr[1] : 0
            write_to_file_conventional!(sim_tracker, output_fh, sim_options, sim_state, jld2_path, m_count, writing_ctr[2]; corr_mat_bin_count=corr_bin_for_write)
            if collect_corr_mat && z_ctr_before_write > 0
                stage3_corr_bin_count += 1
            end
            reset_conventional_measurement!(sim_tracker, sim_options)
        else
            write_to_file_swap(sim_tracker, output_fh, sim_options, sim_state)
            reset_swap_measurement!(sim_tracker, sim_options)
        end
        if sim_options.save_state
            write_state_to_file(state_fh, sim_state, sim_tracker, rng, m_count)
        end
    end
    if sim_options.save_state
        write_state_to_file(state_fh, sim_state, sim_tracker, rng, -1)
    end
end

function run_sweeps!(sim_state::SimState, sim_tracker::SimTracker, rng::AbstractRNG, n_sweeps::Int64, num_replicas::Int64, perform_swap_update::Bool)
    for _ = 1:n_sweeps
        # ---- single replica update ----  
        for r=1:num_replicas
            # Run a single update on replica r 
            random_mc_update!(rng, sim_state, r, sim_tracker)
        end
        # ---- swap update ----
        if perform_swap_update
            random_swap_update!(rng, sim_state, 1, sim_tracker)
        end
    end
end
  
function write_to_file_conventional!(sim_tracker::SimTracker, output_fh::WriteOutputHandler, sim_options::SimOptions, sim_state::SimState, jld2_path::String="", bin_count::Int64=0, antiworm_count::Int64=0; corr_mat_bin_count::Int64=bin_count)
    # Round out N_tracker since it might have floating point errors after a while
    sim_tracker.Ns = round.(sim_tracker.Ns)
    # Conventional measurements
    jldopen(jld2_path != "" ? jld2_path : "/dev/null", "a+") do file
        file["K_$(bin_count)"] = sim_tracker.kinetic_energy ./ sim_options.bin_size
        file["V_$(bin_count)"] = sim_tracker.diagonal_energy ./ sim_options.bin_size
    end

    if sim_options.measure_tau_resolved_estimators
        write(output_fh, "tr_kinetic_energy", sim_tracker.tr_kinetic_energy[1] ./ sim_options.bin_size)
        write(output_fh, "tr_diagonal_energy", sim_tracker.tr_diagonal_energy[1] ./ sim_options.bin_size) 
    end 
    # write <n> and <n^2> to disk
    if sim_options.measure_n
        write(output_fh, "n", sim_tracker.n_A_accum ./ sim_options.bin_size)
        write(output_fh, "n_squared", sim_tracker.n_A_squared_accum ./ sim_options.bin_size)
    end

    # Write <n_i> and <n^2_i> to JLD2 instead of text files for efficiency
    if sim_options.measure_density
        jldopen(jld2_path != "" ? jld2_path : "/dev/null", "a+") do file
            file["n_$(bin_count)"] = sim_tracker.density ./ sim_options.bin_size
            file["n^2_$(bin_count)"] = sim_tracker.density_squared ./ sim_options.bin_size
        end
    end 

    # Write C_ij to JLD2 file (one dataset per bin for jackknife analysis).
    # corr_mat accumulates across bins to ensure every bin written has data.
    # This ensures consistent number of bins for all observables.
    if sim_options.measure_corr_mat && jld2_path != "" && corr_mat_bin_count > 0
        if sim_tracker.Z_ctr[1] > 0
            C_ij = sim_tracker.corr_mat ./ (sim_state.eta * sim_tracker.Z_ctr[1])
            C_ij .= (C_ij .+ C_ij') ./ 2
            jldopen(jld2_path, "a+") do file
                file["C_$(corr_mat_bin_count)"] = C_ij
            end
        end
        # Always reset accumulators after attempting to write, regardless of whether data was written
        # This prevents accumulation across bins when Z_ctr is 0 or when a bin is not written
        fill!(sim_tracker.corr_mat, 0.0)
        fill!(sim_tracker.Z_ctr, 0)
    end

    # write C and sigma2 to disk
    if sim_options.measure_corr
        write(output_fh, "corr", sim_tracker.corr_accum ./ sim_options.bin_size) 
        
        rho02::Float64 = (sim_state.N/sim_state.L)^2
        for i = 0:length(sim_tracker.sigma2_accum)-1
            sim_tracker.sigma2_accum[i+begin] = sim_tracker.sigma2_accum[i+begin] ./ sim_options.bin_size - rho02 * (i+1) * (i+1)
        end
        write(output_fh, "sigma2", sim_tracker.sigma2_accum)
    end
end
 
function write_to_file_swap(sim_tracker::SimTracker, output_fh::WriteOutputHandler, sim_options::SimOptions, sim_state::SimState)
    # Write SWAP histogram
    write(output_fh, "SWAP_histogram", sim_tracker.SWAP_histogram)  
                    
    if !sim_options.no_accessible
        for mA = 1:sim_state.m_A
            # Write Pn 
            write(output_fh, (@sprintf "Pn-mA%d" mA), sim_tracker.Pn[mA])
            # Write Pn_squared
            write(output_fh, (@sprintf "PnSquared-mA%d" mA), sim_tracker.Pn_squared[mA])
            # Write SWAPn_histograms
            write(output_fh, (@sprintf "SWAPn-mA%d" mA), sim_tracker.SWAPn_histograms[mA])
        end
    end
end

function write_state_to_file(state_fh::SnapshotHandler, sim_state::SimState, sim_tracker::SimTracker, rng::AbstractRNG, m_count::Int64)
    if time_for_snapshot(state_fh, "state", m_count)
        # write state to file and reset counter (after any measurement)  
        write(state_fh, "state", sim_state)
        # write out trackers 
        write(state_fh, "tracker", sim_tracker)
        # write out rng 
        write(state_fh, "rng", rng) 
    end
end
