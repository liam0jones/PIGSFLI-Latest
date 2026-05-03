using LinearAlgebra, Statistics, LatticeModels, SparseArrays, Arpack
using DelimitedFiles, Plots, GLMakie, Measures, LaTeXStrings, Printf, Optim, Roots, JLD2

default(fontfamily="serif-roman", fmt=:svg)
set_theme!(theme_latexfonts())

# ── Lattice helpers ───────────────────────────────────────────────────────────

function get_lattice(geometry::String, Ls::Vector{Int64}, bcs::Vector{String})
    if geometry == "square"
        D = length(Ls)
        lattice = SquareLattice(Ls...)
        if D == 1
            periodic = bcs[1] == "pbc"
            return periodic ? setboundaries(lattice, PeriodicBoundary([Ls[1], 0])) : lattice
        else
            bc_periodic = [bcs[i] == "pbc" for i in 1:D]
            boundaries = [PeriodicBoundary(Ls[i] * ((1:D) .== i)) for i in findall(bc_periodic)]
            return isempty(boundaries) ? lattice : setboundaries(lattice, boundaries...)
        end

    elseif geometry == "triangular" && length(Ls) == 2
        periodic = all(bc == "pbc" for bc in bcs)
        return TriangularLattice(Ls[1], Ls[2], boundaries=(:axis1 => periodic, :axis2 => periodic))

    elseif geometry == "honeycomb" && length(Ls) == 2
        periodic = all(bc == "pbc" for bc in bcs)
        return HoneycombLattice(Ls[1], Ls[2], boundaries=(:axis1 => periodic, :axis2 => periodic))

    elseif geometry == "kagome" && length(Ls) == 2
        periodic = all(bc == "pbc" for bc in bcs)
        return KagomeLattice(Ls[1], Ls[2], boundaries=(:axis1 => periodic, :axis2 => periodic))

    else
        return "Invalid Geometry"
    end
end

function get_M(geometry::String, Ls::Vector{Int64})
    if geometry == "square"
        return prod(Ls)
    elseif geometry == "triangular"
        return prod(Ls)
    elseif geometry == "honeycomb"
        return 2 * prod(Ls)
    elseif geometry == "kagome"
        return 3 * prod(Ls)
    else
        return "Invalid Geometry"
    end
end

function adjacency_square(Ls::Vector{Int64}, bcs::Vector{String})
    D = length(Ls)
    lattice = SquareLattice(Ls...)
    if D == 1
        if bcs[1] == "pbc"
            lattice = setboundaries(lattice, PeriodicBoundary([Ls[1], 0]))
        end
    else
        pbcs = [PeriodicBoundary([(j == i ? Ls[i] : 0) for j=1:D]) for i=1:D if bcs[i]=="pbc"]
        isempty(pbcs) || (lattice = setboundaries(lattice, pbcs...))
    end
    return AdjacencyMatrix(lattice, NearestNeighbor()).mat
end

function adjacency_triangular(Ls::Vector{Int64}, bcs::Vector{String})
    lattice = TriangularLattice(Ls[1], Ls[2],
        boundaries=(:axis1 => bcs[1]=="pbc", :axis2 => bcs[2]=="pbc"))
    return AdjacencyMatrix(lattice, NearestNeighbor()).mat
end

function adjacency_honeycomb(Ls::Vector{Int64}, bcs::Vector{String})
    lattice = HoneycombLattice(Ls[1], Ls[2],
        boundaries=(:axis1 => bcs[1]=="pbc", :axis2 => bcs[2]=="pbc"))
    return AdjacencyMatrix(lattice, NearestNeighbor()).mat
end

function adjacency_kagome(Ls::Vector{Int64}, bcs::Vector{String})
    lattice = KagomeLattice(Ls[1], Ls[2],
        boundaries=(:axis1 => bcs[1]=="pbc", :axis2 => bcs[2]=="pbc"))
    return AdjacencyMatrix(lattice, NearestNeighbor()).mat
end

# ── Exact Diagonalization ─────────────────────────────────────────────────────

function BH_Hamiltonian(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64)
    lattice = get_lattice(geometry, Ls, bcs)
    return Symmetric(real.(bosehubbard(lattice, N; U=U, t1=-1.0).data))
end

function BH_ground_state(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64)
    H = BH_Hamiltonian(geometry, Ls, N, bcs, U)
    E0, Ψ0 = eigs(H; nev=1, which=:SR, tol=eps(Float64))
    return E0[1], Ψ0[:,1]
end

function get_ground_state(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64)
    param_tup = (geometry, Ls, N, bcs, U)
    stored_vals = readlines("/Users/liamjones/Bose_Hubbard.txt")
    if "$(param_tup)" in stored_vals
        tup_idx = findfirst(==("$(param_tup)"), stored_vals)
        E0 = parse(Float64, stored_vals[tup_idx+1])
        Ψ0 = parse.(Float64, split(stored_vals[tup_idx+2][2:end-1], ", "))
    else
        E0, Ψ0 = BH_ground_state(geometry, Ls, N, bcs, U)
        open("/Users/liamjones/Bose_Hubbard.txt","a") do f
            write(f,"$(param_tup)","\n","$(E0)","\n","$(Ψ0)","\n")
        end
    end
    return E0, Ψ0
end

function BH_interaction(bra::Vector{Int64}, U::Float64)
    interaction_sum = 0
    nonzero = findall(x -> x!=0, bra)
    for i in nonzero
        interaction_sum += U*bra[i]*(bra[i]-1)/2
    end
    return interaction_sum
end

function get_energies(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64)
    lattice = get_lattice(geometry, Ls, bcs)
    lat_basis = LatticeBasis(lattice)
    basis = ManyBodyBasis(lat_basis, bosonstates(lat_basis, N))
    configs = [Vector{Int64}(s) for s in basis.occupations]
    E0, Ψ0 = get_ground_state(geometry, Ls, N, bcs, U)
    V0 = sum(BH_interaction(c, U)*p^2 for (c,p) in zip(configs, Ψ0))
    K0 = E0 - V0
    return Dict{String,Float64}("E"=>E0, "K"=>K0, "V"=>V0)
end

function get_density(configs::Vector{Vector{Int64}}, psi::Vector{Float64}, site::Int64, squared::Bool)
    n = 0
    if !squared
        for s=1:length(configs)
            n += configs[s][site]*psi[s]^2
        end
    else
        for s=1:length(configs)
            n += configs[s][site]^2*psi[s]^2
        end
    end
    return n
end

function get_densities(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64)
    M = get_M(geometry, Ls)
    lattice = get_lattice(geometry, Ls, bcs)
    lat_basis = LatticeBasis(lattice)
    basis = ManyBodyBasis(lat_basis, bosonstates(lat_basis, N))
    configs = [Vector{Int64}(s) for s in basis.occupations]
    Ψ0 = get_ground_state(geometry, Ls, N, bcs, U)[2]
    n = [get_density(configs, Ψ0, s, false) for s=1:M]
    n_squared = [get_density(configs, Ψ0, s, true) for s=1:M]
    return Dict{String,Vector{Float64}}("n"=>n, "n^2"=>n_squared, "σ^2_n"=>n_squared.-(n.^2))
end

# ── QMC data loading ──────────────────────────────────────────────────────────

_size_str(Ls::Vector{Int64}) =
    all(==(Ls[1]), Ls) ? string(Ls[1]) : join(string.(Ls), "x")

_bc_label(bcs::Vector{String}) =
    all(==(bcs[1]), bcs) ? (bcs[1]=="pbc" ? "PBC" : "OBC") :
    join([bc=="pbc" ? "P" : "O" for bc in bcs], "")

_sim_folder(geometry, Ls, N, bcs, U) =
    @sprintf "/%s_%dD/%sL_%dN/%s/%.1fU" geometry length(Ls) _size_str(Ls) N _bc_label(bcs) U

_sim_path(geometry, Ls, N, bcs, U, beta) =
    _sim_folder(geometry, Ls, N, bcs, U) * @sprintf "/%.1fbeta" beta

function QMC_data(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64, beta::Float64)
    folderpath = _sim_path(geometry, Ls, N, bcs, U, beta)
    basepath = "/Users/liamjones/Pigsfli.jl/out"
    fullpath  = basepath * folderpath

    jld2_files = filter(f -> endswith(f, ".jld2"), readdir(fullpath))
    isempty(jld2_files) && error("No JLD2 files found in $(fullpath)")

    num_seeds = length(jld2_files)
    nt        = Threads.maxthreadid()

    # K and V are scalars — keep as arrays so E^2 = mean((K+V)^2) stays exact
    TK = [Float64[] for _ in 1:nt]
    TV = [Float64[] for _ in 1:nt]

    # n and C are large per-bin arrays; store each bin and truncate later to the
    # minimum common bin count for all observables.
    Tn_bins    = [Vector{Vector{Float64}}() for _ in 1:nt]
    Tn2_bins   = [Vector{Vector{Float64}}() for _ in 1:nt]
    Tn_sq_bins = [Vector{Vector{Float64}}() for _ in 1:nt]
    TC_bins    = [Vector{Matrix{Float64}}() for _ in 1:nt]

    Threads.@threads :static for jld2_file in jld2_files
        tid = Threads.threadid()
        jldopen(joinpath(fullpath, jld2_file), "r") do file
            for k in keys(file)
                if startswith(k, "K_")
                    push!(TK[tid], file[k])
                elseif startswith(k, "V_")
                    push!(TV[tid], file[k])
                elseif startswith(k, "n^2_")
                    push!(Tn2_bins[tid], file[k]::Vector{Float64})
                elseif startswith(k, "n_")
                    v = file[k]::Vector{Float64}
                    push!(Tn_bins[tid], v)
                    push!(Tn_sq_bins[tid], v .^ 2)
                elseif startswith(k, "C^2_")
                    # Skip C^2_ since we'll compute it from normalized C
                    continue
                elseif startswith(k, "C_")
                    v = file[k]::Matrix{Float64}
                    trace_C = sum(diag(v))
                    if trace_C > 0
                        v_norm = v * (N / trace_C)
                        push!(TC_bins[tid], v_norm)
                    end
                end
            end
        end
    end

    K_bins      = reduce(vcat, TK)
    V_bins      = reduce(vcat, TV)
    n_bins_all  = reduce(vcat, Tn_bins)
    n2_bins_all = reduce(vcat, Tn2_bins)
    n_sq_all    = reduce(vcat, Tn_sq_bins)
    C_bins_all  = reduce(vcat, TC_bins)

    total_n = length(n_bins_all)
    total_C = length(C_bins_all)
    scalar_bins = (!isempty(K_bins) && !isempty(V_bins)) ? min(length(K_bins), length(V_bins)) : typemax(Int)
    positive_bins = filter(>=(1), [scalar_bins, total_n, total_C])
    isempty(positive_bins) && error("No observables found in JLD2 files for $(fullpath)")
    total_bins = minimum(positive_bins)
    out = Dict{String,Any}("bins" => div(total_bins, num_seeds), "seeds" => num_seeds)

    if !isempty(K_bins) && !isempty(V_bins)
        n_KV   = min(length(K_bins), length(V_bins), total_bins)
        K_bins = K_bins[1:n_KV]
        V_bins = V_bins[1:n_KV]
        E_bins = K_bins .+ V_bins
        out["K"]  = mean(K_bins);  out["K^2"] = mean(abs2, K_bins)
        out["V"]  = mean(V_bins);  out["V^2"] = mean(abs2, V_bins)
        out["E"]  = mean(E_bins);  out["E^2"] = mean(abs2, E_bins)
    end

    if total_n > 0
        n_bins = n_bins_all[1:total_bins]
        n2_bins = n2_bins_all[1:total_bins]
        n_sq_bins = n_sq_all[1:total_bins]
        out["n"]        = sum(n_bins)    ./ total_bins
        out["n^2"]      = sum(n2_bins)   ./ total_bins
        out["n^2_stat"] = sum(n_sq_bins) ./ total_bins
    end

    if total_C > 0
        C_bins = C_bins_all[1:total_bins]
        out["C"]   = sum(C_bins) ./ total_bins
        out["C^2"] = reduce(+, (c .^ 2 for c in C_bins)) ./ total_bins
        out["C_bins"] = div(total_bins, num_seeds)
    end

    out["geometry"] = geometry
    out["Ls"] = Ls
    out["N"] = N
    out["bcs"] = bcs
    out["U"] = U
    out["beta"] = beta

    return out
end

function QMC_error(means::Dict{String,Any})
    new_dict = Dict{String,Any}()
    N_s = means["bins"] * means["seeds"]
    for key in ["K","V","E"]
        if haskey(means, key) && haskey(means, key*"^2")
            new_dict[key] = @. sqrt((means[key*"^2"] - means[key]^2) / N_s)
        end
    end
    if haskey(means, "n") && haskey(means, "n^2_stat")
        new_dict["n"] = @. sqrt((means["n^2_stat"] - means["n"]^2) / N_s)
    end
    if haskey(means, "C") && haskey(means, "C^2")
        new_dict["C"] = @. sqrt((means["C^2"] - means["C"]^2) / N_s)
    end
    return new_dict
end

QMC_error(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64, beta::Float64) =
    QMC_error(QMC_data(geometry, Ls, N, bcs, U, beta))

function QMC_raw_data(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64, beta::Float64, observable::String)
    # Returns a vector of raw bin data for the specified observable from all JLD2 files.
    # Supported observables: "K", "V", "n", "n^2", "C" (normalized as in QMC_data).
    # For "C", each bin is normalized so that trace(C) = N.
    folderpath = _sim_path(geometry, Ls, N, bcs, U, beta)
    basepath = "/Users/liamjones/Pigsfli.jl/out"
    fullpath  = basepath * folderpath

    jld2_files = filter(f -> endswith(f, ".jld2"), readdir(fullpath))
    isempty(jld2_files) && error("No JLD2 files found in $(fullpath)")

    raw_data = []

    for jld2_file in jld2_files
        jldopen(joinpath(fullpath, jld2_file), "r") do file
            for k in keys(file)
                if observable == "K" && startswith(k, "K_")
                    push!(raw_data, file[k])
                elseif observable == "V" && startswith(k, "V_")
                    push!(raw_data, file[k])
                elseif observable == "n" && startswith(k, "n_")
                    push!(raw_data, file[k])
                elseif observable == "n^2" && startswith(k, "n^2_")
                    push!(raw_data, file[k])
                elseif observable == "C" && startswith(k, "C_")
                    v = file[k]::Matrix{Float64}
                    trace_C = sum(diag(v))
                    if trace_C > 0
                        v_norm = v * (N / trace_C)
                        push!(raw_data, v_norm)
                    end
                end
            end
        end
    end

    return raw_data
end

# ── Plotting: beta convergence ────────────────────────────────────────────────

function get_betas(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64)
    folderpath = _sim_folder(geometry, Ls, N, bcs, U)
    betas = filter(!=(".DS_Store"), readdir("/Users/liamjones/Pigsfli.jl/out" * folderpath))
    sort!(betas, by=length)
    betas = parse.(Float64,[b[1:findlast(isdigit,b)] for b in betas])
    data = [QMC_data(geometry, Ls, N, bcs, U, b) for b in betas]
    return data
end

function fit_beta_extrapolation(betas::Vector{Float64}, y::Vector{Float64}, Δy::Vector{Float64})
    length(betas) < 3 && return nothing
    A0 = y[end]
    B0 = y[1] - y[end]; B0 == 0.0 && (B0 = 1e-8)
    C0 = max(1.0 / betas[end], 1e-3)
    w  = all(iszero, Δy) ? ones(length(y)) : 1.0 ./ max.(Δy, 1e-15) .^ 2
    function loss(p)
        A, B, logC = p[1], p[2], p[3]
        return sum(w[i] * (y[i] - A - B * exp(-exp(logC) * betas[i]))^2 for i in eachindex(betas))
    end
    try
        res = optimize(loss, [A0, B0, log(C0)], LBFGS(),
                       Optim.Options(iterations=50_000, x_abstol=1e-12); autodiff=:forward)
        p = Optim.minimizer(res)
        return (A=p[1], B=p[2], C=exp(p[3]))
    catch
        return nothing
    end
end

function plot_betas(data::Vector{Dict{String, Any}}; sites::Vector{Int64} = [1,2], extrapolate::Bool=true)
    errors = [QMC_error(d) for d in data]
    geometry = data[1]["geometry"]
    Ls = data[1]["Ls"]
    N = data[1]["N"]
    bcs = data[1]["bcs"]
    U = data[1]["U"]
    betas = [data[i]["beta"] for i=1:length(data)]
    sort!(betas, by=length)
    values = ["K","V","E","n"]
    ED_energies  = get_energies(geometry, Ls, N, bcs, U)
    ED_densities = get_densities(geometry, Ls, N, bcs, U)
    y  = Vector{Vector{Float64}}()
    Δy = Vector{Vector{Float64}}()
    ED = Vector{Vector{Float64}}()
    M  = get_M(geometry, Ls)
    for v in values
        if v in ["K","V","E"]
            push!(y,  [d[v]   for d in data])
            push!(Δy, [err[v] for err in errors])
            push!(ED, [ED_energies[v] for _ in betas])
        elseif v == "n"
            for i in sites
                push!(y,  [d[v][i]   for d in data])
                push!(Δy, [err[v][i] for err in errors])
                push!(ED, [ED_densities["n"][i] for _ in betas])
            end
        end
    end
    β_dense = collect(range(minimum(betas), maximum(betas), length=300))
    fits    = extrapolate ? [fit_beta_extrapolation(betas, y[k], Δy[k]) for k in 1:length(y)] :
                           fill(nothing, length(y))

    function make_panel(k::Int, color::String, obs_label, ed_val::Float64; kwargs...)
        fit = fits[k]
        if fit !== nothing
            y_fit = [fit.A + fit.B * exp(-fit.C * β) for β in β_dense]
            curve_label = @sprintf("β→∞: %.5g", fit.A)
            p = Plots.plot(β_dense, y_fit; lw=2, ls=:dash, seriescolor=color,
                           label=curve_label, kwargs...)
        else
            p = Plots.plot(betas, y[k]; lw=2, ls=:dash, seriescolor=color,
                           label="", kwargs...)
        end
        Plots.hline!(p, [ed_val]; lw=2, ls=:solid, seriescolor=color, label= @sprintf("ED: %.5g", ed_val))
        Plots.plot!(p, betas, y[k]; yerror=Δy[k], seriestype=:scatter,
                    ms=4, msc="black", lw=1, seriescolor=color, label=obs_label)
        return p
    end

    K_p  = make_panel(1, "red",        L"$\langle K\:\rangle$",                    ED[1][1]; xticks=betas, topmargin=4mm)
    V_p  = make_panel(2, "green",      L"$\langle V\;\rangle$",                    ED[2][1]; xticks=betas)
    E_p  = make_panel(3, "dodgerblue", L"$\langle E\:\rangle$",                    ED[3][1]; xticks=betas, xlabel=L"$\beta$")
    n1_p = make_panel(4, "turquoise",  L"$\langle n\:\rangle$"*" (i=$(sites[1]))", ED[4][1]; xticks=betas, topmargin=4mm)
    n2_p = make_panel(5, "purple",     L"$\langle n\:\rangle$"*" (i=$(sites[2]))", ED[5][1]; xticks=betas, xlabel=L"$\beta$")
    l = @layout [grid(3,1) grid(2,1)]
    Plots.plot(K_p, V_p, E_p, n1_p, n2_p, layout=l,
        plot_title="Ground State Observable Convergence"*"\n($(geometry), $(_bc_label(bcs)), M=$(M), N=$(N), U/t=$(U))",
        size=(800,500), margin=0.6mm, plot_titlefontsize=12, legendfontsize=8, xlabelfontsize=12, tickfontsize=9, fmt=:svg)
end

plot_betas(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64; sites::Vector{Int64} = [1,2]) = plot_betas(geometry, Ls, N, bcs, U, get_betas(geometry, Ls, N, bcs, U)...)

function plot_density(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64,
                      val::String, beta::Float64, ED::Bool, osc_range::Tuple{Int64,Int64})
    vals = Dict{String,Tuple{String,String}}(
        "n"     =>(L"$\langle n_i \rangle$","Density"),
        "n^2"   =>(L"$\langle n^2_i \rangle$","Density Squared"),
        "σ^2_n" =>(L"\langle \sigma^2_{n_i} \rangle","Density Variance"))
    if ED
        y = get_densities(geometry, Ls, N, bcs, U)[val][osc_range[1]:osc_range[2]]
        Plots.plot(osc_range[1]:osc_range[2], y, xlabel=L"$i$", ylabel=vals[val][1],
            lw=2, ls=:dash, primary=false, seriescolor="lightseagreen")
        Plots.plot!(osc_range[1]:osc_range[2], y, seriestype=:scatter, primary=false, mw=1, seriescolor="lightseagreen")
        M = get_M(geometry, Ls)
        Plots.plot!(title="ED Ground State Boson "*vals[val][2]*"\n(M=$(M), N=$(N), U/t=$(U), $(_bc_label(bcs)))\n",
            margin=0.5mm, plot_titlefontsize=12, legendfontsize=10, xlabelfontsize=12, tickfontsize=9)
    else
        folderpath = _sim_path(geometry, Ls, N, bcs, U, beta)
        data   = QMC_data(geometry, Ls, N, bcs, U, beta)
        y      = data[val][osc_range[1]:osc_range[2]]
        yerror = val == "n" ? QMC_error(data)["n"][osc_range[1]:osc_range[2]] : zeros(Int64, length(y))
        Plots.plot(osc_range[1]:osc_range[2], y, xlabel=L"$i$", ylabel=vals[val][1],
            lw=2, ls=:dash, primary=false, seriescolor="slateblue")
        Plots.plot!(osc_range[1]:osc_range[2], y; yerror, seriestype=:scatter, primary=false, mw=1, seriescolor="slateblue")
        M = get_M(geometry, Ls)
        Plots.plot!(title="Ground State Boson $(vals[val][2])\n(M=$(M), N=$(N), U/t=$(U), $(_bc_label(bcs)), β=$(beta))\n",
            margin=0.5mm, plot_titlefontsize=12, legendfontsize=10, xlabelfontsize=12, tickfontsize=9)
    end
end

plot_density(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64, beta::Float64) =
    plot_density(geometry, Ls, N, bcs, U, "n", beta, false, (1, get_M(geometry, Ls)))

plot_density(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64, val::String, beta::Float64) =
    plot_density(geometry, Ls, N, bcs, U, val, beta, false, (1, get_M(geometry, Ls)))

plot_density(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64, beta::Float64, osc_range::Tuple{Int64,Int64}) =
    plot_density(geometry, Ls, N, bcs, U, "n", beta, false, osc_range)

plot_density_ED(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64) =
    plot_density(geometry, Ls, N, bcs, U, "n", 0.0, true, (1, get_M(geometry, Ls)))

plot_density_ED(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64, val::String) =
    plot_density(geometry, Ls, N, bcs, U, val, 0.0, true, (1, get_M(geometry, Ls)))

plot_density_ED(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64, osc_range::Tuple{Int64,Int64}) =
    plot_density(geometry, Ls, N, bcs, U, "n", 0.0, true, osc_range)

# ── Correlation matrix ────────────────────────────────────────────────────────

function correlation(configs::Vector{Vector{Int64}}, psi::Vector{Float64}, i::Int64, j::Int64)
    if i == j
        return get_density(configs, psi, i, false)
    end
    hilbert_size = length(configs)
    corr = zeros(Float64, hilbert_size)
    for x=1:hilbert_size
        config = configs[x]
        if config[j] != 0
            new_config = copy(config)
            new_config[j] -= 1
            new_config[i] += 1
            corr[findfirst(==(new_config), configs)] = sqrt(config[j])*sqrt(config[i]+1)*psi[x]
        end
    end
    return abs(psi'*corr)
end

function get_corr_mat(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64)
    M = get_M(geometry, Ls)
    lattice = get_lattice(geometry, Ls, bcs)
    lat_basis = LatticeBasis(lattice)
    basis = ManyBodyBasis(lat_basis, bosonstates(lat_basis, N))
    configs = [Vector{Int64}(s) for s in basis.occupations]
    Ψ0 = get_ground_state(geometry, Ls, N, bcs, U)[2]
    c_mat = zeros(Float64, M, M)
    for j=1:M
        c_mat[:,j] = [correlation(configs, Ψ0, i, j) for i=1:M]
    end
    return c_mat
end

function plot_corr_mat(data::Dict{String, Any}; network::Bool=false)
    stat_error = QMC_error(data)["C"]
    geometry   = data["geometry"]
    Ls         = data["Ls"]
    N          = data["N"]
    bcs        = data["bcs"]
    U          = data["U"]
    beta       = data["beta"]
    M          = get_M(geometry, Ls)
    C_qmc      = data["C"]
    C_ed       = get_corr_mat(geometry, Ls, N, bcs, U)

    indices  = CartesianIndices(C_ed)
    mi       = [indices[k][1] for k in eachindex(indices)]
    mj       = [indices[k][2] for k in eachindex(indices)]
    err_bars = [(0.4, 0.4, abs(C_qmc[indices[k]] - C_ed[indices[k]]) /
                              max(abs(C_ed[indices[k]]), 1e-12))
                for k in eachindex(indices)]
    errors   = [e[3] for e in err_bars]

    e_el = 0.38pi;  azim = 0.25pi;  persp = 0.1;  prot = 10

    f = Figure(size=(1000, 400))

    if network
        lattice   = get_lattice(geometry, Ls, bcs)
        node_lv   = LatticeValue(lattice, diag(C_qmc))
        lx        = Float64[node_lv.latt[i].x for i in eachindex(node_lv.latt)]
        ly        = Float64[node_lv.latt[i].y for i in eachindex(node_lv.latt)]
        node_vals = node_lv.values

        segs   = Vector{Tuple{Float64,Float64}}()
        widths = Float64[]
        for i = 1:M, j = i+1:M
            w = abs(C_qmc[i, j])
            push!(segs, (lx[i], ly[i]));  push!(segs, (lx[j], ly[j]))
            push!(widths, w)
        end
        C_off_max     = isempty(widths) ? 1.0 : maximum(widths)
        widths_scaled = widths ./ C_off_max .* 8

        ax_net = f[1,1] = Axis(f,
            xlabel=L"$x$", ylabel=L"$y$",
            title="Correlation Network\n($(geometry), $(_bc_label(bcs)), M=$(M), N=$(N), U/t=$(U), β=$(beta))",
            titlefont=:regular, titlesize=18, xlabelsize=18, ylabelsize=18)
        linesegments!(ax_net, segs;
            linewidth  = repeat(widths_scaled, inner=2),
            color      = repeat(widths, inner=2),
            colormap   = :viridis,
            colorrange = (0, C_off_max))
        Colorbar(f[1,2], colormap=:viridis, limits=(0, C_off_max),
            label=L"\langle C_{ij}\rangle", labelsize=16,
            flipaxis=false, vertical=true, size=10)
        n_max_val = maximum(node_vals)
        GLMakie.scatter!(ax_net, lx, ly,
            markersize=node_vals ./ n_max_val .* 30 .+ 6, marker=:circle, color="black")
        GLMakie.scatter!(ax_net, lx, ly,
            markersize=node_vals ./ n_max_val .* 30 .+ 4, marker=:circle,
            color=node_vals, colormap=:navia)
    else
        z   = [(0.4, 0.4, C_qmc[indices[k]]) for k in eachindex(indices)]
        bar = 10

        f[1,1] = Axis3(f,
            xlabel=L"i", ylabel=L"j", zlabel=L"\langle C_{ij}\rangle",
            zlabelrotation=0.5pi,
            title="Data ($(geometry), $(_bc_label(bcs)), M=$(M), N=$(N), U/t=$(U), β=$(beta))",
            titlefont=:regular, titlesize=18, aspect=(12,12,12),
            elevation=e_el, perspectiveness=persp, azimuth=azim, protrusions=prot,
            xlabeloffset=30, ylabeloffset=30, zlabeloffset=60,
            xlabelsize=18, ylabelsize=18, zlabelsize=18)
        linesegments!(f[1,1],
            [mi[k] for k in eachindex(indices) for _ = 1:2],
            [mj[k] for k in eachindex(indices) for _ = 1:2],
            [z[k][3] + t*stat_error[indices[k]]*bar
             for k in eachindex(indices) for t in [1,-1]],
            color="black", linewidth=1, linecap=:square, fxaa=true)
        linesegments!(f[1,1],
            [mi[k]+t*0.08 for k in eachindex(indices) for t in [1,-1]],
            [mj[k]+t*0.08 for k in eachindex(indices) for t in [-1,1]],
            [z[k][3]+stat_error[indices[k]]*bar
             for k in eachindex(indices) for _ = 1:2],
            color="black", linewidth=2, linecap=:square, fxaa=false)
        meshscatter!(f[1,1], mi, mj,
            markersize=z, marker=Rect3f((-0.5,-0.5,0),(1,1,1)),
            color=[t[3] for t in z])
        Colorbar(f[1,2],
            limits=(minimum(C_qmc), maximum(C_qmc)), colormap=:viridis,
            flipaxis=false, vertical=true, size=10)
    end

    f[1,3] = Axis3(f,
        xlabel=L"i", ylabel=L"j", zlabel="", aspect=(12,12,12),
        title="Relative Error", titlefont=:regular, titlesize=18,
        elevation=e_el, perspectiveness=persp, azimuth=azim, protrusions=prot,
        xlabeloffset=30, ylabeloffset=30, zlabeloffset=40,
        xlabelsize=18, ylabelsize=18, zlabelsize=18)
    meshscatter!(f[1,3], mi, mj,
        markersize=err_bars, marker=Rect3f((-0.5,-0.5,0),(1,1,1)),
        color=[t[3] for t in err_bars], colormap=:acton)
    Colorbar(f[1,4],
        colormap=:acton, flipaxis=false, halign=:left,
        limits=(minimum(errors), maximum(errors)), vertical=true, size=10)

    resize_to_layout!(f)
    f
end

plot_corr_mat(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64, beta::Float64; network::Bool=false) =
    plot_corr_mat(QMC_data(geometry, Ls, N, bcs, U, beta); network=network)

function plot_C_betas(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64)
    data_list = get_betas(geometry, Ls, N, bcs, U)
    betas     = [d["beta"] for d in data_list]
    C_ed      = get_corr_mat(geometry, Ls, N, bcs, U)
    M         = size(C_ed, 1)

    mean_rel_err = map(data_list) do d
        haskey(d, "C") || return NaN
        C = d["C"]
        errs = [abs(C[i,j] - C_ed[i,j]) / abs(C_ed[i,j])
                for i in 1:M for j in 1:M if abs(C_ed[i,j]) > 1e-10]
        isempty(errs) ? NaN : mean(errs)
    end

    Plots.plot(betas, mean_rel_err,
        xlabel=L"$\beta$", ylabel="Mean Relative Error",
        title="Correlation Matrix β-Convergence\n"*
              "($(geometry), $(_bc_label(bcs)), M=$(get_M(geometry, Ls)), N=$(N), U/t=$(U))",
        xticks=betas, seriestype=:scatter, marker=(:circle, 5),
        lw=2, seriescolor="mediumpurple", legend=false,
        plot_titlefontsize=12, tickfontsize=9, xlabelfontsize=12,
        ylabelfontsize=12, margin=3mm, fmt=:svg)
end

# ── 2D visualization ──────────────────────────────────────────────────────────

function lattice_bonds(x::Vector{Float64}, y::Vector{Float64}, adj_sparse::AbstractSparseMatrix, M::Int64)
    bonds = Vector{Vector{Tuple{Float64,Float64}}}[]
    for i = 1:M
        for j in adj_sparse.rowval[adj_sparse.colptr[i]:adj_sparse.colptr[i+1]-1]
            if i < j
                push!(bonds, [(x[i], y[i]), (x[j], y[j])])
            end
        end
    end
    return [bond[k] for bond in bonds for k = 1:2]
end

function plot_density_2D(data::Dict{String,Any})
    geometry = data["geometry"]
    Ls       = data["Ls"]
    N        = data["N"]
    bcs      = data["bcs"]
    U        = data["U"]
    beta     = data["beta"]
    M        = get_M(geometry, Ls)
    lattice  = get_lattice(geometry, Ls, bcs)

    ED_n       = get_densities(geometry, Ls, N, bcs, U)["n"]
    density_lv = LatticeValue(lattice, data["n"])
    err_lv     = LatticeValue(lattice,
                     [ED_n[i] > 1e-12 ? abs(ED_n[i] - data["n"][i]) / ED_n[i] : 0.0
                      for i in 1:M])

    x       = Float64[density_lv.latt[i].x for i in eachindex(density_lv.latt)]
    y       = Float64[density_lv.latt[i].y for i in eachindex(density_lv.latt)]
    density = density_lv.values
    err     = err_lv.values

    adj_mat = geometry == "square"     ? adjacency_square(Ls, bcs)     :
              geometry == "triangular" ? adjacency_triangular(Ls, bcs) :
              geometry == "honeycomb"  ? adjacency_honeycomb(Ls, bcs)  :
              adjacency_kagome(Ls, bcs)
    bonds = lattice_bonds(x, y, adj_mat, M)

    f = Figure(size=(900, 400))

    f[1,1] = Axis(f, xlabel=L"$x$", ylabel=L"$y$",
        title="Ground State Boson Density\n($(geometry), $(_bc_label(bcs)), M=$(M), N=$(N), U/t=$(U), β=$(beta))",
        titlesize=20, xlabelsize=20, ylabelsize=20, titlefont=:regular)
    linesegments!(f[1,1], bonds, linewidth=3, linecap=:round, fxaa=true, color="turquoise2")
    GLMakie.scatter!(f[1,1], x, y,
        markersize=density ./ maximum(density) .* 40 .+ 4, marker=:circle, color="black")
    GLMakie.scatter!(f[1,1], x, y,
        markersize=density ./ maximum(density) .* 40, marker=:circle, color=density, colormap=:navia)
    Colorbar(f[1,2], colormap=:navia, limits=(minimum(density), maximum(density)),
        label=L"\langle n_{x,y}\:\rangle", labelsize=20)

    f[1,3] = Axis(f, xlabel=L"$x$", ylabel=L"$y$",
        title="Relative Error", titlesize=20, xlabelsize=20, ylabelsize=20, titlefont=:regular)
    linesegments!(f[1,3], bonds, linewidth=3, linecap=:round, fxaa=true, color="gold")
    GLMakie.scatter!(f[1,3], x, y,
        markersize=err ./ maximum(err) .* 40 .+ 4, marker=:circle, color="black")
    GLMakie.scatter!(f[1,3], x, y,
        markersize=err ./ maximum(err) .* 40, marker=:circle, color=err, colormap=:lajolla)
    Colorbar(f[1,4], limits=(minimum(err), maximum(err)), colormap=:lajolla, label="", labelsize=20)

    resize_to_layout!(f)
    f
end

plot_density_2D(geometry::String, Ls::Vector{Int64}, N::Int64, bcs::Vector{String}, U::Float64, beta::Float64) = plot_density_2D(QMC_data(geometry, Ls, N, bcs, U, beta))

# ── Gutzwiller mean-field ─────────────────────────────────────────────────────

"""
──────────────────────────────────────────────────────────────────────────────
Gutzwiller ansatz
──────────────────────────────────────────────────────────────────────────────

  |Ψ_G⟩ = ∏_i  Σ_{n=0}^{n_max}  f_n^(i) |n⟩_i

The wavefunction factorises over sites; entanglement between sites enters only
through the self-consistency condition on the mean-field order parameter ⟨b⟩.
For a translationally invariant state (possibly with sublattice structure) the
amplitudes depend only on the sublattice index σ ∈ {A, B, C, …}:

  f_n^(σ)  ∝  κ_σ^n / √(n!)     (coherent-state form),   normalised: Σ_n |f_n|² = 1

This family interpolates between the Mott insulator (κ_σ → 0, f_n → δ_{n,n̄})
and the superfluid (κ_σ finite, Poissonian-like distribution).  The single
variational parameter κ_σ > 0 per sublattice is optimised in log-space,
  κ_σ = exp(κ̃_σ),  κ̃_σ ∈ ℝ,
so that positivity is enforced throughout without any constraints.

──────────────────────────────────────────────────────────────────────────────
Site expectation values
──────────────────────────────────────────────────────────────────────────────

All real for the coherent-state ansatz:

  ⟨b⟩   = Σ_{n=0}^{n_max−1}  f_n √(n+1) f_{n+1}       (hopping order parameter)
  ⟨n⟩   = Σ_n  n |f_n|²                                 (mean occupation)
  ⟨n(n−1)⟩ = Σ_n  n(n−1) |f_n|²                         (interaction energy weight)

──────────────────────────────────────────────────────────────────────────────
Variational energy via mean-field decoupling
──────────────────────────────────────────────────────────────────────────────

Hopping is decoupled by replacing one factor in each bond:
  −t ⟨b†_i b_j⟩  →  −t ⟨b†_i⟩ ⟨b_j⟩  =  −t ⟨b⟩_σ ⟨b⟩_σ'

This yields the following variational energies (per unit cell in each case):

  Square / triangular  (1 sublattice, coordination z = 4 or 6):
    E/site = (U/2) ⟨n(n−1)⟩  −  μ ⟨n⟩  −  z t ⟨b⟩²

  Honeycomb  (2 sublattices A, B;  z = 3;  3 A–B bonds per unit cell):
    E/UC = (U/2)(⟨n(n−1)⟩_A + ⟨n(n−1)⟩_B)  −  μ(⟨n⟩_A + ⟨n⟩_B)  −  3t ⟨b⟩_A ⟨b⟩_B

  Kagome  (3 sublattices A, B, C;  z = 4;  2 bonds of each type per unit cell):
    E/UC = (U/2) Σ_σ ⟨n(n−1)⟩_σ  −  μ Σ_σ ⟨n⟩_σ
           − 2t (⟨b⟩_A ⟨b⟩_B  +  ⟨b⟩_B ⟨b⟩_C  +  ⟨b⟩_A ⟨b⟩_C)

──────────────────────────────────────────────────────────────────────────────
Canonical-ensemble optimization
──────────────────────────────────────────────────────────────────────────────

For a system with N_target particles on N_sites sites the canonical problem is:

  min_{κ}  E_kin(κ) + E_int(κ)      subject to  ⟨N⟩(κ) = N_target

This is equivalent to a grand-canonical minimization at the unique chemical
potential μ* that enforces the constraint.  Since ∂⟨N⟩/∂μ > 0 (grand-canonical
compressibility is positive for the Bose-Hubbard model), ⟨N⟩(μ) is strictly
monotone and μ* is unique.

Implementation — nested two-level scheme:

  Outer (bisection on μ): bracket μ with bounds
    μ_lo = −2(U·n_max + z·t),    μ_hi = +2(U·n_max + z·t)
  which guarantee ⟨N⟩(μ_lo) ≈ 0 and ⟨N⟩(μ_hi) ≈ n_max · N_sites.
  Bisect until |μ_hi − μ_lo| < 10⁻¹⁰ (≤ 80 iterations).

  Inner (LBFGS over κ at fixed μ): minimize E(κ; μ) using LBFGS with
  forward-mode automatic differentiation (ForwardDiff via Optim.jl).
  Convergence: |Δκ| < 10⁻¹², |ΔE|/|E| < 10⁻¹⁴.
"""

function gutzwiller_amplitudes(κ::T, n_max::Int, log_fact::Vector{Float64}) where T
    log_κ = κ  # κ is in log space; physical parameter is exp(κ) > 0
    f = [exp(n * log_κ - 0.5 * log_fact[n+1]) for n in 0:n_max]
    norm_sq = sum(abs2, f)
    f ./= sqrt(norm_sq)
    return f
end

function site_expectations(f::Vector{T}, n_max::Int) where T
    b     = sum(f[n+1] * f[n+2] * sqrt(Float64(n+1)) for n in 0:n_max-1)
    n_avg = sum(f[n+1]^2 * n           for n in 0:n_max)
    nn    = sum(f[n+1]^2 * n * (n - 1) for n in 0:n_max)
    return (b=b, n=n_avg, nn=nn)
end

function sum_sublattice_filling(κ::Vector{T}, n_max::Int, log_fact::Vector{Float64}) where T
    total = zero(T)
    for κ_σ in κ
        f = gutzwiller_amplitudes(κ_σ, n_max, log_fact)
        total += site_expectations(f, n_max).n
    end
    return total
end

"""
Single-sublattice Gutzwiller energy per site (square and triangular lattices).
z: coordination number (square=4, triangular=6).
"""
function gutzwiller_energy_general(κ::Vector{T}, t::Float64, U::Float64,
                                    μ::T, z::Int, n_max::Int,
                                    log_fact::Vector{Float64}) where T
    f    = gutzwiller_amplitudes(κ[1], n_max, log_fact)
    exp_ = site_expectations(f, n_max)
    return (U / 2) * exp_.nn  -  μ * exp_.n  -  z * t * exp_.b^2
end

"""
Honeycomb lattice Gutzwiller energy per unit cell.
Two sublattices A and B (z=3 each); all bonds are A–B type.
"""
function honeycomb_gutzwiller_energy(κ::Vector{T}, t::Float64, U::Float64,
                                      μ::T, n_max::Int,
                                      log_fact::Vector{Float64}) where T
    exp_A = site_expectations(gutzwiller_amplitudes(κ[1], n_max, log_fact), n_max)
    exp_B = site_expectations(gutzwiller_amplitudes(κ[2], n_max, log_fact), n_max)
    return (U / 2) * (exp_A.nn + exp_B.nn) -
           μ       * (exp_A.n  + exp_B.n)  -
           3t      *  exp_A.b  * exp_B.b
end

"""
Kagome lattice Gutzwiller energy per unit cell.
Three sublattices A, B, C (z=4 each); 2 bonds of each type per unit cell.
"""
function kagome_gutzwiller_energy(κ::Vector{T}, t::Float64, U::Float64,
                                   μ::T, n_max::Int,
                                   log_fact::Vector{Float64}) where T
    κ_A, κ_B, κ_C = κ
    exp_A = site_expectations(gutzwiller_amplitudes(κ_A, n_max, log_fact), n_max)
    exp_B = site_expectations(gutzwiller_amplitudes(κ_B, n_max, log_fact), n_max)
    exp_C = site_expectations(gutzwiller_amplitudes(κ_C, n_max, log_fact), n_max)
    return (U / 2) * (exp_A.nn + exp_B.nn + exp_C.nn) -
           μ       * (exp_A.n  + exp_B.n  + exp_C.n)  -
           2t      * (exp_A.b * exp_B.b + exp_B.b * exp_C.b + exp_A.b * exp_C.b)
end

"""
Canonical Gutzwiller optimizer for square, triangular, honeycomb, and kagome
lattices. Uses a nested scheme: bisect on μ (outer) to enforce the particle
number constraint, minimize the variational energy over κ at fixed μ (inner).
"""
function optimize_gutzwiller(lattice::String, U::Float64,
                              N_target::Int, N_sites::Int;
                              t::Float64=1.0,
                              n_max::Int=10)

    N_target / N_sites ≤ n_max ||
        error("Target filling $(N_target/N_sites) exceeds n_max=$n_max")

    log_fact = [sum(log(Float64(k)) for k in 1:n; init=0.0) for n in 0:n_max]

    lattice_props = Dict(
        "square"     => (z=4, n_sub=1),
        "triangular" => (z=6, n_sub=1),
        "honeycomb"  => (z=3, n_sub=2),
        "kagome"     => (z=4, n_sub=3)
    )
    haskey(lattice_props, lattice) ||
        error("Unknown lattice: $lattice. Choose from: $(keys(lattice_props))")

    props = lattice_props[lattice]
    n_sub = props.n_sub
    z     = props.z

    function energy(κ::Vector{T}, μ::T) where T
        if lattice == "square" || lattice == "triangular"
            return gutzwiller_energy_general(κ, t, U, μ, z, n_max, log_fact)
        elseif lattice == "honeycomb"
            return honeycomb_gutzwiller_energy(κ, t, U, μ, n_max, log_fact)
        else
            return kagome_gutzwiller_energy(κ, t, U, μ, n_max, log_fact)
        end
    end

    # Total particle count: sum of ⟨n_σ⟩ over sublattices × number of unit cells
    filling_total(κ::Vector{T}) where T =
        sum_sublattice_filling(κ, n_max, log_fact) * (N_sites / n_sub)

    # Inner optimization: minimize E(κ; μ_val) over log-space Gutzwiller params
    function solve_at_mu(μ_val::Float64)
        obj(log_κs::Vector{T}) where T = energy(log_κs, T(μ_val))
        res = optimize(obj, zeros(Float64, n_sub), LBFGS(),
                       Optim.Options(x_abstol=1e-12, f_reltol=1e-14,
                                     iterations=10_000);
                       autodiff=:forward)
        κ_log = Optim.minimizer(res)
        return κ_log, filling_total(κ_log), res
    end

    # Outer bisection: ⟨N⟩(μ) is monotonically increasing, so bisection is exact.
    # Bracket with bounds wide enough to cover [0, n_max * N_sites].
    scale  = abs(U) * n_max + z * abs(t)
    μ_lo   = -2.0 * scale
    μ_hi   =  2.0 * scale
    n_lo   = solve_at_mu(μ_lo)[2]
    n_hi   = solve_at_mu(μ_hi)[2]

    (n_lo ≤ N_target ≤ n_hi) ||
        error("Could not bracket N_target=$N_target in μ ∈ [$μ_lo, $μ_hi] " *
              "(got N=$(round(n_lo,digits=2)) .. $(round(n_hi,digits=2))). " *
              "Try increasing n_max.")

    for _ in 1:80
        μ_mid = (μ_lo + μ_hi) / 2
        n_mid = solve_at_mu(μ_mid)[2]
        n_mid < N_target ? (μ_lo = μ_mid) : (μ_hi = μ_mid)
        μ_hi - μ_lo < 1e-10 && break
    end

    μ_opt = (μ_lo + μ_hi) / 2
    κ_log_opt, _, result = solve_at_mu(μ_opt)
    κ_opt = exp.(κ_log_opt)

    sym_broken = n_sub > 1 && !all(isapprox(κ_opt[1], κ_opt[k], rtol=1e-4)
                                    for k in 2:n_sub)
    return (
        κ_opt      = κ_opt,
        μ_opt      = μ_opt,
        converged  = Optim.converged(result),
        sym_broken = sym_broken
    )
end

print("Benchmarking functions loaded succesfully.")