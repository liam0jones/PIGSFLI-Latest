using Plots

function plot_kink!(plt, kink::Kink, M::Int64;color=:red,markercolor=:red,i=0,marker=:circle,markersize=4) 
    if kink.src == -1 || kink.dest == -1
        return nothing
    end
    if kink.src == kink.dest
        plot!(plt, [kink.src-0.3, kink.dest+0.3], [kink.tau, kink.tau], lw= kink.n > 0 ? kink.n : 0.5, color=color, linestyle= kink.n>0 ? :solid : :dot) 
    end
    if (kink.src == 0 && kink.dest == M-1) || (kink.src == M-1 && kink.dest == 0)
        plot!(plt, [0, -0.5], [kink.tau, kink.tau], lw= kink.n > 0 ? kink.n : 0.5, color=color, linestyle= kink.n>0 ? :solid : :dot) 
        plot!(plt, [M-1, M-1+0.5], [kink.tau, kink.tau], lw= kink.n > 0 ? kink.n : 0.5, color=color, linestyle= kink.n>0 ? :solid : :dot) 
    else 
        plot!(plt, [kink.src, kink.dest], [kink.tau, kink.tau], lw= kink.n > 0 ? kink.n : 0.5, color=color)
    end
    if i > 0 
        plot!(plt, [kink.src], [kink.tau], marker=marker, markersize=markersize, color=markercolor, label="$(kink.prev),$(kink.next),$(i-1)")  
    else
        plot!(plt, [kink.src], [kink.tau], marker=marker, markersize=markersize, color=markercolor, label="$(kink.prev),$(kink.next)")  
    end
    return nothing
end

function plot_path(sim_state::SimState, r::Int64)
    println("Plotting path for replica $r")
    plotlyjs()
    plt = plot(xlims=(0-0.5, sim_state.M-1+0.5), ylims=(0, sim_state.beta), xlabel="Site", ylabel="tau", legend=false, size=(2000, 1000), grid=false, xticks=0:sim_state.M-1 )

    for i = 0:sim_state.M-1
        plot!(plt, [i, i], [0, sim_state.beta], color=:black, lw=0.5, linestyle=:dash)
    end

    for (i,kink) in enumerate(sim_state.paths[r].data[1:sim_state.num_kinks[r]])
        color = :red
        markercolor = :red
        marker = :circle
        markersize = 4
        if i > sim_state.num_kinks[r] - 7 
            color = :blue
            markercolor = :blue  
        end
        if i-1 == sim_state.last_kinks[r][kink.src+begin] 
            markercolor = :purple
        end
        if i-1 == sim_state.head_idx[r]
            marker = :hexagon 
            markersize = 6
        end
        if i-1 == sim_state.tail_idx[r]
            marker = :diamond
            markersize = 6
        end
        plot_kink!(plt, kink, sim_state.M; color=color, markercolor=markercolor, i=i, marker=marker, markersize=markersize)
        if isnothing(find_partner(kink, sim_state,r))
            plot!(plt, [kink.src], [kink.tau], marker=:square, markersize=9,markerstrokewidth=1, color=:green, label="$(kink.prev),$(kink.next),$(i-1)",  markerfacecolor=:none)  
        end
    end

    for i in 0:length(sim_state.paths[r].data[1:sim_state.num_kinks[r]])-1 
        kink = sim_state.paths[r][i+begin]
        # kink to next kink 
        if kink.next != -1
            kink_next = sim_state.paths[r][kink.next+begin]
            start_x = kink.src
            start_y = kink.tau
            end_x = kink_next.src
            end_y = kink_next.tau 
            if end_y == -1
                end_y = sim_state.beta 
            end
            if start_y == -1
                start_y = sim_state.beta 
            end
            if start_x != -1 && end_x != -1
                plot!(plt, [start_x, end_x], [start_y, end_y], line = :arrow, color=:black)
                # poormans arrowhead
                plot!(plt, [start_x + 0.95*(end_x-start_x)], [start_y + 0.95*(end_y-start_y)], marker=:^ ,color=:black, markersize=4)
            end
        end 
    end 
    
    sorted_kinks = create_sorted_kink_vectors(sim_state, r)
    # look for broken links 
    broken_links, found_a_broken_link = find_broken_links(sim_state, r, sorted_kinks)
    if found_a_broken_link
        println("Found broken links: $(broken_links)")
        for broken_link in broken_links
            plot!(plt, [broken_link[1], broken_link[1]], [broken_link[2], broken_link[3]], color=:yellow, lw=4, linestyle=:dash)
        end
    end
    # look for duplicate links 
    duplicate_links, found_a_duplicate_link = find_duplicate_links(sim_state, r, sorted_kinks)
    if found_a_duplicate_link
        println("Found duplicate links: $(duplicate_links)")
        for dupe_link in duplicate_links
            plot!(plt, [dupe_link[1]], [dupe_link[2]], color=:green, markersize=9, marker=:octagon,alpha=0.5)
        end
    end

    Base.invokelatest(display, plt) 
    savefig("plot.html")
    return plt
end