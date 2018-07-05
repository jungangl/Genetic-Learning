using Parameters, NLsolve, StatsBase, Plots, Distributions, QuantEcon
@with_kw type SL_model
    ## Structural Parameters
    β::Float64 = 0.99
    σ::Float64 = 2.
    κ::Float64 = 0.02
    i_star::Float64 = inv(β) - 1
    ϕ_π::Float64 = 1.5
    ϕ_y::Float64 = 0.125
    ## rn_t Process
    rn_H::Float64 = 0.0093
    rn_L::Float64 = -0.0093
    rn::Vector{Float64} = [rn_H; rn_L]
    ρ_H::Float64 = 0.675
    ρ_L::Float64 = 0.675
    P::Matrix{Float64} = [ρ_H 1 - ρ_H; 1 - ρ_L ρ_L]
    mc::MarkovChain = MarkovChain(P)
    ## Social Learning Algorithm
    T0::Int64 = 100
    T1::Int64 = 1_000
    N::Int64 = 300
    pc::Float64 = 0.1
    pm::Float64 = 0.1
    Σ::Matrix{Float64} = diagm([0.0123; 0.0103; 0.0123; 0.0103])
    Σ_m::Matrix{Float64} = diagm([0.0123; 0.0103; 0.0123; 0.0103])
end



## Construct matrices from the structural parameters
function build_matrices(para)
    @unpack κ, σ, β, ϕ_y, ϕ_π, ρ_H, ρ_L, rn_H, rn_L = para
    G = [1. .0; -κ 1.]
    a = [.0 .0]'
    b = inv(G) * [1 inv(σ); .0 β]
    c = inv(G) * [-inv(σ) .0]'
    d = -c
    Φ_H = [ϕ_y ϕ_π .0 .0]
    Φ_L = [.0 .0 ϕ_y ϕ_π]
    Φ = [Φ_H; Φ_L]
    Γ_H = [ρ_H * eye(2)  (1 - ρ_H) * eye(2)]
    Γ_L = [(1 - ρ_L) * eye(2)  ρ_L * eye(2)]
    r_n = [rn_H rn_L]'
    return a, b, c, d, Φ_H, Φ_L, Φ, Γ_H, Γ_L, r_n
end



## Solve for the REE (never binds) - RE_NB
function RE_NB(para)
    @unpack i_star = para
    a, b, c, d, Φ_H, Φ_L, Φ, Γ_H, Γ_L, r_n = build_matrices(para)
    A_nb = [a; a]
    B_nb = [b * Γ_H + c * Φ_H;
            b * Γ_L + c * Φ_L]
    D = [d zeros(d); zeros(d) d]
    z_nb = inv(eye(B_nb) - B_nb) * (A_nb + D * r_n)
    if !prod(Φ * z_nb .> -i_star)
        z_nb = NaN
    end
    return z_nb
end



## Solve for the REE (always binds) - RE_AB
function RE_AB(para)
    @unpack i_star = para
    a, b, c, d, Φ_H, Φ_L, Φ, Γ_H, Γ_L, r_n = build_matrices(para)
    A_ab = [a - c * i_star; a - c * i_star]
    B_ab = [b * Γ_H; b * Γ_L]
    D = [d zeros(d); zeros(d) d]
    z_ab = inv(eye(B_ab) - B_ab) * (A_ab + D * r_n)
    if !prod(Φ * z_ab .<= -i_star)
        z_ab = NaN
    end
    return z_ab
end



## Solve for the REE (ocasionally binds) - RE_OB
function RE_OB(para)
    @unpack i_star = para
    a, b, c, d, Φ_H, Φ_L, Φ, Γ_H, Γ_L, r_n = build_matrices(para)
    A_ob = [a; a - c * i_star]
    B_ob = [b * Γ_H + c * Φ_H; b * Γ_L]
    D = [d zeros(d); zeros(d) d]
    z_ob = inv(eye(B_ob) - B_ob) * (A_ob + D * r_n)
    if ((Φ_H * z_ob)[1] <= -i_star) || ((Φ_L * z_ob)[1] > -i_star)
        z_ob = NaN
    end
    return z_ob
end



## Define a function that maps a vector of PLM to a conditional ALM
## The vector of PLM is defined as z_i = Matrix{Float64}(4, N)
## The ALM is define as z = Vector{Float64}(4) conditional on the rn for that period
function PLM_to_ALM(para, z_i, rn)
    @unpack ρ_H, ρ_L, i_star, rn_H, rn_L, ϕ_y, ϕ_π = para
    a, b, c, d, Φ_H, Φ_L, Φ, Γ_H, Γ_L, r_n = build_matrices(para)
    z_i_H = z_i[1:2, :]
    z_i_L = z_i[3:4, :]
    Ez_i = (rn == rn_H) * (ρ_H * z_i_H + (1 - ρ_H) * z_i_L) +
           (rn == rn_L) * ((1 - ρ_L) * z_i_H + ρ_L * z_i_L)
    Ez = mean(Ez_i, 2)
    z = inv(eye(2) - c * [ϕ_y ϕ_π]) * (a + b * Ez + d * rn)
    if dot([ϕ_y ϕ_π], z) < -i_star
        z = a + b * Ez + c * (-i_star) + d * rn
    end
    return z
end



## Social Learning Algorithm Part 1) - Crossover
function crossover!(para, z_i)
    @unpack pc, N = para
    pair_num = div(N, 2)
    group1_indx = sample(1:N, pair_num; replace = false)
    group2_indx = sample(setdiff(1:N, group1_indx), pair_num)
    group_indx = hcat(group1_indx, group2_indx)
    crossover_indx = find(rand(pair_num) .< pc)
    for pair in crossover_indx
        cross_bools = rand(Bool, 4)
        z_temp = deepcopy(z_i[:, group_indx[pair, :]])
        z_swap = hcat(z_temp[:, 2], z_temp[:, 1])
        z_i[:, group_indx[pair, :]] = cross_bools .* z_swap + .!cross_bools .* z_temp
    end
end



## Social Learning Algorithm Part 2) - Mutation
function mutation!(para, z_i)
    @unpack pm, N, Σ_m = para
    dist = MvNormal(zeros(4), eye(Σ_m))
    mutate_indx = find(rand(N) .< pm)
    for i in mutate_indx
        z_i[:, i] = z_i[:, i] + Σ_m * rand(dist)
    end
end



## Social Learning Algorithm Part 3) - Tournament Selection
function tournament(para, z_i, history)
    @unpack N, rn_H, rn_L, ρ_H, ρ_L = para
    ## Unpack the history
    y_h, π_h, rn_h = history
    T_forc = length(y_h) - 1
    ## Preallocation for speed
    z_i′ = zeros(z_i)
    agent1 = 1
    agent2 = 1
    y_scores = zeros(T_forc)
    π_scores = zeros(T_forc)
    ŷ, π̂, F_y, F_π, yH, πH, yL, πL = zeros(8)
    ## Preallocation for matches required to attend Tournament
    matches = Vector{Vector{Int64}}(N)
    matches_unique = zeros(Int64, N, 2)
    matches_tie_indx = zeros(Bool, N)
    ## Pick matches for tournament
    same_agent = true
    for i in 1:N
        same_agent = true
        while same_agent
            agent1, agent2 = rand(1:N, 2)
            if agent1 != agent2
                same_agent = false
            end
            matches[i] = [agent1, agent2]
            matches_unique[i, :] = [agent1, agent2]
        end
    end
    ## Eliminate macthes that are tied (agents with same beliefs)
    tie_indx = 0
    for (i, m) in enumerate(matches)
        if z_i[:, m[1]] == z_i[:, m[2]]
            tie_indx += 1
            matches_tie_indx[i] = true
            z_i′[:, tie_indx] = z_i[:, m[1]]
        end
    end
    ## Pick out the unique agents for the tournament
    match_nontie_indx = .!matches_tie_indx
    competitors = unique(matches_unique[match_nontie_indx, :])
    comp_num = length(competitors)
    #println(comp_num)
    ## Compute the fitness scores for each agent required for tournament
    y_fitness = -Inf * ones(N)
    π_fitness = -Inf * ones(N)
    for i in competitors
        yH, πH, yL, πL = z_i[:, i]
        for t in 1:T_forc
            if rn_h[t]
                ŷ = ρ_H * yH + (1 - ρ_H) * yL
                π̂ = ρ_H * πH + (1 - ρ_H) * πL
            else
                ŷ = (1 - ρ_L) * yH + ρ_L * yL
                π̂ = (1 - ρ_L) * πH + ρ_L * πL
            end
            y_scores[t] = (ŷ - y_h[t + 1]) ^ 2
            π_scores[t] = (π̂ - π_h[t + 1]) ^ 2
        end
        y_fitness[i] = -mean(y_scores)
        π_fitness[i] = -mean(π_scores)
    end
    ## Tournament!
    for (i, match_indx) in enumerate(find(match_nontie_indx))
        agent1, agent2 = matches[match_indx]
        F_y1 = y_fitness[agent1]
        F_π1 = π_fitness[agent1]
        F_y2 = y_fitness[agent2]
        F_π2 = π_fitness[agent2]
        if F_y1 > F_y2
            z_i′[[1; 3], tie_indx + i] = z_i[[1; 3], agent1]
        else
            z_i′[[1; 3], tie_indx + i] = z_i[[1; 3], agent2]
        end
        ## inflation tournament
        if F_π1 > F_π2
            z_i′[[2; 4], tie_indx + i] = z_i[[2; 4], agent1]
        else
            z_i′[[2; 4], tie_indx + i] = z_i[[2; 4], agent2]
        end
    end
    return z_i′
end



## Save the data as it goes
function write_data(path, t_start, t_end, z̄_t, σz_t, y_t, π_t)
    writedlm("../data/$(path)/z_bar/$(t_start)-$(t_end).csv", z̄_t, ',')
    writedlm("../data/$(path)/z_sigma/$(t_start)-$(t_end).csv", σz_t, ',')
    writedlm("../data/$(path)/y/$(t_start)-$(t_end).csv", y_t, ',')
    writedlm("../data/$(path)/pi/$(t_start)-$(t_end).csv", π_t, ',')
end



## Read all data saved in the data folder
function read_data(path, T0, T1, save_gap)
    z̄_t, σz_t =[zeros(T0 + T1, 4) for i in 1:2]
    y_t, π_t = [zeros(T0 + T1) for i in 1:2]
    z̄_t[1:T0, :] = readdlm("../data/$(path)/z_bar/1-$(T0).csv", ',')
    σz_t[1:T0, :] = readdlm("../data/$(path)/z_sigma/1-$(T0).csv", ',')
    y_t[1:T0] = readdlm("../data/$(path)/y/1-$(T0).csv", ',')
    π_t[1:T0] = readdlm("../data/$(path)/pi/1-$(T0).csv", ',')
    for i in 1:(T1 ÷ save_gap)
        t_start = T0 + (i - 1) * save_gap + 1
        t_end = T0 + i * save_gap
        z̄_t[t_start:t_end, :] = readdlm("../data/$(path)/z_bar/$(t_start)-$(t_end).csv", ',')
        σz_t[t_start:t_end, :] = readdlm("../data/$(path)/z_sigma/$(t_start)-$(t_end).csv", ',')
        y_t[t_start:t_end] = readdlm("../data/$(path)/y/$(t_start)-$(t_end).csv", ',')
        π_t[t_start:t_end] = readdlm("../data/$(path)/pi/$(t_start)-$(t_end).csv", ',')
    end
    return z̄_t, σz_t, y_t, π_t
end



## Simulate Social Learning
## case 1) initialize the economy with T0 periods of RE_AB for all agents
## case 2) initialize the economy with T0 periods of RE_NB for all agents
## Boolean variable mean_preserve controls the perturbation at time period T0 + 1
## if mean_preserve == true, then do mean preserving spread as the paper
## if mean_preserve == false, then impose a degenerate initial distribution
## of beliefs on every agent (two kinds of beliefs)
## case 1) every agent revert back to RE_NB at period T0 + 1
## case 2) every agent revert back to RE_AB at period T0 + 1
function simulate_SL(para, path, save_gap; case = 1, mean_preseve = true, finite_mem = false, mem_size = 1000)
    @unpack T0, T1, Σ, N, rn, mc, rn_H, rn_L = para
    #------------------------initialize beliefs------------------------#
    ## Initialize objects of interest
    z̄_t = zeros(T0 + T1, 4)
    σz_t = zeros(T0 + T1, 4)
    ## Initialize the beliefs from time 0 to time T0
    z_init = case * RE_AB(para) + (1 - case) * RE_NB(para)
    z̄_t[1:T0, :] = z_init' .* ones(T0)
    z_i = z_init .* ones(1, N)
    ## Perturb the bliefs at time T0 + 1
    ## If it is a mean preserving pertubation
    if mean_preseve
        dist = MvNormal(zeros(4), eye(Σ))
        ν_vec = rand(dist, N)
        z_i = z_i + Σ * ν_vec
    ## If it is not a mean preserving perturbation -
    ## Instead, revert to a degenerate belief distribution
    ## opposite to the belief initialization
    else
        z_perturb = (1 - case) * RE_AB(para) + case * RE_NB(para)
        z_i = z_perturb .* ones(1, N)
    end
    z̄_t[T0 + 1, :] = mean(z_i, 2)
    σz_t[T0 + 1, :] = std(z_i, 2)
    #-----------------------initialize data history----------------------#
    ## Initialize data from time 1 to time T0
    y_t, π_t = [zeros(T0 + T1) for _ in 1:2]
    rn_indices = simulate(mc, T0 + T1)
    rn_t = rn[rn_indices]
    rn_high_t = rn_indices .== 1
    y_t[1:T0] = (z_init[[1, 3]])[rn_indices[1:T0]]
    π_t[1:T0] = (z_init[[2, 4]])[rn_indices[1:T0]]
    t_start = 1
    t_end = T0
    write_data(path, t_start, t_end, z̄_t[t_start:t_end, :], σz_t[t_start:t_end, :], y_t[t_start:t_end], π_t[t_start:t_end])
    ## Initialize data for time T0 + 1
    y_t[T0 + 1], π_t[T0 + 1] = PLM_to_ALM(para, z_i, rn_t[T0 + 1])
    #-----------Simulate the periods from T0 + 2 to T0 + T1--------------#
    for t in T0 + 1:T0 + T1 - 1
        if mod(t, 1_000) == 0 println(t) end
        crossover!(para, z_i)
        mutation!(para, z_i)
        t_init = !finite_mem * 1 + finite_mem * max(1, t - mem_size + 1)
        history = y_t[t_init:t], π_t[t_init:t], rn_high_t[t_init:t]
        z_i = tournament(para, z_i, history)
        z̄_t[t + 1, :] = mean(z_i, 2)
        σz_t[t + 1, :] = std(z_i, 2)
        y_t[t + 1], π_t[t + 1] = PLM_to_ALM(para, z_i, rn_t[t + 1])
        if mod(t - T0 + 1, save_gap) == 0
            t_start = t + 2 - save_gap
            t_end = t + 1
            write_data(path, t_start, t_end, z̄_t[t_start:t_end, :], σz_t[t_start:t_end, :], y_t[t_start:t_end], π_t[t_start:t_end])
        end
    end
    return z̄_t, σz_t, y_t, π_t
end



## Define the function that plots all time series
function plot_all(para, z̄_t, σz_t, y_t, π_t)
    gr()
    z_ab = RE_AB(para)
    z_nb = RE_NB(para)
    z̄_upper_t = z̄_t .+ σz_t
    z̄_lower_t = z̄_t .- σz_t
    titles = hcat("y_H", "pi_H", "y_L", "pi_L")
    pz = plot(size = (1000, 2000), title = titles, grid = false, layout = (4, 1))#, ylims = (-0.04, 0.02))
    for i in 1:4
        plot!(pz, z̄_t[:, i], label = "", color = :red, lw = 1.5, subplot = i)
        plot!(pz, z̄_upper_t[:, i] , label = "", color = :blue, lw = 1, subplot = i)
        plot!(pz, z̄_lower_t[:, i], label = "", color = :blue, lw = 1, subplot = i)
        plot!(x -> z_nb[i], label = "RENB", ls = :dash, lw = 1.5, subplot = i)
        plot!(x -> z_ab[i], label = "REAB", ls = :dash, color = :black, lw = 1.5, subplot = i)
        if i != 1 plot!(pz, legend = false, subplot = i) end
    end
    py = scatter(y_t, title = "output y", grid = false, label = "", alpha = 0.5, s = :dash)
    pπ = scatter(π_t, title = "inflation pi", grid = false, label = "", alpha = 0.5, s = :dash)
    return pz, py, pπ
end



## Housekeeping
path = "RE-AB/IFM"
save_gap = 1_000
T0 = 100
T1 = 10_000
para = SL_model(T0 = T0, T1 = T1, N = 300)
@time z̄_t, σz_t, y_t, π_t = simulate_SL(para, path, save_gap; case = 1, finite_mem = false, mem_size = 1_000)
#z̄_t1, σz_t1, y_t1, π_t1 = read_data(path, T0, T1, save_gap)

#periods = 1:(para.T0 + 300)
#pz, py, pπ = plot_all(para, z̄_t[:, periods], σz_t[:, periods], y_t[periods], π_t[periods])

#pz, py, pπ = plot_all(para, z̄_t, σz_t, y_t, π_t)
#savefig(pz, "../figures/$(path)/z.pdf")
#savefig(py, "../figures/$(path)/y.pdf")
#savefig(pπ, "../figures/$(path)/pi.pdf")
