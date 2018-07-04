using Parameters, NLsolve, StatsBase, Plots, Distributions, QuantEcon, Devectorize
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


@with_kw type Agent
    z::Vector{Float64} = zeros(4)
    F_y::Float64 = -Inf
    F_π::Float64 = -Inf
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
    return vec(z_nb)
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
    return vec(z_ab)
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
        return NaN
    end
    return vec(z_ob)
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
function crossover!(para, agents, history)
    @unpack pc, N = para
    pair_num = div(N, 2)
    group1_indx = sample(1:N, pair_num; replace = false)
    group2_indx = sample(setdiff(1:N, group1_indx), pair_num)
    group_indx = hcat(group1_indx, group2_indx)
    crossover_indices = find(rand(pair_num) .< pc)
    for pair in crossover_indices
        cross_bools = rand(Bool, 4)
        z_temp = deepcopy(z_i[:, group_indx[pair, :]])
        z_swap = hcat(z_temp[:, 2], z_temp[:, 1])
        z_i[:, group_indx[pair, :]] = cross_bools .* z_swap + .!cross_bools .* z_temp
        update_fit_indi!(para, agents[group1_indx[pair]], history)
        update_fit_indi!(para, agents[group2_indx[pair]], history)
    end
end


## Social Learning Algorithm Part 2) - Mutation
function mutation!(para, agents, history)
    @unpack pm, N, Σ_m = para
    dist = MvNormal(zeros(4), eye(Σ_m))
    mutate_indx = find(rand(N) .< pm)
    for i in mutate_indx
        agents[i].z = agents[i].z + Σ_m * rand(dist)
        update_fit_indi!(para, agents[i], history)
    end
end


## Social Learning Algorithm Part 3) - Tournament Selection
function tournament!(N, agents)
    agents_temp = deepcopy(agents)
    same_agent = true
    indx1, indx2 = ones(Int64, 2)
    ## N matches with replacement
    ## tournament for each match
    for i in 1:N
        same_agent = true
        while same_agent
            indx1, indx2 = rand(1:N, 2)
            if indx1 != indx2
                same_agent = false
            end
        end
        ## output tournament
        if agents_temp[indx1].F_y > agents_temp[indx2].F_y
            agents[indx2].z[[1; 3]] = agents_temp[indx1].z[[1; 3]]
        else
            agents[indx1].z[[1; 3]] = agents_temp[indx2].z[[1; 3]]
        end
        ## inflation tournament
        if agents_temp[indx1].F_π > agents_temp[indx2].F_π
            agents[indx2].z[[2; 4]] = agents_temp[indx1].z[[2; 4]]
        else
            agents[indx1].z[[2; 4]] = agents_temp[indx2].z[[2; 4]]
        end
    end
end


## Extract the belief matrix from an agent array
function extract_belief(agents, N)
    z_i = zeros(4, N)
    for i in 1:N
        z_i[:, i] = agents[i].z
    end
    return z_i
end


## Get the history of data depending on if it is finite memory
function get_history(finite_mem, mem_size, t, y_t, π_t, rn_high_t)
    t_init = 1
    if finite_mem
        t_init = max(1, t - mem_size + 1)
    end
    history = y_t[t_init:t], π_t[t_init:t], rn_high_t[t_init:t]
    history_length = t - t_init + 1
    return history, history_length
end


## Update the fitness based on the history for each agent
function update_fit_indi!(para, agent, history)
    @unpack rn_H, rn_L, ρ_H, ρ_L, N = para
    yH, πH, yL, πL = agent.z
    y_h, π_h, rn_h = history
    T_forc = length(y_h) - 1
    π_scores = zeros(T_forc)
    y_scores = zeros(T_forc)
    ŷ, π̂ = zeros(2)
    for t in 1:T_forc
        if rn_h[t] == true
            ŷ = ρ_H * yH + (1 - ρ_H) * yL
            π̂ = ρ_H * πH + (1 - ρ_H) * πL
        else
            ŷ = (1 - ρ_L) * yH + ρ_L * yL
            π̂ = (1 - ρ_L) * πH + ρ_L * πL
        end
        y_scores[t] = (ŷ - y_h[t + 1]) ^ 2
        π_scores[t] = (π̂ - π_h[t + 1]) ^ 2
    end
    agent.F_y = -mean(y_scores)
    agent.F_π = -mean(π_scores)
end


## Update the fitness recursively when new data arrive
function update_fit_all!(para, agents, y_new, π_new, rn, history_length)
    @unpack rn_H, rn_L, ρ_H, ρ_L, N = para
    yH, πH, yL, πL, ŷ, π̂ = zeros(6)
    τ = history_length - 1
    for i in 1:N
        yH, πH, yL, πL = agents[i].z
        if rn == true
            ŷ = ρ_H * yH + (1 - ρ_H) * yL
            π̂ = ρ_H * πH + (1 - ρ_H) * πL
        else
            ŷ = (1 - ρ_L) * yH + ρ_L * yL
            π̂ = (1 - ρ_L) * πH + ρ_L * πL
        end
        agents[i].F_y = -( (-agents[i].F_y) * τ + (y_new - ŷ) ^ 2 ) / (τ + 1)
        agents[i].F_π = -( (-agents[i].F_π) * τ + (π_new - π̂) ^ 2 ) / (τ + 1)
    end
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
function simulate_SL(para; case = 1, mean_preseve = true, finite_mem = false, mem_size = 100)
    @unpack T0, T1, Σ, N, rn, mc, rn_H, rn_L = para
    #------------------------initialize beliefs------------------------#
    ## Initialize objects of interest
    agents = [Agent() for i in 1:N]
    z̄_t = zeros(4, T0 + T1)
    σz_t = zeros(4, T0 + T1)
    ## Initialize the beliefs from time 0 to time T0
    z_init = case * RE_AB(para) + (1 - case) * RE_NB(para)
    z̄_t[:, 1:T0] = z_init .* ones(1, T0)
    for i in 1:N
        agents[i].z = z_init
    end
    ## Perturb the bliefs at time T0 + 1
    ## If it is a mean preserving pertubation
    if mean_preseve
        dist = MvNormal(zeros(4), eye(Σ))
        for i in 1:N
            agents[i].z = agents[i].z + Σ * rand(dist)
        end
    ## If it is not a mean preserving perturbation -
    ## Instead, revert to a degenerate belief distribution
    ## opposite to the belief initialization
    else
        z_perturb = (1 - case) * RE_AB(para) + case * RE_NB(para)
        for i in 1:N
            agents[i].z = z_perturb
        end
    end
    z_i = extract_belief(agents, N)
    z̄_t[:, T0 + 1] = mean(z_i, 2)
    σz_t[:, T0 + 1] = std(z_i, 2)
    #-----------------------initialize data history----------------------#
    ## Initialize data from time 1 to time T0
    y_t, π_t = [zeros(T0 + T1) for _ in 1:2]
    rn_indices = simulate(mc, T0 + T1)
    rn_t = rn[rn_indices]
    rn_high_t = rn_indices .== 1
    y_t[1:T0] = (z_init[[1, 3]])[rn_indices[1:T0]]
    π_t[1:T0] = (z_init[[2, 4]])[rn_indices[1:T0]]
    ## Initialize data for time T0 + 1
    y_t[T0 + 1], π_t[T0 + 1] = PLM_to_ALM(para, z_i, rn_t[T0 + 1])
    ## Update the fitness for each agent at time t
    history, history_length = get_history(finite_mem, mem_size, T0 + 1, y_t, π_t, rn_high_t)
    for i in 1:N
        update_fit_indi!(para, agents[i], history)
    end
    #-----------Simulate the periods from T0 + 2 to T0 + T1--------------#
    for t in T0 + 1:T0 + T1 - 1
        if mod(t, 100) == 0 println(t) end
        history, history_length = get_history(finite_mem, mem_size, t, y_t, π_t, rn_high_t)
        crossover!(para, agents, history)
        mutation!(para, agents, history)
        tournament!(N, agents)
        z_i = extract_belief(agents, N)
        z̄_t[:, t + 1] = mean(z_i, 2)
        σz_t[:, t + 1] = std(z_i, 2)
        y_new, π_new = PLM_to_ALM(para, z_i, rn_t[t + 1])
        y_t[t + 1], π_t[t + 1] = y_new, π_new
        update_fit_all!(para, agents, y_new, π_new, rn_high_t[t], history_length)
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
        plot!(pz, z̄_t[i, :], label = "", color = :red, lw = 1.5, subplot = i)
        plot!(pz, z̄_upper_t[i, :] , label = "", color = :blue, lw = 1, subplot = i)
        plot!(pz, z̄_lower_t[i, :], label = "", color = :blue, lw = 1, subplot = i)
        plot!(x -> z_nb[i], label = "RENB", ls = :dash, lw = 1.5, subplot = i)
        plot!(x -> z_ab[i], label = "REAB", ls = :dash, color = :black, lw = 1.5, subplot = i)
        if i != 1 plot!(pz, legend = false, subplot = i) end
    end
    py = scatter(y_t, title = "output y", grid = false, label = "", alpha = 0.5, s = :dash)
    pπ = scatter(π_t, title = "inflation pi", grid = false, label = "", alpha = 0.5, s = :dash)
    return pz, py, pπ
end


## Write all data generated by the code
function write_all(z̄_t, σz_t, y_t, π_t; str = "")
    writedlm("../data/$(str)/z_bar.csv", z̄_t, ',')
    writedlm("../data/$(str)/z_sigma.csv", σz_t, ',')
    writedlm("../data/$(str)/y.csv", y_t, ',')
    writedlm("../data/$(str)/pi.csv", π_t, ',')
end


## Read all data saved in the data folder
function read_all(str)
    z̄_t = readdlm("../data/$(str)/z_bar.csv", ',')
    σz_t = readdlm("../data/$(str)/z_sigma.csv", ',')
    y_t = readdlm("../data/$(str)/y.csv", ',')
    π_t = readdlm("../data/$(str)/pi.csv", ',')
    return z̄_t, σz_t, y_t, π_t
end


## Housekeeping
para = SL_model(T0 = 100, T1 = 1000, pm = 0.1, pc = 0.1, N = 300)
@time z̄_t, σz_t, y_t, π_t = simulate_SL(para; mean_preseve = true, case = 1, finite_mem = false)

path = "RE-AB/IFM/T3"
#z̄_t, σz_t, y_t, π_t = read_all(path)
#write_all(z̄_t, σz_t, y_t, π_t; str = path)
z̄_t1, σz_t1, y_t1, π_t1 = read_all(path)
pz1, py1, pπ1 = plot_all(para, z̄_t1, σz_t1, y_t1, π_t1)
pz, py, pπ = plot_all(para, z̄_t, σz_t, y_t, π_t)
#periods = 1:(para.T0 + 300)
#pz, py, pπ = plot_all(para, z̄_t[:, periods], σz_t[:, periods], y_t[periods], π_t[periods])
#pz, py, pπ = plot_all(para, z̄_t, σz_t, y_t, π_t)

#savefig(pz, "../figures/$(path)/z.pdf")
#savefig(py, "../figures/$(path)/y.pdf")
#savefig(pπ, "../figures/$(path)/pi.pdf")
