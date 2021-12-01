using LightGraphs
using Printf
using CSV
using DataFrames
using SpecialFunctions
using LinearAlgebra
using GraphRecipes
using Plots
using TickTock
#4121.321856718706	29.886494781413937	112.9033949999986	3978.531966937293

# Write policy
function write_policy(policy, filename)
    open(filename, "w") do io
        for p in policy
            @printf(io, "%d\n", floor(Int8, p))
        end
    end
end

# Qlearning
mutable struct QLearning
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    gamma # discount
    Q # action value function
    alpha # learning rate
end

lookahead(model::QLearning, s, a) = model.Q[s,a]

function update!(model::QLearning, s, a, r, sp)
    gamma, Q, alpha = model.gamma, model.Q, model.alpha
    Q[s,a] += alpha*(r + gamma*maximum(Q[sp,:]) - Q[s,a])
    return model
end

# Sarsa
mutable struct Sarsa
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    gamma # discount
    Q # action value function
    α # learning rate
    l # most recent experience tuple (s,a,r)
end

lookahead(model::Sarsa, s, a) = model.Q[s,a]

function update!(model::Sarsa, s, a, r, sp)
    if model.l != nothing
        gamma, Q, α, l = model.gamma, model.Q, model.α, model.l
        model.Q[l.s,l.a] += α*(l.r + gamma*Q[s,a] - Q[l.s,l.a])
    end
    model.l = (s=s, a=a, r=r)
    return model
end

# SarsaLambda
mutable struct SarsaLambda
    S # state space (assumes 1:nstates)
    A # action space (assumes 1:nactions)
    gamma # discount
    Q # action value function
    N # trace
    α # learning rate
    lambda # trace decay rate
    l # most recent experience tuple (s,a,r)
end

lookahead(model::SarsaLambda, s, a) = model.Q[s,a]

function update!(model::SarsaLambda, s, a, r, sp)
    if model.l != nothing
        gamma, lambda, Q, α, l = model.gamma, model.lambda, model.Q, model.α, model.l
        model.N[l.s,l.a] += 1
        δ = l.r + gamma*Q[s,a] - Q[l.s,l.a]
        for s in model.S
            for a in model.A
                model.Q[s,a] += α*δ*model.N[s,a]
                model.N[s,a] *= gamma*lambda
            end
        end
    else
        model.N[:,:] .= 0.0
    end
    model.l = (s=s, a=a, r=r)
    return model
end

# Epsilon Greedy Exploration
mutable struct EpsilonGreedyExploration
    epsilon # probability of random action
    alpha # exploration decay factor
end

function (pi::EpsilonGreedyExploration)(model, s)
    A, epsilon = model.A, pi.epsilon
    if rand() < epsilon
        return rand(A)
    end
    Q(s,a) = lookahead(model, s, a)
    return argmax(a->Q(s,a), A)
end

function compute(infile, outfile)    
    df = DataFrame(CSV.File(infile)) # load data

    tick()
    gamma = 0.95 # discount factor
    print("S may be defined")
    if infile[1]=='d'
        # For small dataset
        print("S is defined")
        S = collect(1:28634)
        A = collect(1:20)
    elseif infile[1]=='v'
        # For medium dataset
        S = collect(1:55322)
        A = collect(1:20)
    elseif infile[1]=='a'
        # For medium dataset
        S = collect(1:109532)
        A = collect(1:20)
    end
    Q = zeros(length(S), length(A))
    alpha = 0.01 # learning rate
    model = QLearning(S, A, gamma, Q, alpha)
    # model = Sarsa(S, A, gamma, Q, alpha, nothing)
    # model = SarsaLambda(S, A, gamma, Q, Q, alpha, 0.5, nothing)

    n_episodes = 0
    converge_thre = 0.001
    convergence = 100
    while convergence > converge_thre && n_episodes<2000
        n_episodes += 1
        Qp = copy(model.Q)
        for j in 1:length(df.s)
            update!(model, df.s[j], df.a[j], df.r[j], df.sp[j])
        end
        convergence = norm(Qp-model.Q,2)
    end
    tock()

    epsilon = 0.01 # probability of random action
    alpha1 = 1.0
    pi = EpsilonGreedyExploration(epsilon,alpha1)
    policy = zeros(length(S),1)
    for i in 1:length(S)
        policy[i] = pi(model,S[i])
    end
    print(n_episodes)

    write_policy(policy, outfile)
end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>.gph")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

compute(inputfilename, outputfilename)
