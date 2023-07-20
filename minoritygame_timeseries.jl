using Plots, RCall, Statistics

# Parameters
κ = 100 # payoff differential sensitivity
ℓⁱ = 0.1 # rate of individual learning
ℓˢ = 0.1 # rate of social learning
M = 6 # memory length
N = 1001 # number of agents
num_turns = 500 # number of turns
S = 2 # number of strategy tables per individual

rng = MersenneTwister()

# Variables
action = Array{Int,1}(undef,N) # actions taken: buy=1, sell=0
history = Array{Int,1}(undef,M) # history of the winning strategy for the last M turns

# Outputs
attendance = Array{Int,1}(undef,num_turns)
num_winners = Array{Int,1}(undef,num_turns)

# Initialize game
history = rand(1:2^M)
strategy_tables = rand(0:1,S*N,2^M) # S strategy tables for the N players
update_strategy_tables = rand(rng,0:1,S*N,2^M) # for updating strategy tables
virtual_points = zeros(Int64,N,S) # virtual points for all players' strategy tables
update_virtual_points = zeros(Int64,N,S) # for udpating virtual points

for turn=1:num_turns

    # Actions taken
    for i=1:N
        best_strat = 2*(i-1) + findmax(virtual_points[i,:])[2]
        action[i] = strategy_tables[best_strat,history]
    end
    cur_attendance = sum(action)
    if cur_attendance <= (N-1)/2
        minority = 1
    else
        minority = 0
    end
    attendance[turn] = sum(action)-(N-sum(action))
    num_winners[turn] = minimum([cur_attendance,N-cur_attendance])

    # Determine virtual payoffs and win rate
    for i=1:N
        virtual_points[i,1] += (-1)^(minority+strategy_tables[2*i-1,history])
        virtual_points[i,2] += (-1)^(minority+strategy_tables[2*i,history])
    end
    history = Int(mod(2*history,2^M) + minority + 1)

    # # Individual learning
    # for i=1:N
    #     if ℓⁱ > rand(rng)
    #         new_strat = rand(rng,0:1)
    #         strategy_tables[i+new_strat,:] = rand(rng,0:1,2^M)
    #         virtual_points[i,new_strat+1] = 0
    #     end
    # end

    # # Social learning
    # update_strategy_tables = strategy_tables
    # update_virtual_points = virtual_points
    # for i=1:N
    #     if ℓˢ > rand(rng)
    #         # Find worst strategy and its points of focal player
    #         worst_points,worst_strat = findmin(virtual_points[i,:])
    #         # Select random other player and find its best strat and points
    #         player = rand(filter(x -> x ∉ [i], 1:N))
    #         best_points,best_strat = findmax(virtual_points[player,:])
    #         if 1/(1+exp(κ*(worst_points-best_points))) > rand(rng)
    #             update_strategy_tables[2*(i-1)+worst_strat,:] = strategy_tables[2*(player-1)+best_strat,:]
    #             update_virtual_points[i,worst_strat] = virtual_points[player,best_strat]
    #         end
    #     end
    # end
    # strategy_tables = update_strategy_tables
    # virtual_points = update_virtual_points

end

plot(attendance,size = (500, 200),ylims=(-1100,1100),xlabel = "Time",ylabel="A",
legend=false,thickness_scaling = 1.5)
savefig("ts_M6_N1001_S2.pdf")
