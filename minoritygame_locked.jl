using Plots, Random, Statistics

# Outputs
probability_locked = zeros(55,3)

# Parameters
κ = 100 # payoff differential sensitivity
M = 6 # memory length
num_games = 50 # number of games to average over
num_turns = 500 # number of turns
S = 2 # number of strategy tables per individual

X = [51,101,251,501,1001]
rng = MersenneTwister()

global count = 1
for x = 1:5
    N = X[x] # number of agents
    action = Array{Int,1}(undef,N) # actions taken: buy=1, sell=0
    for soc_learn = 1:11
        ℓˢ = (soc_learn-1)/10 # rate of social learning
        for game=1:num_games
            # Initialize game
            attendance = 0
            history = rand(rng,1:2^M)
            strategy_tables = rand(rng,0:1,S*N,2^M) # S strategy tables for the N players
            virtual_points = zeros(Int64,N,S) # virtual points for all players' strategy tables

            # Run simulation
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
                attendance = sum(action)-(N-sum(action))

                # Determine virtual payoffs and win rate
                for i=1:N
                    virtual_points[i,1] += (-1)^(minority+strategy_tables[2*i-1,history])
                    virtual_points[i,2] += (-1)^(minority+strategy_tables[2*i,history])
                end
                history = Int(mod(2*history,2^M) + minority + 1)

                # Social learning
                update_strategy_tables = strategy_tables
                update_virtual_points = virtual_points
                for i=1:N
                    if ℓˢ >= rand(rng)
                        # Find worst strategy and its points of focal player
                        worst_points,worst_strat = findmin(virtual_points[i,:])
                        # Select random other player and find its best strat and points
                        player = rand(filter(x -> x ∉ [i], 1:N))
                        best_points,best_strat = findmax(virtual_points[player,:])
                        if 1/(1+exp(κ*(worst_points-best_points))) > rand(rng)
                            update_strategy_tables[2*(i-1)+worst_strat,:] = strategy_tables[2*(player-1)+best_strat,:]
                            update_virtual_points[i,worst_strat] = virtual_points[player,best_strat]
                        end
                    end
                end
                strategy_tables = update_strategy_tables
                virtual_points = update_virtual_points
            end

            # Check if final turn had full or zero attendance
            if attendance == N || attendance == 0
                probability_locked[count,1] += 1/num_games
            end
        end
        probability_locked[count,2] = ℓˢ
        probability_locked[count,3] = x
        global count += 1
    end
end

probability_locked[:,3] = Int.(probability_locked[:,3])

x1 = probability_locked[1:11,2]
x2 = probability_locked[12:22,2]
x3 = probability_locked[23:33,2]
x4 = probability_locked[34:44,2]
x5 = probability_locked[45:55,2]

y1 = probability_locked[1:11,1]
y2 = probability_locked[12:22,1]
y3 = probability_locked[23:33,1]
y4 = probability_locked[34:44,1]
y5 = probability_locked[45:55,1]

z1 = Int.(probability_locked[1:11,3])
z2 = Int.(probability_locked[12:22,3])
z3 = Int.(probability_locked[23:33,3])
z4 = Int.(probability_locked[34:44,3])
z5 = Int.(probability_locked[45:55,3])

scatter([x1 x2 x3 x4 x5], [y1 y2 y3 y4 y5], markercolor=[z1 z2 z3 z4 z5],
xlims=(-0.05,1.05), ylims=(-0.05,1.05),
label=["N=51" "N=101" "N=251" "N=501" "N=1001"],
xlabel = "ℓˢ", ylabel="Probability", legend=:topleft,
thickness_scaling = 1.5)
savefig("locked_strats.pdf")

