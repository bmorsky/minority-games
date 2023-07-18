using Plots, Random, Statistics

# Outputs
avg_attendance_volatility = zeros(11,11)

# Parameters
κ = 100 # payoff differential sensitivity
M = 6 # memory length
N = 1001 # number of players
num_games = 20 # number of games to average over
num_turns = 500 # number of turns
S = 2 # number of strategy tables per individual

# Variables
attendance = Array{Int,1}(undef,num_turns)

# Random numbers
rng = MersenneTwister()

for ind_learn = 0:10
    ℓⁱ = ind_learn/10 # rate of individual learning
    action = Array{Int,1}(undef,N) # actions taken: buy=1, sell=0
    for soc_learn = 0:10
        ℓˢ = soc_learn/10 # rate of social learning
        for game=1:num_games
            # Initialize game
            history = rand(rng,1:2^M)
            strategy_tables = rand(rng,0:1,S*N,2^M) # S strategy tables for the N players
            update_strategy_tables = rand(rng,0:1,S*N,2^M) # for updating strategy tables
            virtual_points = zeros(Int64,N,S) # virtual points for all players' strategy tables
            update_virtual_points = zeros(Int64,N,S) # for udpating virtual points
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
                attendance[turn] = sum(action)-(N-sum(action))

                # Determine virtual payoffs and win rate
                for i=1:N
                    virtual_points[i,1] += (-1)^(minority+strategy_tables[2*i-1,history])
                    virtual_points[i,2] += (-1)^(minority+strategy_tables[2*i,history])
                end
                history = Int(mod(2*history,2^M) + minority + 1)

                # Individual learning
                for i=1:N
                    if ℓⁱ > rand(rng)
                        new_strat = rand(rng,0:1)
                        strategy_tables[i+new_strat,:] = rand(rng,0:1,2^M)
                        virtual_points[i,new_strat+1] = 0
                    end
                end

                # Social learning
                for i=1:N
                    if ℓˢ > rand(rng)
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
            avg_attendance_volatility[ind_learn+1,soc_learn+1] += var(attendance)/(num_games*N)
        end
    end
end

heatmap(0:0.1:1, 0:0.1:1, avg_attendance_volatility, c=:thermal, xlabel="ℓⁱ", ylabel="ℓˢ", margin=5Plots.mm)
savefig("attendance_heatmap.pdf")

# soclearn = avg_attendance_volatility
# indlearn = avg_attendance_volatility

