using Plots, Random, Statistics, StatsBase

# Outputs
avg_attendance_volatility = zeros(6,6)
avg_entropy = zeros(6,6)
avg_payoffs = zeros(6,6)

# Parameters
κ = 100 # payoff differential sensitivity
M = 6 # memory length
N = 101 # number of players
num_games = 20 # number of games to average over
num_turns = 500 # number of turns
S = 3 # number of strategy tables per individual

# Variables
attendance = Array{Int,1}(undef,num_turns)

# Random numbers
rng = MersenneTwister()

for ind_learn = 1:6
    ℓⁱ = (ind_learn-1)/20 # rate of individual learning
    action = Array{Int,1}(undef,N) # actions taken: buy=1, sell=0
    for soc_learn = 1:6
        ℓˢ = (soc_learn-1)/20 # rate of social learning
        for game=1:num_games
            # Initialize game
            entropy = 0
            history = rand(rng,1:2^M)
            payoffs = 0
            strategy_tables = rand(rng,0:1,S*N,2^M) # S strategy tables for the N players
            virtual_points = zeros(Int64,N,S) # virtual points for all players' strategy tables

            # Run simulation
            for turn=1:num_turns
                # Actions taken
                for i=1:N
                    best_strat = S*(i-1) + findmax(virtual_points[i,:])[2]
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
                    for j=1:S
                        virtual_points[i,j] += (-1)^(minority+strategy_tables[S*(i-1)+j,history])
                        payoffs +=  (-1)^(minority+action[i])/(N*num_turns)
                    end
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
                update_strategy_tables = strategy_tables
                update_virtual_points = virtual_points
                for i=1:N
                    if ℓˢ > rand(rng)
                        # Find worst strategy and its points of focal player
                        worst_points,worst_strat = findmin(virtual_points[i,:])
                        # Select random other player and find its best strat and points
                        player = rand(filter(x -> x ∉ [i], 1:N))
                        best_points,best_strat = findmax(virtual_points[player,:])
                        if 1/(1+exp(κ*(worst_points-best_points))) > rand(rng)
                            update_strategy_tables[S*(i-1)+worst_strat,:] = strategy_tables[S*(player-1)+best_strat,:]
                            update_virtual_points[i,worst_strat] = virtual_points[player,best_strat]
                        end
                    end
                end
                strategy_tables = update_strategy_tables
                virtual_points = update_virtual_points

                # Calculate entropy
                strat_frequencies = values(proportionmap(collect(eachrow(strategy_tables))))
                entropy += sum(-log.(strat_frequencies).*strat_frequencies)/num_turns

            end
            avg_attendance_volatility[ind_learn,soc_learn] += var(attendance)/(num_games*N)
            avg_entropy[ind_learn,soc_learn] += entropy/num_games
            avg_payoffs[ind_learn,soc_learn] += payoffs/num_games

        end
    end
end

pyplot()

heatmap(0:0.05:0.25, 0:0.05:0.25, log10.(avg_attendance_volatility), xlabel="ℓˢ", ylabel="ℓⁱ",
colorbar_ticks=[-1, 0, 1, 2], clims=(-1,2), colorbar_title="log(σ²/N)", thickness_scaling = 1.5)
savefig("volatility_heatmap_S3.pdf")

heatmap(0:0.05:0.25, 0:0.05:0.25, avg_entropy, xlabel="ℓˢ", ylabel="ℓⁱ", thickness_scaling = 1.5)
savefig("entropy_heatmap_S3.pdf")

