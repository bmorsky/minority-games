using Plots, Random, Statistics

# Outputs
max_M = 12
avg_attendance_volatility = zeros(max_M*5,3)

# Parameters
κ = 100 # payoff differential sensitivity
ℓⁱ = 0.1 # rate of individual learning
ℓˢ = 0.1 # rate of social learning
num_games = 20 # number of games to average over
num_turns = 500 # number of turns
S = 4 # number of strategy tables per individual

# Variables
attendance = Array{Int,1}(undef,num_turns)

X = [51,101,251,501,1001]
rng = MersenneTwister()

global count = 1
for x = 1:5
    N = X[x] # number of agents
    action = Array{Int,1}(undef,N) # actions taken: buy=1, sell=0
    for M = 1:max_M
        for game=1:num_games
            # Initialize game
            history = rand(rng,1:2^M)
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
                    end
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

                # Social learning
                # update_strategy_tables = strategy_tables
                # update_virtual_points = virtual_points
                # for i=1:N
                #     if ℓˢ >= rand(rng)
                #         # Find worst strategy and its points of focal player
                #         worst_points,worst_strat = findmin(virtual_points[i,:])
                #         # Select random other player and find its best strat and points
                #         player = rand(filter(x -> x ∉ [i], 1:N))
                #         best_points,best_strat = findmax(virtual_points[player,:])
                #         if 1/(1+exp(κ*(worst_points-best_points))) > rand(rng)
                #             update_strategy_tables[S*(i-1)+worst_strat,:] = strategy_tables[S*(player-1)+best_strat,:]
                #             update_virtual_points[i,worst_strat] = virtual_points[player,best_strat]
                #         end
                #     end
                # end
                # strategy_tables = update_strategy_tables
                # virtual_points = update_virtual_points

            end
            avg_attendance_volatility[count,1] += var(attendance)/(num_games*N)
        end
        avg_attendance_volatility[count,2] = (2^M)/N
        avg_attendance_volatility[count,3] = x
        global count += 1
    end
end

avg_attendance_volatility[:,3] = Int.(avg_attendance_volatility[:,3])

x1 = avg_attendance_volatility[1:12,2]
x2 = avg_attendance_volatility[13:24,2]
x3 = avg_attendance_volatility[25:36,2]
x4 = avg_attendance_volatility[37:48,2]
x5 = avg_attendance_volatility[49:60,2]

y1 = avg_attendance_volatility[1:12,1]
y2 = avg_attendance_volatility[13:24,1]
y3 = avg_attendance_volatility[25:36,1]
y4 = avg_attendance_volatility[37:48,1]
y5 = avg_attendance_volatility[49:60,1]

z1 = Int.(avg_attendance_volatility[1:12,3])
z2 = Int.(avg_attendance_volatility[13:24,3])
z3 = Int.(avg_attendance_volatility[25:36,3])
z4 = Int.(avg_attendance_volatility[37:48,3])
z5 = Int.(avg_attendance_volatility[49:60,3])

scatter([x1 x2 x3 x4 x5], [y1 y2 y3 y4 y5], markercolor=[z1 z2 z3 z4 z5],
xlims=(0.001,100), ylims=(0.1,1000), xscale=:log10, yscale=:log10,
label=["N=51" "N=101" "N=251" "N=501" "N=1001"],
xlabel = "\\alpha", ylabel="\\sigma ²/N", legend=:topright,
thickness_scaling = 1.5)
savefig("var_S4.pdf")
