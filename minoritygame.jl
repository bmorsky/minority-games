using RCall, Statistics

# Outputs
# avg_attendance = zeros(11,11)
# avg_attendance_volatility = zeros(11,11)
# avg_num_winners = zeros(11,11)
# avg_num_winners_volatility = zeros(11,11)
max_M = 8
avg_attendance = zeros(max_M*5,2)
avg_attendance_volatility = zeros(max_M*5,2)
avg_num_winners = zeros(max_M*5,2)
avg_num_winners_volatility = zeros(max_M*5,2)

# Parameters
κ = 100 # payoff differential sensitivity
# M = 3 # memory length
# N = 100 # number of agents
num_games = 20 # number of games to average over
num_turns = 500 # number of turns
S = 2 # number of strategy tables per individual

X = [51,101,251,501,1001]

count = 1
for x = 1:5
    N = X[x] # number of agents
for M = 1:max_M

    # Variables
    action = Array{Int,1}(undef,N) # actions taken: buy=1, sell=0
    history = Array{Int,1}(undef,M) # history of the winning strategy for the last M turns
    strategy_tables = Array{BigInt,2}(undef,N,S) # S strategy tables for the N players

for game=1:num_games

    # Initialize game
    total_buys = 0
    history = rand(0:2^M-1)
    for i=1:N
        strategy_tables[i,:] = rand(0:big(2)^(2^M)-1,S)
    end
    virtual_points = zeros(Int64,N,S) # virtual points for all players' strategy tables

    # Outputs
    attendance = Array{Int,1}(undef,num_turns)
    entropy = 0
    num_winners = Array{Int,1}(undef,num_turns)

    for turn=1:num_turns

        # Actions taken
        for i=1:N
            best_strat = strategy_tables[i,findmax(virtual_points[i,:])[2]]
            action[i] = BigInt(mod((best_strat-mod(best_strat,big(2)^history))/big(2)^history,2))
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
            for j=1:S
                virtual_points[i,j] += (-1)^(minority+Int(mod((strategy_tables[i,j]-mod(strategy_tables[i,j],big(2)^history))/big(2)^history,2)))
            end
            # win_rate[turn] += (-1)^(minority+action[j])/N
        end
        history = Int(2*(history - mod((history-mod(history,2^(M-1)))/(2^(M-1)),2)*2^(M-1)) + minority)

        # Individual learning
        # for i=1:N
        #     if ℓⁱ < rand()
        #         strategy_tables[i,rand(1:3)] = rand(0:big(2)^(2^M)-1)
        #     end
        # end

    end

    # Update outputs
    avg_attendance[count,1] += mean(attendance)/num_games
    avg_attendance_volatility[count,1] += var(attendance)/(num_games*N)
    avg_num_winners[count,1] += mean(num_winners)/num_games
    avg_num_winners_volatility[count,1] += var(num_winners)/(num_games*N)
end
avg_attendance[count,2] = (2^M)/N
avg_attendance_volatility[count,2] = (2^M)/N
avg_num_winners[count,2] = (2^M)/N
avg_num_winners_volatility[count,2] = (2^M)/N
count += 1
end
end

using Plots

plot(avg_num_winners_volatility[:,2], avg_num_winners_volatility[:,1],seriestype=:scatter,xscale=:log10,yscale=:log10)
plot(avg_attendance_volatility[:,2], avg_attendance_volatility[:,1],seriestype=:scatter,xscale=:log10,yscale=:log10)
plot(avg_attendance[:,2], avg_attendance[:,1],seriestype=:scatter,xscale=:log10,yscale=:log10)


@rput avg_num_winners avg_num_winners_volatility
R"""
library(ggplot2)
library(cowplot); theme_set(theme_cowplot())
library(viridis)
library(viridisLite)

avg_num_winners <- as.data.frame(avg_num_winners)
avg_num_winners_volatility <- as.data.frame(avg_num_winners_volatility)

q <- ggplot() +
geom_raster(data=iotaVell,aes(x=V1,y=V2,fill=factor(V3))) +
theme(plot.margin=grid::unit(c(0,0,0,0), "mm"),legend.position="none") +
scale_fill_manual(labels=c("limit cycle","1 stable","2 stable","crash","stable & crash"),values=cols) +
scale_x_continuous(expand=c(0,0), limits=iotaphilims, breaks=iotaphicols, labels=iotaphicols) +
scale_y_continuous(expand=c(0,0), limits=ellomegalims, breaks=ellomegacols, labels=ellomegacols) +
ylab(expression(paste("Learning rate, ", '\u2113'))) +
xlab(expression(paste("Inflow rate, ", iota))) +
coord_fixed(ratio = 0.75)
ggsave(q,filename="avg_num_winners.png", width = 3.5, height = 3.5)

"""
