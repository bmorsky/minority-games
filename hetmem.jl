using Statistics
# Parameters
N = 101 # number of players
R = 100 # number of rounds

# Variables and outcome
action = rand(0:1,N) # players' actions
history = zeros(R+1) # history of plays
memory = rand(1:10,N) # players' memory lengths
output = zeros(R+1,3) # timeseries output
payoff = zeros(N) # players' payoffs

# Initializations
history[1] = mean(action)
output[1,:] = [mean(action) mean(memory) mean(payoff)]

for m=1:R
    # Determine players' actions given their strategies and
    # the history of previous plays
    for n=1:N
        if mean(history[max(1,m-memory[n]):m]) < 0.5
            action[n] = 1
        else
            action[n] = 0
        end
    end

    # Determine payoffs
    if sum(action) < N/2
        minority = 1
    else
        minority = 0
    end
    for n=1:N
        if action[n] == minority
            payoff[n] = 1
        else
            payoff[n] = 0
        end
    end

    # Determine change in strategy (i.e. change in memory) by
    # selecting another random players and comparing payoffs
    for n=1:N
        player = rand(1:N)
        if payoff[n] < payoff[player]
            memory[n] = memory[player]
        end
    end

    # Update history and save output
    history[m+1] = mean(action)
    output[m+1,:] = [mean(action), mean(memory), mean(payoff)]

end

# Plot timeseries
using PyPlot

PyPlot.plot(output,label=["mean action","mean memory length","mean payoff"])
plt.xlabel("Time")
plt.legend()
plt.savefig("hetmem N$N R$R.png")
PyPlot.display_figs()
