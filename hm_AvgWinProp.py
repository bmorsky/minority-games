import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

from mg_ImitateMutate import MinorityGameWithStrategyTable

if __name__ == "__main__":
    m = MinorityGameWithStrategyTable(101, 50, 3, 3)
    mean_list,stdd_list,avg_win = m.run_game()
    # print(avg_win)
    # avg_win = np.array(avg_win)
    # print(avg_win)
    df = pd.DataFrame.from_records(avg_win)
    print(df)

    # Define the plot
    fig, ax = plt.subplots(figsize=(13,7))

    # Add title to the Heat map
    title = "Imitation-Mutation Rate about average winning proportion"

    # Set the font size and the distance of the title from the plot
    plt.title(title,fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5,1.05])

    # Hide ticks for X & Y axis
    # ax.set_xticks([])
    # ax.set_yticks([])

    # Remove the axes
    # ax.axis('off')

    # Use the heatmap function from the seaborn package
    heatmap = sns.heatmap(df,fmt="",linewidths=0.30,ax=ax)
    # heatmap = heatmap.pivot("imitate", "mutate")
    heatmap.invert_yaxis()
    plt.yticks(rotation=0) 

    # Display the Pharma Sector Heatmap
    plt.savefig("heatmap")