import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

from mg_shannon import MinorityGameWithStrategyTable

if __name__ == "__main__":
    m = MinorityGameWithStrategyTable(101, 50, 3, 3)
    mean_list,stdd_list,avg_win,volatility,avolatility,shanno = m.run_game()
    labels=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    df = pd.DataFrame(avg_win,index=labels,columns=labels)
    print(df)

    # Define the plot for average winning rate
    fig, ax = plt.subplots(figsize=(13,7))

    # Add title to the Heat map
    title = "Average Winning Proportion"

    # Set the font size and the distance of the title from the plot
    plt.title(title,fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5,1.05])

    # Use the heatmap function from the seaborn package
    heatmap = sns.heatmap(df,fmt="",linewidths=0.30,ax=ax)
    plt.xlabel('Social Learning Rate')
    plt.ylabel('Asocial Learning Rate')
    heatmap.invert_yaxis()
    plt.yticks(rotation=0) 

    # Display the Heatmap
    plt.savefig("heatmap_avg_win")


    df = pd.DataFrame(volatility,index=labels,columns=labels)
    print(df)
    # Define the plot for volatility
    fig, ax = plt.subplots(figsize=(13,7))

    # Add title to the Heat map
    title = "Volatility"

    # Set the font size and the distance of the title from the plot
    plt.title(title,fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5,1.05])

    # Use the heatmap function from the seaborn package
    heatmap = sns.heatmap(df,fmt="",linewidths=0.30,ax=ax)
    plt.xlabel('Social Learning Rate')
    plt.ylabel('Asocial Learning Rate')
    heatmap.invert_yaxis()
    plt.yticks(rotation=0) 

    # Display the Heatmap
    plt.savefig("heatmap_volatility")


    df = pd.DataFrame(avolatility,index=labels,columns=labels)
    print(df)
    # Define the plot for volatility
    fig, ax = plt.subplots(figsize=(13,7))

    # Add title to the Heat map
    title = "Attendance volatility"

    # Set the font size and the distance of the title from the plot
    plt.title(title,fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5,1.05])

    # Use the heatmap function from the seaborn package
    heatmap = sns.heatmap(df,fmt="",linewidths=0.30,ax=ax)
    plt.xlabel('Social Learning Rate')
    plt.ylabel('Asocial Learning Rate')
    heatmap.invert_yaxis()
    plt.yticks(rotation=0) 

    # Display the Heatmap
    plt.savefig("heatmap_volatility_attendance")

    df = pd.DataFrame(shanno,index=labels,columns=labels)
    print(df)

    # Define the plot for average winning rate
    fig, ax = plt.subplots(figsize=(13,7))

    # Add title to the Heat map
    title = "Shannon Entropy"

    # Set the font size and the distance of the title from the plot
    plt.title(title,fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5,1.05])

    # Use the heatmap function from the seaborn package
    heatmap = sns.heatmap(df,fmt="",linewidths=0.30,ax=ax)
    plt.xlabel('Social Learning Rate')
    plt.ylabel('Asocial Learning Rate')
    heatmap.invert_yaxis()
    plt.yticks(rotation=0)

    # Display the Heatmap
    plt.savefig("heatmap_shannon")