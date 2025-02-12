import seaborn.objects
from jax import lax
from numpy import acos
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
# Initialize a wandb API object
api = wandb.Api()

# Replace with your wandb project and sweep ID
project_name = "ntk-analysis"
sweep_id = "zy9qnode"

# Fetch the sweep runs
sweep = api.sweep(f"{project_name}/{sweep_id}")
runs = sweep.runs

results = []

for run in tqdm(runs):
    summary = run.summary
    config = run.config

    lin_measure = summary.get("lin_measure")
    condition_number = summary.get("ntk_condition_number")

    config.update({"lin_measure": lin_measure, "ntk_condition_number": condition_number})
    results.append(config)


df = pd.DataFrame(results)
df.rename(columns={
    "layer_type": "Layer Type",
    "lin_measure": "Linear Diagonal Strength",
    "ntk_condition_number": "Condition Number",
}, inplace=True)


df["Log Linear Diagonal Strength"] = np.log(df["Linear Diagonal Strength"])

df = df[df["Layer Type"] != "inr_layers.LaplacianLayer"]
df = df[df["Layer Type"] != "inr_layers.MultiQuadraticLayer"]




def plot_lineplot(df, y = "Linear Diagonal Strength"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, x='w0', y=y, hue='Layer Type', ax=ax)
    plt.title(f'{y} of NTK')
    plt.savefig(
        f"results/{y.replace(' ', '_')}_lineplot.png"
    )
    plt.show()


plot_lineplot(df)
plot_lineplot(df, y = "Condition Number")



def tabify(df):
    """
    Return a dataframe with the max condition number of each layer type and its corresponding w0.
    """
    return df.loc[df.groupby("Layer Type")["Condition Number"].idxmax(), ["Layer Type", "Condition Number", "w0"]]

table = tabify(df)
print(table)







def plot_heatmap(df):

    grouped = df.groupby('layer_type')

    for layer_type, group in grouped:
        pivot_table = group.pivot(index='s0', columns='w0', values='lin_measure')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, cmap="viridis")
        plt.title(f'Heatmap for {layer_type}')
        plt.xlabel('w0')
        plt.ylabel('s0')
        plt.savefig(f"results/heatmap_{layer_type}.png")


