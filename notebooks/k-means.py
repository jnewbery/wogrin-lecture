# /// script
# dependencies = ["pyomo"]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Time Series Aggregation for Generation Expansion Planning with Energy Storage System: A Step-by-Step Tutorial**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook is designed as a tutorial accompanying [Prof. Sonja Wogrin](https://www.tugraz.at/en/institutes/iee/institute/team/wogrin-sonja)'s talk at the [DTU PES Summer School 2026](https://energy-markets-school.dk/).

    **Content Summary**: This tutorial covers time series aggregation for generation expansion planning (GEP) with intertemporal constraints. By the end of the tutorial, you will be tasked with two **challenges**. The first challenge is to cluster the input time series of a specific GEP problem while minimizing the **output error**, i.e., the difference between the optimal objective function values of the full-scale and the aggregated models. The second challenge is to derive with the aggregated model **upper and lower bounds** for the optimal full-scale solution.

    The notebook is structured as follows:

    1. **Environment Configuration**: Configures the necessary environment for the simulation (dependencies, libraries, and input data sources).
    2. **Set Up Simulation Parameters and Visualize the Input Time Series**: Characterizes the case study by setting up the parameters and visualizing the input time series for analysis.
    3. **Full-Scale Generation Expansion Planning Model**: Implements the full-scale GEP model.
    4. **Time Series Aggregation**: Applies clustering techniques for time series aggregation.
    5. **Aggregated Generation Expansion Planning Model**: Implements the aggregated GEP model, i.e., a reduced version of the full-scale model solved on a selected set of representative time steps (or representative days).
    6. **Evaluation**: Presents the evaluation methodology for the clustering techniques.
    7. **Challenge**: Provides the instructions for the tutorial challenges.

    This tutorial was developed as part of the European Research Council (ERC) project [NetZero-Opt](https://www.tugraz.at/en/institutes/iee/research/current-projects/netzero-opt) (Grant No. 101116212).

    If you would like to reference this tutorial, please cite the following paper, as the content presented here is based on the findings of this research:

    *   S. Wogrin, "Time Series Aggregation for Optimization: One-Size-Fits-All?," *IEEE Transactions on Smart Grid*, vol. 14, no. 3, pp. 2489-2492, May 2023, doi: 10.1109/TSG.2023.3242467. [Link](https://ieeexplore.ieee.org/abstract/document/10037240).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **1. Environment Configuration**
    """)
    return


@app.cell
def _():
    import copy
    import os
    from pathlib import Path
    import requests
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyomo.environ as pyo
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler

    return KMeans, MinMaxScaler, Path, copy, np, os, pd, plt, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **2. Set Up the Simulation Parameters and Visualize the Input Time Series**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1. Set the values for the simulation parameters

    * **Time Horizon**: The cardinality of the set of time steps $\mathcal{T}$ considered in the simulation, is denoted by $T$ (`T`).
    * **Operational Costs**: The electricity generation costs (€/MWh) for wind and thermal power plants are denoted by $C^\mathrm{op,w}$ (`OPER_COST_WIND`) and $C^\mathrm{op,th}$ (`OPER_COST_THERMAL`), respectively. The costs associated with energy storage charging and discharging (€/MWh) are represented by $C^{\mathrm{c,s}}$ (`OPER_COST_STOR_CH`) and $C^{\mathrm{d,s}}$ (`OPER_COST_STOR_CH`) respectively.
    * **Storage Parameters**: The charging and discharging efficiencies of the storage are denoted by $\eta^{\mathrm{c,s}}$ (`STOR_EFF_CH`) and $\eta^{\mathrm{d,s}}$ (`STOR_EFF_DIS`) respectively. The energy to power ratio (h) is denoted by $\tau$ (`STOR_ETP`).
    * **Non-Supplied Energy Cost**: The penalty cost (€/MWh) for non-supplied energy is denoted by $C^\mathrm{nse}$ (`OPER_COST_STOR_CH`).
    * **Investment Costs**: The capital cost (€/MW) for wind, thermal and storage capacity expansion are denoted by $C^\mathrm{inv,w}$ (`INV_COST_WIND`), $C^\mathrm{inv,th}$ (`INV_COST_THERMAL`) and $C^\mathrm{inv,s}$ (`INV_COST_STOR`), respectively.

    **Do not modify the values assigned to these parameters.**
    """)
    return


@app.cell
def _():
    # Number of time steps
    T = 8736

    # Define operational costs
    OPER_COST_WIND = 3  # for wind power (€/MWh)
    OPER_COST_THERMAL = 60  # for thermal power (€/MWh)
    OPER_COST_NSE = 5000 # for non-supplied energy (€/MWh)
    OPER_COST_STOR_DIS = 1.5 # storage discharging cost (€/MWh)
    OPER_COST_STOR_CH = 0 # storage charging cost (€/MWh)

    # Define storage parameters
    STOR_EFF_CH = 0.9 # storage charging efficiency
    STOR_EFF_DIS = 0.9 # storage discharging efficiency
    STOR_ETP = 4 # storage energy to power ratio (h)

    # Define investment costs
    INV_COST_WIND = 4e4  # for wind power (€/MW)
    INV_COST_THERMAL = 4e4  # for thermal power (€/MW)
    INV_COST_STOR =1e4 # storage investment cost (€/MW)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2. Load input data

    Fetch `input_data` from `JakubRybka/Tutorial_input_data` repo if not already fetched.

    Load `input_data.xlsx` from `/data` directory.

    The `input_data` file is an excel sheet with three columns:

    * **Time Step Index (-)**: Contains the time step indices for the simulation, $t = 1, \dots, T$.
    * **Wind Capacity Factor (p.u.)**: Contains the wind capacity factors, denoted as $CF^\mathrm{w}_t$ for $t = 1, \dots, T$.
    * **Demand (MWh)**: Contains the energy demand, denoted as $D_t$ for $t = 1, \dots, T$.
    """)
    return


@app.cell
def _(Path, os, pd, requests):
    _DATA_DIR = Path(__file__).parent / 'data'
    if not os.path.exists(_DATA_DIR / 'input_data.xlsx'):
        os.makedirs(_DATA_DIR, exist_ok=True)
        input_data = requests.get('https://raw.githubusercontent.com/JakubRybka/Tutorial_input_data/main/input_data.xlsx', allow_redirects=True).content
        input_data_path = _DATA_DIR / 'input_data.xlsx'
        with open(input_data_path, 'wb') as f:
            f.write(input_data)
    # Load input data
    input_data = pd.read_excel(_DATA_DIR / 'input_data.xlsx')

    input_data['Demand (MWh)'] =(2 + input_data['Demand (MWh)'] *3)*100

    # Display the first few rows of the input data
    input_data
    return (input_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3. Graphical representation of the input data

    Display the input data in two ways:

    *   Distributions of demand and wind capacity factors (frequency across all hours in the data set)
    *   Distribution of demand and wind capacity factors for each hour-of-day
    """)
    return


@app.cell
def _(input_data, plt):
    # Create subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Figure 1: Normalized demand data histogram (on the first subplot)
    axes[0].hist(input_data["Demand (MWh)"], bins=30, color='red', edgecolor='black', alpha=0.7)
    axes[0].set_title('Demand Data (MWh) Distribution', fontsize=16,weight='bold')
    axes[0].set_xlabel('Value', fontsize=16)
    axes[0].set_ylabel('Frequency', fontsize=16)
    axes[0].set_yticks([0, 100, 200, 300, 400, 500, 600])
    axes[0].tick_params(axis='both', labelsize=15)
    axes[0].margins(x=0)

    # Figure 2: Wind capacity factors data histogram (on the second subplot)
    axes[1].hist(input_data["Wind Capacity Factor (p.u.)"], bins=30, color='blue', edgecolor='black', alpha=0.7)
    axes[1].set_title('Wind Capacity Factor (p.u.) Data Distribution', fontsize=16,weight='bold')
    axes[1].set_xlabel('Value', fontsize=16)
    axes[1].set_ylabel('Frequency', fontsize=16)
    axes[1].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    axes[1].set_yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    axes[1].set_xlim([0, 1])
    axes[1].tick_params(axis='both', labelsize=15)
    axes[1].margins(x=0)

    # Display plots
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(input_data, np, pd, plt):
    data_temp = input_data.copy()
    dem = data_temp['Demand (MWh)'].values.reshape(364, 24)
    cf = data_temp['Wind Capacity Factor (p.u.)'].values.reshape(364, 24)
    dem_daily = pd.DataFrame(dem, columns=range(1, 25, 1))
    cf_daily = pd.DataFrame(cf, columns=range(1, 25, 1))

    # Boxplot properties for consistent styling
    boxprops_red = dict(linestyle='-', linewidth=1.5, color='red')
    medianprops_red = dict(linestyle='-', linewidth=1.5, color='black')
    whiskerprops_red = dict(linestyle='-', linewidth=1.5, color='red')
    capprops_red = dict(linestyle='-', linewidth=1.5, color='red')
    flierprops_red = dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none', alpha=0.6)
    boxprops_blue = dict(linestyle='-', linewidth=1.5, color='blue')
    medianprops_blue = dict(linestyle='-', linewidth=1.5, color='black')
    whiskerprops_blue = dict(linestyle='-', linewidth=1.5, color='blue')
    capprops_blue = dict(linestyle='-', linewidth=1.5, color='blue')
    flierprops_blue = dict(marker='o', markerfacecolor='blue', markersize=6, linestyle='none', alpha=0.6)

    # Create subplots (1 row, 2 columns)
    _, _axes = plt.subplots(1, 2, figsize=(19, 4))

    # Demand Boxplot
    dem_bp = dem_daily.boxplot(ax=_axes[0], boxprops=boxprops_red, medianprops=medianprops_red, whiskerprops=whiskerprops_red, capprops=capprops_red, flierprops=flierprops_red, return_type='dict', patch_artist=True)
    _axes[0].set_title('Demand (MWh) Daily Distribution', fontsize=16, weight='bold')
    _axes[0].set_xlabel('Time (hours)', fontsize=16)
    _axes[0].tick_params(axis='both', labelsize=16)
    _axes[0].grid(True, linestyle='--', alpha=0.7)
    _axes[0].margins(x=0)
    for patch in dem_bp['boxes']:
        patch.set_facecolor('r')
        patch.set_alpha(0.3)

    # Capacity Factor Boxplot
    cf_bp = cf_daily.boxplot(ax=_axes[1], boxprops=boxprops_blue, medianprops=medianprops_blue, whiskerprops=whiskerprops_blue, capprops=capprops_blue, flierprops=flierprops_blue, return_type='dict', patch_artist=True)
    _axes[1].set_title('Wind Capacity Factor (p.u.) Daily Distribution', fontsize=16, weight='bold')
    _axes[1].set_xlabel('Time (hours)', fontsize=16)
    _axes[1].tick_params(axis='both', labelsize=16)  # Added return_type='dict' and patch_artist=True
    _axes[1].grid(True, linestyle='--', alpha=0.7)
    _axes[1].margins(x=0)
    _axes[1].set_yticks(np.arange(0, 1.1, 0.2))
    for patch in cf_bp['boxes']:
        patch.set_facecolor('b')
        patch.set_alpha(0.3)

    # Display plots
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **4. Time Series Aggregation**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This section presents examples of **clustering techniques** that can be used for time series aggregation to construct the aggregated GEP model. These clustering techniques define **mappings** from the original time steps to representative time steps.

    Clustering input time series is motivated by the necessity to **reduce the computational complexity** of large-scale optimization problems. However, this simplification requires alternative formulations, namely **the aggregated optimization models**, detailed in the section 5.

    The presence of **storage intertemporal constraints** requires clustering techniques and alternative model formulations that **preserve chronology**. This is challenging because standard clustering techniques often disregard temporal chronology.

    Solving the aggregated optimization model generally results in an **output error**, i.e., a difference in the optimal objective function values between the full-scale and aggregated optimization models. The following clustering techniques are implemented to analyze their impact on the accuracy of the aggregated models:

    * **K-Means clustering (rep. hours + chronology)** (`chronological_kmeans_clustering`): A widely used clustering technique, [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) partitions the data into $K$ clusters by iteratively assigning each data point to the nearest centroid and updating the centroids based on the mean of the assigned samples. This process continues until convergence, minimizing the within-cluster variance (i.e., the sum of squared distances from samples to their respective centroids). Since standard clustering techniques do not preserve chronology, we include a ``chronologize`` function $^{[1]}$ which preserves temporal continuity by grouping only consecutive time steps assigned to the same cluster.

    * **K-Means clustering (rep. days)** (`rep_clustering`): Rather than clustering individual time steps, this technique clusters complete 24-hour profiles $^{[2]}$. K-Means clustering is applied to these daily profiles, and a representative day is selected for each cluster.

    * **Chronological hierarchical clustering** (`CH_clustering`): A clustering technique that iteratively merges consecutive time steps with minimal Ward distance in input space$^{[3]}$. The clustering process starts with each time step constituting its cluster, and proceeds until the desired number of clusters is reached.

    ---
    <small>$^{[1]}$ J. Mannhardt, L. Kunz, and G. Sansavini, "Accurately modeling long-term storage with minimum representative hours in large-scale renewable energy systems," 2025. arXiv:2512.00892. [Link](https://arxiv.org/abs/2512.00892)</small>

    <small>$^{[2]}$ L. Kotzur, P. Markewitz, M. Robinius, D. Stolten, "Time series aggregation for energy system design: Modeling seasonal storage," *Applied Energy*, vol. 213, pp. 123-135, Mar. 2018, doi:10.1016/j.apenergy.2018.01.023. [Link](https://www.sciencedirect.com/science/article/pii/S0306261918300242)</small>

    <small>$^{[3]}$ S. Pineda and J. M. Morales, "Chronological time-period clustering for optimal capacity expansion planning with storage," *IEEE Transactions on Power Systems*, vol. 33, no. 6, pp. 7162-7170, Nov. 2018, doi: 10.1109/TPWRS.2018.2842093. [Link](https://doi.org/10.1109/TPWRS.2018.2842093)</small>
    """)
    return


@app.cell
def _(KMeans, MinMaxScaler, copy, np, pd):
    def kmeans_clustering(df, K):
        """Traditional K-means clustering"""
        # Extract the relevant features from df: Wind Capacity Factors and Demand
        features = df[['Demand (MWh)', 'Wind Capacity Factor (p.u.)']]

        # Initialize and fit scaler
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        # Perform K-means clustering with K clusters
        kmeans = KMeans(n_clusters=K, random_state=42)
        kmeans.fit(scaled_features)

        # Inverse transform centroids to original scale
        original_scale_centroids = scaler.inverse_transform(kmeans.cluster_centers_)

        # Create a mapping of time step index to cluster assignments
        return (kmeans.labels_, original_scale_centroids, {index: label for index, label in zip(df['Time Step Index (-)'], kmeans.labels_)})

    def chronologize(mapping):
        """Make any mapping chronological"""
        # Create a copy of the original mapping
        n_mapping = copy.deepcopy(mapping)
        it = 0
        for i in range(len(mapping)):
            if i == 0 or mapping[i] != mapping[i - 1]:
                it = it + 1
            n_mapping[i] = it
        return n_mapping

    def CH_clustering(df, K):
        """Chronological hierarchical clustering"""
        # Number of initial steps (hours)
        T = len(df)

        # Extract features for scaling
        features = df[['Demand (MWh)', 'Wind Capacity Factor (p.u.)']]
        scaler = MinMaxScaler()
        scaled_features_df = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=df.index)

        # Initially, every hour is its own cluster
        # clusters[i] stores the list of original indices (t)
        clusters = [[i] for i in range(T)]

        def calc_ward_dist(idx1, idx2):
            # Cluster sizes
            n1 = len(clusters[idx1])
            n2 = len(clusters[idx2])

            # Cluster means (centroids) using scaled features
            mean1 = scaled_features_df.iloc[clusters[idx1]].mean()
            mean2 = scaled_features_df.iloc[clusters[idx2]].mean()

            # Ward linkage formula: (n1*n2)/(n1+n2) * squared_euclidean_dist
            squared_dist = np.sum((mean1 - mean2) ** 2)
            return n1 * n2 / (n1 + n2) * squared_dist

        # Calculate initial distances between all adjacent clusters
        dists = [calc_ward_dist(i, i + 1) for i in range(T - 1)]

        # Merge until we reach K clusters
        while len(clusters) > K:
            # 1. Find the pair of adjacent clusters with the minimum Ward distance
            idx = np.argmin(dists)

            # 2. Merge cluster idx+1 into cluster idx
            clusters[idx].extend(clusters[idx + 1])  
            clusters.pop(idx + 1)

            # 3. Remove the distance corresponding to the merged pair
            dists.pop(idx)

            # 4. Update neighbors' distances
            # New distance between merged cluster and the one following it
            if idx < len(clusters) - 1:
                dists[idx] = calc_ward_dist(idx, idx + 1)

            # New distance between merged cluster and the one preceding it
            if idx > 0:
                dists[idx - 1] = calc_ward_dist(idx - 1, idx)

        # Convert clusters list to the mapping dictionary for Pyomo
        mapping_dict = {}
        for cluster_id, hours in enumerate(clusters):
            for t in hours:
                mapping_dict[t] = cluster_id

        return mapping_dict

    def rep_clustering(input_data, K):
        cf = input_data['Wind Capacity Factor (p.u.)'].values.reshape(364, 24)
        demand = input_data['Demand (MWh)'].values.reshape(364, 24)
        X = np.concatenate([demand, cf], axis=1)

        # Initialize and fit scaler
        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X)

        km = KMeans(n_clusters=K, random_state=42)
        km.fit(scaled_X)

        # Inverse transform centroids to original scale
        original_scale_centroids = scaler.inverse_transform(km.cluster_centers_)

        return (km.labels_, original_scale_centroids, {index: label for index, label in zip(input_data['Time Step Index (-)'], km.labels_)})

    return chronologize, kmeans_clustering, rep_clustering


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.1. Run **K-Means representative hours clustering**
    """)
    return


@app.cell
def _(mo):
    clusters = mo.ui.slider(
        start=1, stop=10, value=3, step=1,
        label="Number of clusters",
        show_value=True,
    )
    clusters
    return (clusters,)


@app.cell
def _(chronologize, clusters, input_data, kmeans_clustering):
    # K-Means representative hours clustering
    K = clusters.value  # Number of desired clusters (before chronologizing)
    kmeans_labels_shifted, kmeans_centroids_shifted, kmeans_mapping_shifted = kmeans_clustering(input_data, K)

    kmeans_mapping = {i: int(kmeans_mapping_shifted[old_key]) for i, old_key in enumerate(sorted(kmeans_mapping_shifted.keys()))}
    chronological_kmeans_mapping = chronologize(kmeans_mapping)

    # Display the mapping
    print(f"Number of clusters in chronological mapping: {len(set(chronological_kmeans_mapping.values()))}")
    print(chronological_kmeans_mapping)
    return (
        K,
        chronological_kmeans_mapping,
        kmeans_centroids_shifted,
        kmeans_labels_shifted,
        kmeans_mapping,
    )


@app.cell
def _(K, input_data, kmeans_centroids_shifted, kmeans_labels_shifted, np, plt):
    cluster_colors = ['red', 'limegreen', 'orange', 'magenta', 'darkgray', 'green', 'black', 'blue', 'yellow', 'pink', 'purple']
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(input_data['Demand (MWh)'], input_data['Wind Capacity Factor (p.u.)'],
                          c=[cluster_colors[x] for x in kmeans_labels_shifted], marker='o', s=70, alpha=0.7)
    plt.scatter(kmeans_centroids_shifted[:, 0], kmeans_centroids_shifted[:, 1], c='black', s=200, marker='X', label='Centroids')
    plt.xlabel('Demand (MWh)', fontsize=16)
    plt.ylabel('Wind capacity factor (p.u.)', fontsize=16)
    plt.title(f'Traditional K-Means Clustering with {K} Clusters', fontsize=16,weight='bold')
    plt.tick_params(axis='both', labelsize=15)

    # Manually create legend elements for clusters and centroids
    unique_clusters = np.unique(kmeans_labels_shifted)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'{cid+1}',
                   markerfacecolor=cluster_colors[cid], markersize=10)
        for cid in unique_clusters
    ]

    plt.legend(handles=legend_elements, loc='upper right', title="Cluster", title_fontsize=16, fontsize=16, markerscale=1, framealpha=1)
    plt.gca()
    return (cluster_colors,)


@app.cell
def _(
    chronological_kmeans_mapping,
    cluster_colors,
    input_data,
    kmeans_mapping,
    np,
    plt,
):
    R = 72  # Exemplary hours (a day)
    P = 22  # which day to show?

    # 1. Calculate aggregated means for demand and wind for each unique chronological K-Means cluster
    aggregated_demand_means = {}
    aggregated_wind_means = {}
    unique_chrono_clusters = sorted(list(set(chronological_kmeans_mapping.values())))

    for k_chrono in unique_chrono_clusters:
        # Get original time indices that map to this chronologized cluster
        original_indices_in_chrono_cluster = [t for t, cluster_id in chronological_kmeans_mapping.items() if cluster_id == k_chrono]

        if original_indices_in_chrono_cluster:
            aggregated_demand_means[k_chrono] = input_data['Demand (MWh)'].iloc[original_indices_in_chrono_cluster].mean()
            aggregated_wind_means[k_chrono] = input_data['Wind Capacity Factor (p.u.)'].iloc[original_indices_in_chrono_cluster].mean()
        else:
            aggregated_demand_means[k_chrono] = 0.0  # Fallback
            aggregated_wind_means[k_chrono] = 0.0  # Fallback

    # 2. Select an exemplary 24-hour period (e.g., the first day)
    exemplary_start_idx = P * R  # Starting from the first hour of the dataset
    exemplary_hours = range(exemplary_start_idx, exemplary_start_idx + R)

    # 3. Extract original demand and wind for the exemplary period
    original_demand_profile = input_data['Demand (MWh)'].iloc[exemplary_hours].values
    original_wind_profile = input_data['Wind Capacity Factor (p.u.)'].iloc[exemplary_hours].values

    # 4. Prepare aggregated profiles for the exemplary period
    aggregated_demand_profile = np.zeros(R)
    aggregated_wind_profile = np.zeros(R)
    aggregated_colors = []

    for i, t in enumerate(exemplary_hours):
        chrono_cluster_id = chronological_kmeans_mapping[t]
        original_kmeans_cluster_id = kmeans_mapping[t]
        aggregated_demand_profile[i] = aggregated_demand_means[chrono_cluster_id]
        aggregated_wind_profile[i] = aggregated_wind_means[chrono_cluster_id]
        aggregated_colors.append(cluster_colors[original_kmeans_cluster_id % len(cluster_colors)])

    # 5. Plotting
    #_fig, _axes = plt.subplots(2, 1, figsize=(19, 4))
    _fig, _axes = plt.subplots(2, 1, figsize=(20, 12))
    # Plot Demand
    _axes[0].plot(range(1, R + 1), original_demand_profile, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Original Profile')
    _axes[0].plot(range(1, R + 1), aggregated_demand_profile, color='black', linestyle='-', linewidth=2, zorder=4)  # Line connecting aggregated points
    scatter_demand = _axes[0].scatter(range(1, R + 1), aggregated_demand_profile, c=aggregated_colors, s=70, marker='o', edgecolor='black', zorder=5)  # No label here, legend will be built manually
    _axes[0].set_title(f'Demand (MWh)', fontsize=16, weight='bold')
    _axes[0].set_xlabel('Time (hours)', fontsize=16)
    _axes[0].set_xticks(range(1, R + 1, R // 24))  # Ensure color index is valid
    _axes[0].set_xticklabels(range(1, R + 1, R // 24))
    _axes[0].tick_params(axis='both', labelsize=12)
    _axes[0].grid(True, linestyle='--', alpha=0.7)
    _axes[0].margins(x=0)

    # Create custom legend for clarity
    legend_lines_handles = [plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Original Profile'), plt.Line2D([0], [0], color='black', linestyle='-', marker='o', markerfacecolor='black', markeredgecolor='black', markersize=7, label='Aggregated Profile')]
    first_legend = _axes[0].legend(handles=legend_lines_handles, fontsize=12)
    _axes[0].add_artist(first_legend)

    # Legend for K-Means cluster colors
    legend_elements_colors = [plt.Line2D([0], [0], marker='o', color='w', label=f'{cid}', markerfacecolor=cluster_colors[cid], markersize=10) for cid in sorted(set(kmeans_mapping.values()))]

    # Plot wind capacity factor
    _axes[1].plot(range(1, R + 1), original_wind_profile, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Original Profile')
    _axes[1].plot(range(1, R + 1), aggregated_wind_profile, color='black', linestyle='-', linewidth=2, zorder=4)  # Line connecting aggregated points
    scatter_wind = _axes[1].scatter(range(1, R + 1), aggregated_wind_profile, c=aggregated_colors, s=70, marker='o', edgecolor='black', zorder=5)  # No label here
    _axes[1].set_title(f'Wind Capacity Factor (p.u.)', fontsize=16, weight='bold')
    _axes[1].set_xlabel('Time (hours)', fontsize=16)

    _axes[1].set_xticks(range(1, R + 1, R // 24))
    _axes[1].set_xticklabels(range(1, R + 1, R // 24))
    _axes[1].set_yticks(np.arange(0, 1.1, 0.2))  # Updated to show line and marker
    _axes[1].tick_params(axis='both', labelsize=12)
    _axes[1].grid(True, linestyle='--', alpha=0.7)
    _axes[1].margins(x=0)

    second_legend = _axes[1].legend(handles=legend_lines_handles, fontsize=12)
    _axes[1].add_artist(second_legend)

    # Display plot
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.3. Run **Representative day clustering**
    """)
    return


@app.cell
def _(input_data, rep_clustering):
    K_rep = 30 # Number of desired representative days

    rep_labels_shifted, rep_centroids_shifted, rep_mapping_shifted = rep_clustering(input_data, K_rep)
    rep_mapping = {i: int(rep_mapping_shifted[old_key]) for i, old_key in enumerate(sorted(rep_mapping_shifted.keys()))}

    # Display mapping
    print(rep_mapping)
    return rep_centroids_shifted, rep_labels_shifted


@app.cell
def _(input_data, plt, rep_centroids_shifted, rep_labels_shifted):
    _R = 24  # Number of hours in a representative period (a day)

    # Reshape input data into daily profiles
    demand_profiles = input_data['Demand (MWh)'].values.reshape(-1, _R)
    wind_profiles = input_data['Wind Capacity Factor (p.u.)'].values.reshape(-1, _R)

    # Get all unique cluster IDs
    unique_cluster_ids = [5, 12, 15]

    # Choose a colormap for distinct colors for each cluster
    num_clusters = len(unique_cluster_ids)
    # Use a colormap suitable for categorical data, e.g., 'tab20' if num_clusters <= 20
    # If more, use 'rainbow' or a custom cycle
    cmap_1 = plt.colormaps.get_cmap('viridis').resampled(num_clusters) if num_clusters <= 20 else plt.colormaps.get_cmap('rainbow').resampled(num_clusters)

    # Create subplots (1 row, 2 columns)
    _, _axes = plt.subplots(1, 2, figsize=(19, 4))  # Increased width for legend, height for better visibility

    # Lists to hold handles for centroids for the combined legend
    centroid_handles = []
    centroid_labels = []

    for idx, current_cluster_id in enumerate(unique_cluster_ids):
        # Find all day indices that belong to the chosen cluster
        member_day_indices = [i for i, label in enumerate(rep_labels_shifted) if label == current_cluster_id]

        # Skip if no members found for this cluster
        if not member_day_indices:
            continue

        # Assign a color for the current cluster
        cluster_color = cmap_1(idx)

        # Extract centroid data for the chosen cluster
        centroid_demand = rep_centroids_shifted[current_cluster_id, :_R]
        centroid_wind = rep_centroids_shifted[current_cluster_id, _R:]

        # Extract member data for the chosen cluster
        member_demand_profiles = demand_profiles[member_day_indices]
        member_wind_profiles = wind_profiles[member_day_indices]

        # Plot Demand for the current cluster
        for day_demand in member_demand_profiles:
            # Plot members with lighter color and higher transparency
            _axes[0].plot(range(1, _R + 1), day_demand, color=cluster_color, linestyle='-', linewidth=0.8, alpha=0.4)

        # Plot centroid with a darker version of the color and a distinct marker
        line_demand, = _axes[0].plot(range(1, _R + 1), centroid_demand, color=cluster_color, linestyle='-', marker='o', markersize=6, linewidth=2.5, zorder=5, label=f'Centroid {current_cluster_id}')
        centroid_handles.append(line_demand)
        centroid_labels.append(f'Centroid {current_cluster_id}')

        # Plot Wind Capacity Factor for the current cluster
        for day_wind in member_wind_profiles:
            # Plot members with lighter color and higher transparency
            _axes[1].plot(range(1, _R + 1), day_wind, color=cluster_color, linestyle='-', linewidth=0.8, alpha=0.4)

        # Plot centroid with a darker version of the color and a distinct marker
        line_wind, = _axes[1].plot(range(1, _R + 1), centroid_wind, color=cluster_color, linestyle='-', marker='o', markersize=6, linewidth=2.5, zorder=5)  
        # No need to add to centroid_handles for wind as one common legend will be created

    # --- Common plot settings after all clusters are plotted ---

    # Demand plot settings
    _axes[0].set_title(f'Demand (MWh) (Centroids vs. Members)', fontsize=16, weight='bold')
    _axes[0].set_xlabel('Time (hours)', fontsize=16)
    _axes[0].set_xticks(range(2, _R + 1, 2))
    _axes[0].set_xticklabels(range(2, _R + 1, 2))
    _axes[0].tick_params(axis='both', labelsize=14)  
    _axes[0].grid(True, linestyle='--', alpha=0.7)
    _axes[0].margins(x=0)

    # Wind plot settings
    _axes[1].set_title(f'Wind Capacity Factor (p.u.) (Centroids vs. Members)', fontsize=16, weight='bold')  
    _axes[1].set_xlabel('Time (hours)', fontsize=16)
    _axes[1].set_xticks(range(2, _R + 1, 2))
    _axes[1].set_xticklabels(range(2, _R + 1, 2))
    _axes[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  
    _axes[1].tick_params(axis='both', labelsize=14)
    _axes[1].grid(True, linestyle='--', alpha=0.7)
    _axes[1].margins(x=0)

    # Adjust rect to make space for the legend on the right
    plt.tight_layout(rect=[0, 0, 0.9, 1])  
    plt.show()
    return


if __name__ == "__main__":
    app.run()
