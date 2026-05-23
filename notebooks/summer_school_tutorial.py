# /// script
# dependencies = ["pyomo"]
# ///

import marimo

__generated_with = "0.19.6"
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
    return (
        KMeans,
        MinMaxScaler,
        Path,
        copy,
        np,
        os,
        pd,
        plt,
        pyo,
        requests,
        time,
    )


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
    return (
        INV_COST_STOR,
        INV_COST_THERMAL,
        INV_COST_WIND,
        OPER_COST_NSE,
        OPER_COST_STOR_CH,
        OPER_COST_STOR_DIS,
        OPER_COST_THERMAL,
        OPER_COST_WIND,
        STOR_EFF_CH,
        STOR_EFF_DIS,
        STOR_ETP,
        T,
    )


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

    *   First, we plot the distributions of demand and wind capacity factors.
    *   Next, we plot the demand and wind capacity factors for random days within the simulation period.
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
    fig_1, axes_1 = plt.subplots(1, 2, figsize=(19, 4))

    # Demand Boxplot
    dem_bp = dem_daily.boxplot(ax=axes_1[0], boxprops=boxprops_red, medianprops=medianprops_red, whiskerprops=whiskerprops_red, capprops=capprops_red, flierprops=flierprops_red, return_type='dict', patch_artist=True)
    axes_1[0].set_title('Demand (MWh) Daily Distribution', fontsize=16, weight='bold')
    axes_1[0].set_xlabel('Time (hours)', fontsize=16)
    axes_1[0].tick_params(axis='both', labelsize=16)
    axes_1[0].grid(True, linestyle='--', alpha=0.7)
    axes_1[0].margins(x=0)
    for patch in dem_bp['boxes']:
        patch.set_facecolor('r')
        patch.set_alpha(0.3)

    # Capacity Factor Boxplot
    cf_bp = cf_daily.boxplot(ax=axes_1[1], boxprops=boxprops_blue, medianprops=medianprops_blue, whiskerprops=whiskerprops_blue, capprops=capprops_blue, flierprops=flierprops_blue, return_type='dict', patch_artist=True)
    axes_1[1].set_title('Wind Capacity Factor (p.u.) Daily Distribution', fontsize=16, weight='bold')
    axes_1[1].set_xlabel('Time (hours)', fontsize=16)
    axes_1[1].tick_params(axis='both', labelsize=16)  # Added return_type='dict' and patch_artist=True
    axes_1[1].grid(True, linestyle='--', alpha=0.7)
    axes_1[1].margins(x=0)
    axes_1[1].set_yticks(np.arange(0, 1.1, 0.2))
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
    # **3. Full-Scale Generation Expansion Planning Model**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This section introduces the **full-scale generation expansion planning model**.

    The problem is formulated as a discrete-time optimization model, spanning a time horizon $t = 1, \dots, T$, with a sampling time of $\Delta$ (1 hour).

    Let $x^\mathrm{w}$, $x^\mathrm{th}$ and $x^\mathrm{s}$ represent the installed capacities (MW) of the wind, thermal, and storage units, respectively. Similarly, let $p^\mathrm{w}_t$ and $p^\mathrm{th}_t$ denote the power generation (MW) of the wind and thermal units at time step $t$, respectively, while $p^\mathrm{c}_t$ and $p^\mathrm{d}_t$ denote the charging and discharging power (MW) of the energy storage system. Let the storage state of charge (MWh) be denoted by $e^\mathrm{s}_t$, while $e^\mathrm{ns}_t$ denotes the non-supplied energy demand (MWh) at time step $t$.

    The goal of the full-scale optimization model is to determine the optimal values of the decision variables $\left\{x^\mathrm{w}, x^\mathrm{th},x^\mathrm{s}, p^\mathrm{w}_t, p^\mathrm{th}_t, e^\mathrm{ns}_t , p^\mathrm{c}_t , p^\mathrm{d}_t, e^{\mathrm{s}}_t|\, t \in \mathcal{T}\right\}$ that minimize the objective function $J$ defined as

    $J := C^{\mathrm{inv,w}} x^\mathrm{w} + C^\mathrm{inv,s} x^\mathrm{s} + C^\mathrm{inv,th} x^\mathrm{th} + \sum_{t=1}^T \left(C^\mathrm{op,w} p^\mathrm{w}_t \Delta + C^\mathrm{op,th} p^\mathrm{th}_t \Delta + C^\mathrm{nse} e^\mathrm{ns}_t + C^\mathrm{c,s} p^\mathrm{c}_t \Delta + C^\mathrm{d,s} p^\mathrm{d}_t \Delta\right)$,

    subject to the following constraints:

    * Thermal power generation limits: $0 \leq p^\mathrm{th}_t \leq x^\mathrm{th}, \, \forall t$.
    * Wind power generation limits: $0 \leq p^\mathrm{w}_t \leq CF^\mathrm{w}_t x^\mathrm{w}, \, \forall t$.
    * Energy balance constraints: $\left(p^\mathrm{th}_t + p^\mathrm{w}_t - p^\mathrm{c}_t + p^\mathrm{d}_t \right) \Delta + e^\mathrm{ns}_t = D_t, \, \forall t$.
    * Energy storage state of charge dynamics: $e^{\mathrm{s}}_{t+1} = e^{\mathrm{s}}_t +\left(\eta^\mathrm{c,s} p_{t}^\mathrm{c} - \frac{p_{t}^\mathrm{d}}{\eta^\mathrm{d,s}}\right)Δ, \forall t \in \mathcal{T} \setminus \{T\}$.
    * Energy storage boundary constraints: $e^{\mathrm{s}}_T + \left(\eta^\mathrm{c,s} p_{T}^\mathrm{c} - \frac{p_{T}^\mathrm{d}}{\eta^\mathrm{d,s}}\right)Δ = e^{\mathrm{s}}_1$,<br>
    $\qquad\qquad\qquad\qquad\qquad\qquad\qquad \,\, e^{\mathrm{s}}_1=0$.
    * Energy storage state of charge limits: $0\leq e^s_t \leq x^\mathrm{s}\tau, \forall t$.
    * Storage charging power limits: $0\leq p^\mathrm{c}_t \leq x^\mathrm{s}, \forall t$.
    * Storage discharging power limits: $0\leq p^\mathrm{d}_t \leq x^\mathrm{s}, \forall t$.
    """)
    return


@app.cell
def _(
    INV_COST_STOR,
    INV_COST_THERMAL,
    INV_COST_WIND,
    OPER_COST_NSE,
    OPER_COST_STOR_CH,
    OPER_COST_STOR_DIS,
    OPER_COST_THERMAL,
    OPER_COST_WIND,
    STOR_EFF_CH,
    STOR_EFF_DIS,
    STOR_ETP,
    T,
    input_data,
    pyo,
):
    def create_full_model(input_data,T, inv_cost_wind, inv_cost_thermal, oper_cost_wind,
                          oper_cost_thermal, oper_cost_nse, stor_etp, inv_cost_stor):
        # Create optimization model
        full_model = pyo.ConcreteModel(name="Investment_Problem")

        full_model.T = pyo.Set(initialize=list(range(T)))

        # Define parameters
        full_model.demand = pyo.Param(full_model.T, initialize=input_data['Demand (MWh)'].to_dict())  # Demand (MWh)
        full_model.capacity_factor = pyo.Param(full_model.T, initialize=input_data[
            'Wind Capacity Factor (p.u.)'].to_dict())  # Capacity factor (p.u.)

        # Define decision variables
        full_model.x_thermal = pyo.Var(within=pyo.NonNegativeReals)  # Thermal installed capacity (MW)
        full_model.x_wind = pyo.Var(within=pyo.NonNegativeReals)  # Wind installed capacity (MW)
        full_model.x_storage = pyo.Var(within=pyo.NonNegativeReals)  # Storage installed capacity (MW)
        full_model.p_thermal = pyo.Var(full_model.T, within=pyo.NonNegativeReals)  # Thermal power generation (MW)
        full_model.p_wind = pyo.Var(full_model.T, within=pyo.NonNegativeReals)  # Wind power generation (MW)
        full_model.p_storage_ch = pyo.Var(full_model.T, within=pyo.NonNegativeReals)  # Storage charging power (MW)
        full_model.p_storage_dis = pyo.Var(full_model.T, within=pyo.NonNegativeReals)  # Storage discharging power (MW)
        full_model.e_SOC = pyo.Var(full_model.T, within=pyo.NonNegativeReals)  # Storage state of charge (MWh)
        full_model.e_ns = pyo.Var(full_model.T, within=pyo.NonNegativeReals)  # Non-supplied energy demand (MWh)

        # Define constraints
        # 1. Thermal power generation limits
        def eThermal_Limits(mdl, t):
            return mdl.p_thermal[t] <= mdl.x_thermal

        full_model.eThermal_Limits = pyo.Constraint(full_model.T, rule=eThermal_Limits)

        # 2. Wind power generation limits
        def eWind_Limits(mdl, t):
            return mdl.p_wind[t] <= mdl.x_wind * mdl.capacity_factor[t]

        full_model.eWind_Limits = pyo.Constraint(full_model.T, rule=eWind_Limits)

        # 3. Storage state of charge dynamics
        def eSOC_Dynamics(mdl, t):
            if t != T - 1:
                return mdl.e_SOC[t + 1] == mdl.e_SOC[t] + (
                            STOR_EFF_CH * mdl.p_storage_ch[t] - (mdl.p_storage_dis[t] / STOR_EFF_DIS))
            return pyo.Constraint.Skip

        full_model.eSOC_Dynamics = pyo.Constraint(full_model.T, rule=eSOC_Dynamics)

        # 4. Storage boundary constraints
        def eSOC_Boundary_start(mdl):
            return mdl.e_SOC[0] == 0

        def eSOC_Boundary_end(mdl):
            return mdl.e_SOC[T - 1] + STOR_EFF_CH * mdl.p_storage_ch[T - 1] - mdl.p_storage_dis[T - 1] / STOR_EFF_DIS == mdl.e_SOC[0]

        full_model.eSOC_start = pyo.Constraint(rule=eSOC_Boundary_start)
        full_model.eSOC_end = pyo.Constraint(rule=eSOC_Boundary_end)

        # 5. Storage state of charge limits
        def eSOC_Limits(mdl, t):
            return mdl.e_SOC[t] <= mdl.x_storage * stor_etp

        full_model.eSOC_Limits = pyo.Constraint(full_model.T, rule=eSOC_Limits)

        # 6. Storage charging power limits
        def eStorage_Charging_Limits(mdl, t):
            return mdl.p_storage_ch[t] <= mdl.x_storage

        full_model.eStorage_Charging_Limits = pyo.Constraint(full_model.T, rule=eStorage_Charging_Limits)

        # 7. Storage discharging power limits
        def eStorage_Discharging_Limits(mdl, t):
            return mdl.p_storage_dis[t] <= mdl.x_storage

        full_model.eStorage_Discharging_Limits = pyo.Constraint(full_model.T, rule=eStorage_Discharging_Limits)

        # 8. Power balance constraint
        def ePower_Balance(mdl, t):
            return mdl.p_thermal[t] + mdl.p_wind[t] - mdl.p_storage_ch[t] + mdl.p_storage_dis[t] + mdl.e_ns[t] == \
                mdl.demand[t]

        full_model.ePower_Balance = pyo.Constraint(full_model.T, rule=ePower_Balance)

        # 9. Objective function
        def obj_rule(mdl):
            return inv_cost_wind * mdl.x_wind + inv_cost_stor * mdl.x_storage + inv_cost_thermal * mdl.x_thermal \
                + sum(oper_cost_wind * mdl.p_wind[t] + OPER_COST_STOR_CH * mdl.p_storage_ch[t] + OPER_COST_STOR_DIS *
                      mdl.p_storage_dis[t] + oper_cost_thermal * mdl.p_thermal[t] + oper_cost_nse * mdl.e_ns[t] for t
                      in mdl.T)

        full_model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Solve model
        full_model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        return full_model

    # Create and solve the model
    full_model = create_full_model(input_data, T, INV_COST_WIND, INV_COST_THERMAL, OPER_COST_WIND,
                                OPER_COST_THERMAL, OPER_COST_NSE, STOR_ETP, INV_COST_STOR)
    solver = pyo.SolverFactory('highs')
    res = solver.solve(full_model)

    # Check the status of the solution
    if res.solver.termination_condition == 'optimal':
        print(f"Optimal Objective Function Value: {pyo.value(full_model.obj)/1e6:.2f} mln €")
    else:
        print("No optimal solution found.")
    return (full_model,)


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
def _(KMeans, MinMaxScaler, copy, input_data, np, pd):
    def chronlogical_kmeans_clustering(df, K):
        """Chronological K-means clustering"""
        kmeans_labels_shifted, kmeans_centroids_shifted, kmeans_mapping_shifted = kmeans_clustering(input_data, K)
        kmeans_mapping = {i: int(kmeans_mapping_shifted[old_key]) for i, old_key in enumerate(sorted(kmeans_mapping_shifted.keys()))}
        chronological_kmeans_mapping = chronologize(kmeans_mapping)
        return chronological_kmeans_mapping

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
            else:
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

    def rep_clustering(inpud_data, K):
        cf = inpud_data['Wind Capacity Factor (p.u.)'].values.reshape(364, 24)
        demand = inpud_data['Demand (MWh)'].values.reshape(364, 24)
        X = np.concatenate([demand, cf], axis=1)

        # Initialize and fit scaler
        scaler = MinMaxScaler()
        scaled_X = scaler.fit_transform(X)
    
        km = KMeans(n_clusters=K, random_state=42)
        km.fit(scaled_X)

        # Inverse transform centroids to original scale
        original_scale_centroids = scaler.inverse_transform(km.cluster_centers_)
    
        return (km.labels_, original_scale_centroids, {index: label for index, label in zip(inpud_data['Time Step Index (-)'], km.labels_)})
    return CH_clustering, chronologize, kmeans_clustering, rep_clustering


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.1. Run **K-Means representative hours clustering**
    """)
    return


@app.cell
def _(chronologize, input_data, kmeans_clustering):
    # K-Means representative hours clustering
    K = 3  # Number of desired clusters (before chronologizing)
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
def _(mo):
    mo.md(r"""
    **Figure: K-Means clustering**
    """)
    return


@app.cell
def _(K, input_data, kmeans_centroids_shifted, kmeans_labels_shifted, np, plt):
    cluster_colors = ['red', 'limegreen', 'orange', 'magenta', 'darkgray']
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
    plt.show()
    return (cluster_colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Visualization:**
    """)
    return


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
    _fig, _axes = plt.subplots(1, 2, figsize=(19, 4))
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
    plt.show()
    return (P,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.2. Run **Chronological hierarchical clustering**
    """)
    return


@app.cell
def _(CH_clustering, input_data):
    K_CH = 720 # Number of desired clusters
    CH_mapping = CH_clustering(input_data, K_CH)

    # Display mapping
    print(CH_mapping)
    return (CH_mapping,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Visualization:**
    """)
    return


@app.cell
def _(CH_mapping, P, input_data, np, plt):
    R_1 = 72  # Exemplary hours (a day)

    # 1. Calculate aggregated means for demand and wind for each unique CH cluster
    aggregated_demand_means_ch = {}
    aggregated_wind_means_ch = {}
    unique_ch_clusters = sorted(list(set(CH_mapping.values())))

    for k_ch in unique_ch_clusters:
        # Get original time indices that map to this CH cluster
        original_indices_in_ch_cluster = [t for t, cluster_id in CH_mapping.items() if cluster_id == k_ch]
        if original_indices_in_ch_cluster:
            aggregated_demand_means_ch[k_ch] = input_data['Demand (MWh)'].iloc[original_indices_in_ch_cluster].mean()
            aggregated_wind_means_ch[k_ch] = input_data['Wind Capacity Factor (p.u.)'].iloc[original_indices_in_ch_cluster].mean()
        else:
            aggregated_demand_means_ch[k_ch] = 0.0  # Fallback
            aggregated_wind_means_ch[k_ch] = 0.0  # Fallback

    # 2. Select an exemplary 24-hour period (e.g., a random day)
    random_day = P  # Ensures we pick a full day
    exemplary_start_idx_1 = P * R_1
    exemplary_hours_1 = range(exemplary_start_idx_1, exemplary_start_idx_1 + R_1)

    # 3. Extract original demand and wind for the exemplary period
    original_demand_profile_ch = input_data['Demand (MWh)'].iloc[exemplary_hours_1].values
    original_wind_profile_ch = input_data['Wind Capacity Factor (p.u.)'].iloc[exemplary_hours_1].values

    # 4. Prepare aggregated profiles for the exemplary period
    aggregated_demand_profile_ch = np.zeros(R_1)
    aggregated_wind_profile_ch = np.zeros(R_1)
    aggregated_colors_ch = []
    unique_clusters_in_day_ch = set()

    for i_1, t_1 in enumerate(exemplary_hours_1):
        ch_cluster_id = CH_mapping[t_1]
        aggregated_demand_profile_ch[i_1] = aggregated_demand_means_ch[ch_cluster_id]

        aggregated_wind_profile_ch[i_1] = aggregated_wind_means_ch[ch_cluster_id]
        unique_clusters_in_day_ch.add(ch_cluster_id)
        aggregated_colors_ch.append(ch_cluster_id)  # Store the cluster ID for coloring

    # Map cluster IDs to colors using a cyclical colormap for better visual distinction
    # Using 'tab10' for up to 10 distinct colors, or cycle if more
    num_unique_clusters_in_day = len(unique_clusters_in_day_ch)
    if num_unique_clusters_in_day <= 10 and num_unique_clusters_in_day > 0:
        cmap = plt.colormaps.get_cmap('tab10')
        # Create a consistent mapping from cluster ID to colormap index
        cluster_id_to_idx = {cid: i for i, cid in enumerate(sorted(list(unique_clusters_in_day_ch)))}
        color_map_func = lambda cluster_id: cmap(cluster_id_to_idx[cluster_id])
    elif num_unique_clusters_in_day > 0:
        # Cycle through a predefined list of colors if too many unique clusters in the day
        basic_colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_map_func = lambda cluster_id: basic_colors[list(unique_clusters_in_day_ch).index(cluster_id) % len(basic_colors)]  
    else:
        # Handle case with no clusters in the exemplary day (should not happen for R=24 usually)
        color_map_func = lambda cluster_id: 'black'  # Default color if no clusters

    final_aggregated_colors_ch = [color_map_func(cid) for cid in aggregated_colors_ch]

    # 5. Plotting
    fig_3, axes_3 = plt.subplots(1, 2, figsize=(19, 4))

    # Plot Demand
    axes_3[0].plot(range(1, R_1 + 1), original_demand_profile_ch, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Original Profile')
    axes_3[0].plot(range(1, R_1 + 1), aggregated_demand_profile_ch, color='black', linestyle='-', linewidth=2, zorder=4)  # Line connecting aggregated points
    scatter_demand_ch = axes_3[0].scatter(range(1, R_1 + 1), aggregated_demand_profile_ch, c=final_aggregated_colors_ch, s=70, marker='o', edgecolor='black', zorder=5)
    axes_3[0].set_title(f'Demand (MWh)', fontsize=16, weight='bold')
    axes_3[0].set_xlabel('Time (hours)', fontsize=16)
    axes_3[0].set_xticks(range(1, R_1 + 1, R_1 // 24))
    axes_3[0].tick_params(axis='both', labelsize=14)
    axes_3[0].grid(True, linestyle='--', alpha=0.7)
    axes_3[0].margins(x=0)

    # Create custom legend for clarity (similar to K-Means plot)
    legend_lines_handles_ch = [plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Original Profile'), plt.Line2D([0], [0], color='black', linestyle='-', marker='o', markerfacecolor='black', markeredgecolor='black', markersize=7, label='Aggregated Profile')]  
    first_legend_ch = axes_3[0].legend(handles=legend_lines_handles_ch, fontsize=12)
    axes_3[0].add_artist(first_legend_ch)
    legend_elements_colors_ch = []

    for cid in sorted(list(unique_clusters_in_day_ch)):
        legend_elements_colors_ch.append(plt.Line2D([0], [0], marker='o', color='w', label=f'{cid}', markerfacecolor=color_map_func(cid), markersize=10))

    # Plot Wind Capacity Factor
    axes_3[1].plot(range(1, R_1 + 1), original_wind_profile_ch, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Original Profile')
    axes_3[1].plot(range(1, R_1 + 1), aggregated_wind_profile_ch, color='black', linestyle='-', linewidth=2, zorder=4)
    scatter_wind_ch = axes_3[1].scatter(range(1, R_1 + 1), aggregated_wind_profile_ch, c=final_aggregated_colors_ch, s=70, marker='o', edgecolor='black', zorder=5)
    axes_3[1].set_title(f'Wind Capacity Factor (p.u.)', fontsize=16, weight='bold')
    axes_3[1].set_xlabel('Time (hours)', fontsize=16)
    axes_3[1].set_xticks(range(1, R_1 + 1, R_1 // 24))
    axes_3[1].set_xticklabels(range(1, R_1 + 1, R_1 // 24))
    axes_3[1].set_yticks(np.arange(0, 1.1, 0.2))
    axes_3[1].tick_params(axis='both', labelsize=12)
    axes_3[1].grid(True, linestyle='--', alpha=0.7)
    axes_3[1].margins(x=0)
    second_legend_ch = axes_3[1].legend(handles=legend_lines_handles_ch, fontsize=12)
    axes_3[1].add_artist(second_legend_ch)

    # Display plot
    plt.tight_layout()
    plt.show()
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

    rep_labels_shifted, rep_centroids_shifted, rep_mapping_shifted = rep_clustering(input_data,K_rep)
    rep_mapping = {i: int(rep_mapping_shifted[old_key]) for i, old_key in enumerate(sorted(rep_mapping_shifted.keys()))}

    # Display mapping
    print(rep_mapping)
    return rep_centroids_shifted, rep_labels_shifted, rep_mapping


@app.cell
def _(input_data, plt, rep_centroids_shifted, rep_labels_shifted):
    R_2 = 24  # Number of hours in a representative period (a day)

    # Reshape input data into daily profiles
    demand_profiles = input_data['Demand (MWh)'].values.reshape(-1, R_2)
    wind_profiles = input_data['Wind Capacity Factor (p.u.)'].values.reshape(-1, R_2)

    # Get all unique cluster IDs
    unique_cluster_ids = [5, 12, 15]

    # Choose a colormap for distinct colors for each cluster
    num_clusters = len(unique_cluster_ids)
    # Use a colormap suitable for categorical data, e.g., 'tab20' if num_clusters <= 20
    # If more, use 'rainbow' or a custom cycle
    cmap_1 = plt.colormaps.get_cmap('viridis').resampled(num_clusters) if num_clusters <= 20 else plt.colormaps.get_cmap('rainbow').resampled(num_clusters)

    # Create subplots (1 row, 2 columns)
    fig_4, axes_4 = plt.subplots(1, 2, figsize=(19, 4))  # Increased width for legend, height for better visibility

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
        centroid_demand = rep_centroids_shifted[current_cluster_id, :R_2]
        centroid_wind = rep_centroids_shifted[current_cluster_id, R_2:]

        # Extract member data for the chosen cluster
        member_demand_profiles = demand_profiles[member_day_indices]
        member_wind_profiles = wind_profiles[member_day_indices]

        # Plot Demand for the current cluster
        for i_2, day_demand in enumerate(member_demand_profiles):
            # Plot members with lighter color and higher transparency
            axes_4[0].plot(range(1, R_2 + 1), day_demand, color=cluster_color, linestyle='-', linewidth=0.8, alpha=0.4)

        # Plot centroid with a darker version of the color and a distinct marker
        line_demand, = axes_4[0].plot(range(1, R_2 + 1), centroid_demand, color=cluster_color, linestyle='-', marker='o', markersize=6, linewidth=2.5, zorder=5, label=f'Centroid {current_cluster_id}')
        centroid_handles.append(line_demand)
        centroid_labels.append(f'Centroid {current_cluster_id}')

        # Plot Wind Capacity Factor for the current cluster
        for i_2, day_wind in enumerate(member_wind_profiles):
            # Plot members with lighter color and higher transparency
            axes_4[1].plot(range(1, R_2 + 1), day_wind, color=cluster_color, linestyle='-', linewidth=0.8, alpha=0.4)

        # Plot centroid with a darker version of the color and a distinct marker
        line_wind, = axes_4[1].plot(range(1, R_2 + 1), centroid_wind, color=cluster_color, linestyle='-', marker='o', markersize=6, linewidth=2.5, zorder=5)  
        # No need to add to centroid_handles for wind as one common legend will be created

    # --- Common plot settings after all clusters are plotted ---

    # Demand plot settings
    axes_4[0].set_title(f'Demand (MWh) (Centroids vs. Members)', fontsize=16, weight='bold')
    axes_4[0].set_xlabel('Time (hours)', fontsize=16)
    axes_4[0].set_xticks(range(2, R_2 + 1, 2))
    axes_4[0].set_xticklabels(range(2, R_2 + 1, 2))
    axes_4[0].tick_params(axis='both', labelsize=14)  
    axes_4[0].grid(True, linestyle='--', alpha=0.7)
    axes_4[0].margins(x=0)

    # Wind plot settings
    axes_4[1].set_title(f'Wind Capacity Factor (p.u.) (Centroids vs. Members)', fontsize=16, weight='bold')  
    axes_4[1].set_xlabel('Time (hours)', fontsize=16)
    axes_4[1].set_xticks(range(2, R_2 + 1, 2))
    axes_4[1].set_xticklabels(range(2, R_2 + 1, 2))
    axes_4[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  
    axes_4[1].tick_params(axis='both', labelsize=14)
    axes_4[1].grid(True, linestyle='--', alpha=0.7)
    axes_4[1].margins(x=0)

    # Adjust rect to make space for the legend on the right
    plt.tight_layout(rect=[0, 0, 0.9, 1])  
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **5. Aggregated Generation Expansion Planning Model**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.1. Chronologically aggregated model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This subsection introduces the **chronologically aggregated generation expansion planning model**.

    Let $K$ denote the size of the set of clusters $\mathcal{K}$, indexed by $k=1, \dots, K$. Each time step $t=1, \dots, T$ is assigned to one of the $K$ clusters. Let $W_k$ denote the number of the original time steps assigned to the $k$-th cluster, and let $\mathcal{T}_k$ denote the set of consecutive time steps assigned to the $k$-th cluster.

    The goal of the aggregated GEP model is to determine the optimal values of the (aggregated) decision variables $\left\{\bar{x}^\mathrm{w}, \bar{x}^\mathrm{th}, \bar{x}^\mathrm{s},\bar{p}^\mathrm{w}_k, \bar{p}^\mathrm{th}_k,\bar{p}^\mathrm{d}_k,\bar{p}^\mathrm{c}_k, \bar{e}^\mathrm{ns}_k ,\bar{e}^\mathrm{s}_k\, |\, k \in \mathcal{K}\right\}$ that minimize the objective function $\bar{J}$ defined as

    $\displaystyle \bar{J} := C^{\mathrm{inv,w}} \bar{x}^\mathrm{w} + C^\mathrm{inv,th} \bar{x}^\mathrm{th} + C^\mathrm{inv,s}\bar{x}^\mathrm{s} + \sum_{k=1}^{K} W_k \Big( C^\mathrm{op,w} \bar{p}^\mathrm{w}_k \Delta + C^\mathrm{op,th} \bar{p}^\mathrm{th}_k \Delta + C^\mathrm{nse} \bar{e}^\mathrm{ns}_k + C^\mathrm{c,s} \bar{p}^\mathrm{c}_k \Delta + C^\mathrm{d,s} \bar{p}^\mathrm{d}_k \Delta \Big)$

    subject to the following constraints:

    * Thermal power generation limits: $0 \leq \bar{p}^\mathrm{th}_k \leq \bar{x}^\mathrm{th}, \, \forall k$.
    * Wind power generation limits: $0 \leq \bar{p}^\mathrm{w}_k \leq \bar{x}^\mathrm{w} \frac{1}{W_k} \sum_{t \in \mathcal{T}_k} CF^\mathrm{w}_t, \, \forall k$.
    * Energy balance constraints: $\left(\bar{p}^\mathrm{th}_k + \bar{p}^\mathrm{w}_k - \bar{p}^\mathrm{c}_k + \bar{p}^\mathrm{d}_k\right) \Delta + \bar{e}^\mathrm{ns}_k = \frac{1}{W_k} \sum_{t \in \mathcal{T}_k} D_t, \, \forall k$.
    * Energy storage state of charge dynamics: $\bar{e}^{\mathrm{s}}_{k+1} = \bar{e}^{\mathrm{s}}_k +\left(\eta^\mathrm{c,s} \bar{p}_{k}^\mathrm{c} - \frac{\bar{p}_{k}^\mathrm{d}}{\eta^\mathrm{d,s}}\right)W_k Δ, \forall k \in \mathcal{K} \setminus \{K\}$.
    * Energy storage boundary constraints: $\bar{e}^{\mathrm{s}}_{K} +\left(\eta^\mathrm{c,s} \bar{p}_{K}^\mathrm{c} - \frac{\bar{p}_{K}^\mathrm{d}}{\eta^\mathrm{d,s}}\right)W_KΔ = \bar{e}^{\mathrm{s}}_1$, <br>
    $\qquad\qquad\qquad\qquad\qquad\qquad\qquad \, \bar{e}^{\mathrm{s}}_1=0$.
    * Energy storage state of charge limits: $0\leq \bar{e}^s_k \leq \bar{x}^\mathrm{s}\tau, \forall k$.
    * Storage charging power limits: $0\leq \bar{p}^\mathrm{c}_k \leq \bar{x}^\mathrm{s}, \forall k$.
    * Storage discharging power limits: $0\leq \bar{p}^\mathrm{d}_k \leq \bar{x}^\mathrm{s}, \forall k$.
    """)
    return


@app.cell
def _(OPER_COST_STOR_CH, OPER_COST_STOR_DIS, STOR_EFF_CH, STOR_EFF_DIS, pyo):
    def create_aggregated_model(input_data, mapping, inv_cost_wind, inv_cost_thermal, oper_cost_wind,
                                oper_cost_thermal, oper_cost_nse, stor_etp, inv_cost_stor):

      # Create optimization model
      model = pyo.ConcreteModel(name="Aggregated Model")

      unique_clusters = sorted(mapping.values())
      unique_clusters = list(dict.fromkeys(unique_clusters))

      # Aggregate the parameters
      wind_cap_factor_mean = {}
      demand_mean = {}
      clusters_cardinalities = {}

      for k in unique_clusters:
        # Get indices for this cluster
        indices = [t for t, cluster_id in mapping.items() if cluster_id == k]

        if not indices:
            continue

        # Calculate Aggregated Parameters
        wind_cap_factor_mean[k] = input_data["Wind Capacity Factor (p.u.)"].iloc[indices].mean()
        demand_mean[k] = input_data["Demand (MWh)"].iloc[indices].mean()
        clusters_cardinalities[k] = len(indices)

      last_k = unique_clusters[-1]  # Define this for constraints
      first_k = unique_clusters[0]

      # Define sets
      model.K = pyo.Set(initialize=list(unique_clusters))  # Set with clusters

      # Define parameters
      model.aggregated_demand = pyo.Param(model.K, initialize=demand_mean)  # Demand (MWh)
      model.aggregated_capacity_factor = pyo.Param(model.K, initialize=wind_cap_factor_mean)  # Capacity factor (p.u.)
      model.clusters_cardinalitie = pyo.Param(model.K, initialize=clusters_cardinalities)  # Cluster cardinalities

      # Define decision variables
      model.x_thermal = pyo.Var(within=pyo.NonNegativeReals)  # Thermal installed capacity (MW)
      model.x_wind = pyo.Var(within=pyo.NonNegativeReals)  # Wind installed capacity (MW)
      model.x_storage = pyo.Var(within=pyo.NonNegativeReals) # Storage installed capacity (MW)
      model.aggregated_p_thermal = pyo.Var(model.K, within=pyo.NonNegativeReals)  # Thermal power generation (MW)
      model.aggregated_p_wind = pyo.Var(model.K, within=pyo.NonNegativeReals)  # Wind power generation (MW)
      model.aggregated_p_storage_ch = pyo.Var(model.K, within=pyo.NonNegativeReals)  # Storage charging power (MW)
      model.aggregated_p_storage_dis = pyo.Var(model.K, within=pyo.NonNegativeReals)  # Storage discharging power (MW)
      model.aggregated_e_SOC = pyo.Var(model.K, within=pyo.NonNegativeReals)  # Storage state of charge (MWh)
      model.aggregated_e_ns = pyo.Var(model.K, within=pyo.NonNegativeReals)  # Non-supplied energy demand (MWh)

      # Define constraints
      # 1. Thermal power generation limits
      def eThermal_Limits(mdl, k):
        return mdl.aggregated_p_thermal[k] <= mdl.x_thermal
      model.eThermal_Limits = pyo.Constraint(model.K, rule=eThermal_Limits)

      # 2. Wind power generation limits
      def eWind_Limits(mdl, k):
        return mdl.aggregated_p_wind[k] <= mdl.x_wind * mdl.aggregated_capacity_factor[k]
      model.eWind_Limits = pyo.Constraint(model.K, rule=eWind_Limits)

      # 3. Energy storage state of charge dynamics
      def eSOC(mdl, k):
        if k != last_k:
          return mdl.aggregated_e_SOC[k+1] == mdl.aggregated_e_SOC[k] + (STOR_EFF_CH * mdl.aggregated_p_storage_ch[k] - (mdl.aggregated_p_storage_dis[k] / STOR_EFF_DIS))*mdl.clusters_cardinalitie[k]
        return pyo.Constraint.Skip
      model.eSOC = pyo.Constraint(model.K, rule=eSOC)

      # 4. Storage boundary constraints
      def eSOC_Boundary_start(mdl):
        return mdl.aggregated_e_SOC[first_k] == 0
      model.eSOC_start = pyo.Constraint(rule=eSOC_Boundary_start)

      def eSOC_Boundary_end(mdl):
        return mdl.aggregated_e_SOC[last_k] + (STOR_EFF_CH * mdl.aggregated_p_storage_ch[last_k] - (mdl.aggregated_p_storage_dis[last_k] / STOR_EFF_DIS))*mdl.clusters_cardinalitie[last_k] == mdl.aggregated_e_SOC[first_k]
      model.eSOC_end = pyo.Constraint(rule=eSOC_Boundary_end)

      # 5. Storage state of charge limits
      def eSOC_Limits(mdl, k):
        return mdl.aggregated_e_SOC[k] <= mdl.x_storage * stor_etp
      model.eSOC_Limits = pyo.Constraint(model.K, rule=eSOC_Limits)

      # 6. Storage charging power limits
      def eStorage_Charging_Limits(mdl, k):
        return mdl.aggregated_p_storage_ch[k]<= mdl.x_storage
      model.eStorage_Charging_Limits = pyo.Constraint(model.K, rule=eStorage_Charging_Limits)

      # 7. Storage discharging power limits
      def eStorage_Discharging_Limits(mdl, k):
        return mdl.aggregated_p_storage_dis[k]<= mdl.x_storage

      model.eStorage_Discharging_Limits = pyo.Constraint(model.K, rule=eStorage_Discharging_Limits)

      # 8. Power balance constraint
      def ePower_Balance(mdl, k):
        return (mdl.aggregated_p_thermal[k] +
                mdl.aggregated_p_wind[k] -
                mdl.aggregated_p_storage_ch[k] +
                mdl.aggregated_p_storage_dis[k]) + \
                mdl.aggregated_e_ns[k] == mdl.aggregated_demand[k]

      model.ePower_Balance = pyo.Constraint(model.K, rule=ePower_Balance)

      # 9. Objective function
      def obj_rule(mdl):
        investment_costs = (inv_cost_wind * mdl.x_wind +
                            inv_cost_thermal * mdl.x_thermal +
                            inv_cost_stor * mdl.x_storage)
        operational_costs = sum(
            (oper_cost_wind * mdl.aggregated_p_wind[k] +
             oper_cost_thermal * mdl.aggregated_p_thermal[k] +
             OPER_COST_STOR_CH * mdl.aggregated_p_storage_ch[k] +
             OPER_COST_STOR_DIS * mdl.aggregated_p_storage_dis[k] +
             oper_cost_nse * mdl.aggregated_e_ns[k])
             * mdl.clusters_cardinalitie[k]
            for k in mdl.K
        )
        return investment_costs + operational_costs

      model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

      return model
    return (create_aggregated_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1.1. Run the aggregated model with **K-Means representative hours clusters**
    """)
    return


@app.cell
def _(
    INV_COST_STOR,
    INV_COST_THERMAL,
    INV_COST_WIND,
    OPER_COST_NSE,
    OPER_COST_THERMAL,
    OPER_COST_WIND,
    STOR_ETP,
    chronological_kmeans_mapping,
    create_aggregated_model,
    input_data,
    pyo,
    time,
):
    aggregated_model_kmeans = create_aggregated_model(input_data, chronological_kmeans_mapping, INV_COST_WIND, INV_COST_THERMAL, OPER_COST_WIND, OPER_COST_THERMAL, OPER_COST_NSE, STOR_ETP, INV_COST_STOR)

    # Solve pyomo model with highs
    _solver = pyo.SolverFactory('highs')
    start = time.time()
    res_1 = _solver.solve(aggregated_model_kmeans)
    end = time.time()
    print(f'Time taken: {end - start:.2f} seconds')

    # Check the status of the solution
    if res_1.solver.termination_condition == 'optimal':
        print(f'K-Means aggregated model optimal obj. fun. value = {pyo.value(aggregated_model_kmeans.obj) / 1000000.0:.2f} mln €')
    else:
        print('No optimal solution found.')
    return (aggregated_model_kmeans,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1.2. Run the aggregated model with **Chronological hierarchical clusters**
    """)
    return


@app.cell
def _(
    CH_mapping,
    INV_COST_STOR,
    INV_COST_THERMAL,
    INV_COST_WIND,
    OPER_COST_NSE,
    OPER_COST_THERMAL,
    OPER_COST_WIND,
    STOR_ETP,
    create_aggregated_model,
    input_data,
    pyo,
    time,
):
    aggregated_model_CH = create_aggregated_model(input_data, CH_mapping, INV_COST_WIND, INV_COST_THERMAL, OPER_COST_WIND, OPER_COST_THERMAL, OPER_COST_NSE, STOR_ETP, INV_COST_STOR)

    # Solve pyomo model with highs
    _solver = pyo.SolverFactory('highs')
    start_1 = time.time()
    res_2 = _solver.solve(aggregated_model_CH)
    end_1 = time.time()
    print(f'Time taken: {end_1 - start_1:.2f} seconds')

    # Check the status of the solution
    if res_2.solver.termination_condition == 'optimal':
        print(f'Chronological hierarchical aggregated model optimal obj. fun. value = {pyo.value(aggregated_model_CH.obj) / 1000000.0:.2f} mln €')
    else:
        print('No optimal solution found.')
    return (aggregated_model_CH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.2. Aggregated model with representative days
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This subsection introduces an aggregated model formulation based on representative days, following the methodology presented in Kotzur et al. $^{[1]}$.

    Let $\mathcal{D}$ denote the set of representative days, indexed by $d = 1, \dots, D$, and $\mathcal{R}$ denote the set of time steps within each representative day, indexed by $r = 1, \dots, R$. The set of original time horizon days $\mathcal{N}$ is indexed by $n= 1 ,.., N$, where the function $f:\mathcal{N}\rightarrow \mathcal{D}$ assigns a representative day to each original day. Additionally, the mappings of representative time steps to original time steps are denoted by $\mathcal{T}_{d,r}$. Finally, $W_{d}$ denotes the number of occurrences of representative day $d$.

    As the mapping is connecting non-consecutive days, the storage model needs to incorporate the storage state of charge dynamics within (intra) and between (inter) representative days. Intra-day dynamics are described with $\delta\bar{e}^\mathrm{s}_{d,r}$, which represents the difference in state of charge at time step $r$ relative to the beginning of representative day $d$. Inter-day dynamics are modeled with $\bar{e}^{\mathrm{s}}_n$ which represents the chronological state of charge across representative days.


    <div align="center">
      <img src="https://drive.google.com/uc?export=view&id=1FsYrAZa8Up_6ioV9LCTY7EXUXRxiji0X" width="500">
      <br>
      <i>Figure: Schematic diagram of the representative-day aggregated model. </i>
    </div>


    As before, the goal of the aggregated optimization model is to determine the optimal values of the (aggregated) decision variables $\left\{\bar{x}^\mathrm{w}, \bar{x}^\mathrm{th}, \bar{x}^\mathrm{s},\bar{p}^\mathrm{w}_{d,r}, \bar{p}^\mathrm{th}_{d,r},\bar{p}^\mathrm{d}_{d,r},\bar{p}^\mathrm{c}_{d,r}, \bar{e}^\mathrm{ns}_{d,r} ,\bar{e}^\mathrm{s}_{n}, \delta\bar{e}^\mathrm{s}_{d,r}  \, |\, d \in \mathcal{D}, r \in \mathcal{R}, n \in \mathcal{N}\right\}$ that minimize the objective function $\bar{J}$, defined as

    $\displaystyle \bar{J} := C^{\mathrm{inv,w}} \bar{x}^\mathrm{w} + C^\mathrm{inv,th} \bar{x}^\mathrm{th} + C^\mathrm{inv,s}\bar{x}^\mathrm{s} + \sum_{d=1}^{D} W_d \sum_{r=1}^{R} \Big( C^\mathrm{op,w} \bar{p}^\mathrm{w}_{d,r} \Delta + C^\mathrm{op,th} \bar{p}^\mathrm{th}_{d,r} \Delta + C^\mathrm{nse} \bar{e}^\mathrm{ns}_{d,r} + C^\mathrm{c,s} \bar{p}^\mathrm{c}_{d,r} \Delta + C^\mathrm{d,s}\bar{p}^\mathrm{d}_{d,r} \Delta \Big)$

    subject to:

    * Thermal power generation limits: $0 \leq \bar{p}^\mathrm{th}_{d,r} \leq \bar{x}^\mathrm{th}, \, \forall d,\forall r$.
    * Wind power generation limits: $0 \leq \bar{p}^\mathrm{w}_{d,r} \leq \bar{x}^\mathrm{w} \frac{1}{W_{d}} \sum_{t \in \mathcal{T}_{d,r} } CF^\mathrm{w}_t, \, \forall d,\forall r$.
    * Energy balance constraints: $\left(\bar{p}^\mathrm{th}_{d,r} + \bar{p}^\mathrm{w}_{d,r} - \bar{p}^\mathrm{c}_{d,r} + \bar{p}^\mathrm{d}_{d,r}\right) \Delta + \bar{e}^\mathrm{ns}_{d,r} = \frac{1}{W_d} \sum_{t \in \mathcal{T}_{d,r} } D_t, \, \forall d,\forall r$.
    * Intra-day energy storage dynamics constraints: $\delta\bar{e}^\mathrm{s}_{d,r+1} = \delta\bar{e}^\mathrm{s}_{d,r} +\left(\eta^\mathrm{c,s} \bar{p}_{d,r}^\mathrm{c} - \frac{\bar{p}_{d,r}^\mathrm{d}}{\eta^\mathrm{d,s}}\right)Δ, \forall d, \forall r \in \mathcal{R} \setminus \{R\}$,<br>
    $\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\,\,\,\,$$\delta\bar{e}^\mathrm{s}_{d,1}=0, \forall d$.
    * Inter-day energy storage dynamics constraints: $\bar{e}^{\mathrm{s}}_{n+1} = \bar{e}^{\mathrm{s}}_{n}+\delta\bar{e}^\mathrm{s}_{d=f(n),R}+ \left(\eta^\mathrm{c,s} \bar{p}_{d=f(n),R}^\mathrm{c} - \frac{\bar{p}_{d=f(n),R}^\mathrm{d}}{\eta^\mathrm{d,s}}\right)Δ\,,\forall n \in \mathcal{N} \setminus \{N\}$,<br>
    $\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\,\,\,\,$$\bar{e}^\mathrm{s}_{N}+\delta\bar{e}^\mathrm{s}_{d=f(N),R}+\left(\eta^\mathrm{c,s} \bar{p}_{d=f(N),R}^\mathrm{c} - \frac{\bar{p}_{d=f(N),R}^\mathrm{d}}{\eta^\mathrm{d,s}}\right)Δ=\bar{e}^\mathrm{s}_1$, <br>
    $\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad \,\,\,\, \bar{e}^{\mathrm{s}}_1=0$.
    * Energy storage state of charge limits: $0\leq \bar{e}^\mathrm{s}_{n} + \delta\bar{e}^\mathrm{s}_{d=f(n),r} \leq \bar{x}^\mathrm{s}\tau, \forall n, \forall r$.
    * Storage charging limits: $0\leq \bar{p}^\mathrm{c}_{d,r} \leq \bar{x}^\mathrm{s}, \forall d, \forall r$.
    * Storage discharging limits: $0\leq \bar{p}^\mathrm{d}_{d,r} \leq \bar{x}^\mathrm{s}, \forall d, \forall r$.

    ---
    <small>$^{[1]}$ L. Kotzur, P. Markewitz, M. Robinius, D. Stolten, "Time series aggregation for energy system design: Modeling seasonal storage," *Applied Energy*, vol. 213, pp. 123-135, Mar. 2018, doi:10.1016/j.apenergy.2018.01.023. [Link](https://www.sciencedirect.com/science/article/pii/S0306261918300242)
    """)
    return


@app.cell
def _(OPER_COST_STOR_CH, OPER_COST_STOR_DIS, STOR_EFF_CH, STOR_EFF_DIS, pyo):
    def create_rep_model(input_data, mapping, R,inv_cost_wind, inv_cost_thermal, oper_cost_wind,
                                oper_cost_thermal, oper_cost_nse, stor_etp,inv_cost_stor):

        # Data Aggregation
        total_periods = len(input_data) // R
        cf_matrix = input_data['Wind Capacity Factor (p.u.)'].values.reshape(total_periods, R)
        demand_matrix = input_data['Demand (MWh)'].values.reshape(total_periods, R)

        unique_clusters = sorted(set(mapping.values()))
        cluster_cf = {}
        cluster_demand = {}
        cluster_weights = {}

        for c_id in unique_clusters:
            n_indices = [n for n, cluster in mapping.items() if cluster == c_id]
            cluster_weights[c_id] = len(n_indices)
            # Mean across the days in the cluster for each hour r
            cluster_cf[c_id] = cf_matrix[n_indices, :].mean(axis=0)
            cluster_demand[c_id] = demand_matrix[n_indices, :].mean(axis=0)

        # Model Definition
        model = pyo.ConcreteModel(name="Aggregated_Model")

        # Define sets
        model.D = pyo.Set(initialize=unique_clusters)
        model.R = pyo.Set(initialize=range(R))
        model.N = pyo.Set(initialize=range(total_periods))

        # Define variables
        model.x_wind = pyo.Var(within=pyo.NonNegativeReals)  # Wind installed capacity (MW)
        model.x_thermal = pyo.Var(within=pyo.NonNegativeReals)  # Thermal installed capacity (MW)
        model.x_storage = pyo.Var(within=pyo.NonNegativeReals)  # Storage installed capacity (MW)
        model.p_th = pyo.Var(model.D, model.R, within=pyo.NonNegativeReals)  # Thermal power generation (MW)
        model.p_w = pyo.Var(model.D, model.R, within=pyo.NonNegativeReals)  # Wind power generation (MW)
        model.p_c = pyo.Var(model.D, model.R, within=pyo.NonNegativeReals)  # Storage charging power (MW)
        model.p_d = pyo.Var(model.D, model.R, within=pyo.NonNegativeReals)  # Storage discharging power (MW)
        model.e_ns = pyo.Var(model.D, model.R, within=pyo.NonNegativeReals)  # Non-supplied energy demand (MWh)
        model.delta_e_s = pyo.Var(model.D, model.R, within=pyo.Reals)  # Storage intra state of charge difference (MWh)
        model.e_s_inter = pyo.Var(model.N, within=pyo.NonNegativeReals)  # Storage inter-day state of charge (MWh)

        # Define constraints
        # 1. Thermal power generation limits
        model.ethermal_limit = pyo.Constraint(model.D, model.R,
            rule=lambda mdl, d, r: mdl.p_th[d, r] <= mdl.x_thermal)

        # 2. Wind power generation limits
        def rule_wind_limits(mdl,d, r):
            return mdl.p_w[d, r] <= mdl.x_wind * cluster_cf[d][r]
        model.ewind_limit = pyo.Constraint(model.D, model.R, rule=rule_wind_limits)

        # 4. Intra-day storage dynamics
        def rule_intra_storage(mdl,d,  r):
            if r == R - 1: return pyo.Constraint.Skip
            return mdl.delta_e_s[d, r+1] == mdl.delta_e_s[d, r] + \
                   (STOR_EFF_CH * mdl.p_c[d, r] - mdl.p_d[d, r] / STOR_EFF_DIS)
        model.eintra_storage = pyo.Constraint(model.D, model.R, rule=rule_intra_storage)

        model.eintra_start = pyo.Constraint(model.D,
            rule=lambda mdl, d: mdl.delta_e_s[d, 0] == 0)

        # 5. Inter-day storage dynamics and boundaries
        def rule_inter_storage(mdl, n):
          day_net_change = sum(STOR_EFF_CH * mdl.p_c[mapping[n], r] -
                              mdl.p_d[mapping[n], r] / STOR_EFF_DIS for r in mdl.R)

          if n == max(model.N):
              # Cyclic constraint for the end of the year
              return mdl.e_s_inter[n] + day_net_change == mdl.e_s_inter[0]

          # SOC at start of next day = SOC at start of current day + net change today
          return mdl.e_s_inter[n+1] == mdl.e_s_inter[n] + day_net_change

        model.einter_storage = pyo.Constraint(model.N, rule=rule_inter_storage)
        model.einter_start = pyo.Constraint(rule=lambda mdl: mdl.e_s_inter[0] == 0)

        # 6. Storage SOC limits (Inter + Intra)
        def rule_soc_limits(mdl, n, r):
            return mdl.e_s_inter[n] + mdl.delta_e_s[mapping[n], r] <= mdl.x_storage * stor_etp
        model.esoc_limits = pyo.Constraint(model.N, model.R, rule=rule_soc_limits)
        def rule_soc_non(mdl, n, r):
            return mdl.e_s_inter[n] + mdl.delta_e_s[mapping[n], r] >= 0
        model.esoc_non = pyo.Constraint(model.N, model.R, rule=rule_soc_non)

        # 7. Charging power limits
        model.echarge_limit = pyo.Constraint(model.D, model.R,
            rule=lambda mdl,d,  r: mdl.p_c[d, r] <= mdl.x_storage)

        # 8. Discharging power limits
        model.edischarge_limit = pyo.Constraint(model.D, model.R,
            rule=lambda mdl,d,  r: mdl.p_d[d, r] <= mdl.x_storage)

        # 9. Power balance constraint
        def rule_energy_balance(mdl,d,  r):
            return (mdl.p_th[d, r] + mdl.p_w[d, r] - mdl.p_c[d, r] + mdl.p_d[d, r]) + \
                   mdl.e_ns[d, r] == cluster_demand[d][r]
        model.ePower_balance = pyo.Constraint(model.D, model.R, rule=rule_energy_balance)

        # 10. Objective Function
        def rule_objective(mdl):
            inv = inv_cost_wind * mdl.x_wind + inv_cost_thermal * mdl.x_thermal + inv_cost_stor * mdl.x_storage

            oper = sum(cluster_weights[d] * sum(
                (oper_cost_wind * mdl.p_w[d, r] +
                 oper_cost_thermal * mdl.p_th[d, r] +
                 oper_cost_nse * mdl.e_ns[d, r] +
                 OPER_COST_STOR_CH * mdl.p_c[d, r] +
                 OPER_COST_STOR_DIS * mdl.p_d[d, r])
                for r in mdl.R) for d in mdl.D)
            return inv + oper

        model.obj = pyo.Objective(rule=rule_objective, sense=pyo.minimize)

        return model
    return (create_rep_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2.1. Run **Representative day aggregated model**
    """)
    return


@app.cell
def _(
    INV_COST_STOR,
    INV_COST_THERMAL,
    INV_COST_WIND,
    OPER_COST_NSE,
    OPER_COST_THERMAL,
    OPER_COST_WIND,
    STOR_ETP,
    create_rep_model,
    input_data,
    pyo,
    rep_mapping,
    time,
):
    rep_model = create_rep_model(input_data, rep_mapping, 24, INV_COST_WIND, INV_COST_THERMAL, OPER_COST_WIND, OPER_COST_THERMAL, OPER_COST_NSE, STOR_ETP, INV_COST_STOR)

    # Solve pyomo model with highs
    _solver = pyo.SolverFactory('highs')
    start_2 = time.time()
    res_3 = _solver.solve(rep_model)
    end_2 = time.time()
    print(f'Time taken: {end_2 - start_2:.2f} seconds')

    # Check the status of the solution
    if res_3.solver.termination_condition == 'optimal':
        print(f'K-Means aggregated model optimal obj. fun. value = {pyo.value(rep_model.obj) / 1000000.0:.2f} mln €')
    else:
        print('No optimal solution found.')
    return (rep_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **6. Evaluation**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once the **mappings** have been generated using the clustering techniques, the following `evaluate_mapping` function is used to assess the quality of the aggregated solution.

    This function computes the optimal objective value of the aggregated model introduced in Section 5 using the generated mapping. It then computes the relative difference between this value and the optimal objective value of the full-scale model introduced in Section 3, referred to as the **output error**.

    To analyze the investment decisions, the function `plot_investments_results` is used. The function generates a bar plot comparing the optimal investment capacities obtained from the full-scale and aggregated models.
    """)
    return


@app.cell
def _(
    INV_COST_STOR,
    INV_COST_THERMAL,
    INV_COST_WIND,
    OPER_COST_NSE,
    OPER_COST_THERMAL,
    OPER_COST_WIND,
    STOR_ETP,
    create_aggregated_model,
    create_rep_model,
    input_data,
    np,
    plt,
    pyo,
):
    def evaluate_mapping(mapping,full_model,aggregated_model = None):
      if aggregated_model is None:
        temp = len(mapping)
        if temp == 364:
          aggregated_model = create_rep_model(input_data, mapping, 24,INV_COST_WIND, INV_COST_THERMAL, OPER_COST_WIND, OPER_COST_THERMAL, OPER_COST_NSE, STOR_ETP, INV_COST_STOR)
        else:
          aggregated_model =(create_aggregated_model(
          input_data, mapping, INV_COST_WIND, INV_COST_THERMAL, OPER_COST_WIND,
          OPER_COST_THERMAL, OPER_COST_NSE, STOR_ETP, INV_COST_STOR))

        # Solve pyomo model with highs
        solver = pyo.SolverFactory('highs')
        res = solver.solve(aggregated_model)

      return (pyo.value(full_model.obj) - pyo.value(aggregated_model.obj)) / pyo.value(full_model.obj) * 100

    def plot_investments_results(mapping,full_model,aggregated_model = None):
      if aggregated_model is None:
        temp = len(mapping)
        if temp == 364:
          aggregated_model = create_rep_model(input_data, mapping, 24, INV_COST_WIND, INV_COST_THERMAL, OPER_COST_WIND, OPER_COST_THERMAL, OPER_COST_NSE, STOR_ETP, INV_COST_STOR)
        else:
          aggregated_model =(create_aggregated_model(
          input_data, mapping, INV_COST_WIND, INV_COST_THERMAL, OPER_COST_WIND,
          OPER_COST_THERMAL, OPER_COST_NSE, STOR_ETP, INV_COST_STOR))

        # Solve pyomo model with highs
        solver = pyo.SolverFactory('highs')
        res = solver.solve(aggregated_model)

      investments_fs = [full_model.x_thermal(), full_model.x_wind(), full_model.x_storage()]
      investments_agg = [aggregated_model.x_thermal(), aggregated_model.x_wind(), aggregated_model.x_storage()]
      labels = ['Thermal', 'Wind', 'Storage']
      x = np.arange(len(labels))

      colors = ['#1f77b4', '#2ca02c']
      width = 0.35

      fig, ax = plt.subplots(figsize=(9, 4))

      rects1 = ax.bar(x - width/2, investments_fs, width, label='Full-Scale Model',
                        color=colors[0], edgecolor='black', linewidth=1.5)
      rects2 = ax.bar(x + width/2, investments_agg, width, label='Aggregated Model',
                        color=colors[1], edgecolor='black', linewidth=1.5)

      def autolabel(rects):
          for rect in rects:
              height = rect.get_height()
              ax.annotate(f'{height:.2f}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 5),  # Increased offset slightly
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=14, fontweight='bold')

      autolabel(rects1)
      autolabel(rects2)


      max_height = max(max(investments_fs), max(investments_agg))
      ax.set_ylim(0, max_height * 1.15)

      ax.set_ylabel('Investment capacity (MW)', fontsize=16, labelpad=10)
      ax.set_title('Investment Results: Full-scale vs. Aggregated Model', fontsize=16,weight='bold')
      ax.set_xticks(x)
      ax.set_xticklabels(labels, fontsize=16)
      ax.tick_params(axis='both', labelsize=16)

      ax.legend(fontsize=16, frameon=True, loc='upper left')

      ax.yaxis.grid(True, linestyle='--', alpha=0.7)
      ax.set_axisbelow(True)

      plt.tight_layout()
      plt.show()

    def length(mapping):
      temp = len(mapping)
      if temp == 364:
        return len(set(mapping.values())) * 24
      else:
        return len(set(mapping.values()))
    return evaluate_mapping, length, plot_investments_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.1. Evaluate **K-Means representative hours clustering**
    """)
    return


@app.cell
def _(
    aggregated_model_kmeans,
    chronological_kmeans_mapping,
    evaluate_mapping,
    full_model,
):
    # Calculate Output error
    print(f"Relative output error = {evaluate_mapping(chronological_kmeans_mapping,full_model,aggregated_model_kmeans):.2f} %")
    return


@app.cell
def _(
    aggregated_model_kmeans,
    chronological_kmeans_mapping,
    full_model,
    plot_investments_results,
):
    # Visualize investment results
    plot_investments_results(chronological_kmeans_mapping, full_model, aggregated_model_kmeans)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.2. Evaluate **Chronological hierarchical clustering**
    """)
    return


@app.cell
def _(CH_mapping, aggregated_model_CH, evaluate_mapping, full_model):
    # Calculate Output error
    print(f"Relative output error = {evaluate_mapping(CH_mapping, full_model, aggregated_model_CH):.2f} %")
    return


@app.cell
def _(CH_mapping, aggregated_model_CH, full_model, plot_investments_results):
    # Visualize investment results
    plot_investments_results(CH_mapping,full_model,aggregated_model_CH)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.3. Evaluate **Representative day aggregation**
    """)
    return


@app.cell
def _(evaluate_mapping, full_model, kmeans_mapping, rep_model):
    # Calculate Output error
    print(f"Relative output error = {evaluate_mapping(kmeans_mapping, full_model, rep_model):.2f} %")
    return


@app.cell
def _(full_model, plot_investments_results, rep_mapping, rep_model):
    # Visualize investment results
    plot_investments_results(rep_mapping,full_model,rep_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **7. Challenges**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7.1. Create your own mapping (Maximum of 500 time steps)
    Now it's your turn to cluster input time series for optimization!

    **Challenge Objective**

    Your task is to aggregate the `input_data` time series in order to minimize the **output error** of one of the aggregated GEP models introduced in Section 5. Produce a mapping similar to the mappings from Section 4. The objective is to achieve the lowest possible **output error** using **no more than 500 representative time steps.** Evaluate your mapping using the evaluation procedure introduced in the previous section, and compare the resulting investment decisions with those of the full-scale model.

    **Rules**:

    * Submission with the **lowest output error** will be ranked highest, provided that the **methodology is clearly justified**.
    * **Do not modify the existing code cells** in this notebook. If additional coding is required, use the designated workspace below.

    **Please submit your final output error using the following form: [Submission form](https://docs.google.com/forms/d/e/1FAIpQLSelQAIpdcc9bSgenZ5eM1auYsJ5Qx6dZ7qb8iz3pbYBPYEEYg/viewform?usp=publish-editor)**
    """)
    return


@app.cell
def _(
    evaluate_mapping,
    full_model,
    length,
    mo,
    my_mapping,
    plot_investments_results,
):
    # Evaluate your mapping
    size = length(my_mapping)
    if size > 500:
      print("You exceeded 500 timesteps!")
      print(f"Size of aggregated model: {size}")
      print(f"Relative output error of the aggregated model: {evaluate_mapping(my_mapping,full_model):.2f} %")
      mo.stop()
    else:
      print(f"Size of aggregated model: {size}")
      print(f"Relative output error of the aggregated model: {evaluate_mapping(my_mapping,full_model):.2f} %")
      # Visualize the optimal investment results for your mapping
      plot_investments_results(my_mapping,full_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Feeling a bit lost? Here are some suggestions to guide you:

    * You may reduce the number of time steps used in the chronological hierarchical clustering approach or decrease the number of representative days to remain below the limit of 500 representative time steps.
    * You may adapt the `CH_clustering` function from Section 4 and modify it by altering its *distance measure*.
    * Alternatively, you may implement other clustering techniques available in [clustering techniques](https://scikit-learn.org/stable/modules/clustering.html).
    * You could also download and analyze the [input data](https://github.com/JakubRybka/Tutorial_input_data.git), then assign time steps to **the 500 representative time steps** based on a custom policy (e.g., assign a time step $t$ to a specific cluster $k$ if the demand at time $t$ exceeds a certain threshold).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7.2. Time Series Aggregation with performance guarantees
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now it’s your turn to develop bounds for time series aggregation in optimization!

    The time series aggregation is commonly used to estimate the full-scale model solution. In a real-world setting, the optimal value is unknown, and heuristic aggregation methods provide no direct measure of how close the aggregated solution is to the true optimum. However, it has been shown that the aggregated model (as described in Section 5.1) provides a lower bound for the full-scale solution $^{[1]}$.

    **Challenge Objective:**

    Your task is to develop a method to derive an upper bound for the true solution. Your objective is then to apply this method to an aggregated model, and achieving an optimality gap of $\le 3\%$ while minimizing the number of representative time steps.

    $ϵ:=\left(\frac{\mathrm{UpperBound}−\mathrm{LowerBound}}{\mathrm{UpperBound}}\right)×100\%$


    **Rules:**

    - The group achieving **the smallest number of representative time steps** with an **optimality gap $≤3\%$ wins**.
    - The full-scale solution **may not be used.**
    - Be prepared to **justify** your methodology.
    - Do not modify any code snippets in this notebook. If coding is necessary, use the designated space below.

    **Submission:**

    Please submit the number of used representative time steps here: [Submission form](https://forms.gle/Cyi6Zff6tn2mpRxi6)

    ---
    <small> $^{[1]}$ L. Santosuosso, B. Klinz, and S. Wogrin, "What Are We Clustering For? Establishing Performance Guarantees for Time Series Aggregation in Generation Expansion Planning," 2025, arXiv:2510.09357. [Link](https://arxiv.org/abs/2510.09357)
    """)
    return


@app.cell
def _(chronological_kmeans_mapping):
    # Here implement your solution for obtaining Lower and Upper Bounds of optimal objective function value
    my_mapping_1 = chronological_kmeans_mapping
    Lower_Bound = 0
    Upper_Bound = 100
    return Lower_Bound, Upper_Bound, my_mapping_1


@app.cell
def _(Lower_Bound, Upper_Bound, length, my_mapping_1):
    # Evaluate your solution
    print(f'Optimality gap = {(Upper_Bound - Lower_Bound) / Upper_Bound * 100:.2f}%')
    print(f'Number of representative time steps = {length(my_mapping_1)}')
    return


if __name__ == "__main__":
    app.run()
