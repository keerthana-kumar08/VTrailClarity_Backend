import pandas as pd
import numpy as np
import pymc as pm

# import antigravity as az
import arviz as az
import seaborn as sns
from io import BytesIO
from pingouin import corr
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import ttest_ind
from.utils import convert_numpy_types, clean_data_for_json, get_bayes_interp, interpolate_color, generate_graph_image, v1_get_bayes_interp



async def case_1_graph(df, data):
    try:
        df_copy = df.copy()
        result = data['result']
        bayes_pairwise_corr_comparisons = {}
        basefactor_data = {}
        output = [0, result, {'style': 'class/data', 'class': data['categorical_col'], 'data': data['continuous_col']}]

        if output[0] == 1:
            raise Exception(output[1])
        else:
            if output[-1] == 'allcat':
                res_dict = {}
                res_dict['variant'] = 'allcat'
                res_dict['bf_dict'] = output[1]
                res_dict['keys'] = list(res_dict['bf_dict'].keys())
                res_dict['cols'] = output[2]
                res_dict['colour'] = {}
                res_dict['interp'] = {}
                res_dict['bf01_dict'] = {} 
                for key in res_dict['bf_dict']:
                    res_dict['bf01_dict'][key] = round(1 / float(res_dict['bf_dict'][key]), 2)
                    res_dict['colour'][key] = interpolate_color(res_dict['bf01_dict'][key])
                    res_dict['interp'][key] = v1_get_bayes_interp(1, res_dict['bf01_dict'][key]).capitalize()
            else:
                res_dict = output[1]
                res_dict['variant'] = 'normal'
                res_dict['cols'] = output[2]
                
                # Handle both t-test and ANOVA cases
                if 'BF10' in res_dict:  # t-test case
                    bf01_value = float(res_dict['BF10'].get('Bayes Factor', 1.0))
                    res_dict['BF01'] = bf01_value
                    res_dict['BF10_rounded'] = round(bf01_value, 2)   # USE BF10 as bayes factor
                    basefactor_data = {
                        "bayes_factors": bf01_value,
                        "bayes_interp": v1_get_bayes_interp(1, bf01_value).capitalize()
                        }
                elif 'pairwise_comparisons' in res_dict:  # ANOVA case
                    bayes_pairwise_corr_comparisons = {
                        "corr_bayes_factors_data": res_dict['bayes_pairwise_corr_comparisons']['pairwise_corr_bayes_factor'],
                        "bar_chart":res_dict['bayes_pairwise_corr_comparisons']['graph_points']
                        }
                
                res_dict['colour'] = interpolate_color(res_dict.get('BF01', 1.0))
                res_dict['interp'] = v1_get_bayes_interp(1, res_dict.get('BF01', 1.0)).capitalize()


        style = res_dict['cols']['style']
        class_col = res_dict['cols']['class']
        data_col = res_dict['cols']['data']


        df_classes = []  # List to store dataframes
        if style == 'class/data' and class_col and data_col:
            unique_classes = df_copy[class_col].dropna().unique()  # Get all unique class values
            
            if len(unique_classes) >= 2:
                for class_value in unique_classes:
                    df_class = df_copy[df_copy[class_col] == class_value][[data_col]].copy()  # Ensure DataFrame
                    df_class['class'] = class_value  # Assign class label
                    df_classes.append(df_class)  # Store in list
            else:
                return 1, 'ERROR! Not enough unique classes in class column.'
        else:
            return 1, 'ERROR! Data not in proper format.'
        df = pd.concat(df_classes, ignore_index=True)  # Merge all class dataframes
        df_long = pd.melt(df, id_vars='class', var_name='time', value_name='value')

        # await generate_graph_image(df_long)

        # Changed: Storing graph data instead of saving images
        graph_data = {
            "pointplot": df_long.groupby(['time', 'class'])['value'].agg(['mean', 'std', 'count']).reset_index().to_dict(orient='records'),
            "barplot": df_long.groupby(['time', 'class'])['value'].agg(['mean', 'std']).reset_index().to_dict(orient='records'),
            "boxplot": df_long.groupby(['time', 'class'])['value'].apply(list).reset_index().to_dict(orient='records'),
            "stripplot": df_long.to_dict(orient='records')
        }

        result = {
            "status" : 200,
            "message": "Success",
            "data": {
                "file_name": "",
                "type": data["type"],
                "bayes_factors_data": basefactor_data,
                "bayes_pairwise_corr_comparisons":bayes_pairwise_corr_comparisons,
                "output": convert_numpy_types(result),   # dont need ask raja and remove
                "graph_points": convert_numpy_types(graph_data)
            }        
        }

        result = clean_data_for_json(result)
        return result
    except Exception as e:
        raise ValueError(f"Error parsing XML: {e}")

async def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x.tolist(), y.tolist()

async def case_3_graph(drug, placebo, drug_col_name, placebo_col_name, value):
    x_drug, y_drug = await ecdf(drug)
    x_placebo, y_placebo = await ecdf(placebo)
    
    # Plot ECDF Graph
    fig, ax = plt.subplots()
    ax.plot(x_drug, y_drug, label=f'{drug_col_name}, n={len(drug)}')
    ax.plot(x_placebo, y_placebo, label=f'{placebo_col_name}, n={len(placebo)}')
    ax.legend(), ax.set_xlabel('IQ Score'), ax.set_ylabel('Cumulative Frequency')
    ax.hlines(0.5, ax.get_xlim()[0], ax.get_xlim()[1], linestyle='--')
    # fig.savefig("ecdf_plot.png")
    plt.close(fig)

    t_stat, p_value = ttest_ind(drug, placebo)
    
    try:
        correlation = corr(drug, placebo, alternative='two-sided', method='pearson')
        bf10 = float(correlation['BF10'].iloc[0]) if 'BF10' in correlation and not correlation['BF10'].empty else None
        # bayes_interp = v1_get_bayes_interp(3, bf10).capitalize() if bf10 else "BF10 not available"
    except Exception as e:
        bf10 = None
    
    with pm.Model() as model:
        mu_drug, mu_placebo = pm.Normal('mu_drug', 0, 100**2), pm.Normal('mu_placebo', 0, 100**2)
        sigma_drug, sigma_placebo = pm.HalfCauchy('sigma_drug', 100), pm.HalfCauchy('sigma_placebo', 100)
        nu = pm.Exponential('nu', 1/29) + 1
        
        pm.StudentT('drug', nu=nu, mu=mu_drug, sigma=sigma_drug, observed=drug)
        pm.StudentT('placebo', nu=nu, mu=mu_placebo, sigma=sigma_placebo, observed=placebo)
        
        diff_means, pooled_sd = pm.Deterministic('diff_means', mu_drug - mu_placebo), pm.Deterministic('pooled_sd', np.sqrt((sigma_drug**2 + sigma_placebo**2) / 2))
        effect_size = pm.Deterministic('effect_size', diff_means / pooled_sd)
    
    with model:
        trace = pm.sample(2000, cores=2, return_inferencedata=True)
    
    # Trace Plot Graph
    posterior = trace.posterior
    if "draw" in posterior.dims:
        burn_in = posterior.sizes["draw"] // 2
        if burn_in < posterior.sizes["draw"]:
            sliced_posterior = posterior.sel(draw=slice(burn_in, None))
            pm.plot_trace(sliced_posterior,['mu_drug', 'mu_placebo'])
            # plt.savefig("trace_plot.png")
            plt.close(fig)
    
    # Bayesian Posterior Graph
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    pm.plot_posterior(trace, color='#87ceeb', var_names=['mu_drug', 'mu_placebo', 'diff_means', 'pooled_sd'], ax=axes.flatten())
    fig.tight_layout()
    # fig.savefig("bayesian_posterior.png")
    plt.close(fig)

    pval = value.get("result", {}).get("pval")
    bayes_interp = v1_get_bayes_interp(3, pval).capitalize() if bf10 else "BF10 not available"
    
    return {
        "status": 200, "message": "Success", "data": {
            "type": "two_continuous_variables",
            "output": {
                "bayesian_value_corr": {"BF10": bf10, "bayes_interp": bayes_interp},
                "t_test": {"t_statistic": t_stat, "p_value": p_value},
                "correlation": value.get("result", {})
            },
            "graph_points": {
                "ecdf_plot": {"drug": {"x": x_drug, "y": y_drug}, "placebo": {"x": x_placebo, "y": y_placebo}},
                "trace_plot": sliced_posterior[['mu_drug', 'mu_placebo']].to_dict() if sliced_posterior else None,
                "bayesian_posterior": {
                    "diff_means_samples": posterior['diff_means'].values.flatten()[::50].tolist(),
                    "effect_size_samples": posterior['effect_size'].values.flatten()[::50].tolist()
                }
            }
        }
    }

async def case_4_graph(data, result, save_paths={'bar_chart': 'coin_toss_graph.png', 'trace': 'trace_plot.png', 'posterior': 'posterior_distribution.png'}):
    # Set plotting style
    plt.style.use('fivethirtyeight')
    sns.set_style('white')
    sns.set_context('poster')

    tosses = data

    def plot_coins():
        fig, ax = plt.subplots()
        counts = Counter(tosses)
        ax.bar(counts.keys(), counts.values(), color=['blue', 'green'])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Tails', 'Heads'])
        ax.set_ylim(0, 20)
        ax.set_yticks(np.arange(0, 21, 5))
        ax.set_ylabel("Frequency")
        ax.set_title("Coin Toss Outcomes")
        return fig, counts

    # Plot and save the coin toss results
    fig, counts = plot_coins()
    # fig.savefig(save_paths['bar_chart'])
    plt.close(fig)

    # Define Bayesian model for coin toss
    with pm.Model() as coin_model:
        p_prior = pm.Uniform('p', 0, 1)
        likelihood = pm.Bernoulli('likelihood', p=p_prior, observed=tosses)
        step = pm.Metropolis()
        coin_trace = pm.sample(2000, step=step, return_inferencedata=True)

    # Save trace plot
    fig_trace = az.plot_trace(coin_trace)
    # plt.savefig(save_paths['trace'])
    plt.close()

    # Extract posterior samples
    posterior = coin_trace.posterior['p'].values.flatten()

    # Slice posterior from the 100th sample onward
    sliced_posterior = posterior[100:]

    # Store histogram data for frontend use
    hist_values, bin_edges = np.histogram(sliced_posterior, bins=20)
    posterior_data = {
        "hist_values": hist_values.tolist(),
        "bin_edges": bin_edges.tolist(),
        "mean": float(np.mean(sliced_posterior)),
        "median": float(np.median(sliced_posterior)),
        "mode": float(bin_edges[np.argmax(hist_values)]),
    }

    # Save posterior distribution plot
    fig_posterior = plt.figure()
    # pm.plot_posterior(sliced_posterior, color='#87ceeb', rope=[0.48, 0.52], point_estimate='mean', ref_val=0.5)
    # plt.savefig(save_paths['posterior'])
    # plt.close(fig_posterior)

    return {
            "status" : 200,
            "message": "Success",
            "data": {
                "type": "two_category_one_continous",
                "output":result,
                "graph_points": {
                    "bar_chart": dict(counts),
                    "posterior_distribution": posterior_data
                }
                
            }        
        }

