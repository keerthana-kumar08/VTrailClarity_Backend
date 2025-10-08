import itertools, json
import numpy as np
import pandas as pd

import pingouin as pg
from pingouin import ttest, corr
import matplotlib.pyplot as plt
from fastapi import HTTPException
from pingouin import kruskal as pingouin_kruskal, bayesfactor_binom
from scipy.stats import chi2_contingency, kruskal, pearsonr
from ..utils.utils import get_bayes_message, sanitize_bf_value
from .genrate_graph import get_bayes_interp, v1_get_bayes_interp


async def case_1_continuous_categorical(df):
    try:
        # Identify the continuous and categorical columns
        continuous_col = [col for col in df.columns if df[col].dtype.name in ['float64', 'int64', 'Int64']][0]
        categorical_col = [col for col in df.columns if df[col].dtype.name == 'object'][0]

        # Convert continuous column to float if needed
        df[continuous_col] = pd.to_numeric(df[continuous_col], errors='coerce')

        # Split the continuous column into groups based on the categorical column
        groups = df.groupby(categorical_col)[continuous_col].apply(list)
        
        # Check the number of unique groups
        num_groups = len(groups)

        if num_groups < 2:
            raise HTTPException(status_code=400, detail=f"At least 2 groups are required for analysis. Found {num_groups} groups.")
    
        if num_groups == 2:
            # Perform the independent t-test for exactly 2 groups
            ttest_result = ttest(groups.iloc[0], groups.iloc[1], paired=False)
            # Calculate Bayes Factor using pingouin
            bayes_result = ttest(groups.iloc[0], groups.iloc[1], paired=False, alternative='two-sided')

            raw_bf10 = bayes_result['BF10'].values[0]
            try:
                bf10_value = float(raw_bf10) if raw_bf10 is not None else None
            except (ValueError, TypeError):
                bf10_value = None
            bf01_value = bf10_value
            if bf10_value not in (None, 0):
                bf01_value = round(1 / bf10_value, 2)
            return {
                "status": "success",
                "continuous_col": continuous_col, 
                "categorical_col": categorical_col, 
                "result": {
                    "T": {"T-test": ttest_result['T'].values[0]},
                    "dof": {"T-test": ttest_result['dof'].values[0]},
                    "alternative": {"T-test": "two-sided"},
                    "p-val": {"T-test": ttest_result['p-val'].values[0]},
                    "CI95%": {"T-test": ttest_result['CI95%'].values[0].tolist()},
                    "cohen-d": {"T-test": ttest_result['cohen-d'].values[0]},
                    "BF10": {"Bayes Factor": bf01_value},  # kept BF01 in BF10 key 
                    "power": {"T-test": ttest_result['power'].values[0]}
                },
                "type": "continuous_categorical"
            }
        
        else:
            # Perform ANOVA for more than 2 groups
            df[categorical_col] = df[categorical_col].astype("category")

            if len(df[continuous_col].unique())==1:
                raise HTTPException(status_code=400, detail=f"All the continuous column value should not be same")
        
            anova_result = pg.anova(data=df, dv=continuous_col, between=categorical_col)

            threshold = 2  # Set minimum occurrences
            rare_categories = df[categorical_col].value_counts()[df[categorical_col].value_counts() < threshold].index
            df[categorical_col] = df[categorical_col].astype(str)
            df[categorical_col] = df[categorical_col].replace(rare_categories.tolist(), "others")

            # Perform pairwise comparisons with Bayes Factors
            pairwise_results = pg.pairwise_tests(data=df, dv=continuous_col, between=categorical_col, padjust='bonf', effsize='cohen', alternative='two-sided')

            bf_dict = {}
            for _, row in pairwise_results.iterrows():
                key = f"{row['A'].strip()} | {row['B'].strip()}"
                bf10 = float(row['BF10']) if 'BF10' in row and not pd.isna(row['BF10']) else None
                if bf10 is not None and np.isinf(bf10):
                    bf10 = 1e308
                bf01_value = None
                if bf10 not in (None, 0):
                    bf01_value = round(1 / bf10, 2)
                bayes_interp = v1_get_bayes_interp(1, bf01_value).capitalize() if bf01_value is not None else "BF01 not available"
                bayesian_value_corr = {"BF10": bf01_value, "bayes_interp": bayes_interp}
                bf_dict[key] = bayesian_value_corr

            # Group by the categorical column and collect continuous values as lists
            grouped_data = df.groupby(categorical_col)[continuous_col].apply(list).to_dict()
            max_len = max(len(lst) for lst in grouped_data.values())
            # Pad shorter lists with NaN
            for key in grouped_data:
                grouped_data[key] += [np.nan] * (max_len - len(grouped_data[key]))
            df_grouped = pd.DataFrame(grouped_data).dropna()
            means = df_grouped.mean()
            sds = df_grouped.std()


            # pairwise_corr_results = pg.pairwise_corr(df_grouped, method="pearson", alternative="two-sided") # only support greater then 4 values in the pair other wish it will ignore that column pair check
            # print("pairwise_corr_results", pairwise_corr_results)
            # pairwise_corr_bayes_factor = {}
            # for _, row in pairwise_corr_results.iterrows():
            #     col1, col2 = row['X'], row['Y']
            #     if 'BF10' in row and not pd.isna(row['BF10']):
            #         bf10 = float(row['BF10'])
            #         if np.isinf(bf10):  # Handle Infinity values
            #             bf10 = 1e308  # Replace with a large number
            #     else:
            #         bf10 = None
            #     bayes_interp = v1_get_bayes_interp(1, bf10).capitalize() if bf10 else "BF10 not available"
                
            #     bayesian_value_corr = {"BF10": bf10, "bayes_interp": bayes_interp}
            #     pairwise_corr_bayes_factor[f"{col1} | {col2}"] = bayesian_value_corr
            # print("pairwise_corr_bayes_factor", pairwise_corr_bayes_factor)
            

            return {
                "status" : 200,
                "message": "Success",
                "continuous_col": continuous_col, 
                "categorical_col": categorical_col,
                "result": {
                    "anova": {
                        "F": anova_result['F'].values[0],
                        "pval": anova_result['p-unc'].values[0],
                        "dof": anova_result['ddof1'].values[0]  # Use 'ddof1' instead of 'DF'
                    },
                    "pairwise_comparisons": bf_dict,
                    "bayes_pairwise_corr_comparisons":{
                        "pairwise_corr_bayes_factor": bf_dict,
                        "graph_points": {
                                "means": {
                                    category: (None if pd.isna(value) else value)
                                    for category, value in means.to_dict().items()
                                },
                                "std_deviation": {
                                    category: (None if pd.isna(value) else value)
                                    for category, value in sds.to_dict().items()
                                }
                            }
                        }        
                },
                "type": "continuous_categorical"
            }
        
    except HTTPException as e:
        raise e  
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


async def case_2_multiple_categorical(df, file_name):
    try:
        columns = df.columns.tolist()
        col_unique_vals = sorted([[col, df[col].dropna().nunique()] for col in columns], key=lambda x: x[1])
        class_col = col_unique_vals[0][0]
        data_cols = [i[0] for i in col_unique_vals[1:]]

        cx_tab = pd.crosstab([df[data_col] for data_col in data_cols], df[class_col])
        chi2, pval, dof, ex = chi2_contingency(cx_tab)
        class_vals = sorted(df[class_col].dropna().unique())
       
        observed_table = json.loads(cx_tab.to_json())
        expected_table = json.loads(pd.DataFrame(ex, index=cx_tab.index, columns=cx_tab.columns).to_json())

        bf_dict = {}
        for cls_val in class_vals:
            for idx in cx_tab.index: 
                bf10 = bayesfactor_binom(int(cx_tab[cls_val][idx]), int(cx_tab[cls_val].sum()))
                bf10 = sanitize_bf_value(bf10)
                if type(idx) == str: key = cls_val + ' vs ' + idx
                else: key = cls_val + ' vs ' + ', '.join(idx)
                bf01_value = None
                if bf10 not in (None, 0):
                    bf01_value = round(1 / bf10, 2)
                bayes_interp = v1_get_bayes_interp(2, bf01_value).capitalize() if bf01_value else "BF01 not available"
                bf_dict[key] =  {"BF10": bf01_value, "bayes_interp": bayes_interp}

        return  {
                "status" : 200,
                "message": "Success",
                "data": {
                    "type": "multiple_categorical",
                    "file_name": file_name,
                    "output": {
                        "crosstabulation": {
                            "observed": observed_table,
                            "expected": expected_table
                        },
                        "chi_square_test": {
                            "chi2": chi2,
                            "pval": pval,
                            "dof": dof
                        },
                        "bayes_factors": bf_dict
                    },
                    "bayes_message": await get_bayes_message(),
                }        
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

async def case_3_two_continuous(df):
    
    try:
        # Drop rows with missing values in the two columns
        df_clean = df.dropna(subset=[df.columns[0], df.columns[1]])

        # Check if there are enough valid rows
        if len(df_clean) < 2:
            return {
                "status": "error",
                "message": "At least 2 valid rows are required to compute correlation."
            }

        # Extract the two columns
        col1 = df_clean[df.columns[0]]
        col2 = df_clean[df.columns[1]]

        # Compute Pearson correlation and p-value
        corr, pval = pearsonr(col1, col2)

        # Compute 95% confidence interval for the correlation coefficient
        n = len(col1)  # Sample size
        r_z = np.arctanh(corr)  # Fisher transformation
        se = 1 / np.sqrt(n - 3)  # Standard error
        z = 1.96  # Z-score for 95% confidence
        lo_z, hi_z = r_z - z * se, r_z + z * se
        ci_low, ci_high = np.tanh((lo_z, hi_z))  # Transform back to r

        # Return the results
        return {
            "status": "success",
            "result": {
                "correlation": corr,
                "pval": pval,
                "CI95%": [ci_low, ci_high],  # 95% confidence interval
                "sample_size": n  # Number of valid rows used
            },
            "type": "two_continuous",
            "data": {
                "col1": col1,
                "col2": col2
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

async def case_4_multiple_categorical_continuous(df):
    try:
        # Identify the continuous and categorical columns
        continuous_cols = [col for col in df.columns if df[col].dtype.name in ['float64', 'int64', 'Int64']]
        categorical_cols = [col for col in df.columns if df[col].dtype.name == 'object']

        if len(continuous_cols) == 0 or len(categorical_cols) == 0:
            raise ValueError("Dataframe must have at least one categorical and one continuous column.")

        continuous_col = continuous_cols[0]

        # Create group names based on categorical column combinations
        df['group'] = df[categorical_cols].agg('_'.join, axis=1)
        df['group'] = df['group'] + '_' + continuous_col

        # Create dictionary with grouped values
        grouped_data = df.groupby('group')[continuous_col].apply(list).to_dict()

        # Find the maximum length among all lists
        max_len = max(len(lst) for lst in grouped_data.values())

        # Pad shorter lists with NaN
        for key in grouped_data:
            grouped_data[key] += [np.nan] * (max_len - len(grouped_data[key]))

        # Convert to DataFrame
        df_grouped = pd.DataFrame(grouped_data).dropna()
        
        pairwise_results = pg.pairwise_corr(df_grouped, method="pearson", alternative="two-sided") # only support greater then 4 values in the pair other wish it will ignore that column pair check

        bayes_factor = {}
        for _, row in pairwise_results.iterrows():
            col1, col2 = row['X'], row['Y']
            if 'BF10' in row and not pd.isna(row['BF10']):
                bf10 = float(row['BF10'])
                if np.isinf(bf10):  # Handle Infinity values
                    bf10 = 1e308  # Replace with a large number
            else:
                bf10 = None
            bayes_interp = v1_get_bayes_interp(4, bf10).capitalize() if bf10 else "BF10 not available"   
            bayesian_value_corr = {"BF10": bf10, "bayes_interp": bayes_interp}
            bayes_factor[f"{col1} | {col2}"] = bayesian_value_corr

        means = df_grouped.mean()
        sds = df_grouped.std()


        # Save the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(means.index, means.values, yerr=sds.values, capsize=5)
        plt.xlabel("Groups")
        plt.ylabel(f"Mean {continuous_col} with SD")
        plt.title(f"Mean {continuous_col} by Group with Standard Deviations")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        # plt.savefig("bar_chart_case4.png")  # Save instead of showing
        plt.close()

        # Perform Kruskal-Wallis test
        df_clean = df.dropna(subset=[continuous_col] + categorical_cols)
        df_rearranged= df_clean[["group"] + [continuous_col]]
        kruskal = pingouin_kruskal(data=df_rearranged, dv=continuous_col, between='group')

        return {
                "status" : 200,
                "message": "Success",
                "data": {
                    "type": "multiple_categorical_continuous",
                    "bayes_factor": bayes_factor,
                    "kruskal": kruskal.to_dict(orient="records"),
                    "graph_points": {
                            "means": {
                                category: (None if pd.isna(value) else value)
                                for category, value in means.to_dict().items()
                            },
                            "std_deviation": {
                                category: (None if pd.isna(value) else value)
                                for category, value in sds.to_dict().items()
                            }
                        }        
                }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

async def case_5_discontinuous_crosstab(df):
    cx_tab = pd.crosstab(df[df.columns[0]], df[df.columns[1]])
    chi2, pval, dof, ex = chi2_contingency(cx_tab)
    return {
        "status": "success",
        "result": {
            "chi2": chi2,
            "pval": pval,
            "dof": dof,
            "expected": ex.tolist()
        },
        "type": "discontinuous_crosstab"
    }














#---------------------------------- CASE-4 BACKUP -------------------------------------------

# async def case_4_multiple_categorical_continuous(df):
#     try:
#         # Identify the continuous and categorical columns
#         continuous_cols = [col for col in df.columns if df[col].dtype.name in ['float64', 'int64', 'Int64']]
#         categorical_cols = [col for col in df.columns if df[col].dtype.name == 'object']

#         if len(continuous_cols) == 0 or len(categorical_cols) == 0:
#             raise ValueError("Dataframe must have at least one categorical and one continuous column.")

#         continuous_col = continuous_cols[0]

#         # Create group names based on categorical column combinations
#         df['group'] = df[categorical_cols].agg('_'.join, axis=1)
#         df['group'] = df['group'] + '_' + continuous_col

#         # Create dictionary with grouped values
#         grouped_data = df.groupby('group')[continuous_col].apply(list).to_dict()

#         # Find the maximum length among all lists
#         max_len = max(len(lst) for lst in grouped_data.values())

#         # Pad shorter lists with NaN
#         for key in grouped_data:
#             grouped_data[key] += [np.nan] * (max_len - len(grouped_data[key]))

#         # Convert to DataFrame
#         df_grouped = pd.DataFrame(grouped_data).dropna()

#         # Compute means and standard deviations
#         means = df_grouped.mean()
#         sds = df_grouped.std()

#         # Save the bar chart
#         plt.figure(figsize=(10, 6))
#         plt.bar(means.index, means.values, yerr=sds.values, capsize=5)
#         plt.xlabel("Groups")
#         plt.ylabel(f"Mean {continuous_col} with SD")
#         plt.title(f"Mean {continuous_col} by Group with Standard Deviations")
#         plt.xticks(rotation=45, ha="right")
#         plt.tight_layout()
#         plt.savefig("bar_chart_case4.png")  # Save instead of showing
#         plt.close()

#         # Prepare graph values for frontend
#         graph_values = {
#             "x_labels": means.index.tolist(),
#             "means": means.values.tolist(),
#             "std_devs": sds.values.tolist(),
#             "grouped_data": grouped_data
#         }# Select the fir

#         # Rearranging data for statistical analysis
#         df_rearranged = pd.DataFrame({
#         continuous_col: [val for lst in grouped_data.values() for val in lst if not pd.isna(val)],
#         "groups": [key for key, lst in grouped_data.items() for val in lst if not pd.isna(val)]
#         })

#         continuous_values_list = [val for lst in grouped_data.values() for val in lst if not pd.isna(val)]

#         pingouin_kruskal_data = pingouin_kruskal(data=df_rearranged, dv=continuous_col, between='groups').to_dict(orient="records")
        
#         unc = pingouin_kruskal_data[0].get("p-unc", 0) if pingouin_kruskal_data else 0

#         bayes_factor = await bayes_factor_by_unc(unc)

#         # Perform Kruskal-Wallis test
#         unique_groups = df_rearranged["groups"].unique()
#         group_values = [df_rearranged[df_rearranged["groups"] == grp][continuous_col] for grp in unique_groups]

#         if len(group_values) > 1:  # Ensure we have at least 2 groups
#             h_stat, pval = kruskal(*group_values)
#             kruskal_result = {"h_stat": h_stat, "pval": pval}
#         else:
#             kruskal_result = {"error": "Not enough groups for Kruskal-Wallis test"}

#         # Store Kruskal test results
#         kruskal_data = {
#             "graph_values": graph_values,
#             "kruskal_result": kruskal_result,
#             "pingouin_kruskal": pingouin_kruskal_data
#         }

#         # Bayesian analysis for all pairs
#         df_clean = df.dropna(subset=[continuous_col] + categorical_cols)
#         results = {}

#         for col in categorical_cols:
#             unique_vals = df_clean[col].dropna().unique()
#             if len(unique_vals) < 2:
#                 continue  # Skip if there aren't at least two unique values

#             groups = [df_clean[df_clean[col] == val][continuous_col] for val in unique_vals]
#             if len(groups) < 2:
#                 continue  # Skip if not enough groups

#             h_stat, pval = kruskal(*groups)  # Kruskal-Wallis ANOVA

#             # Bayesian analysis
#             bf_dict = {}
#             for i in range(len(unique_vals)):
#                 for j in range(i + 1, len(unique_vals)):
#                     group1 = df_clean[df_clean[col] == unique_vals[i]][continuous_col]
#                     group2 = df_clean[df_clean[col] == unique_vals[j]][continuous_col]

#                     if len(group1) >= 2 and len(group2) >= 2:
#                         bf_result = pg.ttest(group1, group2, paired=False, alternative='two-sided')
#                         bf10 = bf_result['BF10'].values[0]
#                         key = f'{unique_vals[i]} vs {unique_vals[j]}'
#                         bf_dict[key] = bf10

#             group_sizes = {val: len(df_clean[df_clean[col] == val]) for val in unique_vals}
#             # results[col] = {
#             #     "kruskal_wallis": {
#             #         "h_stat": h_stat,
#             #         "pval": pval,
#             #         "group_sizes": group_sizes
#             #     },
#             #     "bayesian_analysis": bf_dict,
#             #     "kruskal_wallis_test_data": kruskal_data
#             # }
#         results.update({"bayes_factor": bayes_factor})
#         return {
#             "status": "success",
#             "result": results,
#             "continuous_values_list": continuous_values_list,
#             "type": "multiple_categorical_continuous"
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": str(e)
#         }


