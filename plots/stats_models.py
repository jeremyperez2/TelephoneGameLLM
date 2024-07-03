import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import pickle
import numpy as array
from plotting import Plotter
import pymc as pm
import arviz as az
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


#####################################
### FUNCTIONS FOR DATA PROCESSING ###
#####################################

def get_cumulativeness(all_data, plotter):
    cumulativeness = {}
    for measure in plotter.measures:
        cumulativeness[measure] = {}
        for m in plotter.models:
            cumulativeness[measure][m] = {}
            for p in plotter.prompts:
                cumulativeness[measure][m][p] = all_data["plot_cumulativeness"][f"cumulativeness_{measure}"][m][p]
    return cumulativeness


def get_all_evolutions(all_data, plotter):
    all_evolutions = {}
    for measure, key in zip(plotter.measures, ["all_seeds_toxicity", "all_seeds_positivity", "all_seeds_difficulty", "all_seeds_length"]):
    # for measure, key in zip(plotter.measures, ["all_seeds_toxicity", "all_seeds_positivity", "all_seeds_difficulty", "length"]):
        all_evolutions[measure] = {}
        for m in plotter.models:
            all_evolutions[measure][m] = {}
            for p in plotter.prompts:
                all_evolutions[measure][m][p] = []
                for s in plotter.stories:
                    if measure == "length":
                        all_evolutions[measure][m][p].append(np.array(all_data["evolution"][p][s][f"Results/{m}/{p}/{s}"][key], dtype=int))
                    else:
                        all_evolutions[measure][m][p].append(all_data["evolution"][p][s][f"Results/{m}/{p}/{s}"][key])
    return all_evolutions

def get_change_per_generation(all_data, plotter):
    change_per_generation = {}
    for measure, key in zip(plotter.measures, ["all_seeds_toxicity", "all_seeds_positivity", "all_seeds_difficulty", "all_seeds_length"]):
        change_per_generation[measure] = {}
        for m in plotter.models:
            change_per_generation[measure][m] = {}
            for p in plotter.prompts:
                change = []
                for s in plotter.stories:
                    all_evolutions = np.array(all_data["evolution"][p][s][f"Results/{m}/{p}/{s}"][key])


                    total_change = np.sum(np.abs(all_evolutions[:,1:] - all_evolutions[:,:-1]), axis=1)
                        
                    change.append([np.abs(all_evolutions[:,i] - all_evolutions[:,i-1]) / total_change for i in range(1, all_evolutions.shape[1])])

                   
                change_per_generation[measure][m][p] = change 
                
    return change_per_generation

def get_directional_change_per_generation(all_data, plotter):
    directional_change_per_generation = {}
    for measure, key in zip(plotter.measures, ["all_seeds_toxicity", "all_seeds_positivity", "all_seeds_difficulty", "all_seeds_length"]):
        directional_change_per_generation[measure] = {}
        for m in plotter.models:
            directional_change_per_generation[measure][m] = {}
            for p in plotter.prompts:
                change = []
                for s in plotter.stories:
                    all_evolutions = np.array(all_data["evolution"][p][s][f"Results/{m}/{p}/{s}"][key])

                    total_change = np.abs(all_evolutions[:,0] - all_evolutions[:,-1])
                        
                    change.append([(np.abs(all_evolutions[:,i-1] - all_evolutions[:,-1])  - np.abs(all_evolutions[:,i] - all_evolutions[:,-1])) / total_change for i in range(1, all_evolutions.shape[1])])
                directional_change_per_generation[measure][m][p] = change 
                
    return directional_change_per_generation



def get_all_initial_vs_final(all_data, plotter):
    all_initial_cumuls = {}
    all_final_cumuls = {}
    all_after_10_cumuls = {}
    for measure, key in zip(plotter.measures, ["all_seeds_toxicity", "all_seeds_positivity", "all_seeds_difficulty", "all_seeds_length"]):
        all_initial_cumuls[measure] = {}
        all_final_cumuls[measure] = {}
        all_after_10_cumuls[measure] = {}
        for m_i, m in zip(plotter.model_indices, plotter.models):
            all_initial_cumuls[measure][m] = {}
            all_final_cumuls[measure][m] = {}
            all_after_10_cumuls[measure][m] = {}
            for p in plotter.prompts:
                all_initial_cumuls[measure][m][p] = []
                all_final_cumuls[measure][m][p] = []
                all_after_10_cumuls[measure][m][p] = []

                for intial_story in plotter.stories:
                    all_initial_cumuls[measure][m][p].append(np.array(all_data["evolution"][p][intial_story][f"Results/{m}/{p}/{intial_story}"][key])[:,0])
                    all_final_cumuls[measure][m][p].append(np.array(all_data["evolution"][p][intial_story][f"Results/{m}/{p}/{intial_story}"][key])[:,-1])
                    all_after_10_cumuls[measure][m][p].append(np.array(all_data["evolution"][p][intial_story][f"Results/{m}/{p}/{intial_story}"][key])[:,10])
                    
    return all_initial_cumuls, all_final_cumuls, all_after_10_cumuls


def get_all_attr_positions_strengths(all_data, plotter):
    all_attr_positions = {}
    all_attr_strengths = {}
    all_attr_positions_10 = {}
    all_attr_strengths_10 = {}
    for measure in plotter.measures:
        all_attr_positions[measure] = {}
        all_attr_strengths[measure] = {}
        all_attr_positions_10[measure] = {}
        all_attr_strengths_10[measure] = {}
        for m in plotter.models:
            all_attr_positions[measure][m] = {}
            all_attr_strengths[measure][m] = {}
            all_attr_positions_10[measure][m] = {}
            all_attr_strengths_10[measure][m] = {}
            for p in plotter.prompts:
                all_attr_positions[measure][m][p] = all_data["plot_attractors"][f"attractor_{measure}_position"][m][p]
                all_attr_strengths[measure][m][p] = all_data["plot_attractors"][f"attractor_{measure}_strength"][m][p]
                all_attr_positions_10[measure][m][p] = all_data["plot_attractors"][f"attractor_{measure}_position_10"][m][p]
                all_attr_strengths_10[measure][m][p] = all_data["plot_attractors"][f"attractor_{measure}_strength_10"][m][p]


    return all_attr_positions, all_attr_strengths, all_attr_positions_10, all_attr_strengths_10



def get_data_attractors(plotter):
    all_attr_positions, all_attr_strengths = plotter.all_attr_positions, plotter.all_attr_strengths


    data = []

    for measure in plotter.measures:
        for m in plotter.models:
            for p in plotter.prompts:
                
                data.append({"model": m, "prompt": p, "measure": measure, "position": all_attr_positions[measure][m][p], "strength": all_attr_strengths[measure][m][p]})
    df = pd.DataFrame(data)
    return data

def get_data_initial_vs_final(plotter):
    data = []
    maxim = {}
    minin = {}
    for measure, key in zip(plotter.measures, ["all_seeds_toxicity", "all_seeds_positivity", "all_seeds_difficulty", "all_seeds_length"]):
        max_i = 0
        min_i = 0
        for m in plotter.models:
            for p in plotter.prompts:
                for s in plotter.stories:
                    for seed in range(plotter.n_seeds):

                        all_evolutions = np.array(all_data["evolution"][p][s][f"Results/{m}/{p}/{s}"][key])
                        try: 
                            difference = np.abs((all_evolutions[seed,1][0] - all_evolutions[seed,-1][0]))  ## For some reason the length measure is sometimes a list of lists ? 
                        except: 
                            difference = np.abs(all_evolutions[seed,1] - all_evolutions[seed,-1]) 
                        if  np.max(all_evolutions[seed,:]) > max_i:
                            max_i = np.max(all_evolutions[seed,:])
                        if np.min(all_evolutions[seed,:]) < min_i:
                            min_i = np.min(all_evolutions[seed,:])
                        data.append({"model": m, "prompt": p, "measure": measure, "story": s, "seed": seed, "difference": difference, "initial": all_evolutions[seed,1], "final": all_evolutions[seed,-1]}) 
        maxim[measure] = max_i
        minin[measure] = min_i
    for entry in data:
        if maxim[entry["measure"]] - minin[entry["measure"]] == 0:
            entry["normalized_difference"] = 0
        entry["normalized_difference"] = (entry["difference"]) / (maxim[entry["measure"]] - minin[entry["measure"]])

    df = pd.DataFrame(data)
    return data
    


def get_initial_vs_final_sim(all_data, plotter):
    initial_sim = {}
    final_sim = {}
    for m_i, m in zip(plotter.model_indices, plotter.models):
        initial_sim[m] = {}
        final_sim[m] = {}
        for p in plotter.prompts:
            initial_sim[m][p] = []
            final_sim[m][p] = []
            for ss_i in range(plotter.n_stories*plotter.n_seeds*(plotter.n_stories*plotter.n_seeds-1)):
                initial_sim[m][p].append(np.squeeze(all_data["plot_convergence"][p]["all_similaritys"][m_i][ss_i][0]))
                final_sim[m][p].append(np.squeeze(all_data["plot_convergence"][p]["all_similaritys"][m_i][ss_i][-1]))
            initial_sim[m][p] = np.array(initial_sim[m][p])
            final_sim[m][p] = np.array(final_sim[m][p])
    return initial_sim, final_sim



#############################################
### FUNCTIONS FOR FITTING BAYESIAN MODELS ###
#############################################
import itertools


def run_bayesian_model(df, variables = ['model', 'prompt', 'story' ], measures = ['punct', 'area'], control_stories = False):


    # Convert categorical variables to numeric codes
    for v in variables:
        df[f'{v}_code'] = pd.Categorical(df[v]).codes
    for m in measures:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    
    def fit_model(response, df, control_stories = False, intercept = False):
        with pm.Model() as model:
            if intercept:
                intercept = pm.Normal('intercept', mu=0, sigma=1)

                mu = intercept
            else:
                mu = 0

            for v in variables:
                if v == 'story' and control_stories:
                    coef = pm.Normal(f'{v}_coef', mu=0, sigma=1, shape=len(df[f'{v}_code'].unique()))
                    mu += coef[df[f'{v}_code']]
                else:
                    # Ensure the variable code is present in the dataframe
                    if f'{v}_code' in df.columns:
                        # Handle other categorical variables
                        unique_codes = df[f'{v}_code'].unique()
                        coef = pm.Normal(f'{v}_coef', mu=0, sigma=1, shape=len(unique_codes))
                        code_indices = df[f'{v}_code'].map({code: idx for idx, code in enumerate(unique_codes)}).values
                        mu += coef[code_indices]
                    else:
                        raise KeyError(f"Column '{v}_code' not found in dataframe.")

           
            sigma = pm.HalfNormal('sigma', sigma=1)
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=df[response])
            
            trace = pm.sample(4000, tune=1000, return_inferencedata=True,
                                                init='jitter+adapt_diag',  
                                                target_accept=0.95,  
                                                cores=1, 
                                                progressbar=True,
                                                chains=10)

            
            return model, trace

    # Plot the posterior distributions with HDI intervals
    def plot_posterior(trace, title):
        az.plot_forest(trace, hdi_prob=0.95, combined=True, figsize=(8, 4))
        plt.title(title)
        plt.show()
    
    # Additional: Plot specific parameter estimates
    def plot_param_estimates(summary, title, intercept = False):
        if intercept:
            xticks = ['Intercept']
        else:
            xticks = []

        for v in variables:
            xticks += list(df[v].unique())

        xticks += ['sigma']
        
        summary_df = summary.reset_index()


        summary_df = summary.reset_index()
        plt.errorbar(summary_df['index'], summary_df['mean'], 
                    yerr=[summary_df['mean'] - summary_df['hdi_2.5%'], summary_df['hdi_97.5%'] - summary_df['mean']],
                    fmt='o', capsize=5)
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(title)
        plt.xlabel('Parameter')
        plt.ylabel('Estimated Value')
        plt.xticks(range(len(summary_df['index'])), xticks, rotation=90)
        plt.show()

    def get_credibility_intervals(summary):
        credibility_intervals = summary[['mean', 'hdi_2.5%', 'hdi_97.5%']]
        return credibility_intervals
    
    def get_combination_effects(summary, variables):
        # Extract the coefficients
        coef_dict = {}
        for v in variables:
            coef_dict[v] = summary.loc[summary.index.str.startswith(v)]['mean'].values
        
        # Generate combinations of variable levels
        combinations = list(itertools.product(*coef_dict.values()))
        combined_effects = [sum(comb) for comb in combinations]

        # Calculate the mean and HDI for the combined effects
        mean_effect = sum(combined_effects) / len(combined_effects)
        hdi_2_5 = sorted(combined_effects)[int(0.025 * len(combined_effects))]
        hdi_97_5 = sorted(combined_effects)[int(0.975 * len(combined_effects))]
        
        return pd.DataFrame({
            'mean': [mean_effect],
            'hdi_2.5%': [hdi_2_5],
            'hdi_97.5%': [hdi_97_5],
            'significant': [not (hdi_2_5 <= 0 <= hdi_97_5)]
        })
    
    




    # Fit model for 'area'
    models, traces, summaries = [], [], []
    for m in measures:
        model, trace = fit_model(m, df)
        models.append(model)
        traces.append(trace)
        summary = az.summary(trace, hdi_prob=0.95)
        plot_posterior(trace, f'{m} model')
        plot_param_estimates(summary,f'{m} parameters estimate' )
        summaries.append(summary)

    combination_effects_dict = {m: get_combination_effects(summary, variables) for m, summary in zip(measures, summaries)}



    # # Output credibility intervals as tables
    # for m, ci in combination_effects_dict.items():
    #     print(f"\n95% Credibility Intervals for {m}:\n")
    #     print(ci)

    return models, traces#, combination_effects_dict


###################################################
### FUNCTION FOR PLOTTING PARAMETER COMPARISON ###
###################################################


def plot_coeff_matrix(df, trace, name = 'Area', param = 'model'):

    param_names = df[param].unique()

    # Extract posterior samples for the model coefficients from the aggregated trace
    model_coef_samples = trace.posterior[f'{param}_coef'].stack(samples=("chain", "draw"))
    summary_area = az.summary(trace, hdi_prob=0.95)

    # Compare model 3 (index 2) with model 1 (index 0)
    diff_3_1 = model_coef_samples[:, 0] - model_coef_samples[:, 3]

    # Calculate differences between model estimates
    param_count = len(param_names)
    differences = np.zeros((param_count, param_count))
    credible_interval = np.zeros((param_count, param_count, 2))
    significant = np.zeros((param_count, param_count))
    diff_hdi_lower = np.zeros((param_count, param_count))
    diff_hdi_upper = np.zeros((param_count, param_count))
    for i in range(param_count):
        for j in range(param_count):
            if param == 'model':
                param1_samples = trace.posterior[f'{param}_coef'].sel(model_coef_dim_0=i).stack(samples=("chain", "draw")).values
                param2_samples = trace.posterior[f'{param}_coef'].sel(model_coef_dim_0=j).stack(samples=("chain", "draw")).values
            elif param == 'prompt':
                param1_samples = trace.posterior[f'{param}_coef'].sel(prompt_coef_dim_0=i).stack(samples=("chain", "draw")).values
                param2_samples = trace.posterior[f'{param}_coef'].sel(prompt_coef_dim_0=j).stack(samples=("chain", "draw")).values
            elif param == 'measure':
                param1_samples = trace.posterior[f'{param}_coef'].sel(measure_coef_dim_0=i).stack(samples=("chain", "draw")).values
                param2_samples = trace.posterior[f'{param}_coef'].sel(measure_coef_dim_0=j).stack(samples=("chain", "draw")).values

            # Compute the posterior differences
            diff_samples = param1_samples - param2_samples

            # Calculate the mean and 95% credible interval
            mean_diff = np.mean(diff_samples)
            cred_interval = np.percentile(diff_samples, [2.5, 97.5])

            differences[i, j] = mean_diff

            if cred_interval[0] <= 0 and cred_interval[1] >= 0: ## if the 95% credible interval contains 0
                significant[i, j] = False
            else:
                significant[i, j] = True


    # Plot the differences matrix
    fig, ax =  plt.subplots(figsize=(10, 10))
    plt.imshow(differences, cmap='coolwarm')

    # Add annotations
    for i in range(param_count):
        for j in range(param_count):
            if 'toxicity' in name:
                 annotation = f'{differences[i, j]:.4f}'
            else:
                annotation = f'{differences[i, j]:.2f}'
            if significant[i,j]:
                annotation += ' *'
            if 'toxicity' in name:
                plt.text(j, i, annotation, ha='center', va='center', color='black', fontsize=8)
            else:
                plt.text(j, i, annotation, ha='center', va='center', color='black')



    plt.xticks(range(param_count), param_names, rotation=90)
    plt.yticks(range(param_count), param_names, rotation=0)
    plt.xlabel(param.capitalize())
    plt.ylabel(param.capitalize())
    plt.title(f'{param.capitalize()} Coefficient Differences - {name.capitalize()}')
    plt.tight_layout()
    plt.colorbar()
    return fig, ax
    

import datetime
date = datetime.datetime.now()
saving_name = "StatModels" + str(date).replace(" ", "_").replace(":", "_").replace(".", "_") 

save_data = True


##################
### LOAD DATA  ###
##################

os.chdir("..")


with open("Results/data-for-plotting/all_data.pkl", "rb") as file:
    all_data = pickle.load(file)







#####################
### PROCESS DATA  ###
#####################

store_name = "processed_data"

plotter = Plotter(all_data)

# cumulativeness = get_cumulativeness(all_data, plotter)
all_evolutions = get_all_evolutions(all_data, plotter)
all_initial_cumuls, all_final_cumuls, all_after_10_cumuls = get_all_initial_vs_final(all_data, plotter)
all_attr_positions, all_attr_strengths, all_attr_positions_10, all_attr_strengths_10 = get_all_attr_positions_strengths(all_data, plotter)
initial_sim, final_sim = get_initial_vs_final_sim(all_data, plotter)
change_per_generation = get_change_per_generation(all_data, plotter)
directional_change_per_generation = get_directional_change_per_generation(all_data, plotter)

os.makedirs(f"Results/{store_name}", exist_ok=True)    
if save_data:
    # with open(f"Results/{store_name}/cumulativeness.pkl", "wb") as file:
    #     pickle.dump(cumulativeness, file)
    with open(f"Results/{store_name}/all_evolutions.pkl", "wb") as file:
        pickle.dump(all_evolutions, file)

    with open(f"Results/{store_name}/all_initial_cumuls.pkl", "wb") as file:
        pickle.dump(all_initial_cumuls, file)
    with open(f"Results/{store_name}/all_final_cumuls.pkl", "wb") as file:
        pickle.dump(all_final_cumuls, file)
    with open(f"Results/{store_name}/all_after_10_cumuls.pkl", "wb") as file:
        pickle.dump(all_after_10_cumuls, file)

    with open(f"Results/{store_name}/all_attr_positions.pkl", "wb") as file:
        pickle.dump(all_attr_positions, file)
    with open(f"Results/{store_name}/all_attr_strengths.pkl", "wb") as file:
        pickle.dump(all_attr_strengths, file)
    with open(f"Results/{store_name}/all_attr_positions_10.pkl", "wb") as file:
        pickle.dump(all_attr_positions_10, file)
    with open(f"Results/{store_name}/all_attr_strengths_10.pkl", "wb") as file:
        pickle.dump(all_attr_strengths_10, file)

    with open(f"Results/{store_name}/initial_sim.pkl", "wb") as file:
        pickle.dump(initial_sim, file)
    with open(f"Results/{store_name}/final_sim.pkl", "wb") as file:
        pickle.dump(final_sim, file)
    with open(f"Results/{store_name}/all_after_10_cumuls.pkl", "wb") as file:
        pickle.dump(all_after_10_cumuls, file)
    with open(f"Results/{store_name}/change_per_generation.pkl", "wb") as file:
        pickle.dump(change_per_generation, file)
    
    with open(f"Results/{store_name}/directional_change_per_generation.pkl", "wb") as file:
        pickle.dump(directional_change_per_generation, file)
    


plotter.load_results(store_name)


os.makedirs(f"Results/{saving_name}", exist_ok=True)



#######################################################################################################
### FIT STATISTICAL MODEL PREDICTING ATTRACTOR STRENGTH AS A FUNCTION OF MODEL, PROMPT, AND MEASURE ###
#######################################################################################################

df = get_data_attractors(plotter)

df = pd.DataFrame(df)

variables = ['model', 'prompt', 'measure']

models, traces = run_bayesian_model(df, measures = ['strength'], variables = ['model', 'prompt', 'measure'])

for i,m in enumerate(['strength']):
    for v in variables:

        fig, ax = plot_coeff_matrix(df, traces[i], name = f'\nAttractor {m}', param = v)
        fig.savefig(f"Results/{saving_name}/attractor_{m}_{v}.png")
        

   



###############################################################################################################
### FIT STATISTICAL MODEL PREDICTING ATTRACTOR POSITION AS A FUNCTION OF MODEL AND PROMPT, FOR EACH MEASURE ###
###############################################################################################################

df = get_data_attractors(plotter)

df = pd.DataFrame(df)

variables = ['model', 'prompt']

for measure in plotter.measures:
    models, traces = run_bayesian_model(df[df['measure'] == measure], measures = ['position'], variables = ['model', 'prompt'])

    for i,m in enumerate(['position']):
        for v in variables:

            fig, ax = plot_coeff_matrix(df, traces[i], name = f'\nAttractor Position - {measure}', param = v)
            fig.savefig(f"Results/{saving_name}/attractor_{m}_{v}_{measure}.png")



# ###############################################################################################################
# ### FIT STATISTICAL MODEL PREDICTING CUMULATIVENESS AS A FUNCTION OF MODEL, PROMPT AND  MEASURE ###
# ###############################################################################################################

# df = get_data_initial_vs_final(plotter)

# df = pd.DataFrame(df)

# variables = ['model', 'prompt', 'measure']

# models, traces = run_bayesian_model(df, measures = ['normalized_difference'], variables = ['model', 'prompt', 'measure'])

# for i,m in enumerate(['normalized_difference']):
#     for v in variables:

#         fig, ax = plot_coeff_matrix(df, traces[i], name = f'Difference', param = v)
#         fig.savefig(f"Results/{saving_name}/cumulativeness_{v}.png")
        




####################################################################
### Kolmogorov-Smirnov (KS) Test for comparing the distributions ###
####################################################################

from scipy.stats import ks_2samp

def ks_test(df, variable, data1, data2):
    ks_stat, p_val = ks_2samp(data1[variable], data2[variable])
    return ks_stat, p_val

def compare_all_generations(prompt, measure_tag, successive = True):
    data = []
    p_values = {model: [] for model in plotter.models}
    for model in plotter.models:
        all_evolutions = np.array([plotter.all_data["evolution"][prompt][s][f"Results/{model}/{prompt}/{s}"][measure_tag] for s in plotter.stories])

        for g in range(plotter.n_generations):
            data.append(pd.DataFrame({
                "Value": all_evolutions[:,:, g].flatten(),
                "Generation": g,
                "Type": "Final",
                "Model": model,
                "Color": "initial"
            }))
    df = pd.concat(data)

    for m in plotter.models:
       for gen in range(1, plotter.n_generations):
            if successive:
                data1 = df[(df["Generation"] == gen - 1) & (df["Model"] == m)]
            else:
                data1 = df[(df["Generation"] == 1) & (df["Model"] == m)]
            data2 = df[(df["Generation"] == gen) & (df["Model"] == m)]
            ks_stat, p_val = ks_test(df, "Value", data1, data2)
            p_values[m].append(p_val)
    return p_values




fig, axes = plt.subplots(len(plotter.measures), len(plotter.prompts), figsize=(20, 20))


for i, (measure, measure_key) in enumerate(zip(plotter.measures, ["all_seeds_toxicity", "all_seeds_positivity", "all_seeds_difficulty", "all_seeds_length"])):
    for j, prompt in enumerate(plotter.prompts):

        p_values = compare_all_generations(prompt, measure_key, successive = False)

        for model in plotter.models:
            axes[i, j].plot(range(1, plotter.n_generations), p_values[model], label=model, color = plotter.model_colors[model])
        
        ## Add a area for the 0.05 significance level
        axes[i, j].fill_between(range(1, plotter.n_generations), 0.05, 0, color='grey', alpha=0.2)

        
        if i == 0:
            if prompt == "inspiration":
                axes[i, j].set_title("Take Inspiration")
            else:
                axes[i, j].set_title(prompt.capitalize())
        if j == 0:
            axes[i, j].set_ylabel(f'{measure.capitalize()}\n KS p-value')
        if i == len(plotter.measures) - 1:
            axes[i, j].set_xlabel("Generation")

plt.tight_layout()
plt.suptitle("Distribution similarities with first generation", fontweight='bold')
fig.subplots_adjust(top=0.9)
plt.legend()

plt.savefig(f"Results/{saving_name}/KS_test_first_generation.png")
plt.show()