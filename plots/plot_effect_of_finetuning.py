import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import pickle
from plotting import Plotter
import pymc as pm
import arviz as az
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pdf2image import convert_from_path
import datetime
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


# os.chdir("..")


##############################################
####### FUNCTIONS FOR LOADING DATA  ##########
##############################################

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
        all_evolutions[measure] = {}
        for m in plotter.models:
            all_evolutions[measure][m] = {}
            for p in plotter.prompts:
                all_evolutions[measure][m][p] = []
                for s in plotter.stories:
                    # if measure == "length":
                    #     all_evolutions[measure][m][p].append(np.array(all_data["evolution"][p][s][f"Results/{m}/{p}/{s}"][key], dtype=int))
                    # else:
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
                    print(key)
                    print(len(all_data["evolution"][p][intial_story][f"Results/{m}/{p}/{intial_story}"][key]))
                    print(len(all_data["evolution"][p][intial_story][f"Results/{m}/{p}/{intial_story}"][key][0]))
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

def get_all_similarities(all_data, plotter):
    all_similarities = {}
    for m_i, m in zip(plotter.model_indices, plotter.models):
        all_similarities[m] = {}
        for p in plotter.prompts:
            all_similarities[m][p] = {}
            for gen in range(plotter.n_generations):
                all_similarities[m][p][gen] = []
                for ss_i in range(plotter.n_stories*plotter.n_seeds*(plotter.n_stories*plotter.n_seeds-1)):
                    all_similarities[m][p][gen].append(np.squeeze(all_data["plot_convergence"][p]["all_similaritys"][m_i][ss_i][gen]))
            



############################
####### LOAD DATA ##########
############################


print("Loading data ...")



saving_name = "processed_data"




with open("../Results/data-for-plotting-finetuning/all_data.pkl", "rb") as file:
    all_data = pickle.load(file)


plotter = Plotter(all_data, models = ['Mixtral-8x7B-Instruct-v0.1', 'Mixtral-8x7B-Base-v0.1', 'Meta-Llama-3-70B-Base', 'Meta-Llama-3-70B-Instruct'], prompts = ['rephrase', 'inspiration', 'continue'])


all_attr_positions, all_attr_strengths, all_attr_positions_10, all_attr_strengths_10 = get_all_attr_positions_strengths(all_data, plotter)




os.makedirs(f"Results/{saving_name}", exist_ok=True)    



with open(f"Results/{saving_name}/all_attr_positions.pkl", "wb") as file:
    pickle.dump(all_attr_positions, file)
with open(f"Results/{saving_name}/all_attr_strengths.pkl", "wb") as file:
    pickle.dump(all_attr_strengths, file)
with open(f"Results/{saving_name}/all_attr_positions_10.pkl", "wb") as file:
    pickle.dump(all_attr_positions_10, file)
with open(f"Results/{saving_name}/all_attr_strengths_10.pkl", "wb") as file:
    pickle.dump(all_attr_strengths_10, file)

all_initial_cumuls, all_final_cumuls, all_after_10_cumuls = get_all_initial_vs_final(all_data, plotter)
with open(f"Results/{saving_name}/all_initial_cumuls.pkl", "wb") as file:
    pickle.dump(all_initial_cumuls, file)
with open(f"Results/{saving_name}/all_final_cumuls.pkl", "wb") as file:
    pickle.dump(all_final_cumuls, file)
with open(f"Results/{saving_name}/all_after_10_cumuls.pkl", "wb") as file:
    pickle.dump(all_after_10_cumuls, file)


plotter.load_results(saving_name)



date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

saving_name = "figures_" + date


os.makedirs(f"Figures_finetuning/{saving_name}", exist_ok=True)

print("Data loaded successfully")
print("Figures will be saved in Figures/" + saving_name)




fig, axs = plt.subplots(2, len(plotter.measures), figsize=(6*(len(plotter.measures)), 12))



_, _ = plotter.compare_base_instruct("position", ylim=None, fig=fig, axs=axs[0,:], legend_i=3, with_suptitle=True, legend_pos=(0.8, 0.8), labelpad = 5)
_, _ = plotter.compare_base_instruct("strength", ylim=None, fig=fig, axs=axs[1,:], legend_i=None, with_suptitle=False, labelpad = 40)
plt.savefig(f"Figures_finetuning/{saving_name}/effect_of_finetuning.pdf", bbox_inches="tight")


