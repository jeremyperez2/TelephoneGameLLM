
from llm_culture.analysis.utils import get_difficulties, get_lengths, get_positivities, get_stories, get_plotting_infos, get_toxicities, preprocess_stories
from llm_culture.analysis.utils import get_initial_story
from llm_culture.analysis.plots import *
from tqdm import trange
import pickle
import os


def main_analysis(folders, plot = False, scale_y_axis = False, labels = None, sizes = [], saving_folder = None, all_data = None):
    if saving_folder is None:
        saving_folder = '-'.join(os.path.basename(label) for label in labels)


    if len(saving_folder) > 250:
        saving_folder = saving_folder[:200] + '_NAME_TOO_LONG'

    if all_data is not None:
        data = all_data
    else:
        
        try:
            with open(f"Results/Comparisons/{saving_folder}/data.pkl", 'rb') as f:
                data = pickle.load(f)
        except:
            data = {}
                
            for i in trange(len(folders)):
                folder = folders[i]
                print(f"Running analysis for {folder} ...")
                # Compute all the metric that will be used for plotting
                all_seeds_stories = get_stories(folder)
                intial_story = get_initial_story(folder)

                n_gen, n_agents, x_ticks_space = get_plotting_infos(all_seeds_stories[0])
                all_seed_flat_stories = preprocess_stories(all_seeds_stories)
                all_seed_difficulties = get_difficulties(intial_story, all_seed_flat_stories)
                os.makedirs(f"Results/data-for-plotting/{folder}", exist_ok=True)
                with open(f"Results/data-for-plotting/{folder}/toxicity.pkl", 'wb') as f:
                    pickle.dump(all_seed_difficulties, f)
                all_seed_toxicities = get_toxicities(intial_story, all_seed_flat_stories, folder)
                with open(f"Results/data-for-plotting/{folder}/positivity.pkl", 'wb') as f:
                    pickle.dump(all_seed_toxicities, f)
                all_seed_positivities = get_positivities(intial_story, all_seed_flat_stories, folder)
                with open(f"Results/data-for-plotting/{folder}/difficulty.pkl", 'wb') as f:
                    pickle.dump(all_seed_positivities, f)

                

                all_seed_length = get_lengths(intial_story, all_seed_flat_stories)


                label = labels[i]
                data[folder] = {
                    'all_seed_stories': all_seeds_stories,
                    'initial_story': intial_story,
                    'n_gen': n_gen,
                    'n_agents': n_agents,
                    'x_ticks_space': x_ticks_space,
                    'all_seeds_flat_stories': all_seed_flat_stories,
                    'all_seeds_difficulty': all_seed_difficulties,
                    'all_seeds_toxicity': all_seed_toxicities,
                    'all_seeds_positivity': all_seed_positivities,
                    'all_seeds_length':  all_seed_length,
                    'label': label
                    }
        
            os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
            with open(f"Results/Comparisons/{saving_folder}/data.pkl", 'wb') as f:
                pickle.dump(data, f)


    return data
