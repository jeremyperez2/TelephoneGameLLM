import argparse
import json
import os

import scipy
from llm_culture.analysis.utils import get_difficulties, get_positivities, get_stories, get_toxicities, preprocess_stories
from llm_culture.analysis.utils import get_initial_story
from llm_culture.analysis.plots import *
import seaborn as sns
import pickle
from tqdm import trange



def get_pvalue(x, y, slope, std_err): ## get the p-value of the linear regression where the null hypothesis is that the absolute value of the slope is lower than 1

    # Degrees of freedom
    df = len(x) - 2

    # Calculate t-statistics
    t_stat_positive = (slope - 1) / std_err
    t_stat_negative = (slope + 1) / std_err

    # Calculate p-values for each t-statistic
    p_value_positive = 1 - scipy.stats.t.cdf(t_stat_positive, df)
    p_value_negative = scipy.stats.t.cdf(t_stat_negative, df)

    # Combine p-values
    combined_p_value = p_value_positive + p_value_negative

    return combined_p_value




def plot_intitial_vs_final(folders, saving_name, stories, models, data = None):

    if data is None:
        data = {'toxicity': [], 'positivity': [], 'difficulty': [], 'length': []}
    

        ##
        try:
            with open(f"Results/Comparisons/init_vs_last/{saving_name}/toxicity.pkl", 'rb') as f:
                all_toxicities = pickle.load(f)
            with open(f"Results/Comparisons/init_vs_last/{saving_name}/positivity.pkl", 'rb') as f:
                all_positivities = pickle.load(f)
            with open(f"Results/Comparisons/init_vs_last/{saving_name}/difficulty.pkl", 'rb') as f:
                all_difficulties = pickle.load(f)
            with open(f"Results/Comparisons/init_vs_last/{saving_name}/length.pkl", 'rb') as f:
                all_lengths = pickle.load(f)

        
        except:
            all_toxicities = []
            all_positivities = []
            all_difficulties = []
            all_lengths = []

            for i in trange(len(folders)):
                model_folder = folders[i]
                toxicities = []
                positivities = []
                difficulties = []
                lengths = []
                model = models[i]


                cumulativeness = []


                for story_i, init_story in enumerate(stories):
                    print(f"story: {story_i} / {len(stories)}")
                    sub_folder = model_folder + '/' + init_story
                    all_seeds_stories = get_stories(sub_folder)
                    intial_story = get_initial_story(sub_folder)
                    all_seed_flat_stories = preprocess_stories(all_seeds_stories)
                    all_seed_difficulties = get_difficulties(intial_story, all_seed_flat_stories)
                    all_seed_toxicities = get_toxicities(intial_story, all_seed_flat_stories, sub_folder)
                    all_seed_positivities = get_positivities(intial_story, all_seed_flat_stories, sub_folder)
                    all_seed_lengths = [[len(story) for story in seeds] for seeds in all_seed_flat_stories]

                    ## Check if a file exists
                    try:
                        with open(f"Results/data-for-plotting/{model_folder}/toxicity/{init_story}.pkl", 'rb') as f:
                            all_seed_toxicities = pickle.load(f)
                    except:
                        all_seed_toxicities = get_toxicities(intial_story, all_seed_flat_stories, sub_folder)
                        os.makedirs(f"Results/data-for-plotting/{model_folder}/toxicity", exist_ok=True)
                        with open(f"Results/data-for-plotting/{model_folder}/toxicity/{init_story}.pkl", 'wb') as f:
                            pickle.dump(all_seed_toxicities, f)

                    try:
                        with open(f"Results/data-for-plotting/{model_folder}/positivity/{init_story}.pkl", 'rb') as f:
                            all_seed_positivities = pickle.load(f)
                    except:
                        all_seed_positivities = get_positivities(intial_story, all_seed_flat_stories, sub_folder)
                        os.makedirs(f"Results/data-for-plotting/{model_folder}/positivity", exist_ok=True)
                        with open(f"Results/data-for-plotting/{model_folder}/positivity/{init_story}.pkl", 'wb') as f:
                            pickle.dump(all_seed_positivities, f)

                    try:
                        with open(f"Results/data-for-plotting/{model_folder}/difficulty/{init_story}.pkl", 'rb') as f:
                            all_seed_difficulties = pickle.load(f)

                    except:
                        all_seed_difficulties = get_difficulties(intial_story, all_seed_flat_stories)
                        os.makedirs(f"Results/data-for-plotting/{model_folder}/difficulty", exist_ok=True)
                        with open(f"Results/data-for-plotting/{model_folder}/difficulty/{init_story}.pkl", 'wb') as f:
                            pickle.dump(all_seed_difficulties, f)

                    try:
                        with open(f"Results/data-for-plotting/{model_folder}/length/{init_story}.pkl", 'rb') as f:
                            all_seed_lengths = pickle.load(f)
                    except:
                        all_seed_lengths = [[len(story) for story in seeds] for seeds in all_seed_flat_stories]
                        
                        os.makedirs(f"Results/data-for-plotting/{model_folder}/length", exist_ok=True)
                        with open(f"Results/data-for-plotting/{model_folder}/length/{init_story}.pkl", 'wb') as f:
                            pickle.dump(all_seed_lengths, f)
                    #cumulativeness.append(get_cumulativeness_all_seeds(intial_story, all_seed_flat_stories))



                    for s in range(len(all_seed_toxicities)):
                        # toxicities.append([all_seed_toxicities[s][0], all_seed_toxicities[s][-1]])
                        # positivities.append([all_seed_positivities[s][0], all_seed_positivities[s][-1]])
                        # difficulties.append([all_seed_difficulties[s][0], all_seed_difficulties[s][-1]])
                        # lengths.append([all_seed_lengths[s][0], all_seed_lengths[s][-1]])
                        toxicities.append(all_seed_toxicities[s])
                        positivities.append(all_seed_positivities[s])
                        difficulties.append(all_seed_difficulties[s])
                        lengths.append(all_seed_lengths[s])
                        


                os.makedirs(f"Results/Comparisons/init_vs_last/{saving_name}/{model}", exist_ok=True)
                with open(f"Results/Comparisons/init_vs_last/{saving_name}/{model}/toxicity.pkl", 'wb') as f:
                    pickle.dump(toxicities, f)
                with open(f"Results/Comparisons/init_vs_last/{saving_name}/{model}/positivity.pkl", 'wb') as f:
                    pickle.dump(positivities, f)
                with open(f"Results/Comparisons/init_vs_last/{saving_name}/{model}/difficulty.pkl", 'wb') as f:
                    pickle.dump(difficulties, f)
                with open(f"Results/Comparisons/init_vs_last/{saving_name}/{model}/length.pkl", 'wb') as f:
                    pickle.dump(lengths, f)

                all_toxicities.append(toxicities)
                all_positivities.append(positivities)
                all_difficulties.append(difficulties)
                all_lengths.append(lengths)

            with open(f"Results/Comparisons/init_vs_last/{saving_name}/toxicity.pkl", 'wb') as f:
                pickle.dump(all_toxicities, f)
            with open(f"Results/Comparisons/init_vs_last/{saving_name}/positivity.pkl", 'wb') as f:
                pickle.dump(all_positivities, f)
            with open(f"Results/Comparisons/init_vs_last/{saving_name}/difficulty.pkl", 'wb') as f:
                pickle.dump(all_difficulties, f)
            with open(f"Results/Comparisons/init_vs_last/{saving_name}/length.pkl", 'wb') as f:
                pickle.dump(all_lengths, f)

        data['toxicity'] = all_toxicities
        data['positivity'] = all_positivities
        data['difficulty'] = all_difficulties
        data['length'] = all_lengths

        print("Data loaded:")
        print(f"toxicity: {np.array(all_toxicities).shape}")
        print(f"positivity: {np.array(all_positivities).shape}")
        print(f"difficulty: {np.array(all_difficulties).shape}")
        print(f"length: {np.array(all_lengths).shape}")
            

    else:
        all_toxicities = data['toxicity']
        all_positivities = data['positivity']
        all_difficulties = data['difficulty']
        all_lengths = data['length']

        print("Data loaded:")
        print(f"toxicity: {np.array(all_toxicities).shape}")
        print(f"positivity: {np.array(all_positivities).shape}")
        print(f"difficulty: {np.array(all_difficulties).shape}")
        print(f"length: {np.array(all_lengths).shape}")
    
    for i, model in enumerate(models):
        toxicities = all_toxicities[i]
        positivities = all_positivities[i]
        difficulties = all_difficulties[i]
        lengths = all_lengths[i]
        all_task_stories = []
        if np.max(all_lengths[i]) > 10000:
            model_folder = folders[i]
            for init_story in stories:
                sub_folder = model_folder + '/' + init_story
                all_seeds_stories = get_stories(sub_folder)
                all_task_stories.append(all_seeds_stories)



            amax = np.argmax(all_lengths[i], axis=None)
            amaxes = ind = np.unravel_index(amax, np.array(all_lengths[i]).shape)
         

        min_toxicity = np.min(np.array(toxicities)[:, 0])
        max_toxicity = np.max(np.array(toxicities)[:, 0])
        min_positivity = np.min(np.array(positivities)[:, 0])
        max_positivity = np.max(np.array(positivities)[:, 0])
        min_difficulty = np.min(np.array(difficulties)[:, 0])
        max_difficulty = np.max(np.array(difficulties)[:, 0])
        min_length = np.min(np.array(lengths)[:, 0])
        max_length = np.max(np.array(lengths)[:, 0])


        
        x = np.array(toxicities)[:, 0]
        y = np.array(toxicities)[:, -1]

        
        slope, intercept, r, p, sterr = scipy.stats.linregress(x=np.array(toxicities)[:, 0],
                                                        y=np.array(toxicities)[:, -1])
        

        
        
       
        if -1 < slope < 1:
            attractor_position = np.clip(intercept / (1 - slope), -1, 1)
            attractor_strength = 1 - np.abs(slope)
        else:
            attractor_position = 0 ## No attractor
            attractor_strength = 0 ## No attractor
            
        os.makedirs(f"Results/data-for-plotting/attractors/{saving_name}/{model}", exist_ok=True)

        p_value = get_pvalue(x, y, slope, sterr)


        with open(f"Results/data-for-plotting/attractors/{saving_name}/{model}/toxicity.json", 'w') as f:
            json.dump({'attractor_position': attractor_position, 'attractor_strength': attractor_strength, 'p_value': p_value}, f)




        ## Computing attractors after 10 generations
        slope, intercept, r, p, sterr = scipy.stats.linregress(x=np.array(toxicities)[:, 0],
                                                        y=np.array(toxicities)[:, 10])
        if -1 < slope < 1:
            attractor_position = np.clip(intercept / (1 - slope), -1, 1)
            attractor_strength = 1 - np.abs(slope)
        else:
            attractor_position = 0
            attractor_strength = 0


        p_value = get_pvalue(x, y, slope, sterr)

        with open(f"Results/data-for-plotting/attractors/{saving_name}/{model}/toxicity_10.json", 'w') as f:
            json.dump({'attractor_position': attractor_position, 'attractor_strength': attractor_strength, 'p_value': p_value}, f)





        x = np.array(positivities)[:, 0]
        y = np.array(positivities)[:, -1]
        

        slope, intercept, r, p, sterr = scipy.stats.linregress(x=x,
                                                        y=y)        
        if -1 < slope < 1:
            attractor_position = np.clip(intercept / (1 - slope), -1, 1)
            attractor_strength = 1 - np.abs(slope)
        else:
            attractor_position = 0  ## No attractor
            attractor_strength = 0 

        p_value = get_pvalue(x, y, slope, sterr)
            
        with open(f"Results/data-for-plotting/attractors/{saving_name}/{model}/positivity.json", 'w') as f:
            json.dump({'attractor_position': attractor_position, 'attractor_strength': attractor_strength, 'p_value': p_value}, f)





        ## Computing attractors after 10 generations
        slope, intercept, r, p, sterr = scipy.stats.linregress(x=np.array(positivities)[:, 0],
                                                        y=np.array(positivities)[:, 10])
        if -1 < slope < 1:
            attractor_position = np.clip(intercept / (1 - slope), -1, 1)
            attractor_strength = 1 - np.abs(slope)
        else:
            attractor_position = 0
            attractor_strength = 0

        p_value = get_pvalue(x, y, slope, sterr)

        with open(f"Results/data-for-plotting/attractors/{saving_name}/{model}/positivity_10.json", 'w') as f:
            json.dump({'attractor_position': attractor_position, 'attractor_strength': attractor_strength, 'p_value': p_value}, f)

        


        x = np.array(difficulties)[:, 0]
            
        y = np.array(difficulties)[:, -1]
       

        slope, intercept, r, p, sterr = scipy.stats.linregress(x=x,
                                                        y=y)        
        if -1 < slope < 1:
            attractor_position = np.max(intercept / (1 - slope), 0)
            attractor_strength = 1 - np.abs(slope)
        else:
            attractor_position = 0  ## No attractor
            attractor_strength = 0 
        
        p_value = get_pvalue(x, y, slope, sterr)

        with open(f"Results/data-for-plotting/attractors/{saving_name}/{model}/difficulty.json", 'w') as f:
            json.dump({'attractor_position': attractor_position, 'attractor_strength': attractor_strength, 'p_value': p_value}, f)

        



        ## Computing attractors after 10 generations
        slope, intercept, r, p, sterr = scipy.stats.linregress(x=np.array(difficulties)[:, 0],
                                                        y=np.array(difficulties)[:, 10])
        if -1 < slope < 1:
            attractor_position = np.max(intercept / (1 - slope), 0)
            attractor_strength = 1 - np.abs(slope)
        else:
            attractor_position = 0
            attractor_strength = 0

        p_value = get_pvalue(x, y, slope, sterr)

        with open(f"Results/data-for-plotting/attractors/{saving_name}/{model}/difficulty_10.json", 'w') as f:
            json.dump({'attractor_position': attractor_position, 'attractor_strength': attractor_strength, 'p_value': p_value}, f)

        

        x = np.array(lengths)[:, 0]
        y = np.array(lengths)[:, -1]
        

        slope, intercept, r, p, sterr = scipy.stats.linregress(x=x,
                                                        y=y)    
        if -1 < slope < 1:
            attractor_position = np.max(intercept / (1 - slope), 0)
            attractor_strength = 1 - np.abs(slope)
        else:
            attractor_position = 0  ## No attractor
            attractor_strength = 0 

        p_value = get_pvalue(x, y, slope, sterr)

        with open(f"Results/data-for-plotting/attractors/{saving_name}/{model}/length.json", 'w') as f:
            json.dump({'attractor_position': attractor_position, 'attractor_strength': attractor_strength, 'p_value': p_value}, f)

        



        ## Computing attractors after 10 generations
        slope, intercept, r, p, sterr = scipy.stats.linregress(x=np.array(lengths)[:, 0],
                                                        y=np.array(lengths)[:, 10])
        if -1 < slope < 1:
            attractor_position = np.max(intercept / (1 - slope), 0)
            attractor_strength = 1 - np.abs(slope)
        else:
            attractor_position = 0
            attractor_strength = 0

        p_value = get_pvalue(x, y, slope, sterr)

        with open(f"Results/data-for-plotting/attractors/{saving_name}/{model}/length_10.json", 'w') as f:
            json.dump({'attractor_position': attractor_position, 'attractor_strength': attractor_strength, 'p_value': p_value}, f)

        


    return data

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', type=str, nargs='+', help='Folders containing the stories.')
    parser.add_argument('saving_name', type=str, help='Name of the folder to save the plots.')
    args = parser.parse_args()
    plot_intitial_vs_final(args.folders, args.saving_name)