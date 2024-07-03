import argparse
import pickle

from llm_culture.analysis.utils import get_SBERT_similarity, get_stories, preprocess_stories
from llm_culture.analysis.utils import get_initial_story
from llm_culture.analysis.plots import *
import seaborn as sns
from tqdm import trange

def plot_convergence(folders, saving_name, stories, models, data = None, prompt = None):

    if data is None:
        all_similaritys = []

        for i in trange(len(folders)):
            model = models[i]

            try:
                with open(f"Results/Comparisons/convergence/{saving_name}/{model}/simildaritys.pkl", 'rb') as f:
                    similaritys = pickle.load(f)
            except:

                model_folder = folders[i]
                similaritys = []
                model = models[i]


                for i in range(len(stories)):
                    init_story1 = stories[i]
                    sub_folder1 = model_folder + '/' + init_story1
                    all_seeds_stories1 = get_stories(sub_folder1)
                    intial_story1 = get_initial_story(sub_folder1)
                    all_seed_flat_stories1 = preprocess_stories(all_seeds_stories1)
                    for j in trange(len(stories)):
                        init_story2 = stories[j]
                        sub_folder2 = model_folder + '/' + init_story2


                        all_seeds_stories2 = get_stories(sub_folder2)
                        intial_story2 = get_initial_story(sub_folder2)

                        all_seed_flat_stories2 = preprocess_stories(all_seeds_stories2)
                        initial_sim = get_SBERT_similarity(intial_story1, intial_story2)
                        for k in range(len(all_seed_flat_stories1)):
                            story_1 = all_seed_flat_stories1[k][-1]
                            for l in range(len(all_seed_flat_stories2)):
                                story_2 = all_seed_flat_stories2[l][-1]
                                if not( i==j and k == l):
                                    final_sim = get_SBERT_similarity(story_1, story_2)
                                    similaritys.append([initial_sim, final_sim])

                       
                        # for gen in range(len(all_seed_flat_stories1[0])):
                        #     similaritys.append([])
                        #     for k in range(len(all_seed_flat_stories1)):
                        #         story_1 = all_seed_flat_stories1[k][gen]
                        #         for l in range(len(all_seed_flat_stories2)):
                        #             story_2 = all_seed_flat_stories2[l][gen]
                        #             if not( k == l):
                        #                 similaritys[-1].append(get_SBERT_similarity(story_1, story_2))
           
            all_similaritys.append(similaritys)

        data = {'all_similaritys': all_similaritys}
    else:
        all_similaritys = data['all_similaritys']


    return data

            



            
                

