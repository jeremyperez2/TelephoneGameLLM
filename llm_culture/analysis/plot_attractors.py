


import json

from matplotlib import pyplot as plt
import os
from tqdm import trange
import numpy as np


def plot_attractors(models, prompts, stories, saving_name, n_seeds = 5, data = None):

    if data is None:

        attractor_toxicity_position = {m: {p: [] for p in prompts} for m in models}
        attractor_toxicity_strength = {m: {p: [] for p in prompts} for m in models}
        attractor_toxicity_pvalue = {m: {p: [] for p in prompts} for m in models}

        attractor_toxicity_position_10 = {m: {p: [] for p in prompts} for m in models}
        attractor_toxicity_strength_10 = {m: {p: [] for p in prompts} for m in models}
        attractor_toxicity_pvalue_10 = {m: {p: [] for p in prompts} for m in models}



        attractor_positivity_position = {m: {p: [] for p in prompts} for m in models}
        attractor_positivity_strength = {m: {p: [] for p in prompts} for m in models}
        attractor_positivity_pvalue = {m: {p: [] for p in prompts} for m in models}

        attractor_positivity_position_10 = {m: {p: [] for p in prompts} for m in models}
        attractor_positivity_strength_10 = {m: {p: [] for p in prompts} for m in models}
        attractor_positivity_pvalue_10 = {m: {p: [] for p in prompts} for m in models}


        attractor_difficulty_position = {m: {p: [] for p in prompts} for m in models}
        attractor_difficulty_strength = {m: {p: [] for p in prompts} for m in models}
        attractor_difficulty_pvalue = {m: {p: [] for p in prompts} for m in models}

        attractor_difficulty_position_10 = {m: {p: [] for p in prompts} for m in models}
        attractor_difficulty_strength_10 = {m: {p: [] for p in prompts} for m in models}
        attractor_difficulty_pvalue_10 = {m: {p: [] for p in prompts} for m in models}

        attractor_length_position = {m: {p: [] for p in prompts} for m in models}
        attractor_length_strength = {m: {p: [] for p in prompts} for m in models}
        attractor_length_pvalue = {m: {p: [] for p in prompts} for m in models}

        attractor_length_position_10 = {m: {p: [] for p in prompts} for m in models}
        attractor_length_strength_10 = {m: {p: [] for p in prompts} for m in models}
        attractor_length_pvalue_10 = {m: {p: [] for p in prompts} for m in models}

        
        
        for model in models:
            for prompt in prompts:
                
                with open(f"Results/data-for-plotting/attractors/{saving_name}/{prompt}/{model}/toxicity.json", 'r') as f:
                    attractor_toxicity = json.load(f)
                    attractor_toxicity_position[model][prompt] = attractor_toxicity['attractor_position']
                    attractor_toxicity_strength[model][prompt] = attractor_toxicity['attractor_strength']
                    attractor_toxicity_pvalue[model][prompt] = attractor_toxicity['p_value']

                with open(f"Results/data-for-plotting/attractors/{saving_name}/{prompt}/{model}/toxicity_10.json", 'r') as f:
                    attractor_toxicity = json.load(f)
                    attractor_toxicity_position_10[model][prompt] = attractor_toxicity['attractor_position']
                    attractor_toxicity_strength_10[model][prompt] = attractor_toxicity['attractor_strength']
                    attractor_toxicity_pvalue_10[model][prompt] = attractor_toxicity['p_value']
                

                
                with open(f"Results/data-for-plotting/attractors/{saving_name}/{prompt}/{model}/positivity.json", 'r') as f:
                    attractor_positivity = json.load(f)
                    attractor_positivity_position[model][prompt] = attractor_positivity['attractor_position']
                    attractor_positivity_strength[model][prompt] = attractor_positivity['attractor_strength']
                    attractor_positivity_pvalue[model][prompt] = attractor_positivity['p_value']

                with open(f"Results/data-for-plotting/attractors/{saving_name}/{prompt}/{model}/positivity_10.json", 'r') as f:
                    attractor_positivity = json.load(f)
                    attractor_positivity_position_10[model][prompt] = attractor_positivity['attractor_position']
                    attractor_positivity_strength_10[model][prompt] = attractor_positivity['attractor_strength']
                    attractor_positivity_pvalue_10[model][prompt] = attractor_positivity['p_value']

                with open(f"Results/data-for-plotting/attractors/{saving_name}/{prompt}/{model}/difficulty.json", 'r') as f:
                    attractor_difficulty = json.load(f)
                    attractor_difficulty_position[model][prompt] = attractor_difficulty['attractor_position']
                    attractor_difficulty_strength[model][prompt] = attractor_difficulty['attractor_strength']
                    attractor_difficulty_pvalue[model][prompt] = attractor_difficulty['p_value']

                with open(f"Results/data-for-plotting/attractors/{saving_name}/{prompt}/{model}/difficulty_10.json", 'r') as f:
                    attractor_difficulty = json.load(f)
                    attractor_difficulty_position_10[model][prompt] = attractor_difficulty['attractor_position']
                    attractor_difficulty_strength_10[model][prompt] = attractor_difficulty['attractor_strength']
                    attractor_difficulty_pvalue_10[model][prompt] = attractor_difficulty['p_value']

                with open(f"Results/data-for-plotting/attractors/{saving_name}/{prompt}/{model}/length.json", 'r') as f:
                    attractor_length = json.load(f)
                    attractor_length_position[model][prompt] = attractor_length['attractor_position']
                    attractor_length_strength[model][prompt] = attractor_length['attractor_strength']
                    attractor_length_pvalue[model][prompt] = attractor_length['p_value']

                with open(f"Results/data-for-plotting/attractors/{saving_name}/{prompt}/{model}/length_10.json", 'r') as f:
                    attractor_length = json.load(f)
                    attractor_length_position_10[model][prompt] = attractor_length['attractor_position']
                    attractor_length_strength_10[model][prompt] = attractor_length['attractor_strength']
                    attractor_length_pvalue_10[model][prompt] = attractor_length['p_value']

        data = {'attractor_toxicity_position': attractor_toxicity_position,
                'attractor_toxicity_strength': attractor_toxicity_strength,
                'attractor_positivity_position': attractor_positivity_position,
                'attractor_positivity_strength': attractor_positivity_strength,
                'attractor_difficulty_position': attractor_difficulty_position,
                'attractor_difficulty_strength': attractor_difficulty_strength,
                'attractor_length_position': attractor_length_position,
                'attractor_length_strength': attractor_length_strength,
                'attractor_toxicity_position_10': attractor_toxicity_position_10,
                'attractor_toxicity_strength_10': attractor_toxicity_strength_10,
                'attractor_positivity_position_10': attractor_positivity_position_10,
                'attractor_positivity_strength_10': attractor_positivity_strength_10,
                'attractor_difficulty_position_10': attractor_difficulty_position_10,
                'attractor_difficulty_strength_10': attractor_difficulty_strength_10,
                'attractor_length_position_10': attractor_length_position_10,
                'attractor_length_strength_10': attractor_length_strength_10,
                'attractor_toxicity_pvalue': attractor_toxicity_pvalue,
                'attractor_positivity_pvalue': attractor_positivity_pvalue,
                'attractor_difficulty_pvalue': attractor_difficulty_pvalue,
                'attractor_length_pvalue': attractor_length_pvalue,
                'attractor_toxicity_pvalue_10': attractor_toxicity_pvalue_10,
                'attractor_positivity_pvalue_10': attractor_positivity_pvalue_10,
                'attractor_difficulty_pvalue_10': attractor_difficulty_pvalue_10,
                'attractor_length_pvalue_10': attractor_length_pvalue_10}
                  

    else:
        attractor_toxicity_position = data['attractor_toxicity_position']
        attractor_toxicity_strength = data['attractor_toxicity_strength']
        attractor_positivity_position = data['attractor_positivity_position']
        attractor_positivity_strength = data['attractor_positivity_strength']
        attractor_difficulty_position = data['attractor_difficulty_position']
        attractor_difficulty_strength = data['attractor_difficulty_strength']
        attractor_length_position = data['attractor_length_position']
        attractor_length_strength = data['attractor_length_strength']



    return data




        



