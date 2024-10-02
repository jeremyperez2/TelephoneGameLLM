import argparse
import os
import pickle

from llm_culture.analysis.plot_attractors import plot_attractors
from llm_culture.analysis.plot_convergence import plot_convergence
from llm_culture.analysis.run_comparison_analysis import main_analysis
from llm_culture.analysis.run_initial_vs_last_story import plot_intitial_vs_final
import sys
from tqdm import trange


def main(models, saving_name, prompts = ['rephrase', 'inspiration', 'continue']):
    sizes = {'ticks': 16,
                  'labels': 18,
                  'legend': 16,
                  'title': 23,
                  'matrix': 8}
    
 
    

    print(f"Models: {models}")

    stories = ['Difficulty22.9',
                'Length4986.0',
                'Difficulty24.83',
                'Positivity-0.5106',
                'Toxicity0.1864273',
                'Length1635.0',
                'Toxicity0.8434329',
                'Toxicity0.9934216',
                'Positivity0.5994',
                'Length518.0',
                'Length2752.0',
                'Difficulty14.78',
                'Difficulty32.24',
                'Toxicity0.00113261',
                'Difficulty11.57',
                'Toxicity0.46547398',
                'Positivity-0.9738',
                'Length3869.0',
                'Positivity0.9019',
                'Positivity-0.0653']


    try:
        with open(f"Results/data-for-plotting/all_dataa.pkl", 'rb') as f: #TODO change to all_data.pkl
            all_data = pickle.load(f)
        data_loaded = True

        print("Data loaded successfully")
    except:

        all_data = {'evolution' : {p: {s : [] for s in stories} for p in prompts},
                    'plot_intitial_vs_final' : {p: [] for p in prompts},
                    'plot_convergence' : {p: [] for p in prompts},
                    'plot_cumulativeness' : [],
                    'plot_attractors' : []}
        data_loaded = False
    
        

    for i in trange(len(prompts)):
        p = prompts[i]
        print(f"\n\nRunning evolution analysis for {p} ...")
        for j in trange(len(stories)):
            s = stories[j]
        
            sub_folders = [f"Results/{model}/{p}/{s}" for model in models]

            if not data_loaded:
                all_data['evolution'][p][s] = main_analysis(sub_folders, plot = False, scale_y_axis = False, labels = models, sizes = sizes, saving_folder = f'{saving_name}/{p}/{s}')
            else:

                main_analysis(sub_folders, plot = False, scale_y_axis = False, labels = models, sizes = sizes, saving_folder = f'{saving_name}/{p}/{s}', all_data=all_data['evolution'][p][s])


        sub_folders = [f"Results/{model}/{p}" for model in models]

        print(f"\n\nRunning attraction analysis for {p} ...")
        if data_loaded:

            plot_intitial_vs_final(sub_folders, f'{saving_name}/{p}', stories, models, all_data['plot_intitial_vs_final'][p])
        else:
            all_data['plot_intitial_vs_final'][p] = plot_intitial_vs_final(sub_folders, f'{saving_name}/{p}', stories, models)

        # print(f"\n\nRunning convergence analysis for {p} ...")
        # if data_loaded:
        #     plot_convergence(sub_folders, f'{saving_name}/{p}', stories, models, all_data['plot_convergence'][p])
        # else:
        #     all_data['plot_convergence'][p] = plot_convergence(sub_folders, f'{saving_name}/{p}', stories, models, prompt = p)
    
   

    print(f"\n\nPlotting attractors ...")
    if data_loaded: 
        plot_attractors(models, prompts, stories, saving_name, all_data['plot_attractors'])
    else:
        all_data['plot_attractors'] = plot_attractors(models, prompts, stories, saving_name)
        os.makedirs(f"Results/data-for-plotting/{saving_name}", exist_ok=True)
        with open(f"Results/data-for-plotting/all_data.pkl", 'wb') as f:
            pickle.dump(all_data, f)
    
    # os.makedirs(f"Results/data-for-plotting/{saving_name}", exist_ok=True)
    # with open(f"Results/data-for-plotting/all_data_temp.pkl", 'wb') as f:
    #     pickle.dump(all_data, f)
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the analysis on the output of the simulation')
    parser.add_argument('-m', '--models', type=str, nargs='+', help='Models to compare')
    parser.add_argument('-sn', '--saving_name', type=str, help='Name of the folder to save the results')
    parser.add_argument('-p', '--prompt', nargs='+', help='Prompt to compare', default=['rephrase', 'inspiration', 'continue'])
    args = parser.parse_args()

    main(args.models, args.saving_name, args.prompt)

