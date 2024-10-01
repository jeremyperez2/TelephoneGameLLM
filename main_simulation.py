import os
import json
import argparse
from pathlib import Path
import networkx as nx
import numpy as np
from tqdm import trange
from llm_culture.simulation.utils import run_simul

# from vllm import LLM, SamplingParams
import os
os.environ['CURL_CA_BUNDLE'] = ''

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run a simulation.')
    parser.add_argument('-na', '--n_agents', type=int, default=2, help='Number of agents.')
    parser.add_argument('-nt', '--n_timesteps', type=int, default=2, help='Number of timesteps.')
    # add an optional argument that will select a preset of parameters from parameters_sets in data
    # argument to select the network structure
    parser.add_argument('-ns', '--network_structure', type=str, default='sequence',
                        choices=['sequence','fully_connected' 'circle', 'caveman', 'random'], help='Network structure.')
    parser.add_argument('-nc', '--n_cliques', type=int, default=2, help='Number of cliques for the Caveman graph')
    parser.add_argument('-ne', '--n_edges', type=int, default=2, help='Number of edges for the Random graph')
    # argument to select the prompt_init from the list of prompts
    parser.add_argument('-pi', '--prompt_init', type=str, default='kid',
                        help='Initial prompt.')
    # argument to select the prompt_update from the list of prompts
    parser.add_argument('-pu', '--prompt_update', type=str, default='kid',
                        help='Update prompt.')    
    # select a personality from the list of personalities (no choices)
    parser.add_argument('-pl', '--personality_list', type=str, default= 'Empty',
                        help='Personality list.')
    # add an option output folder to save the results
    parser.add_argument('-o', '--output_dir', type=str, default='Results/default_folder', help='Output folder.')
    # create optional argument for the output file name to save in the output folder
    parser.add_argument('-of', '--output_file', type=str, default='output.json', help='Output file name.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('-url', '--access_url', type=str, default='', help='URL to send the prompt to.')
    parser.add_argument('-s', '--n_seeds', type=int, default=2, help='Number of seeds')
    parser.add_argument('-f', '--format_prompt', type=str, default='Empty', help='Format of the prompt')
    parser.add_argument('-sf', '--start_flag', type=str, default=None, help='Start flag')
    parser.add_argument('-ef', '--end_flag', type=str, default=None, help='End flag')
    parser.add_argument('-vllm', '--use_vllm', default=True, help='Use VLLM model')
    parser.add_argument('-m', '--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.1', help='VLLM model')
    parser.add_argument('-is', '--initial_story', type=str, default='Empty', help='Initial story')
    parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Temperature for token generation')


    return parser.parse_args()


def prepare_simu(args):
    pass


def main(args=None):

    json_prompt_init = 'llm_culture/data/parameters/prompt_init.json'
    json_prompt_update = 'llm_culture/data/parameters/prompt_update.json'
    json_structure = 'llm_culture/data/parameters/network_structure.json'

    json_personnalities = 'llm_culture/data/parameters/personnalities.json'
    json_format = 'llm_culture/data/parameters/format_prompt.json'
    json_stories = 'llm_culture/data/parameters/stories.json'


    if args is None:
        args = parse_arguments()

    output_dict = {}
    debug = args.debug

    sequence = False

    # If we use a preset, we can use the parameters_sets in data

    # Use the arguments
    use_vllm = args.use_vllm
    model = args.model

    if use_vllm:
        try:
            from vllm import LLM, SamplingParams

            sampling_params = SamplingParams(temperature=0.8, max_tokens=32768)
            llm = LLM(model=model,tensor_parallel_size=2, gpu_memory_utilization=0.7, seed=np.random.randint(0,2**31))
        except Exception as e:
            print("Not using vllm:", str(e))
            use_vllm = False
            llm = None
            sampling_params = None







    n_agents = args.n_agents
    n_timesteps = args.n_timesteps

    network_structure = None
    if args.network_structure == 'sequence':
        network_structure = nx.DiGraph()
        for i in range(n_agents - 1):
            network_structure.add_edge(i, i + 1)
        sequence = True
    elif args.network_structure == 'circle':
        network_structure = nx.cycle_graph(n_agents)
    elif args.network_structure == 'caveman':
        network_structure = nx.connected_caveman_graph(int(args.n_cliques), n_agents // int(args.n_cliques))

    elif args.network_structure == 'fully_connected':
                network_structure = nx.complete_graph(n_agents)
                
    
    elif args.network_structure == 'random':
         network_structure = nx.dense_gnm_random_graph(n_agents, args.n_edges )
    
    ## Adding self-loops:
    for i in range(n_agents):
        network_structure.add_edge(i, i)

    # save adjacency matrix to output_dict
    output_dict["adjacency_matrix"] = nx.to_numpy_array(network_structure).tolist()

    # prompt_init = prompts.prompt_init_dict[args.prompt_init]
    with open(json_prompt_init, 'r') as file:
        data = json.load(file)
        for d in data:
            if d['name'] == args.prompt_init:
                prompt_init = d['prompt']

    # prompt_update = prompts.prompt_update_dict[args.prompt_update]
    with open(json_prompt_update, 'r') as file:
        data = json.load(file)
        for d in data:
            if d['name'] == args.prompt_update:
                prompt_update = d['prompt']

    with open(json_format, 'r') as file:
        data = json.load(file)
        for d in data:
            if d['name'] == args.format_prompt:
                format_prompt = d['prompt']
    
    with open(json_stories, 'r') as file:
        data = json.load(file)
        for d in data:
            if d['name'] == args.initial_story:
                initial_story = d['prompt']


        # personality_dict = getattr(prompts, args.personality_dict)
        # personality_list = prompts.personality_dict_of_lists[args.personality_list]
    if args.personality_list == 'Empty':
        personality_list = [''] * n_agents
    else:
        personality_list = []
        with open(json_personnalities, 'r') as file:
                    data = json.load(file)
                    for perso in args.personality_list:
                        for d in data:
                            if d['name'] == perso:
                                personality_list.append(d['prompt'])


    output_dict["prompt_init"] = [prompt_init]
    output_dict["prompt_update"] = [prompt_update]
    output_dict["personality_list"] = personality_list
    output_dict["format_prompt"] = [format_prompt]
    output_dict["initial_story"] = initial_story

    os.makedirs(os.path.dirname(str(args.output_dir) + '/'), exist_ok=True)
    # t = input(args.output)
    for i in trange(args.n_seeds):
        stories = run_simul(args.access_url, n_timesteps, network_structure, prompt_init,
                            prompt_update, initial_story, personality_list, n_agents, format_prompt=format_prompt,
                            start_flag=args.start_flag, end_flag=args.end_flag,
                            sequence=sequence, output_folder=args.output_dir,
                            debug=debug, use_vllm=use_vllm, model=llm, sampling_params=sampling_params, temperature=args.temperature)
        output_dict["stories"] = stories

        # Save the output to a file
        if args.output_dir:
            with open(Path(args.output_dir, 'output'+str(i)+'.json'), "w") as f:
                json.dump(output_dict, f, indent=4)
        else:
            with open(Path("Results/", 'output'+str(i)+'.json'), "w") as f:
                json.dump(output_dict, f, indent=4)
            return output_dict
        
    # get_all_figures(stories, folder_name)


if __name__ == "__main__":
    main()