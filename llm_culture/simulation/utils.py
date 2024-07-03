# This file contains the main functions to run the simulation. 
#It is called by the run_simulation.py file.
from llm_culture.simulation.agent import Agent
from tqdm import trange


def init_agents(n_agents, network_structure, prompt_init, prompt_update, initial_story, personality_list, access_url, format_prompt='', start_flag=None, end_flag=None,
                sequence=False, debug=False, model = None, use_vllm = False, sampling_params=None):
    agent_list = []
    wait = 0

    for a in range(n_agents):
        perso = personality_list[a]
        agent = Agent(a, network_structure, prompt_init, prompt_update, initial_story, perso, 
                      access_url= access_url, format_prompt = format_prompt, 
                      start_flag= start_flag, end_flag=end_flag , wait=wait,
                      debug=debug, sequence = sequence, model=model, use_vllm = use_vllm, sampling_params=sampling_params)
        agent_list.append(agent)
        if sequence:
            wait += 1

    return agent_list


def run_simul(access_url, n_timesteps=5, network_structure=None, prompt_init=None, prompt_update=None, initial_story = '', personality_list=None, 
              n_agents=5, format_prompt='', start_flag=None, end_flag=None, sequence=False, output_folder=None, debug=False, model = None, use_vllm = False, sampling_params=None):
    #STRORAGE
    stories_history = []

    #INTIALIZE AGENTS
    agent_list = init_agents(n_agents, network_structure, prompt_init, 
                             prompt_update, initial_story, personality_list, access_url, format_prompt = format_prompt, 
                            start_flag= start_flag, end_flag=end_flag ,
                             sequence=sequence, debug=debug, model=model, use_vllm = use_vllm, sampling_params=sampling_params)
    


    # print the agent id and wait time
    for agent in agent_list:
        agent.update_neighbours(network_structure, agent_list )
        # print(f'Agent: {agent.agent_id}, wait: {agent.wait}')

    #MAIN LOOP
    if output_folder is None:
        state_history_path = 'Results/state_history.json'
    else:
        state_history_path = f'{output_folder}/state_history.json'
    for t in trange(n_timesteps):
        new_stories = update_step(agent_list, t, state_history_path)
        #print(f'Timestep: {t}')
        #print(f'Number of new_stories: {len(new_stories)}')
        stories_history.append(new_stories)

    return stories_history

def update_step(agent_list, timestep, state_history_path):
    #UPDATE LOOP
    new_stories = []

    for agent in agent_list:
        agent.update_prompt()

    for a in trange(len(agent_list)):
        agent = agent_list[a]
       # print(f'Agent: {agent.agent_id}')
        story = agent.get_updated_story()
        if story is not None:
            new_stories.append(story)
        # update the state history of the agent at the current timestep
        # update_state_history(agent, timestep, state_history_path)

    return new_stories