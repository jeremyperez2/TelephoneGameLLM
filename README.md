# When LLMs Play the Telephone Game: Cumulative Changes and Attractors in Iterated Cultural Transmissions
![introduction_figure](https://github.com/jeremyperez2/TelephoneGameLLM/assets/152488508/6ec97899-f8fd-48bd-a5da-d8fac458bfe8)

This repo contains code for the paper [_"When LLMs Play the Telephone Game: Cumulative Changes and Attractors in Iterated Cultural Transmissions"_](https://sites.google.com/view/llms-play-telephone)

The paper is also accompanied by a [website](https://sites.google.com/view/llms-play-telephone) that features a Data Explorer tool, allowing to look at the simulated data used in the paper. 

# How to use
```
conda env create -f telephone_llm.yml
conda activate telephone_llm
```



## Set-up environment variables

- To use OpenAI models with the OpenAI API, set the OPENAI_API_KEY env variable (e.g. in your .bashrc):
```
export OPENAI_API_KEY="<your_key>"
```

- To use OpenAI model with the Azure API, set the variables for each model, for example:
```
export AZURE_OPENAI_ENDPOINT_gpt_35_turbo_0125="<your_endpoint>"
export AZURE_OPENAI_KEY_gpt_35_turbo_0125="<your_key>"
export AZURE_OPENAI_API_VERSION_gpt_35_turbo_0125="<your_version>"
```
- To use huggingface models, set the HF_HOME env variable to define your cache directory:

```
export HF_HOME="$HOME/.cache/huggingface"
```

- To use huggingface gated models, set the HF_TOKEN env variable

```
export HF_TOKEN="<your_token>"
```

## Reproducing the simulations comparing different models and prompts

### Run the simulations

The script _slurm_script.sh_ takes in argument a _model name_ and a _prompt name_. The model name is the name of LLM that you want to use (e.g. meta-llama/Meta-Llama-3-8B). Prompt names are either "_rephrase_", "_inspiration_" or "_continue_". 

- Regular machine:
```
for i in {0..99}; do SLURM_ARRAY_TASK_ID=$i bash slurm_script.sh <model_name> <prompt_name> ; done
```
This will sequentially launch 100 chains (20 initial texts * 5 seeds) for the specified <_model_name_> and <_prompt_name_>


- Slurm-based machine:
```
sbatch slurm_script.sh <model_name> <prompt_name>
```

Generated texts will be stored in Results/<_model_name_>/<_prompt_name_>

### Run the analyses

- Perform the analyses:
  
  After running simulations for several models, you can compare them by running:
  
  ```
  python3 run_all_analyses.py -m <_model_name1_> <_model_name2_> <_model_name3_> <...> -sn data-for-plotting
  ```

  This will create a file 'data-for-plotting/all_data.pkl' that can be used for generating figures and running the statistical tests

### Generate the figures
 ```
cd plots
python3 plot_all_figures.py <_model_name1_> <_model_name2_> <_model_name3_> <...>
 ```
Figures will be stored in Figures/figures_<current_date>

- Run the statistical models

```
  python3 stat_models.py <_model_name1_> <_model_name2_> <_model_name3_> <...>
  ```

  Figures and tables will be stored in Results/StatModels<current_date>



## Reproducing the experiments on effect of temperature

### Run the simulations
- Regular machine:
```
for i in {0..99}; do SLURM_ARRAY_TASK_ID=$i bash slurm_script_temperature.sh <model_name> <prompt_name> ; done
```
This will sequentially launch 100 chains (20 initial texts * 5 seeds) for the specified <_model_name_> and <_prompt_name_>, with temperature taking values in [0, 0.4, 1.2, 1.6]


- Slurm-based machine:
```
sbatch slurm_script_temperature.sh <model_name> <prompt_name>
```

Generated texts will be stored in Results/<_model_name_>_<_temperature_value_>/<_prompt_name_>

### Run the analyses

 ```
  python3 run_all_analyses.py -m <_model_name1_>_<_temperature_value1_> <_model_name1_>_<_temperature_value2_> <...> -sn data-for-plotting-temperature

  ```

  This will create a file 'data-for-plotting-temperature/all_data.pkl' that can be used for generating figures

### Generate the figures

 ```
  python3 plot_effect_of_temperature.py

  ```

  Figures will be stored in Figures_temperature/figures_<current_date>



## Reproducing the experiments on effect of fine-tuning

### Run the simulations

To compare Base and Instruct models, follow to same steps as when comparing differnt models, but pass as <_model_name_> arguments both the Base and Instruct version. For instance, with "Meta-Llama-3-70B-Base" and "Meta-Llama-3-70B-Instruct"

- Regular machine:
```
for i in {0..99}; do SLURM_ARRAY_TASK_ID=$i bash slurm_script_temperature.sh <model_name> <prompt_name> ; done
```
This will sequentially launch 100 chains (20 initial texts * 5 seeds) for the specified <_model_name_> and <_prompt_name_>, with temperature taking values in [0, 0.4, 1.2, 1.6]


- Slurm-based machine:
```
sbatch slurm_script_temperature.sh <model_name> <prompt_name>
```


Generated texts will be stored in Results/<_model_name_>/<_prompt_name_>

### Run the analyses

 ```
  python3 run_all_analyses.py -m <_model_name1_> <_model_name2_> <...> -sn data-for-plotting-finetuning

  ```

  where <_model_name1_> is the Base version and <_model_name2_> is the Instruct version. 

  This will create a file 'data-for-plotting-finetuning/all_data.pkl' that can be used for generating figures

### Generate the figures

 ```
  python3 plot_effect_of_finetuning.py

  ```

  Figures will be stored in Figures_finetuning/figures_<current_date>
