# When LLMs Play the Telephone Game: Cumulative Changes and Attractors in Iterated Cultural Transmissions
![introduction_figure](https://github.com/jeremyperez2/TelephoneGameLLM/assets/152488508/6ec97899-f8fd-48bd-a5da-d8fac458bfe8)

This repo contains code for the paper [_"When LLMs Play the Telephone Game: Cumulative Changes and Attractors in Iterated Cultural Transmissions"_](https://sites.google.com/view/telephone-game-llm)

The paper is also accompanied by a [website](https://sites.google.com/view/telephone-game-llm) that features a Data Explorer tool, allowing to look at the simulated data used in the paper. 

## How to use
```
conda env create -f telephone_llm.yml
conda activate telephone_llm
```

### Reproducing the simulations

#### Set-up environment variables

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


#### Run the simulations

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

### Reproducing the analysis and figures

- Perform the analyses:
  
  After running simulations for several models, you can compare them by running:
  
  ```
  python3 run_all_analyses.py -m <_model_name1_> <_model_name2_> <_model_name3_> <...> -sn <saving_name>
  ```

- Generate the figures
 ```
cd plots
python3 plots/plot_all_figures.py
 ```
Figures will be stored in Figures/figures_<current_date>




