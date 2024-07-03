import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import pandas as pd
import numpy as np

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import joypy
from matplotlib.ticker import FormatStrFormatter


class Plotter():

    def __init__(self, 
                 all_data,
                saving_name="gaia",
                n_generations=50,
                models=["gpt-3.5-turbo-0125", "Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2", "Meta-Llama-3-70B-Instruct", "Mixtral-8x7B-Instruct-v0.1"],
                model_indices=[0, 1, 2, 3, 4, 5],
                measures=["toxicity", "positivity", "difficulty", "length"], 
                prompts=["rephrase", "inspiration", "continue"],
                prompt_names=["Rephrase", "Take Inspiration", "Continue"],

                stories=["Difficulty22.9",
                            "Length4986.0",
                            "Difficulty24.83",
                            "Positivity-0.5106",
                            "Toxicity0.1864273",
                            "Length1635.0",
                            "Toxicity0.8434329",
                            "Toxicity0.9934216",
                            "Positivity0.5994",
                            "Length518.0",
                            "Length2752.0",
                            "Difficulty14.78",
                            "Difficulty32.24",
                            "Toxicity0.00113261",
                            "Difficulty11.57",
                            "Toxicity0.46547398",
                            "Positivity-0.9738",
                            "Length3869.0",
                            "Positivity0.9019",
                            "Positivity-0.0653"],
                example_stories={"toxicity": "Toxicity0.46547398",
                                 "positivity": "Positivity-0.9738",
                                 "difficulty": "Difficulty32.24",
                                 "length": "Length4986.0"},
                collpase_stories=['Positivity-0.0653'],
                n_seeds = 5,
                model_nicks={"gpt-3.5-turbo-0125": "GPT3.5", 
                            "Meta-Llama-3-8B-Instruct": "Llama3-8B", 
                            "Mistral-7B-Instruct-v0.2": "Mistral7B", 
                            "Meta-Llama-3-70B-Instruct": "Llama3-70B",
                            "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B"},
                model_colors={"initial": "#808080",
                            "gpt-3.5-turbo-0125": "#7aa16a", 
                            "Meta-Llama-3-8B-Instruct": "#004c97", 
                            "Mistral-7B-Instruct-v0.2": "#c04732",
                            "Meta-Llama-3-70B-Instruct": "#DA70D6",
                            "Mixtral-8x7B-Instruct-v0.1": "#f0a632"},
                font_size=16):
        self.all_data = all_data
        self.saving_name = saving_name
        self.n_generations = n_generations
        self.n_stories = len(stories)
        self.n_seeds = n_seeds
        self.models = models
        self.model_indices = model_indices
        self.model_nicks = model_nicks
        self.n_models = len(models)
        self.measures = measures
        self.prompts = prompts
        self.prompt_names = prompt_names
        self.stories = stories
        self.example_stories = example_stories
        self.example_indices = {measure: stories.index(example_stories[measure]) for measure in measures}
        self.collpase_stories = collpase_stories
        self.collpase_indices = [stories.index(story) for story in collpase_stories]

        # set up plotting style
        self.font_size = font_size
        self.model_colors = model_colors
        self._set_plot_params()
    
    def load_results(self, saving_name="gaia"):
        
        

        # load evolution results
        with open(f"Results/{saving_name}/all_evolutions.pkl", "rb") as file:
            all_evolutions = pickle.load(file)
        self.all_evolutions = all_evolutions

        # load cumulativeness results 
        with open(f"Results/{saving_name}/cumulativeness.pkl", "rb") as file:
            cumulativeness = pickle.load(file)
        self.cumulativeness = cumulativeness
  
        # load initial vs final cumulativeness
        with open(f"Results/{saving_name}/all_initial_cumuls.pkl", "rb") as file:
            all_initial_cumuls = pickle.load(file)
        with open(f"Results/{saving_name}/all_final_cumuls.pkl", "rb") as file:
            all_final_cumuls = pickle.load(file)
        self.all_initial_cumuls = all_initial_cumuls
        self.all_final_cumuls = all_final_cumuls

        # load initial vs final similarities
        with open(f"Results/{saving_name}/initial_sim.pkl", "rb") as file:
            initial_sim = pickle.load(file)
        with open(f"Results/{saving_name}/final_sim.pkl", "rb") as file:
            final_sim = pickle.load(file)
        self.initial_sim = initial_sim
        self.final_sim = final_sim

        with open(f"Results/{saving_name}/all_after_10_cumuls.pkl", "rb") as file:
            all_after_10_cumuls = pickle.load(file)
        self.all_after_10_cumuls = all_after_10_cumuls

        # with open(f"Results/{saving_name}/all_after_10_sim", "rb") as file:

        # load attractor positions and strengths
        with open(f"Results/{saving_name}/all_attr_positions.pkl", "rb") as file:
            all_attr_positions = pickle.load(file)
        with open(f"Results/{saving_name}/all_attr_strengths.pkl", "rb") as file:
            all_attr_strengths = pickle.load(file)
        self.all_attr_positions = all_attr_positions
        self.all_attr_strengths = all_attr_strengths

        with open(f"Results/{saving_name}/all_attr_positions_10.pkl", "rb") as file:
            all_attr_positions_10 = pickle.load(file)
        with open(f"Results/{saving_name}/all_attr_strengths_10.pkl", "rb") as file:
            all_attr_strengths_10 = pickle.load(file)
        
        with open(f"Results/{saving_name}/change_per_generation.pkl", "rb") as file:
            self.change_per_generation = pickle.load(file)
        
        with open(f"Results/{saving_name}/directional_change_per_generation.pkl", "rb") as file:
            self.directional_change_per_generation = pickle.load(file)

        self.all_attr_positions_10 = all_attr_positions_10
        self.all_attr_strengths_10 = all_attr_strengths_10


    def _set_plot_params(self):
        """
        Set parameters for plotting
        """
        plt.rcParams["errorbar.capsize"] = 5
        self.legend_font_size = self.font_size-2
        # plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = self.font_size

        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.labelspacing"] = 0
        plt.rcParams["legend.columnspacing"] = 0.7
        plt.rcParams["legend.handletextpad"] = 0.1

        plt.rcParams["lines.linewidth"] = 2
        plt.rcParams["axes.spines.right"] = False
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.default"] = "regular"
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"   
        # plt.rcParams["mathtext.rm"] = "Arial"
        # plt.rcParams["mathtext.it"] = "Arial:italic"
        # plt.rcParams["mathtext.bf"] = "Arial:bold"
        self.fade = 0.4

    
    def _get_model_handles(self, marker="_", markersize=6):
        handles = []
        for cidx, c in enumerate(self.models):
            handles.append(mlines.Line2D([], [], c=self.model_colors[c], marker=marker, markersize=markersize,
                                        markerfacecolor=self.model_colors[c], markeredgecolor=self.model_colors[c], ls="None", 
                                        label=self.models[cidx]))
        return handles



    def get_all_similarities(self, all_data, model, prompt):
        model_i = self.models.index(model)
        all_similarities = {}
        for gen in range(self.n_generations):
            all_similarities[gen] = []
            for ss_i in range(self.n_stories*self.n_seeds*(self.n_stories*self.n_seeds-1)):
                all_similarities[gen].append(np.squeeze(all_data["plot_convergence"][prompt]["all_similaritys"][model_i][ss_i]))
                all_similarities[gen].append(np.squeeze(all_data["plot_convergence"][prompt]["all_similaritys"][model_i][ss_i][gen]))
        return all_similarities


    def plot_cumulativeness(self, n_rows=1, w=6, h=4, ylim=(-0.5, 15), legend_i=0, models_first=False, fig=None, axs=None):
        n_cols = len(self.measures)//n_rows
        
        if axs is None:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(w*n_cols, h*n_rows))
                    
        for i, (ax, measure) in enumerate(zip(axs.ravel(), self.measures)):
            if models_first: # compare across prompts
                xticks = [p.capitalize() for m in self.models for p in self.prompts_names]
                means = [np.nanmean(self.cumulativeness[measure][m][p]) for m in self.models for p in self.prompts]
                sems = [np.nanstd(self.cumulativeness[measure][m][p] / np.sqrt(len(self.cumulativeness[measure][m][p]))) for m in self.models for p in self.prompts]
                ax.bar(range(len(xticks)), means, yerr=sems, color=[self.model_colors[m] for m in self.models for p in self.prompts])
                ax.set_xticks(range(len(xticks)), xticks, rotation=45, ha="right")
            else: # prompts first, compare across models
                xticks = [p.capitalize() for p in self.prompts_names for m in self.models]
                xlabels = [xtick for xi, xtick in enumerate(xticks) if xi in np.arange(self.n_models//2, self.n_models*len(self.prompts), self.n_models)]
                means = [np.nanmean(self.cumulativeness[measure][m][p]) for p in self.prompts for m in self.models]
                sems = [np.nanstd(self.cumulativeness[measure][m][p] / np.sqrt(len(self.cumulativeness[measure][m][p]))) for p in self.prompts for m in self.models]
                ax.bar(range(len(xticks)), means, yerr=sems, color=[self.model_colors[m] for p in self.prompts for m in self.models ])
                ax.set_xticks(np.arange(self.n_models//2, self.n_models*len(self.prompts), self.n_models), xlabels)
            
            
            ax.set_ylabel("Cumulativeness")
            ax.set_title(measure.capitalize())
            # if measure != "length": 
            ax.set_ylim(ylim)
            if i == legend_i:
                legend = ax.legend(handles=self._get_model_handles("s"), 
                                   loc="upper right", fontsize=self.legend_font_size, title=f"Model")


        fig.tight_layout()
        return fig, axs
    

        
    def plot_ridge_distribution(self, measure, w=6, h=4, mmpp=True, legend_i=0, legend_pos=(0.7,1), ylim=None, with_suptitle=False, fig=None, axs=None):
        initial = self.all_initial_cumuls[measure]
        final = self.all_final_cumuls[measure]

        if mmpp:  # multiple models per plot
            data = []
            labels = []
            for i, p in enumerate(self.prompts):
                for m in self.models:
                    for value in np.array(initial[m][p]).flatten():
                        data.append(value)
                        labels.append(f"Initial {p} - {m}")
                    for value in np.array(final[m][p]).flatten():
                        data.append(value)
                        labels.append(f"Final {p} - {m}")
            
            df = pd.DataFrame({'Value': data, 'Label': labels})
            fig, ax = joypy.joyplot(df, by='Label', figsize=(w, h), fade=True, legend=False)
        
        else:
            fig, axs = plt.subplots(self.n_models, len(self.prompts), figsize=(w*(len(self.prompts)), h*(self.n_models)))
            for i, m in enumerate(self.models):
                data = []
                labels = []
                for j, p in enumerate(self.prompts):
                    for value in np.array(initial[m][p]).flatten():
                        data.append(value)
                        labels.append(f"Initial {p}")
                    for value in np.array(final[m][p]).flatten():
                        data.append(value)
                        labels.append(f"Final {p}")

                df = pd.DataFrame({'Value': data, 'Label': labels})
                joypy.joyplot(df, by='Label', ax=axs[i, j], fade=True, legend=False)
                axs[i, 0].annotate(m, xy=(-0.2, 0.5), xycoords="axes fraction", fontsize=plt.rcParams["font.size"], ha="center", va="center", rotation="vertical")
        
        fig.tight_layout()
        if with_suptitle:
            fig.suptitle(f"Initial and Final {measure.capitalize()} Distribution", y=1.01)

        return fig, axs

    
    def plot_convergence(self, w=6, h=4, mmpp=True, legend_i=0, with_suptitle=False, markerfill_by_pos=True, alpha=0.2, fig=None, axs=None, first_10=False):
    
        if first_10:
            final = self.all_after_10_cumuls
        else:
            final = self.final_sim
        initial = self.initial_sim
        
        num_prompts = len(self.prompts)
        
        if mmpp:  # multiple models per plot
            if axs is None:
                fig, axs = plt.subplots(1, num_prompts, figsize=(w * num_prompts, h))

            for i, (ax, p) in enumerate(zip(axs, self.prompts)):
                ax.set_xlabel("Initial similarity")
                ax.set_ylim(-0.1, 1.1)
                ax.set_xlim(-0.1, 1.1)
                if i != 0:
                    ax.set_yticks([])
                ax.set_title(self.prompt_names[i])
                ax.plot([0, 1], [0, 1], ls="--", c="k")

                for m in self.models:
                    x = initial[m][p]
                    y = final[m][p]
                    if markerfill_by_pos:
                        ax.scatter(x=x[y < x], y=y[y < x], color="none", edgecolor=self.model_colors[m], alpha=alpha)
                        ax.scatter(x=x[y >= x], y=y[y >= x], color="none", edgecolor=self.model_colors[m], alpha=alpha)
                        sns.regplot(x=x, y=y, line_kws={"color": self.model_colors[m]}, scatter=False, ax=ax)
                    else:
                        sns.regplot(x=x, y=y, line_kws={"color": self.model_colors[m]}, scatter_kws={"color": self.model_colors[m], "alpha": alpha}, ax=ax)

                if i == legend_i:
                    ax.legend(handles=self._get_model_handles("o"), loc="best", fontsize=self.legend_font_size, title="Model")

            # Create ridge plots for the bottom row
            sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
            
            # Define a custom scatterplot function to use different alphas
            def kdeplot_with_alpha(data, x, **kwargs):
                alpha_values = {
                    "Initial": 0.0,
                    "First": 0.5,
                    "Final": 1.0,
                }
                for type, alpha in alpha_values.items():
                    subset = data[data["Type"] == type]
                    sns.kdeplot(data=subset, x=x, alpha=alpha, bw_adjust=1, clip_on=False, fill=True, linewidth=1.5, **kwargs)

            # Example loop over prompts and models, assuming self.prompts and self.models are defined
            for p in self.prompts:
                ridge_data = []

                for m in self.models:
                    ridge_data.append(pd.DataFrame({
                        "Similarity": initial[self.models[0]][p],
                        "Type": "Initial",
                        "Model": m,
                        "Prompt": p,
                        "Color": "initial"
                    }))
                    # ridge_data.append(pd.DataFrame({
                    #     "Similarity": final_10[m][p],
                    #     "Type": "First",
                    #     "Model": m,
                    #     "Prompt": p,
                    #     "Color": m
                    # }))
                    ridge_data.append(pd.DataFrame({
                        "Similarity": final[m][p],
                        "Type": "Final",
                        "Model": m,
                        "Prompt": p,
                        "Color": m
                    }))
                ridge_data = pd.concat(ridge_data)

                # Different color for each model, different line style for each type
                g = sns.FacetGrid(ridge_data, row="Model", hue="Color", aspect=15, height=0.5, palette=self.model_colors)
                g.map_dataframe(kdeplot_with_alpha, x="Similarity")
                g.map(sns.kdeplot, "Similarity", bw_adjust=1, clip_on=False, color="black", lw=0.1)
                g.map(plt.axhline, lw=0.5, clip_on=False)
                for i, model in enumerate(self.models):
                    ax = g.axes.flat[i]
                    ax.text(-0.18, 0.3, model, fontweight='bold', fontsize=6, color=self.model_colors[model])
                    ax.set_xlim(-0.2, 1.1)
                g.axes.flat[0].text(-0.18, 0.3, 'Initial', fontweight='bold', fontsize=6, color='grey')
                g.fig.subplots_adjust(hspace=-0.4)
                g.set_titles("")
                g.set(yticks=[])
                # g.despine(bottom=True, left=True)
                plt.setp(g.axes.flat[-1].get_xticklabels(), fontsize=15, fontweight='bold')
                g.set_xlabels('Similarity', fontweight='bold', fontsize=15)
                g.set_ylabels('')
                if with_suptitle:
                    g.fig.suptitle('Convergence', ha='right', fontsize=20, fontweight='bold')

        else:
            if axs is None:
                fig, axs = plt.subplots(self.n_models, num_prompts, figsize=(w * num_prompts, h * self.n_models))

            for i, m in enumerate(self.models):
                axs[i, 0].annotate(m, xy=(-0.2, 0.5), xycoords="axes fraction", fontsize=plt.rcParams["font.size"], ha="center", va="center", rotation="vertical")
                axs[i, 0].set_ylabel("Final similarity")
                
                for j, (ax, p) in enumerate(zip(axs[i, :], self.prompts)):
                    if i == 0:
                        ax.set_title(self.prompt_names[j])
                    if j != 0:
                        ax.set_yticks([])

                    if i == len(self.models) - 1:
                        ax.set_xlabel("Initial similarity")
                    else:
                        ax.set_xticks([])

                    ax.set_ylim(-0.1, 1.1)
                    ax.set_xlim(-0.1, 1.1)
                    ax.plot([0, 1], [0, 1], ls="--", c="k")
                    x = np.array(initial[m][p])
                    y = np.array(final[m][p])

                    if markerfill_by_pos:
                        ax.scatter(x=x[y < x], y=y[y < x], color="none", edgecolor=self.model_colors[m])
                        ax.scatter(x=x[y >= x], y=y[y >= x], color=self.model_colors[m])
                        sns.regplot(x=x, y=y, line_kws={"color": self.model_colors[m]}, scatter=False, ax=ax)
                    else:
                        sns.regplot(x=x, y=y, line_kws={"color": self.model_colors[m]}, scatter_kws={"color": self.model_colors[m]}, ax=ax)

            # Create ridge plots for the bottom row
            ridge_data = []
            for p in self.prompts:
                for m in self.models:
                    ridge_data.append(pd.DataFrame({"Similarity": initial[m][p], "Type": "Initial", "Model": m, "Prompt": p}))
                    ridge_data.append(pd.DataFrame({"Similarity": final[m][p], "Type": "Final", "Model": m, "Prompt": p}))
            ridge_data = pd.concat(ridge_data)

            g = sns.FacetGrid(ridge_data, row="Prompt", hue="Model", aspect=15, height=0.5, palette=self.model_colors)
            g.map(sns.kdeplot, "Similarity", bw_adjust=1, clip_on=False, fill=True, alpha=1, linewidth=1.5)
            g.map(sns.kdeplot, "Similarity", bw_adjust=1, clip_on=False, color="w", lw=2)
            g.map(plt.axhline, y=0, lw=2, clip_on=False)

            for i, ax in enumerate(g.axes.flat):
                ax.text(-0.1, 0.02, self.prompt_names[i], fontweight='bold', fontsize=15, color=ax.lines[-1].get_color())
            
            g.fig.subplots_adjust(hspace=-0.3)
            g.set_titles("")
            g.set(yticks=[])
            g.despine(bottom=True, left=True)
            plt.setp(g.axes[-1].get_xticklabels(), fontsize=15, fontweight='bold')
            g.set_xlabels('Similarity', fontweight='bold', fontsize=15)
            if with_suptitle:
                g.fig.suptitle('Convergence', ha='right', fontsize=20, fontweight='bold')

        return fig, axs, g.fig
    

    def plot_attraction_landscape(self, measures, model, prompt, w=6, h=4, mmpp=True, legend_i=0, legend_pos=(0.7, 1), ylim=None, with_suptitle=False, fig=None, axs=None):
        arrow_start = []
        arrow_end = []


        for s_i, s in enumerate(self.stories):
            for seed in range(self.n_seeds):
                for g in range(self.n_generations - 1):
                    x_start = self.all_evolutions[measures[0]][model][prompt][s_i][seed][g]
                    y_start = self.all_evolutions[measures[1]][model][prompt][s_i][seed][g]
                    x_end = self.all_evolutions[measures[0]][model][prompt][s_i][seed][g + 1]
                    y_end = self.all_evolutions[measures[1]][model][prompt][s_i][seed][g + 1]
                    arrow_start.append([x_start, y_start])
                    arrow_end.append([x_end, y_end])

        arrow_start = np.array(arrow_start)
        arrow_end = np.array(arrow_end)
        
        # Define grid size
        n_bins = 50
        grid_size_x = (np.max(arrow_start[:, 0]) - np.min(arrow_start[:, 0])) / n_bins
        grid_size_y = (np.max(arrow_start[:, 1]) - np.min(arrow_start[:, 1])) / n_bins

        # Compute bins
        x_bins = np.arange(np.min(arrow_start[:, 0]), np.max(arrow_start[:, 0]) + grid_size_x, grid_size_x)
        y_bins = np.arange(np.min(arrow_start[:, 1]), np.max(arrow_start[:, 1]) + grid_size_y, grid_size_y)

        # Create grid for average arrows
        avg_arrows = np.zeros((len(x_bins) - 1, len(y_bins) - 1, 2))

        # Bin arrows and compute mean direction for each bin
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                bin_mask = (
                    (arrow_start[:, 0] >= x_bins[i]) & (arrow_start[:, 0] < x_bins[i + 1]) &
                    (arrow_start[:, 1] >= y_bins[j]) & (arrow_start[:, 1] < y_bins[j + 1])
                )
                if np.any(bin_mask):
                    avg_dx = np.mean(arrow_end[bin_mask, 0] - arrow_start[bin_mask, 0])
                    avg_dy = np.mean(arrow_end[bin_mask, 1] - arrow_start[bin_mask, 1])
                    avg_arrows[i, j, 0] = avg_dx
                    avg_arrows[i, j, 1] = avg_dy
        
        ## Normalize arrows
        avg_arrows /= np.max(np.abs(avg_arrows))

        # avg_arrows[:, :, 0] /= np.max(np.abs(avg_arrows[:, :, 0]))
        # avg_arrows[:, :, 1] /= np.max(np.abs(avg_arrows[:, :, 1]))

        



        if mmpp:  # multiple models per plot
            if axs is None:
                fig, axs = plt.subplots(1, len(self.prompts), figsize=(w * len(self.prompts), h))
            for i, (ax, p) in enumerate(zip(axs.ravel(), self.prompts)):
                ax.set_xlabel(measures[0].capitalize())
                ax.set_ylabel(measures[1].capitalize())
                ax.set_title(self.prompt_names[i])
                if ylim is not None:
                    ax.set_ylim(ylim)
                X, Y = np.meshgrid(x_bins[:-1] + grid_size_x / 2, y_bins[:-1] + grid_size_y / 2)
                ax.quiver(X, Y, avg_arrows[:, :, 0], avg_arrows[:, :, 1], color='black', alpha=0.8)
                if i == legend_i:
                    legend = ax.legend(handles=self._get_model_handles("s"), bbox_to_anchor=legend_pos, loc="upper right",
                                    fontsize=self.legend_font_size, title=f"Model")
        else:
            if axs is None:
                fig, axs = plt.subplots(self.n_models, len(self.prompts), figsize=(w * len(self.prompts), h * self.n_models))
            for i, (ax, m) in enumerate(zip(axs, self.models)):
                axs[i, 0].annotate(m, xy=(-0.2, 0.5),
                                xycoords="axes fraction", fontsize=plt.rcParams["font.size"], ha="center", va="center",
                                rotation="vertical")
                axs[i, 0].set_ylabel(measures[1].capitalize())
                for j, p in enumerate(self.prompts):
                    if i == 0:
                        axs[i, j].set_title(self.prompt_names[j])
                    if j != 0:
                        axs[i, j].set_yticks([])
                    if i == len(self.models) - 1:
                        axs[i, j].set_xlabel(measures[0].capitalize())
                    else:
                        axs[i, j].set_xticks([])
                    if ylim is not None:
                        axs[i, j].set_ylim(ylim)
                    X, Y = np.meshgrid(x_bins[:-1] + grid_size_x / 2, y_bins[:-1] + grid_size_y / 2)
                    axs[i, j].quiver(X, Y, avg_arrows[:, :, 0], avg_arrows[:, :, 1], color='black', alpha=0.8)
        fig.tight_layout()
        if with_suptitle:
            fig.suptitle(f"Trajectories of {measures[0].capitalize()} and {measures[1].capitalize()}", y=1.01)

        return fig, axs

                                
                                
        


        
    
    def plot_initial_final_distribution_(self, measure, w=6, h=4, mmpp=True, legend_i=0, legend_pos=(0.7,1), ylim=None, with_suptitle=False, fig=None, axs=None):
        initial = self.all_initial_cumuls[measure]
        final = self.all_final_cumuls[measure]

        fig, axs = plt.subplots(1, len(self.prompts), figsize=(w*len(self.prompts), h))
        


        axs[0].set_ylabel("Density")

        if mmpp:  # multiple models per plot
            labels = []
            for i, p in enumerate(self.prompts):
                axs[i].set_title(self.prompt_names[i], pad=25)
                axs[i].set_xlabel(measure.capitalize())
                if i == legend_i:
                    axs[i].legend(handles=self._get_model_handles("o"), loc="best", fontsize=self.legend_font_size, title="Model")
                ax2 = axs[i].twinx()
                x,y = sns.kdeplot(data=np.array(initial[self.models[0]][p]).flatten(), label=f"Initial {p}", color='lightgray', fill=False, bw_adjust=0.75, clip=ylim, ax=ax2).get_lines()[0].get_data()
                y_max = y.max()
                sns.kdeplot(data=np.array(initial[self.models[0]][p]).flatten(), label=f"Initial {p}", color='lightgray', fill=True, bw_adjust=0.75, clip=ylim, ax=ax2, alpha=0.3)
                ax2.set_ylim(0, y_max)
                ax2.set_yticks([])
                ax2.set_ylabel("")
                ax2.spines["right"].set_visible(False)

                for m in self.models:
                    ax2 = axs[i].twinx()
                    x,y = sns.kdeplot(data=np.array(final[m][p]).flatten(), label=f"Final {p}", color=self.model_colors[m], fill=False, bw_adjust=0.75, clip=ylim, ax=ax2).get_lines()[0].get_data()
                    y_max = y.max()
                    sns.kdeplot(data=np.array(final[m][p]).flatten(), label=f"Final {p}", color=self.model_colors[m], fill=False, bw_adjust=0.75, clip=ylim, ax=ax2)

                    ax2.set_ylim(0, y_max)
                    ax2.set_yticks([])
                    ax2.set_ylabel("")
                    ax2.spines["right"].set_visible(False)

                    xlim = ax2.get_xlim()

                    if self.all_attr_positions_10[measure][m][p] > xlim[1]:
                        plt.text(xlim[1], y_max, "-> X", color=self.model_colors[m], fontsize= 20 + 20 * self.all_attr_strengths_10[measure][m][p], alpha=0.3)
                    else:
                    
                        plt.text(self.all_attr_positions_10[measure][m][p], y_max, "X", color=self.model_colors[m], fontsize= 20 + 20 * self.all_attr_strengths_10[measure][m][p], alpha=0.3)

        fig.tight_layout()
        if with_suptitle:
            fig.suptitle(f"Initial and Final {measure.capitalize()} Distribution", y=1.01)

        return fig, axs
    


    def plot_metric_distributions(self, measure, measure_tag, prompt, model, w=6, h=4, mmpp=True, legend_i=0, legend_pos=(0.7,1), ylim=None, with_suptitle=False, fig=None, axs=None):
        if measure != 'similarity':
            
            initial = self.all_initial_cumuls[measure]
            final = self.all_final_cumuls[measure]

        if measure == "difficulty":
            clip = (0, 50)
            linewidth = 1.5
        elif measure == "length":
            clip = (0, 3500)
            linewidth = 3
        elif measure == "toxicity":
            clip = (0, 1)
            linewidth = 0.1 
        elif measure == "positivity":
            clip = (-1, 1)
            linewidth = 3
        elif measure == "similarity":
            clip = (-1, 1)
            linewidth = 3

        if mmpp:
            sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

            ridge_data = []
            if measure == 'similarity':
                all_evolutions = self.get_all_similarities(self.all_data, model, prompt)
            else:
                all_evolutions = np.array([self.all_data["evolution"][prompt][s][f"Results/{model}/{prompt}/{s}"][measure_tag] for s in self.stories])
            
            for g in range(self.n_generations):
                if measure == 'similarity':

                    ridge_data.append(pd.DataFrame({
                        "Value": np.array(all_evolutions[g]).flatten(),
                        "Generation": g,
                        "Type": "Final",
                        "Model": model,
                        "Prompt": prompt,
                        "Color": "initial"
                    }))
                else:
                    ridge_data.append(pd.DataFrame({
                        "Value": all_evolutions[:,:, g].flatten(),
                        "Generation": g,
                        "Type": "Final",
                        "Model": model,
                        "Prompt": prompt,
                        "Color": "initial"
                    }))
            if measure == 'similarity':

                ridge_data.append(pd.DataFrame({
                    "Value": np.array(self.initial_sim[model][prompt]).flatten(),
                    "Type": "Initial",
                    "Model": model,
                    "Prompt": prompt,
                    "Color": "initial"
                }))
            else:
                ridge_data.append(pd.DataFrame({
                    "Value": np.array(initial[self.models[0]][prompt]).flatten(),
                    "Type": "Initial",
                    "Model": model,
                    "Prompt": prompt,
                    "Color": "initial"
                }))
            ridge_data = pd.concat(ridge_data)

            g = sns.FacetGrid(ridge_data, row="Generation", hue="Model", aspect=15, height=0.5, palette=self.model_colors, sharex=False)
            
            def normalize_kde(ax, data, i):
                ax2 = ax.twinx()
                # lines = sns.kdeplot(data=data, x="Value", hue="Model", bw_adjust=0.75, clip_on=True, fill=False, linewidth=1.5, legend=False, ax=ax2, palette=self.model_colors, clip=clip).get_lines()
                # if len(lines) == 0:
                #     print("No data")
                #     print('generation', i)  
                #     print(np.var(data["Value"]))
                #     print(prompt)
                #     print(model)
                #     print(measure)
                #     print(data)
                    
                x,y = sns.kdeplot(data=data, x="Value", hue="Model", bw_adjust=0.75, clip_on=True, fill=False, linewidth=1.5, legend=False, ax=ax2, palette=self.model_colors, clip=clip).get_lines()[0].get_data()
                max_density = y.max()
                sns.kdeplot(data=data, x="Value", hue="Model", bw_adjust=0.75, clip_on=True, fill=True, linewidth=1, legend=False, ax=ax2, palette=self.model_colors, alpha=1, clip=clip)
                sns.kdeplot(data=data, x="Value", color = 'black', bw_adjust=0.75, clip_on=True, fill=False, linewidth=linewidth, legend=False, ax=ax2, clip=clip)
                plt.plot([data["Value"].iloc[0], data["Value"].iloc[0]], [0, 1], color='black', linewidth=linewidth)
                if measure == "toxicity":
                    ax2.set_xlim([-0.1, 1])
                elif measure == "positivity":
                    ax2.set_xlim([-1.1, 1.1])
                else:
                    ax2.set_xlim(clip)
                if i == self.n_generations - 1:
                    ax2.set_xlabel(measure.capitalize())
                    ax2.set_xticks(np.linspace(clip[0], clip[1], 3))
                    ax.set_xticks(np.linspace(clip[0], clip[1], 3))
                else:
                    ax2.set_xticks([])

                xtick_keep = ax2.get_xticks()  
                ax2.set_ylim(0, max_density)
                ax2.set_yticks([])
                ax2.set_ylabel("")
                ax2.set_axis_off()
                ax.set_axis_off()
                return xtick_keep
            
            for i, ax_row in enumerate(g.axes.flat):
                generation_data = ridge_data[ridge_data["Generation"] == i]
                xtick_keep = normalize_kde(ax_row, generation_data, i)
            
            g.fig.subplots_adjust(hspace=-0.8)

            g.axes.flat[-1].set_xlabel(measure.capitalize())
            g.set_titles("")
            g.set(yticks=[])
            g.axes.flat[0].set_ylabel("")
            for i, ax_row in enumerate(g.axes.flat):
                if i == self.n_generations - 1:
                    for tick in xtick_keep:
                        ax_row.text(tick, -0.25, f'{tick:.1f}', ha='center', va='center', fontsize=30, fontweight='bold')
                    # mid_pos = (xtick_keep[0] + xtick_keep[-1]) / 2
                    # ax_row.text(mid_pos, -0.4, measure.capitalize(), ha='center', va='center', fontsize=35, fontweight='bold')
            if with_suptitle:
                g.fig.suptitle('Convergence', ha='right', fontsize=20, fontweight='bold')

        return g


    def plot_cumulativeness_ivsf(self, measure, w=6, h=4, mmpp=True, legend_i=0, legend_pos=(0.7,1), ylim=None, xlim=None, with_suptitle=False, alpha=0.4, fig=None, axs=None, after10 = False):

        initial = self.all_initial_cumuls[measure]
        final = self.all_final_cumuls[measure]
        if after10:
            final = self.all_after_10_cumuls[measure]

        val_min = np.min([score[m][p] for m in self.models for p in self.prompts for score in [initial, final]])
        val_min_initial = np.min([initial[m][p] for m in self.models for p in self.prompts])
        val_min_final = np.min([final[m][p] for m in self.models for p in self.prompts])
        val_max = np.max([score[m][p] for m in self.models for p in self.prompts for score in [initial, final]])
        val_max_initial = np.max([initial[m][p] for m in self.models for p in self.prompts])
        val_max_final = np.max([final[m][p] for m in self.models for p in self.prompts])

        if mmpp: # multiple models per plot
            if axs is None:
                fig, axs = plt.subplots(1, len(self.prompts), figsize=(w*(len(self.prompts)), h)) 

            axs[0].set_ylabel(f"Final {measure}")

            for i, (ax, p) in enumerate(zip(axs.ravel(), self.prompts)):
                ax.set_xlabel(f"Initial {measure}")
                if ylim is None:
                    ax.set_ylim((val_min_final*0.99, val_max_final*1.1))
                else:
                    ax.set_ylim(ylim)
                if xlim is None:
                    ax.set_xlim((val_min_initial*0.99, val_max_initial*1.1))
                else:
                    ax.set_xlim(xlim)
                if i != 0:
                    ax.set_yticks([])
                ax.set_title(self.prompt_names[i])
                ax.plot([val_min, val_max], [val_min, val_max], ls="--", c="k")
                for m in self.models:
                    sns.regplot(x=np.array(initial[m][p]).flatten(), y=np.array(final[m][p]).flatten(), line_kws={"color": self.model_colors[m]}, scatter_kws={"color": self.model_colors[m], "alpha": alpha}, ax=ax)
                if i == legend_i:
                    legend = ax.legend(handles=self._get_model_handles("o"), bbox_to_anchor=legend_pos, loc="upper center", fontsize=self.legend_font_size, title=f"Model")



        else:
            if axs is None:
                fig, axs = plt.subplots(self.n_models, len(self.prompts), figsize=(w*(len(self.prompts)), h*(self.n_models)))
            for i, (ax, m) in enumerate(zip(axs, self.models)):
                axs[i, 0].annotate(m, xy=(-0.2, 0.5), 
                                xycoords="axes fraction", fontsize=plt.rcParams["font.size"], ha="center", va="center", rotation="vertical")
                axs[i, 0].set_ylabel(f"Final {measure}")
                for j, p in enumerate(self.prompts):
                    if i == 0:
                        axs[i, j].set_title(self.prompt_names[j])
                    if j != 0:
                        axs[i, j].set_yticks([])
                    if i == len(self.models)-1:
                        axs[i, j].set_xlabel(f"Initial {measure}")
                    else:
                        axs[i, j].set_xticks([])
                    axs[i, j].plot([val_min, val_max], [val_min, val_max], ls="--", c="k")
                    sns.regplot(x=initial[m][p], y=final[m][p], 
                                line_kws={"color": self.model_colors[m]}, scatter_kws={"color": self.model_colors[m], "alpha": alpha}, ax=axs[i, j])
                    if ylim is None:
                        axs[i, j].set_ylim((val_min_final*0.99, val_max_final*1.1))
                    else:
                        axs[i, j].set_ylim(ylim)
                    if xlim is None:
                        axs[i, j].set_ylim((val_min_initial*0.99, val_max_initial*1.1))
                    else:
                        axs[i, j].set_xlim(xlim)

        fig.tight_layout()
        if with_suptitle:
            fig.suptitle(measure.capitalize(), y=1.01)

        return fig, axs


    # def plot_convergence(self, w=6, h=4, mmpp=True, legend_i=0, with_suptitle=False, markerfill_by_pos=True, alpha=0.2, fig=None, axs=None, first_10 = False):
        
    #     if first_10:
    #         final = self.all_after_10_cumuls
    #     else:
    #         final = self.final_sim
    #     initial = self.initial_sim
        


    #     if mmpp: # multiple models per plot
    #         if axs is None:
    #             fig, axs = plt.subplots(1, len(self.prompts), figsize=(w*(len(self.prompts)), h)) 

    #         axs[0].set_ylabel(f"Final similarity")

    #         for i, (ax, p) in enumerate(zip(axs.ravel(), self.prompts)):
    #             ax.set_xlabel(f"Initial similarity")
    #             ax.set_ylim(-0.1,1.1)
    #             ax.set_xlim(-0.1,1.1)
    #             if i != 0:
    #                 ax.set_yticks([])
    #             ax.set_title(self.prompt_names[i])
    #             ax.plot([0, 1], [0, 1], ls="--", c="k")
    #             for m in self.models:
    #                 x = initial[m][p]
    #                 y = final[m][p]
    #                 if markerfill_by_pos:
    #                     ax.scatter(x=x[y<x], y=y[y<x], color="none", edgecolor=self.model_colors[m], alpha=alpha)
    #                     ax.scatter(x=x[y>=x], y=y[y>=x], color="none", edgecolor=self.model_colors[m], alpha=alpha)
    #                     sns.regplot(x=x, y=y, line_kws={"color": self.model_colors[m]}, scatter=False, ax=ax)
    #                 else:
    #                     sns.regplot(x=x, y=y, 
    #                             line_kws={"color": self.model_colors[m]}, scatter_kws={"color": self.model_colors[m], "alpha": alpha}, ax=ax)
    #             if i == legend_i:
    #                 legend = ax.legend(handles=self._get_model_handles("o"), loc="best", fontsize=self.legend_font_size, title=f"Model")

    #     else:
    #         if axs is None:
    #             fig, axs = plt.subplots(self.n_models, len(self.prompts), figsize=(w*(len(self.prompts)), h*(self.n_models)))
    #         for i, (ax, m) in enumerate(zip(axs, self.models)):
    #             axs[i, 0].annotate(m, xy=(-0.2, 0.5), 
    #                             xycoords="axes fraction", fontsize=plt.rcParams["font.size"], ha="center", va="center", rotation="vertical")
    #             axs[i, 0].set_ylabel(f"Final similarity")
    #             for j, p in enumerate(self.prompts):
                    
    #                 if i == 0:
    #                     axs[i, j].set_title(self.prompt_names[j])
    #                 if j != 0:
    #                     axs[i, j].set_yticks([])
    #                 if i == len(self.models)-1:
    #                     axs[i, j].set_xlabel(f"Initial similarity")
    #                 else:
    #                     axs[i, j].set_xticks([])
    #                 axs[i, j].set_ylim((-0.1,1.1))
    #                 axs[i, j].set_xlim((-0.1,1.1))
    #                 axs[i, j].plot([0, 1], [0, 1], ls="--", c="k")
    #                 x = np.array(initial[m][p])
    #                 y = np.array(final[m][p])
    #                 if markerfill_by_pos:
    #                     axs[i, j].scatter(x=x[y<x], y=y[y<x], color="none", edgecolor=self.model_colors[m])
    #                     axs[i, j].scatter(x=x[y>=x], y=y[y>=x], color=self.model_colors[m])
    #                     sns.regplot(x=x, y=y, line_kws={"color": self.model_colors[m]}, scatter=False, ax=axs[i, j])
    #                 else:
    #                     sns.regplot(x=x, y=y, line_kws={"color": self.model_colors[m]}, scatter_kws={"color": self.model_colors[m]}, ax=axs[i, j])
            
    #     fig.tight_layout()
    #     if with_suptitle:
    #         fig.suptitle("Convergence", y=1.01)

    #     return fig, axs
    
    def plot_all_evolutions(self, measure, w=6, h=4, mmpp=False, legend_i=0, legend_j=0, with_suptitle=True, with_grid=True, fig=None, axs=None):
        x = np.arange(0, self.n_generations+1)
        xticks = np.arange(0, self.n_generations+1, 10)
        xticks[1:] -= 1

        N = self.n_stories * self.n_seeds

        if mmpp:
            if axs is None:
                fig, axs = plt.subplots(1, len(self.prompts), figsize=(w*(len(self.prompts)), h))
            axs.set_ylabel(f"{measure.capitalize()} score")
            for i, (ax, p) in enumerate(zip(axs.ravel(), self.prompts)):
                ax.set_xlabel("Generation")
                ax.set_title(self.prompt_names[i])
                if with_grid:
                    ax.grid(True, axis="y")
                if i != 0:
                    ax.tick_params(left=False, labelleft=False)
                if i == len(self.prompts)-1:
                    ax.set_xlabel(f"Generation")
                    ax.set_xticks(xticks, xticks)
                else:
                    ax.set_xticks([])

                for m in self.models:
                    y = self.all_evolutions[m][p]
                    y_mean = np.squeeze(np.mean(y, axis=(0,1)))
                    y_sem = np.squeeze(np.std(y, axis=(0,1))/np.sqrt(N))
                    ax.plot(x, y_mean, color=self.model_colors[m])
                    ax.fill_between(x, y_mean-y_sem, y_mean+y_sem, alpha=0.3, color=self.model_colors[m])
                if i == legend_i:
                    legend = ax.legend(handles=self._get_model_handles("_"), loc="best", fontsize=self.legend_font_size, title=f"Model")
                
        fig.tight_layout()
        if with_suptitle:
            fig.suptitle("Convergence", y=1.01)

        return fig, axs
                


    
    

    def plot_evolution(self, examples=True, w=6, h=4, legend_i=0, legend_j=0, with_suptitle=True, with_grid=True, fig=None, axs=None, single_seeds=False, specific_story=None, all_figures = False):

        if axs is None:
            fig, axs = plt.subplots(len(self.measures), len(self.prompts), figsize=(w*(len(self.prompts)), h*(len(self.measures))))
        
        # if all:
        #     fig, axs = plt.subplots(len(self.measures), len(self.prompts), figsize=(w*(len(self.prompts)), h*(len(self.measures)))


        x = np.arange(0, self.n_generations+1)
        xticks = np.arange(0, self.n_generations+1, 10)
        xticks[1:] -= 1

        N = self.n_stories * self.n_seeds

        for i, (ax, measure) in enumerate(zip(axs, self.measures)):

            evolution = self.all_evolutions[measure]
            if specific_story is not None:
                s = self.stories.index(specific_story)
            
            else:
                s = self.example_indices[measure]
            c = self.collpase_indices

            if examples:
                val_max = np.max([np.squeeze(np.mean(evolution[m][p][s], axis=0))+np.squeeze(np.std(evolution[m][p][s], axis=0)/np.sqrt(self.n_seeds)) for m in self.models for p in self.prompts])
                val_min = np.min([np.squeeze(np.mean(evolution[m][p][s], axis=0))-np.squeeze(np.std(evolution[m][p][s], axis=0)/np.sqrt(self.n_seeds)) for m in self.models for p in self.prompts])
            # elif all:
            #     for s in range(self.n_stories):
            #         val_max = np.max([np.squeeze(np.mean(evolution[m][p][s], axis=0))+np.squeeze(np.std(evolution[m][p][s], axis=0)/np.sqrt(self.n_seeds)) for m in self.models for p in self.prompts])
            #         val_min = np.min([np.squeeze(np.mean(evolution[m][p][s], axis=0))-np.squeeze(np.std(evolution[m][p][s], axis=0)/np.sqrt(self.n_seeds)) for m in self.models for p in self.prompts])
            
            else:
                val_max = np.max([np.squeeze(np.mean(evolution[m][p], axis=(0,1)))+np.squeeze(np.std(evolution[m][p], axis=(0,1))/np.sqrt(N)) for m in self.models for p in self.prompts])
                val_min = np.min([np.squeeze(np.mean(evolution[m][p], axis=(0,1)))-np.squeeze(np.std(evolution[m][p], axis=(0,1))/np.sqrt(N)) for m in self.models for p in self.prompts])

            # axs[i, 0].annotate(measure, xy=(-0.2, 0.5), 
                            # xycoords="axes fraction", fontsize=plt.rcParams["font.size"], ha="center", va="center", rotation="vertical")
            axs[i, 0].set_ylabel(f"{measure.capitalize()} score")

            for j, p in enumerate(self.prompts):
                
                if i == 0:
                    axs[i, j].set_title(self.prompt_names[j])
                if with_grid:
                    axs[i, j].grid(True, axis="y")
                if j != 0:
                    axs[i, j].tick_params(left=False, labelleft=False)
                if i == len(self.measures)-1:
                    axs[i, j].set_xlabel(f"Generation")
                    axs[i, j].set_xticks(xticks, xticks)
                else:
                    axs[i, j].set_xticks([])

                axs[i, j].set_ylim([val_min, val_max])


                for m in self.models:
                    if single_seeds: ## Only plot collpasing seeds for "Mistra-7B-Instruct-v0.2" for length
                            if m == "Mistral-7B-Instruct-v0.2" and measure == "length":
                                for seed in range(self.n_seeds):
                                    y = evolution[m][p][c[0]][:][seed]
                                    y_mean = np.squeeze(np.mean(y, axis=0))
                                    axs[0, 0].plot(x, y_mean, color=self.model_colors[m])
                                return
                                
                    else:
                                
                        if examples:  # pick a single story, average over seeds


                                y = evolution[m][p][s]
                                # if measure == "length":
                                #     y = y[:, 1:] 
                                y_mean = np.squeeze(np.mean(y, axis=0))
                                y_sem = np.squeeze(np.std(y, axis=0)/np.sqrt(self.n_seeds))
                        else:
                            y = evolution[m][p] # all stories, all seeds
                            y_mean = np.squeeze(np.mean(y, axis=(0,1)))
                            y_sem = np.squeeze(np.std(y, axis=(0,1))/np.sqrt(N))

                        axs[i, j].plot(x, y_mean, color=self.model_colors[m])
                        axs[i, j].fill_between(x, y_mean-y_sem, y_mean+y_sem, alpha=0.3, color=self.model_colors[m])


        legend = axs[legend_i, legend_j].legend(handles=self._get_model_handles("_", 15), loc="best", fontsize=self.legend_font_size, title=f"Model")

        if with_suptitle:
            fig.suptitle("Evolution of metrics over generations")
        fig.tight_layout()

        return fig, axs
    
    def get_attr_var(self, after10 = False):
                
        values_strength = self.all_attr_strengths
        
        values_position = self.all_attr_positions

        df = []

        for m in self.models:
            for p in self.prompts:

        
                for measure in self.measures:
                    df.append({"model": m, "prompt": p, "measure": measure, "position": values_position[measure][m][p], "strength": values_strength[measure][m][p]})



        return pd.DataFrame(df)


    def plot_attr_var(self, var, n_rows=1, w=6, h=4, ylim=(0, 1), legend_i=0, 
                        legend_pos=(0.58, 0.7), models_first=False, with_suptitle=True, 
                        fig=None, axs=None, after10 = False, ylabel = 'first', labelpad = 0):
        n_cols = len(self.measures)//n_rows

        if var == "strength":
            if after10:
                values = self.all_attr_strengths_10
            else:
                values = self.all_attr_strengths
        elif var == "position":
            if after10:
                values = self.all_attr_positions_10
            else:
                values = self.all_attr_positions

        else:
            print(f"Invalid variable {var}. Please choose between 'strength' and 'position'.")
            return 

        data = []

        for m in self.models:
            for p in self.prompts:
                for measure in self.measures:
                    data.append({"model": m, "prompt": p, "measure": measure, "value": values[measure][m][p]})

        df = pd.DataFrame(data)
                
        
        

        
        
        if axs is None:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(w*n_cols, h*n_rows))
                    
        for i, (ax, measure) in enumerate(zip(axs.ravel(), self.measures)):
            
            sns.barplot(x='prompt', y='value', hue='model', data=df[df["measure"] == measure], ax=ax, palette=[self.model_colors[m] for m in self.models])
            ## Set xticks
            xticks = [self.prompt_names[i] for i in range(len(self.prompts))]
            ax.set_xticks(range(len(xticks)), xticks, fontsize=12)  
            if ylabel == 'first' and i % n_cols == 0:

                ax.set_ylabel(f"Attractor {var.capitalize()}", fontsize=24, fontweight='bold', labelpad = labelpad)
            else:
                ax.set_ylabel("")

            ax.set_xlabel("Task")

            ## Increase yticks fontsize
            ax.tick_params(axis='y', labelsize=16)

            ## offset title
            ax.title.set_position([.5, 1.2])


            # ax.set_title('Attractor ' + var.capitalize(), fontsize=12, fontweight='bold')



            # if models_first: # compare across prompts
            #     xticks = [p.capitalize() for m in self.models for p in self.prompts]
            #     y = [values[measure][m][p] for m in self.models for p in self.prompts]
            #     ax.bar(range(len(xticks)), y, color=[self.model_colors[m] for m in self.models for p in self.prompts])
            #     ax.set_xticks(range(len(xticks)), xticks, rotation=45, ha="right")
            # else: # prompts first, compare across models
            #     
            #     xlabels = [xtick for xi, xtick in enumerate(xticks) if xi in np.arange(self.n_models//2, self.n_models*len(self.prompts), self.n_models)]
            #     y = [values[measure][m][p] for p in self.prompts for m in self.models ]
            #     ax.bar(range(len(xticks)), y, color=[self.model_colors[m] for p in self.prompts for m in self.models ])
            #     ax.set_xticks(np.arange(self.n_models//2, self.n_models*len(self.prompts), self.n_models), xlabels)
            
            
            # ax.set_ylabel(f"Attractor {var}")
            if with_suptitle:
                ax.set_title(measure.capitalize(), fontweight='bold', fontsize=24, y=1.2)
            ax.set_ylim(ylim)
            if i == legend_i:
                legend = ax.legend(handles=self._get_model_handles("s"), bbox_to_anchor=legend_pos,
                                    loc="lower right", fontsize=self.legend_font_size, title=f"Model")
            else:
                ax.get_legend().remove()


        fig.tight_layout()
        return fig, axs
    


    def get_data_punctuatedness(self, measure, directional = False):
            def difference_to_diagonal(x, y):
                return np.abs(y - x / 50)
            data = []

            for p_i, p in enumerate(self.prompts):
                for m in self.models:
                    if directional:
                        values = np.array(self.directional_change_per_generation[measure][m][p]).squeeze()
                    else:
                        values = np.array(self.change_per_generation[measure][m][p]).squeeze()
                    summed_changes = np.cumsum(values, axis=1)
                    shape = summed_changes.shape

                    for story in range(shape[0]):
                        for seed in range(shape[2]):
                            differences = np.array([difference_to_diagonal(np.arange(50), summed_changes[story,:,seed])]) 
                            punct = np.trapz(differences, np.arange(50))[0] / 25
                            data.append({'model': m, 'prompt': p, 'story': story, 'seed': seed, 'punct': punct, 'area': np.trapz(summed_changes[story,:,seed] , np.arange(50)) / 50})
                    
                    # Compute the differences
                    differences = np.array([[difference_to_diagonal(np.arange(50), summed_changes[i,:,j]) for i in range(shape[0])] for j in range(shape[2])])



                    punct = np.array([[np.trapz(differences[j,i,:], np.arange(50)) for i in range(shape[0])] for j in range(shape[2])])

                    
                    

                    # Normalize 


                    area = np.array([[np.trapz(summed_changes[i,:,j], np.arange(50)) for i in range(shape[0])] for j in range(shape[2])])

                    punct = punct / 25


                    ## Normalize with max area = 50

                    area = area / 50


            
            df = pd.DataFrame(data)
            return df
    

    def plot_punctuatedness(self, measure,which = 'area',  w=6, h=4, mmpp=True, legend_i=0, legend_pos=(0.7,1), ylim=None, with_suptitle=False, fig=None, axs=None, directional = False):
    
        
        
        df = self. get_data_punctuatedness(measure, directional=directional)
        
        


        


        if fig is None:
            fig, axs = plt.subplots(1, 1, figsize=(2 * w, h))
        
        
        
        if which == 'punctuatedness':
            ## barplot with error bars
            sns.barplot(x='prompt', y='punct', hue='model', data=df,  ax=axs, color=[self.model_colors[m] for m in self.models])
            
            axs.set_ylabel('Absolute area to diagonal', fontsize=plt.rcParams["font.size"])
            axs.set_xlabel('Prompt', fontsize=12)
            axs.set_title('Absolute area to diagonal', fontsize=plt.rcParams["font.size"], fontweight='bold')
            axs.legend(loc='upper right', bbox_to_anchor=legend_pos, ncol=1, fontsize=plt.rcParams["font.size"])
            axs.set_ylim(0, 1)
        
        elif which == 'area':

            ## area barplot
            sns.barplot(x='prompt', y='area', hue='model', data=df,  ax=axs, color=[self.model_colors[m] for m in self.models])
            axs.set_ylabel('Area under Curve', fontsize=12)
            axs.set_xlabel('Prompt', fontsize=12)
            axs.set_title('Area under Curve', fontsize=12, fontweight='bold')
            axs.legend(loc='upper right', bbox_to_anchor=legend_pos, ncol=1, fontsize=plt.rcParams["font.size"])
            axs.set_ylim(0, 1)





            
        return fig, axs

    def plot_change_distribution(self, measure, w=6, h=4, mmpp=True, legend_i=0, legend_pos=(0.7,1), ylim=None, with_suptitle=False, fig=None, axs=None, directional = False):



        def distance_to_scaled_diagonal(x, y, n_generations = 49):
            return np.abs(y - x / n_generations) / np.sqrt(1 + (1 / n_generations) ** 2)
        


        if axs is None:
        
            fig, axs = plt.subplots(1, len(self.prompts), figsize=(w*len(self.prompts), h))
        

        for p_i, p in enumerate(self.prompts):
            data = []
            punctuatedness = []
            data = []
            for m in self.models:
                if directional:
                    values = np.array(self.directional_change_per_generation[measure][m][p]).squeeze()
                else:
                    values = np.array(self.change_per_generation[measure][m][p]).squeeze()
                summed_values = [0]
                summ = 0
                
                data.append({'model': m, 'generation': 0, 'value_sum': 0})

                for gen in range(values.shape[1]):
                    
                    summ += np.mean(values[:,gen,:])
                    summed_values.append(summ)
                    data.append({'model': m, 'generation': gen + 1, 'value_sum': summ})
                    

            df = pd.DataFrame(data)
        
                

            
            sns.lineplot(x='generation', y='value_sum', hue='model', data=df, palette="tab10", linewidth=2.5, ax=axs[p_i])

            

            axs[p_i].plot([0, 50], [0, 1], 'k--', linewidth=2.5)

            if p_i == 0:
                axs[p_i].set_ylabel(f'Cumulative {measure} Change', fontsize=plt.rcParams["font.size"], fontweight='bold')
            else:
                axs[p_i].set_ylabel(None)
            
            axs[p_i].set_xlabel('Generation', fontsize=plt.rcParams["font.size"])

            axs[p_i].set_title(p.capitalize(), fontsize=plt.rcParams["font.size"], fontweight='bold')

            # Adding labels and title
        plt.xlabel('Generation')
        plt.ylim(0, 50)
        plt.plot([25, 25], [0, 50], 'k--', linewidth=2.5)

        return fig, axs
    

        
        
    def get_data_initial_vs_final(self):
        data = []
        maxim = {}
        minin = {}
        for measure, key in zip(self.measures, ["all_seeds_toxicity", "all_seeds_positivity", "all_seeds_difficulty", "all_seeds_length"]):
            max_i = 0
            min_i = 0
            for m in self.models:
                for p in self.prompts:
                    for s in self.stories:
                        for seed in range(self.n_seeds):

                            all_evolutions = np.array(self.all_data["evolution"][p][s][f"Results/{m}/{p}/{s}"][key])
                            try: 
                                difference = np.abs((all_evolutions[seed,1][0] - all_evolutions[seed,-1][0]))  ## For some reason the length measure is sometimes a list of lists ? 
                            except: 
                                difference = np.abs(all_evolutions[seed,1] - all_evolutions[seed,-1]) 
                            if  np.max(all_evolutions[seed,:]) > max_i:
                                max_i = np.max(all_evolutions[seed,:])
                            if np.min(all_evolutions[seed,:]) < min_i:
                                min_i = np.min(all_evolutions[seed,:])
                            data.append({"model": m, "prompt": p, "measure": measure, "story": s, "seed": seed, "difference": difference, "initial": all_evolutions[seed,1], "final": all_evolutions[seed,-1]}) 
            maxim[measure] = max_i
            minin[measure] = min_i
        for entry in data:
            if maxim[entry["measure"]] - minin[entry["measure"]] == 0:
                entry["normalized_difference"] = 0
            entry["normalized_difference"] = (entry["difference"]) / (maxim[entry["measure"]] - minin[entry["measure"]])

        df = pd.DataFrame(data)
        return data


    def plot_norm_difference(self):
        df = self.get_data_initial_vs_final()
        df = pd.DataFrame(df)
        fig, axes = plt.subplots(1, len(self.measures), figsize=(len(self.measures)*7, 4), sharey=True)
        for i, measure in enumerate(self.measures):
            sns.barplot(x='prompt', y='normalized_difference', hue='model', data=df[df['measure'] == measure], errorbar='se', ax=axes[i], palette=[self.model_colors[m] for m in self.models])
            axes[i].set_ylabel(f'Normalized difference \nbetween first and last generations')
            xticks = ['Rephrase', 'Inspiration', 'Continue']
            axes[i].set_xlabel('Task')
            axes[i].set_xticks(range(3))
            axes[i].set_xticklabels(xticks)
            axes[i].set_ylim(0,0.4)
            axes[i].set_title(measure.capitalize(), fontweight='bold')
            if i != 0:
                axes[i].legend().remove()
            else:
                axes[i].legend(loc='upper right', bbox_to_anchor=(1.3, 1), ncol=1, fontsize=plt.rcParams["font.size"], title='Model')
        return axes

        
        

    


