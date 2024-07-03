import json
import pickle
import re

from detoxify import Detoxify
from scipy import stats
import numpy as np
import ssl

from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sentence_transformers import SentenceTransformer, util
from tqdm import trange
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")



try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def get_stories(folder):

    ## Iterate over all subfolders and get the stories
    
    all_stories = []


    for subfolder in os.listdir(folder):
        if os.path.isdir(folder + '/' + subfolder):
            json_files = [file for file in os.listdir(folder + '/' + subfolder) if file.endswith('.json')]
            for json_file in json_files:
                with open(folder + '/' + subfolder + '/' + json_file, 'r') as file:
                    data = json.load(file)
                    stories = data['stories']

                    all_stories.append(stories)

    return all_stories
            
    



def get_initial_story(folder):
    for subfolder in os.listdir(folder):
        if os.path.isdir(folder + '/' + subfolder):
            json_files = [file for file in os.listdir(folder + '/' + subfolder) if file.endswith('.json')]
            for json_file in json_files:
                with open(folder + '/' + subfolder + '/' + json_file, 'r') as file:
                    data = json.load(file)
                    initial_story = data["initial_story"]
                    return initial_story
    

    

def get_plotting_infos(stories):  
    n_gen, n_agents = len(stories), len(stories[0])
    x_ticks_space = n_gen // 10 if n_gen >= 20 else 1
    return n_gen, n_agents, x_ticks_space

def preprocess_single_seed(stories, remove_hashtags = True):
    flat_stories = [stories[i][j] for i in range(len(stories)) for j in range(len(stories[0]))]
    if remove_hashtags:
        flat_stories = [re.sub(r'#\w+', '', story) for story in flat_stories]

    return flat_stories

def preprocess_stories(all_seeds_stories):
    all_seeds_flat_stories = []
    all_seeds_keywords = []
    all_seeds_stem_words = []
    for stories in all_seeds_stories:
      
        flat_stories= preprocess_single_seed(stories)
        all_seeds_flat_stories.append(flat_stories)

    
    return all_seeds_flat_stories

def get_similarity_matrix_single_seed(flat_stories, initial_story = None):
    vect = TfidfVectorizer(min_df=1, stop_words="english", norm="l2")
    if initial_story is not None:
        flat_stories = [initial_story] + flat_stories
    tfidf = vect.fit_transform(flat_stories)                                                                                                                                                                                                                       
    similarity_matrix = tfidf * tfidf.T 
    return similarity_matrix.toarray()

def get_embeddings(stories):
    embeddings = model.encode(stories, convert_to_tensor=True)
    return embeddings.cpu().numpy()

def get_SBERT_similarity(story1, story2, embedded = False):

    if not embedded:

        embeddings1 = model.encode(story1, convert_to_tensor=True)
        embeddings2 = model.encode(story2, convert_to_tensor=True)
    else:
        embeddings1 = story1
        embeddings2 = story2
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores.cpu().numpy()

def get_similarity_matrix_single_seed_SBERT(flat_stories):
    model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    embeddings = model.encode(flat_stories, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    return cosine_scores.cpu().numpy()
    similarity_matrix = embeddings @ embeddings.T / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embeddings, axis=1)[:, None])
    print(similarity_matrix)
    return similarity_matrix

def get_similarity_matrix(all_seed_flat_stories):
    all_seeds_similarity_matrix = []
    for flat_stories in all_seed_flat_stories:
        similarity_matrix = get_similarity_matrix_single_seed_SBERT(flat_stories)
        all_seeds_similarity_matrix.append(similarity_matrix)
    return all_seeds_similarity_matrix


def convert_to_json_serializable(obj):
    """
    Recursively converts non-JSON serializable objects to JSON serializable format.
    """
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {convert_to_json_serializable(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj




def get_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0  
    
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    
    similarity = max(-1, min(1, similarity))
    
    return similarity



def get_difficulty(text):

    return textstat.gunning_fog(text)


def get_difficulties_single_seed(intial_story, stories):
    
    difficulties = []
    
    difficulties.append(get_difficulty(intial_story))

    for story in stories:
        
        difficulties.append(get_difficulty(story))
    
    return difficulties

def get_difficulties(intial_story, all_seed_stories):
    all_seeds_difficulties = []
    for stories in all_seed_stories:
        difficulties = get_difficulties_single_seed(intial_story, stories)
        all_seeds_difficulties.append(difficulties)
    return all_seeds_difficulties
    
    



def get_toxicity(text):
    return Detoxify('original').predict(text)['toxicity']


def get_toxicities_single_seed(intial_story, stories):

    toxicities = []
    
    toxicities.append(get_toxicity(intial_story))

    for i in range(len(stories)):
        story = stories[i]
        
        toxicities.append(get_toxicity(story))
    
    return toxicities


def get_toxicities(intial_story, all_seed_stories, folder):

    all_seeds_toxicities = []
    for i in range(len(all_seed_stories)):
        try:
            file = open(f"{folder}/toxicities{i}.obj",'rb')
            toxicities = pickle.load(file)
            file.close()
        except:
            stories = all_seed_stories[i]
            toxicities = get_toxicities_single_seed(intial_story, stories)
            filehandler = open(f"{folder}/toxicities{i}.obj","wb")
            pickle.dump(toxicities, filehandler)
            filehandler.close()
            
        all_seeds_toxicities.append(toxicities)
            
                


    


    return all_seeds_toxicities
   

def get_positivity(text):
    sid = SentimentIntensityAnalyzer()    
    return sid.polarity_scores(text)['compound']

def get_positivities_single_seed(intial_story, stories):
        
        positivities = []
        
        positivities.append(get_positivity(intial_story))
    
        for story in stories:
            
            positivities.append(get_positivity(story))
        
        return positivities

def get_positivities(intial_story, all_seed_stories, folder):
    all_seeds_positivities = []
    for stories in all_seed_stories:
        positivities = get_positivities_single_seed(intial_story, stories)
        all_seeds_positivities.append(positivities)
    return all_seeds_positivities


def get_length_scores_single_seed(intial_story, stories):
    lengths = []
    lengths.append(len(intial_story))
    for story in stories:
        lengths.append(len(story))
    return lengths

def get_lengths(intial_story, all_seed_flat_stories):
    all_seeds_lengths = []
    for flat_stories in all_seed_flat_stories:
        lengths = get_length_scores_single_seed(intial_story, flat_stories)
        all_seeds_lengths.append(lengths)
    return all_seeds_lengths


def get_cumulativeness(values):


    total_change = np.max(values) - np.min(values)


    total_change_after_first = np.max(values[1:]) - np.min(values[1:])

    if total_change == 0:
        return 0.0
    
    return total_change_after_first / total_change

    



def get_cumulativeness_all_seeds(values):
    all_seeds_cumulativeness = []
    for seed in values:
        cumulativeness = get_cumulativeness(seed)
        all_seeds_cumulativeness.append(cumulativeness)
    return all_seeds_cumulativeness




# Calculate the confidence interval
def conf_interval(x, y, slope, intercept, std_err, confidence=0.95):
    # Compute standard deviation of residuals
    residuals = y - (slope * x + intercept)
    std_residual = np.std(residuals)

    # Compute the critical value for the confidence interval
    t_critical = stats.t.ppf((1 + confidence) / 2, len(x) - 2)
    margin_of_error = t_critical * std_residual / np.sqrt(len(x))

    return margin_of_error

