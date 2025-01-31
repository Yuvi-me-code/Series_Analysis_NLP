import torch
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import os
import sys
import pathlib
import nltk
nltk.download('punkt_tab')
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles_dataset

class ThemeClassifier():
    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)

    model_name = "facebook/bart-large-mnli"

    def load_model(self, device):
        theme_classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device = device
        )

        return theme_classifier
    
    def get_themes_inference(self, script):
        script_sentences = sent_tokenize(script)

        sentence_batch_size = 20
        script_batches = []
        for index in range(0, len(script_sentences), sentence_batch_size):
            sent = " ".join(script_sentences[index : index + sentence_batch_size])
            script_batches.append(sent)

        
        theme_output = self.theme_classifier(
            script_batches,
            self.theme_list, 
            multi_label = True
        )

        themes = {}
        for output in theme_output:
            for label, score in zip(output['labels'], output['scores']):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)
        
        themes = {key : np.mean(np.array(value)) for key, value in themes.items()}

        return themes
    
    def get_themes(self, dataset_path, save_path = None):

        if save_path is not None and os.path.exists(save_path):
            data = pd.read_csv(save_path)
            return data
        
        data = load_subtitles_dataset(dataset_path)
        output_themes = data['scripts'].apply(self.get_themes_inference)
        themes_data = pd.DataFrame(output_themes.tolist())
        data[themes_data.columns] = themes_data

        if save_path is not None and save_path != "":
            data.to_csv(save_path, index = False)
        
        return data