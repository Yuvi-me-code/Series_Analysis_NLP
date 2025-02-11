import spacy
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
from ast import literal_eval
import os, sys, pathlib
folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles_dataset

class NamedEntityRecognizer:
    def __init__(self):
        self.nlp_model = self.load_model()
        pass

    def load_model(self):
        nlp = spacy.load("en_core_web_trf")
        return nlp

    def get_ners_inference(self, script):
        script_sentences = sent_tokenize(script)
        ner_output = []

        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = []  # Use a list instead of a set to preserve order
            for entity in doc.ents:
                if entity.label_ == "PERSON":  # Use label_ instead of label
                    full_name = entity.text
                    first_name = full_name.split(" ")[0].strip()  # Extract first name
                    ners.append(first_name)  # Preserve order instead of using a set

            ner_output.append(ners)

        return ner_output
    
    def get_ners(self, dataset_path, save_path = None):
        if save_path is not None and os.path.exists(save_path):
            data = pd.read_csv(save_path)
            data['ners'] = data['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return data
        data = load_subtitles_dataset(dataset_path)

        data = data.head(10)

        data['ners'] = data['scripts'].apply(self.get_ners_inference)
        if save_path is not None:
            data.to_csv(save_path, index=False)

        return data