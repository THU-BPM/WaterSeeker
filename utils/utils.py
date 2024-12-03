# ==============================
# utils.py
# Description: Utility functions
# ==============================

import json
import time
import openai
import random
import nltk
from nltk.corpus import wordnet

def load_config_file(path: str) -> dict:
    """Load a JSON configuration file from the specified path and return it as a dictionary."""
    try:
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return config_dict

    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{path}': {e}")
        # Handle other potential JSON decoding errors here
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Handle other unexpected errors here
        return None


class GPTParaphraser():
    """Paraphrase a text using the GPT model."""

    def __init__(self, openai_model: str, prompt: str) -> None:
        """
            Initialize the GPT paraphraser.

            Parameters:
                openai_model (str): The OpenAI model to use for paraphrasing.
                prompt (str): The prompt to use for paraphrasing.
        """
        self.openai_model = openai_model
        self.prompt = prompt

    def edit(self, text: str, reference=None):
        """Paraphrase the text using the GPT model."""
        openai_util = OpenAIAPI(model=self.openai_model, temperature=0.2, system_content="Your are a helpful assistant to rewrite the text.")
        paraphrased_text = openai_util.get_result(self.prompt + text)
        return paraphrased_text

class OpenAIAPI:
    """API class for OpenAI API."""
    def __init__(self, model, temperature, system_content):
        """
            Initialize OpenAI API with model, temperature, and system content.

            Parameters:
                model (str): Model name for OpenAI API.
                temperature (float): Temperature value for OpenAI API.
                system_content (str): System content for OpenAI API.
        """

        self.model = model
        self.api_base = "xxx"
        self.api_key = "sk-xxx"
        self.temperature = temperature
        self.system_content = system_content
        self.client = openai.OpenAI(base_url=self.api_base, api_key=self.api_key)

    def get_result_from_gpt4(self, query):
        """get result from GPT-4 model."""
        response = self.client.chat.completions.create(
            model="gpt-4-0613",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": query},
            ]
        )
        return response
    
    def get_result_from_gpt3_5(self, query):
        """get result from GPT-3.5 model."""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0613",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": query},
            ]
        )
        return response
    
    def get_result(self, query):
        """get result from OpenAI API. Try 5 times before giving up."""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if self.model == 'gpt-3.5-turbo':
                    result = self.get_result_from_gpt3_5(query)
                elif self.model == 'gpt-4':
                    result = self.get_result_from_gpt4(query)
                return result.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API error: {str(e)}")
                retry_count += 1
                if retry_count == max_retries:
                    return ""
                time.sleep(10)

class SynonymSubstitution():
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        nltk.download('wordnet')

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        num_words = len(words)
        
        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)
        
            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text

class WordDeletion():
    """Delete words randomly from the text."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the word deletion editor.

            Parameters:
                ratio (float): The ratio of words to delete.
        """
        self.ratio = ratio

    def edit(self, text: str, reference=None):
        """Delete words randomly from the text."""

        # Handle empty string input
        if not text:  
            return text

        # Split the text into words and randomly delete each word based on the ratio
        word_list = text.split()
        edited_words = [word for word in word_list if random.random() >= self.ratio]

        # Join the words back into a single string
        deleted_text = ' '.join(edited_words)

        return deleted_text