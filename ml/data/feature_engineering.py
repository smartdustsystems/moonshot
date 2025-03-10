
from bs4 import BeautifulSoup
import pandas as pd
import spacy

class FeatureEngineer:
    def __init__(self, df):
        self.df = df
        self.nlp = spacy.load("en_core_web_md")

    def extract_html_features(self, html):
        soup = BeautifulSoup(html, "html.parser")
        meta_desc = soup.find("meta", attrs={"name": "description"})
        meta_desc_length = len(meta_desc["content"]) if meta_desc else 0
        num_headers = len(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]))
        num_links = len(soup.find_all("a"))
        return meta_desc_length, num_headers, num_links

    def apply_html_extraction(self):
        self.df[["meta_desc_length", "num_headers", "num_links"]] = self.df["html_content"].apply(
            lambda html: pd.Series(self.extract_html_features(html))
        )

    def extract_spacy_embeddings(self):
        self.df["spacy_embedding"] = self.df["html_content"].apply(
            lambda html: self.nlp(BeautifulSoup(html, "html.parser").get_text()).vector.tolist()
        )

    def get_features(self):
        return self.df