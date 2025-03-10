
import pandas as pd
import numpy as np
import spacy
from bs4 import BeautifulSoup
import requests
import zipfile
import os
import dask.dataframe as dd
import csv

class DataLoader:
    def __init__(self, num_samples=20):
        self.num_samples = num_samples
        self.nlp = spacy.load("en_core_web_md")
        self.data = None

    def read_gsc_data(self, file_path="/app/ingestion/dumps/gsc_202503041305.csv"):
        """
        Reads an unzipped CSV file using Dask for memory-efficient processing.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = dd.read_csv(file_path)  # Use Dask to read CSV in a lazy way
        return df


    def read_details_data(self, file_path="/app/ingestion/dumps/details_202503100434.csv"):
        """
        Reads an unzipped CSV file using Dask for memory-efficient processing, handling bad lines and encoding issues.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = dd.read_csv(
            file_path, 
            sample=256000000,   # Increase sample size
            sample_rows=100,    # Ensure Dask reads enough rows
            on_bad_lines="skip",  # Ignore badly formatted lines
            engine="python",   # Use Python engine to better handle errors
            quoting=csv.QUOTE_NONE,  # Prevent issues with unexpected quotes
            encoding="utf-8",  # Ensure proper encoding
            dtype={"html_code": "object"},  # Explicitly set dtype to object (string)
            na_values=[],  # Prevents Dask from treating empty strings as NaN
            keep_default_na=False  # Ensures empty values are not converted to NaN

        )

        return df

    
    def generate_synthetic_data(self):
        np.random.seed(42)
        data = {
            "page_title": [f"Page {i}" for i in range(self.num_samples)],
            "keyword_density": np.random.uniform(0.5, 5.0, self.num_samples),
            "readability_score": np.random.uniform(30, 90, self.num_samples),
            "page_speed": np.random.uniform(1, 5, self.num_samples),
            "mobile_friendly": np.random.choice([0, 1], self.num_samples),
            "structured_data": np.random.choice([0, 1], self.num_samples),
            "bounce_rate": np.random.uniform(10, 90, self.num_samples),
            "time_on_page": np.random.uniform(10, 300, self.num_samples),
            "ctr": np.random.uniform(0.01, 0.3, self.num_samples),
            "backlinks": np.random.randint(0, 1000, self.num_samples),
            "serp_rank": np.random.randint(1, 100, self.num_samples),
            "target_search": 'Best universities'
        }
        self.data = pd.DataFrame(data)

    def fetch_real_html(self, url):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.text
        except:
            pass
        return None
    
    def generate_fake_html(self, title, keyword_density):
        keywords = "SEO, ranking, Google, search engine, optimization"
        content = " ".join(["This is SEO content."] * 50)
        content += " " + (" ".join([keywords] * int(keyword_density * 10)))
        
        html = f"""
        <html>
        <head>
            <title>{title}</title>
            <meta name="description" content="SEO optimized page.">
        </head>
        <body>
            <h1>{title}</h1>
            <p>{content}</p>
        </body>
        </html>
        """
        return html

    def enrich_with_html(self):
        urls = [
            "https://moz.com/learn/seo/what-is-seo",
            "https://backlinko.com/hub/seo",
        ]
        self.data["html_content"] = self.data.index.map(lambda i: 
            self.fetch_real_html(urls[i % len(urls)]) or 
            self.generate_fake_html(self.data.loc[i, "page_title"], self.data.loc[i, "keyword_density"])
        )

    def get_data(self):
        return self.data
