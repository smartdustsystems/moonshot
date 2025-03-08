import pandas as pd
import numpy as np
import spacy
from bs4 import BeautifulSoup
import requests

class DataLoader:
    def __init__(self, num_samples=20):
        self.num_samples = num_samples
        self.nlp = spacy.load("en_core_web_md")
        self.data = None
    
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
