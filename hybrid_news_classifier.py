import time
class HybridNewsClassifier:
    def __init__(self, client):
        self.client = client
    
    def check_url_credibility(self, news_url, openai_engine=None,dataset=None):
        message = f"""
        You are an expert in evaluating news website credibility. Determine whether this source is **trusted or untrusted**.

        **Instructions:**
        - If the website is a **reputable source** (e.g., BBC, CNN, Reuters), return **"Trusted"**.
        - If the website is known for **misinformation or fake news**, return **"Untrusted"**.

        **News URL:** "{news_url}"

        **Classification (Trusted / Untrusted):**
        """
        start_time_detection=time.time()
        response = self.client.chat.completions.create(
            model=openai_engine,
            messages=[{"role": "system", "content": "You are a news credibility classifier."},
                      {"role": "user", "content": message}],
            temperature=0
        )
        end_time_detection= time.time()
        detection_time=end_time_detection - start_time_detection

        result = response.choices[0].message.content.strip()
        token_usage = {
            "Data":dataset,
            "Engine":openai_engine,
            "Method": "hybrid",
            "Prompt Type": "hybrid",
            "Prompt Tokens": response.usage.prompt_tokens,
            "Completion Tokens": response.usage.completion_tokens,
            "Total Tokens": response.usage.total_tokens,
            "Detection Time per Article":detection_time
        }
        return result, token_usage

    def check_title_sensationalism(self, title, openai_engine=None,dataset=None):
        message = f"""
        Analyze the given news title to determine if it contains **sensationalist language**.

        **Instructions:**
        - If the title has **clickbait, emotionally charged words, or exaggeration**, return **"Sensational"**.
        - If the title is **neutral and factual**, return **"Neutral"**.

        **News Title:** "{title}"

        **Classification (Sensational / Neutral):**
        """
        start_time_detection=time.time()
        response = self.client.chat.completions.create(
            model=openai_engine,
            messages=[{"role": "system", "content": "You detect sensationalism in news titles."},
                      {"role": "user", "content": message}],
            temperature=0
        )
        end_time_detection= time.time()
        detection_time=end_time_detection - start_time_detection
        result = response.choices[0].message.content.strip()
        token_usage = {
            "Data":dataset,
            "Engine":openai_engine,
            "Method": "hybrid",
            "Prompt Type": "hybrid",
            "Prompt Tokens": response.usage.prompt_tokens,
            "Completion Tokens": response.usage.completion_tokens,
            "Total Tokens": response.usage.total_tokens,
            "Detection Time per Article":detection_time
        }
        
        return result, token_usage

    def check_article_logic(self, article_text, openai_engine=None,dataset=None):
        message = f"""
        Analyze the given news article and determine if its content is **logical** or **illogical**.

        **Instructions:**
        - If the article **makes sense and follows factual reporting**, return **"Logical"**.
        - If the article contains **unbelievable claims, conspiracy theories, or contradictions**, return **"Illogical"**.

        **Article Text (Except):** "{article_text}"

        **Classification (Logical / Illogical):**
        """
        start_time_detection=time.time()
        response = self.client.chat.completions.create(
            model=openai_engine,
            messages=[{"role": "system", "content": "You verify the logical consistency of news articles."},
                      {"role": "user", "content": message}],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        end_time_detection= time.time()
        detection_time=end_time_detection - start_time_detection
        token_usage = {
            "Data":dataset,
            "Engine":openai_engine,
            "Method": "hybrid",
            "Prompt Type": "hybrid",
            "Prompt Tokens": response.usage.prompt_tokens,
            "Completion Tokens": response.usage.completion_tokens,
            "Total Tokens": response.usage.total_tokens,
            "Detection Time per Article":detection_time
        }

        return result, token_usage

    def classify_news(self, title, url, article_text, openai_engine=None,dataset=None):
        url_verdict, url_tokens = self.check_url_credibility(url, dataset=dataset, openai_engine=openai_engine)
        if url_verdict == "Untrusted":
            return "fake", url_tokens  
        
        title_verdict, title_tokens = self.check_title_sensationalism(title, dataset=dataset, openai_engine=openai_engine)
        if title_verdict == "Sensational":
            return "fake", title_tokens 

        text_verdict, text_tokens = self.check_article_logic(article_text, dataset=dataset, openai_engine=openai_engine)
        if text_verdict == "Illogical":
            return "fake", text_tokens 

        # If all checks pass, classify as real
        return "real", {
            "Data":dataset,
            "Engine":openai_engine,
            "Method": "hybrid",
            "Prompt Type": "hybrid",
            "Prompt Tokens": url_tokens["Prompt Tokens"] + title_tokens["Prompt Tokens"] + text_tokens["Prompt Tokens"],
            "Completion Tokens": url_tokens["Completion Tokens"] + title_tokens["Completion Tokens"] + text_tokens["Completion Tokens"],
            "Total Tokens": url_tokens["Total Tokens"] + title_tokens["Total Tokens"] + text_tokens["Total Tokens"],
            "Detection Time per Article": url_tokens["Detection Time per Article"] + title_tokens["Detection Time per Article"] + text_tokens["Detection Time per Article"]
        }