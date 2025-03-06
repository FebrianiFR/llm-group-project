class HybridNewsClassifier:
    def __init__(self, client):
        self.client = client
    
    def check_url_credibility(self, news_url):
        prompt = f"""
        You are an expert in evaluating news website credibility. Determine whether this source is **trusted or untrusted**.

        **Instructions:**
        - If the website is a **reputable source** (e.g., BBC, CNN, Reuters), return **"Trusted"**.
        - If the website is known for **misinformation or fake news**, return **"Untrusted"**.

        **News URL:** "{news_url}"

        **Classification (Trusted / Untrusted):**
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a news credibility classifier."},
                      {"role": "user", "content": prompt}],
            max_tokens=2
        )
        result = response.choices[0].message.content.strip()
        token_usage = {
            "Method": "hybrid",
            "Prompt Type": "hybrid",
            "Prompt Tokens": response.usage.prompt_tokens,
            "Completion Tokens": response.usage.completion_tokens,
            "Total Tokens": response.usage.total_tokens
        }
        return result, token_usage

    def check_title_sensationalism(self, title):
        prompt = f"""
        Analyze the given news title to determine if it contains **sensationalist language**.

        **Instructions:**
        - If the title has **clickbait, emotionally charged words, or exaggeration**, return **"Sensational"**.
        - If the title is **neutral and factual**, return **"Neutral"**.

        **News Title:** "{title}"

        **Classification (Sensational / Neutral):**
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You detect sensationalism in news titles."},
                      {"role": "user", "content": prompt}],
            max_tokens=2
        )
        result = response.choices[0].message.content.strip()
        token_usage = {
            "Method": "hybrid",
            "Prompt Type": "hybrid",
            "Prompt Tokens": response.usage.prompt_tokens,
            "Completion Tokens": response.usage.completion_tokens,
            "Total Tokens": response.usage.total_tokens
        }
        return result, token_usage

    def check_article_logic(self, article_text):
        prompt = f"""
        Analyze the given news article and determine if its content is **logical** or **illogical**.

        **Instructions:**
        - If the article **makes sense and follows factual reporting**, return **"Logical"**.
        - If the article contains **unbelievable claims, conspiracy theories, or contradictions**, return **"Illogical"**.

        **Article Text (Excerpt):** "{article_text[:500]}"

        **Classification (Logical / Illogical):**
        """
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You verify the logical consistency of news articles."},
                      {"role": "user", "content": prompt}],
            max_tokens=2
        )
        result = response.choices[0].message.content.strip()
        token_usage = {
            "Method": "hybrid",
            "Prompt Type": "hybrid",
            "Prompt Tokens": response.usage.prompt_tokens,
            "Completion Tokens": response.usage.completion_tokens,
            "Total Tokens": response.usage.total_tokens
        }
        return result, token_usage

    def classify_news(self, title, url, article_text):
        url_verdict, tokens = self.check_url_credibility(url)
        if url_verdict in ["Trusted", "Untrusted"]:
            return "real", tokens if url_verdict == "Trusted" else "fake"

        title_verdict, tokens = self.check_title_sensationalism(title)
        if title_verdict == "Neutral":
            return "real", tokens

        text_verdict, tokens = self.check_article_logic(article_text)
        return "fake", tokens if text_verdict == "Illogical" else "real"
