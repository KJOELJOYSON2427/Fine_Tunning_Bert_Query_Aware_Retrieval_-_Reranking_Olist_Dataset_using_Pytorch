import pandas as pd

class SearchCorpusBuilder:
    def __init__(self,df, max_docs=30000):
        self.df = df
        self.max_docs = max_docs
        self.corpus = None


    def build_corpus(self):
        self.corpus = self.df[[
            'review_id',
            'review_text',
            'review_score',
            'product_category_name_english',
            'order_id'
        ]].copy()

        self.corpus = self.corpus.reset_index(drop=True)

        if len(self.corpus) > self.max_docs:
            self.corpus =(
                 self.corpus
                 .sample(n=self.max_docs, random_state = 42)
                 .reset_index(drop =True)
            )  
            print(f"Using a sample of {self.max_docs} reviews for the search system")
        else:
            print(f"Using all {len(self.corpus)} reviews")

        print(f"\nFinal corpus size: {len(self.corpus)} documents")
        print(f"Average text length: {self.corpus['review_text'].str.len().mean():.0f} characters")
