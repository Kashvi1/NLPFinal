import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def aspect_extraction(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    grammar = r"""
NP: {<JJ|JJR|JJS>*<NN|NNS>+}
NP2: {<NN|NNS>+<VB.*>*<JJ|JJR|JJS>+}
NP3: {<JJ|JJR|JJS>+<NN|NNS>+<JJ|JJR|JJS>+}
NP_CONJ: {<NN|NNS>+(<CC><NN|NNS>+)+}
ADJ: {<JJ|JJR|JJS>+}
"""
    tree = RegexpParser(grammar).parse(tagged)
    return [
        " ".join(word for word, _ in subtree.leaves())
        for subtree in tree.subtrees()
        if subtree.label() in ['NP', 'NP2', 'NP3', 'NP_CONJ', 'ADJ']
    ]

def get_sentiment(text):
    return sia.polarity_scores(text)['compound']

def label(score):
    return -1 if score <= -0.05 else 1 if score >= 0.05 else 0

def process_chunk(chunk):
    chunk['aspects'] = chunk['text'].apply(lambda x: aspect_extraction(x) if isinstance(x, str) else [])
    chunk = chunk.explode('aspects').reset_index(drop=True)
    chunk = chunk[chunk['aspects'].notna() & (chunk['aspects'] != '')]
    chunk['aspect_score'] = chunk['aspects'].apply(get_sentiment)
    chunk['aspect_sentiment'] = chunk['aspect_score'].apply(label)
    return chunk

if __name__ == '__main__':
    doc_path = "final_review1.csv"
    chunksize = 50000
    reader = pd.read_csv(doc_path, chunksize=chunksize)
    temp_file = "processed_aspects.csv"
    business_agg = {}

    with open(temp_file, 'w', encoding='utf-8', newline='') as out:
        header_written = False
        for chunk in reader:
            result = process_chunk(chunk)
            for _, row in result.iterrows():
                bid = row['business_id']
                score = row['aspect_score']
                name = row['name']
                if bid in business_agg:
                    business_agg[bid][0] += score
                    business_agg[bid][1] += 1
                else:
                    business_agg[bid] = [score, 1, name]
            result.to_csv(out, index=False, header=not header_written)
            header_written = True

    business_summary = pd.DataFrame(
        [[bid, total / count, name] for bid, (total, count, name) in business_agg.items()],
        columns=['business_id', 'overall_score', 'name']
    )

    final_file = "final_aspect_with_overall.csv"
    reader2 = pd.read_csv(temp_file, chunksize=chunksize)

    with open(final_file, 'w', encoding='utf-8', newline='') as out:
        header_written = False
        for chunk in reader2:
            chunk = chunk.merge(business_summary, on='business_id', suffixes=('', '_overall'))
            if 'name_overall' in chunk.columns:
                chunk.drop(columns=['name_overall'], inplace=True)
            chunk = chunk[['business_id', 'name', 'text', 'aspects', 'aspect_score', 'aspect_sentiment', 'overall_score']]
            chunk.to_csv(out, index=False, header=not header_written)
            header_written = True

    print("Saved aspect‑level + overall sentiment to", final_file)
