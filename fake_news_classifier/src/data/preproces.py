def prepross_data(text):
    import re 
    from nltk import word_tokenize
    from nltk.corpus import stopwords

    text = re.sub(r'[^a-zA-z]'," ", text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def build_vocab(data):
    from collections import Counter
    counter = Counter([token for tokens in data for token in tokens])
    filtered_tokens = [word for word, count in counter.items() if count >= 2]
    vocab = {word:idx+2 for idx,word in enumerate(filtered_tokens)}
    vocab['<PAD>'] = 0  # Padding token
    vocab['<UNK>'] = 1  # Unknown token
    return vocab

def build_seq(tokens, vocab, max_len=100):
    seq = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

