import pandas as pd
def load_data(fake='data/raw/Fake.csv', true='data/raw/True.csv'):
    import os 
    print(os.getcwd())
    fake_news = pd.read_csv(fake)
    true_news = pd.read_csv(true)
    fake_news_labels = pd.Series(0, index=fake_news.index, name='label')
    true_news_labels = pd.Series(1, index=true_news.index, name='label')
    fake_news = pd.concat([fake_news, fake_news_labels], axis=1)
    true_news = pd.concat([true_news, true_news_labels], axis=1)
    data = pd.concat([fake_news, true_news],ignore_index=True)
    return data 

# print(load_data()['title'][0])