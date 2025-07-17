import torch
from gensim.models import Doc2Vec
from tqdm import tqdm
import numpy as np

doc2vec_model_path = './wiki_doc2vec/'
predata_file_path = './cresci15/'
model_128 = Doc2Vec.load(doc2vec_model_path + "pretrained_wiki_doc2vec_128.model")

graph = torch.load(predata_file_path + 'cache/pre_graph.pt')

user_tweet_num = torch.zeros(graph['user'].x.size(0))
for edge in tqdm(graph['user', 'post', 'tweet'].edge_index.t().tolist(), desc='聚合推文信息'):
    user_tweet_num[edge[0]] += 1
user_tweet_feature = torch.load(predata_file_path + 'cache/user_tweet_feature.pt')
user_tweet_num = torch.where(torch.tensor(user_tweet_num == 0), 1, user_tweet_num)
graph['user'].x = torch.cat((graph['user'].x, user_tweet_feature / user_tweet_num.view(-1, 1)), dim=1)

origin_tweet_feature = []
for text in tqdm(graph['tweet'].x, desc='编码每篇推文'):
    origin_tweet_feature.append(model_128.infer_vector(text.split()))
graph['tweet'].x1 = torch.tensor(np.array(origin_tweet_feature))
del graph['tweet'].x

max_tweets = 200
each_user_tweet = [[] for _ in range(graph['user'].x.size(0))]
for edge in tqdm(graph['user', 'post', 'tweet'].edge_index.t().tolist()):
    if len(each_user_tweet[edge[0]]) < max_tweets:
        each_user_tweet[edge[0]].append(graph['tweet'].x1[edge[1]].tolist())
max_len = 128
each_user_tweet = [[[0]*max_len] * (max_tweets - len(tweets)) + tweets for tweets in each_user_tweet]
each_user_tweet = torch.tensor(np.array(each_user_tweet), dtype=torch.float)

print(each_user_tweet.shape)
graph['user'].x1 = each_user_tweet
torch.save(graph, './predata/graph1.pt')

