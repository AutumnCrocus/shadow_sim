import torch
import card2vec
from gensim.models import doc2vec
names, texts = card2vec.load_json("all")
d2v_model = doc2vec.Doc2Vec.load("all.model")
d2v_ini_weight = torch.Tensor([[0.0]*len(d2v_model.docvecs[0])]+[d2v_model.docvecs[i] for i in range(len(d2v_model.docvecs))])
