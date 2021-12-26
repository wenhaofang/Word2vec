import torch

def save_checkpoint(loader, module, save_path):
    vocab = loader.get_vocab_self()
    embed = module.get_embeddings()
    checkpoint = {
        'id2word': vocab['id2word'],
        'word2id': vocab['word2id'],
        'embedding': embed
    }
    torch.save (checkpoint, save_path)

def load_checkpoint(load_path):
    checkpoint = torch.load(load_path)
    return checkpoint

def get_similar_tokens(checkpoint, query_tok, query_num):
    embeddings = checkpoint['embedding']
    word2id = checkpoint['word2id']
    id2word = checkpoint['id2word']
    if query_tok in word2id:
        W = torch.tensor(embeddings)
        x = torch.tensor(embeddings[word2id[query_tok]])
        id2cos = torch.matmul(W, x) / (torch.sum(W * W, dim = 1) * torch.sum(x * x) + 1e-9).sqrt()
        _,topk = torch.topk(id2cos, k = query_num + 1)
        topk = topk.cpu().numpy()
        return [(id2word[word_id], id2cos[word_id]) for word_id in topk[1:]]
    else:
        return []
