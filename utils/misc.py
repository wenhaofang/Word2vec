import torch

def save_embedding(loader, module, save_path):
    vocab = loader.get_vocab_self()
    embed = module.get_embeddings()
    checkpoint = {
        'id2word': vocab['id2word'],
        'word2id': vocab['word2id'],
        'embedding': embed
    }
    torch.save(checkpoint, save_path)
