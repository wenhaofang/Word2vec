import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW_NS_Module(nn.Module):
    def __init__(self, emb_size, emb_dims, device):
        super(CBOW_NS_Module, self).__init__()
        self.device = device
        self.emb_size = emb_size
        self.emb_dims = emb_dims
        self.u_embeddings = nn.Embedding(emb_size, emb_dims)
        self.v_embeddings = nn.Embedding(emb_size, emb_dims)

    def forward(self, src_words, trg_words, wmasks, labels):
        p_src_emb = []
        for src_word in src_words:
            p_src_emb.append(self.u_embeddings(
                torch.tensor(src_word, dtype = torch.long).to(self.device)
            ).sum(dim = 0))
        src_emb = torch.stack(p_src_emb)
        trg_emb = self.v_embeddings(
            torch.tensor(trg_words, dtype = torch.long).to(self.device)
        )

        wmasks = torch.tensor(wmasks, dtype = torch.float).to(self.device)
        labels = torch.tensor(labels, dtype = torch.float).to(self.device)

        pred = torch.bmm(src_emb.unsqueeze(1), trg_emb.permute(0, 2, 1)).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(pred.float(), labels, reduction = 'none', weight = wmasks)
        loss = (loss.mean(dim = 1) * wmasks.shape[1] / wmasks.sum(dim = 1)).mean()

        return loss

    def get_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()

def get_module(option, vocab_size, device):
    return CBOW_NS_Module(vocab_size, option.emb_dim, device)

if __name__ == '__main__':
    import random

    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    vocab_size = 10000

    module = get_module(option, vocab_size, device).to(device)

    src_words = [[random.randint(0, vocab_size - 1) for _ in range(random.randint(1, 6))] for _ in range(option.batch_size)]
    trg_words = [[random.randint(0, vocab_size - 1) for _ in range(7)] for _ in range(option.batch_size)]
    wmasks = [[random.randint(0, 1) for _ in range(7)] for _ in range(option.batch_size)]
    labels = [[random.randint(0, 1) for _ in range(7)] for _ in range(option.batch_size)]

    loss = module(src_words, trg_words, wmasks, labels)

    print(loss)
