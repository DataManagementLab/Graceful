from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModel
# needed to not show a certain warning all the time
from transformers import logging

logging.set_verbosity_error()


###################################################

# current strategy for representation of the input string
# concatenate the last 4 hidden layers => see BERT paper since this worked best
@lru_cache(maxsize=1000)
def get_string_vec(string, tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
                   model=AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)):
    model.eval()  # enable feed forward operation
    tokens = tokenizer.tokenize(string)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [1] * len(tokens)
    token_tensor = torch.tensor([token_ids])
    seg_tensor = torch.tensor([segment_ids])

    with torch.no_grad():
        outputs = model(token_tensor, seg_tensor)
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    token_vecs_cat = []
    for token in token_embeddings:
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        token_vecs_cat.append(cat_vec)

    # take the sum of all tokens to get the vector for the entire string
    out_ten = torch.zeros(token_vecs_cat[0].size())
    for vec in token_vecs_cat:
        out_ten = torch.add(out_ten, vec)
    return out_ten.numpy()
