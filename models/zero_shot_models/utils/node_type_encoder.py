import numpy as np
import torch
from torch import nn

from models.preprocessing.feature_statistics import FeatureType
from models.zero_shot_models.utils.embeddings import EmbeddingInitializer
from models.zero_shot_models.utils.fc_out_model import FcOutModel


class NodeTypeEncoder(FcOutModel):
    """
    Model to encode one type of nodes in the graph (with particular features)
    """

    def __init__(self, features, feature_statistics, max_emb_dim=32, drop_whole_embeddings=False,
                 one_hot_embeddings=True, allow_gradients_on_embeddings: bool = False, **kwargs):

        # for f in features:
        #     if f not in feature_statistics:
        #         cleaned_feature_statistics = dict()
        #         for k, v in feature_statistics.items():
        #             feature_info = v.copy()
        #             if 'value_dict' in feature_info:
        #                 feature_info.pop('value_dict')
        #             cleaned_feature_statistics[k] = feature_info
        #         raise ValueError(f"Did not find {f} in feature statistics: {cleaned_feature_statistics}")

        self.features = features
        self.feature_types = [FeatureType[feature_statistics[feat]['type']] for feat in features]
        self.feature_idxs = []

        # initialize embeddings and input dimension

        self.input_dim = 0
        self.input_feature_idx = 0
        embeddings = dict()
        for i, (feat, type) in enumerate(zip(self.features, self.feature_types)):
            if type == FeatureType.numeric:
                # a single value is encoded here
                self.feature_idxs.append(np.arange(self.input_feature_idx, self.input_feature_idx + 1))
                self.input_feature_idx += 1

                self.input_dim += 1
            elif type == FeatureType.categorical:
                if feat in ["in_dts", "ops", "cmdtypes"]:  # multi-label features
                    # similarly, a single value is encoded here
                    target_list_len = max(10, feature_statistics[feat]['no_vals'])
                    self.feature_idxs.append(
                        np.arange(self.input_feature_idx, self.input_feature_idx + feature_statistics[feat]['no_vals']))
                    self.input_feature_idx += target_list_len

                    embd = EmbeddingInitializer(feature_statistics[feat]['no_vals'], max_emb_dim, kwargs['p_dropout'],
                                                drop_whole_embeddings=drop_whole_embeddings, one_hot=one_hot_embeddings,
                                                allow_gradients_on_embeddings=allow_gradients_on_embeddings)
                    embeddings[feat] = embd
                    self.input_dim += embd.emb_dim
                else:
                    # similarly, a single value is encoded here
                    self.feature_idxs.append(np.arange(self.input_feature_idx, self.input_feature_idx + 1))
                    self.input_feature_idx += 1

                    embd = EmbeddingInitializer(feature_statistics[feat]['no_vals'], max_emb_dim, kwargs['p_dropout'],
                                                drop_whole_embeddings=drop_whole_embeddings, one_hot=one_hot_embeddings,
                                                allow_gradients_on_embeddings=allow_gradients_on_embeddings)
                    embeddings[feat] = embd
                    self.input_dim += embd.emb_dim

                # addition for UDFs; lib vector
            elif type == FeatureType.vector:
                self.feature_idxs.append(
                    np.arange(self.input_feature_idx, self.input_feature_idx + feature_statistics[feat]['vec_len']))
                self.input_feature_idx += feature_statistics[feat]['vec_len']

                self.input_dim += feature_statistics[feat]['vec_len']
            else:
                raise NotImplementedError

        super().__init__(input_dim=self.input_dim, **kwargs)

        self.embeddings = nn.ModuleDict(embeddings)

    def forward(self, input):
        if self.no_input_required:
            return self.replacement_param.repeat(input.shape[0], 1)

        # assert input.shape[1] == self.input_feature_idx
        encoded_input = []
        for feat, feat_type, feat_idxs in zip(self.features, self.feature_types, self.feature_idxs):
            feat_data = input[:, feat_idxs]

            if feat_type == FeatureType.numeric:
                encoded_input.append(feat_data)
            elif feat_type == FeatureType.categorical:
                # convert to long
                feat_data = feat_data.long()
                if feat not in ["in_dts", "ops", "cmdtypes"]:
                    feat_data = torch.reshape(feat_data, (-1,))
                    embd_data = self.embeddings[feat](feat_data)
                    encoded_input.append(embd_data)
                else:
                    # print(f'{feat}: {feat_data.shape}', flush=True)
                    # if len(encoded_input) > 0:
                    #     print(f'{encoded_input[-1].shape}', flush=True)
                    # encoded_input.append(feat_data)

                    # create mask for -1 values
                    ignore_mask = feat_data == -1

                    # replace -1 with 0 in input data (to avoid errors in the embedding layer)
                    t_adj = torch.where(ignore_mask, torch.zeros_like(feat_data), feat_data)

                    # embed the data
                    embedded = self.embeddings[feat](t_adj)

                    # create mask for embedded data
                    repeated_mask = torch.repeat_interleave(ignore_mask, self.embeddings[feat].emb_dim, dim=1)
                    repeated_mask = torch.reshape(repeated_mask,
                                                  (-1, feat_data.size(dim=1), self.embeddings[feat].emb_dim))

                    # replace embedded '0' values with 0 where the input data was -1
                    embedded_adj = torch.where(repeated_mask, torch.zeros_like(embedded), embedded)

                    # sum the embedded values of each row
                    out_gpu = torch.sum(embedded_adj, dim=1)

                    # assert torch.equal(out_gpu,out_cpu), f"GPU and CPU results are not equal: {out_gpu} vs {out_cpu}"

                    assert out_gpu.shape == (input.shape[0], self.embeddings[feat].emb_dim)

                    encoded_input.append(out_gpu)

            elif feat_type == FeatureType.vector:
                encoded_input.append(feat_data)
            else:
                raise NotImplementedError

        input_enc = torch.cat(encoded_input, dim=1)

        return self.fcout(input_enc)
