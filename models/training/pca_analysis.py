import json
import os.path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def perform_pca_analysis(repr: np.ndarray, labels: np.ndarray, predictions: np.ndarray, label: str, prefix: str,
                         query_stats: Dict, fit_repr: np.ndarray = None, save_path: str = None,
                         return_wandb_image_dict: bool = True, title: str = None, ):
    if fit_repr is None:
        fit_repr = repr

    if len(fit_repr) > 10000:
        fit_repr = fit_repr[:10000]

    if len(repr) > 10000:
        repr = repr[:10000]
        labels = labels[:10000]
        predictions = predictions[:10000]

        for key in query_stats.keys():
            query_stats[key] = query_stats[key][:10000]

    assert len(repr) > 0, 'Representation must not be empty'

    # perform standard scaling
    scalar = StandardScaler()
    scaled_fit_repr = scalar.fit_transform(fit_repr)
    scaled_repr = scalar.transform(repr)

    # perform pca
    pca = PCA(n_components=2)
    pca.fit(scaled_fit_repr)
    repr_reduced_pca = pca.transform(scaled_repr)

    # perform tsne

    default_perplexity = 30
    if len(scaled_repr) < default_perplexity:
        default_perplexity = int(len(scaled_repr) / 2)

    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=default_perplexity)
    try:
        repr_reduced_tsne = tsne.fit_transform(scaled_repr)  # tsne does not support splitting fit and transform
    except ValueError as e:
        print(f'perplexity: {default_perplexity}, repr shape: {scaled_repr.shape}')
        raise e

    # store the results in the query_stats
    query_stats['pca'] = repr_reduced_pca
    query_stats['tsne'] = repr_reduced_tsne

    def plot(repr_reduced, labels, text: str, ax, fig, lognorm=False):
        # plot the labels in the reduced space
        if lognorm:
            scatter = ax.scatter(repr_reduced[:, 0], repr_reduced[:, 1], c=labels, norm=matplotlib.colors.LogNorm())
        else:
            scatter = ax.scatter(repr_reduced[:, 0], repr_reduced[:, 1], c=labels)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.colorbar(scatter, ax=ax)  # add colorbar legend
        ax.set_title(text)

    plt.close()
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    qerror = np.maximum(labels / predictions, predictions / labels)

    plot(repr_reduced_pca, labels, f'PCA (color=label)', ax=axs[0, 0], fig=fig)
    plot(repr_reduced_pca, predictions, f'PCA (color=prediction)', ax=axs[0, 1], fig=fig)
    plot(repr_reduced_pca, qerror, f'PCA (color=qerror)', lognorm=True, ax=axs[0, 2], fig=fig)
    plot(repr_reduced_tsne, labels, f'tSNE (color=label)', ax=axs[1, 0], fig=fig)
    plot(repr_reduced_tsne, predictions, f'tSNE (color=prediction)', ax=axs[1, 1], fig=fig)
    plot(repr_reduced_tsne, qerror, f'tSNE (color=qerror)', lognorm=True, ax=axs[1, 2], fig=fig)
    if title is not None:
        assert title != '', 'Title must not be empty'
        fig.suptitle(title)
    plt.show()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{prefix}_repr_{label}.png'))

        # convert numpy to list for json serialization
        dump_dict = dict()
        for key, value in query_stats.items():
            if isinstance(value, np.ndarray):
                dump_dict[key] = value.tolist()
            else:
                dump_dict[key] = value
        with open(os.path.join(save_path, f'{prefix}_stats_{label}.json'), 'w') as f:
            json.dump(dump_dict, f)

    if return_wandb_image_dict:
        return {f'{prefix}_repr_{label}': wandb.Image(plt)}
    else:
        return None
