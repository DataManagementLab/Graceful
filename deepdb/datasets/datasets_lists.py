from dataclasses import dataclass
from typing import Dict, List


@dataclass()
class SourceDataset:
    name: str
    osf: bool = False


@dataclass()
class Dataset:
    db_name: str
    _source_dataset: str = None
    max_no_joins: int = 4
    downscale: bool = False
    scale_individually: bool = False
    scale: int = 1
    contain_unicode: bool = False

    @property
    def source_dataset(self) -> str:
        if self._source_dataset is None:
            return self.db_name
        return self._source_dataset

    @property
    def data_folder(self) -> str:
        if self.scale is not None and self.scale > 1:
            return f'{self.db_name}_scaled{self.scale}'
        return self.db_name


# datasets that can be downloaded from osf and should be unzipped
source_dataset_list = [
    # original datasets
    SourceDataset('airline', osf=True),
    SourceDataset('imdb', osf=True),
    SourceDataset('ssb', osf=True),
    SourceDataset('tpc_h', osf=True),
    SourceDataset('walmart'),
    SourceDataset('financial'),
    SourceDataset('basketball'),
    SourceDataset('accidents'),
    SourceDataset('movielens'),
    SourceDataset('baseball'),
    SourceDataset('hepatitis'),
    SourceDataset('tournament'),
    SourceDataset('genome'),
    SourceDataset('credit'),
    SourceDataset('employee'),
    SourceDataset('carcinogenesis'),
    SourceDataset('consumer'),
    SourceDataset('geneea'),
    SourceDataset('seznam'),
    SourceDataset('fhnk'),
]

zs_dataset_list = [
    # unscaled
    Dataset('airline', max_no_joins=5),
    Dataset('imdb'),
    Dataset('ssb', max_no_joins=3),
    Dataset('tpc_h', max_no_joins=5),
    Dataset('walmart', max_no_joins=2),

    # scaled batch 1
    Dataset('financial', scale=4),
    Dataset('basketball', scale=200),
    Dataset('accidents', scale=1, contain_unicode=True),
    Dataset('movielens', scale=8),
    Dataset('baseball', scale=10),

    # scaled batch 2
    Dataset('hepatitis', scale=2000),
    Dataset('tournament', scale=50),
    Dataset('credit', scale=5),
    Dataset('employee', scale=3),
    Dataset('consumer', scale=6),
    Dataset('geneea', scale=23, contain_unicode=True),
    Dataset('genome', scale=6),
    Dataset('carcinogenesis', scale=674),
    Dataset('seznam', scale=2),
    Dataset('fhnk', scale=2)
]

zs_dataset_less_scaled_list = [
    # unscaled
    Dataset('airline', max_no_joins=5),
    Dataset('imdb'),
    Dataset('ssb', max_no_joins=3),
    Dataset('tpc_h', max_no_joins=5),
    Dataset('walmart', max_no_joins=2),
    Dataset('seznam'),

    # scaled batch 1
    Dataset('financial', scale=2),
    Dataset('basketball', scale=200),
    Dataset('accidents', scale=1, contain_unicode=True),
    Dataset('movielens', scale=4),
    Dataset('baseball', scale=5),

    # scaled batch 2
    Dataset('hepatitis', scale=1000),
    Dataset('tournament', scale=50),
    Dataset('credit', scale=5),
    Dataset('employee', scale=3),
    Dataset('consumer', scale=6),
    Dataset('geneea', scale=23, contain_unicode=True),
    Dataset('genome', scale=6),
    Dataset('carcinogenesis', scale=374),
    Dataset('fhnk', scale=2)
]

udf_dataset_list = [
    Dataset('airline', max_no_joins=5, downscale=True),
    Dataset('ssb', max_no_joins=3, downscale=True),
    Dataset('tpc_h', max_no_joins=5, downscale=True),
    Dataset('walmart', max_no_joins=2, downscale=True),
    Dataset('financial', downscale=True),
    Dataset('basketball', downscale=True),
    Dataset('accidents', contain_unicode=True, downscale=True),
    Dataset('movielens', downscale=True),
    Dataset('baseball', downscale=True),
    Dataset('hepatitis', downscale=True),
    Dataset('tournament', downscale=True),
    Dataset('credit', downscale=True),
    Dataset('employee', downscale=True),
    Dataset('geneea', downscale=True, contain_unicode=True),
    Dataset('genome', downscale=True),
    Dataset('fhnk', downscale=True)
]

dataset_list_dict: Dict[str, List[Dataset]] = {
    'zs': zs_dataset_list,
    'zs_less_scaled': zs_dataset_less_scaled_list,
    'udf': udf_dataset_list
}
