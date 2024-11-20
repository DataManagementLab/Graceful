import logging
import os
from pathlib import Path

from deepdb.aqp_spn.aqp_spn import AQPSPN
from deepdb.data_preparation.join_data_preparation import JoinDataPreparator
from deepdb.ensemble_compilation.spn_ensemble import SPNEnsemble

logger = logging.getLogger(__name__)

RATIO_MIN_INSTANCE_SLICE = 1 / 100


def check_exists(ensemble_path):
    if os.path.exists(ensemble_path):
        print(f"Skipping training since ensemble {ensemble_path} already exists.")
        return True

    start_path = ensemble_path + '_start'
    if os.path.exists(start_path):
        print(f"Skipping training since ensemble _start {start_path} already exists.")
        return True
    else:
        Path(start_path).touch()

    return False


def create_naive_all_split_ensemble(schema, hdf_path, sample_size, ensemble_path, dataset, bloom_filters,
                                    rdc_threshold, max_table_data, post_sampling_factor, incremental_learning_rate,
                                    min_min_instance_slice=500):
    meta_data_path = hdf_path + '/meta_data.pkl'
    prep = JoinDataPreparator(meta_data_path, schema, max_table_data=max_table_data)
    spn_ensemble = SPNEnsemble(schema)
    ensemble_path += '/ensemble_single_' + dataset + '_' + str(rdc_threshold) + '_' + str(sample_size) + '.pkl'

    if check_exists(ensemble_path):
        return

    logger.info(f"Creating naive ensemble.")

    for table_obj in schema.tables:
        logger.info(f"Learning SPN for {table_obj.table_name}.")

        if incremental_learning_rate > 0:
            df_samples, df_inc_samples, meta_types, null_values, full_join_est = prep.generate_n_samples_with_incremental_part(
                sample_size,
                single_table=table_obj.table_name,
                post_sampling_factor=post_sampling_factor,
                incremental_learning_rate=incremental_learning_rate)
            logger.debug(f"Requested {sample_size} samples and got {len(df_samples)} + {len(df_inc_samples)} "
                         f"(for incremental learning)")
        else:
            df_samples, meta_types, null_values, full_join_est = prep.generate_n_samples(sample_size,
                                                                                         single_table=table_obj.table_name,
                                                                                         post_sampling_factor=post_sampling_factor)

        # learn spn
        aqp_spn = AQPSPN(meta_types, null_values, full_join_est, schema, None, full_sample_size=len(df_samples),
                         table_set={table_obj.table_name}, column_names=list(df_samples.columns),
                         table_meta_data=prep.table_meta_data)
        min_instance_slice = max(RATIO_MIN_INSTANCE_SLICE * min(sample_size, len(df_samples)), min_min_instance_slice)
        logger.debug(f"Using min_instance_slice parameter {min_instance_slice}.")
        logger.info(f"SPN training phase with {len(df_samples)} samples")
        aqp_spn.learn(df_samples.values, min_instances_slice=min_instance_slice, bloom_filters=bloom_filters,
                      rdc_threshold=rdc_threshold)
        if incremental_learning_rate > 0:
            logger.info(f"additional incremental SPN training phase with {len(df_inc_samples)} samples "
                        f"({incremental_learning_rate}%)")
            aqp_spn.learn_incremental(df_inc_samples.values)
        spn_ensemble.add_spn(aqp_spn)

    logger.info(f"Saving ensemble to {ensemble_path}")
    spn_ensemble.save(ensemble_path)


def naive_every_relationship_ensemble(schema, hdf_path, sample_size, ensemble_path, dataset, bloom_filters,
                                      rdc_threshold, max_table_data, post_sampling_factor,
                                      incremental_learning_rate=0, min_min_instance_slice=500):
    meta_data_path = hdf_path + '/meta_data.pkl'
    prep = JoinDataPreparator(meta_data_path, schema, max_table_data=max_table_data)
    spn_ensemble = SPNEnsemble(schema)
    ensemble_path += '/ensemble_relationships_' + dataset + '_' + str(rdc_threshold) + '_' + str(sample_size) + '.pkl'

    if check_exists(ensemble_path):
        return

    logger.info(f"Creating naive ensemble for every relationship.")
    for relationship_obj in schema.relationships:
        logger.info(f"Learning SPN for {relationship_obj.identifier}.")

        if incremental_learning_rate > 0:
            df_samples, df_inc_samples, meta_types, null_values, full_join_est = prep.generate_n_samples_with_incremental_part(
                sample_size, relationship_list=[relationship_obj.identifier], post_sampling_factor=post_sampling_factor,
                incremental_learning_rate=incremental_learning_rate)
        else:
            df_samples, meta_types, null_values, full_join_est = prep.generate_n_samples(
                sample_size, relationship_list=[relationship_obj.identifier], post_sampling_factor=post_sampling_factor)
        logger.debug(f"Requested {sample_size} samples and got {len(df_samples)}")

        # learn spn
        print(f"Full join size {full_join_est}")
        aqp_spn = AQPSPN(meta_types, null_values, full_join_est, schema,
                         [relationship_obj.identifier], full_sample_size=len(df_samples),
                         column_names=list(df_samples.columns), table_meta_data=prep.table_meta_data)
        min_instance_slice = max(RATIO_MIN_INSTANCE_SLICE * min(sample_size, len(df_samples)), min_min_instance_slice)
        logger.debug(f"Using min_instance_slice parameter {min_instance_slice}.")
        logger.info(f"SPN training phase with {len(df_samples)} samples")
        aqp_spn.learn(df_samples.values, min_instances_slice=min_instance_slice, bloom_filters=bloom_filters,
                      rdc_threshold=rdc_threshold)
        if incremental_learning_rate > 0:
            logger.info(f"additional incremental SPN training phase with {len(df_inc_samples)} samples "
                        f"({incremental_learning_rate}%)")
            aqp_spn.learn_incremental(df_inc_samples)
        spn_ensemble.add_spn(aqp_spn)

    logger.info(f"Saving ensemble to {ensemble_path}")
    spn_ensemble.save(ensemble_path)
