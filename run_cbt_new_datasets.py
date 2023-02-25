import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'framework'))

import argparse
import traceback
from functools import partial

import numpy as np
import scipy.sparse as sps
from skopt.space import Categorical, Integer

from framework.Base.Evaluation.Evaluator import EvaluatorHoldout
from framework.Base.NonPersonalizedRecommender import TopPop, Random
from framework.Base.Recommender_utils import check_matrix
from conferences.ECML.lkt_fm import lktfm_names
from conferences.ECML.lkt_fm.lkt_fm import LKTFMRecommender
from conferences.IJCAI.cbt import cbt_names
from conferences.IJCAI.cbt.cbt import CBTRecommender
from conferences.IJCAI.cbt.cbt_similarity import CBTSimilarityRecommender
from conferences.IJCAI.cbt.data import parse_data
from conferences.IJCAI.cbt.data.parse_data import fill_row_miss_value
from conferences.IJCAI.cbt.datasets_provided.amazon_movies_tv_reduced_reader import AmazonMoviesTVReducedReader
from conferences.IJCAI.cbt.datasets_provided.movielens_hetrec_2011_reduced_reader import MovielensHetrec2011ReducedReader
from conferences.IJCAI.cbt.datasets_provided.netflix_prize_reduced_reader import NetflixPrizeReducedReader
from conferences.IJCAI.cbt.evaluator_loss_wrapper import EvaluatorLossWrapper
from conferences.IJCAI.cbt.significance_tests import read_permutation_results
from framework.Data_manager.DataSplitter_k_fold_random import DataSplitter_k_fold_random_fromDataSplitter
from framework.Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from framework.Data_manager.Movielens.Movielens20MReader import Movielens20MReader
from framework.Data_manager.Movielens.MovielensHetrec2011Reader import MovielensHetrec2011Reader
from framework.Data_manager.NetflixPrize.NetflixPrizeReader import NetflixPrizeReader
from framework.Data_manager.Utility import filter_urm
from framework.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from framework.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from framework.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from framework.ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from framework.Utils.ResultFolderLoader import ResultFolderLoader
from framework.Utils.assertions_on_data_for_experiments import assert_disjoint_matrices


def read_data_split_and_search(source_dataset_name,
                               target_dataset_name,
                               flag_baselines_tune=False,
                               flag_cbt_article_default=False,
                               flag_print_results=False):

    output_folder_path = 'result_experiments/new datasets/{}/{}_{}-to-{}/'.format(CONFERENCE_NAME, ALGORITHM_NAME, source_dataset_name, target_dataset_name)

    metric_to_optimize = 'MAP'
    n_folds = 3
    k_out_value = 1
    n_cases = 50
    n_random_starts = 15
    cutoff_list_test = [20]

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    if source_dataset_name == 'movielens':
        source_data_reader = Movielens20MReader()
        source_loaded_dataset = source_data_reader.load_data()
        URM_source = source_loaded_dataset.AVAILABLE_URM['URM_all'].copy()
        for i in range(4):
            URM_source = filter_urm(URM_source, user_min_number_ratings=645, item_min_number_ratings=645)
        URM_source = URM_source[:500, :500]
    elif source_dataset_name == 'netflix':
        source_data_reader = NetflixPrizeReader()
        source_loaded_dataset = source_data_reader.load_data()
        URM_source = source_loaded_dataset.AVAILABLE_URM['URM_all'].copy()
        for i in range(10):
            URM_source = filter_urm(URM_source, user_min_number_ratings=1075, item_min_number_ratings=1075)
        URM_source = URM_source[:500, :500]
    else:
        print('Dataset name not supported, current is {}'.format(source_dataset_name))
        return

    print('Current source dataset is: {}'.format(source_dataset_name))

    n_users, n_items = URM_source.shape
    print("Source rating matrix shape: {0} * {1}".format(n_users, n_items))
    sparsity = float(URM_source.nnz)
    sparsity /= (n_users * n_items)
    print("Density in source dataset: {:.4f}%".format(sparsity * 100.))
    URM_source = fill_row_miss_value(URM_source)
    URM_source = sps.csr_matrix(URM_source).astype(np.float32)
    URM_source.eliminate_zeros()

    # min_user_interactions = k_out_value for test and validation + 1 for training
    if target_dataset_name == 'movielens-dense':
        target_data_reader = MovielensHetrec2011ReducedReader(dense=True, min_user_interactions=k_out_value * 2 + 1)
    elif target_dataset_name == 'movielens-sparse':
        target_data_reader = MovielensHetrec2011ReducedReader(dense=False, min_user_interactions=k_out_value * 2 + 1)
    elif target_dataset_name == 'netflix-dense':
        target_data_reader = NetflixPrizeReducedReader(dense=True, min_user_interactions=k_out_value * 2 + 1)
    elif target_dataset_name == 'netflix-sparse':
        target_data_reader = NetflixPrizeReducedReader(dense=False, min_user_interactions=k_out_value * 2 + 1)
    elif target_dataset_name == 'amazon-dense':
        target_data_reader = AmazonMoviesTVReducedReader(dense=True, min_user_interactions=k_out_value * 2 + 1)
    elif target_dataset_name == 'amazon-sparse':
        target_data_reader = AmazonMoviesTVReducedReader(dense=False, min_user_interactions=k_out_value * 2 + 1)
    else:
        print('Dataset name not supported, current is {}'.format(source_dataset_name))
        return

    dataSplitter_kwargs = {
        'k_out_value': k_out_value,
        'use_validation_set': True,
        'leave_random_out': True
    }

    dataSplitter_k_fold = DataSplitter_k_fold_random_fromDataSplitter(target_data_reader, DataSplitter_leave_k_out,
                                                                      dataSplitter_kwargs=dataSplitter_kwargs,
                                                                      n_folds=n_folds,
                                                                      preload_all=False)

    dataSplitter_k_fold.load_data(save_folder_path=output_folder_path + 'data/folds/')

    for fold_index, dataSplitter_fold in enumerate(dataSplitter_k_fold):

        URM_target_train, URM_target_validation, URM_target_test = dataSplitter_fold.get_holdout_split()
        URM_target_train.eliminate_zeros()
        URM_target_validation.eliminate_zeros()
        URM_target_test.eliminate_zeros()
        URM_target_train = URM_target_train.astype(np.float32)
        URM_target_validation = URM_target_validation.astype(np.float32)
        URM_target_test = URM_target_test.astype(np.float32)

        URM_target_train_last_test = URM_target_train + URM_target_validation

        # Ensure disjoint test-train split
        assert_disjoint_matrices([URM_target_train, URM_target_validation, URM_target_test])
        print('Train set nnz: ' + str(URM_target_train.count_nonzero()))
        print('Validation set nnz: ' + str(URM_target_validation.count_nonzero()))
        print('Test set nnz: ' + str(URM_target_test.count_nonzero()))

        result_folder_path = os.path.join(output_folder_path, 'results/{}/'.format(fold_index))

        evaluator_validation = EvaluatorHoldout(URM_target_validation, cutoff_list=cutoff_list_test, exclude_seen=True)
        evaluator_test = EvaluatorHoldout(URM_target_test, cutoff_list=cutoff_list_test, exclude_seen=True)

        collaborative_algorithm_list = [
            TopPop,
            Random,
            UserKNNCFRecommender
        ]

        if flag_baselines_tune:

            print('')
            print('################################################################################################')
            print('######')
            print('######      BASELINES')
            print('######')
            print('')

            # This will generate a filled target matrix with CBT and then test it against various similarities.
            # It is used to avoid generating the filled target matrix in every iteration.
            # run_fast_tests(source_dataset_name=source_dataset_name, target_dataset_name=target_dataset_name,
            #                evaluator_validation=evaluator_validation, evaluator_test=evaluator_test,
            #                metric_to_optimize=metric_to_optimize, n_cases=n_cases, n_random_starts=n_random_starts, result_folder_path=result_folder_path,
            #                URM_source=URM_source, URM_target_train=URM_target_train, URM_target_validation=URM_target_validation)

            runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                               URM_train=URM_target_train,
                                                               URM_train_last_test=URM_target_train_last_test,
                                                               similarity_type_list=KNN_similarity_to_report_list,
                                                               evaluator_validation=evaluator_validation,
                                                               evaluator_test=evaluator_test,
                                                               metric_to_optimize=metric_to_optimize,
                                                               n_cases=n_cases,
                                                               n_random_starts=n_random_starts,
                                                               parallelizeKNN=False,
                                                               allow_weighting=True,
                                                               output_folder_path=result_folder_path,
                                                               resume_from_saved=True)

            for recommender_class in collaborative_algorithm_list:
                try:
                    runParameterSearch_Collaborative_partial(recommender_class)
                except Exception as e:
                    print('On recommender {} Exception {}'.format(recommender_class, str(e)))
                    traceback.print_exc()

            run_lktfm(evaluator_validation=evaluator_validation, evaluator_test=evaluator_test,
                      metric_to_optimize=metric_to_optimize, n_cases=n_cases, n_random_starts=n_random_starts, result_folder_path=result_folder_path,
                      urm_source=URM_source, urm_target_train=URM_target_train, urm_target_train_last_test=URM_target_train_last_test)

            URM_target_train_averaged = parse_data.fill_row_miss_value(URM_target_train.copy())
            URM_target_train_averaged = sps.csr_matrix(URM_target_train_averaged)
            URM_target_train_averaged.eliminate_zeros()
            URM_target_train_averaged = check_matrix(URM_target_train_averaged, 'csr', dtype=np.float32)

            run_cbt(evaluator_validation=evaluator_validation, evaluator_test=evaluator_test,
                    metric_to_optimize=metric_to_optimize, n_cases=n_cases, n_random_starts=n_random_starts, result_folder_path=result_folder_path,
                    urm_source=URM_target_train_averaged, urm_target_train=URM_target_train, urm_target_train_last_test=URM_target_train_last_test, baseline=True)

        if flag_cbt_article_default:

            print('')
            print('################################################################################################')
            print('######')
            print('######      CBT ALGORITHM')
            print('######')
            print('')

            run_cbt(evaluator_validation=evaluator_validation, evaluator_test=evaluator_test,
                    metric_to_optimize=metric_to_optimize, n_cases=n_cases, n_random_starts=n_random_starts, result_folder_path=result_folder_path,
                    urm_source=URM_source, urm_target_train=URM_target_train, urm_target_train_last_test=URM_target_train_last_test, baseline=False)

        if flag_print_results:

            print('')
            print('################################################################################################')
            print('######')
            print('######      PRINT RESULTS')
            print('######')
            print('')

            file_name = '{}/{}_{}-to-{}_'.format(result_folder_path, ALGORITHM_NAME, source_dataset_name, target_dataset_name)

            result_loader = ResultFolderLoader(result_folder_path,
                                               base_algorithm_list=collaborative_algorithm_list,
                                               other_algorithm_list=[
                                                   cbt_names.CBTRecommenderPearson,
                                                   cbt_names.CBTRecommenderCosine,
                                                   cbt_names.CBTBaselineRecommenderPearson,
                                                   cbt_names.CBTBaselineRecommenderCosine,
                                                   lktfm_names.LKTFMRecommenderPearson,
                                                   lktfm_names.LKTFMRecommenderCosine
                                               ],
                                               KNN_similarity_list=KNN_similarity_to_report_list,
                                               UCM_names_list=None)

            result_loader.generate_latex_results(file_name + 'latex_results.txt',
                                                 metrics_list=['MAP', 'NDCG', 'PRECISION', 'RECALL', 'DIVERSITY_GINI', 'COVERAGE_ITEM'],
                                                 cutoffs_list=[20],
                                                 table_title=None,
                                                 highlight_best=True)

    read_permutation_results(output_folder_path, n_folds, 20, ['MAP', 'NDCG', 'PRECISION', 'RECALL', 'DIVERSITY_GINI', 'COVERAGE_ITEM'],
                             cbt_models_names=[
                                 cbt_names.CBTRecommenderPearson.RECOMMENDER_NAME,
                                 cbt_names.CBTRecommenderCosine.RECOMMENDER_NAME,
                                 lktfm_names.LKTFMRecommenderPearson.RECOMMENDER_NAME,
                                 lktfm_names.LKTFMRecommenderCosine.RECOMMENDER_NAME
                             ],
                             baseline_models_names=[
                                 Random.RECOMMENDER_NAME,
                                 TopPop.RECOMMENDER_NAME,
                                 UserKNNCFRecommender.RECOMMENDER_NAME + '_pearson',
                                 UserKNNCFRecommender.RECOMMENDER_NAME + '_cosine',
                                 cbt_names.CBTBaselineRecommenderPearson.RECOMMENDER_NAME,
                                 cbt_names.CBTBaselineRecommenderCosine.RECOMMENDER_NAME
                             ])


def read_data_split_and_search_single(source_dataset_name,
                                      target_dataset_name,
                                      flag_print_results=False,
                                      flag_random_type='none'):

    if flag_random_type == 'mix':
        output_folder_path = 'result_experiments/full dataset - mixed source/{}/{}_{}-to-{}/'.format(CONFERENCE_NAME, ALGORITHM_NAME, source_dataset_name, target_dataset_name)
    elif flag_random_type == 'random':
        output_folder_path = 'result_experiments/full dataset - random source/{}/{}_{}-to-{}/'.format(CONFERENCE_NAME, ALGORITHM_NAME, source_dataset_name, target_dataset_name)
    elif flag_random_type == 'removal':
        output_folder_path = 'result_experiments/full dataset - source removal/{}/{}_{}-to-{}/'.format(CONFERENCE_NAME, ALGORITHM_NAME, source_dataset_name, target_dataset_name)
    else:
        output_folder_path = 'result_experiments/full dataset - no random/{}/{}_{}-to-{}/'.format(CONFERENCE_NAME, ALGORITHM_NAME, source_dataset_name, target_dataset_name)
    metric_to_optimize = 'MAP'
    n_folds = 10
    k_out_value = 1
    n_cases = 1
    n_random_starts = 1
    cutoff_list_test = [20]

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    if flag_random_type == 'random':
        URM_source = np.random.uniform(0, 5, (500, 500)).astype(dtype=np.float32)
    else:
        if source_dataset_name == 'netflix':
            source_data_reader = NetflixPrizeReader()
            source_loaded_dataset = source_data_reader.load_data()
            URM_source = source_loaded_dataset.AVAILABLE_URM['URM_all'].copy()
            for i in range(10):
                URM_source = filter_urm(URM_source, user_min_number_ratings=1075, item_min_number_ratings=1075)
            URM_source = URM_source[:500, :500]
        else:
            print('Dataset name not supported, current is {}'.format(source_dataset_name))
            return

        print('Current source dataset is: {}'.format(source_dataset_name))

        n_users, n_items = URM_source.shape
        print("Source rating matrix shape: {0} * {1}".format(n_users, n_items))
        sparsity = float(URM_source.nnz)
        sparsity /= (n_users * n_items)
        print("Density in source dataset: {:.4f}%".format(sparsity * 100.))

        URM_source = URM_source.toarray()
        if flag_random_type == 'mix':
            source_columns = URM_source.shape[1]
            URM_source = URM_source.flatten()
            np.random.shuffle(URM_source)
            URM_source = np.reshape(URM_source, (-1, source_columns))

    if target_dataset_name == 'movielens':
        target_data_reader = MovielensHetrec2011Reader()
    else:
        print('Dataset name not supported, current is {}'.format(source_dataset_name))
        return

    dataSplitter_kwargs = {
        'k_out_value': k_out_value,
        'use_validation_set': True,
        'leave_random_out': True
    }

    dataSplitter_k_fold = DataSplitter_k_fold_random_fromDataSplitter(target_data_reader, DataSplitter_leave_k_out,
                                                                      dataSplitter_kwargs=dataSplitter_kwargs,
                                                                      n_folds=n_folds,
                                                                      preload_all=False)

    dataSplitter_k_fold.load_data(save_folder_path=output_folder_path + 'data/folds/')

    for fold_index, dataSplitter_fold in enumerate(dataSplitter_k_fold):

        URM_source_processed = URM_source.copy()

        if flag_random_type == 'removal':
            source_columns = URM_source_processed.shape[1]
            URM_source_processed = URM_source_processed.flatten()
            indices = np.where(URM_source_processed != 0)[0]
            indices = np.random.choice(indices, int(len(indices) / 10), replace=False)
            URM_source_processed[indices] = 0
            URM_source_processed = np.reshape(URM_source_processed, (-1, source_columns))

        URM_source_processed = sps.csr_matrix(URM_source_processed).astype(np.float32)
        URM_source_processed = fill_row_miss_value(URM_source_processed)
        URM_source_processed = sps.csr_matrix(URM_source_processed).astype(np.float32)
        URM_source_processed.eliminate_zeros()

        URM_target_train, URM_target_validation, URM_target_test = dataSplitter_fold.get_holdout_split()
        URM_target_train.eliminate_zeros()
        URM_target_validation.eliminate_zeros()
        URM_target_test.eliminate_zeros()
        URM_target_train = URM_target_train.astype(np.float32)
        URM_target_validation = URM_target_validation.astype(np.float32)
        URM_target_test = URM_target_test.astype(np.float32)

        URM_target_train_last_test = URM_target_train + URM_target_validation

        # Ensure disjoint test-train split
        assert_disjoint_matrices([URM_target_train, URM_target_validation, URM_target_test])
        print('Train set nnz: ' + str(URM_target_train.count_nonzero()))
        print('Validation set nnz: ' + str(URM_target_validation.count_nonzero()))
        print('Test set nnz: ' + str(URM_target_test.count_nonzero()))

        result_folder_path = os.path.join(output_folder_path, 'results/{}/'.format(fold_index))

        evaluator_validation = EvaluatorHoldout(URM_target_validation, cutoff_list=cutoff_list_test, exclude_seen=True)
        evaluator_test = EvaluatorHoldout(URM_target_test, cutoff_list=cutoff_list_test, exclude_seen=True)

        collaborative_algorithm_list = [
            TopPop,
            Random,
            UserKNNCFRecommender
        ]

        print('')
        print('################################################################################################')
        print('######')
        print('######      BASELINES')
        print('######')
        print('')

        runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                           URM_train=URM_target_train,
                                                           URM_train_last_test=URM_target_train_last_test,
                                                           similarity_type_list=KNN_similarity_to_report_list,
                                                           evaluator_validation=evaluator_validation,
                                                           evaluator_test=evaluator_test,
                                                           metric_to_optimize=metric_to_optimize,
                                                           n_cases=n_cases,
                                                           n_random_starts=n_random_starts,
                                                           parallelizeKNN=False,
                                                           allow_weighting=True,
                                                           output_folder_path=result_folder_path,
                                                           resume_from_saved=True)

        for recommender_class in collaborative_algorithm_list:
            try:
                runParameterSearch_Collaborative_partial(recommender_class)
            except Exception as e:
                print('On recommender {} Exception {}'.format(recommender_class, str(e)))
                traceback.print_exc()

        print('')
        print('################################################################################################')
        print('######')
        print('######      LKT ALGORITHM')
        print('######')
        print('')

        run_lktfm_single(evaluator_validation=evaluator_validation, evaluator_test=evaluator_test,
                         metric_to_optimize=metric_to_optimize, n_cases=n_cases, n_random_starts=n_random_starts, result_folder_path=result_folder_path,
                         urm_source=URM_source_processed, urm_target_train=URM_target_train, urm_target_train_last_test=URM_target_train_last_test)

        print('')
        print('################################################################################################')
        print('######')
        print('######      CBT ALGORITHM')
        print('######')
        print('')

        run_cbt_single(evaluator_validation=evaluator_validation, evaluator_test=evaluator_test,
                       metric_to_optimize=metric_to_optimize, n_cases=n_cases, n_random_starts=n_random_starts, result_folder_path=result_folder_path,
                       urm_source=URM_source_processed, urm_target_train=URM_target_train, urm_target_train_last_test=URM_target_train_last_test, baseline=False)

        if flag_print_results:

            print('')
            print('################################################################################################')
            print('######')
            print('######      PRINT RESULTS')
            print('######')
            print('')

            file_name = '{}/{}_{}-to-{}_'.format(result_folder_path, ALGORITHM_NAME, source_dataset_name, target_dataset_name)

            result_loader = ResultFolderLoader(result_folder_path,
                                               base_algorithm_list=collaborative_algorithm_list,
                                               other_algorithm_list=[
                                                   cbt_names.CBTRecommenderCosine,
                                                   lktfm_names.LKTFMRecommenderCosine
                                               ],
                                               KNN_similarity_list=['cosine'],
                                               UCM_names_list=None)

            result_loader.generate_latex_results(file_name + 'latex_results.txt',
                                                 metrics_list=['MAP', 'NDCG', 'PRECISION', 'RECALL', 'DIVERSITY_GINI', 'COVERAGE_ITEM'],
                                                 cutoffs_list=[20],
                                                 table_title=None,
                                                 highlight_best=True)

    read_permutation_results(output_folder_path, n_folds, 20, ['MAP', 'NDCG', 'PRECISION', 'RECALL', 'DIVERSITY_GINI', 'COVERAGE_ITEM'],
                             cbt_models_names=[
                                 cbt_names.CBTRecommenderCosine.RECOMMENDER_NAME,
                                 lktfm_names.LKTFMRecommenderCosine.RECOMMENDER_NAME
                             ],
                             baseline_models_names=[
                                 Random.RECOMMENDER_NAME,
                                 TopPop.RECOMMENDER_NAME,
                                 UserKNNCFRecommender.RECOMMENDER_NAME + '_cosine'
                             ])


def run_cbt(evaluator_validation, evaluator_test,
            metric_to_optimize, n_cases, n_random_starts, result_folder_path,
            urm_source, urm_target_train, urm_target_train_last_test, baseline):

    earlystopping_hyperparameters = {
        'es_validation_every_n': 1,
        'es_stop_on_validation': True,
        'es_lower_validations_allowed': 2000,
        'es_evaluator_object': EvaluatorLossWrapper(),
        'es_validation_metric': 'loss'
    }

    recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[urm_target_train, urm_source, baseline], FIT_KEYWORD_ARGS=earlystopping_hyperparameters)
    recommender_input_args_last_test = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[urm_target_train_last_test, urm_source, baseline], FIT_KEYWORD_ARGS=earlystopping_hyperparameters)
    parameterSearch = SearchBayesianSkopt(CBTRecommender, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

    for similarity_type in KNN_similarity_to_report_list:

        recommender_name = cbt_names.names['baseline' if baseline else 'standard'][similarity_type]

        hyperparameters_range_dictionary = {
            'n_user_clusters': Integer(5, 100),
            'n_item_clusters': Integer(5, 100),
            'epochs': Categorical([20000]),  # maximum_construct_iterations
            'transfer_attempts': Categorical([30]),
            'maximum_fill_iterations': Categorical([100]),
            'topK': Integer(5, 1000),
            'shrink': Integer(0, 1000),
            'similarity': Categorical([similarity_type]),
            'normalize': Categorical([True, False])
        }

        local_parameter_search_space = {**hyperparameters_range_dictionary}
        parameterSearch.search(recommender_input_args=recommender_input_args,
                               recommender_input_args_last_test=recommender_input_args_last_test,
                               parameter_search_space=local_parameter_search_space,
                               metric_to_optimize=metric_to_optimize,
                               n_cases=n_cases,
                               n_random_starts=n_random_starts,
                               output_folder_path=result_folder_path,
                               output_file_name_root=recommender_name,
                               save_model='all',
                               resume_from_saved=True)


def run_cbt_single(evaluator_validation, evaluator_test,
                   metric_to_optimize, n_cases, n_random_starts, result_folder_path,
                   urm_source, urm_target_train, urm_target_train_last_test, baseline):

    earlystopping_hyperparameters = {
        'es_validation_every_n': 1,
        'es_stop_on_validation': True,
        'es_lower_validations_allowed': 2000,
        'es_evaluator_object': EvaluatorLossWrapper(),
        'es_validation_metric': 'loss'
    }

    recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[urm_target_train, urm_source, baseline], FIT_KEYWORD_ARGS=earlystopping_hyperparameters)
    recommender_input_args_last_test = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[urm_target_train_last_test, urm_source, baseline], FIT_KEYWORD_ARGS=earlystopping_hyperparameters)
    parameterSearch = SearchBayesianSkopt(CBTRecommender, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

    recommender_name = cbt_names.names['baseline' if baseline else 'standard']['cosine']

    hyperparameters_range_dictionary = {
        'n_user_clusters': Categorical([5]),
        'n_item_clusters': Categorical([5]),
        'epochs': Categorical([20000]),  # maximum_construct_iterations
        'transfer_attempts': Categorical([30]),
        'maximum_fill_iterations': Categorical([100]),
        'topK': Categorical([550]),
        'shrink': Categorical([120]),
        'similarity': Categorical(['cosine']),
        'normalize': Categorical([False])
    }

    local_parameter_search_space = {**hyperparameters_range_dictionary}
    parameterSearch.search(recommender_input_args=recommender_input_args,
                           recommender_input_args_last_test=recommender_input_args_last_test,
                           parameter_search_space=local_parameter_search_space,
                           metric_to_optimize=metric_to_optimize,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=result_folder_path,
                           output_file_name_root=recommender_name,
                           save_model='all',
                           resume_from_saved=True)


def run_lktfm(evaluator_validation, evaluator_test,
              metric_to_optimize, n_cases, n_random_starts, result_folder_path,
              urm_source, urm_target_train, urm_target_train_last_test):

    earlystopping_hyperparameters = {
        'es_validation_every_n': 1,
        'es_stop_on_validation': True,
        'es_lower_validations_allowed': 2000,
        'es_evaluator_object': EvaluatorLossWrapper(),
        'es_validation_metric': 'loss'
    }

    recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[urm_target_train, urm_source], FIT_KEYWORD_ARGS=earlystopping_hyperparameters)
    recommender_input_args_last_test = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[urm_target_train_last_test, urm_source], FIT_KEYWORD_ARGS=earlystopping_hyperparameters)
    parameterSearch = SearchBayesianSkopt(LKTFMRecommender, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

    for similarity_type in KNN_similarity_to_report_list:
        recommender_name = lktfm_names.names[similarity_type]

        hyperparameters_range_dictionary = {
            'n_clusters': Integer(5, 100),
            'epochs': Categorical([10]),  # maximum_construct_iterations
            'transfer_attempts': Categorical([30]),
            'maximum_fill_iterations': Categorical([100]),
            'topK': Integer(5, 1000),
            'shrink': Integer(0, 1000),
            'similarity': Categorical([similarity_type]),
            'normalize': Categorical([True, False])
        }

        local_parameter_search_space = {**hyperparameters_range_dictionary}
        parameterSearch.search(recommender_input_args=recommender_input_args,
                               recommender_input_args_last_test=recommender_input_args_last_test,
                               parameter_search_space=local_parameter_search_space,
                               metric_to_optimize=metric_to_optimize,
                               n_cases=n_cases,
                               n_random_starts=n_random_starts,
                               output_folder_path=result_folder_path,
                               output_file_name_root=recommender_name,
                               save_model='all',
                               resume_from_saved=True)


def run_lktfm_single(evaluator_validation, evaluator_test,
                     metric_to_optimize, n_cases, n_random_starts, result_folder_path,
                     urm_source, urm_target_train, urm_target_train_last_test):

    earlystopping_hyperparameters = {
        'es_validation_every_n': 1,
        'es_stop_on_validation': True,
        'es_lower_validations_allowed': 2000,
        'es_evaluator_object': EvaluatorLossWrapper(),
        'es_validation_metric': 'loss'
    }

    recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[urm_target_train, urm_source], FIT_KEYWORD_ARGS=earlystopping_hyperparameters)
    recommender_input_args_last_test = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[urm_target_train_last_test, urm_source], FIT_KEYWORD_ARGS=earlystopping_hyperparameters)
    parameterSearch = SearchBayesianSkopt(LKTFMRecommender, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

    recommender_name = lktfm_names.names['cosine']

    hyperparameters_range_dictionary = {
        'n_clusters': Categorical([5]),
        'epochs': Categorical([10]),  # maximum_construct_iterations
        'transfer_attempts': Categorical([3]),
        'maximum_fill_iterations': Categorical([100]),
        'topK': Categorical([550]),
        'shrink': Categorical([120]),
        'similarity': Categorical(['cosine']),
        'normalize': Categorical([False])
    }

    local_parameter_search_space = {**hyperparameters_range_dictionary}
    parameterSearch.search(recommender_input_args=recommender_input_args,
                           recommender_input_args_last_test=recommender_input_args_last_test,
                           parameter_search_space=local_parameter_search_space,
                           metric_to_optimize=metric_to_optimize,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           output_folder_path=result_folder_path,
                           output_file_name_root=recommender_name,
                           save_model='all',
                           resume_from_saved=True)


def run_fast_tests(source_dataset_name, target_dataset_name,
                   evaluator_validation, evaluator_test,
                   metric_to_optimize, n_cases, n_random_starts, result_folder_path,
                   URM_source, URM_target_train, URM_target_validation):

    cbt = CBTRecommender(URM_target_train, URM_source, baseline=False)
    cbt.fit(es_evaluator_object=EvaluatorLossWrapper())
    URM_target_train_filled = cbt.URM_target_train_filled.toarray()
    URM_target_train_filled = sps.csr_matrix(URM_target_train_filled)
    URM_target_train_filled.eliminate_zeros()

    cbt = CBTRecommender(URM_target_train + URM_target_validation, URM_source, baseline=False)
    cbt.fit(es_evaluator_object=EvaluatorLossWrapper())
    URM_target_train_filled_last_test = cbt.URM_target_train_filled.toarray()
    URM_target_train_filled_last_test = sps.csr_matrix(URM_target_train_filled_last_test)
    URM_target_train_filled_last_test.eliminate_zeros()

    recommender_input_args = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_target_train, URM_target_train_filled])
    recommender_input_args_last_test = SearchInputRecommenderArgs(CONSTRUCTOR_POSITIONAL_ARGS=[URM_target_train + URM_target_validation, URM_target_train_filled_last_test])
    parameterSearch = SearchBayesianSkopt(CBTSimilarityRecommender, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

    similarity_types = [
        'pearson',
        'cosine'
    ]
    for similarity_type in similarity_types:
        recommender_name = cbt_names.names['similarity'][similarity_type]

        hyperparameters_range_dictionary = {
            'topK': Integer(5, 1000),
            'shrink': Integer(0, 1000),
            'similarity': Categorical([similarity_type]),
            'normalize': Categorical([True, False]),
        }

        local_parameter_search_space = {**hyperparameters_range_dictionary}
        parameterSearch.search(recommender_input_args,
                               parameter_search_space=local_parameter_search_space,
                               n_cases=n_cases,
                               n_random_starts=n_random_starts,
                               resume_from_saved=True,
                               output_folder_path=result_folder_path,
                               output_file_name_root=recommender_name,
                               metric_to_optimize=metric_to_optimize,
                               recommender_input_args_last_test=recommender_input_args_last_test)

    file_name = '{}/{}_{}-to-{}_'.format(result_folder_path, ALGORITHM_NAME, source_dataset_name, target_dataset_name)

    result_loader = ResultFolderLoader(result_folder_path,
                                       base_algorithm_list=[],
                                       other_algorithm_list=[
                                           cbt_names.CBTSimilarityRecommenderPearson,
                                           cbt_names.CBTSimilarityRecommenderCosine
                                       ],
                                       KNN_similarity_list=similarity_types,
                                       UCM_names_list=None)

    result_loader.generate_latex_results(file_name + 'fast_latex_results.txt',
                                         metrics_list=['MAP', 'NDCG', 'PRECISION', 'RECALL', 'DIVERSITY_GINI', 'COVERAGE_ITEM'],
                                         cutoffs_list=[20],
                                         table_title=None,
                                         highlight_best=True)


if __name__ == '__main__':

    ALGORITHM_NAME = 'cbt'
    CONFERENCE_NAME = 'ijcai'

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune', help='Baseline hyperparameter search', type=bool, default=True)
    parser.add_argument('-a', '--cbt_article_default', help='Train the CBT model with hyperparameter search', type=bool, default=True)
    parser.add_argument('-p', '--print_results', help='Print results', type=bool, default=True)
    parser.add_argument('-r', '--random_type', help='Randomization type', type=str, default='none')

    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = [
        'pearson',
        'cosine'
    ]

    dataset_list = [
        # source, target
        ['netflix', 'amazon-sparse'],
        ['netflix', 'amazon-dense'],
        ['netflix', 'movielens-sparse'],
        ['netflix', 'movielens-dense'],
        ['movielens', 'amazon-sparse'],
        ['movielens', 'amazon-dense'],
        ['movielens', 'netflix-sparse'],
        ['movielens', 'netflix-dense']
    ]

    read_data_split_and_search_single('netflix', 'movielens',
                                      flag_print_results=input_flags.print_results,
                                      flag_random_type=input_flags.random_type)
    exit(1)

    for datasets in dataset_list:
        read_data_split_and_search(datasets[0], datasets[1],
                                   flag_baselines_tune=input_flags.baseline_tune,
                                   flag_cbt_article_default=input_flags.cbt_article_default,
                                   flag_print_results=input_flags.print_results)
