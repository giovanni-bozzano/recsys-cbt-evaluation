import numpy as np
import pandas as pd

from Base.DataIO import DataIO
from Base.Evaluation.KFold_SignificanceTest import KFold_SignificanceTest


def get_metric_value_list(result_list, cutoff, metric):
    metric_value_list = [None] * len(result_list)

    for index in range(len(metric_value_list)):
        metric_value = result_list[index][str(cutoff)][metric]
        metric_value_list[index] = metric_value

    return metric_value_list


def read_permutation_results(output_folder_path, n_folds, cutoff, metrics, cbt_models_names, baseline_models_names):
    for cbt_model_name in cbt_models_names:
        test_cbt = KFold_SignificanceTest(n_folds, log_label=cbt_model_name)
        test_baselines = np.empty(len(baseline_models_names), dtype=KFold_SignificanceTest)
        for i_baseline in range(len(baseline_models_names)):
            test_baselines[i_baseline] = KFold_SignificanceTest(n_folds, log_label=baseline_models_names[i_baseline])

        results_cbt = np.empty(n_folds, dtype=dict)
        results_baselines = np.empty(len(baseline_models_names), dtype=np.ndarray)
        for i_baseline in range(len(baseline_models_names)):
            results_baselines[i_baseline] = np.empty(n_folds, dtype=dict)

        dataframe_baselines = np.empty(len(baseline_models_names), dtype=pd.DataFrame)

        for i_fold in range(n_folds):
            dataIO = DataIO(folder_path=output_folder_path + 'results/{}/'.format(i_fold))
            search_metadata = dataIO.load_data(cbt_model_name + '_metadata')
            results_cbt[i_fold] = search_metadata['result_on_last']
            test_cbt.set_results_in_fold(i_fold, search_metadata['result_on_last'][str(cutoff)])

        for i_baseline in range(len(baseline_models_names)):

            for i_fold in range(n_folds):
                dataIO = DataIO(folder_path=output_folder_path + 'results/{}/'.format(i_fold))
                search_metadata = dataIO.load_data(baseline_models_names[i_baseline] + '_metadata')
                results_baselines[i_baseline][i_fold] = search_metadata['result_on_last']
                test_baselines[i_baseline].set_results_in_fold(i_fold, search_metadata['result_on_last'][str(cutoff)])

            dataframe_baselines[i_baseline] = test_baselines[i_baseline].run_significance_test(test_cbt,
                                                                                               dataframe_path=output_folder_path + 'latex_results_significance_' +
                                                                                                              cbt_model_name + '-' + baseline_models_names[i_baseline] + '.csv')

        output_file = open(output_folder_path + 'latex_results_table_' + cbt_model_name + '.txt', 'w')
        output_file.write('\t&' + '\t&'.join(metric for metric in metrics) + '\\\\ \n')

        output_file.write('{} \t&'.format(cbt_model_name))
        for metric in metrics:
            metric_value = get_metric_value_list(results_cbt, cutoff, metric)
            output_file.write('{:.4f} $\pm$ {:.4f}\t{}'.format(np.mean(metric_value), np.std(metric_value), '&' if metric != metrics[-1] else ''))
        output_file.write('\\\\ \n')

        for i_baseline in range(len(baseline_models_names)):
            output_file.write('{} \t&'.format(baseline_models_names[i_baseline]))
            for metric in metrics:
                metric_value = get_metric_value_list(results_baselines[i_baseline], cutoff, metric)
                output_file.write('{:.4f} $\pm$ {:.4f}\t{}'.format(np.mean(metric_value), np.std(metric_value), '&' if metric != metrics[-1] else ''))
            output_file.write('\\\\ \n')
            output_file.write('Is significant\t&')
            for metric in metrics:
                is_significant = dataframe_baselines[i_baseline].loc[metric]['difference_is_significant_pass']
                output_file.write('{}\t{}'.format(is_significant, '&' if metric != metrics[-1] else ''))
            output_file.write('\\\\ \n')

        output_file.close()
