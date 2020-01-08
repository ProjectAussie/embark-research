"""Post-process germline output with {-bits 1} to correct -w_extend bug."""

import sys
import os
import subprocess
import errno
import logging

from contextlib import closing
from multiprocessing import Pool, cpu_count

import begin
import numpy as np
import pandas as pd


# I/O AND PARALLELIZATION
def call_shell_cmd(cmd, shell=False):
    logging.info('call_shell_cmd: {}'.format(cmd))
    try:
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=shell)
    except OSError as err:
        logging.error(cmd)
        raise err
    stdout, stderr = child.communicate()
    returncode = child.returncode
    logging.info('stdout: {}'.format(stdout))
    if stderr:
        logging.error(cmd)
        logging.error(stderr)
    else:
        logging.debug('No stderr')
    return returncode


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def run_plink_with_options(options):
    """Run plink with list of options.

    Assumes you have a working version of plink in PATH.

    """
    return call_shell_cmd(['plink', '--dog'] + options)


class WithExtraArgs(object):
    """Helper class to add extra args to parallelized function.

    adapted from https://chisqr.wordpress.com/2017/03/26/parallel-apply-in-python/
    """

    def __init__(self, func, **args):
        self.func = func
        self.args = args

    def __call__(self, df):
        return self.func(df, **self.args)


def apply_parallel(df_grouped, func, kwargs=None):
    """Parallelize apply over a full dataframe for a function with any number of kwargs, given as a dict.

        dfGrouped -> groupby object with dataframe for each group
        func -> method to parallelize, must return a pandas Series object (each row of output DataFrame)
        kwargs -> Dictionary with keyword arguments for func {"keyword": argument}

    Usage:
        dataframe_from_parallelized_method = applyParallel(dfGrouped, func, kwargs)
    adapted from https://chisqr.wordpress.com/2017/03/26/parallel-apply-in-python/
    """
    with closing(Pool(cpu_count())) as pool:
        if kwargs:
            ret_list = pool.map(
                WithExtraArgs(func, **kwargs),
                [group for name, group in df_grouped])
        else:
            ret_list = pool.map(func, [group for name, group in df_grouped])
        pool.terminate()
    return ret_list


def apply_parallel_series_method(df_grouped, func, columns, kwargs=None):
    """Join into DataFrame results from method that returns Series."""
    return pd.DataFrame(
        apply_parallel(df_grouped, func, kwargs=kwargs), columns=columns).reindex()


def read_match_file(match_file):
    # output format from germline at http://www.cs.columbia.edu/~gusev/germline/
    match_file_columns = ['fid1', 'iid1', 'fid2', 'iid2', 'chr', 'start_bp',
                          'end_bp', 'start_snp', 'end_snp', 'tot_snps',
                          'tract_len', 'unit', 'num_mismat', 'iid1_is_homoz',
                          'iid2_is_homoz']
    match_file_dtypes = {'fid1': 'object', 'iid1': 'object', 'fid2': 'object',
                         'iid2': 'object', 'chr': 'int32', 'start_bp': 'int64',
                         'end_bp': 'int64', 'start_snp': 'object',
                         'end_snp': 'object', 'tot_snps': 'int32',
                         'tract_len': 'float64', 'unit': 'object',
                         'num_mismat': 'int32', 'iid1_is_homoz': 'int32',
                         'iid2_is_homoz': 'int32'}
    match_df = pd.read_csv(
        match_file, delim_whitespace=True, header=None,
        names=match_file_columns, dtype=match_file_dtypes)
    match_df['tract_len_bp'] = match_df['end_bp'] - match_df['start_bp']
    return match_df


# HOMOZYGOSITY POST-PROCESSING METHODS
def assign_groups_to_dataframe_splitting_tracts_on_gaps(
        match_df, minimum_gap_size_kilobases):
    minimum_gap_size_basepairs = minimum_gap_size_kilobases * 1000
    match_df_sorted = match_df.sort_values(by=['chr', 'start_bp', 'end_bp'])

    # identify large gaps by shifting end position column
    match_df_sorted['end_bp_shifted'] = match_df_sorted['end_bp'].shift(1)
    match_df_sorted['inter_tract_distance'] = (
        match_df_sorted['start_bp'] - match_df_sorted['end_bp_shifted'])
    match_df_sorted['is_large_gap'] = np.where(
        match_df_sorted['inter_tract_distance'] > minimum_gap_size_basepairs,
        1, 0)

    # make a new shifted chromosome column to find distance between chromosomes in adjacent tracts, flag new chromosomes
    match_df_sorted['chr_shifted'] = match_df_sorted['chr'].shift(1).fillna(1)
    match_df_sorted['is_new_chr'] = (match_df_sorted['chr'] -
                                     match_df_sorted['chr_shifted'])

    # make a column to group on by finding all places where there is a large
    #   gap or new chromosome
    match_df_sorted['is_start_of_new_group'] = np.where(
        match_df_sorted['is_large_gap'] + match_df_sorted['is_new_chr'] > 0,
        1, 0)
    match_df_sorted['group_label'] = match_df_sorted[
        'is_start_of_new_group'].cumsum()
    return match_df_sorted


def merge_tract_group(tract_subgroup_df, minimum_number_of_markers):
    """Process a group within a group of tracts to merge them together."""
    row_index_for_generating_constants = tract_subgroup_df.index[0]
    index_values = ['chr', 'start_bp', 'end_bp', 'tract_len_bp',
                    'tot_snps']
    markers_per_interval = tract_subgroup_df.sum()['tot_snps']
    if markers_per_interval < minimum_number_of_markers:
        return [np.nan] * len(index_values)

    chromosome = int(tract_subgroup_df.loc[
        row_index_for_generating_constants, 'chr'])
    start_bp = int(tract_subgroup_df.min()['start_bp'])
    end_bp = int(tract_subgroup_df.max()['end_bp'])
    tract_length_basepairs = end_bp - start_bp

    output = [chromosome, start_bp, end_bp, tract_length_basepairs,
              markers_per_interval]
    return output


def merge_and_filter_raw_homozygozity_tracts(
        homozygosity_df, minimum_gap_size_kilobases, minimum_number_of_markers,
        minimum_tract_length_kilobases):
    """Merge and filter raw homozygosity tracts."""
    message = 'More than one ID present in homozygosity dataframe.'
    unique_id1 = homozygosity_df['iid1'].unique()
    unique_id2 = homozygosity_df['iid2'].unique()
    if not ((len(unique_id1) == 1) and (len(unique_id2) == 1) and
            (unique_id1[0] == unique_id2[0])):
        logging.error(message)
        sys.exit(1)

    homozygosity_df_sorted = assign_groups_to_dataframe_splitting_tracts_on_gaps(
        homozygosity_df, minimum_gap_size_kilobases)
    homoz_tract_groups = homozygosity_df_sorted.groupby('group_label')
    columns = ['chr', 'start_bp', 'end_bp', 'tract_len_bp',
               'tot_snps']
    merged_tracts_of_all_lengths = apply_parallel_series_method(
        homoz_tract_groups, merge_tract_group, columns=columns,
        kwargs={"minimum_number_of_markers": minimum_number_of_markers}).dropna().sort_values(
            by=['chr', 'start_bp', 'end_bp']).astype(int)

    # filter tracts on length after gap filling
    minimum_tract_length_basepairs = minimum_tract_length_kilobases * 1000
    merged_tracts_that_meet_tract_length_minimum = merged_tracts_of_all_lengths[
        merged_tracts_of_all_lengths['tract_len_bp'] >=
        minimum_tract_length_basepairs]
    return merged_tracts_that_meet_tract_length_minimum


def process_homozygosity_for_all_individuals_in_match_file(
        homozygosity_df, minimum_gap_size_kilobases, minimum_number_of_markers,
        minimum_tract_length_kilobases):
    matches_grouped_by_iid1 = homozygosity_df.groupby('iid1')

    def process_individual_homozygosity_dataframe((individual, homozygosity_df)):
        logging.info(
            'Processing homozygosity tracts for ' + individual + '...')
        individual_df = merge_and_filter_raw_homozygozity_tracts(
            homozygosity_df, minimum_gap_size_kilobases,
            minimum_number_of_markers,
            minimum_tract_length_kilobases)
        individual_df['id'] = individual
        final_individual_df = individual_df[[
            'id', 'chr', 'start_bp', 'end_bp']]
        return final_individual_df

    all_processed_tract_dfs = map(
        process_individual_homozygosity_dataframe,
        matches_grouped_by_iid1)
    return pd.concat(all_processed_tract_dfs)


@begin.start(auto_convert=True)
@begin.logging
def main(
        match_file,
        output_file=os.path.join(os.getcwd(), 'postprocessed_match_file.tsv'),
        minimum_tract_length_kilobases=500,
        minimum_gap_size_kilobases=50,
        minimum_number_of_markers=41):
    '''A post-processing algorithm for filling gaps and filtering on
    marker counts in merged homozygosity tracts.

    This script is intended to be used on the output of
    GERMLINE (Gusev et al. 2009) run with the following flags |

        -homoz-only, -bits 1, -w_extend |

    and was designed to correct for a bug in germline in which the -w_extend
    algorithm does not work properly for runs of homozygosity; it extends
    beyond the first heterozygous marker to the end of the extended slice.
    Requirements:
        begins==0.9
        numpy==1.11.3
        pandas==0.22.0
    '''
    match_df = read_match_file(match_file)
    processed_matches_df = process_homozygosity_for_all_individuals_in_match_file(
        match_df, minimum_gap_size_kilobases, minimum_number_of_markers,
        minimum_tract_length_kilobases)
    processed_matches_df.to_csv(
        output_file, header=False, index=False, sep='\t')
