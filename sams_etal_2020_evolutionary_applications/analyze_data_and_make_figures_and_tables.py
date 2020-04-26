#! /usr/bin/env python3
import os
import sys
import errno
import copy
from functools import reduce

import begin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import dabest

print(f"We're using DABEST v{dabest.__version__}")


### GLOBALS ###
COLORS = [
    "#FFCE34",  # primary
    "#57C1E9",
    "#F15B5C",
    "#A776A6",  # secondary
    "#F99F1E",
    "#95D600",
    "#CFD21C",  # tertiary
    "#FCB92A",
    "#73CB7F",
    "#7B9FCB",
    "#CF677E",
    "#F57A40",
]

ones = list(range(900, ((120 * 25) - 1), 120))
twos = [160, 200, 280, 320, 400, 440, 520, 560, 640, 680, 760, 800]
threes = list(range(30, 119, 30))
MS_POSITIONS = pd.Series(threes + twos + ones)

ORDERED_SLIM_PARAMETERS = [
    "model_type",
    "population_size_burn_in",
    "population_size_bottleneck",
    "population_size_mate_choice",
    "number_of_genome_wide_founder_haplotypes",
    "number_of_ms_founder_alleles",
    "generations_in_burn_in",
    "generations_in_bottleneck",
    "generations_in_mate_choice",
    "mating_pool_size",
    "maximum_number_of_matings",
    "proportion_of_mates_for_layered_mate_choice",
]

VALID_MODEL_TYPES = [
    "random",
    "pedigree",
    "gw_het",
    "gw_rel",
    "ms33_het",
    "ms33_ir",
    "ms33_agr",
    "ms33_ir_agr",
]
PRINTED_NAMES_OF_MODELS = [
    "Random",
    "Pedigree",
    "GW Het",
    "GW Rel",
    "MS33 Het",
    "MS33 IR",
    "MS33 AGR",
    "MS33 IR + AGR",
]
LONG_PRINTED_NAMES_OF_MODELS = [
    "Random",
    "Pedigree",
    "Genome-Wide Heterozygosity",
    "Genome-Wide Relatedness",
    "MS33 Heterozygosity",
    "MS33 Internal Relatedness",
    "MS33 Avg. Genetic Relatedness",
    "MS33 IR + AGR",
]
MODEL_NAME_DICT = {
    VALID_MODEL_TYPES[i]: PRINTED_NAMES_OF_MODELS[i]
    for i in range(len(VALID_MODEL_TYPES))
}
LONG_MODEL_NAME_DICT = {
    VALID_MODEL_TYPES[i]: LONG_PRINTED_NAMES_OF_MODELS[i]
    for i in range(len(VALID_MODEL_TYPES))
}

MS_MODEL_LIST = ["random", "pedigree", "ms33_het", "ms33_ir", "ms33_agr", "ms33_ir_agr"]
WHOLE_GENOME_MODEL_LIST = ["random", "pedigree", "gw_het", "gw_rel"]

STATISTIC_TYPES = [
    "gw_heterozygosity",
    "ms_heterozygosity",
    "gw_richness",
    "ms_richness",
    "gw_over_5_percent_richness",
    "ms_over_5_percent_richness",
    "coi",
    "ir",
]

STATISTIC_TYPE_TO_FILE_SUFFIX_DICT = {
    "gw_heterozygosity": "gw_heterozygosity",
    "ms_heterozygosity": "ms_heterozygosity",
    "gw_richness": "gw_richness",
    "ms_richness": "ms_richness",
    "gw_over_5_percent_richness": "gw_richness",
    "ms_over_5_percent_richness": "ms_richness",
    "coi": "gw_coi",
    "ir": "ms_internal_relatedness",
}

STATISTIC_TYPE_TO_COLUMN_NAME_PREFIX_DICT = {
    "gw_heterozygosity": "heterozygosity_",
    "ms_heterozygosity": "heterozygosity_",
    "gw_richness": "total_richness_",
    "ms_richness": "total_richness_",
    "gw_over_5_percent_richness": "alleles_over_five_percent_richness_",
    "ms_over_5_percent_richness": "alleles_over_five_percent_richness_",
    "coi": "mean_coi_",
    "ir": "mean_ir_",
}

STATISTIC_TYPE_TO_Y_AXIS_LABEL_PREFIX = {
    "gw_heterozygosity": "Mean Heterozygosity ",
    "ms_heterozygosity": "Mean Heterozygosity ",
    "gw_richness": "Mean Richness ",
    "ms_richness": "Mean Richness ",
    "gw_over_5_percent_richness": "Mean Richness (>5%) ",
    "ms_over_5_percent_richness": "Mean Richness (>5%) ",
    "coi": "Mean COI ",
    "ir": "Mean IR ",
}

DABEST_STAT_TYPES = ["within_gen", "change_between_gens", "pct_change_between_gens"]

MATE_CHOICE_GENERATION_TO_FILE_SUFFIX_DICT = {
    0: "start",
    5: "gen_5",
    10: "gen_10",
    15: "gen_15",
    20: "gen_20",
    25: "gen_25",
    30: "gen_30",
    35: "gen_35",
    40: "end",
}
FILE_EXTENSIONS_FOR_MATE_CHOICE_GENERATIONS = [
    MATE_CHOICE_GENERATION_TO_FILE_SUFFIX_DICT[gen]
    for gen in sorted(MATE_CHOICE_GENERATION_TO_FILE_SUFFIX_DICT.keys())
]


PARAMETER_SETS_DICT = {
    1: {
        "population_size_burn_in": 2500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    2: {
        "population_size_burn_in": 1000,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    3: {
        "population_size_burn_in": 500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    4: {
        "population_size_burn_in": 2500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 50,
        "number_of_ms_founder_alleles": 50,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    5: {
        "population_size_burn_in": 1000,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 50,
        "number_of_ms_founder_alleles": 50,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    6: {
        "population_size_burn_in": 500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 50,
        "number_of_ms_founder_alleles": 50,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    7: {
        "population_size_burn_in": 2500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 0,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    8: {
        "population_size_burn_in": 1000,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 0,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    9: {
        "population_size_burn_in": 500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 0,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    10: {
        "population_size_burn_in": 2500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 50,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    11: {
        "population_size_burn_in": 1000,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 50,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    12: {
        "population_size_burn_in": 500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 50,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    13: {
        "population_size_burn_in": 2500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 5,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    14: {
        "population_size_burn_in": 1000,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 5,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    15: {
        "population_size_burn_in": 500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 5,
        "mating_pool_size": 50,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    16: {
        "population_size_burn_in": 2500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 25,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    17: {
        "population_size_burn_in": 1000,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 25,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    18: {
        "population_size_burn_in": 500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 25,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    19: {
        "population_size_burn_in": 2500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 10,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    20: {
        "population_size_burn_in": 1000,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 10,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
    21: {
        "population_size_burn_in": 500,
        "population_size_bottleneck": 50,
        "population_size_mate_choice": 200,
        "number_of_genome_wide_founder_haplotypes": 20,
        "number_of_ms_founder_alleles": 20,
        "generations_in_burn_in": 200,
        "generations_in_bottleneck": 5,
        "generations_in_mate_choice": 40,
        "maximum_number_of_matings": 200,
        "mating_pool_size": 10,
        "proportion_of_mates_for_layered_mate_choice": 0.5,
    },
}


### IMPORT AND PARSE DATA ###
def merge_dictionaries(a, b, path=None):
    copy_of_a = copy.deepcopy(a)  # we don't want to mutate the input object
    if path is None:
        path = []
    for key in b:
        if key in copy_of_a:
            if isinstance(copy_of_a[key], dict) and isinstance(b[key], dict):
                merge_dictionaries(copy_of_a[key], b[key], path + [str(key)])
            elif copy_of_a[key] == b[key]:
                pass
            else:
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            copy_of_a[key] = b[key]
    return copy_of_a


def make_dir_if_does_not_exist(directory):
    try:
        os.makedirs(directory)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            pass
        else:
            raise


def get_ordered_parameter_folder(slim_parameters_dict, with_model_type=True):
    ordered_params = (
        ORDERED_SLIM_PARAMETERS
        if with_model_type
        else [p for p in ORDERED_SLIM_PARAMETERS if p != "model_type"]
    )
    param_folders = [
        f"{param_name}/{slim_parameters_dict[param_name]}"
        for param_name in ordered_params
    ]
    return "/".join(param_folders)


def get_partial_simulation_outputs_path_for_simulation_parameters_formatted_for_simulation_number(
    population_size_burn_in,
    population_size_bottleneck,
    population_size_mate_choice,
    number_of_genome_wide_founder_haplotypes,
    number_of_ms_founder_alleles,
    generations_in_burn_in,
    generations_in_bottleneck,
    generations_in_mate_choice,
    mating_pool_size,
    maximum_number_of_matings,
    proportion_of_mates_for_layered_mate_choice,
    model_type,
):
    return get_ordered_parameter_folder(locals()) + "/{}/"


def get_full_simulation_outputs_path(
    parameter_dict_without_model_type,
    model_type,
    simulation_index,
    simulation_output_file,
    simulation_results_root_directory,
):
    all_possible_parameters_for_path = merge_dictionaries(
        parameter_dict_without_model_type, {"model_type": model_type}
    )
    valid_slim_parameters_for_path = {
        key: all_possible_parameters_for_path[key] for key in ORDERED_SLIM_PARAMETERS
    }
    simulation_result_file_path = os.path.join(
        simulation_results_root_directory,
        get_partial_simulation_outputs_path_for_simulation_parameters_formatted_for_simulation_number(
            **valid_slim_parameters_for_path
        ).format(
            simulation_index
        )
        + simulation_output_file,
    )
    return simulation_result_file_path


def get_output_file_name(
    parameter_set_dict, statistic_type, output_file_root_dir, output_file_suffix
):
    param_folders_path = get_ordered_parameter_folder(
        parameter_set_dict, with_model_type=False
    )
    make_dir_if_does_not_exist(
        f"{output_file_root_dir}/{param_folders_path}/{output_file_suffix}"
    )
    return f"{output_file_root_dir}/{param_folders_path}/{output_file_suffix}/{statistic_type}"


### SUMMARIZE DATA ###


def get_average_heterozygosity_across_simulations_and_parameter_sets(
    model_type_list,
    start_simulation_index,
    end_simulation_index,
    simulation_results_file_type,
    simulation_results_root_directory,
    analysis_data_output_dir,
    get_data_from_results_files=False,
):
    simulation_index_range = list(
        range(start_simulation_index, end_simulation_index + 1)
    )
    paremeter_set_index_string = "_".join(
        sorted([str(i) for i in list(PARAMETER_SETS_DICT.keys())])
    )
    all_model_dfs = []

    if get_data_from_results_files:
        print("Getting Heterozygosity data from results files...")
        for index in sorted(PARAMETER_SETS_DICT.keys()):
            print(f"Parameter Set: {index}")
            parameter_set_sub_dict = PARAMETER_SETS_DICT[index]
            for model in model_type_list:
                print(f"Model Type: {model}")
                heterozygosity_df = pd.DataFrame(
                    [],
                    columns=[
                        f"heterozygosity_{file_suffix}"
                        for file_suffix in FILE_EXTENSIONS_FOR_MATE_CHOICE_GENERATIONS
                    ],
                    index=simulation_index_range,
                    dtype=float,
                )
                for simulation_index in simulation_index_range:
                    newline = "\n" if simulation_index == end_simulation_index else ""
                    sys.stdout.write(
                        "\r" + f"Simulation Index: {simulation_index}" + newline
                    )
                    sys.stdout.flush()
                    for file_suffix in FILE_EXTENSIONS_FOR_MATE_CHOICE_GENERATIONS:
                        file_name_from_mate_choice_gen = (
                            f"{simulation_results_file_type}_{file_suffix}.txt"
                        )
                        file_path = get_full_simulation_outputs_path(
                            parameter_set_sub_dict,
                            model_type=model,
                            simulation_index=simulation_index,
                            simulation_output_file=file_name_from_mate_choice_gen,
                            simulation_results_root_directory=simulation_results_root_directory,
                        )
                        try:
                            file_object = open(file_path, "r")
                        except OSError:
                            print(
                                f"{file_name_from_mate_choice_gen} for simulation {simulation_index} and model {model} was not found!"
                            )
                            continue
                        heterozygosity = float(file_object.read().strip())
                        heterozygosity_df.loc[
                            simulation_index, f"heterozygosity_{file_suffix}"
                        ] = heterozygosity

                heterozygosity_df["model_type"] = model
                heterozygosity_df["parameter_set_index"] = index
                all_model_dfs.append(heterozygosity_df)
        final_df = (
            pd.concat(all_model_dfs)
            .reset_index()
            .rename(columns={"index": "simulation_index"})
        )
        ordered_generation_columns = [
            "model_type",
            "parameter_set_index",
            "simulation_index",
            "heterozygosity_start",
            "heterozygosity_gen_5",
            "heterozygosity_gen_10",
            "heterozygosity_gen_15",
            "heterozygosity_gen_20",
            "heterozygosity_gen_25",
            "heterozygosity_gen_30",
            "heterozygosity_gen_35",
            "heterozygosity_end",
        ]
        final_df[ordered_generation_columns].to_csv(
            analysis_data_output_dir
            + f"/{simulation_results_file_type}_{paremeter_set_index_string}.csv",
            header=True,
            index=False,
            na_rep="NaN",
            float_format="%.4f",
        )
    else:
        dtypes = {
            "simulation_index": "int32",
            "heterozygosity_start": "float64",
            "heterozygosity_gen_5": "float64",
            "heterozygosity_gen_10": "float64",
            "heterozygosity_gen_15": "float64",
            "heterozygosity_gen_20": "float64",
            "heterozygosity_gen_25": "float64",
            "heterozygosity_gen_30": "float64",
            "heterozygosity_gen_35": "float64",
            "heterozygosity_end": "float64",
            "model_type": "object",
            "parameter_set_index": "int32",
        }
        final_df = pd.read_csv(
            analysis_data_output_dir
            + f"/{simulation_results_file_type}_{paremeter_set_index_string}.csv",
            header=0,
            dtype=dtypes,
        )
    return final_df[final_df["model_type"].isin(model_type_list)]


def get_per_locus_and_per_individual_average_richness_across_simulations(
    model_type_list,
    start_simulation_index,
    end_simulation_index,
    simulation_results_file_type,
    simulation_results_root_directory,
    analysis_data_output_dir,
    get_data_from_results_files=False,
):
    simulation_index_range = list(
        range(start_simulation_index, end_simulation_index + 1)
    )
    paremeter_set_index_string = "_".join(
        sorted([str(i) for i in list(PARAMETER_SETS_DICT.keys())])
    )
    per_locus_all_model_dfs = []
    per_individual_all_model_dfs = []
    if get_data_from_results_files:
        print("Getting Heterozygosity data from results files...")
        for index in sorted(PARAMETER_SETS_DICT.keys()):
            print(f"Parameter Set: {index}")
            parameter_set_sub_dict = PARAMETER_SETS_DICT[index]
            for model in model_type_list:
                print(f"Model Type: {model}")
                per_locus_model_dfs = []
                per_individual_model_dfs = []
                for simulation_index in simulation_index_range:
                    newline = "\n" if simulation_index == end_simulation_index else ""
                    sys.stdout.write(
                        "\r" + f"Simulation Index: {simulation_index}" + newline
                    )
                    sys.stdout.flush()
                    richness_dfs_for_simulation_index = []
                    for file_suffix in FILE_EXTENSIONS_FOR_MATE_CHOICE_GENERATIONS:
                        file_name_from_mate_choice_gen = (
                            f"{simulation_results_file_type}_{file_suffix}.txt"
                        )
                        file_path = get_full_simulation_outputs_path(
                            parameter_set_sub_dict,
                            model_type=model,
                            simulation_index=simulation_index,
                            simulation_output_file=file_name_from_mate_choice_gen,
                            simulation_results_root_directory=simulation_results_root_directory,
                        )
                        try:
                            file_object = open(file_path, "r")
                        except OSError:
                            print(
                                f"{file_name_from_mate_choice_gen} for simulation {simulation_index} and model {model} was not found!"
                            )
                            continue
                        richness_for_mate_choice_gen_df = pd.read_csv(
                            file_object,
                            header=None,
                            sep="\t",
                            names=[
                                "position",
                                f"total_richness_{file_suffix}",
                                f"alleles_over_five_percent_richness_{file_suffix}",
                            ],
                        )
                        richness_dfs_for_simulation_index.append(
                            richness_for_mate_choice_gen_df
                        )
                    merged_richness_df = reduce(
                        lambda left, right: pd.merge(
                            left, right, on="position", how="inner"
                        ),
                        richness_dfs_for_simulation_index,
                    )
                    per_locus_model_dfs.append(merged_richness_df)
                    single_simulation_df = pd.DataFrame(merged_richness_df.mean()).T
                    single_simulation_df["simulation_index"] = simulation_index
                    per_individual_model_dfs.append(single_simulation_df)
                # per locus wrap up
                per_locus_model_across_simulation_runs_df = pd.concat(
                    per_locus_model_dfs
                )
                mean_richness_for_model_df = per_locus_model_across_simulation_runs_df.groupby(
                    "position"
                ).mean()
                mean_richness_for_model_df["model_type"] = model
                mean_richness_for_model_df["parameter_set_index"] = index
                per_locus_all_model_dfs.append(mean_richness_for_model_df)

                # per individual wrap up
                per_individual_model_across_simulation_runs_df = pd.concat(
                    per_individual_model_dfs
                )
                per_individual_model_across_simulation_runs_df["model_type"] = model
                per_individual_model_across_simulation_runs_df[
                    "parameter_set_index"
                ] = index
                per_individual_all_model_dfs.append(
                    per_individual_model_across_simulation_runs_df
                )

        per_locus_final_df = pd.concat(per_locus_all_model_dfs)
        per_individual_final_df = pd.concat(per_individual_all_model_dfs).drop(
            "position", axis=1
        )
        ordered_generation_columns = [
            "model_type",
            "parameter_set_index",
            "total_richness_start",
            "total_richness_gen_5",
            "total_richness_gen_10",
            "total_richness_gen_15",
            "total_richness_gen_20",
            "total_richness_gen_25",
            "total_richness_gen_30",
            "total_richness_gen_35",
            "total_richness_end",
            "alleles_over_five_percent_richness_start",
            "alleles_over_five_percent_richness_gen_5",
            "alleles_over_five_percent_richness_gen_10",
            "alleles_over_five_percent_richness_gen_15",
            "alleles_over_five_percent_richness_gen_20",
            "alleles_over_five_percent_richness_gen_25",
            "alleles_over_five_percent_richness_gen_30",
            "alleles_over_five_percent_richness_gen_35",
            "alleles_over_five_percent_richness_end",
        ]
        per_locus_final_df[ordered_generation_columns].to_csv(
            analysis_data_output_dir
            + f"/{simulation_results_file_type}_per_locus_{paremeter_set_index_string}.csv",
            header=True,
            index=True,
            na_rep="NaN",
            float_format="%.4f",
        )
        per_individual_final_df[ordered_generation_columns].to_csv(
            analysis_data_output_dir
            + f"/{simulation_results_file_type}_per_individual_{paremeter_set_index_string}.csv",
            header=True,
            index=False,
            na_rep="NaN",
            float_format="%.4f",
        )
    else:
        dtypes = {
            "position": "int32",
            "total_richness_start": "float64",
            "alleles_over_five_percent_richness_start": "float64",
            "total_richness_gen_5": "float64",
            "alleles_over_five_percent_richness_gen_5": "float64",
            "total_richness_gen_10": "float64",
            "alleles_over_five_percent_richness_gen_10": "float64",
            "total_richness_gen_15": "float64",
            "alleles_over_five_percent_richness_gen_15": "float64",
            "total_richness_gen_20": "float64",
            "alleles_over_five_percent_richness_gen_20": "float64",
            "total_richness_gen_25": "float64",
            "alleles_over_five_percent_richness_gen_25": "float64",
            "total_richness_gen_30": "float64",
            "alleles_over_five_percent_richness_gen_30": "float64",
            "total_richness_gen_35": "float64",
            "alleles_over_five_percent_richness_gen_35": "float64",
            "total_richness_end": "float64",
            "alleles_over_five_percent_richness_end": "float64",
            "model_type": "object",
            "parameter_set_index": "int32",
        }
        per_locus_final_df = pd.read_csv(
            analysis_data_output_dir
            + f"/{simulation_results_file_type}_per_locus_{paremeter_set_index_string}.csv",
            header=0,
            index_col="position",
            dtype=dtypes,
        )
        per_individual_final_df = pd.read_csv(
            analysis_data_output_dir
            + f"/{simulation_results_file_type}_per_individual_{paremeter_set_index_string}.csv",
            header=0,
            dtype=dtypes,
        )
    per_locus_filtered_to_model_type_list_df = per_locus_final_df[
        per_locus_final_df["model_type"].isin(model_type_list)
    ]
    per_individual_filtered_to_model_type_list_df = per_individual_final_df[
        per_individual_final_df["model_type"].isin(model_type_list)
    ]
    return (
        per_locus_filtered_to_model_type_list_df,
        per_individual_filtered_to_model_type_list_df,
    )


def get_average_coi_across_simulations(
    model_type_list,
    start_simulation_index,
    end_simulation_index,
    simulation_results_file_type,
    simulation_results_root_directory,
    analysis_data_output_dir,
    get_data_from_results_files=False,
):
    simulation_index_range = list(
        range(start_simulation_index, end_simulation_index + 1)
    )
    paremeter_set_index_string = "_".join(
        sorted([str(i) for i in list(PARAMETER_SETS_DICT.keys())])
    )
    all_model_dfs = []
    if get_data_from_results_files:
        print("Getting COI data from S3...")
        for index in sorted(PARAMETER_SETS_DICT.keys()):
            print(f"Parameter Set: {index}")
            parameter_set_sub_dict = PARAMETER_SETS_DICT[index]
            for model in model_type_list:
                print(f"Model Type: {model}")
                model_results_list = []
                for simulation_index in simulation_index_range:
                    newline = "\n" if simulation_index == end_simulation_index else ""
                    sys.stdout.write(
                        "\r" + f"Simulation Index: {simulation_index}" + newline
                    )
                    sys.stdout.flush()
                    coi_dfs = []
                    for file_suffix in FILE_EXTENSIONS_FOR_MATE_CHOICE_GENERATIONS:
                        file_name_from_mate_choice_gen = (
                            f"{simulation_results_file_type}_{file_suffix}.txt"
                        )
                        file_path = get_full_simulation_outputs_path(
                            parameter_set_sub_dict,
                            model_type=model,
                            simulation_index=simulation_index,
                            simulation_output_file=file_name_from_mate_choice_gen,
                            simulation_results_root_directory=simulation_results_root_directory,
                        )
                        try:
                            file_object = open(file_path, "r")
                        except OSError:
                            print(
                                f"{file_name_from_mate_choice_gen} for simulation {simulation_index} and model {model} was not found!"
                            )
                            continue
                        coi = pd.read_csv(
                            file_object,
                            header=None,
                            sep="\t",
                            names=[f"mean_coi_{file_suffix}"],
                        )  # in reality these aren't means, but using this name to avoid having to change names in the .mean df below.
                        coi_dfs.append(coi)
                    coi_dfs_merged = reduce(
                        lambda left, right: pd.merge(
                            left, right, left_index=True, right_index=True
                        ),
                        coi_dfs,
                    )  # merging dataframes by index which represents individuals in the population, order doesn't matter since we're averaging.
                    mean_coi_df = pd.DataFrame(coi_dfs_merged.mean()).T
                    ordered_generation_columns = mean_coi_df.columns.tolist()
                    mean_coi_df["model_type"] = model
                    mean_coi_df["simulation_index"] = simulation_index
                    mean_coi_df["parameter_set_index"] = index
                    column_order = (
                        ["model_type"]
                        + ordered_generation_columns
                        + ["simulation_index", "parameter_set_index"]
                    )
                    model_results_list.append(mean_coi_df[column_order])
                all_model_dfs.append(pd.concat(model_results_list))
        final_df = pd.concat(all_model_dfs)
        ordered_generation_columns = [
            "model_type",
            "parameter_set_index",
            "simulation_index",
            "mean_coi_start",
            "mean_coi_gen_5",
            "mean_coi_gen_10",
            "mean_coi_gen_15",
            "mean_coi_gen_20",
            "mean_coi_gen_25",
            "mean_coi_gen_30",
            "mean_coi_gen_35",
            "mean_coi_end",
        ]
        final_df[ordered_generation_columns].to_csv(
            analysis_data_output_dir
            + f"/{simulation_results_file_type}_mean_{paremeter_set_index_string}.csv",
            header=True,
            index=False,
            na_rep="NaN",
            float_format="%.4f",
        )
    else:
        dtypes = {
            "model_type": "object",
            "mean_coi_start": "float64",
            "mean_coi_gen_5": "float64",
            "mean_coi_gen_10": "float64",
            "mean_coi_gen_15": "float64",
            "mean_coi_gen_20": "float64",
            "mean_coi_gen_25": "float64",
            "mean_coi_gen_30": "float64",
            "mean_coi_gen_35": "float64",
            "mean_coi_end": "float64",
            "simulation_index": "int32",
            "parameter_set_index": "int32",
        }
        final_df = pd.read_csv(
            analysis_data_output_dir
            + f"/{simulation_results_file_type}_mean_{paremeter_set_index_string}.csv",
            header=0,
            dtype=dtypes,
        )
    return final_df[final_df["model_type"].isin(model_type_list)]


def get_average_ms_internal_relatedness_across_simulations(
    model_type_list,
    start_simulation_index,
    end_simulation_index,
    simulation_results_file_type,
    simulation_results_root_directory,
    analysis_data_output_dir,
    get_data_from_results_files=False,
):
    simulation_index_range = list(
        range(start_simulation_index, end_simulation_index + 1)
    )
    paremeter_set_index_string = "_".join(
        sorted([str(i) for i in list(PARAMETER_SETS_DICT.keys())])
    )
    all_model_dfs = []
    if get_data_from_results_files:
        print("Getting IR data from S3...")
        for index in sorted(PARAMETER_SETS_DICT.keys()):
            print(f"Parameter Set: {index}")
            parameter_set_sub_dict = PARAMETER_SETS_DICT[index]
            for model in model_type_list:
                print(f"Model Type: {model}")
                model_results_list = []
                for simulation_index in simulation_index_range:
                    newline = "\n" if simulation_index == end_simulation_index else ""
                    sys.stdout.write(
                        "\r" + f"Simulation Index: {simulation_index}" + newline
                    )
                    sys.stdout.flush()
                    ir_dfs = []
                    for file_suffix in FILE_EXTENSIONS_FOR_MATE_CHOICE_GENERATIONS:
                        file_name_from_mate_choice_gen = (
                            f"{simulation_results_file_type}_{file_suffix}.txt"
                        )
                        file_path = get_full_simulation_outputs_path(
                            parameter_set_sub_dict,
                            model_type=model,
                            simulation_index=simulation_index,
                            simulation_output_file=file_name_from_mate_choice_gen,
                            simulation_results_root_directory=simulation_results_root_directory,
                        )
                        try:
                            file_object = open(file_path, "r")
                        except OSError:
                            print(
                                f"{file_name_from_mate_choice_gen} for simulation {simulation_index} and model {model} was not found!"
                            )
                            continue
                        ir = pd.read_csv(
                            file_object,
                            header=None,
                            sep="\t",
                            names=[f"mean_ir_{file_suffix}"],
                        )  # in reality these aren't means, but using this name to avoid having to change names in the .mean df below.
                        ir_dfs.append(ir)
                    ir_dfs_merged = reduce(
                        lambda left, right: pd.merge(
                            left, right, left_index=True, right_index=True
                        ),
                        ir_dfs,
                    )  # merging dataframes by index which represents individuals in the population, order doesn't matter since we're averaging.
                    mean_ir_df = pd.DataFrame(ir_dfs_merged.mean()).T
                    ordered_generation_columns = mean_ir_df.columns.tolist()
                    mean_ir_df["model_type"] = model
                    mean_ir_df["ir_change_during_mate_choice"] = (
                        ir_dfs_merged["mean_ir_end"] - ir_dfs_merged["mean_ir_start"]
                    ).mean()
                    mean_ir_df["percent_ir_change_during_mate_choice"] = (
                        ir_dfs_merged["mean_ir_end"] - ir_dfs_merged["mean_ir_start"]
                    ).divide(ir_dfs_merged["mean_ir_start"]).mean() * 100
                    mean_ir_df["simulation_index"] = simulation_index
                    mean_ir_df["parameter_set_index"] = index
                    column_order = (
                        ["model_type"]
                        + ordered_generation_columns
                        + ["simulation_index", "parameter_set_index"]
                    )
                    model_results_list.append(mean_ir_df[column_order])
                all_model_dfs.append(pd.concat(model_results_list))
        final_df = pd.concat(all_model_dfs)
        ordered_generation_columns = [
            "model_type",
            "parameter_set_index",
            "simulation_index",
            "mean_ir_start",
            "mean_ir_gen_5",
            "mean_ir_gen_10",
            "mean_ir_gen_15",
            "mean_ir_gen_20",
            "mean_ir_gen_25",
            "mean_ir_gen_30",
            "mean_ir_gen_35",
            "mean_ir_end",
        ]
        final_df[ordered_generation_columns].to_csv(
            analysis_data_output_dir
            + f"/{simulation_results_file_type}_mean_{paremeter_set_index_string}.csv",
            header=True,
            index=False,
            na_rep="NaN",
            float_format="%.4f",
        )
    else:
        dtypes = {
            "model_type": "object",
            "mean_ir_start": "float64",
            "mean_ir_gen_5": "float64",
            "mean_ir_gen_10": "float64",
            "mean_ir_gen_15": "float64",
            "mean_ir_gen_20": "float64",
            "mean_ir_gen_25": "float64",
            "mean_ir_gen_30": "float64",
            "mean_ir_gen_35": "float64",
            "mean_ir_end": "float64",
            "simulation_index": "int32",
            "parameter_set_index": "int32",
        }
        final_df = pd.read_csv(
            analysis_data_output_dir
            + f"/{simulation_results_file_type}_mean_{paremeter_set_index_string}.csv",
            header=0,
            dtype=dtypes,
        )
    return final_df[final_df["model_type"].isin(model_type_list)]


### PLOTTING ###


def get_plotting_and_stats_params_from_statistic_type(
    statistic_type, data_df, generation_tuple, dabest_stat_type, ignore_data_df=False
):
    if dabest_stat_type == "single_gen":
        y_axis_column_name = (
            STATISTIC_TYPE_TO_COLUMN_NAME_PREFIX_DICT[statistic_type]
            + MATE_CHOICE_GENERATION_TO_FILE_SUFFIX_DICT[generation_tuple[0]]
        )
        y_axis_label = (
            STATISTIC_TYPE_TO_Y_AXIS_LABEL_PREFIX[statistic_type]
            + f"Gen {generation_tuple[0]}"
        )
        output_file_suffix = "gen_{}".format(generation_tuple[0])
    elif dabest_stat_type == "change_between_gens":
        y_axis_column_name = (
            STATISTIC_TYPE_TO_COLUMN_NAME_PREFIX_DICT[statistic_type]
            + f"gen_{generation_tuple[0]}_to_{generation_tuple[1]}"
        )
        y_axis_label = (
            "Change in "
            + STATISTIC_TYPE_TO_Y_AXIS_LABEL_PREFIX[statistic_type]
            + f"Gens {generation_tuple[0]}-{generation_tuple[1]}"
        )
        output_file_suffix = (
            f"change_gen_{generation_tuple[0]}_to_{generation_tuple[1]}"
        )
        start_column = (
            STATISTIC_TYPE_TO_COLUMN_NAME_PREFIX_DICT[statistic_type]
            + MATE_CHOICE_GENERATION_TO_FILE_SUFFIX_DICT[generation_tuple[0]]
        )
        end_column = (
            STATISTIC_TYPE_TO_COLUMN_NAME_PREFIX_DICT[statistic_type]
            + MATE_CHOICE_GENERATION_TO_FILE_SUFFIX_DICT[generation_tuple[1]]
        )
        if not ignore_data_df:
            data_df.loc[:, y_axis_column_name] = (
                data_df[start_column] - data_df[end_column]
            )
    elif dabest_stat_type == "pct_change_between_gens":
        y_axis_column_name = (
            STATISTIC_TYPE_TO_COLUMN_NAME_PREFIX_DICT[statistic_type]
            + f"gen_{generation_tuple[0]}_to_{generation_tuple[1]}"
        )
        y_axis_label = (
            "% Change in\n"
            + STATISTIC_TYPE_TO_Y_AXIS_LABEL_PREFIX[statistic_type]
            + f"\nGens {generation_tuple[0]}-{generation_tuple[1]}"
        )
        output_file_suffix = (
            f"pct_change_gen_{generation_tuple[0]}_to_{generation_tuple[1]}"
        )
        start_column = (
            STATISTIC_TYPE_TO_COLUMN_NAME_PREFIX_DICT[statistic_type]
            + MATE_CHOICE_GENERATION_TO_FILE_SUFFIX_DICT[generation_tuple[0]]
        )
        end_column = (
            STATISTIC_TYPE_TO_COLUMN_NAME_PREFIX_DICT[statistic_type]
            + MATE_CHOICE_GENERATION_TO_FILE_SUFFIX_DICT[generation_tuple[1]]
        )
        if not ignore_data_df:
            data_df.loc[:, y_axis_column_name] = (
                data_df[start_column] - data_df[end_column]
            ).divide(data_df[start_column]) * 100
    return data_df, y_axis_column_name, y_axis_label, output_file_suffix


def make_shared_control_plot_for_statistic_type(
    data_df,
    model_type_list,
    statistic_type,
    dabest_stat_type,
    parameter_set_index,
    generation_tuple,
    stats_dir_formatted_for_model_type_list,
    figure_dir_formatted_for_model_type_list,
    write_stats_tsv=True,
    write_figure_png=True,
):
    (
        data_df,
        y_axis_column_name,
        y_axis_label,
        output_file_suffix,
    ) = get_plotting_and_stats_params_from_statistic_type(
        statistic_type=statistic_type,
        data_df=data_df,
        generation_tuple=generation_tuple,
        dabest_stat_type=dabest_stat_type,
    )
    plot_df = data_df[
        (data_df["parameter_set_index"] == parameter_set_index)
        & (data_df["model_type"].isin(model_type_list))
    ]

    # Setting the X-axis labels to "title"
    index = [MODEL_NAME_DICT[model] for model in model_type_list]
    plot_df["model_type"] = plot_df["model_type"].map(MODEL_NAME_DICT)

    # load dataset and make plot
    shared_control = dabest.load(
        plot_df, idx=index, x="model_type", y=y_axis_column_name
    )

    if write_stats_tsv:
        stats_dir = stats_dir_formatted_for_model_type_list.format(
            "|".join(model_type_list)
        )
        make_dir_if_does_not_exist(stats_dir)
        stats_output_file = "{}.tsv".format(
            get_output_file_name(
                parameter_set_dict=PARAMETER_SETS_DICT[parameter_set_index],
                statistic_type=statistic_type,
                output_file_root_dir=stats_dir,
                output_file_suffix=output_file_suffix,
            )
        )
        shared_control.mean_diff.results.to_csv(
            stats_output_file, header=True, index=True, sep="\t"
        )
    if write_figure_png:
        shared_control.mean_diff.plot(
            custom_palette=COLORS,
            swarm_label=y_axis_label,
            contrast_label="Mean Difference",
        )
        figure_dir = figure_dir_formatted_for_model_type_list.format(
            "|".join(model_type_list)
        )
        make_dir_if_does_not_exist(figure_dir)
        plt.savefig(
            "{}.png".format(
                get_output_file_name(
                    parameter_set_dict=PARAMETER_SETS_DICT[parameter_set_index],
                    statistic_type=statistic_type,
                    output_file_root_dir=figure_dir,
                    output_file_suffix=output_file_suffix,
                )
            ),
            dpi=600,
        )
        plt.clf()
        plt.close()


def make_grouped_control_plot_for_statistic_type_and_two_parameter_sets(
    data_df,
    model_type_list,
    statistic_type,
    dabest_stat_type,
    parameter_set_index_control,
    parameter_set_index_test,
    generation_tuple,
    stats_dir_formatted_for_model_type_list,
    figure_dir_formatted_for_model_type_list,
    output_png=None,
    return_stats_df=False,
    write_stats_tsv=True,
    write_figure_png=True,
    dpi=800,
):
    # one parameter set against another across all models
    (
        data_df,
        y_axis_column_name,
        y_axis_label,
        output_file_suffix,
    ) = get_plotting_and_stats_params_from_statistic_type(
        statistic_type=statistic_type,
        data_df=data_df,
        generation_tuple=generation_tuple,
        dabest_stat_type=dabest_stat_type,
    )

    # have to make a column that combines the model and parameter set indices
    plot_df = data_df[
        (
            data_df["parameter_set_index"].isin(
                [parameter_set_index_control, parameter_set_index_test]
            )
            & (data_df["model_type"].isin(model_type_list))
        )
    ]
    plot_df.loc[:, "test"] = "-"
    plot_df.loc[:, "model_and_parameter_set_index"] = (
        plot_df["model_type"]
        + plot_df["test"]
        + plot_df["parameter_set_index"].astype(str)
    )

    # index needs to be tuple of tuples with pairs of
    custom_palette_dict = {}
    for i, model in enumerate(model_type_list):
        for parameter_set_index in [
            parameter_set_index_control,
            parameter_set_index_test,
        ]:
            custom_palette_dict[model + "-" + str(parameter_set_index)] = COLORS[i]
    index_for_plotting = tuple(
        [
            tuple(
                [
                    model + "-" + str(parameter_set_index_control),
                    model + "-" + str(parameter_set_index_test),
                ]
            )
            for model in model_type_list
        ]
    )
    paired_control = dabest.load(
        plot_df,
        idx=index_for_plotting,
        x="model_and_parameter_set_index",
        y=y_axis_column_name,
    )

    if write_stats_tsv:
        stats_dir = (
            stats_dir_formatted_for_model_type_list.format("|".join(model_type_list))
            + f"/parameter_sets_paired/{statistic_type}"
        )
        make_dir_if_does_not_exist(stats_dir)
        stats_output_file = f"{stats_dir}/{parameter_set_index_control}_{parameter_set_index_test}_paired_control_{output_file_suffix}.tsv"
        paired_control.mean_diff.results.to_csv(
            stats_output_file, header=True, index=True, sep="\t"
        )
    if return_stats_df:
        return paired_control.mean_diff.results
    if write_figure_png:
        plt.rc("font", size=18)
        plt.rc("ytick", labelsize=14)
        plt.rc("xtick", labelsize=15)

        fig = paired_control.mean_diff.plot(
            swarm_label=y_axis_label,
            contrast_label="Mean Difference",
            custom_palette=custom_palette_dict,
            fig_size=(18, 12),
        )
        if output_png is not None:
            path_to_output_png = output_png
        else:
            figure_dir = (
                figure_dir_formatted_for_model_type_list.format(
                    "|".join(model_type_list)
                )
                + f"/parameter_sets_paired/{statistic_type}"
            )
            make_dir_if_does_not_exist(figure_dir)
            path_to_output_png = f"{figure_dir}/{parameter_set_index_control}_{parameter_set_index_test}_paired_control_{output_file_suffix}.png"
        # Rotate ticks 45 degs, and also set custom tick labels
        # See https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html
        for j, ax in enumerate(fig.axes):
            xticks_text = ax.get_xticklabels()
            ax.set_xticklabels(xticks_text, rotation=45, horizontalalignment="center")
        plt.subplots_adjust(hspace=0.2, bottom=0.15)
        plt.savefig(path_to_output_png, dpi=dpi)
        plt.clf()
        plt.close()


def make_richness_distance_to_ms_plots(
    richness_per_locus_df,
    model_type_list,
    statistic_type,
    dabest_stat_type,
    parameter_set_index,
    generation_tuple,
    stats_dir_formatted_for_model_type_list,
    figure_dir_formatted_for_model_type_list,
    output_png=None,
    maximum_haplotype_distance_to_ms=4,
    write_stats_tsv=True,
    write_figure_png=True,
    dpi=800,
):

    plot_df = richness_per_locus_df[
        (richness_per_locus_df["parameter_set_index"] == parameter_set_index)
        & (richness_per_locus_df["model_type"].isin(model_type_list))
    ]
    plot_df.reset_index(inplace=True)
    plot_df.loc[:, "distance_to_closest_ms"] = plot_df["position"].apply(
        lambda x: (MS_POSITIONS - x).abs().min()
    )
    plot_df_final = plot_df[
        plot_df["distance_to_closest_ms"] <= maximum_haplotype_distance_to_ms
    ]
    plot_df_final.loc[:, "test"] = "-"
    plot_df_final.loc[:, "model_and_distance"] = (
        plot_df_final["model_type"]
        + plot_df_final["test"]
        + plot_df_final["distance_to_closest_ms"].astype(str)
    )
    custom_palette_dict = {}
    for model in model_type_list:
        for distance in range(maximum_haplotype_distance_to_ms + 1):
            custom_palette_dict[model + "-" + str(distance)] = COLORS[distance]
    unique_distances_as_strings = [
        str(i) for i in sorted(plot_df_final["distance_to_closest_ms"].unique())
    ]
    index_for_plotting = tuple(
        [
            tuple([model + "-" + distance for model in model_type_list])
            for distance in unique_distances_as_strings
        ]
    )

    (
        plot_df_final,
        y_axis_column_name,
        y_axis_label,
        output_file_suffix,
    ) = get_plotting_and_stats_params_from_statistic_type(
        statistic_type=statistic_type,
        data_df=plot_df_final,
        generation_tuple=generation_tuple,
        dabest_stat_type=dabest_stat_type,
    )

    paired_control = dabest.load(
        plot_df_final,
        idx=index_for_plotting,
        x="model_and_distance",
        y=y_axis_column_name,
    )
    if write_stats_tsv:
        stats_dir = stats_dir_formatted_for_model_type_list.format(
            "|".join(model_type_list)
        )
        make_dir_if_does_not_exist(stats_dir)
        stats_output_file = "{}_distance_to_ms.tsv".format(
            get_output_file_name(
                parameter_set_dict=PARAMETER_SETS_DICT[parameter_set_index],
                statistic_type=statistic_type,
                output_file_root_dir=stats_dir,
                output_file_suffix=output_file_suffix,
            )
        )
        paired_control.mean_diff.results.to_csv(
            stats_output_file, header=True, index=True, sep="\t"
        )
    if write_figure_png:
        plt.rc("font", size=18)
        plt.rc("ytick", labelsize=14)
        plt.rc("xtick", labelsize=15)

        fig = paired_control.mean_diff.plot(
            swarm_label=y_axis_label,
            contrast_label="Mean Difference",
            custom_palette=custom_palette_dict,
            fig_size=(18, 12),
        )
        if output_png is not None:
            path_to_output_png = output_png
        else:
            figure_dir = figure_dir_formatted_for_model_type_list.format(
                "|".join(model_type_list)
            )
            make_dir_if_does_not_exist(figure_dir)
            path_to_output_png = "{}_distance_to_ms.png".format(
                get_output_file_name(
                    parameter_set_dict=PARAMETER_SETS_DICT[parameter_set_index],
                    statistic_type=statistic_type,
                    output_file_root_dir=figure_dir,
                    output_file_suffix=output_file_suffix,
                )
            )
        # Rotate ticks 45 degs, and also set custom tick labels
        # See https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html
        for j, ax in enumerate(fig.axes):
            if j == 1:
                number_of_labels = len(ax.get_xticklabels())
                xticks_text = [
                    f"{(0.25 * (i - 1)):.1f} Mb" if i % 2 == 1 else ""
                    for i in range(number_of_labels)
                ]
            else:
                xticks_text = ax.get_xticklabels()
            ax.set_xticklabels(xticks_text, rotation=45, horizontalalignment="center")
        plt.subplots_adjust(hspace=0.2, bottom=0.15)
        plt.savefig(path_to_output_png, dpi=dpi)
        plt.clf()
        plt.close()


def make_plot_of_stats_over_time(
    statistic_type,
    data_df,
    model_type_list,
    parameter_set_index,
    figure_dir_formatted_for_model_type_list,
    write_figure_png=True,
):
    fig, ax = plt.subplots()

    for i, model_type in enumerate(model_type_list):
        maximum_generation_number = PARAMETER_SETS_DICT[parameter_set_index][
            "generations_in_mate_choice"
        ]
        generation_times = list(range(0, maximum_generation_number + 1, 5))
        percentile_5 = []
        medians = []
        percentile_95 = []
        model_df = data_df[
            (data_df["model_type"] == model_type)
            & (data_df["parameter_set_index"] == parameter_set_index)
        ]
        for suffix in FILE_EXTENSIONS_FOR_MATE_CHOICE_GENERATIONS:
            full_suffix = (
                STATISTIC_TYPE_TO_COLUMN_NAME_PREFIX_DICT[statistic_type] + suffix
            )
            percentile_5.append(model_df[full_suffix].quantile(0.05))
            medians.append(model_df[full_suffix].quantile(0.5))
            percentile_95.append(model_df[full_suffix].quantile(0.95))
        ax.plot(generation_times, percentile_5, color=COLORS[i], alpha=0.25)
        ax.plot(generation_times, percentile_95, color=COLORS[i], alpha=0.25)
        ax.plot(
            generation_times,
            medians,
            color=COLORS[i],
            label=LONG_MODEL_NAME_DICT[model_type],
        )
        ax.fill_between(
            generation_times, percentile_5, percentile_95, color=COLORS[i], alpha=0.1
        )
        ax.set(
            xlabel="Generation of Mate Choice",
            ylabel=STATISTIC_TYPE_TO_Y_AXIS_LABEL_PREFIX[statistic_type][:-1],
        )
        ax.legend()
        if write_figure_png:
            figure_dir = figure_dir_formatted_for_model_type_list.format(
                "|".join(model_type_list)
            )
            plt.savefig(
                "{}_change_over_mate_choice.png".format(
                    get_output_file_name(
                        parameter_set_dict=PARAMETER_SETS_DICT[parameter_set_index],
                        statistic_type=statistic_type,
                        output_file_root_dir=figure_dir,
                        output_file_suffix="change_over_time",
                    )
                ),
                dpi=300,
            )
    plt.close()


def make_heterozygosity_by_richness_stats_all_model_types_and_parameter_sets_scatter_plot(
    het_mean_difference_df, richness_mean_difference_df
):
    # merge data to one df
    data_df = het_mean_difference_df.merge(
        richness_mean_difference_df,
        on=["test", "parameter_set_index"],
        suffixes=("_heterozygosity", "_richness"),
        how="inner",
    )
    models = PRINTED_NAMES_OF_MODELS[1:]
    model_to_color_dict = {models[i]: COLORS[i] for i in list(range(len(models)))}
    data_df["color"] = data_df["test"].map(model_to_color_dict)
    data_df["width"] = (
        data_df["bca_high_heterozygosity"] - data_df["bca_low_heterozygosity"]
    )
    data_df["height"] = data_df["bca_high_richness"] - data_df["bca_low_richness"]
    xmin = data_df["difference_heterozygosity"].min() - 1
    xmax = data_df["difference_heterozygosity"].max() + 1
    ymin = data_df["difference_richness"].min() - 1
    ymax = data_df["difference_richness"].max() + 1
    # set up figure
    fig, ax = plt.subplots()
    # remove spines
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    # remove ticks
    plt.xticks([])
    plt.yticks([])
    # place axes through origin
    plt.axhline(0, xmin=xmin, xmax=xmax, color="k", linewidth=1)  # horizontal line
    plt.axvline(0, ymin=ymin, ymax=ymax, color="k", linewidth=1)  # vertical line
    # add patch to draw attention to good quadrant
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            width=0 - xmin,
            height=0 - ymin,
            color="#b6bbc2",
            linestyle="-",
            linewidth=1.5,
            zorder=0,
        )
    )

    # plot data
    for i, model in enumerate(models):
        sub_df = data_df[data_df["test"] == model]
        color = COLORS[i]
        ax.scatter(
            sub_df["difference_heterozygosity"].tolist(),
            sub_df["difference_richness"].tolist(),
            color=color,
            label=model,
            s=3,
        )
        for j, row in sub_df.iterrows():
            ax.add_patch(
                Rectangle(
                    xy=(row["bca_low_heterozygosity"], row["bca_low_richness"]),
                    width=row["width"],
                    height=row["height"],
                    linewidth=0.5,
                    color=color,
                    fill=False,
                )
            )
    ax.legend()
    ax.set(xlabel="Pct Loss in Heterozygosity", ylabel="Pct Loss in Richness")
    plt.savefig("/Users/aaronsams/Desktop/test.png", dpi=300)


def make_heterozygosity_by_richness_stats_all_model_types_all_generations_one_parameter_set_scatter_plot(
    model_type_list,
    parameter_set_index,
    stats_dir_formatted_for_model_type_list,
    output_png,
    dabest_stat_type="pct_change_between_gens",
    dpi=800,
    import_csv_instead_of_making=True,
):

    all_heterozygosity_rows = []
    all_richness_rows = []

    for generation_end in range(5, 41, 5):
        gw_heterozygosity_df = make_data_merged_with_dabest_stats_csv(
            statistic_type="gw_heterozygosity",
            parameter_set_index_list=[parameter_set_index],
            model_type_list=model_type_list,
            data_df=None,
            dabest_stat_type="pct_change_between_gens",
            generation_tuple=(0, generation_end),
            stats_dir_formatted_for_model_type_list=stats_dir_formatted_for_model_type_list,
            write_tsv=False,
            return_df=True,
            import_csv_instead_of_making=import_csv_instead_of_making,
        )  # This plotting function only works if you have already made the necessary dabest stats files with make_data_merged_with_dabest_stats_csv
        gw_heterozygosity_df.loc[:, "generation"] = generation_end
        gw_richness_df = make_data_merged_with_dabest_stats_csv(
            statistic_type="gw_richness",
            parameter_set_index_list=[parameter_set_index],
            model_type_list=model_type_list,
            data_df=None,
            dabest_stat_type="pct_change_between_gens",
            generation_tuple=(0, generation_end),
            stats_dir_formatted_for_model_type_list=stats_dir_formatted_for_model_type_list,
            write_tsv=False,
            return_df=True,
            import_csv_instead_of_making=import_csv_instead_of_making,
        )  # This plotting function only works if you have already made the necessary dabest stats files with make_data_merged_with_dabest_stats_csv
        gw_richness_df.loc[:, "generation"] = generation_end
        all_heterozygosity_rows.append(gw_heterozygosity_df)
        all_richness_rows.append(gw_richness_df)

    all_heterozygosity_df = pd.concat(all_heterozygosity_rows)
    all_richness_df = pd.concat(all_richness_rows)

    # merge data to one df
    data_df = all_heterozygosity_df.merge(
        all_richness_df,
        on=["test", "parameter_set_index", "generation"],
        suffixes=("_heterozygosity", "_richness"),
        how="inner",
    )
    models = PRINTED_NAMES_OF_MODELS[1:]
    model_to_color_dict = {models[i]: COLORS[i] for i in list(range(len(models)))}
    data_df["color"] = data_df["test"].map(model_to_color_dict)
    data_df.loc[:, "edgecolor"] = np.where(
        data_df["generation"] < 40, data_df["color"], "k"
    )
    data_df.loc[:, "linewidth"] = np.where(data_df["generation"] < 40, 0.5, 0.6)
    data_df["width"] = (
        data_df["bca_high_heterozygosity"] - data_df["bca_low_heterozygosity"]
    )
    data_df["height"] = data_df["bca_high_richness"] - data_df["bca_low_richness"]
    xmin = data_df["difference_heterozygosity"].min() - 1
    xmax = data_df["difference_heterozygosity"].max() + 1
    ymin = data_df["difference_richness"].min() - 1
    ymax = data_df["difference_richness"].max() + 1
    # set up figure
    fig, ax = plt.subplots()
    # remove spines
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # place axes through origin
    plt.axhline(0, xmin=xmin, xmax=xmax, color="k", linewidth=1)  # horizontal line
    plt.axvline(0, ymin=ymin, ymax=ymax, color="k", linewidth=1)  # vertical line

    # plot data
    for i, model in enumerate(models):
        sub_df = data_df[data_df["test"] == model]
        color = COLORS[i]
        ax.scatter(
            sub_df["difference_heterozygosity"].tolist(),
            sub_df["difference_richness"].tolist(),
            color=color,
            label=model,
            s=3,
        )
        for j, row in sub_df.iterrows():
            ax.add_patch(
                Rectangle(
                    xy=(row["bca_low_heterozygosity"], row["bca_low_richness"]),
                    width=row["width"],
                    height=row["height"],
                    linewidth=row["linewidth"],
                    facecolor=row["color"],
                    edgecolor=row["edgecolor"],
                    fill=False,
                )
            )
    ax.legend()
    ax.set(
        xlabel="% Loss in Heterozygosity (Model - Random)",
        ylabel="% Loss in Richness (Model - Random)",
    )

    # add patch to draw attention to good quadrant
    recymin = ax.get_ylim()[0]
    recxmin = ax.get_xlim()[0]
    ax.add_patch(
        Rectangle(
            (recxmin, recymin),
            width=0 - recxmin,
            height=0 - recymin,
            color="#b6bbc2",
            linestyle="-",
            linewidth=0,
            zorder=0,
            alpha=0.25,
        )
    )

    plt.savefig(output_png, dpi=dpi)


def make_inbreeding_by_richness_stats_all_model_types_all_generations_one_parameter_set_scatter_plot(
    model_type_list,
    parameter_set_index,
    stats_dir_formatted_for_model_type_list,
    output_png,
    dabest_stat_type="pct_change_between_gens",
    dpi=800,
    import_csv_instead_of_making=True,
):

    all_inbreeding_rows = []
    all_richness_rows = []

    for generation_end in range(5, 41, 5):
        coi_df = make_data_merged_with_dabest_stats_csv(
            statistic_type="coi",
            parameter_set_index_list=[parameter_set_index],
            model_type_list=model_type_list,
            data_df=None,
            dabest_stat_type="pct_change_between_gens",
            generation_tuple=(0, generation_end),
            stats_dir_formatted_for_model_type_list=stats_dir_formatted_for_model_type_list,
            write_tsv=False,
            return_df=True,
            import_csv_instead_of_making=import_csv_instead_of_making,
        )  # This plotting function only works if you have already made the necessary dabest stats files with make_data_merged_with_dabest_stats_csv
        coi_df.loc[:, "generation"] = generation_end
        # reversing the signs of the inbreeding stats for clarity (inbreeding increases over time with drift and we're interested here in how much inbreeding is increasing in focal model relative to random mating)
        coi_df.loc[:, "difference"] *= -1
        coi_df.loc[:, "bca_low"] *= -1
        coi_df.loc[:, "bca_high"] *= -1
        coi_df.rename(
            columns={"bca_low": "bca_high", "bca_high": "bca_low"}, inplace=True
        )

        gw_richness_df = make_data_merged_with_dabest_stats_csv(
            statistic_type="gw_richness",
            parameter_set_index_list=[parameter_set_index],
            model_type_list=model_type_list,
            data_df=None,
            dabest_stat_type="pct_change_between_gens",
            generation_tuple=(0, generation_end),
            stats_dir_formatted_for_model_type_list=stats_dir_formatted_for_model_type_list,
            write_tsv=False,
            return_df=True,
            import_csv_instead_of_making=import_csv_instead_of_making,
        )  # This plotting function only works if you have already made the necessary dabest stats files with make_data_merged_with_dabest_stats_csv
        gw_richness_df.loc[:, "generation"] = generation_end
        all_inbreeding_rows.append(coi_df)
        all_richness_rows.append(gw_richness_df)

    all_coi_df = pd.concat(all_inbreeding_rows)
    all_richness_df = pd.concat(all_richness_rows)

    # merge data to one df
    data_df = all_coi_df.merge(
        all_richness_df,
        on=["test", "parameter_set_index", "generation"],
        suffixes=("_coi", "_richness"),
        how="inner",
    )
    models = PRINTED_NAMES_OF_MODELS[1:]
    model_to_color_dict = {models[i]: COLORS[i] for i in list(range(len(models)))}
    data_df["color"] = data_df["test"].map(model_to_color_dict)
    data_df.loc[:, "edgecolor"] = np.where(
        data_df["generation"] < 40, data_df["color"], "k"
    )
    data_df.loc[:, "linewidth"] = np.where(data_df["generation"] < 40, 0.5, 0.6)
    data_df["width"] = data_df["bca_high_coi"] - data_df["bca_low_coi"]
    data_df["height"] = data_df["bca_high_richness"] - data_df["bca_low_richness"]
    xmin = data_df["difference_coi"].min() - 1
    xmax = data_df["difference_coi"].max() + 1
    ymin = data_df["difference_richness"].min() - 1
    ymax = data_df["difference_richness"].max() + 1
    # set up figure
    fig, ax = plt.subplots()
    # remove spines
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # place axes through origin
    plt.axhline(0, xmin=xmin, xmax=xmax, color="k", linewidth=1)  # horizontal line
    plt.axvline(0, ymin=ymin, ymax=ymax, color="k", linewidth=1)  # vertical line

    # plot data
    for i, model in enumerate(models):
        sub_df = data_df[data_df["test"] == model]
        color = COLORS[i]
        ax.scatter(
            sub_df["difference_coi"].tolist(),
            sub_df["difference_richness"].tolist(),
            color=color,
            label=model,
            s=3,
        )
        for j, row in sub_df.iterrows():
            ax.add_patch(
                Rectangle(
                    xy=(row["bca_low_coi"], row["bca_low_richness"]),
                    width=row["width"],
                    height=row["height"],
                    linewidth=row["linewidth"],
                    facecolor=row["color"],
                    edgecolor=row["edgecolor"],
                    fill=False,
                )
            )
    ax.legend()
    ax.set(
        xlabel="% Increase in Inbreeding (Model - Random)",
        ylabel="% Loss in Richness (Model - Random)",
    )

    # add patch to draw attention to good quadrant
    recymin = ax.get_ylim()[0]
    recxmin = ax.get_xlim()[0]
    ax.add_patch(
        Rectangle(
            (recxmin, recymin),
            width=0 - recxmin,
            height=0 - recymin,
            color="#b6bbc2",
            linestyle="-",
            linewidth=0,
            zorder=0,
            alpha=0.25,
        )
    )

    plt.savefig(output_png, dpi=dpi)


def make_heterozygosity_by_richness_stats_all_model_types_all_generations_all_parameter_sets_scatter_plot(
    model_type_list,
    stats_dir_formatted_for_model_type_list,
    output_png,
    dabest_stat_type="pct_change_between_gens",
    dpi=800,
    import_csv_instead_of_making=True,
):

    all_heterozygosity_rows = []
    all_richness_rows = []

    for generation_end in range(5, 41, 5):
        gw_heterozygosity_df = make_data_merged_with_dabest_stats_csv(
            statistic_type="gw_heterozygosity",
            parameter_set_index_list=PARAMETER_SETS_DICT.keys(),
            model_type_list=model_type_list,
            data_df=None,
            dabest_stat_type="pct_change_between_gens",
            generation_tuple=(0, generation_end),
            stats_dir_formatted_for_model_type_list=stats_dir_formatted_for_model_type_list,
            write_tsv=False,
            return_df=True,
            import_csv_instead_of_making=import_csv_instead_of_making,
        )  # This plotting function only works if you have already made the necessary dabest stats files with make_data_merged_with_dabest_stats_csv
        gw_heterozygosity_df.loc[:, "generation"] = generation_end
        gw_richness_df = make_data_merged_with_dabest_stats_csv(
            statistic_type="gw_richness",
            parameter_set_index_list=PARAMETER_SETS_DICT.keys(),
            model_type_list=model_type_list,
            data_df=None,
            dabest_stat_type="pct_change_between_gens",
            generation_tuple=(0, generation_end),
            stats_dir_formatted_for_model_type_list=stats_dir_formatted_for_model_type_list,
            write_tsv=False,
            return_df=True,
            import_csv_instead_of_making=import_csv_instead_of_making,
        )  # This plotting function only works if you have already made the necessary dabest stats files with make_data_merged_with_dabest_stats_csv
        gw_richness_df.loc[:, "generation"] = generation_end
        all_heterozygosity_rows.append(gw_heterozygosity_df)
        all_richness_rows.append(gw_richness_df)

    all_heterozygosity_df = pd.concat(all_heterozygosity_rows)
    all_richness_df = pd.concat(all_richness_rows)

    # merge data to one df
    data_df = all_heterozygosity_df.merge(
        all_richness_df,
        on=["test", "parameter_set_index", "generation"],
        suffixes=("_heterozygosity", "_richness"),
        how="inner",
    )
    models = PRINTED_NAMES_OF_MODELS[1:]
    model_to_color_dict = {models[i]: COLORS[i] for i in list(range(len(models)))}
    data_df.loc[:, "color"] = data_df["test"].map(model_to_color_dict)
    data_df.loc[:, "edgecolor"] = np.where(
        data_df["generation"] < 40, data_df["color"], "k"
    )
    data_df.loc[:, "width"] = (
        data_df["bca_high_heterozygosity"] - data_df["bca_low_heterozygosity"]
    )
    data_df.loc[:, "height"] = (
        data_df["bca_high_richness"] - data_df["bca_low_richness"]
    )

    # set up figure
    fig, ax = plt.subplots(3, 7, sharex="all", sharey="all")
    parameter_set_index = 1
    for r in range(3):
        for c in range(7):
            data_df_for_psi = data_df[
                data_df["parameter_set_index"] == parameter_set_index
            ]
            xmin = data_df_for_psi["difference_heterozygosity"].min() - 1
            xmax = data_df_for_psi["difference_heterozygosity"].max() + 1
            ymin = data_df_for_psi["difference_richness"].min() - 1
            ymax = data_df_for_psi["difference_richness"].max() + 1
            # remove spines
            for spine in ax[r, c].spines.values():
                spine.set_visible(False)
            # place axes through origin
            ax[r, c].axhline(
                0, xmin=xmin, xmax=xmax, color="k", linewidth=1, zorder=0
            )  # horizontal line
            ax[r, c].axvline(
                0, ymin=ymin, ymax=ymax, color="k", linewidth=1, zorder=0
            )  # vertical line

            # plot data
            for i, model in enumerate(models):
                sub_df = data_df_for_psi[data_df_for_psi["test"] == model]
                color = COLORS[i]

                ax[r, c].scatter(
                    sub_df["difference_heterozygosity"].tolist(),
                    sub_df["difference_richness"].tolist(),
                    color=color,
                    label=model,
                    s=0.01,
                    linewidths=0,
                )
                for j, row in sub_df.iterrows():
                    ax[r, c].add_patch(
                        Rectangle(
                            xy=(row["bca_low_heterozygosity"], row["bca_low_richness"]),
                            width=row["width"],
                            height=row["height"],
                            linewidth=0.25,
                            facecolor=row["color"],
                            edgecolor=row["edgecolor"],
                            fill=True,
                        )
                    )

            ax[r, c].set_xlabel(str(parameter_set_index))

            parameter_set_index += 1

    # add patches to draw attention to good quadrant (have to do this in separate loop after axes clipped)
    for r in range(3):
        for c in range(7):
            recymin = ax[r, c].get_ylim()[0]
            recxmin = ax[r, c].get_xlim()[0]
            ax[r, c].add_patch(
                Rectangle(
                    (recxmin, recymin),
                    width=0 - recxmin,
                    height=0 - recymin,
                    color="#b6bbc2",
                    linestyle="-",
                    linewidth=0,
                    zorder=0,
                    alpha=0.25,
                )
            )

    fig.text(0.5, 0.0, "% Loss in Heterozygosity Relative to Random", ha="center")
    fig.text(
        0.05,
        0.5,
        "% Loss in Richness Relative to Random",
        va="center",
        rotation="vertical",
    )

    plt.savefig(output_png, dpi=dpi, tight_layout=True)


### MAKE STATS ONLY ###


def make_data_merged_with_dabest_stats_csv(
    statistic_type,
    parameter_set_index_list,
    model_type_list,
    data_df,
    dabest_stat_type,
    generation_tuple,
    stats_dir_formatted_for_model_type_list=None,
    output_file_csv=None,
    write_tsv=True,
    return_df=False,
    import_csv_instead_of_making=False,
):

    ignore_data_df = False
    if import_csv_instead_of_making:
        ignore_data_df = True

    (
        data_df,
        y_axis_column_name,
        y_axis_label,
        output_file_suffix,
    ) = get_plotting_and_stats_params_from_statistic_type(
        statistic_type=statistic_type,
        data_df=data_df,
        generation_tuple=generation_tuple,
        dabest_stat_type=dabest_stat_type,
        ignore_data_df=ignore_data_df,
    )
    if output_file_csv:
        final_file_name = output_file_csv
    else:
        stats_dir = stats_dir_formatted_for_model_type_list.format(
            "|".join(model_type_list)
        )
        make_dir_if_does_not_exist(stats_dir)
        final_file_name = "{stats_dir}/{statistic_type}_{output_file_suffix}.csv".format(
            stats_dir=stats_dir,
            statistic_type=statistic_type,
            output_file_suffix=output_file_suffix,
        )

    if import_csv_instead_of_making:
        final_df_temp = pd.read_csv(
            final_file_name, header=0
        )  # assuming this has been made to include all possible parameter set indices
        final_df = final_df_temp[
            final_df_temp["parameter_set_index"].isin(parameter_set_index_list)
        ]
    else:
        data_df_copy = data_df.copy()
        index = [MODEL_NAME_DICT[model] for model in model_type_list]
        x_axis_column_name = "model_type"
        new_x_column = data_df_copy[x_axis_column_name].apply(
            lambda x: MODEL_NAME_DICT[x]
        )
        data_df_copy.loc[:, x_axis_column_name] = new_x_column

        stats_dataframes = []
        for parameter_set_index in parameter_set_index_list:
            data_df_copy_for_parameter_set_index = data_df_copy[
                data_df_copy["parameter_set_index"] == parameter_set_index
            ]
            shared_control = dabest.load(
                data_df_copy_for_parameter_set_index,
                idx=index,
                x=x_axis_column_name,
                y=y_axis_column_name,
            )
            shared_control_results_df = shared_control.mean_diff.results
            stats_df = shared_control_results_df[
                [
                    "control",
                    "test",
                    "difference",
                    "bca_low",
                    "bca_high",
                    "statistic_students_t",
                    "pvalue_students_t",
                ]
            ]
            stats_df.loc[:, "parameter_set_index"] = parameter_set_index
            stats_dataframes.append(stats_df)
        final_df = pd.concat(stats_dataframes).sort_values(by=["control", "test"])
        if write_tsv:
            final_df.to_csv(
                final_file_name, index=False, header=True, float_format="%.4f"
            )
    if return_df:
        return final_df


def get_mean_stats_for_statistic_type(stats_df, path_to_output_csv):
    stats_df.drop("simulation_index", axis=1).groupby(
        ["model_type", "parameter_set_index"]
    ).mean().to_csv(path_to_output_csv, header=True, index=True, float_format="%.4f")


@begin.start
@begin.convert(
    get_data_from_results_files=begin.utils.tobool,
    start_simulation_index=int,
    end_simulation_index=int,
)
def main(
    get_data_from_results_files=False,  # If running this for the first time, need to set this to True to generate data summary files from raw data
    output_root_directory=None,  # must pass to script or add path here as default (e.g. {workding_directory}/simulation_analysis_output)
    simulation_results_root_directory=None,  # must pass to script or add path here as default (e.g. {working_directory}/sams_et_al_2020_evolutionary_applications_simulation_raw_data)
    start_simulation_index=1,
    end_simulation_index=100,
):

    STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST = (
        f"{output_root_directory}/statistics/" + "{}"
    )
    DATA_DIR = f"{output_root_directory}/data_csvs"
    # make dirs if they don't exist
    make_dir_if_does_not_exist(DATA_DIR)
    make_dir_if_does_not_exist(f"{output_root_directory}/figures")
    make_dir_if_does_not_exist(f"{output_root_directory}/statistics")

    gw_heterozygosity_df = get_average_heterozygosity_across_simulations_and_parameter_sets(
        model_type_list=VALID_MODEL_TYPES,
        start_simulation_index=start_simulation_index,
        end_simulation_index=end_simulation_index,
        simulation_results_file_type=STATISTIC_TYPE_TO_FILE_SUFFIX_DICT[
            "gw_heterozygosity"
        ],
        simulation_results_root_directory=simulation_results_root_directory,
        analysis_data_output_dir=DATA_DIR,
        get_data_from_results_files=get_data_from_results_files,
    )
    (
        gw_richness_per_locus_df,
        gw_richness_per_individual_df,
    ) = get_per_locus_and_per_individual_average_richness_across_simulations(
        model_type_list=VALID_MODEL_TYPES,
        start_simulation_index=start_simulation_index,
        end_simulation_index=end_simulation_index,
        simulation_results_file_type=STATISTIC_TYPE_TO_FILE_SUFFIX_DICT["gw_richness"],
        simulation_results_root_directory=simulation_results_root_directory,
        analysis_data_output_dir=DATA_DIR,
        get_data_from_results_files=get_data_from_results_files,
    )
    ms_heterozygosity_df = get_average_heterozygosity_across_simulations_and_parameter_sets(
        model_type_list=VALID_MODEL_TYPES,
        start_simulation_index=start_simulation_index,
        end_simulation_index=end_simulation_index,
        simulation_results_file_type=STATISTIC_TYPE_TO_FILE_SUFFIX_DICT[
            "ms_heterozygosity"
        ],
        simulation_results_root_directory=simulation_results_root_directory,
        analysis_data_output_dir=DATA_DIR,
        get_data_from_results_files=get_data_from_results_files,
    )
    (
        ms_richness_per_locus_df,
        ms_richness_per_individual_df,
    ) = get_per_locus_and_per_individual_average_richness_across_simulations(
        model_type_list=VALID_MODEL_TYPES,
        start_simulation_index=start_simulation_index,
        end_simulation_index=end_simulation_index,
        simulation_results_file_type=STATISTIC_TYPE_TO_FILE_SUFFIX_DICT["ms_richness"],
        simulation_results_root_directory=simulation_results_root_directory,
        analysis_data_output_dir=DATA_DIR,
        get_data_from_results_files=get_data_from_results_files,
    )
    mean_coi_df = get_average_coi_across_simulations(
        model_type_list=VALID_MODEL_TYPES,
        start_simulation_index=start_simulation_index,
        end_simulation_index=end_simulation_index,
        simulation_results_file_type=STATISTIC_TYPE_TO_FILE_SUFFIX_DICT["coi"],
        simulation_results_root_directory=simulation_results_root_directory,
        analysis_data_output_dir=DATA_DIR,
        get_data_from_results_files=get_data_from_results_files,
    )
    mean_ir_df = get_average_ms_internal_relatedness_across_simulations(
        model_type_list=VALID_MODEL_TYPES,
        start_simulation_index=start_simulation_index,
        end_simulation_index=end_simulation_index,
        simulation_results_file_type=STATISTIC_TYPE_TO_FILE_SUFFIX_DICT["ir"],
        simulation_results_root_directory=simulation_results_root_directory,
        analysis_data_output_dir=DATA_DIR,
        get_data_from_results_files=get_data_from_results_files,
    )

    statistic_type_to_data_df_dict = {
        "gw_heterozygosity": gw_heterozygosity_df,
        "ms_heterozygosity": ms_heterozygosity_df,
        "gw_richness": gw_richness_per_individual_df,
        "ms_richness": ms_richness_per_individual_df,
        "gw_over_5_percent_richness": gw_richness_per_individual_df,
        "ms_over_5_percent_richness": ms_richness_per_individual_df,
        "coi": mean_coi_df,
        "ir": mean_ir_df,
    }

    ### Manuscript Supp. Tables ###
    supplementary_tables_dir = (
        output_root_directory
        + "/manuscript_materials_evo_app_revision/supplementary/tables"
    )
    make_dir_if_does_not_exist(supplementary_tables_dir)
    # Table C1
    pd.DataFrame(PARAMETER_SETS_DICT).T.to_csv(
        f"{supplementary_tables_dir}/table_c1_parameter_sets.csv",
        header=True,
        index=True,
        index_label="parameter_set_index",
    )
    # Table C2
    get_mean_stats_for_statistic_type(
        stats_df=mean_coi_df,
        path_to_output_csv=f"{supplementary_tables_dir}/table_c2_mean_gw_coi_stats.csv",
    )
    # Table C3
    get_mean_stats_for_statistic_type(
        stats_df=mean_ir_df,
        path_to_output_csv=f"{supplementary_tables_dir}/table_c3_mean_ms_ir_stats.csv",
    )
    # Table C4
    make_data_merged_with_dabest_stats_csv(
        statistic_type="gw_heterozygosity",
        parameter_set_index_list=sorted(PARAMETER_SETS_DICT.keys()),
        model_type_list=VALID_MODEL_TYPES,
        data_df=gw_heterozygosity_df,
        dabest_stat_type="pct_change_between_gens",
        generation_tuple=(0, 40),
        output_file_csv=f"{supplementary_tables_dir}/table_c4_gw_heterozygosity_mean_difference_estimation_stats.csv",
        import_csv_instead_of_making=False,
    )
    # Table C5
    make_data_merged_with_dabest_stats_csv(
        statistic_type="coi",
        parameter_set_index_list=sorted(PARAMETER_SETS_DICT.keys()),
        model_type_list=VALID_MODEL_TYPES,
        data_df=mean_coi_df,
        dabest_stat_type="pct_change_between_gens",
        generation_tuple=(0, 40),
        output_file_csv=f"{supplementary_tables_dir}/table_c5_gw_coi_mean_difference_estimation_stats.csv",
        import_csv_instead_of_making=False,
    )
    # Table C6
    make_data_merged_with_dabest_stats_csv(
        statistic_type="gw_richness",
        parameter_set_index_list=sorted(PARAMETER_SETS_DICT.keys()),
        model_type_list=VALID_MODEL_TYPES,
        data_df=gw_richness_per_individual_df,
        dabest_stat_type="pct_change_between_gens",
        generation_tuple=(0, 40),
        output_file_csv=f"{supplementary_tables_dir}/table_c6_gw_richness_mean_difference_estimation_stats.csv",
        import_csv_instead_of_making=False,
    )

    # Table C7
    stats_dfs = []
    for control in (1, 2, 3):
        for test_scalar in range(1, 7):
            print(
                f"Making plots for control {control} test {control + (test_scalar * 3)} and statistic type gw_heterozygosity"
            )
            stats_df = make_grouped_control_plot_for_statistic_type_and_two_parameter_sets(
                data_df=statistic_type_to_data_df_dict["gw_heterozygosity"],
                model_type_list=VALID_MODEL_TYPES,
                statistic_type="gw_heterozygosity",
                dabest_stat_type="pct_change_between_gens",
                parameter_set_index_control=control,
                parameter_set_index_test=control + (test_scalar * 3),
                generation_tuple=(0, 40),
                stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
                figure_dir_formatted_for_model_type_list=None,
                return_stats_df=True,
                write_figure_png=False,
            )
            stats_dfs.append(stats_df)
    pd.concat(stats_dfs).to_csv(
        f"{supplementary_tables_dir}/table_c7_gw_heterozygosity_grouped_controls_mean_difference_between_parameter_sets.csv",
        header=True,
        index=False,
        float_format="%.4f",
    )

    # Table C8
    stats_dfs = []
    for control in (1, 2, 3):
        for test_scalar in range(1, 7):
            print(
                f"Making plots for control {control} test {control + (test_scalar * 3)} and statistic type gw_richness"
            )
            stats_df = make_grouped_control_plot_for_statistic_type_and_two_parameter_sets(
                data_df=statistic_type_to_data_df_dict["gw_richness"],
                model_type_list=VALID_MODEL_TYPES,
                statistic_type="gw_richness",
                dabest_stat_type="pct_change_between_gens",
                parameter_set_index_control=control,
                parameter_set_index_test=control + (test_scalar * 3),
                generation_tuple=(0, 40),
                stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
                figure_dir_formatted_for_model_type_list=None,
                return_stats_df=True,
                write_figure_png=False,
            )
            stats_dfs.append(stats_df)
    pd.concat(stats_dfs).to_csv(
        f"{supplementary_tables_dir}/table_c8_gw_richness_grouped_controls_mean_difference_between_parameter_sets.csv",
        header=True,
        index=False,
        float_format="%.4f",
    )

    # Table C9
    stats_dfs = []
    for control in (1, 2, 3):
        for test_scalar in range(1, 7):
            print(
                f"Making plots for control {control} test {control + (test_scalar * 3)} and statistic type coi"
            )
            stats_df = make_grouped_control_plot_for_statistic_type_and_two_parameter_sets(
                data_df=statistic_type_to_data_df_dict["coi"],
                model_type_list=VALID_MODEL_TYPES,
                statistic_type="coi",
                dabest_stat_type="pct_change_between_gens",
                parameter_set_index_control=control,
                parameter_set_index_test=control + (test_scalar * 3),
                generation_tuple=(0, 40),
                stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
                figure_dir_formatted_for_model_type_list=None,
                return_stats_df=True,
                write_figure_png=False,
            )
            stats_dfs.append(stats_df)
    pd.concat(stats_dfs).to_csv(
        f"{supplementary_tables_dir}/table_c9_mean_coi_grouped_controls_mean_difference_between_parameter_sets.csv",
        header=True,
        index=False,
        float_format="%.4f",
    )

    ### Manuscript Main Figures ###
    main_figures_dir = (
        output_root_directory + "/manuscript_materials_evo_app_revision/main/figures"
    )
    make_dir_if_does_not_exist(main_figures_dir)

    # Figure 1a
    make_heterozygosity_by_richness_stats_all_model_types_all_generations_one_parameter_set_scatter_plot(
        model_type_list=VALID_MODEL_TYPES,
        parameter_set_index=1,
        stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
        dabest_stat_type="pct_change_between_gens",
        output_png=f"{main_figures_dir}/figure_1a_heterozygosity_by_richness_ps1.eps",
    )

    # Figure 1b
    make_inbreeding_by_richness_stats_all_model_types_all_generations_one_parameter_set_scatter_plot(
        model_type_list=VALID_MODEL_TYPES,
        parameter_set_index=1,
        stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
        dabest_stat_type="pct_change_between_gens",
        output_png=f"{main_figures_dir}/figure_1b_inbreeding_by_richness_ps1.eps",
    )

    # Figure 2A
    make_richness_distance_to_ms_plots(
        richness_per_locus_df=gw_richness_per_locus_df,
        model_type_list=["random", "ms33_ir"],
        statistic_type="gw_richness",
        dabest_stat_type="pct_change_between_gens",
        parameter_set_index=1,
        generation_tuple=(0, 40),
        stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
        write_stats_tsv=False,
        figure_dir_formatted_for_model_type_list=None,
        maximum_haplotype_distance_to_ms=6,
        output_png=f"{main_figures_dir}/figure_2a_richness_by_distance_to_ms_random_vs_ms33_ir.eps",
    )

    # Figure 2B
    make_richness_distance_to_ms_plots(
        richness_per_locus_df=gw_richness_per_locus_df,
        model_type_list=["random", "gw_het"],
        statistic_type="gw_richness",
        dabest_stat_type="pct_change_between_gens",
        parameter_set_index=1,
        generation_tuple=(0, 40),
        stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
        write_stats_tsv=False,
        figure_dir_formatted_for_model_type_list=None,
        maximum_haplotype_distance_to_ms=6,
        output_png=f"{main_figures_dir}/figure_2b_richness_by_distance_to_ms_random_vs_gw_het.eps",
    )

    # Figure 3A
    make_grouped_control_plot_for_statistic_type_and_two_parameter_sets(
        data_df=statistic_type_to_data_df_dict["gw_heterozygosity"],
        model_type_list=VALID_MODEL_TYPES,
        statistic_type="gw_heterozygosity",
        dabest_stat_type="pct_change_between_gens",
        parameter_set_index_control=1,
        parameter_set_index_test=10,
        generation_tuple=(0, 40),
        stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
        write_stats_tsv=False,
        figure_dir_formatted_for_model_type_list=None,
        output_png=f"{main_figures_dir}/figure_3a_gw_heterozygosity_grouped_controls_1_vs_10.eps",
    )

    # Figure 3B
    make_grouped_control_plot_for_statistic_type_and_two_parameter_sets(
        data_df=statistic_type_to_data_df_dict["gw_heterozygosity"],
        model_type_list=VALID_MODEL_TYPES,
        statistic_type="gw_heterozygosity",
        dabest_stat_type="pct_change_between_gens",
        parameter_set_index_control=1,
        parameter_set_index_test=13,
        generation_tuple=(0, 40),
        stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
        write_stats_tsv=False,
        figure_dir_formatted_for_model_type_list=None,
        output_png=f"{main_figures_dir}/figure_3b_gw_heterozygosity_grouped_controls_1_vs_13.eps",
    )

    # Figure 3C
    make_grouped_control_plot_for_statistic_type_and_two_parameter_sets(
        data_df=statistic_type_to_data_df_dict["gw_heterozygosity"],
        model_type_list=VALID_MODEL_TYPES,
        statistic_type="gw_heterozygosity",
        dabest_stat_type="pct_change_between_gens",
        parameter_set_index_control=1,
        parameter_set_index_test=16,
        generation_tuple=(0, 40),
        stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
        write_stats_tsv=False,
        figure_dir_formatted_for_model_type_list=None,
        output_png=f"{main_figures_dir}/figure_3c_gw_heterozygosity_grouped_controls_1_vs_16.eps",
    )

    # Figure 3D
    make_grouped_control_plot_for_statistic_type_and_two_parameter_sets(
        data_df=statistic_type_to_data_df_dict["gw_heterozygosity"],
        model_type_list=VALID_MODEL_TYPES,
        statistic_type="gw_heterozygosity",
        dabest_stat_type="pct_change_between_gens",
        parameter_set_index_control=1,
        parameter_set_index_test=19,
        generation_tuple=(0, 40),
        stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
        write_stats_tsv=False,
        figure_dir_formatted_for_model_type_list=None,
        output_png=f"{main_figures_dir}/figure_3d_gw_heterozygosity_grouped_controls_1_vs_19.eps",
    )

    ### Manuscript Supp. Figures ###
    supplementary_figures_dir = (
        output_root_directory
        + "/manuscript_materials_evo_app_revision/supplementary/figures"
    )
    make_dir_if_does_not_exist(supplementary_figures_dir)
    # Figures B1-3 made in powerpoint
    # Figure B4
    make_heterozygosity_by_richness_stats_all_model_types_all_generations_all_parameter_sets_scatter_plot(
        model_type_list=VALID_MODEL_TYPES,
        stats_dir_formatted_for_model_type_list=STATS_DIR_FORMATTED_FOR_MODEL_TYPE_LIST,
        output_png=f"{supplementary_figures_dir}/figure_s1_heterozygosity_by_richness.png",
        dabest_stat_type="pct_change_between_gens",
        dpi=800,
        import_csv_instead_of_making=True,
    )
