from typing import Any, Dict, List

import copy
import pandas as pd
import tqdm


__all__ = [
    "load_csv_dict",
    "dictionary_to_dataframe",
    "save_dictionary_in_csv",
    "copy_dictionary",
    "merge_list_of_dictionaries",
    "load_and_merge_csv_dicts_in_dir",

    "AttrDict",
    "make_attrdict",
]


def load_csv_dict(
        csv_path: str,
        index_col: Any = 0,
) -> Dict:
    data = pd.read_csv(csv_path, index_col=index_col)

    try:
        return data.to_dict(orient="list")
    except:
        return data


def dictionary_to_dataframe(
        dictionary: Dict,
        index_col: Any = None,
) -> pd.DataFrame:
    df = pd.DataFrame(dictionary)

    if index_col is not None:
        df.set_index(index_col)

    return df


def save_dictionary_in_csv(
        dictionary: Dict,
        save_dir: str,
        save_name: str,
        index_col: Any = None,
):
    from .pathtools import makedir

    makedir(save_dir)
    save_path = f"{save_dir}/{save_name}.csv"

    df = dictionary_to_dataframe(dictionary, index_col)

    if index_col is not None:
        df.to_csv(save_path, mode='w', index=False)
    else:
        df.to_csv(save_path, mode='w')


def copy_dictionary(
        dictionary: Dict,
) -> Dict:
    return copy.deepcopy(dictionary)


def merge_list_of_dictionaries(
        list_of_dictionaries: List[Dict],
        verbose: bool = False,
) -> Dict:
    from .listools import compare_items_in_list, merge_list_of_lists

    num_dicts = len(list_of_dictionaries)

    # keys
    keys = [key for key in list_of_dictionaries[0]]

    # initialize
    merged_dict = {}
    for key in keys:
        merged_dict[key] = []

    # load
    for dict_index in tqdm.trange(
        num_dicts,
        desc=f"Merging {num_dicts} dictionaries...",
        disable=not verbose,
    ):
        dictionary = list_of_dictionaries[dict_index]

        assert compare_items_in_list(keys, [key for key in dictionary])

        for key in keys:
            merged_dict[key].append(dictionary[key])

    # merge
    for key in keys:
        merged_dict[key] = merge_list_of_lists(merged_dict[key])

    return merged_dict


def load_and_merge_csv_dicts_in_dir(
        csv_dicts_dir: str,
        index_col: Any = 0,
        verbose: bool = False,
) -> Dict:
    from .pathtools import get_file_list

    dicts_list = get_file_list(csv_dicts_dir, fext="csv")

    list_of_dictionaries = []

    for csv_dict_index in tqdm.trange(
        len(dicts_list),
        desc=f"Loading .csv dictionaries in {csv_dicts_dir}...",
        disable=not bool(verbose)
    ):
        csv_dict_fname = dicts_list[csv_dict_index]
        dictionary = load_csv_dict(f"{csv_dicts_dir}/{csv_dict_fname}", index_col=index_col)

        list_of_dictionaries.append(dictionary)

    merged_dict = merge_list_of_dictionaries(list_of_dictionaries, verbose)

    return merged_dict


class AttrDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"No such attribute: {name}")


def make_attrdict(
        dictionary: Dict,
) -> AttrDict:
    dictionary = AttrDict(dictionary)

    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = make_attrdict(value)

    return dictionary
