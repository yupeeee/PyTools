from typing import Any, Dict, Iterable, List

from itertools import chain, compress
from more_itertools import locate
import re


__all__ = [
    "argmax_list",
    "argmin_list",
    "indices_of_value_in_list",
    "index_of_first_zero_value_in_list",
    "index_of_first_nonzero_value_in_list",
    "load_txt_to_list",
    "save_list_in_txt",
    "sort_str_list",
    "remove_empty_string_from_str_list",
    "merge_list_of_lists",
    "merge_lists_in_dictionary",
    "average_of_surroundings",
    "compare_items_in_list",
    "topk_index_of_list",
    "index_of_closest_value_in_list",
    "sort_list_by_list",
    "filter_list",
]


def argmax_list(
        lst: List,
) -> int:
    return lst.index(max(lst))


def argmin_list(
        lst: List,
) -> int:
    return lst.index(min(lst))


def indices_of_value_in_list(
        lst: List,
        value: Any,
) -> List[int]:
    return list(locate(lst, lambda x: x == value))


def index_of_first_zero_value_in_list(
        lst: List,
) -> int:
    return next((i for i, x in enumerate(lst) if x == 0), None)


def index_of_first_nonzero_value_in_list(
        lst: List,
) -> int:
    return next((i for i, x in enumerate(lst) if x), None)


def load_txt_to_list(
        path: str,
        dtype: type,
) -> List:
    txt_in_strings = \
        [line.rstrip('\n') for line in open(path, 'r')]
    lst_mapped = map(dtype, txt_in_strings)
    lst = list(lst_mapped)

    return lst


def save_list_in_txt(
        lst: List[Any],
        path: str,
        save_name: str,
) -> None:
    from .pathtools import makedir

    makedir(path)
    save_path = f"{path}/{save_name}.txt"

    lst_str = []

    for i in range(len(lst)):
        lst_str.append(str(lst[i]) + '\n')

    with open(save_path, 'w') as f:
        f.writelines(lst_str)


def sort_str_list(
        str_list: List[str],
        return_indices: bool = False,
) -> Any:
    convert = lambda text: int(text) if text.isdigit() else text

    sorted_str_list = sorted(
        str_list,
        key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)],
    )

    indices = sorted(
        range(len(str_list)),
        key=lambda key: [convert(c) for c in re.split('([0-9]+)', str_list[key])],
    )

    if return_indices:
        return sorted_str_list, indices

    else:
        return sorted_str_list


def remove_empty_string_from_str_list(
        str_list: List[str],
) -> List[str]:
    return [s for s in str_list if len(s)]


def merge_list_of_lists(
        list_of_lists: List[List],
) -> List[Any]:
    return [item for sublist in list_of_lists for item in sublist]


def merge_lists_in_dictionary(
        dictionary: Dict[Any, List],
) -> List[Any]:
    lst = []

    for key in dictionary:
        value: List = dictionary[key]

        lst.append(value)

    lst = merge_list_of_lists(lst)

    return lst


def average_of_surroundings(
        lst: List[Any],
) -> List[Any]:
    def neighbours(items, fill=None):
        before = chain([fill], items)
        after = chain(items, [fill])
        next(after)
        for a, b, c in zip(before, items, after):
            yield [value for value in (a, b, c) if value is not fill]

    return [sum(v) / len(v) for v in neighbours(items=lst)]


def compare_items_in_list(
        l1: List[Any],
        l2: List[Any],
) -> List[bool]:
    assert len(l1) == len(l2)

    return [i == j for i, j in zip(l1, l2)]


def topk_index_of_list(
        lst: List[Any],
        k: int,
        top: bool = True,
) -> List[int]:
    sorted_lst = sorted(range(len(lst)), key=lambda i: lst[i], reverse=top)

    return sorted_lst[:k]


def index_of_closest_value_in_list(
        lst: List[Any],
        val: Any,
) -> Any:
    return min(range(len(lst)), key=lambda i: abs(lst[i] - val))


def sort_list_by_list(
        target_list: List[Any],
        idx_list: List[int],
        reverse: bool = False,
) -> List[Any]:
    assert len(target_list) == len(idx_list)

    return [x for _, x in sorted(zip(idx_list, target_list), reverse=reverse)]


def filter_list(
        original_list: List[Any],
        bool_filter: Iterable[bool],
        reverse_bool: bool = False,
) -> List[Any]:
    if reverse_bool:
        bool_filter = [not v for v in bool_filter]

    return list(compress(original_list, bool_filter))
