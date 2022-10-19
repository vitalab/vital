import argparse
from typing import Any, Callable, Tuple, Type, TypeVar, Union

import yaml

T = TypeVar("T")


class StoreDictKeyPair(argparse.Action):
    """Action that can parse a python dictionary from comma-separated key-value pairs passed to the parser."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Parses comma-separated key-value pairs passed to the parser into a dict, adding it to the namespace.

        Args:
            parser: Parser object being parsed.
            namespace: Namespace produced from parsing the arguments.
            values: Values specified for the option using the current action.
            option_string: Option flag.
        """
        # Hack converting `values` to a YAML document to use the YAML parser's type inference
        args_map = yaml_flow_collection(values, mapping_separator="=")
        setattr(namespace, self.dest, args_map)


def yaml_flow_collection(
    val: str,
    collection_entries_separator: str = ",",
    mapping_separator: str = ":",
    sequence_markers: Tuple[str, str] = ("[", "]"),
    mapping_markers: Tuple[str, str] = ("{", "}"),
) -> Any:
    """Parses a string as a YAML flow collection.

    References:
        - YAML flow collection specification, for more details: https://yaml.org/spec/1.2.2/#74-flow-collection-styles

    Args:
        val: String representation of the flow collection to parse.
        collection_entries_separator: Separator to use to split collection entries.
        mapping_separator: Separator to use to split key-value pairs in flow mappings.
        sequence_markers: Characters denoting the beginning/end of a flow sequence.
        mapping_markers: Characters denoting the beginning/end of a flow mapping.

    Returns:
        Native Python data structure representation of the YAML flow collection parsed from the string.
    """
    yaml_str = (
        val.replace(collection_entries_separator, ",")
        .replace(mapping_separator, ": ")
        .replace(sequence_markers[0], "[")
        .replace(sequence_markers[1], "]")
        .replace(mapping_markers[0], "{")
        .replace(mapping_markers[1], "}")
    )
    return yaml.safe_load(yaml_str)


def dtype_tuple(val: str, dtype: Callable[[str], T] = str, separator: str = ",") -> Tuple[T, ...]:
    """Parses a string as a tuple of dtypes, separated by `separator`.

    Args:
        val: String representation of the tuple to parse.
        dtype: Callable able to parse a value of the expected dtype from a string.
        separator: Separator to use to split the string.

    Returns:
        Tuple of dtypes parsed from the string.
    """
    return tuple(dtype(x) for x in val.split(separator))


def int_or_float(val: str) -> Union[int, float]:
    """Parses a string as either an integer or a float.

    Args:
        val: String representation of an integer/float to parse.

    Returns:
        Integer/float value parsed from the string.
    """
    try:
        cast_val = int(val)
    except ValueError:
        try:
            cast_val = float(val)
        except ValueError:
            raise
    return cast_val


def get_classpath_group(parser: argparse.ArgumentParser, cls: Type) -> argparse._ArgumentGroup:
    """Fetches an argument group named after the provided class' qualified name from the argument parser.

    Hack because it relies on some private class types and fields of the argument parser.

    Args:
        parser: Argument parser from which to get/create an argument group.
        cls: Class whose qualified name to use for the argument group.

    Returns:
        An argument group in the parser named after the provided class' qualified name.
    """
    cls_path = f"{cls.__module__}.{cls.__qualname__}"
    cls_group = [arg_group for arg_group in parser._action_groups if arg_group.title == cls_path]
    if cls_group:  # If an existing group matches the requested class
        cls_group = cls_group[0]
    else:  # Else create a new group for the requested class
        cls_group = parser.add_argument_group(cls_path)
    return cls_group
