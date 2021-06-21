import argparse
from typing import Type

import yaml


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
        yaml_str = values.replace("=", ": ").replace(",", "\n")
        args_map = yaml.safe_load(yaml_str)

        setattr(namespace, self.dest, args_map)