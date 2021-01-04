import argparse


class StoreDictKeyPair(argparse.Action):
    """Action that can parse a python from comma-separated key-value pairs passed to the parser."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Parses comma-separated key-value pairs passed to the parser into a dict, adding it to the namespace.

        Args:
            parser: Parser object being parsed.
            namespace: Namespace produced from parsing the arguments.
            values: Values specified for the option using the current action.
            option_string: Option flag.
        """
        args_map = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            args_map[k] = v
        setattr(namespace, self.dest, args_map)
