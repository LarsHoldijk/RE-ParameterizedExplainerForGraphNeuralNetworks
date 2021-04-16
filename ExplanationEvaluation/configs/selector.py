import json


class Struct:
    """Helper class to parse dict to object"""
    def __init__(self, entries):
        self.__dict__.update(entries)

class Selector:
    def __init__(self, config_path):
        self.args = self.parse_config(config_path)

    def parse_config(self, config_path):
        try:
            with open(config_path) as config_parser:
                config = json.loads(json.dumps(json.load(config_parser)), object_hook=Struct)
            return config
        except FileNotFoundError:
            print("No config found")
            return None
