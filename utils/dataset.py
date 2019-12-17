from enum import Enum


class ConfigurationLabel(Enum):

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @classmethod
    def list(cls):
        return [e for e in cls]

    @classmethod
    def values(cls):
        return [e.value for e in cls]

    @classmethod
    def count(cls):
        return sum(1 for _ in cls)

    @classmethod
    def from_name(cls, name):
        try:
            return cls[name.upper()]
        except KeyError:
            return name
