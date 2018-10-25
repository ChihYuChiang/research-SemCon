class UniversalContainer():
    """
    Usage
    - Print object to see all key and data (recursive).
    - listKey() shows all attribute keys of this object (current level).
    - listMethod() shows all methods of this object.
    """

    def __repr__(self, level=0):
        keys = [item for item in dir(self) if not callable(getattr(self, item)) and not item.startswith("__")]
        rep = []

        for key in keys:
            attr = getattr(self, key)
            if isinstance(attr, UniversalContainer):
                rep.append('-' * 3 * level + '.' + key)
                rep.append(attr.__repr__(level + 1))
            else:
                rep.append('-' * 3 * level + '.' + key)
                rep.append('-' * 3 * level + ' ' + str(attr))

        return '\n'.join(rep)

    def listKey(self):
        return [item for item in dir(self) if not callable(getattr(self, item)) and not item.startswith("__")]

    def listMethod(self):
        return [item for item in dir(self) if callable(getattr(self, item)) and not item.startswith("__")]


class ConvertedContainer(UniversalContainer):
    """
    Convert data to object form (recursive).
    """
    
    def __new__(cls, data):
        from collections import Iterable

        if isinstance(data, dict):
            return super().__new__(cls)
        elif isinstance(data, Iterable) and not isinstance(data, str):
            return type(data)(ConvertedContainer(value) for value in data)
        else:
            return data

    def __init__(self, data):
        for i in range(len(data.keys())):
            setattr(self, list(data.keys())[i], ConvertedContainer(list(data.values())[i]))


class SettingContainer(UniversalContainer):
    """
    Usage
    - Convenient keyword = parameter setting.
    - Protected attribute setter. Use `update(key=value)` to modify content.
    """

    def __setattr__(self, key, value):
        class ProhibittedOperation(Exception): pass
        raise self.ProhibittedOperation('Use SettingContainer.update to modify attributes.')

    def update(self, **kwarg):
        for key, value in kwarg.items():
            self.__dict__[key] = value
            # setattr(self, key, value)

    __init__ = update


def getConfigObj(path):
    """
    Read config files into an obj container.
    - Support file type: .json, .yml.
    """
    import yaml
    import re

    class UnknownFileType(Exception): pass

    with open(path, 'r', encoding='utf-8') as f:
        ext = re.search('\.(.+)', path).group(1)
        if ext == 'json': config_dic = json.load(f)
        elif ext == 'yml': config_dic = yaml.load(f)
        else: raise UnknownFileType('\'{}\' is not a supported file type.'.format(ext))

    return ConvertedContainer(config_dic)
