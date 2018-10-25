#--Universal container
class UniversalContainer():

    def listData(self):
        data = [(item, getattr(self, item)) for item in dir(self) if not callable(getattr(self, item)) and not item.startswith("__")]
        for item in data:
            try:
                item[1].listData()
                print(item)
            except AttributeError:
                print(item)

    def listMethod(self):
        print([item for item in dir(self) if callable(getattr(self, item)) and not item.startswith("__")])


#--Convert data to object form (recursive)
class ConvertedContainer(UniversalContainer):
    
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


#--Setting container for convenient keyword=parameter setting
class SettingContainer(UniversalContainer):

    def update(self, **kwarg):
        for key, value in kwarg.items():
            setattr(self, key, value)

    __init__ = update


#--Read config files into obj container
def getConfigObj(path):
    """
    Support file type: .json, .yml
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
