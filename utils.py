from operator import itemgetter

def get_indexed_list(items: list, indices: list):
    return itemgetter(*indices)(items)
