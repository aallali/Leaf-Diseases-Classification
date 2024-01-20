

def ft_flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(ft_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def ft_print_pretty(data, level=0):
    for key, value in data.items():
        print('\t' * level + str(key) + (" : " + str(value) if not isinstance(value, dict) else ""))
        if isinstance(value, dict):
            ft_print_pretty(value, level + 1)