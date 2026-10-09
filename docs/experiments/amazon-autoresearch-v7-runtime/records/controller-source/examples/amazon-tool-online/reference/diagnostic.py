def measure(request):
    values = [row['metrics']['HR@10'] for row in request['runs']]
    return {'measurements': {'category_spread': max(values) - min(values)},
            'limitations': ['Development aggregate spread; not a mechanism or utility claim.']}
