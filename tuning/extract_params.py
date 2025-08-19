def extract_hyperparameter_schema(model):
    schema = {}
    params = model.get_params()

    for param, value in params.items():
        lower_name = param.lower()
        if lower_name == 'n_jobs':
            # For n_jobs, set the search space to just [-1]
            schema[param] = [-1]
        elif isinstance(value, bool):
            schema[param] = [True, False]
        elif isinstance(value, str):
            # For known parameters, we supply a list of possible options.
            if lower_name == 'kernel':
                schema[param] = ['linear', 'poly', 'rbf', 'sigmoid']
            elif lower_name == 'gamma':
                schema[param] = ['scale', 'auto']
            else:
                schema[param] = [value]
        elif isinstance(value, (int, float)):
            if any(keyword in lower_name for keyword in ['iteration', 'batch', 'epoch']):
                schema[param] = [100, 1000, 10000]
            else:
                schema[param] = [1, 2, 3, 4, 5]
        else:
            schema[param] = [value]

    return schema

# Example usage:
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    schema = extract_hyperparameter_schema(model)
    for k, v in schema.items():
        print(f"{k}: {v}")

