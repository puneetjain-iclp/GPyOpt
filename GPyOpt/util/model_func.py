import numpy as np
import pandas as pd
import re


def cut_array(x):
    """
        input x: array to cut in multiple arrays
        return x: array of cut arrays
        Cut an array in multiple array of the size of the decision_cols
    """
    x = np.array(x)
    x.resize(int(len(x) / len(decision_cols)), len(decision_cols))
    return x


def model_predict(x):
    a = np.copy(x)
    a[x <= -5] = .8
    a[(x > -5) & (x <= 0)] = .6
    a[(x >0) & (x <= 5 )] = .4
    a[x > 5]= .1
    return a


def evaluate_rec(code, x=None, data_cols=None):
    """
        IN Code, string: Code parameter to evaluate
        IN x: None by default can be a dataframe if your formula work on a specific dataframe
        OUT: Evaluated parameter
        Evaluate parameter from the UI, exploring recursively nested parameters
    """
    try:
        if data_cols is None:
            data_cols = x.columns
        return eval(code)
    except NameError as ne:
        name_error = str(ne).split("'")[1]
        if name_error in parameter_dict.keys():
            return evaluate_rec(re.sub(r'\b%s\b' % name_error, parameter_dict[name_error], code), x,
                                data_cols)
        elif name_error in data_cols:
            return evaluate_rec(re.sub(r'\b%s\b' % name_error, "x['" + name_error + "']", code)
                                .replace("x['x[", "x[")
                                .replace("]']", "]"), x, data_cols)
        else:
            print("Value is neither in parameter nor data: " + name_error)
            raise(ne)


def agg_constraint_min(w, dataset, agg_conv, aggc_rdiff, aggc_when):
    df = cut_array(w)
    df = pd.DataFrame(df, columns=decision_cols)
    df = df.reset_index()
    non_dec = dataset.reset_index()
    non_dec['index'] = non_dec.index.astype('int64')
    df = df.merge(non_dec, on='index', how='inner')
    del(df['index'])

    aggc_conv_e = evaluate_rec(aggc_conv, df)
    aggc_rdiff_e = evaluate_rec(aggc_rdiff, df)
    if aggc_when == 'offering_time':
        if aggregative_contraint['aggc_calc'] == 'mean':
            agg_calc = mean(evaluate_rec(aggc_par, df) * model_predict(df, model_cols_opt))\
                / mean(model_predict(df, model_cols_opt))
        else:
            agg_calc = sum(evaluate_rec(aggc_par, df) * model_predict(df, model_cols_opt)) /\
                sum(model_predict(df, model_cols_opt))
    else:
        if aggregative_contraint['aggc_calc'] == 'mean':
            agg_calc = mean(evaluate_rec(aggc_par, df))
        else:
            agg_calc = sum(evaluate_rec(aggc_par, df))
    return agg_calc - aggc_conv_e * (1 - aggc_rdiff_e)


def agg_constraint_max(w, dataset, agg_conv, aggc_rdiff, aggc_when):
    df = cut_array(w)
    df = pd.DataFrame(df, columns=decision_cols)
    df = df.reset_index()
    non_dec = dataset.reset_index()
    non_dec['index'] = non_dec.index.astype('int64')
    df = df.merge(non_dec, on='index', how='inner')
    del(df['index'])

    aggc_conv_e = evaluate_rec(aggc_conv, df)
    aggc_rdiff_e = evaluate_rec(aggc_rdiff, df)
    if aggc_when == 'offering_time':
        if aggregative_contraint['aggc_calc'] == 'mean':
            agg_calc = mean(evaluate_rec(aggc_par, df) * model.model_function(df, model_cols_opt))\
                / mean(model.model_function(df, model_cols_opt))
        else:
            agg_calc = sum(evaluate_rec(aggc_par, df) * model.model_function(df, model_cols_opt)) /\
                sum(model.model_function(df, model_cols_opt))
    else:
        if aggregative_contraint['aggc_calc'] == 'mean':
            agg_calc = mean(evaluate_rec(aggc_par, df))
        else:
            agg_calc = sum(evaluate_rec(aggc_par, df))
    return aggc_conv_e * (1 + aggc_rdiff_e) - agg_calc
