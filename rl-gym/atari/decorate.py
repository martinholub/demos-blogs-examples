def func_args(logger = None):
    """Function decorator logging arguments of functions.

    func_args: allows passing of parameters to decorator (e.g. Logger instance)
    func_decorator: decorates function and logs its arguments

    Parameters:
    ----------
    logger: instance of Logger class
    """
    def func_decorator(func):

        from functools import wraps
        @wraps(func)
        def wrapper(*dec_fn_args, **dec_fn_kwargs):
            func_name = func.__name__
            # get function params (args and kwargs)
            arg_names = func.__code__.co_varnames
            params = dict(
                args=dict(zip(arg_names, dec_fn_args)),
                kwargs=dec_fn_kwargs)

            if logger is not None:
                logger.debug("Function {} with args:".format(func_name))
                logger.debug(', '.join(['{}={}'.format(str(k), repr(v))
                            for k, v in params.items()]))
            else:
                print("Function {} with args:".format(func_name))
                print(', '.join(['{}={}'.format(str(k), repr(v))
                                for k, v in params.items()]))

            out = func(*dec_fn_args, **dec_fn_kwargs)
            return out

        return wrapper

    return func_decorator
