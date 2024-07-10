""" Utilities related to processing command line arguments """

def substitute_vars(args, substitutions):
    """ Substitute variables in args that are enclosed in curly braces with values from env_vars.

    Example:
        `substitute_vars(args, {**os.environ, 'RUN_ID': args.run_id})`

    """
    substitutions = {k:v.format(**substitutions) for k,v in substitutions.items() if v is not None} # Substitute values in `subsitutions` too in case they also have {} variables. None values are removed.
    for k,v in vars(args).items():
        if type(v) is str:
            v = v.format(**substitutions)
            setattr(args, k, v)
    return args
