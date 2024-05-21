def error_ignore(old):
    def new(*args, **kwargs):
        res = None
        try:
            res = old(*args, **kwargs)
        except Exception:
            pass
        return res
    return new
