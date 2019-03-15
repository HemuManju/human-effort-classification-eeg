class skip(object):
    """A decorator to skip function execution.

    Parameters
    ----------
    f : function
        Any function whose execution need to be skipped.

    Attributes
    ----------
    f

    """

    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        print('skipping : ' + self.f.__name__)
