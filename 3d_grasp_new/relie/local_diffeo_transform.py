class LocalDiffeoTransform:
    """
    This is a generalization of torch.distributions.transforms.Transform,
    where the transform is a local diffeomorphism and thus has a discrete set as inverse.
    """

    event_dim = 0

    def __init__(self, cache_size=1):
        self._cache_size = cache_size
        self._inv = None
        if cache_size == 0:
            pass  # default behavior
        elif cache_size == 1:
            self._cached_x_y = None, None
            self._cached_xset_y = None, None
        else:
            raise ValueError("cache_size must be 0 or 1")

    def __eq__(self, other):
        return self is other

    def __call__(self, x):
        """
        Computes the transform `x => y`.
        """
        if self._cache_size == 0:
            return self._call(x)
        x_old, y_old = self._cached_x_y
        if x is x_old:
            return y_old
        y = self._call(x)
        self._cached_x_y = x, y
        return y

    def inverse_set(self, y):
        """
        Return preimage: `y => {x: f(x) = y}`.
        :param y: batch_shape + event_shape
        :return: (x, xset, mask)
        x : The "primary" inverse of shape of y
        xset : {x' | x'=f(x) && x' != x} of shape (?, *x.shape)
        mask : boolean tensor of shape of xset potentially making elements of xset be ignored
        """
        if self._cache_size == 0:
            return self._inverse_set(y)

        # If xset immedietely available
        xset_old, y_old = self._cached_xset_y
        if y is y_old and xset_old is not None:
            return xset_old

        # If instead, x available and _xset() available
        x_old, y_old = self._cached_x_y
        if y is y_old:
            xset = self._xset(x_old)
            if xset is not None:
                self._cached_xset_y = xset, y
                return xset

        # Otherwise, compute real inverse
        xset = self._inverse_set(y)
        self._cached_xset_y = xset, y
        return xset

    def _call(self, x):
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def _inverse_set(self, y):
        """
        Abstract method to compute inverse set {x : f(x)=y}.
        :param y: Tensor
        :return: (x, xset, mask)
        x : The "primary" inverse of shape of y
        xset : {x' | x'=f(x) && x' != x} of shape (?, *x.shape)
        mask : boolean tensor of shape of xset potentially making elements of xset be ignored
        """
        raise NotImplementedError

    def _xset(self, x):
        """
        Return preimage of f(x), optional.
        :param x:
        :return: (x, xset, mask)
        x : the input
        xset : {x' | x'=f(x) && x' != x} of shape (?, *x.shape)
        mask : boolean tensor of shape of xset potentially making elements of xset be ignored
        """
        return None

    def log_abs_det_jacobian(self, x, y):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """
        raise NotImplementedError
