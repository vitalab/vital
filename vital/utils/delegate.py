import inspect


def _custom_dir(c, add):
    return dir(type(c)) + list(c.__dict__.keys()) + add


def delegate_inheritance(to=None, keep=False):
    """Replaces ``**kwargs`` in signature with params from ``to``.

    Args:
        to: Decorated object.
        keep: If ``True``, keeps ``**kwargs``, with its values, as part of the signature.

    Returns:
        Decorated object where ``**kwargs`` in signature is replaced with params from ``to``.
    """

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop("kwargs")
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd
        }
        sigd.update(s2)
        if keep:
            sigd["kwargs"] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


class DelegateComposition:
    """Base class for attr accesses in ``self._xtra`` passed down to ``self.default``."""

    @property
    def _xtra(self):
        return [o for o in dir(self.default) if not o.startswith("_")]

    def __getattr__(self, k):  # noqa: D105
        if k in self._xtra:
            return getattr(self.default, k)
        raise AttributeError(k)

    def __dir__(self):  # noqa: D105
        return _custom_dir(self, self._xtra)
