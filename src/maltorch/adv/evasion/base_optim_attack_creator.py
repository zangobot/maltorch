import importlib
from abc import abstractmethod
from typing import Type

from maltorch.adv.evasion.backend_attack import BackendAttack


class OptimizerBackends:
    """Available backends."""

    NG = "nevergrad"
    GRADIENT = "gradient"


class BaseOptimAttackCreator:
    """Generic creator for attacks."""

    @classmethod
    def get_implementation(cls, backend: str) -> Type[BackendAttack]:
        """
        Get the implementation of the attack with the given backend.

        Parameters
        ----------
        backend : str
            The backend for the attack. See secmlt.adv.backends for
            available backends.

        Returns
        -------
        BaseEvasionAttack
            Attack implementation.
        """
        implementations = {
            OptimizerBackends.NG: cls.get_nevergrad_implementation,
            OptimizerBackends.GRADIENT: cls._get_native_implementation,
        }
        cls.check_backend_available(backend)
        return implementations[backend]()

    @classmethod
    def check_backend_available(cls, backend: str) -> bool:
        """
        Check if a given backend is available for the attack.

        Parameters
        ----------
        backend : str
            Backend string.

        Returns
        -------
        bool
            True if the given backend is implemented.

        Raises
        ------
        NotImplementedError
            Raises NotImplementedError if the requested backend is not in
            the list of the possible backends (check secmlt.adv.backends).
        """
        if backend in cls.get_backends():
            return True
        msg = "Unsupported or not-implemented backend."
        raise NotImplementedError(msg)

    @classmethod
    def get_nevergrad_implementation(cls) -> BackendAttack:
        """
        Get the Nevergrad implementation of the attack.

        Returns
        -------
        BaseEvasionAttack
            Nevergrad implementation of the attack.

        Raises
        ------
        ImportError
            Raises ImportError if Nevergrad extra is not installed.
        """
        if importlib.util.find_spec("nevergrad", None) is not None:
            return cls._get_nevergrad_implementation()
        msg = "Nevergrad extra not installed."
        raise ImportError(msg)

    @staticmethod
    def _get_nevergrad_implementation() -> BackendAttack:
        msg = "Nevergrad implementation not available."
        raise NotImplementedError(msg)

    @staticmethod
    def _get_native_implementation() -> BackendAttack:
        msg = "Native implementation not available."
        raise NotImplementedError(msg)

    @staticmethod
    @abstractmethod
    def get_backends() -> set[str]:
        """
        Get the available backends for the given attack.

        Returns
        -------
        set[str]
            Set of implemented backends available for the attack.

        Raises
        ------
        NotImplementedError
            Raises NotImplementedError if not implemented in the inherited class.
        """
        msg = "Backends should be specified in inherited class."
        raise NotImplementedError(msg)
