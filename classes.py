from typing import Union, Any, Optional
from dataclasses import dataclass

import numpy as np
import warnings
import data


Real = Union[int, float, np.integer, np.floating]
Float = Union[float, np.floating]
Int = Union[int, np.integer]


@dataclass
class Orbit:
    a: Real                    # Semimajor axis (a): kilometers (km)
    e: Real                    # Eccentricity (e): dimensionless (-)
    i: Real                    # Inclination (i): radians (rad)
    M: np.ndarray              # Mean anomaly (M): radians (rad)
    omega: Real                # Longitude of the ascending node (Ω): radians (rad)
    arg_per: Real              # Argument of periapsis/perigee (ω): radians (rad)

    def __post_init__(self):
        self._validate_attr(self.a, "a", min_lim=0, max_lim=None, min_exclusive=True, max_exclusive=False)
        self._validate_attr(self.e, "e", min_lim=0, max_lim=1, min_exclusive=False, max_exclusive=True)
        self._validate_attr(self.i, "i", min_lim=0, max_lim=np.pi, min_exclusive=False, max_exclusive=False)
        self._validate_attr(self.omega, "omega", min_lim=0, max_lim=2 * np.pi, min_exclusive=False, max_exclusive=False)
        self._validate_attr(self.arg_per, "arg_per", min_lim=0, max_lim=2 * np.pi, min_exclusive=False, max_exclusive=False)

        if isinstance(self.M, np.ndarray):
            if self.M.ndim > 1:
                warnings.warn(f"Expected 1D array, got {self.M.ndim}D. Input will be flattened", UserWarning, stacklevel=2)
                self.M = self.M.ravel()

            if not (np.issubdtype(self.M.dtype, np.floating) or np.issubdtype(self.M.dtype, np.integer)):
                raise TypeError(f"Attribute 'M' must have the dtype that is a subdtype of numpy.integer or numpy.floating")

            for i in range(len(self.M)):
                self._validate_value(self.M[i], f"M[{i}]", min_lim=0, max_lim=2 * np.pi, min_exclusive=False, max_exclusive=False)

    def _validate_attr(self, value: Any, arg_name: str, min_lim: Optional[Real], max_lim: Optional[Real], min_exclusive: bool = False, max_exclusive: bool = False) -> None:
        self._validate_type(value, arg_name)
        self._validate_value(value, arg_name, min_lim, max_lim, min_exclusive, max_exclusive)

    @staticmethod
    def _validate_type(value: Any, arg_name: str) -> None:
        if not isinstance(value, (np.integer, np.floating, int, float)):
            raise TypeError(f"Attribute '{arg_name}' must be a real number")

    @staticmethod
    def _validate_value(value: Real, arg_name: str, min_lim: Optional[Real], max_lim: Optional[Real], min_exclusive: bool = False, max_exclusive: bool = False) -> None:
        if min_lim is None:
            check = value >= max_lim if max_exclusive else value > max_lim
            sym = ")" if max_exclusive else "]"

            if check:
                raise ValueError(f"Attribute '{arg_name}' must be in (-∞; {max_lim}{sym}")
        elif max_lim is None:
            check = value <= min_lim if min_exclusive else value < min_lim
            sym = "(" if min_exclusive else "["

            if check:
                raise ValueError(f"Attribute '{arg_name}' must be in {sym}{min_lim}; +∞)")
        else:
            check_2 = value >= max_lim if max_exclusive else value > max_lim
            sym_2 = ")" if max_exclusive else "]"

            check_1 = value <= min_lim if min_exclusive else value < min_lim
            sym_1 = "(" if min_exclusive else "["

            if check_1 or check_2:
                raise ValueError(f"Attribute '{arg_name}' must be in {sym_1}{min_lim}; {max_lim}{sym_2}")

class CoordValidator:
    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __set__(self, instance, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Parameter '{self.name}' must be a numpy.ndarray")
        elif not (np.issubdtype(value.dtype, np.floating) or np.issubdtype(value.dtype, np.integer)):
            raise TypeError(f"Parameter '{self.name}' must have the dtype that is a subdtype of numpy.integer or numpy.floating")
        elif not (value.ndim == 2 and value.shape[0] == 3):
            raise ValueError(f"Parameter '{self.name}' must be 2D array with the first axis equal three, i. e. (3, n)")

        setattr(instance, self.name, value)

    def __get__(self, instance, owner):
        return getattr(instance, self.name, None)

class CoordTransformer:
    mu = data.mu
    omega_earth = data.omega_earth

    eci_coords = CoordValidator()
    gcs_coords = CoordValidator()

    def __init__(self, orbit: Orbit):
        self.orbit = orbit

        self._eci_coords = None
        self._gcs_coords = None
        self._gscs_coords = None

    def transform_to_eci(self, eps: Float = 1e-6, iter_lim: Int = 1000) -> np.ndarray:
        # Earth-centered inertial coordinate system

        e_anom = self.calculate_eccent_anom(eps=eps, iter_lim=iter_lim)

        true_anom = 2 * np.arctan2(np.sqrt(1 + self.orbit.e) * np.sin(e_anom / 2), np.sqrt(1 - self.orbit.e) * np.cos(e_anom / 2))
        true_anom[true_anom < 0] += 2 * np.pi

        u = true_anom + self.orbit.arg_per
        r_a = self.orbit.a * (1 - self.orbit.e ** 2) / (1 + self.orbit.e * np.cos(true_anom))

        x_a = r_a * (np.cos(u) * np.cos(self.orbit.omega) - np.sin(u) * np.sin(self.orbit.omega) * np.cos(self.orbit.i))
        y_a = r_a * (np.cos(u) * np.sin(self.orbit.omega) + np.sin(u) * np.cos(self.orbit.omega) * np.cos(self.orbit.i))
        z_a = r_a * np.sin(u) * np.sin(self.orbit.i)

        self.eci_coords = np.vstack([x_a, y_a, z_a])

        return u

    def transform_to_gcs(self, tao: Float = 0.0, s0: Float = 0.0) -> None:
        # Geographic coordinate system

        if self.eci_coords is None:
            raise AttributeError("Transforming coordinates to GCS requires first transforming them to ECI.")

        period = 2 * np.pi * np.sqrt((self.orbit.a ** 3) / self.mu)
        n = 2 * np.pi / period
        t = self.orbit.M / n + tao

        s_t = self.omega_earth * t + s0
        s_t = s_t % (2 * np.pi)

        cos_s, sin_s = np.cos(s_t), np.sin(s_t)

        gcs_coords = np.vstack([cos_s * self.eci_coords[0] + sin_s * self.eci_coords[1],
                               -sin_s * self.eci_coords[0] + cos_s * self.eci_coords[1],
                               self.eci_coords[2]])

        self.gcs_coords = gcs_coords

    def transform_to_gscs(self) -> None:
        # Geocentric spherical coordinate system

        if self.gcs_coords is None:
            raise AttributeError("Transforming coordinates to GSCS requires first transforming them to GCS.")

        r = np.sqrt(self.gcs_coords[0] ** 2 + self.gcs_coords[1] ** 2 + self.gcs_coords[2] ** 2)
        phi = np.arcsin(self.gcs_coords[2] / r)

        lmbd = np.arctan2(self.gcs_coords[1], self.gcs_coords[0])
        lmbd[lmbd < 0] += 2 * np.pi

        self.gscs_coords = np.vstack([r, phi, lmbd])

    def calculate_eccent_anom(self, eps: Float = 1e-6, iter_lim: Int = 1000) -> np.ndarray:
        curr_anom, i = self.orbit.M, 0
        next_anom = self.orbit.M + self.orbit.e * np.sin(curr_anom)

        while np.any(np.abs(next_anom - curr_anom) > eps) and (i < iter_lim):
            curr_anom = next_anom
            next_anom = self.orbit.M + self.orbit.e * np.sin(curr_anom)
            i += 1

        return next_anom

    @property
    def gscs_coords(self):
        return self._gscs_coords

    @gscs_coords.setter
    def gscs_coords(self, gscs_coords):
        if not isinstance(gscs_coords, np.ndarray):
            raise TypeError("Parameter 'gscs_coords' must be a numpy.ndarray")
        elif not (np.issubdtype(gscs_coords.dtype, np.floating) or np.issubdtype(gscs_coords.dtype, np.integer)):
            raise TypeError("Parameter 'gscs_coords' must have the dtype that is a subdtype of numpy.integer or numpy.floating")
        elif not (gscs_coords.ndim == 2 and gscs_coords.shape[0] == 3):
            raise ValueError("Parameter 'gscs_coords' must be 2D array with the first axis equal three, i. e. (3, n)")
        elif np.any(gscs_coords[0] <= 0):
            raise ValueError("Parameter 'gscs_coords' must contain positive values in the first row")
        elif np.any((gscs_coords[1] < -np.pi / 2) | (gscs_coords[1] > np.pi / 2)):
            raise ValueError("Parameter 'gscs_coords' must contain values belonging to [-0.5π; 0.5π] in the second row")
        elif np.any((gscs_coords[2] < 0) | (gscs_coords[2] > 2 * np.pi)):
            raise ValueError("Parameter 'gscs_coords' must contain values belonging to [0; 2π] in the third row")

        self._gscs_coords = gscs_coords

    @property
    def orbit(self):
        return self._orbit

    @orbit.setter
    def orbit(self, orbit):
        if not isinstance(orbit, Orbit):
            raise TypeError("Parameter 'orbit' must be an Orbit object")

        self._orbit = orbit


class PositiveInteger:
    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __set__(self, instance, value):
        if not isinstance(value, (np.integer, int)):
            raise TypeError(f"{self.name} must be an integer")
        elif value < 1:
            raise ValueError(f"{self.name} must be positive")

        setattr(instance, self.name, value)

    def __get__(self, instance, owner):
        return getattr(instance, self.name, None)


class AccelerComputer:
    n_max = PositiveInteger()

    J = data.J
    c_h = data.C
    s_h = data.S

    r0, mu = data.r0, data.mu

    def __init__(self, orbit: Orbit, coords: CoordTransformer, n_max: Int):
        self.orbit = orbit
        self.coords = coords
        self.n_max = n_max # inclusive

        if n_max > np.min([data.S.shape[0], data.C.shape[0]]):
            raise ValueError(f"'n_max' must be <= {np.min(data.S.shape[0], data.C.shape[0])}")

        self.__P, self.__dP = None, None
        self.__acceleration, self._comp = None, None

    def calculate_associated_legendre(self) -> None:
        ### When m > n: P_n^m == 0
        n_max = self.n_max
        n_points = self.coords.gscs_coords.shape[1]
        x = np.sin(self.coords.gscs_coords[1, :])
        cos_phi = np.cos(self.coords.gscs_coords[1, :])

        # P[n, m, n_point]
        P = np.zeros((n_max + 1, n_max + 1, n_points), dtype=np.float64)

        P[0, 0, :] = 1.0
        if n_max >= 1:
            P[1, 0, :] = x
            P[1, 1, :] = -cos_phi

        for n in range(2, n_max + 1):
            # Diagonal: P_n^n
            P[n, n, :] = -(2 * n - 1) * cos_phi * P[n - 1, n - 1, :]

            # The first subdiagonal: P_n^(n-1)
            P[n, n - 1, :] = (2 * n - 1) * x * P[n - 1, n - 1, :]

            # The rest: P_n,m (m < n-1)
            for m in range(0, n - 1):
                P[n, m, :] = ((2 * n - 1) * x * P[n - 1, m, :] - (n + m - 1) * P[n - 2, m, :]) / (n - m)

        self.__P = P

    def calculate_associated_legendre_derivative(self) -> None:
        if self.__P is None:
            raise AttributeError("Before running this method you must run calculate_associated_legendre")

        n_max, P = self.n_max, self.__P
        n_points = self.coords.gscs_coords.shape[1]
        x = np.sin(self.coords.gscs_coords[1, :])

        dP = np.zeros((n_max + 1, n_max + 1, n_points), dtype=np.float64)

        denom = x ** 2 - 1.0
        denom = np.clip(denom, -1.0, -1e-12)

        dP[0, 0, :] = 0.0

        for n in range(1, n_max + 1):
            for m in range(0, n + 1):
                dP[n, m, :] = (n * x * P[n, m, :] - (n + m) * P[n - 1, m, :]) / denom

        self.__dP = dP

    def calculate_acceleration(self):
        if self.__P is None or self.__dP is None:
            raise AttributeError("Before running this method you must run calculate_associated_legendre and calculate_associated_legendre_derivative")

        result, coords = np.zeros_like(self.coords.gscs_coords, dtype=np.float64), self.coords.gscs_coords # (r, phi, lmbd)
        P, dP = self.__P, self.__dP
        r0, mu = self.r0, self.mu
        n_max = self.n_max
        J, C, S = self.J, self.c_h, self.s_h

        mu_r0 = mu / (r0 * r0)
        r0_r = r0 / coords[0, :]
        for n in range(2, n_max + 1):
            r0_r_power = r0_r ** (n + 2)

            result[0, :] += (n + 1) * J[n] * r0_r_power * P[n, 0, :]
            result[1, :] += J[n] * r0_r_power * dP[n, 0, :]

            for k in range(1, n + 1):
                cos_k_lmbd, sin_k_lmbd = np.cos(k * coords[2, :]), np.sin(k * coords[2, :])

                result[0, :] -= (n + 1) * r0_r_power * P[n, k, :] * (C[n, k] * cos_k_lmbd + S[n, k] * sin_k_lmbd)
                result[1, :] -= r0_r_power * (C[n, k] * cos_k_lmbd + S[n, k] * sin_k_lmbd) * dP[n, k, :]
                result[2, :] -= r0_r_power * P[n, k, :] * (-C[n, k] * sin_k_lmbd + S[n, k] * cos_k_lmbd)

        cos_phi = np.cos(coords[1, :])
        pole_mask = np.abs(cos_phi) < 1e-10

        result[0, :] *= mu_r0
        result[1, :] *= cos_phi * mu_r0
        result[2, :] *= mu_r0 / np.where(pole_mask, 1.0, cos_phi)
        result[2, pole_mask] = 0.0

        self.__P, self.__dP = None, None
        self.__acceleration = result

    def calculate_components(self, u: np.ndarray):
        if self.__acceleration is None:
            raise AttributeError("Before running this method you must run calculate_acceleration")

        if not isinstance(u, np.ndarray):
            raise TypeError("Parameter 'u' must be a numpy.ndarray")
        elif not (np.issubdtype(u.dtype, np.integer) or np.issubdtype(u.dtype, np.floating)):
            raise TypeError("Parameter 'u' must have a numeric dtype, i. e. either numpy.integer or numpy.floating")
        elif u.ndim != 1 or u.shape[0] != self.coords.gscs_coords.shape[1]:
            raise ValueError(f"Parameter 'u' must have ({self.coords.gscs_coords.shape[1]},) shape")
        elif np.any((u < 0) | (u > 2 * np.pi)):
            raise ValueError(f"Parameter 'u' must have values in [0, 2π]")

        cos_phi = np.cos(self.coords.gscs_coords[1, :])
        pole_mask = np.abs(cos_phi) < 1e-10
        cos_phi = np.where(pole_mask, 1, cos_phi)

        sin_A = np.cos(self.orbit.i) / cos_phi
        cos_A = np.cos(u) * np.sin(self.orbit.i) / cos_phi

        cos_A[pole_mask], sin_A[pole_mask] = np.nan, np.nan

        self.comp = np.empty_like(self.__acceleration, dtype=np.float64)
        self.comp[0, :] = self.__acceleration[0, :]
        self.comp[1, :] = self.__acceleration[1, :] * cos_A + self.__acceleration[2, :] * sin_A
        self.comp[2, :] = self.__acceleration[1, :] * sin_A - self.__acceleration[2, :] * cos_A

    @property
    def comp(self):
        return self._comp

    @comp.setter
    def comp(self, comp):
        if not isinstance(comp, np.ndarray):
            raise TypeError("Parameter 'comp' must be a numpy.ndarray")
        elif not (np.issubdtype(comp.dtype, np.integer) or np.issubdtype(comp.dtype, np.floating)):
            raise TypeError("Parameter 'comp' must have a numeric dtype, i. e. either numpy.integer or numpy.floating")

        self._comp = comp

    @property
    def orbit(self):
        return self._orbit

    @orbit.setter
    def orbit(self, orbit):
        if not isinstance(orbit, Orbit):
            raise TypeError("Parameter 'orbit' must be an Orbit object")

        self._orbit = orbit

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        if not isinstance(coords, CoordTransformer):
            raise TypeError("Parameter 'coords' must be a CoordTransformer")
        elif coords.gscs_coords is None:
            raise AttributeError("Calculation of the acceleration requires first transforming coordinates to GSCS, i. e. coords.gscs_coords != None")

        self._coords = coords