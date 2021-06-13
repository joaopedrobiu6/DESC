import numpy as np

from desc.backend import jnp
from desc.utils import sign, copy_coeffs
from desc.grid import Grid, LinearGrid
from desc.basis import DoubleFourierSeries, ZernikePolynomial
from desc.transform import Transform
from .core import Surface, cart2polvec, pol2cartvec

__all__ = ["FourierRZToroidalSurface", "ZernikeToroidalSection"]


class FourierRZToroidalSurface(Surface):
    """Toroidal surface represented by a double fourier series in poloidal angle
    theta and toroidal angle phi/zeta

    Parameters
    ----------
    R_mn, Z_mn : array-like, shape(k,)
        Fourier coefficients for R and Z in cylindrical coordinates
    modes_R : array-like, shape(k,2)
        poloidal and toroidal mode numbers [m,n] for R_n.
    modes_Z : array-like, shape(k,2)
        mode numbers associated with Z_n, defaults to modes_R
    NFP : int
        number of field periods
    sym : bool
        whether to enforce stellarator symmetry. Default is "auto" which enforces if
        modes are symmetric. If True, non-symmetric modes will be truncated.
    grid : Grid
        default grid for computation
    name : str
        name for this surface

    """

    _io_attrs_ = Surface._io_attrs_ + [
        "_R_mn",
        "_Z_mn",
        "_R_basis",
        "_Z_basis",
        "_R_transform",
        "_Z_transform",
        "_NFP",
    ]

    def __init__(
        self,
        R_mn=None,
        Z_mn=None,
        modes_R=None,
        modes_Z=None,
        NFP=1,
        sym="auto",
        grid=None,
        name=None,
    ):

        if R_mn is None:
            R_mn = np.array([10, 1])
            modes_R = np.array([[0, 0], [1, 0]])
        if Z_mn is None:
            Z_mn = np.array([0, 1])
            modes_Z = np.array([[0, 0], [-1, 0]])
        if modes_Z is None:
            modes_Z = modes_R

        MR = np.max(abs(modes_R[:, 0]))
        NR = np.max(abs(modes_R[:, 1]))
        MZ = np.max(abs(modes_Z[:, 0]))
        NZ = np.max(abs(modes_Z[:, 1]))
        if sym == "auto":
            if np.all(
                R_mn[np.where(sign(modes_R[:, 0]) != sign(modes_R[:, 1]))] == 0
            ) and np.all(
                Z_mn[np.where(sign(modes_Z[:, 0]) == sign(modes_Z[:, 1]))] == 0
            ):
                sym = True
            else:
                sym = False

        self._R_basis = DoubleFourierSeries(
            M=MR, N=NR, NFP=NFP, sym="cos" if sym else False
        )
        self._Z_basis = DoubleFourierSeries(
            M=MZ, N=NZ, NFP=NFP, sym="sin" if sym else False
        )

        self._R_mn = copy_coeffs(R_mn, modes_R, self.R_basis.modes[:, 1:])
        self._Z_mn = copy_coeffs(Z_mn, modes_Z, self.Z_basis.modes[:, 1:])
        self._NFP = NFP
        self._sym = sym

        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self._grid = grid
        self._R_transform, self._Z_transform = self._get_transforms(grid)
        self.name = name

    @property
    def NFP(self):
        """number of toroidal field periods"""
        return self._NFP

    @property
    def R_basis(self):
        """Spectral basis for R double fourier series"""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z double fourier series"""
        return self._Z_basis

    @property
    def grid(self):
        """Default grid for computation"""
        return self._grid

    @grid.setter
    def grid(self, new):
        if isinstance(new, Grid):
            self._grid = new
        elif isinstance(new, (np.ndarray, jnp.ndarray)):
            self._grid = Grid(new, sort=False)
        else:
            raise TypeError(
                f"grid should be a Grid or subclass, or ndarray, got {type(new)}"
            )
        self._R_transform.grid = self.grid
        self._Z_transform.grid = self.grid

    @property
    def R_mn(self):
        """Spectral coefficients for R"""
        return self._R_mn

    @R_mn.setter
    def R_mn(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_mn = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_mn should have the same size as the basis, got {len(new)} for basis with {self.R_basis.num_modes} modes"
            )

    @property
    def Z_mn(self):
        """Spectral coefficients for Z"""
        return self._Z_mn

    @Z_mn.setter
    def Z_mn(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_mn = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_mn should have the same size as the basis, got {len(new)} for basis with {self.Z_basis.num_modes} modes"
            )

    def get_coeffs(self, m, n=0):
        """Get fourier coefficients for given mode number(s)"""
        n = np.atleast_1d(n).astype(int)
        m = np.atleast_1d(m).astype(int)

        m, n = np.broadcast_arrays(m, n)
        R = np.zeros_like(m).astype(float)
        Z = np.zeros_like(m).astype(float)

        mn = np.array([m, n]).T
        idxR = np.where(
            (mn[:, np.newaxis, :] == self.R_basis.modes[np.newaxis, :, 2:]).all(axis=-1)
        )
        idxZ = np.where(
            (mn[:, np.newaxis, :] == self.Z_basis.modes[np.newaxis, :, 2:]).all(axis=-1)
        )

        R[idxR[0]] = self.R_n[idxR[1]]
        Z[idxZ[0]] = self.Z_n[idxZ[1]]
        return R, Z

    def set_coeffs(self, m, n=0, R=None, Z=None):
        """set specific fourier coefficients"""
        m, n, R, Z = (
            np.atleast_1d(m),
            np.atleast_1d(n),
            np.atleast_1d(R),
            np.atleast_1d(Z),
        )
        m, n, R, Z = np.broadcast_arrays([m, n, R, Z])
        for mm, nn, RR, ZZ in zip(m, n, R, Z):
            idxR = self.R_basis.get_idx(0, mm, nn)
            idxZ = self.Z_basis.get_idx(0, mm, nn)
            if RR is not None:
                self.R_n[idxR] = RR
            if ZZ is not None:
                self.Z_n[idxZ] = ZZ

    def _get_transforms(self, grid=None):
        if grid is None:
            return self._R_transform, self._Z_transform
        if not isinstance(grid, Grid):
            if np.isscalar(grid):
                grid = LinearGrid(rho=1, M=grid, N=grid, NFP=self.NFP)
            elif len(grid) == 2:
                grid = LinearGrid(rho=1, M=grid[0], N=grid[1], NFP=self.NFP)
            elif grid.shape[1] == 2:
                grid = np.pad(grid, ((0, 0), (1, 0)))
                grid = Grid(grid, sort=False)
            else:
                grid = Grid(grid, sort=False)
        R_transform = Transform(
            grid,
            self.R_basis,
            derivs=np.array(
                [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 0, 2], [0, 1, 1]]
            ),
        )
        Z_transform = Transform(
            grid,
            self.Z_basis,
            derivs=np.array(
                [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 0, 2], [0, 1, 1]]
            ),
        )
        return R_transform, Z_transform

    def compute_coordinates(self, R_mn=None, Z_mn=None, grid=None, dt=0, dz=0):
        """Compute values using specified coefficients

        Parameters
        ----------
        R_mn, Z_mn: array-like
            fourier coefficients for R, Z. Defaults to self.R_mn, self.Z_mn
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)
        dt, dz: int
            derivative order to compute in theta, zeta

        Returns
        -------
        values : ndarray, shape(k,3)
            R, phi, Z coordinates of the surface at points specified in grid
        """
        if R_mn is None:
            R_mn = self.R_mn
        if Z_mn is None:
            Z_mn = self.Z_mn
        R_transform, Z_transform = self._get_transforms(grid)

        if dz == 0:
            R = R_transform.transform(R_mn, dt=dt, dz=0)
            Z = Z_transform.transform(Z_mn, dt=dt, dz=0)
            phi = R_transform.grid.nodes[:, 2] * (dt == 0)
            return jnp.stack([R, phi, Z], axis=1)
        if dz == 1:
            R0 = R_transform.transform(R_mn, dt=dt, dz=0)
            dR = R_transform.transform(R_mn, dt=dt, dz=1)
            dZ = Z_transform.transform(Z_mn, dt=dt, dz=1)
            dphi = R0 * (dt == 0)
            return jnp.stack([dR, dphi, dZ], axis=1)
        if dz == 2:
            R0 = R_transform.transform(R_mn, dt=dt, dz=0)
            dR = R_transform.transform(R_mn, dt=dt, dz=1)
            d2R = R_transform.transform(R_mn, dt=dt, dz=2)
            d2Z = Z_transform.transform(Z_mn, dt=dt, dz=2)
            R = d2R - R0
            Z = d2Z
            # 2nd derivative wrt to phi = 0
            phi = 2 * dR * (dt == 0)
            return jnp.stack([R, phi, Z], axis=1)
        if dz == 3:
            R0 = R_transform.transform(R_mn, dt=dt, dz=0)
            dR = R_transform.transform(R_mn, dt=dt, dz=1)
            d2R = R_transform.transform(R_mn, dt=dt, dz=2)
            d3R = R_transform.transform(R_mn, dt=dt, dz=3)
            d3Z = Z_transform.transform(Z_mn, dt=dt, dz=3)
            R = d3R - 3 * dR
            Z = d3Z
            phi = (3 * d2R - R0) * (dt == 0)
            return jnp.stack([R, phi, Z], axis=1)
        raise NotImplementedError(
            "Derivatives higher than 3 have not been implemented in cylindrical coordinates"
        )

    def compute_normal(self, R_mn=None, Z_mn=None, grid=None, coords="rpz"):
        """Compute normal vector to surface on default grid

        Parameters
        ----------
        R_mn, Z_mn: array-like
            fourier coefficients for R, Z. Defaults to self.R_mn, self.Z_mn
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)
        coords : {"rpz", "xyz"}
            basis vectors to use for normal vector representation

        Returns
        -------
        N : ndarray, shape(k,3)
            normal vector to surface in specified coordinates
        """
        assert coords.lower() in ["rpz", "xyz"]

        if R_mn is None:
            R_mn = self.R_mn
        if Z_mn is None:
            Z_mn = self.Z_mn
        R_transform, Z_transform = self._get_transforms(grid)

        r_t = self.compute_coordinates(R_mn, Z_mn, grid, dt=1)
        r_z = self.compute_coordinates(R_mn, Z_mn, grid, dz=1)

        N = jnp.cross(r_t, r_z, axis=1)
        N = N / jnp.linalg.norm(N, axis=1)[:, jnp.newaxis]
        if coords.lower() == "xyz":
            phi = R_transform.grid.nodes[:, 2]
            N = pol2cartvec(N, phi=phi)
        return N

    def compute_surface_area(self, R_mn=None, Z_mn=None, grid=None):
        """Compute surface area via quadrature

        Parameters
        ----------
        R_mn, Z_mn: array-like
            fourier coefficients for R, Z. Defaults to self.R_mn, self.Z_mn
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,2pi)

        Returns
        -------
        area : float
            surface area
        """
        if R_mn is None:
            R_mn = self.R_mn
        if Z_mn is None:
            Z_mn = self.Z_mn
        R_transform, Z_transform = self._get_transforms(grid)

        r_t = self.compute_coordinates(R_mn, Z_mn, grid, dt=1)
        r_z = self.compute_coordinates(R_mn, Z_mn, grid, dz=1)

        N = jnp.cross(r_t, r_z, axis=1)
        return jnp.sum(R_transform.grid.weights * jnp.linalg.norm(N, axis=1))


class ZernikeToroidalSection(Surface):
    """A toroidal cross section represented by a zernike polynomial in R,Z

    Parameters
    ----------
    R_lm, Z_lm : array-like, shape(k,)
        zernike coefficients
    modes_R : array-like, shape(k,2)
        radial and poloidal mode numbers [l,m] for R_lm
    modes_Z : array-like, shape(k,2)
        radial and poloidal mode numbers [l,m] for Z_lm. If None defaults to modes_R.
    sym : bool
        whether to enforce stellarator symmetry. Default is "auto" which enforces if
        modes are symmetric. If True, non-symmetric modes will be truncated.
    spectral_indexing : {``'ansi'``, ``'fringe'``}
        Indexing method, default value = ``'fringe'``

        For L=0, all methods are equivalent and give a "chevron" shaped
        basis (only the outer edge of the zernike pyramid of width M).
        For L>0, the indexing scheme defines order of the basis functions:

        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triagle shape. For L == M,
        the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
        to the bottom of the pyramid, increasing L while keeping M constant,
        giving a "house" shape

        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape for L=2*M where
        the traditional fringe/U of Arizona indexing is recovered.
        For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond
    grid : Grid
        default grid for computation
    name : str
        name for this surface

    """

    _io_attrs_ = Surface._io_attrs_ + [
        "_R_lm",
        "_Z_lm",
        "_R_basis",
        "_Z_basis",
        "_R_transform",
        "_Z_transform",
        "_spectral_indexing",
    ]

    def __init__(
        self,
        R_lm=None,
        Z_lm=None,
        modes_R=None,
        modes_Z=None,
        spectral_indexing="fringe",
        sym="auto",
        grid=None,
        name=None,
    ):
        if R_lm is None:
            R_lm = np.array([10, 1])
            modes_R = np.array([[0, 0], [1, 1]])
        if Z_lm is None:
            Z_lm = np.array([0, 1])
            modes_Z = np.array([[0, 0], [1, -1]])
        if modes_Z is None:
            modes_Z = modes_R

        LR = np.max(abs(modes_R[:, 0]))
        MR = np.max(abs(modes_R[:, 1]))
        LZ = np.max(abs(modes_Z[:, 0]))
        MZ = np.max(abs(modes_Z[:, 1]))

        if sym == "auto":
            if np.all(
                R_lm[np.where(sign(modes_R[:, 0]) != sign(modes_R[:, 1]))] == 0
            ) and np.all(
                Z_lm[np.where(sign(modes_Z[:, 0]) == sign(modes_Z[:, 1]))] == 0
            ):
                sym = True
            else:
                sym = False

        self._R_basis = ZernikePolynomial(
            L=max(LR, MR),
            M=max(LR, MR),
            spectral_indexing=spectral_indexing,
            sym="cos" if sym else False,
        )
        self._Z_basis = ZernikePolynomial(
            L=max(LZ, MZ),
            M=max(LZ, MZ),
            spectral_indexing=spectral_indexing,
            sym="sin" if sym else False,
        )

        self._R_lm = copy_coeffs(R_lm, modes_R, self.R_basis.modes[:, :2])
        self._Z_lm = copy_coeffs(Z_lm, modes_Z, self.Z_basis.modes[:, :2])
        self._sym = sym
        self._spectral_indexing = spectral_indexing

        if grid is None:
            grid = Grid(np.empty((0, 3)))
        self._grid = grid
        self._R_transform, self._Z_transform = self._get_transforms(grid)
        self.name = name

    @property
    def spectral_indexing(self):
        return self._spectral_indexing

    @property
    def R_basis(self):
        """Spectral basis for R zernike polynomial"""
        return self._R_basis

    @property
    def Z_basis(self):
        """Spectral basis for Z zernike polynomial"""
        return self._Z_basis

    @property
    def grid(self):
        """Default grid for computation"""
        return self._grid

    @grid.setter
    def grid(self, new):
        if isinstance(new, Grid):
            self._grid = new
        elif isinstance(new, (np.ndarray, jnp.ndarray)):
            self._grid = Grid(new, sort=False)
        else:
            raise TypeError(
                f"grid should be a Grid or subclass, or ndarray, got {type(new)}"
            )
        self._R_transform.grid = self.grid
        self._Z_transform.grid = self.grid

    @property
    def R_lm(self):
        """Spectral coefficients for R"""
        return self._R_lm

    @R_lm.setter
    def R_lm(self, new):
        if len(new) == self.R_basis.num_modes:
            self._R_lm = jnp.asarray(new)
        else:
            raise ValueError(
                f"R_lm should have the same size as the basis, got {len(new)} for basis with {self.R_basis.num_modes} modes"
            )

    @property
    def Z_lm(self):
        """Spectral coefficients for Z"""
        return self._Z_lm

    @Z_lm.setter
    def Z_lm(self, new):
        if len(new) == self.Z_basis.num_modes:
            self._Z_lm = jnp.asarray(new)
        else:
            raise ValueError(
                f"Z_lm should have the same size as the basis, got {len(new)} for basis with {self.Z_basis.num_modes} modes"
            )

    def get_coeffs(self, l, m=0):
        """Get fourier coefficients for given mode number(s)"""
        l = np.atleast_1d(l).astype(int)
        m = np.atleast_1d(m).astype(int)

        l, m = np.broadcast_arrays(l, m)
        R = np.zeros_like(m).astype(float)
        Z = np.zeros_like(m).astype(float)

        lm = np.array([l, m]).T
        idxR = np.where(
            (lm[:, np.newaxis, :] == self.R_basis.modes[np.newaxis, :, :2]).all(axis=-1)
        )
        idxZ = np.where(
            (lm[:, np.newaxis, :] == self.Z_basis.modes[np.newaxis, :, :2]).all(axis=-1)
        )

        R[idxR[0]] = self.R_n[idxR[1]]
        Z[idxZ[0]] = self.Z_n[idxZ[1]]
        return R, Z

    def set_coeffs(self, l, m=0, R=None, Z=None):
        """set specific fourier coefficients"""
        l, m, R, Z = (
            np.atleast_1d(l),
            np.atleast_1d(m),
            np.atleast_1d(R),
            np.atleast_1d(Z),
        )
        l, m, R, Z = np.broadcast_arrays([l, m, R, Z])
        for ll, mm, RR, ZZ in zip(l, m, R, Z):
            idxR = self.R_basis.get_idx(ll, mm, 0)
            idxZ = self.Z_basis.get_idx(ll, mm, 0)
            if RR is not None:
                self.R_n[idxR] = RR
            if ZZ is not None:
                self.Z_n[idxZ] = ZZ

    def _get_transforms(self, grid=None):
        if grid is None:
            return self._R_transform, self._Z_transform
        if not isinstance(grid, Grid):
            if np.isscalar(grid):
                grid = LinearGrid(L=grid, M=grid, zeta=0, NFP=self.NFP)
            elif len(grid) == 2:
                grid = LinearGrid(L=grid[0], M=grid[1], zeta=0, NFP=self.NFP)
            elif grid.shape[1] == 2:
                grid = np.pad(grid, ((0, 0), (0, 1)))
                grid = Grid(grid, sort=False)
            else:
                grid = Grid(grid, sort=False)
        R_transform = Transform(
            grid,
            self.R_basis,
            derivs=np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [1, 1, 0]]
            ),
        )
        Z_transform = Transform(
            grid,
            self.Z_basis,
            derivs=np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [1, 1, 0]]
            ),
        )
        return R_transform, Z_transform

    def compute_coordinates(self, R_lm=None, Z_lm=None, grid=None, dr=0, dt=0):
        """Compute values using specified coefficients

        Parameters
        ----------
        R_lm, Z_lm: array-like
            zernike coefficients for R, Z. Defaults to self.R_lm, self.Z_lm
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,1)x(0,2pi)
        dr, dt: int
            derivative order to compute in rho, theta

        Returns
        -------
        values : ndarray, shape(k,3)
            R, phi, Z coordinates of the surface at points specified in grid
        """
        if R_lm is None:
            R_lm = self.R_lm
        if Z_lm is None:
            Z_lm = self.Z_lm
        R_transform, Z_transform = self._get_transforms(grid)

        R = R_transform.transform(R_lm, dr=dr, dt=dt)
        Z = Z_transform.transform(Z_lm, dr=dr, dt=dt)
        phi = R_transform.grid.nodes[:, 2] * (dr == 0) * (dt == 0)

        return jnp.stack([R, phi, Z], axis=1)

    def compute_normal(self, R_lm=None, Z_lm=None, grid=None, coords="rpz"):
        """Compute normal vector to surface on default grid

        Parameters
        ----------
        R_lm, Z_lm: array-like
            zernike coefficients for R, Z. Defaults to self.R_lm, self.Z_lm
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,1)x(0,2pi)
        coords : {"rpz", "xyz"}
            basis vectors to use for normal vector representation

        Returns
        -------
        N : ndarray, shape(k,3)
            normal vector to surface in specified coordinates
        """
        assert coords.lower() in ["rpz", "xyz"]

        R_transform, Z_transform = self._get_transforms(grid)

        phi = R_transform.grid.nodes[:, -1]

        # normal vector is a constant 1*phihat
        N = jnp.array([jnp.zeros_like(phi), jnp.ones_like(phi), jnp.zeros_like(phi)]).T

        if coords.lower() == "xyz":
            N = pol2cartvec(N, phi=phi)

        return N

    def compute_surface_area(self, R_lm=None, Z_lm=None, grid=None):
        """Compute surface area via quadrature

        Parameters
        ----------
        R_lm, Z_lm: array-like
            zernike coefficients for R, Z. Defaults to self.R_lm, self.Z_lm
        grid : Grid or array-like
            toroidal coordinates to compute at. Defaults to self.grid
            If an integer, assumes that many linearly spaced points in (0,1)x(0,2pi)

        Returns
        -------
        area : float
            surface area
        """
        if R_lm is None:
            R_lm = self.R_lm
        if Z_lm is None:
            Z_lm = self.Z_lm
        R_transform, Z_transform = self._get_transforms(grid)

        r_r = self.compute_coordinates(R_lm, Z_lm, grid, dr=1)
        r_t = self.compute_coordinates(R_lm, Z_lm, grid, dt=1)

        N = jnp.cross(r_r, r_t, axis=1)
        return jnp.sum(R_transform.grid.weights * jnp.linalg.norm(N, axis=1)) / (
            2 * np.pi
        )
