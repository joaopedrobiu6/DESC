"""Compute functions for bootstrap current."""

from scipy.constants import elementary_charge
from scipy.special import roots_legendre

from ..backend import fori_loop, jnp
from ..profiles import PowerSeriesProfile, Profile
from .data_index import register_compute_fun
from .utils import (
    compress,
    expand,
    surface_averages,
    surface_integrals,
)


@register_compute_fun(
    name="trapped fraction",
    label="1 - \\frac{3}{4} \\left< B^2 \\right> \\int_0^{1/Bmax} "
          "\\frac{\\lambda\\; d\\lambda}{\\left< \\sqrt{1 - \\lambda B} \\right>}",
    units="~",
    units_long="None",
    description="Neoclassical effective trapped particle fraction",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="r",
    data=["sqrt(g)", "V_r(r)", "|B|", "<B^2>", "max_tz |B|"],
)
def _trapped_fraction(params, transforms, profiles, data, **kwargs):
    r"""
    Evaluate the effective trapped particle fraction.

    Compute the effective fraction of trapped particles, which enters
    several formulae for neoclassical transport. The trapped fraction
    ``f_t`` has a standard definition in neoclassical theory:

    .. math::
        f_t = 1 - \frac{3}{4} \left< B^2 \right> \int_0^{1/Bmax}
            \frac{\lambda\; d\lambda}{\left< \sqrt{1 - \lambda B} \right>}

    where :math:`\left< \ldots \right>` is a flux surface average.
    """
    # Get nodes and weights for Gauss-Legendre integration:
    n_gauss = kwargs.get("n_gauss", 20)
    base_nodes, base_weights = roots_legendre(n_gauss)
    # Rescale for integration on [0, 1], not [-1, 1]:
    lambd = (base_nodes + 1) * 0.5
    lambda_weights = base_weights * 0.5

    grid = transforms["grid"]
    Bmax = data["max_tz |B|"]
    modB_over_Bmax = data["|B|"] / Bmax
    sqrt_g = jnp.abs(data["sqrt(g)"])
    denominator = data["V_r(r)"]
    Bmax_squared = compress(grid, Bmax * Bmax)

    # Sum over the lambda grid points, using fori_loop for efficiency.
    lambd = jnp.asarray(lambd)
    lambda_weights = jnp.asarray(lambda_weights)

    def body_fun(jlambda, lambda_integral):
        flux_surf_avg_term = surface_averages(
            grid,
            jnp.sqrt(1 - lambd[jlambda] * modB_over_Bmax),
            sqrt_g,
            denominator=denominator,
        )
        return lambda_integral + lambda_weights[jlambda] * lambd[jlambda] / (
            Bmax_squared * compress(grid, flux_surf_avg_term)
        )

    lambda_integral = fori_loop(0, n_gauss, body_fun, jnp.zeros(grid.num_rho))

    trapped_fraction = 1 - 0.75 * compress(grid, data["<B^2>"]) * lambda_integral
    data["trapped fraction"] = expand(grid, trapped_fraction)
    return data


def j_dot_B_Redl(
    geom_data,
    ne,
    Te,
    Ti,
    Zeff=None,
    helicity_N=None,
    plot=False,
):
    r"""Compute the bootstrap current.

    (specifically :math:`\left<\vec{J}\cdot\vec{B}\right>`) using the formulae in
    Redl et al, Physics of Plasmas 28, 022502 (2021). This formula for
    the bootstrap current is valid in axisymmetry, quasi-axisymmetry,
    and quasi-helical symmetry, but not in other stellarators.

    The profiles of ne, Te, Ti, and Zeff should all be instances of
    subclasses of :obj:`desc.Profile`, i.e. they should
    have ``__call__()`` and ``dfds()`` functions. If ``Zeff == None``, a
    constant 1 is assumed. If ``Zeff`` is a float, a constant profile will
    be assumed.

    ``ne`` should have units of 1/m^3. ``Ti`` and ``Te`` should have
    units of eV.

    The argument ``geom_data`` is a Dictionary that should contain the
    following items:

    - rho: 1D array with the effective minor radius.
    - G: 1D array with the Boozer ``G`` coefficient.
    - R: 1D array with the effective value of ``R`` to use in the Redl formula,
      not necessarily the major radius.
    - iota: 1D array with the rotational transform.
    - epsilon: 1D array with the effective inverse aspect ratio to use in
      the Redl formula.
    - psi_edge: float, the boundary toroidal flux, divided by (2 pi).
    - f_t: 1D array with the effective trapped particle fraction

    Parameters
    ----------
    geom_data : dict
        Dictionary containing the data described above.
    ne : A :obj:`~desc.profile.Profile` object
        The electron density profile.
    Te : A :obj:`~desc.profile.Profile` object
        The electron temperature profile.
    Ti : A :obj:`~desc.profile.Profile` object
        The ion temperature profile.
    Zeff : `None`, float, or a :obj:`~Profile` object
        The profile of the average impurity charge :math:`Z_{eff}`. A single
        number can be provided if this profile is constant. Or, if ``None``,
        Zeff = 1 will be used.
    helicity_N : int
        Set to 0 for quasi-axisymmetry, or +/- NFP for quasi-helical symmetry.
        This quantity is used to apply the quasisymmetry isomorphism to map the
        collisionality and bootstrap current from the tokamak expressions to
        quasi-helical symmetry.
    plot : boolean
        Whether to make a plot of many of the quantities computed.

    Returns
    -------
    J_dot_B_data : dict
        Dictionary containing the computed data listed above.
    """
    rho = geom_data["rho"]
    G = geom_data["G"]
    R = geom_data["R"]
    iota = geom_data["iota"]
    epsilon = geom_data["epsilon"]
    psi_edge = geom_data["psi_edge"]
    f_t = geom_data["f_t"]

    if Zeff is None:
        Zeff = PowerSeriesProfile(1.0, modes=[0])
    if not isinstance(Zeff, Profile):
        # Zeff is presumably a number. Convert it to a constant profile.
        Zeff = PowerSeriesProfile([Zeff], modes=[0])

    # Evaluate profiles on the grid:
    ne_rho = ne(rho)
    Te_rho = Te(rho)
    Ti_rho = Ti(rho)
    Zeff_rho = Zeff(rho)
    ni_rho = ne_rho / Zeff_rho
    pe_rho = ne_rho * Te_rho
    pi_rho = ni_rho * Ti_rho
    d_ne_d_s = ne(rho, dr=1) / (2 * rho)
    d_Te_d_s = Te(rho, dr=1) / (2 * rho)
    d_Ti_d_s = Ti(rho, dr=1) / (2 * rho)

    # Eq (18d)-(18e) in Sauter, Angioni, and Lin-Liu, Physics of Plasmas 6, 2834 (1999).
    ln_Lambda_e = 31.3 - jnp.log(jnp.sqrt(ne_rho) / Te_rho)
    ln_Lambda_ii = 30 - jnp.log(Zeff_rho**3 * jnp.sqrt(ni_rho) / (Ti_rho**1.5))

    # Eq (18b)-(18c) in Sauter:
    geometry_factor = abs(R / (iota - helicity_N))
    nu_e = (
        geometry_factor
        * (6.921e-18)
        * ne_rho
        * Zeff_rho
        * ln_Lambda_e
        / (Te_rho * Te_rho * (epsilon**1.5))
    )
    nu_i = (
        geometry_factor
        * (4.90e-18)
        * ni_rho
        * (Zeff_rho**4)
        * ln_Lambda_ii
        / (Ti_rho * Ti_rho * (epsilon**1.5))
    )

    # Redl eq (11):
    X31 = f_t / (
        1
        + (0.67 * (1 - 0.7 * f_t) * jnp.sqrt(nu_e)) / (0.56 + 0.44 * Zeff_rho)
        + (0.52 + 0.086 * jnp.sqrt(nu_e))
        * (1 + 0.87 * f_t)
        * nu_e
        / (1 + 1.13 * jnp.sqrt(Zeff_rho - 1))
    )

    # Redl eq (10):
    Zfac = Zeff_rho**1.2 - 0.71
    L31 = (
        (1 + 0.15 / Zfac) * X31
        - 0.22 / Zfac * (X31**2)
        + 0.01 / Zfac * (X31**3)
        + 0.06 / Zfac * (X31**4)
    )

    # Redl eq (14):
    X32e = f_t / (
        (
            1
            + 0.23 * (1 - 0.96 * f_t) * jnp.sqrt(nu_e) / jnp.sqrt(Zeff_rho)
            + 0.13
            * (1 - 0.38 * f_t)
            * nu_e
            / (Zeff_rho * Zeff_rho)
            * (
                jnp.sqrt(1 + 2 * jnp.sqrt(Zeff_rho - 1))
                + f_t * f_t * jnp.sqrt((0.075 + 0.25 * (Zeff_rho - 1) ** 2) * nu_e)
            )
        )
    )

    # Redl eq (13):
    F32ee = (
        (0.1 + 0.6 * Zeff_rho)
        * (X32e - X32e**4)
        / (Zeff_rho * (0.77 + 0.63 * (1 + (Zeff_rho - 1) ** 1.1)))
        + 0.7
        / (1 + 0.2 * Zeff_rho)
        * (X32e**2 - X32e**4 - 1.2 * (X32e**3 - X32e**4))
        + 1.3 / (1 + 0.5 * Zeff_rho) * (X32e**4)
    )

    # Redl eq (16):
    X32ei = f_t / (
        1
        + 0.87 * (1 + 0.39 * f_t) * jnp.sqrt(nu_e) / (1 + 2.95 * (Zeff_rho - 1) ** 2)
        + 1.53 * (1 - 0.37 * f_t) * nu_e * (2 + 0.375 * (Zeff_rho - 1))
    )

    # Redl eq (15):
    F32ei = (
        -(0.4 + 1.93 * Zeff_rho)
        / (Zeff_rho * (0.8 + 0.6 * Zeff_rho))
        * (X32ei - X32ei**4)
        + 5.5
        / (1.5 + 2 * Zeff_rho)
        * (X32ei**2 - X32ei**4 - 0.8 * (X32ei**3 - X32ei**4))
        - 1.3 / (1 + 0.5 * Zeff_rho) * (X32ei**4)
    )

    # Redl eq (12):
    L32 = F32ei + F32ee

    # Redl eq (19):
    L34 = L31

    # Redl eq (20):
    alpha0 = (
        -(0.62 + 0.055 * (Zeff_rho - 1))
        * (1 - f_t)
        / (
            (0.53 + 0.17 * (Zeff_rho - 1))
            * (1 - (0.31 - 0.065 * (Zeff_rho - 1)) * f_t - 0.25 * f_t * f_t)
        )
    )
    # Redl eq (21):
    alpha = (
        (alpha0 + 0.7 * Zeff_rho * jnp.sqrt(f_t * nu_i)) / (1 + 0.18 * jnp.sqrt(nu_i))
        - 0.002 * nu_i * nu_i * (f_t**6)
    ) / (1 + 0.004 * nu_i * nu_i * (f_t**6))

    # Factor of elementary_charge is included below to convert temperatures from eV to J
    dnds_term = (
        -G
        * elementary_charge
        * (ne_rho * Te_rho + ni_rho * Ti_rho)
        * L31
        * (d_ne_d_s / ne_rho)
        / (psi_edge * (iota - helicity_N))
    )
    dTeds_term = (
        -G
        * elementary_charge
        * pe_rho
        * (L31 + L32)
        * (d_Te_d_s / Te_rho)
        / (psi_edge * (iota - helicity_N))
    )
    dTids_term = (
        -G
        * elementary_charge
        * pi_rho
        * (L31 + L34 * alpha)
        * (d_Ti_d_s / Ti_rho)
        / (psi_edge * (iota - helicity_N))
    )
    J_dot_B = dnds_term + dTeds_term + dTids_term

    # Store all results in the J_dot_B_data dictionary:
    nu_e_star = nu_e
    nu_i_star = nu_i
    variables = [
        "rho",
        "ne_rho",
        "ni_rho",
        "Zeff_rho",
        "Te_rho",
        "Ti_rho",
        "d_ne_d_s",
        "d_Te_d_s",
        "d_Ti_d_s",
        "ln_Lambda_e",
        "ln_Lambda_ii",
        "nu_e_star",
        "nu_i_star",
        "X31",
        "X32e",
        "X32ei",
        "F32ee",
        "F32ei",
        "L31",
        "L32",
        "L34",
        "alpha0",
        "alpha",
        "dnds_term",
        "dTeds_term",
        "dTids_term",
    ]
    J_dot_B_data = geom_data.copy()
    for v in variables:
        J_dot_B_data[v] = eval(v)
    J_dot_B_data["<J*B>"] = J_dot_B

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 7))
        plt.rcParams.update({"font.size": 8})
        nrows = 5
        ncols = 5
        variables = [
            "Bmax",
            "Bmin",
            "epsilon",
            "<B^2>",
            "<1/B>",
            "f_t",
            "iota",
            "G",
            "R",
            "ne_rho",
            "ni_rho",
            "Zeff_rho",
            "Te_rho",
            "Ti_rho",
            "ln_Lambda_e",
            "ln_Lambda_ii",
            "nu_e_star",
            "nu_i_star",
            "dnds_term",
            "dTeds_term",
            "dTids_term",
            "L31",
            "L32",
            "alpha",
            "<J*B>",
        ]
        for j, variable in enumerate(variables):
            plt.subplot(nrows, ncols, j + 1)
            plt.plot(rho, J_dot_B_data[variable])
            plt.title(variable)
            plt.xlabel(r"$\rho$")
        plt.tight_layout()
        plt.show()

    return J_dot_B_data


@register_compute_fun(
    name="<J*B> Redl",
    label="\\langle\\mathbf{J}\\cdot\\mathbf{B}\\rangle_{Redl}",
    units="T A m^{-2}",
    units_long="Tesla Ampere / meter^2",
    description="Bootstrap current profile, Redl model for quasisymmetry",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[
        "electron_density",
        "electron_temperature",
        "ion_temperature",
        "atomic_number",
    ],
    coordinates="r",
    data=["trapped fraction", "G", "I", "iota", "<1/|B|>", "effective r/R0"],
)
def _compute_J_dot_B_Redl(params, transforms, profiles, data, **kwargs):
    r"""Compute the bootstrap current.

    (specifically :math:`\left<\vec{J}\cdot\vec{B}\right>`) using the formulae in
    Redl et al, Physics of Plasmas 28, 022502 (2021). This formula for
    the bootstrap current is valid in axisymmetry, quasi-axisymmetry,
    and quasi-helical symmetry, but not in other stellarators.
    """
    grid = transforms["grid"]

    # Note that the geom_data dictionary provided to j_dot_B_Redl()
    # contains info only as a function of rho, not theta or zeta,
    # i.e. on the compressed grid. In contrast, "data" contains
    # quantities on a 3D grid even for quantities that are flux
    # functions.
    geom_data = {}
    geom_data["rho"] = compress(grid, data["rho"])
    geom_data["f_t"] = compress(grid, data["trapped fraction"])
    geom_data["epsilon"] = compress(grid, data["effective r/R0"])
    geom_data["G"] = compress(grid, data["G"])
    geom_data["I"] = compress(grid, data["I"])
    geom_data["iota"] = compress(grid, data["iota"])
    geom_data["<1/|B|>"] = compress(grid, data["<1/|B|>"])
    geom_data["R"] = (geom_data["G"] + geom_data["iota"] * geom_data["I"]) * geom_data[
        "<1/|B|>"
    ]
    geom_data["psi_edge"] = params["Psi"] / (2 * jnp.pi)

    ne = profiles["electron_density"]
    Te = profiles["electron_temperature"]
    Ti = profiles["ion_temperature"]
    Zeff = profiles["atomic_number"]
    # The "backup" PowerSeriesProfiles that follow here are necessary
    # for test_compute_funs.py::test_compute_everything
    if ne is None:
        ne = PowerSeriesProfile([1e20])
    if Te is None:
        Te = PowerSeriesProfile([1e3])
    if Ti is None:
        Ti = PowerSeriesProfile([1e3])
    if Zeff is None:
        Zeff = PowerSeriesProfile([1.0])

    helicity = kwargs.get("helicity", (1, 0))
    helicity_N = helicity[1]

    j_dot_B_data = j_dot_B_Redl(
        geom_data,
        ne,
        Te,
        Ti,
        Zeff,
        helicity_N,
        plot=False,
    )
    data["<J*B> Redl"] = expand(grid, j_dot_B_data["<J*B>"])
    return data
