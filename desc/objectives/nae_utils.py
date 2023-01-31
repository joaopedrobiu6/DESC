"""Functions to create the O(rho) and O(rho^2) NAE constraints on a DESC equilibrium."""

import numpy as np

from desc.basis import FourierSeries
from desc.grid import LinearGrid
from desc.transform import Transform

from .linear_objectives import FixSumModesR, FixSumModesZ


def _calc_1st_order_NAE_coeffs(qsc, desc_eq):
    """Calculate 1st order NAE coefficients' toroidal Fourier representations.

    Description
    -----------
        uses the passed-in qsc object, and the desc_eq's stellarator symmetry is used.

    Parameters
    ----------
        qsc (Qsc): Qsc object to use as the NAE constraints on the DESC equilibrium
        desc_eq (Equilibrium): desc equilibrium to constrain

    Returns
    -------
        coeffs: dict, dictionary of arrays with keys like 'X_L_M_n', where
                X is R or Z, L is 1 or 2, and M is 0,1, or 2, are the
                NAE Fourier (in toroidal angle phi) coeffs of
                radial order L and poloidal order M
        bases: dict, dictionary of Rbasis_cos, Rbasis_sin, Zbasis_cos, Zbasis_sin,
            the FourierSeries basis objects used to obtain the coefficients, where
            _cos or _sin denotes the symmetry of the (toroidal) Fourier series.
            symmetry is such that the R or Z coefficients is stellarator symmetric
            i.e. R_1_1_n uses the Rbasis_cos, since cos(theta)*cos(phi) is
             stellarator symmetric for R i.e. R(-theta,-phi) = R(theta,phi)
            and Z_1_1_n uses the Zbasis_sin as the term is cos(theta)*sin(phi)
            since Z(-theta,-phi) = - Z(theta,phi) for Z stellarator symmetry
    """
    phi = qsc.phi

    R0 = qsc.R0_func(phi)
    dR0_dphi = qsc.R0p
    dZ0_dphi = qsc.Z0p
    # normal and binormal vector components
    # Spline interpolants for the cylindrical components of the Frenet-Serret frame:
    # these are functions of phi (toroidal cylindrical angle)
    k_dot_R = qsc.normal_R_spline(phi)
    k_dot_phi = qsc.normal_phi_spline(phi)
    k_dot_Z = qsc.normal_z_spline(phi)
    tau_dot_R = qsc.binormal_R_spline(phi)
    tau_dot_phi = qsc.binormal_phi_spline(phi)
    tau_dot_Z = qsc.binormal_z_spline(phi)

    # use untwisted, which accounts for when NAE has QH symmetry,
    # and the poloidal angle is a helical angle.
    # we want the untwisted angle
    X1c = qsc.X1c_untwisted
    X1s = qsc.X1s_untwisted
    Y1c = qsc.Y1c_untwisted
    Y1s = qsc.Y1s_untwisted

    R_1_1 = X1c * (k_dot_R - k_dot_phi * dR0_dphi / R0) + Y1c * (
        tau_dot_R - tau_dot_phi * dR0_dphi / R0
    )
    R_1_neg1 = Y1s * (tau_dot_R - tau_dot_phi * dR0_dphi / R0) + X1s * (
        k_dot_R - k_dot_phi * dR0_dphi / R0
    )

    Z_1_1 = X1c * (k_dot_Z - k_dot_phi * dZ0_dphi / R0) + Y1c * (
        tau_dot_Z - tau_dot_phi * dZ0_dphi / R0
    )
    Z_1_neg1 = Y1s * (tau_dot_Z - tau_dot_phi * dZ0_dphi / R0) + X1s * (
        k_dot_Z - k_dot_phi * dZ0_dphi / R0
    )

    nfp = qsc.nfp
    if desc_eq.sym:
        Rbasis = FourierSeries(N=desc_eq.N, NFP=nfp, sym="cos")
        Zbasis = FourierSeries(N=desc_eq.N, NFP=nfp, sym="cos")
        Rbasis_sin = FourierSeries(N=desc_eq.N, NFP=nfp, sym="sin")
        Zbasis_sin = FourierSeries(N=desc_eq.N, NFP=nfp, sym="sin")
    else:
        Rbasis = FourierSeries(N=desc_eq.N, NFP=nfp, sym=False)
        Zbasis = FourierSeries(N=desc_eq.N, NFP=nfp, sym=False)
        Rbasis_sin = FourierSeries(N=desc_eq.N, NFP=nfp, sym=False)
        Zbasis_sin = FourierSeries(N=desc_eq.N, NFP=nfp, sym=False)

    grid = LinearGrid(M=0, L=0, zeta=phi, NFP=nfp)
    Rtrans = Transform(grid, Rbasis, build_pinv=True, method="auto")
    Ztrans = Transform(grid, Zbasis, build_pinv=True, method="auto")
    Rtrans_sin = Transform(grid, Rbasis_sin, build_pinv=True, method="auto")
    Ztrans_sin = Transform(grid, Zbasis_sin, build_pinv=True, method="auto")

    R_1_1_n = Rtrans.fit(R_1_1)
    R_1_neg1_n = Rtrans_sin.fit(R_1_neg1)

    Z_1_1_n = Ztrans_sin.fit(Z_1_1)
    Z_1_neg1_n = Ztrans.fit(Z_1_neg1)

    bases = {}
    bases["Rbasis_cos"] = Rbasis
    bases["Rbasis_sin"] = Rbasis_sin
    bases["Zbasis_cos"] = Zbasis
    bases["Zbasis_sin"] = Zbasis_sin

    coeffs = {}
    coeffs["R_1_1_n"] = R_1_1_n
    coeffs["R_1_neg1_n"] = R_1_neg1_n
    coeffs["Z_1_1_n"] = Z_1_1_n
    coeffs["Z_1_neg1_n"] = Z_1_neg1_n

    return coeffs, bases


def _make_RZ_cons_R1_Zn1(qsc, desc_eq, coeffs, bases):
    # r is the ratio  r_NAE / rho_DESC
    r = np.sqrt(2 * desc_eq.Psi / qsc.Bbar / 2 / np.pi)

    Rconstraints = ()
    Zconstraints = ()
    Rbasis_cos = bases["Rbasis_cos"]
    Zbasis_cos = bases["Zbasis_cos"]

    # R_11n
    for n, NAEcoeff in zip(Rbasis_cos.modes[:, 2], coeffs["R_1_1_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):  # FIXME: add by 1 desc_eq.L+1
            modes.append([2 * k - 1, 1, n])
            sum_weights.append([(-1) ** k * k])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Rcon = FixSumModesR(target=target, sum_weights=sum_weights, modes=modes)
        desc_eq.R_lmn
        Rconstraints += (Rcon,)
    # Z_1-1n
    for n, NAEcoeff in zip(Zbasis_cos.modes[:, 2], coeffs["Z_1_neg1_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):
            modes.append([2 * k - 1, -1, n])
            sum_weights.append([(-1) ** k * k])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Zcon = FixSumModesZ(target=target, sum_weights=sum_weights, modes=modes)
        Zconstraints += (Zcon,)
    return Rconstraints, Zconstraints


def _make_RZ_cons_Rn1_Z1(qsc, desc_eq, coeffs, bases):
    # r is the ratio  r_NAE / rho_DESC
    r = np.sqrt(2 * desc_eq.Psi / qsc.Bbar / 2 / np.pi)
    Rconstraints = ()
    Zconstraints = ()

    Rbasis_sin = bases["Rbasis_sin"]
    Zbasis_sin = bases["Zbasis_sin"]

    # R_1-1n
    for n, NAEcoeff in zip(Rbasis_sin.modes[:, 2], coeffs["R_1_neg1_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):
            modes.append([2 * k - 1, -1, n])
            sum_weights.append([(-1) ** k * k])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Rcon = FixSumModesR(target=target, sum_weights=sum_weights, modes=modes)
        Rconstraints += (Rcon,)
    # Z_11n
    for n, NAEcoeff in zip(Zbasis_sin.modes[:, 2], coeffs["Z_1_1_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int((desc_eq.L + 1) / 2) + 1):
            modes.append([2 * k - 1, 1, n])
            sum_weights.append([(-1) ** k * k])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Zcon = FixSumModesZ(target=target, sum_weights=sum_weights, modes=modes)
        Zconstraints += (Zcon,)

    return Rconstraints, Zconstraints


def make_RZ_cons_1st_order(qsc, desc_eq):
    """Make the first order NAE constraints for a DESC equilibrium.

    Parameters
    ----------
        qsc (Qsc): Qsc object to use as the NAE constraints on the DESC equilibrium
        desc_eq (Equilibrium): desc equilibrium to constrain

    Returns
    -------
        Rconstraints tuple: tuple of FixSumModesR constraints
           corresponding to constraining the O(rho) DESC coefficients,
           to be used in constraining a DESC equilibrium solve
        Zconstraints tuple: tuple of FixSumModesZ constraints
            corresponding to constraining the O(rho) DESC coefficients,
            to be used in constraining a DESC equilibrium solve
    """
    Rconstraints = ()
    Zconstraints = ()

    coeffs, bases = _calc_1st_order_NAE_coeffs(qsc, desc_eq)

    Rcons1, Zcons1 = _make_RZ_cons_R1_Zn1(qsc, desc_eq, coeffs, bases)
    Rcons2, Zcons2 = _make_RZ_cons_Rn1_Z1(qsc, desc_eq, coeffs, bases)

    Rconstraints += Rcons1 + Rcons2
    Zconstraints += Zcons1 + Zcons2

    return Rconstraints + Zconstraints


""" Order (rho^2)"""


def _calc_2nd_order_NAE_coeffs(qsc, desc_eq):
    """Calculate 2nd order NAE coefficients' Fourier representations.

    Description
    -----------
        uses the passed-in qsc object, and the desc_eq's stellarator symmetry is used.

    Parameters
    ----------
        qsc (Qsc): Qsc object to use as the NAE constraints on the DESC equilibrium
        desc_eq (Equilibrium): desc equilibrium to constrain

    Returns
    -------
        coeffs: dict, dictionary of arrays with keys like 'X_L_M_n', where
                X is R or Z, L is 1 or 2, and M is 0,1, or 2, are the
                NAE Fourier (in tor. phi) coeffs of radial order L and poloidal order M
        bases: dict, dictionary of Rbasis_cos, Rbasis_sin, Zbasis_cos, Zbasis_sin,
            the FourierSeries basis objects used to obtain the coefficients, where
            _cos or _sin denotes the symmetry of the Fourier series.
            symmetry is such that the R or Z coefficients is stellarator symmetric
            i.e. R_1_1_n uses the Rbasis_cos, since cos(theta)*cos(phi) is
             stellarator symmetric for R i.e. R(-theta,-phi) = R(theta,phi)
            and Z_1_1_n uses the Zbasis_sin as the term is cos(theta)*sin(phi)
            since Z(-theta,-phi) = - Z(theta,phi) for Z stellarator symmetry
    """
    # get variables from qsc
    phi = qsc.phi
    R0 = qsc.R0_func(phi)
    dR0_dphi = qsc.R0p
    R0p = dR0_dphi
    R0pp = qsc.R0pp
    dZ0_dphi = qsc.Z0p
    Z0p = dZ0_dphi
    Z0pp = qsc.Z0pp

    # unit vector components
    k_dot_R = qsc.normal_R_spline(phi)
    kR = k_dot_R
    k_dot_phi = qsc.normal_phi_spline(phi)
    kphi = k_dot_phi
    k_dot_Z = qsc.normal_z_spline(phi)
    kZ = k_dot_Z
    tau_dot_R = qsc.binormal_R_spline(phi)
    tauR = tau_dot_R
    tau_dot_phi = qsc.binormal_phi_spline(phi)
    tauphi = tau_dot_phi
    tau_dot_Z = qsc.binormal_z_spline(phi)
    tauZ = tau_dot_Z
    bR = qsc.tangent_R_spline(phi)
    bphi = qsc.tangent_phi_spline(phi)
    bZ = qsc.tangent_z_spline(phi)

    # unit vector component derivatives wrt phi
    ## second arg of the scipy CubicSpline class used is the derivative order
    kRp = qsc.normal_R_spline(phi, 1)
    kphip = qsc.normal_phi_spline(phi, 1)
    kZp = qsc.normal_z_spline(phi, 1)

    tauRp = qsc.binormal_R_spline(phi, 1)
    tauphip = qsc.binormal_phi_spline(phi, 1)
    tauZp = qsc.binormal_z_spline(phi, 1)

    # never hurts to use the untwisted ones, as for QH this is needed
    # and for QA they are the same as the X1c etc variables
    X1c = qsc.X1c_untwisted
    X1s = qsc.X1s_untwisted
    Y1c = qsc.Y1c_untwisted
    Y1s = qsc.Y1s_untwisted

    X2c = qsc.X2c_untwisted
    X2s = qsc.X2s_untwisted
    Y2c = qsc.Y2c_untwisted
    Y2s = qsc.Y2s_untwisted
    X20 = qsc.X20_untwisted
    Y20 = qsc.Y20_untwisted

    Z20NAE = qsc.Z20_untwisted
    Z2sNAE = qsc.Z2s_untwisted
    Z2cNAE = qsc.Z2c_untwisted

    dvphi_dp = qsc.d_varphi_d_phi  # needed to convert dX_d_dvarphi derivs to dX_d_dphi
    # coefficient derivatives
    X1cp = qsc.d_X1c_d_varphi * dvphi_dp
    X1sp = qsc.d_X1s_d_varphi * dvphi_dp
    Y1cp = qsc.d_Y1c_d_varphi * dvphi_dp
    Y1sp = qsc.d_Y1s_d_varphi * dvphi_dp

    ## first order terms ##
    R_1_1 = X1c * (k_dot_R - k_dot_phi * dR0_dphi / R0) + Y1c * (
        tau_dot_R - tau_dot_phi * dR0_dphi / R0
    )
    R_1_neg1 = Y1s * (tau_dot_R - tau_dot_phi * dR0_dphi / R0) + X1s * (
        k_dot_R - k_dot_phi * dR0_dphi / R0
    )

    Z_1_1 = X1c * (k_dot_Z - k_dot_phi * dZ0_dphi / R0) + Y1c * (
        tau_dot_Z - tau_dot_phi * dZ0_dphi / R0
    )
    Z_1_neg1 = Y1s * (tau_dot_Z - tau_dot_phi * dZ0_dphi / R0) + X1s * (
        k_dot_Z - k_dot_phi * dZ0_dphi / R0
    )

    # 2nd order terms
    # expressions found from mathematica and converted to python

    ####### Z_2_2 ###########
    aux0 = 4.0 * (((R0) ** 3.0) * (((tauZ) * (Y2c)) + ((bZ) * (Z2cNAE))))
    aux1 = ((kphi) * (((X1s) * (X1sp)) - ((X1c) * (X1cp)))) + (
        (tauphi) * (((Y1s) * (X1sp)) - ((Y1c) * (X1cp)))
    )
    aux2 = (kphi) * ((tauphi) * (((X1s) * (Y1s)) - ((X1c) * (Y1c))))
    aux3 = (((kphi) ** 2) * (((X1s) ** 2) - ((X1c) ** 2))) + (
        (2.0 * aux2) + (((tauphi) ** 2) * (((Y1s) ** 2) - ((Y1c) ** 2)))
    )
    aux4 = ((X1c) * (((Y1c) * (tauZp)) + ((tauZ) * (Y1cp)))) + (2.0 * ((X2c) * (Z0p)))
    aux5 = ((((X1c) ** 2) * (kZp)) + aux4) - (
        (X1s) * (((Y1s) * (tauZp)) + ((tauZ) * (Y1sp)))
    )
    aux6 = (((Y1c) ** 2) * (tauZp)) + (
        ((tauZ) * ((Y1c) * (Y1cp))) + (2.0 * ((Y2c) * (Z0p)))
    )
    aux7 = (((X1c) * ((Y1c) * (kZp))) + aux6) - ((tauZ) * ((Y1s) * (Y1sp)))
    aux8 = (aux7 - (((Y1s) ** 2) * (tauZp))) - ((X1s) * ((Y1s) * (kZp)))
    aux9 = ((kphi) * (aux5 - (((X1s) ** 2) * (kZp)))) + ((tauphi) * aux8)
    aux10 = -2.0 * (((R0) ** 2) * ((2.0 * ((bphi) * ((Z2cNAE) * (Z0p)))) + aux9))
    aux11 = (tauR) * ((((X1c) * (Y1c)) - ((X1s) * (Y1s))) * (Z0p))
    aux12 = ((tauphi) * ((X1c) * ((Y1cp) * (Z0p)))) + (
        (tauphi) * ((X1c) * ((Y1c) * (Z0pp)))
    )
    aux13 = ((X1c) * ((Y1c) * ((tauphip) * (Z0p)))) + (
        ((tauphi) * ((Y1c) * ((X1cp) * (Z0p)))) + aux12
    )
    aux14 = ((kR) * ((((X1c) ** 2) - ((X1s) ** 2)) * (Z0p))) + (
        aux11 + ((((X1c) ** 2) * ((kphip) * (Z0p))) + aux13)
    )
    aux15 = (aux14 - ((tauphi) * ((X1s) * ((Y1s) * (Z0pp))))) - (
        (tauphi) * ((X1s) * ((Y1sp) * (Z0p)))
    )
    aux16 = (aux15 - ((tauphi) * ((Y1s) * ((X1sp) * (Z0p))))) - (
        (X1s) * ((Y1s) * ((tauphip) * (Z0p)))
    )
    aux17 = (kR) * ((((X1c) * (Y1c)) - ((X1s) * (Y1s))) * (Z0p))
    aux18 = (-2.0 * ((tauphi) * ((Y1s) * ((Y1sp) * (Z0p))))) + (
        (tauphi) * (((Y1c) ** 2) * (Z0pp))
    )
    aux19 = (-2.0 * (((Y1s) ** 2) * ((tauphip) * (Z0p)))) + (
        (2.0 * ((tauphi) * ((Y1c) * ((Y1cp) * (Z0p))))) + aux18
    )
    aux20 = (-2.0 * ((X1s) * ((Y1s) * ((kphip) * (Z0p))))) + (
        (2.0 * (((Y1c) ** 2) * ((tauphip) * (Z0p)))) + aux19
    )
    aux21 = (2.0 * ((tauR) * ((((Y1c) ** 2) - ((Y1s) ** 2)) * (Z0p)))) + (
        (2.0 * ((X1c) * ((Y1c) * ((kphip) * (Z0p))))) + aux20
    )
    aux22 = (tauphi) * (((2.0 * aux17) + aux21) - ((tauphi) * (((Y1s) ** 2) * (Z0pp))))
    aux23 = (2.0 * ((X1c) * ((X1cp) * (Z0p)))) + (((X1c) ** 2) * (Z0pp))
    aux24 = aux23 - ((X1s) * ((2.0 * ((X1sp) * (Z0p))) + ((X1s) * (Z0pp))))
    aux25 = (2.0 * ((kphi) * (aux16 - (((X1s) ** 2) * ((kphip) * (Z0p)))))) + (
        aux22 + (((kphi) ** 2) * aux24)
    )
    aux26 = (2.0 * ((kZ) * (((R0) ** 2) * ((2.0 * ((R0) * (X2c))) + aux1)))) + (
        (2.0 * (aux3 * ((R0p) * (Z0p)))) + (aux10 + ((R0) * aux25))
    )
    Z_2_2 = 0.25 * (((R0) ** -3.0) * (aux0 + aux26))

    ######################### Z_2_0 ###################
    aux0 = 4.0 * (((R0) ** 3.0) * (((tauZ) * (Y20)) + ((bZ) * (Z20NAE))))
    aux1 = ((kphi) * (((X1c) * (X1cp)) + ((X1s) * (X1sp)))) + (
        (tauphi) * (((Y1c) * (X1cp)) + ((Y1s) * (X1sp)))
    )
    aux2 = (kphi) * ((tauphi) * (((X1c) * (Y1c)) + ((X1s) * (Y1s))))
    aux3 = (((kphi) ** 2) * (((X1c) ** 2) + ((X1s) ** 2))) + (
        (2.0 * aux2) + (((tauphi) ** 2) * (((Y1c) ** 2) + ((Y1s) ** 2)))
    )
    aux4 = ((X1c) * (((Y1c) * (tauZp)) + ((tauZ) * (Y1cp)))) + (
        ((tauZ) * ((X1s) * (Y1sp))) + (2.0 * ((X20) * (Z0p)))
    )
    aux5 = (((X1s) ** 2) * (kZp)) + (((X1s) * ((Y1s) * (tauZp))) + aux4)
    aux6 = ((tauZ) * ((Y1c) * (Y1cp))) + (
        ((tauZ) * ((Y1s) * (Y1sp))) + (2.0 * ((Y20) * (Z0p)))
    )
    aux7 = (((Y1c) ** 2) * (tauZp)) + ((((Y1s) ** 2) * (tauZp)) + aux6)
    aux8 = ((X1c) * ((Y1c) * (kZp))) + (((X1s) * ((Y1s) * (kZp))) + aux7)
    aux9 = ((kphi) * ((((X1c) ** 2) * (kZp)) + aux5)) + ((tauphi) * aux8)
    aux10 = -2.0 * (((R0) ** 2) * ((2.0 * ((bphi) * ((Z20NAE) * (Z0p)))) + aux9))
    aux11 = (tauR) * ((((X1c) * (Y1c)) + ((X1s) * (Y1s))) * (Z0p))
    aux12 = ((tauphi) * ((X1c) * ((Y1c) * (Z0pp)))) + (
        (tauphi) * ((X1s) * ((Y1s) * (Z0pp)))
    )
    aux13 = ((tauphi) * ((X1c) * ((Y1cp) * (Z0p)))) + (
        ((tauphi) * ((X1s) * ((Y1sp) * (Z0p)))) + aux12
    )
    aux14 = ((tauphi) * ((Y1c) * ((X1cp) * (Z0p)))) + (
        ((tauphi) * ((Y1s) * ((X1sp) * (Z0p)))) + aux13
    )
    aux15 = ((X1c) * ((Y1c) * ((tauphip) * (Z0p)))) + (
        ((X1s) * ((Y1s) * ((tauphip) * (Z0p)))) + aux14
    )
    aux16 = (((X1c) ** 2) * ((kphip) * (Z0p))) + (
        (((X1s) ** 2) * ((kphip) * (Z0p))) + aux15
    )
    aux17 = ((kR) * ((((X1c) ** 2) + ((X1s) ** 2)) * (Z0p))) + (aux11 + aux16)
    aux18 = (kR) * ((((X1c) * (Y1c)) + ((X1s) * (Y1s))) * (Z0p))
    aux19 = ((tauphi) * (((Y1c) ** 2) * (Z0pp))) + ((tauphi) * (((Y1s) ** 2) * (Z0pp)))
    aux20 = (2.0 * ((tauphi) * ((Y1c) * ((Y1cp) * (Z0p))))) + (
        (2.0 * ((tauphi) * ((Y1s) * ((Y1sp) * (Z0p))))) + aux19
    )
    aux21 = (2.0 * (((Y1c) ** 2) * ((tauphip) * (Z0p)))) + (
        (2.0 * (((Y1s) ** 2) * ((tauphip) * (Z0p)))) + aux20
    )
    aux22 = (2.0 * ((X1c) * ((Y1c) * ((kphip) * (Z0p))))) + (
        (2.0 * ((X1s) * ((Y1s) * ((kphip) * (Z0p))))) + aux21
    )
    aux23 = (2.0 * ((tauR) * ((((Y1c) ** 2) + ((Y1s) ** 2)) * (Z0p)))) + aux22
    aux24 = (((X1c) ** 2) * (Z0pp)) + (
        (X1s) * ((2.0 * ((X1sp) * (Z0p))) + ((X1s) * (Z0pp)))
    )
    aux25 = ((tauphi) * ((2.0 * aux18) + aux23)) + (
        ((kphi) ** 2) * ((2.0 * ((X1c) * ((X1cp) * (Z0p)))) + aux24)
    )
    aux26 = (-2.0 * (aux3 * ((R0p) * (Z0p)))) + (
        aux10 + ((R0) * ((2.0 * ((kphi) * aux17)) + aux25))
    )
    aux27 = (-2.0 * ((kZ) * (((R0) ** 2) * ((-2.0 * ((R0) * (X20))) + aux1)))) + aux26
    Z_2_0 = 0.25 * (((R0) ** -3.0) * (aux0 + aux27))

    ################## Z_2_neg2 #################
    aux0 = 2.0 * (((R0) ** 3.0) * (((tauZ) * (Y2s)) + ((bZ) * (Z2sNAE))))
    aux1 = (2.0 * ((R0) * (X2s))) - ((tauphi) * (((Y1s) * (X1cp)) + ((Y1c) * (X1sp))))
    aux2 = ((R0) ** 2) * (aux1 - ((kphi) * (((X1s) * (X1cp)) + ((X1c) * (X1sp)))))
    aux3 = (((kphi) * (X1s)) + ((tauphi) * (Y1s))) * ((R0p) * (Z0p))
    aux4 = (tauR) * ((((X1s) * (Y1c)) + ((X1c) * (Y1s))) * (Z0p))
    aux5 = ((tauphi) * ((X1s) * ((Y1c) * (Z0pp)))) + (
        (tauphi) * ((X1c) * ((Y1s) * (Z0pp)))
    )
    aux6 = ((tauphi) * ((X1s) * ((Y1cp) * (Z0p)))) + (
        ((tauphi) * ((X1c) * ((Y1sp) * (Z0p)))) + aux5
    )
    aux7 = ((tauphi) * ((Y1s) * ((X1cp) * (Z0p)))) + (
        ((tauphi) * ((Y1c) * ((X1sp) * (Z0p)))) + aux6
    )
    aux8 = ((X1s) * ((Y1c) * ((tauphip) * (Z0p)))) + (
        ((X1c) * ((Y1s) * ((tauphip) * (Z0p)))) + aux7
    )
    aux9 = (2.0 * ((kR) * ((X1c) * ((X1s) * (Z0p))))) + (
        aux4 + ((2.0 * ((X1c) * ((X1s) * ((kphip) * (Z0p))))) + aux8)
    )
    aux10 = (kR) * ((((X1s) * (Y1c)) + ((X1c) * (Y1s))) * (Z0p))
    aux11 = ((tauphi) * ((Y1c) * ((Y1sp) * (Z0p)))) + (
        (tauphi) * ((Y1c) * ((Y1s) * (Z0pp)))
    )
    aux12 = (2.0 * ((Y1c) * ((Y1s) * ((tauphip) * (Z0p))))) + (
        ((tauphi) * ((Y1s) * ((Y1cp) * (Z0p)))) + aux11
    )
    aux13 = ((X1s) * ((Y1c) * ((kphip) * (Z0p)))) + (
        ((X1c) * ((Y1s) * ((kphip) * (Z0p)))) + aux12
    )
    aux14 = (tauphi) * ((2.0 * ((tauR) * ((Y1c) * ((Y1s) * (Z0p))))) + (aux10 + aux13))
    aux15 = ((X1c) * ((X1sp) * (Z0p))) + ((X1s) * (((X1cp) * (Z0p)) + ((X1c) * (Z0pp))))
    aux16 = (-2.0 * ((((kphi) * (X1c)) + ((tauphi) * (Y1c))) * aux3)) + (
        (R0) * (((kphi) * aux9) + (aux14 + (((kphi) ** 2) * aux15)))
    )
    aux17 = (2.0 * ((X1s) * (kZp))) + (((Y1s) * (tauZp)) + ((tauZ) * (Y1sp)))
    aux18 = ((tauZ) * ((X1s) * (Y1cp))) + (((X1c) * aux17) + (2.0 * ((X2s) * (Z0p))))
    aux19 = ((tauZ) * ((Y1s) * (Y1cp))) + (
        ((tauZ) * ((Y1c) * (Y1sp))) + (2.0 * ((Y2s) * (Z0p)))
    )
    aux20 = ((X1c) * ((Y1s) * (kZp))) + ((2.0 * ((Y1c) * ((Y1s) * (tauZp)))) + aux19)
    aux21 = ((kphi) * (((X1s) * ((Y1c) * (tauZp))) + aux18)) + (
        (tauphi) * (((X1s) * ((Y1c) * (kZp))) + aux20)
    )
    aux22 = (aux0 + (((kZ) * aux2) + aux16)) - (
        ((R0) ** 2) * ((2.0 * ((bphi) * ((Z2sNAE) * (Z0p)))) + aux21)
    )
    Z_2_neg2 = 0.5 * (((R0) ** -3.0) * aux22)

    ############### R_2_2 ###################
    aux0 = -2.0 * (((R0) ** 2) * ((tauphi) * ((X1c) * ((Y1c) * (kRp)))))
    aux1 = 2.0 * (((R0) ** 2) * ((tauphi) * ((X1s) * ((Y1s) * (kRp)))))
    aux2 = 2.0 * ((R0) * ((tauphi) * ((tauR) * (((Y1c) ** 2) * (R0p)))))
    aux3 = -2.0 * ((R0) * ((tauphi) * ((tauR) * (((Y1s) ** 2) * (R0p)))))
    aux4 = (R0) * ((tauphi) * ((X1c) * ((Y1c) * ((kphip) * (R0p)))))
    aux5 = (R0) * ((tauphi) * ((X1s) * ((Y1s) * ((kphip) * (R0p)))))
    aux6 = (R0) * ((tauphi) * (((Y1c) ** 2) * ((R0p) * (tauphip))))
    aux7 = (R0) * ((tauphi) * (((Y1s) ** 2) * ((R0p) * (tauphip))))
    aux8 = -2.0 * (((R0) ** 2) * ((tauphi) * (((Y1c) ** 2) * (tauRp))))
    aux9 = (tauphi) * ((((X1c) * (Y1c)) - ((X1s) * (Y1s))) * (R0p))
    aux10 = (R0) * ((tauphi) * (((Y1s) * (X1sp)) - ((Y1c) * (X1cp))))
    aux11 = (kR) * ((R0) * ((2.0 * (((R0) ** 2) * (X2c))) + (aux9 + aux10)))
    aux12 = ((R0) ** 2) * ((tauphi) * ((tauR) * ((Y1c) * (Y1cp))))
    aux13 = 2.0 * ((R0) * (((tauphi) ** 2) * ((Y1c) * ((R0p) * (Y1cp)))))
    aux14 = ((R0) ** 2) * ((tauphi) * ((tauR) * ((Y1s) * (Y1sp))))
    aux15 = -2.0 * ((R0) * (((tauphi) ** 2) * ((Y1s) * ((R0p) * (Y1sp)))))
    aux16 = (tauphi) * ((((X1s) * (Y1s)) - ((X1c) * (Y1c))) * ((R0p) ** 2))
    aux17 = ((kR) * ((X1s) * (X1sp))) + ((tauR) * ((X1s) * (Y1sp)))
    aux18 = (-2.0 * ((X2c) * (R0p))) + (((X1s) * ((Y1s) * (tauRp))) + aux17)
    aux19 = ((tauphi) * (((X1c) * (Y1c)) - ((X1s) * (Y1s)))) + (
        (((X1s) ** 2) * (kRp)) + aux18
    )
    aux20 = ((Y1c) * (tauRp)) + (((kR) * (X1cp)) + ((tauR) * (Y1cp)))
    aux21 = ((R0) ** 2) * ((aux19 - ((X1c) * aux20)) - (((X1c) ** 2) * (kRp)))
    aux22 = (tauR) * ((((X1c) * (Y1c)) - ((X1s) * (Y1s))) * (R0p))
    aux23 = ((tauphi) * ((X1c) * ((R0p) * (Y1cp)))) + (
        (tauphi) * ((X1c) * ((Y1c) * (R0pp)))
    )
    aux24 = ((X1c) * ((Y1c) * ((R0p) * (tauphip)))) + (
        ((tauphi) * ((Y1c) * ((R0p) * (X1cp)))) + aux23
    )
    aux25 = ((kR) * ((((X1c) ** 2) - ((X1s) ** 2)) * (R0p))) + (
        aux22 + ((((X1c) ** 2) * ((kphip) * (R0p))) + aux24)
    )
    aux26 = (aux25 - ((tauphi) * ((X1s) * ((Y1s) * (R0pp))))) - (
        (tauphi) * ((X1s) * ((R0p) * (Y1sp)))
    )
    aux27 = (aux26 - ((tauphi) * ((Y1s) * ((R0p) * (X1sp))))) - (
        (X1s) * ((Y1s) * ((R0p) * (tauphip)))
    )
    aux28 = aux21 + ((R0) * (aux27 - (((X1s) ** 2) * ((kphip) * (R0p)))))
    aux29 = (2.0 * ((X1c) * ((R0p) * (X1cp)))) + (((X1c) ** 2) * (R0pp))
    aux30 = aux29 - ((X1s) * ((2.0 * ((R0p) * (X1sp))) + ((X1s) * (R0pp))))
    aux31 = (2.0 * ((((X1s) ** 2) - ((X1c) ** 2)) * ((R0p) ** 2))) + ((R0) * aux30)
    aux32 = ((kphi) ** 2) * ((((R0) ** 2) * (((X1c) ** 2) - ((X1s) ** 2))) + aux31)
    aux33 = ((R0) * (((tauphi) ** 2) * (((Y1c) ** 2) * (R0pp)))) + (
        (2.0 * ((kphi) * ((2.0 * aux16) + aux28))) + aux32
    )
    aux34 = (2.0 * (((R0) ** 2) * ((tauphi) * (((Y1s) ** 2) * (tauRp))))) + (
        (2.0 * aux11) + ((-2.0 * aux12) + (aux13 + ((2.0 * aux14) + (aux15 + aux33))))
    )
    aux35 = (2.0 * (((tauphi) ** 2) * (((Y1s) ** 2) * ((R0p) ** 2)))) + (
        (2.0 * aux6) + ((-2.0 * aux7) + (aux8 + aux34))
    )
    aux36 = (-2.0 * (((tauphi) ** 2) * (((Y1c) ** 2) * ((R0p) ** 2)))) + aux35
    aux37 = (-4.0 * ((bphi) * (((R0) ** 2) * ((Z2cNAE) * (R0p))))) + (
        (2.0 * aux4) + ((-2.0 * aux5) + aux36)
    )
    aux38 = aux3 + ((-4.0 * (((R0) ** 2) * ((tauphi) * ((Y2c) * (R0p))))) + aux37)
    aux39 = (4.0 * ((bR) * (((R0) ** 3.0) * (Z2cNAE)))) + (
        aux0 + (aux1 + (aux2 + aux38))
    )
    aux40 = (((R0) ** 2) * (((tauphi) ** 2) * ((Y1c) ** 2))) + (
        (4.0 * (((R0) ** 3.0) * ((tauR) * (Y2c)))) + aux39
    )
    aux41 = aux40 - ((R0) * (((tauphi) ** 2) * (((Y1s) ** 2) * (R0pp))))
    aux42 = ((R0) ** -3.0) * (aux41 - (((R0) ** 2) * (((tauphi) ** 2) * ((Y1s) ** 2))))
    R_2_2 = 0.25 * aux42

    ############### R_2_neg2 ##############
    aux0 = (R0) * ((tauphi) * ((tauR) * ((Y1c) * ((Y1s) * (R0p)))))
    aux1 = (R0) * ((tauphi) * ((X1s) * ((Y1c) * ((kphip) * (R0p)))))
    aux2 = (R0) * ((tauphi) * ((X1c) * ((Y1s) * ((kphip) * (R0p)))))
    aux3 = (R0) * ((tauphi) * ((Y1c) * ((Y1s) * ((R0p) * (tauphip)))))
    aux4 = ((R0) ** 2) * ((tauphi) * ((Y1c) * ((Y1s) * (tauRp))))
    aux5 = (tauphi) * ((((X1s) * (Y1c)) + ((X1c) * (Y1s))) * (R0p))
    aux6 = (R0) * ((tauphi) * (((Y1s) * (X1cp)) + ((Y1c) * (X1sp))))
    aux7 = (tauphi) * ((((X1s) * (Y1c)) + ((X1c) * (Y1s))) * ((R0p) ** 2))
    aux8 = ((tauphi) * (((X1s) * (Y1c)) + ((X1c) * (Y1s)))) + (-2.0 * ((X2s) * (R0p)))
    aux9 = ((Y1s) * (tauRp)) + (((kR) * (X1sp)) + ((tauR) * (Y1sp)))
    aux10 = (aux8 - ((X1c) * ((2.0 * ((X1s) * (kRp))) + aux9))) - (
        (tauR) * ((X1s) * (Y1cp))
    )
    aux11 = (aux10 - ((kR) * ((X1s) * (X1cp)))) - ((X1s) * ((Y1c) * (tauRp)))
    aux12 = (tauR) * ((((X1s) * (Y1c)) + ((X1c) * (Y1s))) * (R0p))
    aux13 = ((tauphi) * ((X1s) * ((Y1c) * (R0pp)))) + (
        (tauphi) * ((X1c) * ((Y1s) * (R0pp)))
    )
    aux14 = ((tauphi) * ((X1s) * ((R0p) * (Y1cp)))) + (
        ((tauphi) * ((X1c) * ((R0p) * (Y1sp)))) + aux13
    )
    aux15 = ((tauphi) * ((Y1s) * ((R0p) * (X1cp)))) + (
        ((tauphi) * ((Y1c) * ((R0p) * (X1sp)))) + aux14
    )
    aux16 = ((X1s) * ((Y1c) * ((R0p) * (tauphip)))) + (
        ((X1c) * ((Y1s) * ((R0p) * (tauphip)))) + aux15
    )
    aux17 = (2.0 * ((kR) * ((X1c) * ((X1s) * (R0p))))) + (
        aux12 + ((2.0 * ((X1c) * ((X1s) * ((kphip) * (R0p))))) + aux16)
    )
    aux18 = ((X1c) * ((R0p) * (X1sp))) + ((X1s) * (((R0p) * (X1cp)) + ((X1c) * (R0pp))))
    aux19 = (((R0) ** 2) * ((X1c) * (X1s))) + (
        (-2.0 * ((X1c) * ((X1s) * ((R0p) ** 2)))) + ((R0) * aux18)
    )
    aux20 = ((kphi) * ((-2.0 * aux7) + ((((R0) ** 2) * aux11) + ((R0) * aux17)))) + (
        ((kphi) ** 2) * aux19
    )
    aux21 = ((R0) * (((tauphi) ** 2) * ((Y1c) * ((Y1s) * (R0pp))))) + aux20
    aux22 = ((R0) * (((tauphi) ** 2) * ((Y1c) * ((R0p) * (Y1sp))))) + aux21
    aux23 = ((R0) * (((tauphi) ** 2) * ((Y1s) * ((R0p) * (Y1cp))))) + aux22
    aux24 = ((kR) * ((R0) * (((2.0 * (((R0) ** 2) * (X2s))) + aux5) - aux6))) + aux23
    aux25 = (-2.0 * (((tauphi) ** 2) * ((Y1c) * ((Y1s) * ((R0p) ** 2))))) + (
        (2.0 * aux3) + ((-2.0 * aux4) + aux24)
    )
    aux26 = (-2.0 * ((bphi) * (((R0) ** 2) * ((Z2sNAE) * (R0p))))) + (
        aux1 + (aux2 + aux25)
    )
    aux27 = (2.0 * aux0) + (
        (-2.0 * (((R0) ** 2) * ((tauphi) * ((Y2s) * (R0p))))) + aux26
    )
    aux28 = (2.0 * (((R0) ** 3.0) * ((tauR) * (Y2s)))) + (
        (2.0 * ((bR) * (((R0) ** 3.0) * (Z2sNAE)))) + aux27
    )
    aux29 = (((R0) ** 2) * (((tauphi) ** 2) * ((Y1c) * (Y1s)))) + aux28
    aux30 = ((R0) ** 2) * ((tauphi) * ((tauR) * ((Y1c) * (Y1sp))))
    aux31 = ((R0) ** 2) * ((tauphi) * ((tauR) * ((Y1s) * (Y1cp))))
    aux32 = ((aux29 - aux30) - aux31) - (
        ((R0) ** 2) * ((tauphi) * ((X1c) * ((Y1s) * (kRp))))
    )
    aux33 = aux32 - (((R0) ** 2) * ((tauphi) * ((X1s) * ((Y1c) * (kRp)))))
    R_2_neg2 = 0.5 * (((R0) ** -3.0) * aux33)

    ############# R_2_0 ##############
    aux0 = -2.0 * (((R0) ** 2) * ((tauphi) * ((X1c) * ((Y1c) * (kRp)))))
    aux1 = -2.0 * (((R0) ** 2) * ((tauphi) * ((X1s) * ((Y1s) * (kRp)))))
    aux2 = 2.0 * ((R0) * ((tauphi) * ((tauR) * (((Y1c) ** 2) * (R0p)))))
    aux3 = 2.0 * ((R0) * ((tauphi) * ((tauR) * (((Y1s) ** 2) * (R0p)))))
    aux4 = (R0) * ((tauphi) * ((X1c) * ((Y1c) * ((kphip) * (R0p)))))
    aux5 = (R0) * ((tauphi) * ((X1s) * ((Y1s) * ((kphip) * (R0p)))))
    aux6 = (R0) * ((tauphi) * (((Y1c) ** 2) * ((R0p) * (tauphip))))
    aux7 = (R0) * ((tauphi) * (((Y1s) ** 2) * ((R0p) * (tauphip))))
    aux8 = -2.0 * (((R0) ** 2) * ((tauphi) * (((Y1c) ** 2) * (tauRp))))
    aux9 = -2.0 * (((R0) ** 2) * ((tauphi) * (((Y1s) ** 2) * (tauRp))))
    aux10 = (tauphi) * ((((X1c) * (Y1c)) + ((X1s) * (Y1s))) * (R0p))
    aux11 = (R0) * ((tauphi) * (((Y1c) * (X1cp)) + ((Y1s) * (X1sp))))
    aux12 = (kR) * ((R0) * (((2.0 * (((R0) ** 2) * (X20))) + aux10) - aux11))
    aux13 = ((R0) ** 2) * ((tauphi) * ((tauR) * ((Y1c) * (Y1cp))))
    aux14 = 2.0 * ((R0) * (((tauphi) ** 2) * ((Y1c) * ((R0p) * (Y1cp)))))
    aux15 = ((R0) ** 2) * ((tauphi) * ((tauR) * ((Y1s) * (Y1sp))))
    aux16 = 2.0 * ((R0) * (((tauphi) ** 2) * ((Y1s) * ((R0p) * (Y1sp)))))
    aux17 = (tauphi) * ((((X1c) * (Y1c)) + ((X1s) * (Y1s))) * ((R0p) ** 2))
    aux18 = ((tauphi) * (((X1c) * (Y1c)) + ((X1s) * (Y1s)))) + (-2.0 * ((X20) * (R0p)))
    aux19 = ((Y1c) * (tauRp)) + (((kR) * (X1cp)) + ((tauR) * (Y1cp)))
    aux20 = ((aux18 - ((tauR) * ((X1s) * (Y1sp)))) - ((X1c) * aux19)) - (
        (kR) * ((X1s) * (X1sp))
    )
    aux21 = (aux20 - ((X1s) * ((Y1s) * (tauRp)))) - (((X1s) ** 2) * (kRp))
    aux22 = (tauR) * ((((X1c) * (Y1c)) + ((X1s) * (Y1s))) * (R0p))
    aux23 = ((tauphi) * ((X1c) * ((Y1c) * (R0pp)))) + (
        (tauphi) * ((X1s) * ((Y1s) * (R0pp)))
    )
    aux24 = ((tauphi) * ((X1c) * ((R0p) * (Y1cp)))) + (
        ((tauphi) * ((X1s) * ((R0p) * (Y1sp)))) + aux23
    )
    aux25 = ((tauphi) * ((Y1c) * ((R0p) * (X1cp)))) + (
        ((tauphi) * ((Y1s) * ((R0p) * (X1sp)))) + aux24
    )
    aux26 = ((X1c) * ((Y1c) * ((R0p) * (tauphip)))) + (
        ((X1s) * ((Y1s) * ((R0p) * (tauphip)))) + aux25
    )
    aux27 = (((X1c) ** 2) * ((kphip) * (R0p))) + (
        (((X1s) ** 2) * ((kphip) * (R0p))) + aux26
    )
    aux28 = ((kR) * ((((X1c) ** 2) + ((X1s) ** 2)) * (R0p))) + (aux22 + aux27)
    aux29 = (((R0) ** 2) * (aux21 - (((X1c) ** 2) * (kRp)))) + ((R0) * aux28)
    aux30 = (((X1c) ** 2) * (R0pp)) + (
        (X1s) * ((2.0 * ((R0p) * (X1sp))) + ((X1s) * (R0pp)))
    )
    aux31 = (-2.0 * ((((X1c) ** 2) + ((X1s) ** 2)) * ((R0p) ** 2))) + (
        (R0) * ((2.0 * ((X1c) * ((R0p) * (X1cp)))) + aux30)
    )
    aux32 = ((kphi) ** 2) * ((((R0) ** 2) * (((X1c) ** 2) + ((X1s) ** 2))) + aux31)
    aux33 = ((R0) * (((tauphi) ** 2) * (((Y1s) ** 2) * (R0pp)))) + (
        (2.0 * ((kphi) * ((-2.0 * aux17) + aux29))) + aux32
    )
    aux34 = ((R0) * (((tauphi) ** 2) * (((Y1c) ** 2) * (R0pp)))) + aux33
    aux35 = aux8 + (
        aux9
        + (
            (2.0 * aux12)
            + ((-2.0 * aux13) + (aux14 + ((-2.0 * aux15) + (aux16 + aux34))))
        )
    )
    aux36 = (-2.0 * (((tauphi) ** 2) * (((Y1s) ** 2) * ((R0p) ** 2)))) + (
        (2.0 * aux6) + ((2.0 * aux7) + aux35)
    )
    aux37 = (-2.0 * (((tauphi) ** 2) * (((Y1c) ** 2) * ((R0p) ** 2)))) + aux36
    aux38 = (-4.0 * ((bphi) * (((R0) ** 2) * ((Z20NAE) * (R0p))))) + (
        (2.0 * aux4) + ((2.0 * aux5) + aux37)
    )
    aux39 = aux3 + ((-4.0 * (((R0) ** 2) * ((tauphi) * ((Y20) * (R0p))))) + aux38)
    aux40 = (4.0 * ((bR) * (((R0) ** 3.0) * (Z20NAE)))) + (
        aux0 + (aux1 + (aux2 + aux39))
    )
    aux41 = (((R0) ** 2) * (((tauphi) ** 2) * ((Y1s) ** 2))) + (
        (4.0 * (((R0) ** 3.0) * ((tauR) * (Y20)))) + aux40
    )
    aux42 = ((R0) ** -3.0) * ((((R0) ** 2) * (((tauphi) ** 2) * ((Y1c) ** 2))) + aux41)
    R_2_0 = 0.25 * aux42

    # Fourier Transform in toroidal angle phi
    coeffs = {}
    bases = {}

    nfp = qsc.nfp
    if desc_eq.sym:
        Rbasis = FourierSeries(N=desc_eq.N, NFP=nfp, sym="cos")
        Zbasis = FourierSeries(N=desc_eq.N, NFP=nfp, sym="cos")
        Rbasis_sin = FourierSeries(N=desc_eq.N, NFP=nfp, sym="sin")
        Zbasis_sin = FourierSeries(N=desc_eq.N, NFP=nfp, sym="sin")
    else:
        Rbasis = FourierSeries(N=desc_eq.N, NFP=nfp, sym=False)
        Zbasis = FourierSeries(N=desc_eq.N, NFP=nfp, sym=False)
        Rbasis_sin = FourierSeries(N=desc_eq.N, NFP=nfp, sym=False)
        Zbasis_sin = FourierSeries(N=desc_eq.N, NFP=nfp, sym=False)
    bases["Rbasis_cos"] = Rbasis
    bases["Rbasis_sin"] = Rbasis_sin
    bases["Zbasis_cos"] = Zbasis
    bases["Zbasis_sin"] = Zbasis_sin

    grid = LinearGrid(M=0, L=0, zeta=phi, NFP=nfp)
    Rtrans = Transform(grid, Rbasis, build_pinv=True, method="auto")
    Ztrans = Transform(grid, Zbasis, build_pinv=True, method="auto")
    Rtrans_sin = Transform(grid, Rbasis_sin, build_pinv=True, method="auto")
    Ztrans_sin = Transform(grid, Zbasis_sin, build_pinv=True, method="auto")

    R_1_1_n = Rtrans.fit(R_1_1)
    R_1_neg1_n = Rtrans_sin.fit(R_1_neg1)

    coeffs["R_1_1_n"] = R_1_1_n
    coeffs["R_1_neg1_n"] = R_1_neg1_n

    Z_1_1_n = Ztrans_sin.fit(Z_1_1)
    Z_1_neg1_n = Ztrans.fit(Z_1_neg1)
    coeffs["Z_1_1_n"] = Z_1_1_n
    coeffs["Z_1_neg1_n"] = Z_1_neg1_n

    # if stell sym we should be able to figure out the phi sym for each term
    # R cos terms need a cos phi basis
    # R sin terms need sin phi basis
    # Z cos terms need sin phi basis
    # Z sin terms need cos phi basis

    R_2_0_n = Rtrans.fit(R_2_0)
    R_2_2_n = Rtrans.fit(R_2_2)
    R_2_neg2_n = Rtrans_sin.fit(R_2_neg2)

    coeffs["R_2_0_n"] = R_2_0_n
    coeffs["R_2_2_n"] = R_2_2_n
    coeffs["R_2_neg2_n"] = R_2_neg2_n

    Z_2_0_n = Ztrans_sin.fit(Z_2_0)
    Z_2_2_n = Ztrans_sin.fit(Z_2_2)
    Z_2_neg2_n = Ztrans.fit(Z_2_neg2)

    coeffs["Z_2_0_n"] = Z_2_0_n
    coeffs["Z_2_2_n"] = Z_2_2_n
    coeffs["Z_2_neg2_n"] = Z_2_neg2_n

    return coeffs, bases


def _calc_2nd_order_constraints(qsc, desc_eq, coeffs, bases):
    """Creates 2nd order NAE constraints for a DESC eq based off given qsc eq.

    Parameters
    ----------
        qsc (Qsc): pyQsc Qsc object to use as the NAE constraints on the DESC eq
        desc_eq (Equilibrium): desc equilibrium to constrain
        coeffs: dict, dictionary of arrays with keys like 'X_L_M_n', where
                X is R or Z, L is 1 or 2, and M is 0,1, or 2, are the
                NAE Fourier (in tor. phi) coeffs of radial order L and poloidal order M
        bases: dict, dictionary of Rbasis_cos, Rbasis_sin, Zbasis_cos, Zbasis_sin,
            the FourierSeries basis objects used to obtain the coefficients, where
            _cos or _sin denotes the symmetry of the Fourier series.
            symmetry is such that the R or Z coefficients is stellarator symmetric
            i.e. R_1_1_n uses the Rbasis_cos, since cos(theta)*cos(phi) is
             stellarator symmetric for R i.e. R(-theta,-phi) = R(theta,phi)
            and Z_1_1_n uses the Zbasis_sin as the term is cos(theta)*sin(phi)
            since Z(-theta,-phi) = - Z(theta,phi) for Z stellarator symmetry
    Returns
    -------
        Rconstraints (tuple): tuple of FixSumModesR constraints corresponding
         to constraining the O(rho) DESC coefficients, to be used in
         constraining a DESC equilibrium solve
        Zconstraints (tuple): tuple of FixSumModesZ constraints corresponding
         to constraining the O(rho) DESC coefficients, to be used in
         constraining a DESC equilibrium solve
    Notes
    ----
        follows eqns 30a and 30b in NAE2DESC document
    """
    r = 2 * desc_eq.Psi / qsc.Bbar / 2 / np.pi  # this is the r over rho factor,
    # squared (bc is rho^2 and r^2 terms considering here)
    Rconstraints = ()
    Zconstraints = ()

    Rbasis_cos = bases["Rbasis_cos"]
    Rbasis_sin = bases["Rbasis_sin"]
    Zbasis_cos = bases["Zbasis_cos"]
    Zbasis_sin = bases["Zbasis_sin"]

    # R_20n i.e. L=2, M=0
    for n, NAEcoeff in zip(Rbasis_cos.modes[:, 2], coeffs["R_2_0_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.L / 2) + 1):
            modes.append([2 * k, 0, n])
            sum_weights.append([(-1) ** k * k * (k + 1)])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Rcon = FixSumModesR(target=target, sum_weights=sum_weights, modes=modes)
        desc_eq.R_lmn
        Rconstraints += (Rcon,)
    # R_2_2n
    for n, NAEcoeff in zip(Rbasis_cos.modes[:, 2], coeffs["R_2_2_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.L / 2) + 1):
            modes.append([2 * k, 2, n])
            sum_weights.append([(-1) ** k * k * (k + 1)])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights) / 2
        Rcon = FixSumModesR(target=target, sum_weights=sum_weights, modes=modes)
        Rconstraints += (Rcon,)
    # R_2_neg2n
    for n, NAEcoeff in zip(Rbasis_sin.modes[:, 2], coeffs["R_2_neg2_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.L / 2) + 1):
            modes.append([2 * k, -2, n])
            sum_weights.append([(-1) ** k * k * (k + 1)])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights) / 2
        Rcon = FixSumModesR(target=target, sum_weights=sum_weights, modes=modes)
        Rconstraints += (Rcon,)
    # Z_2_0n
    for n, NAEcoeff in zip(Zbasis_sin.modes[:, 2], coeffs["Z_2_0_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.L / 2) + 1):
            modes.append([2 * k, 0, n])
            sum_weights.append([(-1) ** k * k * (k + 1)])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights)
        Zcon = FixSumModesZ(target=target, sum_weights=sum_weights, modes=modes)
        Zconstraints += (Zcon,)
    # Z_2_neg2n
    for n, NAEcoeff in zip(Zbasis_cos.modes[:, 2], coeffs["Z_2_neg2_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.M / 2) + 1):
            modes.append([2 * k, -2, n])
            sum_weights.append([(-1) ** k * k * (k + 1)])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights) / 2
        Zcon = FixSumModesZ(target=target, sum_weights=sum_weights, modes=modes)
        Zconstraints += (Zcon,)
    # Z_2_2n
    for n, NAEcoeff in zip(Zbasis_sin.modes[:, 2], coeffs["Z_2_2_n"]):
        sum_weights = []
        modes = []
        target = NAEcoeff * r
        for k in range(1, int(desc_eq.M / 2) + 1):
            modes.append([2 * k, 2, n])
            sum_weights.append([(-1) ** k * k * (k + 1)])
        modes = np.atleast_2d(modes)
        sum_weights = -np.atleast_1d(sum_weights) / 2
        Zcon = FixSumModesZ(target=target, sum_weights=sum_weights, modes=modes)
        Zconstraints += (Zcon,)

    return Rconstraints, Zconstraints


def make_RZ_cons_2nd_order(qsc, desc_eq):
    """Make the second order NAE constraints for a DESC equilibrium.

    Parameters
    ----------
        qsc (Qsc): pyQsc Qsc object to use as the NAE constraints on the DESC eq
        desc_eq (Equilibrium): desc equilibrium to constrain

    Returns
    -------
        Rconstraints tuple: tuple of FixSumModesR constraints corresponding to
         constraining the O(rho^2) DESC coefficients,
         to be used in constraining a DESC equilibrium solve
        Zconstraints tuple: tuple of FixSumModesZ constraints corresponding to
         constraining the O(rho^2) DESC coefficients,
         to be used in constraining a DESC equilibrium solve
    """
    Rconstraints = ()
    Zconstraints = ()

    coeffs, bases = _calc_2nd_order_NAE_coeffs(qsc, desc_eq)
    Rconstraints, Zconstraints = _calc_2nd_order_constraints(
        qsc, desc_eq, coeffs, bases
    )

    return Rconstraints + Zconstraints
