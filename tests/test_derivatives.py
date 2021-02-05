import unittest
import numpy as np

from desc.backend import jnp
from desc.derivatives import AutoDiffDerivative, FiniteDiffDerivative

from numpy.random import default_rng


class TestDerivative(unittest.TestCase):
    """Tests Grid classes"""

    def test_finite_diff_vec(self):
        def test_fun(x, y, a):
            return x * y + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jac = FiniteDiffDerivative(test_fun, argnum=0)
        J = jac.compute(x, y, a)
        correct_J = np.diag(y)

        np.testing.assert_allclose(J, correct_J, atol=1e-8)

    def test_finite_diff_scalar(self):
        def test_fun(x, y, a):
            return np.dot(x, y) + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jac = FiniteDiffDerivative(test_fun, argnum=0)
        J = jac.compute(x, y, a)
        correct_J = y

        np.testing.assert_allclose(J, correct_J, atol=1e-8)

        jac.argnum = 1
        J = jac.compute(x, y, a)
        np.testing.assert_allclose(J, x, atol=1e-8)

    def test_auto_diff(self):
        def test_fun(x, y, a):
            return jnp.cos(x) + x * y + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jac = AutoDiffDerivative(test_fun, argnum=0)
        J = jac.compute(x, y, a)
        correct_J = np.diag(-np.sin(x) + y)

        np.testing.assert_allclose(J, correct_J, atol=1e-8)

    def test_compare_AD_FD(self):
        def test_fun(x, y, a):
            return jnp.cos(x) + x * y + a

        x = np.array([1, 5, 0.01, 200])
        y = np.array([60, 1, 100, 0.02])
        a = -2

        jac_AD = AutoDiffDerivative(test_fun, argnum=0)
        J_AD = jac_AD.compute(x, y, a)

        jac_FD = AutoDiffDerivative(test_fun, argnum=0)
        J_FD = jac_FD.compute(x, y, a)

        np.testing.assert_allclose(J_FD, J_AD, atol=1e-8)

    def test_fd_hessian(self):
        rando = default_rng(seed=0)

        n = 5
        A = rando.random((n, n))
        A = A + A.T
        g = rando.random(n)

        def f(x):
            return 5 + g.dot(x) + x.dot(1 / 2 * A.dot(x))

        hess = FiniteDiffDerivative(f, argnum=0, mode="hess")

        y = rando.random(n)
        A1 = hess(y)

        np.testing.assert_allclose(A1, A)


class TestJVP(unittest.TestCase):
    @staticmethod
    def fun(x, c1, c2):
        Amat = np.arange(12).reshape((4, 3))
        return jnp.dot(Amat, (x + c1 * c2) ** 2)

    x = np.ones(3).astype(float)
    c1 = np.arange(3).astype(float)
    c2 = np.arange(3).astype(float) + 2

    dx = np.array([1, 2, 3]).astype(float)
    dc1 = np.array([3, 4, 5]).astype(float)
    dc2 = np.array([-3, 1, -2]).astype(float)

    def test_autodiff_jvp(self):

        df = AutoDiffDerivative.compute_jvp(
            self.fun, 0, self.dx, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([124.0, 340.0, 556.0, 772.0]))
        df = AutoDiffDerivative.compute_jvp(
            self.fun, 1, self.dc1, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([816.0, 2220.0, 3624.0, 5028.0]))
        df = AutoDiffDerivative.compute_jvp(
            self.fun, (0, 2), (self.dx, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([-12.0, 12.0, 36.0, 60.0]))

    def test_finitediff_jvp(self):

        df = FiniteDiffDerivative.compute_jvp(
            self.fun, 0, self.dx, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([124.0, 340.0, 556.0, 772.0]))
        df = FiniteDiffDerivative.compute_jvp(
            self.fun, 1, self.dc1, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([816.0, 2220.0, 3624.0, 5028.0]))
        df = FiniteDiffDerivative.compute_jvp(
            self.fun, (0, 2), (self.dx, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([-12.0, 12.0, 36.0, 60.0]))

    def test_autodiff_jvp2(self):

        df = AutoDiffDerivative.compute_jvp2(
            self.fun, 0, 0, self.dx + 1, self.dx, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([60.0, 180.0, 300.0, 420.0]))
        df = AutoDiffDerivative.compute_jvp2(
            self.fun, 1, 1, self.dc1 + 1, self.dc1, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([2280.0, 6528.0, 10776.0, 15024.0]))
        df = AutoDiffDerivative.compute_jvp2(
            self.fun, 0, 2, self.dx, self.dc2, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([-44.0, -104.0, -164.0, -224.0]))
        df = AutoDiffDerivative.compute_jvp2(
            self.fun, 0, (1, 2), self.dx, (self.dc1, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([244.0, 724.0, 1204.0, 1684.0]))
        df = AutoDiffDerivative.compute_jvp2(
            self.fun,
            (1, 2),
            (1, 2),
            (self.dc1, self.dc2),
            (self.dc1, self.dc2),
            self.x,
            self.c1,
            self.c2,
        )
        np.testing.assert_allclose(df, np.array([706.0, 2476.0, 4246.0, 6016.0]))
        df = AutoDiffDerivative.compute_jvp2(
            self.fun, 0, (1, 2), self.dx, (self.dc1, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([244.0, 724.0, 1204.0, 1684.0]))

    def test_finitediff_jvp2(self):

        df = FiniteDiffDerivative.compute_jvp2(
            self.fun, 0, 0, self.dx + 1, self.dx, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([60.0, 180.0, 300.0, 420.0]))
        df = FiniteDiffDerivative.compute_jvp2(
            self.fun, 1, 1, self.dc1 + 1, self.dc1, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([2280.0, 6528.0, 10776.0, 15024.0]))
        df = FiniteDiffDerivative.compute_jvp2(
            self.fun, 0, 2, self.dx, self.dc2, self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([-44.0, -104.0, -164.0, -224.0]))
        df = FiniteDiffDerivative.compute_jvp2(
            self.fun, 0, (1, 2), self.dx, (self.dc1, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([244.0, 724.0, 1204.0, 1684.0]))
        df = FiniteDiffDerivative.compute_jvp2(
            self.fun,
            (1, 2),
            (1, 2),
            (self.dc1, self.dc2),
            (self.dc1, self.dc2),
            self.x,
            self.c1,
            self.c2,
        )
        np.testing.assert_allclose(df, np.array([706.0, 2476.0, 4246.0, 6016.0]))
        df = FiniteDiffDerivative.compute_jvp2(
            self.fun, 0, (1, 2), self.dx, (self.dc1, self.dc2), self.x, self.c1, self.c2
        )
        np.testing.assert_allclose(df, np.array([244.0, 724.0, 1204.0, 1684.0]))
