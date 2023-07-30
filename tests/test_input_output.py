"""Tests for reading/writing intput/output, both ascii and binary."""

import os
import pathlib
import shutil

import h5py
import numpy as np
import pytest

from desc.basis import FourierZernikeBasis
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.io import InputReader, hdf5Reader, hdf5Writer, load
from desc.io.ascii_io import read_ascii, write_ascii
from desc.transform import Transform
from desc.utils import equals


@pytest.mark.unit
def test_vmec_input(tmpdir_factory):
    """Test converting VMEC to DESC input file."""
    # input.DSHAPE has multi-line inputs and
    # a duplicate AXIS input line as well, so
    # exercises full VMEC capability of input reader
    input_path = "./tests/inputs/input.DSHAPE"
    tmpdir = tmpdir_factory.mktemp("desc_inputs")
    tmp_path = tmpdir.join("input.DSHAPE")
    shutil.copyfile(input_path, tmp_path)
    with pytest.warns(UserWarning):
        ir = InputReader(cl_args=[str(tmp_path)])
    vmec_inputs = ir.inputs
    # ir makes a VMEC file automatically
    path_converted_file = tmpdir.join("input.DSHAPE_desc")
    # also test making a DESC file from the ir.inputs manually
    path = tmpdir.join("desc_from_vmec")
    ir.write_desc_input(path, ir.inputs)
    ir2 = InputReader(cl_args=[str(path)])
    desc_inputs = ir2.inputs
    for d, v in zip(desc_inputs, vmec_inputs):
        d.pop("output_path")
        v.pop("output_path")
    assert all([equals(in1, in2) for in1, in2 in zip(vmec_inputs, desc_inputs)])

    correct_file_path = ".//tests//inputs//input.DSHAPE_desc"

    # check DESC input file matches known correct one line-by-line
    with open(correct_file_path) as f:
        lines_correct = f.readlines()
    with open(path) as f:
        lines_direct = f.readlines()
    with open(path_converted_file) as f:
        lines_converted = f.readlines()
    # skip first 3 lines as they have date and pwd info
    for line1, line2 in zip(lines_correct[3:], lines_converted[4:]):
        assert line1.strip() == line2.strip()
    for line1, line2 in zip(lines_correct[3:], lines_direct):
        assert line1.strip() == line2.strip()


@pytest.mark.unit
def test_write_desc_input_Nones(tmpdir_factory):
    """Test converting writing DESC input file when an input tol is None."""
    # tests how None is handled as a passed-in input to the input writer
    # if None is passed for one of the elements of an input item
    # such as ftol, gtol, xtol or nfev,
    # then that will not be written
    # for example if inputs['ftol'] = [1e-2,None,1e-3]
    # the written file will have ftol = 0.01,0.001
    # and the None will have not been written

    # if only Nones are passed for just one of the tols, only that one will not
    # be written. gtol is used here as that test

    input_path = "./tests/inputs/DSHAPE"
    tmpdir = tmpdir_factory.mktemp("desc_inputs")
    tmp_path = tmpdir.join("DSHAPE")
    shutil.copyfile(input_path, tmp_path)
    ir = InputReader(cl_args=[str(tmp_path)])

    ftols_input_with_none = [1e-2, None, 1e-3]
    for i, inp in enumerate(ir.inputs):
        inp["ftol"] = ftols_input_with_none[i]
        inp["gtol"] = None

    path = tmpdir.join("desc_with_None")
    ir.write_desc_input(path, ir.inputs)
    ir2 = InputReader(cl_args=[str(path)])
    correct_ftols = [1e-2, 1e-3, 1e-3]
    for i, inp in enumerate(ir2.inputs):
        assert inp["ftol"] == correct_ftols[i]

    # now check that the written line is
    # the correct "ftol = 1e-2, 1e-3"
    # and that gtol is NOT written anywhere,
    # since only None was passed in for that input
    no_gtol = True
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        if line.find("ftol") != -1:
            # line is like "ftol = 0.01, 0.001\n"
            line = line.strip().split("=")[1]
            # now we have [" 0.01, 0.001"]
            line = line.strip().split(",")
            for i, string_num in enumerate(line):
                assert float(string_num) == correct_ftols[i]
        elif line.find("gtol") != -1:
            no_gtol = False
    assert no_gtol  # fail test if gtol line is written


@pytest.mark.unit
def test_descout_to_input(tmpdir_factory):
    """Test converting DESC output to a DESC input file. To do that we covert a DESC output file to
    an input file, denoted by desc_input_converted. We compare the boundary Fourier coefficients of
    this file with the true values of the Fourier coefficients, given in desc_input_truth"""

    outfile_path = "./tests/inputs/LandremanPaul2022_QA_reactorScale_lowRes.h5"
    tmpdir = tmpdir_factory.mktemp("desc_inputs")
    tmp_path = tmpdir.join("input_LandremanPaul_QA")
    tmpout_path = tmpdir.join("LandremanPaul2022_QA_reactorScale_lowRes.h5")
    shutil.copyfile(outfile_path, tmpout_path)

    ir = InputReader()
    ir.descout_to_input(str(tmp_path), str(tmpout_path))
    desc_input_truth = ".//tests//inputs//LandremanPaul2022_QA_reactorScale_lowRes"
    desc_input_converted = tmp_path

    with open(desc_input_truth) as f:
        lines_correct = f.readlines()
    with open(desc_input_converted) as f:
        lines_converted = f.readlines()

    # Count the lines before the boundary modes
    counter0 = int(0)
    found0 = int(0)
    while found0 != 1:
        if (
            lines_correct[counter0].split("\n")[0].replace(" ", "")
            == "#fixed-boundarysurfaceshape"
        ):
            found0 = 1
            counter0 += 1
        else:
            counter0 += 1

    counter1 = int(0)
    found0 = int(0)
    while found0 != 1:
        if (
            lines_converted[counter1].split("\n")[0].replace(" ", "")
            == "#fixed-boundarysurfaceshape"
        ):
            found0 = 1
            counter1 += 1
        else:
            counter1 += 1

    # Tells us if the m mode number comes before n in the input file.
    mfirst0 = (
        lines_correct[counter0].split("\t")[1].split(":")[0].replace(" ", "") == "m"
    )

    found0 = 1

    i = int(counter1)
    j = int(counter0)

    # Compare the Fourier modes and ensure that they match, i.e,, found0 = 1
    if mfirst0 == True:
        while i < np.minimum(counter1 + 20, len(lines_converted)) and found0 == 1:
            m1_idx = int(eval(lines_converted[i].split("\t")[1].split(":")[1]))
            n1_idx = int(eval(lines_converted[i].split("\t")[2].split(":")[1]))
            R11 = eval(lines_converted[i].split("\t")[3].split("=")[1])
            Z11 = eval(lines_converted[i].split("\t")[4].split("=")[1])
            j = counter0
            found0 = 0
            while j < int(len(lines_correct)) and found0 == 0:
                m0_idx = int(eval(lines_correct[j].split("\t")[1].split(":")[1]))
                n0_idx = int(eval(lines_correct[j].split("\t")[2].split(":")[1]))
                R10 = eval(lines_correct[j].split("\t")[3].split("=")[1])
                Z10 = eval(lines_correct[j].split("\t")[4].split("=")[1])
                if m0_idx == m1_idx and n0_idx == n1_idx:
                    if (
                        np.minimum(
                            np.linalg.norm(np.array([R11, Z11]) - np.array([R10, Z10])),
                            np.linalg.norm(np.array([R11, Z11]) + np.array([R10, Z10])),
                        )
                        < 1e-3
                    ):
                        found0 = 1
                    else:
                        found0 = 0
                    j = len(lines_correct)
                else:
                    j += 1
            i += 1
    else:
        while i < counter1 + 10 and found0 == 1:
            m1_idx = int(eval(lines_converted[i].split("\t")[1].split(":")[1]))
            n1_idx = int(eval(lines_converted[i].split("\t")[2].split(":")[1]))
            R11 = eval(lines_converted[i].split("\t")[3].split("=")[1])
            Z11 = eval(lines_converted[i].split("\t")[4].split("=")[1])
            j = counter0
            found0 = 0
            while j < int(len(lines_correct)) and found0 == 0:
                m0_idx = int(eval(lines_correct[j].split("\t")[2].split(":")[1]))
                n0_idx = int(eval(lines_correct[j].split("\t")[1].split(":")[1]))
                R10 = eval(lines_correct[j].split("\t")[3].split("=")[1])
                Z10 = eval(lines_correct[j].split("\t")[4].split("=")[1])
                if m0_idx == m1_idx and n0_idx == n1_idx:
                    print(
                        np.array([m1_idx, n1_idx, R11, Z11]),
                        np.array([m0_idx, n0_idx, R10, Z10]),
                    )
                    if (
                        np.minimum(
                            np.linalg.norm(np.array([R11, Z11]) - np.array([R10, Z10])),
                            np.linalg.norm(np.array([R11, Z11]) + np.array([R10, Z10])),
                        )
                        < 1e-3
                    ):
                        found0 = 1
                    else:
                        found0 = 0
                    j = len(lines_correct)
                else:
                    j += 1
            i += 1

    assert found0 == 1
    ## skip first 3 lines as they have date and pwd info
    # for line1, line2 in zip(lines_correct[3:], lines_converted[4:]):
    #    assert line1.strip() == line2.strip()


@pytest.mark.unit
def test_near_axis_input_files():
    """Test that DESC and VMEC input files generated by pyQSC give the same inputs."""
    vmec_path = ".//tests//inputs//input.QSC_r2_5.5_vmec"
    desc_path = ".//tests//inputs//input.QSC_r2_5.5_desc"
    with pytest.warns(UserWarning):
        inputs_vmec = InputReader(vmec_path).inputs[-1]
    inputs_desc = InputReader(desc_path).inputs[-1]
    for arg in ["sym", "NFP", "Psi", "pressure", "current", "surface", "axis"]:
        np.testing.assert_allclose(
            inputs_desc[arg], inputs_vmec[arg], rtol=1e-6, atol=1e-8
        )
    if os.path.exists(".//tests//inputs//input.QSC_r2_5.5_vmec_desc"):
        os.remove(".//tests//inputs//input.QSC_r2_5.5_vmec_desc")


@pytest.mark.unit
def test_vmec_input_surface_threshold():
    """Test ."""
    path = ".//tests//inputs//input.QSC_r2_5.5_vmec"
    with pytest.warns(UserWarning, match="Detected multiple inputs"):
        surf_full = InputReader.parse_vmec_inputs(path)[-1]["surface"]
        surf_trim = InputReader.parse_vmec_inputs(path, threshold=1e-6)[-1]["surface"]
    assert surf_full.shape[0] > surf_trim.shape[0]
    assert surf_full.shape[1] == surf_trim.shape[1] == 5


class TestInputReader:
    """Tests for the InputReader class."""

    argv0 = []
    argv1 = ["nonexistant_input_file"]
    argv2 = ["./tests/inputs/MIN_INPUT"]

    @pytest.mark.unit
    def test_no_input_file(self):
        """Test an error is raised when no input file is given."""
        with pytest.raises(NameError):
            InputReader(cl_args=self.argv0)

    @pytest.mark.unit
    def test_nonexistant_input_file(self):
        """Test error is raised when nonexistent path is given."""
        with pytest.raises(FileNotFoundError):
            InputReader(cl_args=self.argv1)

    @pytest.mark.unit
    def test_min_input(self):
        """Test that minimal input is parsed correctly."""
        ir = InputReader(cl_args=self.argv2)
        assert ir.args.input_file[0] == self.argv2[0], "Input file name does not match"
        assert ir.input_path == str(
            pathlib.Path("./" + self.argv2[0]).resolve()
        ), "Path to input file is incorrect."
        # Test defaults
        assert ir.args.plot == 0, "plot is not default 0"
        assert ir.args.quiet is False, "quiet is not default False"
        assert ir.args.verbose == 1, "verbose is not default 1"
        assert ir.args.numpy is False, "numpy is not default False"
        assert (
            os.environ["DESC_BACKEND"] == "jax"
        ), "numpy environment variable incorrect with default argument"
        assert ir.args.version is False, "version is not default False"
        assert (
            len(ir.inputs[0]) == 27
        ), "number of inputs does not match number expected in MIN_INPUT"
        # test equality of arguments

    @pytest.mark.unit
    def test_np_environ(self):
        """Test setting numpy backend via environment variable."""
        argv = self.argv2 + ["--numpy"]
        InputReader(cl_args=argv)
        assert (
            os.environ["DESC_BACKEND"] == "numpy"
        ), "numpy environment variable incorrect on use"

    @pytest.mark.unit
    def test_quiet_verbose(self):
        """Test setting of quiet and verbose options."""
        ir = InputReader(self.argv2)
        assert (
            ir.inputs[0]["verbose"] == 1
        ), "value of inputs['verbose'] incorrect on no arguments"
        argv = self.argv2 + ["-v"]
        ir = InputReader(argv)
        assert (
            ir.inputs[0]["verbose"] == 2
        ), "value of inputs['verbose'] incorrect on verbose argument"
        argv = self.argv2 + ["-vv"]
        ir = InputReader(argv)
        assert (
            ir.inputs[0]["verbose"] == 3
        ), "value of inputs['verbose'] incorrect on double verbose argument"
        argv = self.argv2 + ["-q"]
        ir = InputReader(argv)
        assert (
            ir.inputs[0]["verbose"] == 0
        ), "value of inputs['verbose'] incorrect on quiet argument"

    @pytest.mark.unit
    def test_vacuum_objective_with_iota_yields_current(self):
        """Test that input file with vacuum objective always uses zero current."""
        input_path = ".//tests//inputs//HELIOTRON_vacuum"
        # load an input file with vacuum obj but also an iota profile specified
        with pytest.warns(UserWarning):
            ir = InputReader(input_path)
        # ensure that a current profile instead of an iota profile is used
        assert "iota" not in ir.inputs[0].keys()
        assert "current" in ir.inputs[0].keys()


class MockObject:
    """Example object for saving/loading tests."""

    def __init__(self):
        self._io_attrs_ = ["a", "b", "c"]


@pytest.mark.unit
def test_writer_given_filename(writer_test_file):
    """Test writing to a given file by filename."""
    writer = hdf5Writer(writer_test_file, "w")
    assert writer.check_type(writer.target) is False
    assert writer.check_type(writer.base) is True
    assert writer._close_base_ is True
    writer.close()
    assert writer._close_base_ is False


@pytest.mark.unit
def test_writer_given_file(writer_test_file):
    """Test writing to given file instance."""
    f = h5py.File(writer_test_file, "w")
    writer = hdf5Writer(f, "w")
    assert writer.check_type(writer.target) is True
    assert writer.check_type(writer.base) is True
    assert writer._close_base_ is False
    assert writer._close_base_ is False
    f.close()


@pytest.mark.unit
def test_writer_close_on_delete(writer_test_file):
    """Test that files are closed when writer is deleted."""
    writer = hdf5Writer(writer_test_file, "w")
    with pytest.raises(OSError):
        newwriter = hdf5Writer(writer_test_file, "w")
    del writer
    newwriter = hdf5Writer(writer_test_file, "w")
    del newwriter


@pytest.mark.unit
def test_writer_write_dict(writer_test_file):
    """Test writing dictionary to hdf5 file."""
    thedict = {"1": 1, "2": 2, "3": 3}
    writer = hdf5Writer(writer_test_file, "w")
    writer.write_dict(thedict)
    with pytest.raises(SyntaxError):
        writer.write_dict(thedict, where="not a writable type")
    writer.close()
    f = h5py.File(writer_test_file, "r")
    for key in thedict.keys():
        assert key in f.keys()
        assert f[key][()] == thedict[key]
    f.close()
    reader = hdf5Reader(writer_test_file)

    dict1 = reader.read_dict()
    assert dict1 == thedict
    reader.close()


@pytest.mark.unit
def test_writer_write_list(writer_test_file):
    """Test writing list to hdf5 file."""
    thelist = ["1", 1, "2", 2, "3", 3]
    writer = hdf5Writer(writer_test_file, "w")
    writer.write_list(thelist)
    with pytest.raises(SyntaxError):
        writer.write_list(thelist, where="not a writable type")
    writer.close()
    reader = hdf5Reader(writer_test_file)

    list1 = reader.read_list()
    assert list1 == thelist
    reader.close()


@pytest.mark.unit
def test_writer_write_obj(writer_test_file):
    """Test writing objects to hdf5 file."""
    mo = MockObject()
    writer = hdf5Writer(writer_test_file, "w")
    # writer should throw runtime warning if any save_attrs are undefined
    with pytest.warns(RuntimeWarning):
        writer.write_obj(mo)
    writer.close()
    writer = hdf5Writer(writer_test_file, "w")
    for name in mo._io_attrs_:
        setattr(mo, name, name)
    writer.write_obj(mo)
    groupname = "initial"
    writer.write_obj(mo, where=writer.sub(groupname))
    writer.close()
    f = h5py.File(writer_test_file, "r")
    for key in mo._io_attrs_:
        assert key in f.keys()
    assert groupname in f.keys()
    initial = f[groupname]
    for key in mo._io_attrs_:
        assert key in initial.keys()
    f.close()


@pytest.mark.unit
def test_reader_given_filename(reader_test_file):
    """Test opening a reader with a given filename."""
    reader = hdf5Reader(reader_test_file)
    assert reader.check_type(reader.target) is False
    assert reader.check_type(reader.base) is True
    assert reader._close_base_ is True
    reader.close()
    assert reader._close_base_ is False


@pytest.mark.unit
def test_reader_given_file(reader_test_file):
    """Test opening a reader from a given file instance."""
    f = h5py.File(reader_test_file, "r")
    reader = hdf5Reader(f)
    assert reader.check_type(reader.target) is True
    assert reader.check_type(reader.base) is True
    assert reader._close_base_ is False
    assert reader._close_base_ is False
    f.close()


@pytest.mark.unit
def test_reader_read_obj(reader_test_file):
    """Test reading an object from hdf5 file."""
    mo = MockObject()
    reader = hdf5Reader(reader_test_file)
    reader.read_obj(mo)
    mo._io_attrs_ += "4"
    with pytest.warns(RuntimeWarning):
        reader.read_obj(mo)
    del mo._io_attrs_[-1]
    submo = MockObject()
    reader.read_obj(submo, where=reader.sub("subgroup"))
    for key in mo._io_attrs_:
        assert hasattr(mo, key)
        assert hasattr(submo, key)


@pytest.mark.unit
@pytest.mark.solve
def test_pickle_io(DSHAPE_current, tmpdir_factory):
    """Test saving and loading equilibrium in pickle format."""
    tmpdir = tmpdir_factory.mktemp("desc_inputs")
    tmp_path = tmpdir.join("solovev_test.pkl")
    eqf = load(load_from=str(DSHAPE_current["desc_h5_path"]))
    eqf.save(tmp_path, file_format="pickle")
    peqf = load(tmp_path, file_format="pickle")
    assert equals(eqf, peqf)


@pytest.mark.unit
@pytest.mark.solve
def test_ascii_io(DSHAPE_current, tmpdir_factory):
    """Test saving and loading equilibrium in ASCII format."""
    tmpdir = tmpdir_factory.mktemp("desc_inputs")
    tmp_path = tmpdir.join("solovev_test.txt")
    eq1 = load(load_from=str(DSHAPE_current["desc_h5_path"]))[-1]
    eq1.iota = eq1.get_profile("iota", grid=LinearGrid(30, 16, 0)).to_powerseries(
        sym=True
    )
    write_ascii(tmp_path, eq1)
    with pytest.warns(UserWarning):
        eq2 = read_ascii(tmp_path)
    assert np.allclose(eq1.R_lmn, eq2.R_lmn)
    assert np.allclose(eq1.Z_lmn, eq2.Z_lmn)
    assert np.allclose(eq1.L_lmn, eq2.L_lmn)


@pytest.mark.unit
def test_copy():
    """Test thing.copy() method of IOAble objects."""
    basis = FourierZernikeBasis(2, 2, 2)
    grid = LinearGrid(2, 2, 2)
    transform1 = Transform(grid, basis, method="direct1")
    transform2 = transform1.copy(deepcopy=False)

    assert transform1.basis is transform2.basis
    np.testing.assert_allclose(
        transform1.matrices["direct1"][0][0][0],
        transform2.matrices["direct1"][0][0][0],
        rtol=1e-10,
        atol=1e-10,
    )

    transform3 = transform1.copy(deepcopy=True)
    assert transform1.basis is not transform3.basis
    assert transform1.basis.eq(transform3.basis)
    np.testing.assert_allclose(
        transform1.matrices["direct1"][0][0][0],
        transform3.matrices["direct1"][0][0][0],
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.unit
def test_save_none(tmpdir_factory):
    """Test that None attributes are saved/loaded correctly."""
    tmpdir = tmpdir_factory.mktemp("none_test")
    eq = Equilibrium()
    eq._iota = None
    eq.save(tmpdir + "none_test.h5")
    eq1 = load(tmpdir + "none_test.h5")
    assert eq1.iota is None


@pytest.mark.unit
def test_load_eq_without_current():
    """Test that loading an eq from DESC < 0.6.0 works correctly."""
    desc_no_current_path = ".//tests//inputs//DSHAPE_output_saved_without_current.h5"
    with pytest.warns(RuntimeWarning):
        eq = load(desc_no_current_path)[-1]
    assert eq.current is None
