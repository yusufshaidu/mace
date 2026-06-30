import importlib.util
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from e3nn import o3

from mace import data
from mace.calculators import MACECalculator
from mace.cli.eval_configs import run as mace_eval_configs_run
from mace.cli.run_train import run as mace_run
from mace.modules import interaction_classes
from mace.modules.extensions import MACEPQEQ
from mace.modules.models import ScaleShiftMACE
from mace.tools import torch_geometric, utils
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.torch_tools import default_dtype

BACENET_AVAILABLE = bool(importlib.util.find_spec("bacenet") is not None)
CUET_AVAILABLE = bool(importlib.util.find_spec("cuequivariance") is not None)
CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(name="fitting_configs")
def fixture_fitting_configs():
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    fit_configs = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3),
    ]
    fit_configs[0].info["REF_energy"] = 0.0
    fit_configs[0].info["config_type"] = "IsolatedAtom"
    fit_configs[1].info["REF_energy"] = 0.0
    fit_configs[1].info["config_type"] = "IsolatedAtom"

    np.random.seed(5)
    for _ in range(20):
        c = water.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        c.info["REF_energy"] = np.random.normal(0.1)
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        c.info["REF_stress"] = np.random.normal(0.1, size=6)
        fit_configs.append(c)

    return fit_configs


_mace_params = {
    "name": "MACE",
    "valid_fraction": 0.05,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "stress_weight": 1.0,
    "model": "s",
    "hidden_irreps": "128x0e",
    "r_max": 3.5,
    "batch_size": 5,
    "max_num_epochs": 10,
    "swa": None,
    "start_swa": 5,
    "ema": None,
    "ema_decay": 0.99,
    "amsgrad": None,
    "restart_latest": None,
    "device": "cpu",
    "seed": 5,
    "loss": "stress",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "eval_interval": 2,
    "use_reduced_cg": False,
}


MODEL_CONFIG = dict(
    r_max=5,
    num_bessel=8,
    num_polynomial_cutoff=6,
    max_ell=2,
    interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
    interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
    num_interactions=5,
    num_elements=2,
    hidden_irreps=o3.Irreps("32x0e + 32x1o"),
    MLP_irreps=o3.Irreps("16x0e"),
    gate=torch.nn.functional.silu,
    atomic_energies=np.zeros(2),
    avg_num_neighbors=8,
    atomic_numbers=[1, 8],
    correlation=3,
    radial_type="bessel",
    atomic_inter_shift=0.0,
    atomic_inter_scale=1.0,
)


@pytest.fixture(name="mace_model_path")
def mace_model_path_fixture(tmp_path: Path) -> Path:
    """Create and save a standard ScaleShiftMACE model."""
    with default_dtype(torch.float32):
        model = ScaleShiftMACE(**MODEL_CONFIG)
        path = tmp_path / "mace.model"
        torch.save(model, path)
    return path


@pytest.fixture(name="macepqeq_model_path")
def macepqeq_model_path_fixture(tmp_path: Path) -> Path:
    """Create and save a MACEPQEQ model."""
    with default_dtype(torch.float32):
        model = MACEPQEQ(**MODEL_CONFIG)
        path = tmp_path / "macepqeq.model"
        torch.save(model, path)
    return path


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_run_train(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in mace_params.items()]
    )

    mace_run(args)

    calc = MACECalculator(model_paths=tmp_path / "MACE.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    ref_Es = [
        0.004919160731848143,
        0.5906680240792959,
        0.47887544882572264,
        0.4176002467254094,
        0.5606673227439406,
        0.40181714730443363,
        0.3367534132795259,
        0.27118917957971056,
        0.47967529915910134,
        0.32077479180773283,
        1.2865402405977537,
        0.3472478715875782,
        0.427734507004752,
        0.8092185237225293,
        0.38348242384362774,
        0.14448973657513398,
        0.5650118900854595,
        0.429029669763921,
        0.4837945154901776,
        0.2244894146891574,
        0.3667896493444026,
        0.23811703879534651,
    ]
    assert np.allclose(Es, ref_Es)


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_run_train_with_mp(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["foundation_model"] = "small"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["multiheads_finetuning"] = False
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in mace_params.items()]
    )

    mace_run(args)

    calc = MACECalculator(model_paths=tmp_path / "MACE.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)


@pytest.mark.skipif(
    not (BACENET_AVAILABLE and CUET_AVAILABLE and CUDA_AVAILABLE),
    reason="Testing MACEPQEQ cueq training requires bacenet, cuequivariance, and CUDA",
)
def test_run_train_macepqeq_cueq(tmp_path, fitting_configs):
    import os

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    mace_params["device"] = "cuda"
    mace_params["enable_cueq"] = True
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in mace_params.items()]
    )
    torch.manual_seed(5)
    torch.use_deterministic_algorithms(True)

    mace_run(args)

    calc = MACECalculator(model_paths=tmp_path / "MACE.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_macepqeq_forward_outputs_charges(macepqeq_model_path: Path, fitting_configs):
    """Tests that MACEPQEQ forward pass outputs charges and dipoles."""
    model = torch.load(f=str(macepqeq_model_path), map_location="cpu")
    model.eval()

    # Use only the periodic water configs (skip isolated atoms)
    periodic_configs = [c for c in fitting_configs if c.pbc.any()][:3]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    configs = [data.config_from_atoms(atoms) for atoms in periodic_configs]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))
            for cfg in configs
        ],
        batch_size=3,
        shuffle=False,
    )

    for batch in data_loader:
        output = model(batch.to_dict(), compute_stress=True)

    assert "charges" in output, "MACEPQEQ output must contain 'charges'"
    assert "dipoles" in output, "MACEPQEQ output must contain 'dipoles'"
    assert "energy" in output
    assert "forces" in output
    assert output["charges"].shape[0] == sum(len(c) for c in periodic_configs)


def test_run_eval_fail_with_wrong_model(
    tmp_path: Path, mace_model_path: Path, fitting_configs
):
    """BEC computation should fail with any non-MACELES model, including ScaleShiftMACE."""
    import argparse

    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    args = argparse.Namespace(
        model=str(mace_model_path),
        configs=str(tmp_path / "fit.xyz"),
        output=str(tmp_path / "output.xyz"),
        device="cpu",
        default_dtype="float32",
        batch_size=1,
        compute_stress=False,
        compute_bec=True,
        enable_cueq=False,
        return_contributions=False,
        return_descriptors=False,
        return_node_energies=False,
        info_prefix="MACE_",
        head=None,
    )

    with pytest.raises(ValueError, match="BEC can only be computed with MACELES model."):
        mace_eval_configs_run(args)


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_run_eval_macepqeq_basic(
    tmp_path: Path, macepqeq_model_path: Path, fitting_configs
):
    """Tests running eval_configs with a MACEPQEQ model (energy/forces/stress, no BEC)."""
    import argparse

    output_path = tmp_path / "output.xyz"
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    args = argparse.Namespace(
        model=str(macepqeq_model_path),
        configs=str(tmp_path / "fit.xyz"),
        output=str(output_path),
        device="cpu",
        default_dtype="float32",
        batch_size=1,
        compute_stress=True,
        compute_bec=False,
        enable_cueq=False,
        return_contributions=False,
        return_descriptors=False,
        return_node_energies=False,
        info_prefix="MACE_",
        head=None,
    )
    mace_eval_configs_run(args)

    assert output_path.exists()
    output_atoms = ase.io.read(str(output_path), index=":")
    assert len(output_atoms) == len(fitting_configs)
    for at in output_atoms:
        assert isinstance(at, Atoms)
        assert "MACE_BEC" not in at.arrays
        assert "MACE_energy" in at.info
        assert "MACE_stress" in at.info
        assert "MACE_forces" in at.arrays
