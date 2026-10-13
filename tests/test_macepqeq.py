import math
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
    "model": "MACEPQEQ",
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
        0.6060475029766712,
        -0.052749521495003504,
        0.5070719259431999,
        0.46189143127896787,
        0.5902770650590669,
        0.4514416741820198,
        0.39827530822714796,
        0.5631236741856696,
        0.4994541009246779,
        0.3722971679946786,
        0.6246452626783902,
        0.40486020764571823,
        0.46728016778530806,
        0.4891797819227403,
        0.44344220898861836,
        0.3326136489539917,
        0.5253753245018001,
        0.4703623534446811,
        0.40089359240297934,
        0.29859704726145814,
        0.44608910287054576,
        0.37184892455631025,
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
    """Tests that MACEPQEQ forward pass outputs charges and the dipole."""
    model = torch.load(f=str(macepqeq_model_path), map_location="cpu")
    model.eval()

    # Use only the periodic water configs (skip isolated atoms)
    periodic_configs = [c for c in fitting_configs if c.pbc.any()][:3]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    configs = [data.config_from_atoms(atoms) for atoms in periodic_configs]
    with default_dtype(torch.float32):
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
    assert "dipole" in output, "MACEPQEQ output must contain 'dipole'"
    assert "polarization" in output, "MACEPQEQ output must contain 'polarization'"
    volume = torch.abs(torch.linalg.det(batch["cell"].reshape(-1, 3, 3)))
    torch.testing.assert_close(output["dipole"], output["polarization"] * volume.unsqueeze(1))
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


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
@pytest.mark.parametrize(
    "pqeq, pqeq_arguments",
    [
        (True, {}),
        (False, {}),
        (True, {"environment_dependent_gaussian_width": True}),
    ],
)
def test_macepqeq_optimizer_covers_all_parameters(pqeq, pqeq_arguments):
    from argparse import Namespace

    from mace.tools.scripts_utils import get_optimizer, get_params_options

    with default_dtype(torch.float64):
        torch.manual_seed(0)
        model = MACEPQEQ(pqeq=pqeq, pqeq_arguments=pqeq_arguments, **MODEL_CONFIG)
    args = Namespace(
        lr_params_factors="{}",
        freeze=None,
        lr=1e-2,
        weight_decay=0.0,
        amsgrad=False,
        beta=0.9,
        optimizer="adam",
    )
    param_options = get_params_options(args, model)
    counts = {}
    for group in param_options["params"]:
        group["params"] = list(group["params"])
        for p in group["params"]:
            counts[id(p)] = counts.get(id(p), 0) + 1
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert counts.get(id(p), 0) == 1, name

    optimizer = get_optimizer(args, param_options)
    heads = [n for n, _ in model.named_parameters() if n.startswith("pqeq_e")]
    heads += [n for n, _ in model.named_parameters() if n.startswith("pqeq_sigma")]
    assert heads
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    sum(p.sum() for p in model.parameters()).backward()
    optimizer.step()
    params = dict(model.named_parameters())
    for name in heads:
        assert not torch.equal(before[name], params[name].detach()), name


FINETUNE_CONFIG = {
    **MODEL_CONFIG,
    "num_interactions": 2,
    "interaction_cls_first": interaction_classes["RealAgnosticInteractionBlock"],
}


class PolarMACE(ScaleShiftMACE):
    """Stand-in for a foundation class with its own long-range electrostatics."""


def _two_head_config(base=None):
    return {
        **(base or MODEL_CONFIG),
        "heads": ["pt_head", "Default"],
        "atomic_energies": np.zeros((2, 2)),
        "atomic_inter_shift": [0.0, 0.0],
        "atomic_inter_scale": [1.0, 1.0],
    }


def _head_batch(model, fitting_configs, heads, graph_heads):
    periodic = [c for c in fitting_configs if c.pbc.any()][: len(graph_heads)]
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    configs = []
    for atoms, head in zip(periodic, graph_heads):
        cfg = data.config_from_atoms(atoms)
        cfg.head = head
        configs.append(cfg)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                cfg, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for cfg in configs
        ],
        batch_size=len(configs),
        shuffle=False,
    )
    return next(iter(loader)).to_dict()


def _randomize(model, scale=0.1):
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn_like(p))


def _run_and_capture(monkeypatch, params):
    import mace.cli.run_train as run_train_module

    captured = {}
    original = run_train_module.get_params_options

    def spy(args, model):
        captured["model"] = model
        captured["initial"] = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }
        return original(args, model)

    monkeypatch.setattr(run_train_module, "get_params_options", spy)
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in params.items()]
    )
    mace_run(args)
    return captured


def _last_layer(readout):
    return readout.linear if hasattr(readout, "linear") else readout.linear_2


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
@pytest.mark.parametrize("analytic", [True, False])
def test_pqeq_heads_mask_electrostatics(fitting_configs, analytic):
    with default_dtype(torch.float64):
        args = {"analytic_ewald_derivative": analytic}
        model = MACEPQEQ(pqeq=True, pqeq_arguments=args, pqeq_heads=["Default"], **_two_head_config())
        _randomize(model)
        reference = MACEPQEQ(pqeq=True, pqeq_arguments=args, pqeq_heads=[], **_two_head_config())
        reference.load_state_dict(model.state_dict())
        assert model.pqeq_head_flags == [0.0, 1.0]
        assert reference.pqeq_head_flags == [0.0, 0.0]

        batch = _head_batch(model, fitting_configs, ["pt_head", "Default"], ["pt_head", "Default"])
        out = model(batch, compute_stress=True)
        ref = reference(_head_batch(model, fitting_configs, ["pt_head", "Default"], ["pt_head", "Default"]),
                        compute_stress=True)

    pt_atoms = batch["batch"] == 0
    assert out["energy_pqeq"][0].item() == 0.0
    assert abs(out["energy_pqeq"][1].item()) > 1e-6
    torch.testing.assert_close(out["energy"][0], ref["energy"][0])
    torch.testing.assert_close(out["forces"][pt_atoms], ref["forces"][pt_atoms])
    torch.testing.assert_close(out["stress"][0], ref["stress"][0])
    torch.testing.assert_close(out["energy"][1] - ref["energy"][1], out["energy_pqeq"][1])
    assert not torch.allclose(out["forces"][~pt_atoms], ref["forces"][~pt_atoms])


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_zero_pqeq_readouts_start_at_priors(fitting_configs):
    with default_dtype(torch.float64):
        model = MACEPQEQ(pqeq=True, pqeq_arguments={"environment_dependent_gaussian_width": True},
                         **MODEL_CONFIG)
        _randomize(model)
        model.zero_pqeq_readouts()
        assert set(model.pqeq_readout_names()) == {
            "pqeq_e1_readouts", "pqeq_e2_readouts", "pqeq_e2d_readouts", "pqeq_sigma_readouts"
        }
        for name in model.pqeq_readout_names():
            for readout in getattr(model, name):
                assert torch.count_nonzero(_last_layer(readout).weight) == 0
        out = model(_head_batch(model, fitting_configs, ["Default"], ["Default"]))
    chi0, J0 = model.pqeq_model.estimate_species_chi0_J0(
        model.pqeq_model.get_species_from_atomic_numbers(model.atomic_numbers)
    )
    assert out["E1"].abs().max() > 0
    assert torch.unique(out["E1"]).numel() <= 2
    assert torch.unique(out["E2"]).numel() <= 2


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_remove_pt_head_keeps_pqeq(fitting_configs):
    from mace.tools.scripts_utils import extract_config_mace_model, remove_pt_head

    with default_dtype(torch.float64):
        model = MACEPQEQ(pqeq=True, pqeq_arguments={"gaussian_width_tanh_scale": 0.4},
                         **_two_head_config())
        _randomize(model)
        config = extract_config_mace_model(model)
        assert config["pqeq"] is True
        assert config["pqeq_arguments"]["gaussian_width_tanh_scale"] == 0.4
        assert config["pqeq_heads"] == ["pt_head", "Default"]

        single = remove_pt_head(model, "Default")
        assert isinstance(single, MACEPQEQ)
        assert single.heads == ["Default"]
        assert single.pqeq_config["gaussian_width_tanh_scale"] == 0.4
        out = model(_head_batch(model, fitting_configs, ["pt_head", "Default"], ["Default"] * 2))
        got = single(_head_batch(single, fitting_configs, ["Default"], ["Default"] * 2))
    for key in ("energy", "forces", "charges", "E1", "E2", "E_d2"):
        torch.testing.assert_close(got[key], out[key], msg=key)


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_load_pqeq_foundation_into_multihead(fitting_configs):
    from mace.tools.finetuning_utils import load_foundations_elements

    with default_dtype(torch.float64):
        foundation = MACEPQEQ(pqeq=True, **FINETUNE_CONFIG)
        _randomize(foundation)
        model = MACEPQEQ(pqeq=True, **_two_head_config(FINETUNE_CONFIG))
        model = load_foundations_elements(
            model,
            foundation,
            utils.AtomicNumberTable([1, 8]),
            load_readout=True,
            max_L=1,
            default_dtype=torch.float64,
        )
        ref = foundation(_head_batch(foundation, fitting_configs, ["Default"], ["Default"] * 2))
        for head in ("pt_head", "Default"):
            out = model(_head_batch(model, fitting_configs, ["pt_head", "Default"], [head] * 2))
            for key in ("energy", "forces", "charges", "E1", "E2", "E_d2"):
                torch.testing.assert_close(out[key], ref[key], msg=f"{head} {key}")


def _finetune_params(tmp_path, foundation_path):
    params = _mace_params.copy()
    params.update(
        checkpoints_dir=str(tmp_path),
        model_dir=str(tmp_path),
        train_file=tmp_path / "fit.xyz",
        foundation_model=str(foundation_path),
        default_dtype="float64",
        multiheads_finetuning=False,
        max_num_epochs=2,
        E0s="average",
    )
    for key in ("hidden_irreps", "r_max"):
        params.pop(key)
    return params


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_finetune_macepqeq_from_mace_foundation(tmp_path, fitting_configs, monkeypatch):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    with default_dtype(torch.float64):
        foundation = ScaleShiftMACE(**FINETUNE_CONFIG)
        _randomize(foundation)
    foundation_path = tmp_path / "foundation.model"
    torch.save(foundation, foundation_path)
    params = _finetune_params(tmp_path, foundation_path)
    params["pqeq"] = None
    captured = _run_and_capture(monkeypatch, params)

    initial = captured["initial"]
    model = captured["model"]
    assert model.pqeq and hasattr(model, "pqeq_e2d_readouts")
    for name, p in foundation.named_parameters():
        if name.startswith("interactions.") or name.startswith("products."):
            torch.testing.assert_close(initial[name], p.detach(), msg=name)
    for name in model.pqeq_readout_names():
        for i, readout in enumerate(getattr(model, name)):
            last = "linear" if hasattr(readout, "linear") else "linear_2"
            assert torch.count_nonzero(initial[f"{name}.{i}.{last}.weight"]) == 0

    trained = torch.load(tmp_path / "MACE.model", map_location="cpu")
    assert torch.count_nonzero(_last_layer(trained.pqeq_e1_readouts[0]).weight) > 0


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_finetune_macepqeq_from_macepqeq_foundation(tmp_path, fitting_configs, monkeypatch):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    with default_dtype(torch.float64):
        foundation = MACEPQEQ(pqeq=True, pqeq_arguments={"gaussian_width_tanh_scale": 0.4},
                              **FINETUNE_CONFIG)
        _randomize(foundation)
    foundation_path = tmp_path / "foundation.model"
    torch.save(foundation, foundation_path)

    captured = _run_and_capture(monkeypatch, _finetune_params(tmp_path, foundation_path))
    model = captured["model"]
    initial = captured["initial"]
    assert model.pqeq
    assert model.pqeq_config["gaussian_width_tanh_scale"] == 0.4
    assert model.pqeq_heads == list(model.heads)
    for name, p in foundation.named_parameters():
        if name.startswith("pqeq_"):
            torch.testing.assert_close(initial[name], p.detach(), msg=name)


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_finetune_macepqeq_rejects_long_range_foundation(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    with default_dtype(torch.float64):
        foundation = PolarMACE(**FINETUNE_CONFIG)
    foundation_path = tmp_path / "polar.model"
    torch.save(foundation, foundation_path)
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}"
         for k, v in _finetune_params(tmp_path, foundation_path).items()]
    )
    with pytest.raises(ValueError, match="PolarMACE"):
        mace_run(args)


def test_pqeq_heads_selection():
    from argparse import Namespace

    from mace.tools.model_script_utils import _pqeq_heads

    heads = ["pt_head", "Default"]
    assert _pqeq_heads(Namespace(pqeq_heads=None), heads) == heads
    assert _pqeq_heads(Namespace(pqeq_heads=None), heads, {}) == ["Default"]
    pqeq_foundation = {"pqeq": True, "pqeq_arguments": {"n_shells": 1}}
    assert _pqeq_heads(Namespace(pqeq_heads=None), heads, pqeq_foundation) == heads
    assert _pqeq_heads(Namespace(pqeq_heads="pt_head, Default"), heads, {}) == heads
    with pytest.raises(ValueError, match="unknown heads"):
        _pqeq_heads(Namespace(pqeq_heads="water"), heads)


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "softplus", "relu"])
def test_sigma_readout_starts_at_zero_and_learns(fitting_configs, activation):
    with default_dtype(torch.float64):
        model = MACEPQEQ(
            pqeq=True,
            pqeq_arguments={
                "environment_dependent_gaussian_width": True,
                "gaussian_width_activation": activation,
            },
            **MODEL_CONFIG,
        )
        last = [_last_layer(readout) for readout in model.pqeq_sigma_readouts]
        for layer in last:
            assert torch.count_nonzero(layer.weight) == 0
        out = model(_head_batch(model, fitting_configs, ["Default"], ["Default"]), training=True)
        out["energy"].sum().backward()
    assert sum(layer.weight.grad.abs().sum() for layer in last) > 0


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
@pytest.mark.parametrize("env", [False, True])
def test_uniform_gaussian_width_reference_starts_every_species_equal(fitting_configs, env):
    with default_dtype(torch.float64):
        model = MACEPQEQ(
            pqeq=True,
            pqeq_arguments={
                "environment_dependent_gaussian_width": env,
                "gaussian_width_reference": 0.5,
                "gaussian_width_activation": "sigmoid",
                "gaussian_width_start": 0.5,
                "gaussian_width_tanh_scale": 1.0,
            },
            **MODEL_CONFIG,
        )
        assert model.pqeq_config["gaussian_width_reference"] == 0.5
        out = model(_head_batch(model, fitting_configs, ["Default"], ["Default"]), training=True)
    widths = out["gaussian_width"]
    torch.testing.assert_close(widths, torch.full_like(widths, math.sqrt(2.0) * 0.75))


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_charge_prior_loss_adds_weighted_prior(fitting_configs):
    from bacenet.models.charge_prior import charge_prior_loss
    from mace.modules import WeightedEnergyForcesLoss
    from mace.modules.extensions import ChargePriorLoss

    with default_dtype(torch.float64):
        model = MACEPQEQ(pqeq=True, **MODEL_CONFIG)
        _randomize(model)
        periodic = [c for c in fitting_configs if c.pbc.any()][:3]
        configs = []
        for atoms in periodic:
            cfg = data.config_from_atoms(
                atoms, key_specification=data.KeySpecification(
                    info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"}
                )
            )
            configs.append(cfg)
        z_table = utils.AtomicNumberTable([1, 8])
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[data.AtomicData.from_config(c, z_table=z_table, cutoff=float(model.r_max)) for c in configs],
            batch_size=3,
            shuffle=False,
        )
        batch = next(iter(loader))
        pred = model(batch.to_dict(), training=True)
        base = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
        loss_fn = ChargePriorLoss(base, [1, 8], weight=2.0, decay_epochs=10)
        expected_prior = charge_prior_loss(
            pred["charges"].reshape(-1),
            torch.tensor([1, 8])[batch["node_attrs"].argmax(dim=-1)],
            batch["batch"],
            batch["total_charge"].reshape(-1),
            scale=0.6,
            temperature=0.1,
            nonmain_penalty=4.0,
        )
        total = loss_fn(ref=batch, pred=pred)
        torch.testing.assert_close(total, base(ref=batch, pred=pred) + 2.0 * expected_prior)
        assert expected_prior.item() > 0.0

        loss_fn.set_epoch(10)
        weight, temperature = loss_fn.current_weight_and_temperature()
        assert weight == pytest.approx(0.2)
        assert temperature == pytest.approx(0.01)


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
@pytest.mark.parametrize("source", ["command_line", "pqeq_arguments"])
def test_run_train_with_charge_prior(tmp_path, fitting_configs, monkeypatch, source):
    from mace.modules import extensions

    calls = {"n": 0, "epochs": set()}
    original = extensions.ChargePriorLoss.prior

    def counting_prior(self, ref, pred):
        calls["n"] += 1
        calls["epochs"].add(self.epoch)
        return original(self, ref, pred)

    monkeypatch.setattr(extensions.ChargePriorLoss, "prior", counting_prior)
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    params = _mace_params.copy()
    params.update(
        checkpoints_dir=str(tmp_path),
        model_dir=str(tmp_path),
        train_file=tmp_path / "fit.xyz",
        max_num_epochs=2,
    )
    if source == "command_line":
        params["charge_prior_weight"] = 1.0
    else:
        params["pqeq_arguments"] = "{'charge_prior_weight': 1.0, 'charge_prior_scale': 0.7}"
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in params.items()]
    )
    mace_run(args)
    assert calls["n"] > 0
    assert calls["epochs"] == {0, 1}
    assert (tmp_path / "MACE.model").exists()



def test_resolve_charge_prior_settings():
    from argparse import Namespace

    from mace.cli.run_train import CHARGE_PRIOR_KEYS, resolve_charge_prior_settings
    from mace.tools.arg_parser import build_default_arg_parser

    parser = build_default_arg_parser()
    defaults = {key: parser.get_default(key) for key in CHARGE_PRIOR_KEYS}
    model = Namespace(pqeq_config={"charge_prior_weight": 2.0, "charge_prior_scale": 0.7})

    settings = resolve_charge_prior_settings(Namespace(**defaults), model)
    assert settings["charge_prior_weight"] == 2.0
    assert settings["charge_prior_scale"] == 0.7
    assert settings["charge_prior_temperature"] == defaults["charge_prior_temperature"]

    cli = Namespace(**{**defaults, "charge_prior_weight": 3.0})
    assert resolve_charge_prior_settings(cli, Namespace(pqeq_config={}))["charge_prior_weight"] == 3.0
    with pytest.raises(ValueError, match="charge_prior_weight"):
        resolve_charge_prior_settings(cli, model)
    same = Namespace(**{**defaults, "charge_prior_weight": 2.0})
    assert resolve_charge_prior_settings(same, model)["charge_prior_weight"] == 2.0
    assert resolve_charge_prior_settings(Namespace(**defaults), Namespace())["charge_prior_weight"] == 0.0


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_run_train_charge_prior_needs_zero_q0(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    params = _mace_params.copy()
    params.update(
        checkpoints_dir=str(tmp_path),
        model_dir=str(tmp_path),
        train_file=tmp_path / "fit.xyz",
        max_num_epochs=1,
        pqeq_arguments="{'charge_prior_weight': 1.0, 'oxidation_states': {'O': -2.0}}",
    )
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in params.items()]
    )
    with pytest.raises(ValueError, match="zero reference charges"):
        mace_run(args)


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("shell_vector", [False, True])
@pytest.mark.parametrize("stiffness,activation", [("softplus", "sigmoid"), ("polarizability", "sigmoid"),
                                                 ("polarizability", "softplus")])
def test_charge_response_gradients_match_finite_differences(fitting_configs, analytic, shell_vector, stiffness,
                                                            activation):
    with default_dtype(torch.float64):
        model = MACEPQEQ(pqeq=True, pqeq_arguments={"analytic_ewald_derivative": analytic,
                                                    "exact_solver": True,
                                                    "shell_vector_term": shell_vector,
                                                    "shell_stiffness_mode": stiffness,
                                                    "polarizability_activation": activation},
                         **FINETUNE_CONFIG)
        _randomize(model)
        atoms = [c for c in fitting_configs if c.pbc.any()][0]
        cfg = data.config_from_atoms(
            atoms, key_specification=data.KeySpecification(
                info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"}
            )
        )
        z_table = utils.AtomicNumberTable([1, 8])
        batch = next(iter(torch_geometric.dataloader.DataLoader(
            dataset=[data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))],
            batch_size=1,
        ))).to_dict()
        observables = {
            "forces": lambda o: o["forces"].square().sum(),
            "stress": lambda o: o["stress"].square().sum(),
            "energy": lambda o: o["energy"].sum(),
            "charges": lambda o: (o["charges"] * torch.arange(o["charges"].numel())).sum(),
            "shells": lambda o: o["shell_disp"].square().sum(),
        }

        def run():
            return model({k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()},
                         training=True, compute_stress=True)

        names = ["pqeq_e1_readouts", "pqeq_e2_readouts", "pqeq_e2d_readouts"]
        if shell_vector:
            names.append("pqeq_ed1_readouts")
        for name in names:
            param = _last_layer(getattr(model, name)[-1]).weight
            direction = torch.randn(param.shape, generator=torch.Generator().manual_seed(1))
            out = run()
            grads = {k: (torch.autograd.grad(f(out), param, retain_graph=True)[0] * direction).sum().item()
                     for k, f in observables.items()}
            h = 1e-4
            with torch.no_grad():
                param.add_(h * direction)
            plus = run()
            with torch.no_grad():
                param.sub_(2 * h * direction)
            minus = run()
            with torch.no_grad():
                param.add_(h * direction)
            for key, f in observables.items():
                fd = (f(plus) - f(minus)).item() / (2 * h)
                assert grads[key] == pytest.approx(fd, rel=1e-4, abs=1e-9), (name, key)


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_run_train_charge_prior_needs_exact_solver(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    params = _mace_params.copy()
    params.update(
        checkpoints_dir=str(tmp_path),
        model_dir=str(tmp_path),
        train_file=tmp_path / "fit.xyz",
        max_num_epochs=1,
        pqeq_arguments="{'charge_prior_weight': 1.0, 'exact_solver': False}",
    )
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in params.items()]
    )
    with pytest.raises(ValueError, match="exact_solver"):
        mace_run(args)


def _single_batch(model, atoms):
    cfg = data.config_from_atoms(atoms)
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    return next(iter(torch_geometric.dataloader.DataLoader(
        dataset=[data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max))],
        batch_size=1,
    ))).to_dict()


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_polarizability_stiffness_starts_at_the_free_atom_value(fitting_configs):
    from bacenet.functions import species_priors
    from bacenet.models.compute_torch import CONV_FACT

    with default_dtype(torch.float64):
        model = MACEPQEQ(pqeq=True, pqeq_arguments={"shell_stiffness_mode": "polarizability",
                                                    "wannier_centriod_mode": True},
                         **FINETUNE_CONFIG)
        assert model.pqeq_config["shell_stiffness_mode"] == "polarizability"
        assert model.pqeq_config["polarizability_scale_max"] == 2.0
        assert model.pqeq_config["polarizability_activation"] == "sigmoid"
        model.zero_pqeq_readouts()
        atoms = [c for c in fitting_configs if c.pbc.any()][0]
        out = model(_single_batch(model, atoms), compute_stress=True)
    numbers = atoms.get_atomic_numbers().tolist()
    zin = model.pqeq_model.estimate_species_nelectrons(numbers, 1).double()
    alpha = species_priors.estimate_species_polarizability(numbers).double()
    expected = CONV_FACT * zin * zin / alpha
    torch.testing.assert_close(out["E_d2"].reshape(len(numbers), 3), expected[:, None].expand(-1, 3))


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_shell_vector_term_starts_at_zero(fitting_configs):
    with default_dtype(torch.float64):
        torch.manual_seed(0)
        on = MACEPQEQ(pqeq=True, pqeq_arguments={"shell_vector_term": True}, **FINETUNE_CONFIG)
        torch.manual_seed(0)
        off = MACEPQEQ(pqeq=True, **FINETUNE_CONFIG)
        atoms = [c for c in fitting_configs if c.pbc.any()][0]
        a = on(_single_batch(on, atoms), compute_stress=True)
        b = off(_single_batch(off, atoms), compute_stress=True)
        assert torch.count_nonzero(a["E_d1"]) == 0
        assert b["E_d1"] is None
        for key in ("energy", "forces", "stress", "charges", "shell_disp"):
            torch.testing.assert_close(a[key], b[key])


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
def test_shell_vector_term_needs_vector_features():
    config = {**FINETUNE_CONFIG, "hidden_irreps": o3.Irreps("32x0e"), "max_ell": 1}
    with pytest.raises(ValueError, match="l=1"):
        MACEPQEQ(pqeq=True, pqeq_arguments={"shell_vector_term": True}, **config)
    with pytest.raises(ValueError, match="pqeq"):
        MACEPQEQ(pqeq=False, pqeq_arguments={"shell_vector_term": True}, **FINETUNE_CONFIG)


@pytest.mark.skipif(not BACENET_AVAILABLE, reason="bacenet library is not available")
@pytest.mark.parametrize("analytic", [True, False])
def test_shell_vector_term_is_equivariant_and_conservative(fitting_configs, analytic):
    with default_dtype(torch.float64):
        model = MACEPQEQ(pqeq=True, pqeq_arguments={"analytic_ewald_derivative": analytic,
                                                    "shell_vector_term": True},
                         **FINETUNE_CONFIG)
        _randomize(model)
        atoms = [c for c in fitting_configs if c.pbc.any()][0].copy()
        torch.manual_seed(3)
        R = o3.rand_matrix().double()
        rotated = atoms.copy()
        rotated.set_cell(atoms.cell.array @ R.numpy().T, scale_atoms=False)
        rotated.positions = atoms.positions @ R.numpy().T
        a = model(_single_batch(model, atoms), compute_stress=True)
        b = model(_single_batch(model, rotated), compute_stress=True)
        assert a["E_d1"].abs().max() > 1e-5
        torch.testing.assert_close(b["energy"], a["energy"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(b["E_d1"], a["E_d1"] @ R.T, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(b["shell_disp"].reshape(-1, 3), a["shell_disp"].reshape(-1, 3) @ R.T,
                                   rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(b["forces"], a["forces"] @ R.T, rtol=1e-6, atol=1e-6)

        h = 1e-5
        for atom, axis in ((0, 0), (1, 2), (2, 1)):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[atom, axis] += h
            minus.positions[atom, axis] -= h
            fd = -(model(_single_batch(model, plus))["energy"]
                   - model(_single_batch(model, minus))["energy"]).item() / (2 * h)
            assert a["forces"][atom, axis].item() == pytest.approx(fd, rel=1e-5, abs=1e-7)
