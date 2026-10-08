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
        0.6020793433333951,
        -0.04897875436345805,
        0.5105856021772198,
        0.46533330254596983,
        0.5930650715827769,
        0.4550156347621205,
        0.401680205241765,
        0.5653813933989743,
        0.5029856165753587,
        0.3757482381662212,
        0.6276906636632577,
        0.4083639152204791,
        0.4706426422216971,
        0.4925230371485189,
        0.44705004540840837,
        0.33613012354948674,
        0.5290351943665962,
        0.47394776659687043,
        0.4043022325669644,
        0.3020163432834793,
        0.44960464855404125,
        0.37537264290874556,
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
