"""The encoder must activate every hidden layer without reinterpreting old weights."""
import copy

import pytest
import torch

from config import CONFIG
from uav_marl.algorithms.masac import (
    HybridMASAC, HybridMASACActor, build_configured_mlp,
    load_masac_checkpoint, save_masac_checkpoint,
)
from uav_marl.algorithms.matd3 import (
    HybridMATD3, HybridMATD3Actor, load_matd3_checkpoint, save_matd3_checkpoint,
)
from uav_marl.envs.uav_search import UAVSearchEnv


@pytest.mark.parametrize("actor_type", [HybridMASACActor, HybridMATD3Actor])
@pytest.mark.parametrize("hidden_dims", [(8,), (8, 8)])
def test_final_hidden_layer_applies_leaky_activation(actor_type, hidden_dims, monkeypatch):
    # Removing the last activation used to leave the final hidden feature linear.
    monkeypatch.setitem(CONFIG, "model_activation", "leaky_relu")
    monkeypatch.setitem(CONFIG, "model_layer_norm", False)
    actor = actor_type(333, 4, 7, hidden_dims)
    linears = [module for module in actor.encoder if isinstance(module, torch.nn.Linear)]
    with torch.no_grad():
        for linear in linears:
            linear.weight.zero_()
            linear.bias.zero_()
        linears[-1].bias.fill_(-2.)
    features = actor.encoder(torch.zeros(1, 333))
    torch.testing.assert_close(features, torch.full((1, 8), -0.02))


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
@pytest.mark.parametrize("activation,layer_norm", [("leaky_relu", False), ("prelu", True)])
def test_legacy_checkpoint_keeps_actions_and_optimizer_state(
    tmp_path, monkeypatch, algorithm, activation, layer_norm,
):
    # A checkpoint without the version marker predates the last activation.
    monkeypatch.setitem(CONFIG, "model_activation", activation)
    monkeypatch.setitem(CONFIG, "model_layer_norm", layer_norm)
    monkeypatch.setitem(CONFIG, algorithm + "_hidden_dims", (8, 8))
    env = UAVSearchEnv(backend_name="simple")
    try:
        cls, save, load = (
            (HybridMASAC, save_masac_checkpoint, load_masac_checkpoint)
            if algorithm == "masac"
            else (HybridMATD3, save_matd3_checkpoint, load_matd3_checkpoint)
        )
        trainer = cls(env, device="cpu")
        # Build the historical architecture explicitly, independent of the fixed helper.
        trainer.actor.encoder = build_configured_mlp(trainer.observation_dim, (8,), 8)
        trainer.actor_optimizer = torch.optim.Adam(
            trainer.actor.parameters(), lr=float(CONFIG[algorithm + "_actor_lr"]),
        )
        if algorithm == "matd3":
            trainer.target_actor = copy.deepcopy(trainer.actor)
        obs = torch.linspace(-.3, .7, trainer.observation_dim).repeat(2, 1)
        outputs = trainer.actor(obs)
        sum(value.sum() for value in outputs).backward()
        trainer.actor_optimizer.step()
        expected = tuple(value.detach().clone() for value in trainer.actor(obs))
        path = tmp_path / "legacy.pt"
        save(path, trainer, include_replay=False)
        payload = torch.load(path, weights_only=True)
        payload["trainer_state"].pop("actor_encoder_version", None)
        torch.save(payload, path)
        restored = load(path, env, device="cpu", load_replay=False, restore_torch_rng=False)["trainer"]
        for actual, wanted in zip(restored.actor(obs), expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        if algorithm == "matd3":
            for actual, wanted in zip(restored.target_actor(obs), trainer.target_actor(obs)):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        original_state = trainer.actor_optimizer.state_dict()
        loaded_state = restored.actor_optimizer.state_dict()
        assert original_state["param_groups"] == loaded_state["param_groups"]
        for key, values in original_state["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(loaded_state["state"][key][name], value, rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_new_checkpoint_records_encoder_version_and_roundtrips(tmp_path, monkeypatch, algorithm):
    monkeypatch.setitem(CONFIG, algorithm + "_hidden_dims", (8, 8))
    env = UAVSearchEnv(backend_name="simple")
    try:
        cls, save, load = (
            (HybridMASAC, save_masac_checkpoint, load_masac_checkpoint)
            if algorithm == "masac"
            else (HybridMATD3, save_matd3_checkpoint, load_matd3_checkpoint)
        )
        trainer = cls(env, device="cpu")
        path = tmp_path / "new.pt"
        save(path, trainer, include_replay=False)
        payload = torch.load(path, weights_only=True)
        assert payload["trainer_state"].get("actor_encoder_version") == 2
        restored = load(path, env, device="cpu", load_replay=False, restore_torch_rng=False)["trainer"]
        obs = torch.linspace(-.3, .7, trainer.observation_dim).repeat(2, 1)
        for actual, wanted in zip(restored.actor(obs), trainer.actor(obs)):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_direct_state_load_rejects_ambiguous_legacy_encoder(monkeypatch, algorithm):
    monkeypatch.setitem(CONFIG, algorithm + "_hidden_dims", (8, 8))
    env = UAVSearchEnv(backend_name="simple")
    try:
        cls = HybridMASAC if algorithm == "masac" else HybridMATD3
        trainer = cls(env, device="cpu")
        state = trainer.state_dict()
        state.pop("actor_encoder_version", None)
        with pytest.raises(ValueError, match="actor encoder version mismatch"):
            trainer.load_state_dict(state)
    finally:
        env.close()
