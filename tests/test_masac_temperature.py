"""Regressions for masked communication entropy in the corner-collapse run."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist


@pytest.fixture
def process_group(tmp_path):
    dist.init_process_group("gloo", init_method=(tmp_path / "gloo").as_uri(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()

from config import CONFIG
from uav_marl.training.gpu import _trainer, _ddp_update_masac, _MASACDDPActorForward


@pytest.mark.parametrize('ddp_path', [False, True])
def test_no_communication_choice_preserves_temperature_and_adam_state(monkeypatch, ddp_path, process_group):
    monkeypatch.setitem(CONFIG, 'masac_hidden_dims', (16, 16))
    monkeypatch.setitem(CONFIG, 'masac_replay_capacity', 16)
    trainer = _trainer(Path.cwd(), 'masac', 'cpu', 44)
    replay = trainer.make_replay_buffer(seed=44)
    obs = np.full((trainer.num_agents, trainer.observation_dim), 0.1, np.float32)
    state = np.full(trainer.state_dim, 0.1, np.float32)
    action = np.zeros((trainer.num_agents, 4), np.float32)
    dest = np.zeros(trainer.num_agents, np.int64)
    mask = np.zeros((trainer.num_agents, trainer.discrete_dim), np.float32)
    mask[:, 0] = 1
    for _ in range(4):
        replay.add(obs, state, action, dest, mask, 0., obs, state, mask, True, False)
    # A preceding low-entropy batch leaves real Adam momentum behind.
    optimizer = trainer.alpha_discrete_optimizer
    optimizer.zero_grad()
    (-trainer.log_alpha_discrete).backward()
    optimizer.step()
    before = trainer.log_alpha_discrete.detach().clone()
    step_before = optimizer.state[trainer.log_alpha_discrete]['step'].clone()
    if ddp_path:
        forward = _MASACDDPActorForward(trainer.actor, trainer._project_namespace)
        _ddp_update_masac(trainer, replay, 4, forward, trainer.critic_1, trainer.critic_2)
    else:
        trainer.update(replay, batch_size=4)
    assert torch.equal(trainer.log_alpha_discrete.detach(), before)
    assert torch.equal(optimizer.state[trainer.log_alpha_discrete]['step'], step_before)


def _distributed_temperature_worker(rank, init_file, output_dir):
    from uav_marl.algorithms.masac import HybridMASAC
    dist.init_process_group('gloo', init_method=Path(init_file).as_uri(), rank=rank, world_size=2)
    try:
        trainer = object.__new__(HybridMASAC)
        trainer.log_alpha_discrete = torch.tensor(0., requires_grad=True)
        trainer.alpha_discrete_optimizer = torch.optim.Adam([trainer.log_alpha_discrete], lr=0.01)
        counts = torch.tensor([[1. if rank == 0 else 2.]])
        loss = trainer.log_alpha_discrete * (0. if rank == 0 else -1.)
        trainer._update_discrete_temperature(loss, counts, distributed=True)
        first = trainer.log_alpha_discrete.detach().clone()
        trainer._update_discrete_temperature(trainer.log_alpha_discrete * 0., torch.ones_like(counts), distributed=True)
        torch.save({'first': first, 'second': trainer.log_alpha_discrete.detach().clone()}, Path(output_dir) / f'{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_two_ranks_update_together_when_only_one_rank_has_choices(tmp_path):
    torch.multiprocessing.spawn(_distributed_temperature_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    rows = [torch.load(tmp_path / f'{rank}.pt', weights_only=True) for rank in range(2)]
    assert rows[0]['first'].item() > 0.
    assert torch.equal(rows[0]['first'], rows[1]['first'])
    assert all(torch.equal(row['first'], row['second']) for row in rows)
