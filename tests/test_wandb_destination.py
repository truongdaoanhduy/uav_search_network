import copy
from types import SimpleNamespace

import pytest

from config import CONFIG
from uav_marl.evaluation import postprocess


def test_cpu_logging_uses_destination_override_without_changing_snapshot(monkeypatch, tmp_path):
    monkeypatch.setitem(CONFIG, 'training_wandb_entity', 'deleted-team')
    monkeypatch.setitem(CONFIG, 'training_wandb_project', 'deleted-project')
    snapshot = copy.deepcopy(CONFIG)
    monkeypatch.setenv('WANDB_ENTITY', 'new-team')
    monkeypatch.setenv('WANDB_PROJECT', 'new-project')
    captured = {}
    class InitCaptured(Exception):
        pass
    def init(**kwargs):
        captured.update(kwargs)
        raise InitCaptured
    fake = SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs)
    original_import = postprocess.importlib.import_module
    monkeypatch.setattr(postprocess.importlib, 'import_module',
                        lambda name, *args, **kwargs: fake if name == 'wandb' else original_import(name, *args, **kwargs))
    monkeypatch.setattr(postprocess, 'running_on_kaggle', lambda: False)
    with pytest.raises(InitCaptured):
        postprocess.postprocess_checkpoint_cpu(output_dir=tmp_path, log_wandb=True)
    assert captured['entity'] == 'new-team'
    assert captured['project'] == 'new-project'
    assert CONFIG == snapshot


def test_deleted_source_run_does_not_discard_completed_visualization():
    class CommError(Exception):
        pass
    def missing_run(path):
        raise CommError('Could not find run ' + path)
    fake = SimpleNamespace(Api=lambda: SimpleNamespace(run=missing_run),
                           errors=SimpleNamespace(CommError=CommError))
    with pytest.warns(RuntimeWarning, match='Source training run'):
        linked = postprocess._update_source_run_summary(fake, 'new-team/project/deleted-run', {'cpu_postprocess/status': 'complete'})
    assert linked is False


def test_available_source_run_receives_visualization_summary():
    class CommError(Exception):
        pass
    summary = {}
    fake = SimpleNamespace(Api=lambda: SimpleNamespace(run=lambda path: SimpleNamespace(summary=summary)),
                           errors=SimpleNamespace(CommError=CommError))
    assert postprocess._update_source_run_summary(fake, 'new-team/project/source', {'cpu_postprocess/status': 'complete'})
    assert summary['cpu_postprocess/status'] == 'complete'


def test_destination_override_keeps_historical_source_namespace(monkeypatch, tmp_path):
    import torch
    monkeypatch.setitem(CONFIG, 'training_wandb_entity', 'deleted-team')
    monkeypatch.setitem(CONFIG, 'training_wandb_project', 'deleted-project')
    monkeypatch.setenv('WANDB_ENTITY', 'new-team')
    monkeypatch.setenv('WANDB_PROJECT', 'new-project')
    captured = {}
    run = SimpleNamespace(id='cpu-run', url='new-cpu-url', summary={},
                          log=lambda data: None, finish=lambda: None)
    def init(**kwargs):
        captured['destination'] = (kwargs['entity'], kwargs['project'])
        return run
    fake = SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs, Image=lambda path: path)
    original_import = postprocess.importlib.import_module
    monkeypatch.setattr(postprocess.importlib, 'import_module',
                        lambda name, *args, **kwargs: fake if name == 'wandb' else original_import(name, *args, **kwargs))
    monkeypatch.setattr(postprocess, 'running_on_kaggle', lambda: False)
    monkeypatch.setattr(postprocess, 'UAVSearchEnv',
                        lambda **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(postprocess, 'load_masac_checkpoint',
                        lambda *args, **kwargs: {'trainer': SimpleNamespace(actor=torch.nn.Linear(1, 1))})
    monkeypatch.setattr(postprocess, 'evaluate_masac',
                        lambda *args, **kwargs: [{'visualization_trace': object()}])
    monkeypatch.setattr(postprocess, 'summarize_evaluation_results', lambda rows: {'return': -1.})
    monkeypatch.setattr(postprocess, 'render_evaluation_trace_2d',
                        lambda *args, **kwargs: SimpleNamespace(savefig=lambda path, **kw: path.write_bytes(b'image')))
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'close', lambda fig: None)
    def update_source(module, path, summary):
        captured['source'] = path
        return True
    monkeypatch.setattr(postprocess, '_update_source_run_summary', update_source)
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    result = postprocess.postprocess_checkpoint_cpu(
        checkpoint_path=checkpoint, network_backend='simple', output_dir=tmp_path,
        source_run_id='historical-run', log_wandb=True,
        upload_huggingface=False, render_video=False)
    assert captured['destination'] == ('new-team', 'new-project')
    assert captured['source'] == 'deleted-team/deleted-project/historical-run'
    assert result['source_run_linked'] is True
