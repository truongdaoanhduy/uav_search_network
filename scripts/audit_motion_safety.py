"""Seeded motion-only audit: 16 worlds, 3,000 steps, full clearance paths.

Run from the repository root: python scripts/audit_motion_safety.py
Does not run sensing, networking, energy accounting, or policy training.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config import CONFIG
from uav_marl.envs.gpu import FullGpuUAVBatchEnv

torch.set_num_threads(2)
env=FullGpuUAVBatchEnv(16,device='cpu',seed=44,strict_cuda=False)
g=torch.Generator().manual_seed(1044)
started=time.monotonic(); minimum=float('inf')
for step in range(3000):
 old=env.positions.clone()
 old_velocity=env.velocities.clone()
 a=torch.rand((16,env.num_uavs,3),generator=g)*2-1
 if step<20:a[...,2]=1
 result=env._motion(a)
 rel=old[:,:,None,:]-old[:,None,:,:]
 move=env.positions-old
 delta=move[:,:,None,:]-move[:,None,:,:]
 t=(-(rel*delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-12)).clamp(0,1)
 distances=torch.linalg.vector_norm(rel+t[...,None]*delta,dim=-1)
 distances.masked_fill_(torch.eye(env.num_uavs,dtype=torch.bool)[None],float('inf'))
 closest=float(distances.min()); minimum=min(minimum,closest)
 hit=env._segment_cylinder_blocked(old,env.positions,obstacle_radius=env.obstacle_radius+float(CONFIG['obstacle_clearance_m'])-1e-3,obstacle_height=env.obstacle_height+float(CONFIG['obstacle_clearance_m'])-1e-3)
 ok=closest>=float(CONFIG['safety_distance'])-1e-3 and not bool(hit.any())
 if not ok:
  torch.save({'step':step+1,'config':dict(CONFIG),'old_velocities':old_velocity,'old_positions':old,'positions':env.positions,'velocities':env.velocities,'actions':a,'obstacle_xy':env.obstacle_xy,'obstacle_radius':env.obstacle_radius,'obstacle_height':env.obstacle_height,'result':result},'/tmp/uav_motion_failure.pt')
  raise AssertionError({'step':step+1,'min_peer_m':closest,'obstacle_hits':int(hit.sum()),'hit_indices':hit.nonzero().tolist()})
 assert torch.isfinite(env.positions).all()
 assert float(torch.linalg.vector_norm(env.velocities,dim=-1).max())<=float(CONFIG['max_speed'])+1e-4
 if (step+1)%500==0:print(json.dumps({'step':step+1,'min_peer_m':minimum,'seconds':time.monotonic()-started}),flush=True)
print(json.dumps({'passed':True,'envs':16,'steps_each':3000,'min_peer_m':minimum,'seconds':time.monotonic()-started}),flush=True)
env.close()
