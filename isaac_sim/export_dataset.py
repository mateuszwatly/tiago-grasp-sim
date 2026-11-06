#!/usr/bin/env python3
"""Export completed Isaac episodes into portable, framework-neutral training records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def export(source, output, successes_only=False, keep_replays=False):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'dataset.json').exists():
        raise ValueError('Use a new output directory; exports are immutable snapshots')
    rows, rejected, seen = [], [], set()
    for metadata_path in sorted(source.glob('*/episodes/*/metadata.json'), reverse=True):
        session = metadata_path.parents[2]
        validation = session / 'validation.json'
        if validation.exists() and not read_json(validation).get('valid_for_training', True):
            continue
        try:
            metadata = read_json(metadata_path)
            if metadata.get('schema_version', 0) < 3:
                raise ValueError('unsupported legacy schema')
            if successes_only and not metadata['success']:
                continue
            trajectory = [json.loads(line) for line in metadata_path.with_name('trajectory.jsonl').read_text().splitlines()]
            if len(trajectory) < 2:
                raise ValueError('fewer than two observations')
            if metadata['success'] and not (all(metadata['phase_results'].values()) and metadata['object_rise'] >= metadata['minimum_verified_lift']):
                raise ValueError('inconsistent success label')
            # Replayed scene seeds belong to the same split, regardless of worker/session.
            scene_key = str(metadata['seed'])
            key = hashlib.sha256(scene_key.encode()).hexdigest()
            if key in seen and not keep_replays:
                continue
            split_bucket = int(key[:8], 16) % 100
            split = 'train' if split_bucket < 80 else 'validation' if split_bucket < 90 else 'test'
            names = trajectory[0]['joint_names']
            if any(t['joint_names'] != names for t in trajectory):
                raise ValueError('joint ordering changed')
            config_path = session / 'config.json'
            config = read_json(config_path).get('generator', {}) if config_path.exists() else {}
            dt = config.get('physics_dt', 1 / 120)
            inferred_time = any('simulation_time_seconds' not in t for t in trajectory)
            times = np.array([t.get('simulation_time_seconds', t['physics_step'] * dt) for t in trajectory], dtype=np.float64)
            if not np.all(np.diff(times) > 0):
                raise ValueError('non-monotonic observation timestamps')
            frames = []
            for t in trajectory:
                frame = {'time_seconds': float(times[len(frames)]), 'frame_id': t['replicator_frame'], 'phase': t['phase'], 'cameras': {}}
                for camera, directory in t['camera_paths'].items():
                    files = sorted((session / directory).glob(f'*_{t["replicator_frame"]:04d}.*'))
                    by_kind = {f.stem.rsplit('_', 1)[0]: f for f in files}
                    for required in ['rgb', 'distance_to_image_plane', 'camera_params', 'semantic_segmentation', 'instance_id_segmentation']:
                        if required not in by_kind:
                            raise ValueError(f'missing {camera}/{required} frame {t["replicator_frame"]}')
                    frame['cameras'][camera] = {kind: os.path.relpath(f, output) for kind, f in by_kind.items()}
                frames.append(frame)
            n = len(trajectory)
            arrays = {
                'time_seconds': times,
                'dt_seconds': np.r_[np.diff(times), 0].astype(np.float32),
                'joint_position': np.array([t['joint_positions'] for t in trajectory], np.float32),
                'joint_velocity': np.array([t['joint_velocities'] for t in trajectory], np.float32),
                'ee_position_world': np.array([t['end_effector_position'] for t in trajectory], np.float32),
                'ee_quaternion_wxyz': np.array([t['end_effector_orientation_wxyz'] for t in trajectory], np.float32),
                'target_position_world': np.array([t['target_object_position'] for t in trajectory], np.float32),
                'achieved_motion': np.array([t['action']['values'][:7] for t in trajectory], np.float32),
                'transition_valid': np.arange(n) < n - 1,
                'is_first': np.arange(n) == 0, 'is_last': np.arange(n) == n - 1,
                'imitation_mask': (np.arange(n) < n - 1) & bool(metadata['success']),
            }
            has_commands = all('joint_position_targets' in t for t in trajectory)
            if has_commands:
                arrays['joint_position_target_at_capture'] = np.array([t['joint_position_targets'] for t in trajectory], np.float32)
                arrays['all_object_pose_world_wxyz'] = np.array([
                    [o['position'] + o['orientation_wxyz'] for o in t['all_object_states']]
                    for t in trajectory], np.float32)
                arrays['all_object_velocity_world'] = np.array([
                    [o['velocity_linear_angular'] for o in t['all_object_states']]
                    for t in trajectory], np.float32)
                arrays['attachment_active'] = np.array([t['attachment_active'] for t in trajectory], bool)
                arrays['robot_link_world_pose_xyzw'] = np.array([t['robot_link_world_pose_xyzw'] for t in trajectory], np.float32)
            for field in ['joint_efforts', 'link_incoming_joint_wrench_local']:
                if all(t.get(field) is not None for t in trajectory):
                    arrays[field] = np.array([t[field] for t in trajectory], np.float32)
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError('non-finite numeric data')
            record_key = hashlib.sha256((scene_key + session.name).encode()).hexdigest() if keep_replays else key
            relative = Path('episodes') / record_key[:20]
            destination = output / relative
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / 'arrays.npz', **arrays)
            (destination / 'frames.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
            shutil.copy2(metadata_path, destination / 'metadata.json')
            shutil.copy2(metadata_path.with_name('trajectory.jsonl'), destination / 'raw_trajectory.jsonl')
            controls = metadata_path.with_name('controls.jsonl')
            if controls.exists():
                shutil.copy2(controls, destination / 'controls.jsonl')
            if config_path.exists():
                shutil.copy2(config_path, destination / 'session_config.json')
            row = {'episode_id': record_key[:20], 'scene_key': scene_key, 'split': split,
                   'success': bool(metadata['success']), 'length': n,
                   'instruction': metadata['instruction'], 'joint_names': names,
                   'arrays': str(relative / 'arrays.npz'), 'frames': str(relative / 'frames.jsonl'),
                   'metadata': str(relative / 'metadata.json'), 'has_commands': has_commands,
                   'controls': str(relative / 'controls.jsonl') if controls.exists() else None,
                   'timestamps_inferred': inferred_time, 'source': os.path.relpath(metadata_path, output)}
            rows.append(row)
            seen.add(key)
        except (ValueError, KeyError, OSError) as error:
            rejected.append({'source': str(metadata_path), 'reason': str(error)})
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (output / 'rejected.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rejected))
    summary = {'schema_version': 1, 'episode_count': len(rows), 'successes': sum(r['success'] for r in rows),
               'rejected': len(rejected), 'keep_replays': keep_replays, 'split_policy': 'sha256(scene_seed) modulo 100: 80/10/10; replay deduplicated',
               'media_storage': 'relative references; retain source recordings',
               'action_semantics': 'achieved_motion is observed next-frame world delta xyz, world rotvec, next left-finger position; not a command',
               'units': 'meters, radians, seconds; poses world frame; quaternions explicitly ordered',
               'success_semantics': 'geometric enclosure plus attachment-assisted verified lift'}
    (output / 'dataset.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


class EpisodeDataset:
    """Framework-neutral loader; each item contains arrays and lazy media references."""
    def __init__(self, root, split=None, successes_only=False):
        self.root = Path(root)
        self.episodes = [r for r in map(json.loads, (self.root / 'episodes.jsonl').read_text().splitlines())
                         if (split is None or r['split'] == split) and (not successes_only or r['success'])]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        record = self.episodes[index]
        with np.load(self.root / record['arrays'], allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        frames = list(map(json.loads, (self.root / record['frames']).read_text().splitlines()))
        controls = list(map(json.loads, (self.root / record['controls']).read_text().splitlines())) if record['controls'] else None
        return {'metadata': record, 'arrays': arrays, 'frames': frames, 'controls': controls, 'media_root': self.root}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--successes-only', action='store_true')
    parser.add_argument('--keep-replays', action='store_true')
    args = parser.parse_args()
    export(args.source.resolve(), args.output.resolve(), args.successes_only, args.keep_replays)
