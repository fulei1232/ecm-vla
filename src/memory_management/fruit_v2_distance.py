"""R9 adopts any_hanging; the maturity-sensitive helper remains for read-only comparisons."""
import numpy as np

from .fruit_distance import distance as legacy_distance, NO_TARGET_M


def any_hanging(tcp,positions,hanging):
    """No maturity argument or access to maturity labels."""
    points=np.asarray(positions,dtype=float)[np.asarray(hanging,dtype=bool)]
    if not len(points):return NO_TARGET_M
    tcp=np.asarray(tcp,dtype=float)
    return float(min(np.linalg.norm(tcp-p) for p in points))


def maturity_sensitive(tcp,positions,ripe,hanging):
    return legacy_distance(tcp,positions,ripe,~np.asarray(hanging,dtype=bool))


def from_gt_row(row):
    """R9 build input: parked reserves, windfalls and harvested fruit are excluded."""
    fruits = row['fruits']
    return any_hanging(row['tcp_pos'], [f['pos'] for f in fruits],
                       [bool(f.get('present', True)) and not f['attached'] for f in fruits])


def from_snapshot(snapshot):
    """Live recorder schema; maturity labels and the legacy distance field are ignored."""
    present = np.asarray(snapshot.get('present', [True] * len(snapshot['detached'])), dtype=bool)
    hanging = present & ~np.asarray(snapshot['detached'], dtype=bool)
    return any_hanging(snapshot['tcp_position'], snapshot['fruit_positions'], hanging)


class RecordedDistance:
    """Opt-in R9 inference signal computed from recorded geometry, never maturity."""
    def reset(self, context):
        from pathlib import Path
        from .validation.v_a_logger import RawFrameTail
        self.tail = RawFrameTail(Path(context['directory']) / 'raw_frames.jsonl')

    def distance(self, frame):
        import hashlib
        row = self.tail.upto(frame.frame_id)[frame.frame_id]
        if hashlib.sha256(np.asarray(frame.image).tobytes(order='C')).hexdigest() != row['front_sha256']:
            raise ValueError(f'frame {frame.frame_id}: image differs from the recorded row')
        return from_snapshot(row['gt'])
