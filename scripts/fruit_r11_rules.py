"""Revision 11 observation rules and atomic data-attempt accounting."""
import threading


def hidden_content_visible(visibility):
    return any(v.get(k, 0) for v in visibility.values()
               for k in ('basket_fruit', 'inactive_reserve'))


def intersection_stop(failures, started):
    # Integer arithmetic preserves the strict >2% boundary.
    return failures >= 3 and failures * 100 > started * 2


class AttemptLedger:
    def __init__(self, persist, previous=None):
        self.lock = threading.RLock()
        self.rows = {r['key']: r for r in (previous or {}).get('attempts', [])}
        self.persist = persist
        self.stopped = bool((previous or {}).get('stopped', False))

    def snapshot(self):
        failures = sum(r.get('outcome') == 'shield_intersection' for r in self.rows.values())
        started = len(self.rows)
        return {'started_attempts': started, 'shield_intersection_attempts': failures,
                'fraction': failures / started if started else 0.,
                'stop_threshold_reached': intersection_stop(failures, started),
                'stopped': self.stopped, 'denominator': 'All started data attempts, both tasks, clean and DART',
                'attempts': list(self.rows.values())}

    def start(self, key, details):
        with self.lock:
            if self.stopped:
                raise RuntimeError('R11 data attempt intersection threshold reached; no further starts')
            if key in self.rows:
                raise RuntimeError('Duplicate data attempt start: ' + key)
            self.rows[key] = {**details, 'key': key}
            self.persist(self.snapshot())

    def finish(self, key, outcome, success):
        with self.lock:
            row = self.rows[key]
            if 'outcome' in row:
                raise RuntimeError('Duplicate data attempt completion: ' + key)
            row.update(outcome=outcome, success=bool(success))
            current = self.snapshot()
            self.stopped |= current['stop_threshold_reached']
            current['stopped'] = self.stopped
            self.persist(current)
            return self.stopped
