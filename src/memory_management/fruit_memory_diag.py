"""Opt-in exploratory frame ablations; frozen D5 selector files stay intact."""
from pathlib import Path
import sys

ECM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ECM/'scripts'))
from fruit_memory_diag_core import select_frames
from .validation.teacher_oracle import OracleLogger
from .validation import event_memory as em


class DiagnosticOracleLogger(OracleLogger):
    def __init__(self, *args, diagnostics_enabled=False, **kwargs):
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self.memory_ablation = None
        self._defer_oracle = False
        super().__init__(*args, **kwargs)

    def reset(self):
        self.memory_ablation = None
        result = super().reset()
        mode = (self.context or {}).get('memory_ablation')
        if mode not in (None, 'recent16', 'no_pregust'):
            raise ValueError(f'Unknown diagnostic mode: {mode}')
        if mode is not None and not self.diagnostics_enabled:
            raise ValueError('Memory diagnostics are disabled by default')
        if mode is not None and self.condition != 'K0':
            raise ValueError('Revision6 only permits K0 inference ablations')
        self.memory_ablation = mode
        return result

    def decide(self, obs, teacher):
        if self.memory_ablation is None:
            return super().decide(obs, teacher)
        self._defer_oracle = True
        try:
            super().decide(obs, teacher)
        finally:
            self._defer_oracle = False
        t = int(self.policy.step_idx)
        if self.policy.exec_start_idx != 30 or t < 30:
            raise ValueError('Diagnostic ablations require the unchanged30-step physical prefix')
        selected = select_frames(t, self.memory_ablation)
        log = dict(self.oracle_log)
        log.update(memory_ablation=self.memory_ablation, selected_frame_ids=selected,
            physical_control_step=t, candidate_min_physical_step=20 if self.memory_ablation == 'no_pregust' else None)
        log['baseline_k0_keyframes'] = log.pop('keyframes', None)
        if self.signal is not None:
            cfg = self.event.for_task(self.context['task'])
            log['event'] = em.memory_stats(selected, log['uniform'], self.signal.distances(t), cfg.tau)
        self.pending, self.oracle_log = (t, selected), log
        self.emit('oracle_decisions.jsonl', log)

    def emit(self, name, row):
        if self._defer_oracle and name == 'oracle_decisions.jsonl':
            return
        if self.memory_ablation is not None and name == 'policy_decisions.jsonl':
            row = {**row, 'memory_ablation': self.memory_ablation}
        return super().emit(name, row)
