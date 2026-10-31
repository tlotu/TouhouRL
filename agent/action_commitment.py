"""Explicit evaluation-only action holding; no teacher or collision override."""
import operator
import time


class ActionCommitment:
    def __init__(self, student, elapsed_frames, hold_frames=4, clock=time.perf_counter):
        if not isinstance(hold_frames, int) or not 2 <= hold_frames <= 16 or hold_frames % 2:
            raise ValueError("hold_frames must be an even integer from 2 to 16")
        self.student, self.elapsed_frames = student, elapsed_frames
        self.hold_frames, self.clock = hold_frames, clock
        self.held_action = self.since = self.last = None
        self.decisions = self.overrides = 0

    def __call__(self, obs):
        if self.last is not None:
            raise RuntimeError("Missing transition annotation")
        frame = self.elapsed_frames()
        if self.since is not None and frame < self.since:
            raise RuntimeError("Reset annotation required before a new episode")
        started = self.clock()
        raw = operator.index(self.student(obs))
        inference_seconds = self.clock() - started
        if not 0 <= raw < 9:
            raise ValueError("Invalid network action")
        active = self.since is not None and frame - self.since < self.hold_frames
        if not active:
            self.held_action, self.since = raw, frame
        self.last = {"raw_student_action": raw, "committed_action": self.held_action,
                     "commitment_active": active, "action_overridden": raw != self.held_action,
                     "hold_age_frames": frame - self.since,
                     "student_inference_seconds": inference_seconds}
        self.decisions += 1
        self.overrides += int(raw != self.held_action)
        return self.held_action

    def annotate(self, row):
        if row.get("event") == "reset":
            self.held_action = self.since = self.last = None
        if row.get("event") == "transition":
            if self.last is None or row["action"] != self.last["committed_action"]:
                raise RuntimeError("Commitment action annotation mismatch")
            row = {**row, **self.last}
            self.last = None
        return row

    def summary(self):
        return {"event": "commitment_summary", "decisions": self.decisions,
                "overridden_decisions": self.overrides, "hold_frames": self.hold_frames}
