"""Mix teacher/student execution while retaining separate supervised labels."""
import random
import time

from .baselines import reactive_attack


class CorrectionPolicy:
    def __init__(self, student, teacher_probability=0.5, seed=0, teacher=reactive_attack,
                 clock=time.perf_counter):
        if not 0 <= teacher_probability <= 1:
            raise ValueError("teacher_probability must be in [0,1]")
        self.student, self.teacher = student, teacher
        self.probability = teacher_probability
        self.rng = random.Random(seed)
        self.clock = clock
        self.last = None
        self.decisions = self.disagreements = self.teacher_steps = 0

    def __call__(self, obs):
        started = self.clock()
        student = int(self.student(obs))
        student_finished = self.clock()
        teacher = int(self.teacher(obs))
        teacher_finished = self.clock()
        if not 0 <= student < 9 or not 0 <= teacher < 9:
            raise ValueError("Invalid student or teacher action")
        use_teacher = self.rng.random() < self.probability
        action = teacher if use_teacher else student
        self.last = {"student_action": student, "teacher_action": teacher,
                     "student_inference_seconds": student_finished - started,
                     "teacher_inference_seconds": teacher_finished - student_finished,
                     "executed_by": "teacher" if use_teacher else "student"}
        self.decisions += 1
        self.disagreements += int(student != teacher)
        self.teacher_steps += int(use_teacher)
        return action

    def annotate(self, row):
        if row.get("event") == "transition":
            if self.last is None:
                raise RuntimeError("Missing correction decision")
            expected = self.last[f"{self.last['executed_by']}_action"]
            if row["action"] != expected:
                raise RuntimeError("Executed correction action mismatch")
            row = {**row, **self.last}
            self.last = None
        return row

    def summary(self):
        return {"event": "correction_summary", "decisions": self.decisions,
                "disagreements": self.disagreements, "teacher_steps": self.teacher_steps,
                "disagreement_rate": self.disagreements / self.decisions if self.decisions else None}
