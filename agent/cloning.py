"""Offline actor fitting with episode-held-out validation."""
from copy import deepcopy

import numpy as np
import torch


def fit_actor(learner, train_data, validation_data, emit, epochs=30, batch_size=256,
              learning_rate=0.001, seed=0):
    if epochs < 1 or batch_size < 1 or not 0 < learning_rate <= 0.01:
        raise ValueError("Invalid cloning hyperparameters")
    policy = learner.model.policy
    device = learner.model.device
    rng = np.random.default_rng(seed)
    # A separate optimizer leaves PPO's optimizer fresh for fine-tuning.
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def batches(data, shuffle):
        observations, actions = data
        indices = rng.permutation(len(actions)) if shuffle else np.arange(len(actions))
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            obs = {k: torch.as_tensor(v[selected], device=device) for k, v in observations.items()}
            yield obs, torch.as_tensor(actions[selected], device=device)

    def evaluate(data):
        policy.set_training_mode(False)
        loss = correct = count = 0
        action_counts = np.zeros(9, dtype=np.int64)
        action_correct = np.zeros(9, dtype=np.int64)
        with torch.no_grad():
            for obs, actions in batches(data, False):
                distribution = policy.get_distribution(obs)
                loss += float(-distribution.log_prob(actions).sum())
                matches = distribution.mode() == actions
                correct += int(matches.sum())
                action_counts += np.bincount(actions.cpu().numpy(), minlength=9)
                action_correct += np.bincount(actions[matches].cpu().numpy(), minlength=9)
                count += len(actions)
        return {"loss": loss / count, "accuracy": correct / count, "samples": count,
                "action_counts": action_counts.tolist(),
                "action_accuracy": [int(n) / int(d) if d else None
                                    for n, d in zip(action_correct, action_counts)]}

    initial = evaluate(validation_data)
    best_loss, best_epoch = initial["loss"], 0
    best = deepcopy(policy.state_dict())
    emit({"event": "cloning_initial", "validation": initial})
    for epoch in range(1, epochs + 1):
        policy.set_training_mode(True)
        loss_sum = count = 0
        for obs, actions in batches(train_data, True):
            loss = -policy.get_distribution(obs).log_prob(actions).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite cloning loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(actions)
            count += len(actions)
        validation = evaluate(validation_data)
        if not np.isfinite(validation["loss"]):
            raise RuntimeError("Non-finite validation loss")
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            best = deepcopy(policy.state_dict())
        emit({"event": "cloning_epoch", "epoch": epoch, "training_loss": loss_sum / count,
              "validation": validation})
    policy.load_state_dict(best)
    policy.set_training_mode(False)
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise RuntimeError("Non-finite cloned parameters")
    return {"selected_epoch": best_epoch, "initial_validation": initial,
            "training": evaluate(train_data), "validation": evaluate(validation_data)}
