"""Offline actor fitting with episode-held-out validation."""
from copy import deepcopy

import numpy as np
import torch


def fit_actor(learner, train_data, validation_data, emit, epochs=30, batch_size=256,
              learning_rate=0.001, seed=0, train_weights=None, validation_weights=None):
    if epochs < 1 or batch_size < 1 or not 0 < learning_rate <= 0.01:
        raise ValueError("Invalid cloning hyperparameters")
    policy = learner.model.policy
    device = learner.model.device
    rng = np.random.default_rng(seed)
    # A separate optimizer leaves PPO's optimizer fresh for fine-tuning.
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def prepare_weights(data, weights):
        weights = np.ones(len(data[1]), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        if weights.shape != (len(data[1]),) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Invalid imitation sample weights")
        return weights

    train_weights = prepare_weights(train_data, train_weights)
    validation_weights = prepare_weights(validation_data, validation_weights)

    def batches(data, weights, shuffle):
        observations, actions = data
        indices = rng.permutation(len(actions)) if shuffle else np.arange(len(actions))
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            obs = {k: torch.as_tensor(v[selected], device=device) for k, v in observations.items()}
            yield obs, torch.as_tensor(actions[selected], device=device), torch.as_tensor(weights[selected], device=device)

    def evaluate(data, weights):
        policy.set_training_mode(False)
        loss = correct = count = 0
        weighted_loss = weight_sum = priority_correct = priority_count = 0
        action_counts = np.zeros(9, dtype=np.int64)
        action_correct = np.zeros(9, dtype=np.int64)
        with torch.no_grad():
            for obs, actions, weight in batches(data, weights, False):
                distribution = policy.get_distribution(obs)
                losses = -distribution.log_prob(actions)
                loss += float(losses.sum())
                weighted_loss += float((losses * weight).sum())
                weight_sum += float(weight.sum())
                matches = distribution.mode() == actions
                correct += int(matches.sum())
                action_counts += np.bincount(actions.cpu().numpy(), minlength=9)
                action_correct += np.bincount(actions[matches].cpu().numpy(), minlength=9)
                count += len(actions)
                priority = weight > 1
                priority_count += int(priority.sum())
                priority_correct += int((matches & priority).sum())
        return {"loss": loss / count, "weighted_loss": weighted_loss / weight_sum,
                "priority_samples": priority_count,
                "priority_accuracy": priority_correct / priority_count if priority_count else None,
                "accuracy": correct / count, "samples": count,
                "action_counts": action_counts.tolist(),
                "action_accuracy": [int(n) / int(d) if d else None
                                    for n, d in zip(action_correct, action_counts)]}

    initial = evaluate(validation_data, validation_weights)
    best_loss, best_epoch = initial["weighted_loss"], 0
    best = deepcopy(policy.state_dict())
    emit({"event": "cloning_initial", "validation": initial})
    for epoch in range(1, epochs + 1):
        policy.set_training_mode(True)
        loss_sum = count = 0
        for obs, actions, weight in batches(train_data, train_weights, True):
            losses = -policy.get_distribution(obs).log_prob(actions)
            loss = (losses * weight).sum() / weight.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite cloning loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            loss_sum += float(loss.detach()) * float(weight.sum())
            count += float(weight.sum())
        validation = evaluate(validation_data, validation_weights)
        if not np.isfinite(validation["loss"]):
            raise RuntimeError("Non-finite validation loss")
        if validation["weighted_loss"] < best_loss:
            best_loss, best_epoch = validation["weighted_loss"], epoch
            best = deepcopy(policy.state_dict())
        emit({"event": "cloning_epoch", "epoch": epoch, "training_loss": loss_sum / count,
              "validation": validation})
    policy.load_state_dict(best)
    policy.set_training_mode(False)
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise RuntimeError("Non-finite cloned parameters")
    return {"selected_epoch": best_epoch, "initial_validation": initial,
            "training": evaluate(train_data, train_weights), "validation": evaluate(validation_data, validation_weights)}
