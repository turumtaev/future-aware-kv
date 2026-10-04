import pytest
import torch

from experiments.many_to_one import (
    MatchingModel,
    Task,
    generate_batch,
    threshold_temperature,
)


def test_unique_sampling_rejects_impossible_task():
    task = Task(groups=3, members=2, member_vocab=5, label_vocab=3)
    with pytest.raises(ValueError, match="more unique IDs"):
        generate_batch(task, 1, torch.Generator().manual_seed(1))


def test_batch_queries_every_member_with_its_group_label():
    task = Task(groups=4, members=3, member_vocab=64, label_vocab=4)
    batch = generate_batch(task, 5, torch.Generator().manual_seed(4))
    assert batch.body.shape == (5, 16)
    assert batch.queries.shape == batch.targets.shape == (5, 12)
    for row in range(5):
        for group in range(4):
            start = group * 3
            assert (batch.targets[row, start : start + 3] == batch.targets[row, start]).all()
            for query in batch.queries[row, start : start + 3]:
                assert (batch.body[row] == query).sum() == 1


def test_random_boundary_shift_targets_the_next_label_to_the_right():
    task = Task(
        groups=3,
        members=4,
        member_vocab=64,
        label_vocab=4,
        random_boundary_shift=True,
    )
    batch = generate_batch(task, 16, torch.Generator().manual_seed(5))
    assert batch.body.shape == (16, 16)
    assert batch.queries.shape == batch.targets.shape == (16, 12)
    observed_shifts = set()
    for row in range(batch.body.shape[0]):
        is_label = batch.body[row] >= task.member_vocab
        first_label = int(is_label.nonzero()[0])
        observed_shifts.add(task.members - first_label)
        assert is_label.sum() == task.groups + 1
        for query, target in zip(batch.queries[row], batch.targets[row]):
            position = int((batch.body[row] == query).nonzero()[0])
            next_label = position + int(is_label[position:].nonzero()[0])
            assert target == batch.body[row, next_label] - task.member_vocab
    assert observed_shifts.issubset(set(range(task.members)))
    assert len(observed_shifts) > 1


def test_queries_are_independent_virtual_final_tokens():
    torch.manual_seed(8)
    task = Task(groups=3, members=2, member_vocab=32, label_vocab=3)
    batch = generate_batch(task, 2, torch.Generator().manual_seed(10))
    for architecture in ("2a", "fakv"):
        model = MatchingModel(architecture, task, width=16, heads=2, head_dim=4).double()
        together = model(batch.body, batch.queries)
        separate = torch.cat([
            model(batch.body, batch.queries[:, index : index + 1])
            for index in range(batch.queries.shape[1])
        ], 1)
        torch.testing.assert_close(together, separate, rtol=2e-12, atol=2e-12)


def test_shifted_models_use_relative_positions():
    torch.manual_seed(9)
    task = Task(
        groups=3,
        members=2,
        member_vocab=32,
        label_vocab=4,
        random_boundary_shift=True,
    )
    batch = generate_batch(task, 2, torch.Generator().manual_seed(11))
    for architecture in ("2a", "fakv"):
        model = MatchingModel(
            architecture, task, width=16, heads=2, head_dim=4
        ).double()
        together = model(batch.body, batch.queries)
        separate = torch.cat(
            [model(batch.body, batch.queries[:, index : index + 1])
             for index in range(batch.queries.shape[1])],
            dim=1,
        )
        torch.testing.assert_close(together, separate, rtol=2e-12, atol=2e-12)


def test_retrieval_copy_is_initialization_not_weight_tying():
    task = Task(groups=3, members=2, member_vocab=32, label_vocab=3)
    model = MatchingModel("fakv", task, width=16, heads=2, head_dim=4)
    assert torch.equal(model.attention.kc0.weight, model.attention.qc.weight)
    assert model.attention.kc0.weight is not model.attention.qc.weight


def test_frozen_recipe_parameter_counts():
    for members in (2, 64):
        task = Task(members=members, label_vocab=5, random_boundary_shift=True)
        model_2a = MatchingModel("2a", task)
        model_fakv = MatchingModel("fakv", task)
        assert sum(p.numel() for p in model_2a.parameters()) == 329_088
        assert sum(p.numel() for p in model_fakv.parameters()) == 320_456


def test_temperature_schedule_and_hard_window_evaluation():
    assert threshold_temperature(1, 100, 4.0, 0.25) == 4.0
    assert threshold_temperature(50, 100, 4.0, 0.25) == 4.0
    assert 0.25 < threshold_temperature(70, 100, 4.0, 0.25) < 4.0
    assert threshold_temperature(90, 100, 4.0, 0.25) == 0.25


def test_one_training_step_has_finite_gradients():
    torch.manual_seed(11)
    task = Task(groups=3, members=2, member_vocab=32, label_vocab=3)
    batch = generate_batch(task, 2, torch.Generator().manual_seed(12))
    for architecture in ("2a", "fakv"):
        model = MatchingModel(architecture, task, width=16, heads=2, head_dim=4)
        logits = model(batch.body, batch.queries)
        torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), batch.targets.flatten()
        ).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
