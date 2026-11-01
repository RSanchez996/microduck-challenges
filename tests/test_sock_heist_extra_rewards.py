from types import SimpleNamespace

import torch

from microduck_challenges.sock_heist_6m import rewards


def mock_env(steps, velocity, commands, term):
    return SimpleNamespace(
        episode_length_buf=torch.tensor(steps) + 1,
        command_manager=SimpleNamespace(
            get_term=lambda name: term,
            get_command=lambda name: torch.tensor(
                commands, dtype=torch.float32
            ),
        ),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_lin_vel_b=torch.tensor(
                        velocity, dtype=torch.float32
                    ),
                )
            )
        },
    )


def test_green_launch():
    term = SimpleNamespace(
        onset=torch.tensor([[60, 200]] * 3),
        length=torch.tensor([[60, 60]] * 3),
    )

    env = mock_env(
        steps=[25, 120, 55],
        velocity=[[0.4, 0, 0], [0.5, 0, 0], [0.8, 0, 0]],
        commands=[[0.8, 0, 0], [1.0, 0, 0], [0.8, 0, 0]],
        term=term,
    )

    result = rewards.green_launch(env, "twist")

    torch.testing.assert_close(
        result,
        torch.tensor([0.5, 0.5, 0.0]),
    )


def test_red_braking():
    term = SimpleNamespace(
        stepped=lambda: (
            torch.tensor([True, True, False]),
            torch.tensor([False, False, False]),
            torch.tensor([False, True, False]),
        )
    )

    env = mock_env(
        steps=[80, 90, 100],
        velocity=[[1.2, 0, 0], [1.2, 0, 0], [0.6, 0, 0]],
        commands=[[0, 0, 0]] * 3,
        term=term,
    )

    result = rewards.red_braking(env, "twist")

    torch.testing.assert_close(
        result,
        torch.tensor([-1.0, 0.0, 0.0]),
    )