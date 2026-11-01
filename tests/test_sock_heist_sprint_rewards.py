from types import SimpleNamespace

import torch

from microduck_challenges.sock_heist_6m import rewards


def make_env(vx, commands, red, steps):
    velocities = torch.tensor(
        [[v, 0.0, 0.0] for v in vx],
        dtype=torch.float32,
    )

    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_b=velocities,
            root_link_lin_vel_w=velocities,
        )
    )

    term = SimpleNamespace(
        stepped=lambda: (
            torch.tensor(red, dtype=torch.bool),
            torch.zeros(len(red), dtype=torch.bool),
            torch.zeros(len(red), dtype=torch.bool),
        )
    )

    return SimpleNamespace(
        scene={"robot": robot},
        episode_length_buf=torch.tensor(steps) + 1,
        command_manager=SimpleNamespace(
            get_term=lambda name: term,
            get_command=lambda name: torch.tensor(
                [[v, 0.0, 0.0] for v in commands],
                dtype=torch.float32,
            ),
        ),
    )


def test_sprint_progress():
    env = make_env(
        vx=[1.7, 2.0, 0.5],
        commands=[1.2, 1.2, 0.0],
        red=[False, False, True],
        steps=[30, 30, 30],
    )

    result = rewards.progress(
        env,
        "twist",
        speed_factor=1.5,
        speed_cap=1.8,
    )

    torch.testing.assert_close(
        result,
        torch.tensor([1.7, 1.8, 0.0]),
    )


def test_green_distance():
    env = make_env(
        vx=[1.7, 2.2, 1.5, 1.6, -1.2],
        commands=[1.2, 1.2, 0.0, 1.2, 1.2],
        red=[False, False, True, False, False],
        steps=[30, 30, 30, 10, 30],
    )

    result = rewards.green_distance(
        env,
        "twist",
        speed_cap=1.8,
    )

    torch.testing.assert_close(
        result,
        torch.tensor([1.7, 1.8, 0.0, 0.0, -0.5]),
    )