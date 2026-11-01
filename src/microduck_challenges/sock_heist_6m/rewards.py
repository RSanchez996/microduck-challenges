"""Sock Heist speed-v2 rewards, bound to Arena-compatible LightsCommand."""
import torch
from . import lights


def freeze(env, command_name: str, std: float) -> torch.Tensor:
    """Pay only while watched; stop drift must stay below 3 cm."""
    term = env.command_manager.get_term(command_name)
    _, _, watch = term.stepped()
    drift = term.drift()
    return torch.where(watch, torch.exp(-((drift / std) ** 2)), torch.zeros_like(drift))


def green_distance(env, command_name: str, speed_factor: float = 2.0,
                   speed_cap: float = 2.4) -> torch.Tensor:
    """World-forward speed on GREEN only; scales target with Arena command.

    The command seen by the policy is unchanged (up to 1.2 m/s), but the
    reward stays positive up to e.g. 1.2 * 2.0 = 2.4 m/s. Reverse progress
    is mildly penalized, not clipped to zero. During reds this term is zero.
    """
    term = env.command_manager.get_term(command_name)
    red, _, _ = term.stepped()
    step = env.episode_length_buf - 1
    green = (step >= lights.steps(lights.SETTLE_S)) & ~red
    cmd = env.command_manager.get_command(command_name)[:, 0].clamp(min=0.0)
    target = (cmd * speed_factor).clamp(max=speed_cap)
    vx_world = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
    rewarded_vx = torch.minimum(vx_world.clamp(min=-0.5), target)
    return torch.where(green, rewarded_vx, torch.zeros_like(rewarded_vx))


def red_stillness(env, command_name: str, std: float = 0.22) -> torch.Tensor:
    """A dense zero-velocity reward on RED and initial SETTLE.

    Complements the watch-only positional freeze reward. It does not pay
    while the robot is green-running, so fast green speed is never penalized.
    """
    term = env.command_manager.get_term(command_name)
    red, _, _ = term.stepped()
    settle = env.episode_length_buf - 1 < lights.steps(lights.SETTLE_S)
    vxy = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    speed_sq = vxy.square().sum(dim=-1)
    stillness = torch.exp(-speed_sq / (std * std))
    return torch.where(red | settle, stillness, torch.zeros_like(stillness))


def red_braking(env, command_name: str) -> torch.Tensor:
    """Penalize speed while the giant turns, before watch begins."""
    term = env.command_manager.get_term(command_name)
    red, _, watch = term.stepped()
    vxy = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    speed_squared = (vxy / 1.2).square().sum(dim=-1)
    return torch.where(red & ~watch, -speed_squared, torch.zeros_like(speed_squared))


def caught(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    _, _, watch = term.stepped()
    return watch & (term.drift() > lights.TOLERANCE_M)


def off_lane(env, command_name: str, half_width: float) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    return (term.trunk[:, 1] - term.lane_y).abs() > half_width


def light_state(env, command_name: str) -> torch.Tensor:
    """Privileged critic-only light state, unchanged from original challenge."""
    term = env.command_manager.get_term(command_name)
    standing, watch = term.ahead()
    k = env.episode_length_buf[:, None]
    settle = torch.full_like(term.onset[:, :1], lights.steps(lights.SETTLE_S))
    changes = torch.cat([settle, term.onset, term.onset + term.length], dim=1)
    until = (changes - k).masked_fill(changes <= k, torch.iinfo(changes.dtype).max).amin(dim=1)
    drift = torch.where(watch, term.drift() / lights.TOLERANCE_M, 0.0)
    return torch.stack([standing.float(), watch.float(), until * lights.CTRL_DT, drift], dim=-1)


def lane_state(env, command_name: str) -> torch.Tensor:
    term = env.command_manager.get_term(command_name)
    trunk = term.trunk
    return torch.stack([trunk[:, 1] - term.lane_y, lights.yaw_of(trunk[:, 3:7])], dim=-1)
