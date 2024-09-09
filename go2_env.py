from etils import epath
from typing import Any, Dict, Sequence, Tuple, Union, List
import os
from ml_collections import config_dict


import jax
from jax import numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags


class Go2Env(PipelineEnv):
    """Environment for training the barkour quadruped joystick policy in MJX."""

    def __init__(
        self,
        action_scale: float = 1.0,
        leg_control: str = "position",
    ):
        file_path = os.path.dirname(os.path.abspath(__file__))
        self.leg_control = leg_control
        if leg_control == "position":
            path = file_path + "/model/scene_mjx_position.xml"
        elif leg_control == "force":
            path = file_path + "/model/scene_mjx_force.xml"
            self.kp = 20.0
            self.kd = 0.4
        sys = mjcf.load(path)
        self._dt = 1.0 / 50.0  # this environment is 50 fps
        if self.leg_control == "position":
            sys = sys.tree_replace({"opt.timestep": 0.01})
        else:
            sys = sys.tree_replace({"opt.timestep": 0.02})

        n_frames = int(self._dt / sys.opt.timestep)
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        # gaits: FL, RL, FR, RR
        self.gait = "trot"
        self.gait_phase = {
            "stand": jnp.zeros(4),
            "walk": jnp.array([0.0, 0.5, 0.75, 0.25]),
            "trot": jnp.array([0.0, 0.5, 0.5, 0.0]),
            "canter": jnp.array([0.0, 0.33, 0.33, 0.66]),
            "gallop": jnp.array([0.0, 0.05, 0.4, 0.35]),
        }
        self.gait_params = {
            #                  ratio, cadence, amplitude
            "stand": jnp.array([1.0, 1.0, 0.0]),
            "walk": jnp.array([0.75, 1.0, 0.08]),
            "trot": jnp.array([0.45, 2, 0.08]),
            "canter": jnp.array([0.4, 4, 0.06]),
            "gallop": jnp.array([0.3, 3.5, 0.10]),
        }

        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "base"
        )
        self._action_scale = action_scale
        self._init_q = jnp.array(sys.mj_model.keyframe("home").qpos)
        self._default_pose = sys.mj_model.keyframe("home").qpos[7:]
        self.ctrl_range = sys.actuator_ctrlrange
        self.joint_range = sys.jnt_range[1:]
        feet_site = [
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        ]
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = jnp.array(feet_site_id)
        self._nv = sys.nv

    def get_foot_step(self, time):
        def step_height(t, footphase, duty_ratio):
            angle = (t + jnp.pi - footphase) % (2 * jnp.pi) - jnp.pi
            angle = jnp.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
            clipped_angle = jnp.clip(angle, -jnp.pi / 2, jnp.pi / 2)
            value = jnp.where(duty_ratio < 1, jnp.cos(clipped_angle), 0)
            final_value = jnp.where(jnp.abs(value) >= 1e-6, jnp.abs(value), 0.0)
            return final_value

        amplitude = self.gait_params[self.gait][2]
        cadence = self.gait_params[self.gait][1]
        duty_ratio = self.gait_params[self.gait][0]
        h_steps = amplitude * jax.vmap(step_height, in_axes=(None, 0, None))(
            time * 2 * jnp.pi * cadence + jnp.pi,
            2 * jnp.pi * self.gait_phase[self.gait],
            duty_ratio,
        )
        return h_steps

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "pos_tar": jnp.array([0.5, 0.0, 0.27]),
            "vel_tar": jnp.array([1.5, 0.0, 0.0]),
            "yaw_tar": 0.0,
            "step": 0,
            "z_feet": jnp.zeros(4),
            "z_feet_tar": jnp.zeros(4),
        }

        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(
            pipeline_state, obs, reward, done, metrics, state_info
        )  # pytype: disable=wrong-arg-types
        return state

    def act2joint(self, act: jax.Array) -> jax.Array:
        joint_targets = self._default_pose + act * self._action_scale
        joint_targets = jnp.clip(
            joint_targets, self.joint_range[:, 0], self.joint_range[:, 1]
        )
        return joint_targets

    def act2tau(self, act: jax.Array, pipline_state) -> jax.Array:
        joint_target = self.act2joint(act)

        q = pipline_state.q[7:]
        qd = pipline_state.qd[6:]
        q_err = joint_target - q
        tau = self.kp * q_err - self.kd * qd

        tau_reshaped = tau.reshape((4, 3))
        tau_limit = jnp.array([24.0, 24.0, 45.0])
        tau_reshaped = jnp.clip(tau_reshaped, -tau_limit, tau_limit)
        tau = tau_reshaped.flatten()

        # tau_limit = jnp.array([24.0, 24.0, 45.0])
        # tau = (act.reshape((4, 3)) * tau_limit).flatten()

        return tau

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        if self.leg_control == "position":
            ctrl = self.act2joint(action)
        elif self.leg_control == "force":
            ctrl = self.act2tau(action, state.pipeline_state)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info)

        # done
        done = 0.0

        # reward
        # gaits reward
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        z_feet_tar = self.get_foot_step(state.info["step"] * self._dt)
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)
        # position reward
        pos_tar = state.info["pos_tar"] + state.info["vel_tar"] * self._dt * state.info["step"]
        pos = x.pos[self._torso_idx - 1]
        reward_pos = -jnp.sum((pos - pos_tar) ** 2)
        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        # yaw orientation reward
        yaw_tar = state.info["yaw_tar"]
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        reward_yaw = -jnp.square(yaw - yaw_tar)
        # stay to norminal pose reward
        # reward_pose = -jnp.sum(jnp.square(joint_targets - self._default_pose))
        # reward
        reward = (
            reward_gaits * 0.1
            + reward_pos * 1.0
            + reward_upright * 1.0
            + reward_yaw * 0.3
            # + reward_pose * 0.0
        )

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_feet_tar

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        return jnp.zeros(0)

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)


if __name__ == "__main__":
    env = Go2Env(action_scale=1.0, leg_control="force")
    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    state = reset_jit(rng)
    rollout = []
    for _ in range(50):
        rng, _ = jax.random.split(rng)
        act = jax.random.uniform(rng, (12,), minval=-1, maxval=1)
        state = step_jit(state, act)
        rollout.append(state.pipeline_state)
        print(f"rew={state.reward:.2f}")

    webpage = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout)
    import flask

    app = flask.Flask(__name__)

    @app.route("/")
    def home():
        return webpage

    app.run(port=5000)
