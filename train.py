"""Walking task for ZBot."""

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Self

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from ksim.types import Metadata

logger = logging.getLogger(__name__)

NUM_JOINTS = 20
NUM_COMMANDS = 6

ACTOR_DIM: dict[str, int] = dict(
    joint_positions=20,
    joint_velocity=20,
    imu_orientation=4,
    cmd_zero=1,
    cmd_linear_velocity=2,
    cmd_yaw_rate=1,
    cmd_base_height_roll_pitch=3,
)

CRITIC_DIM: dict[str, int] = dict(
    joint_positions=20,
    joint_velocity=20,
    imu_quat=4,
    cmd_all=7,
    imu_gyro=3,
    left_touch=1,
    right_touch=1,
    feet_position=6,
    base_pos=3,
    base_quat=4,
    com_inertia=250,
    com_velocity=150,
    base_lin_vel=3,
    base_ang_vel=3,
    act_force=20,
    base_height=1,
)

NUM_ACTOR_INPUTS = sum(ACTOR_DIM.values())
NUM_CRITIC_INPUTS = sum(CRITIC_DIM.values())


# These are in the order of the neural network outputs.
# (joint_name, reference_angle_rad, weight)
JOINT_BIASES: list[tuple[str, float, float]] = [
    ("right_hip_yaw", 0.0, 1.0),  # 0
    ("right_hip_roll", -0.1, 1.0),  # 1
    ("right_hip_pitch", -0.4, 0.01),  # 2
    ("right_knee_pitch", -0.8, 0.01),  # 3
    ("right_ankle_pitch", -0.4, 0.01),  # 4
    ("right_ankle_roll", -0.1, 0.01),  # 5
    ("left_hip_yaw", 0.0, 1.0),  # 6
    ("left_hip_roll", 0.1, 1.0),  # 7
    ("left_hip_pitch", -0.4, 0.01),  # 8
    ("left_knee_pitch", -0.8, 0.01),  # 9
    ("left_ankle_pitch", -0.4, 0.01),  # 10
    ("left_ankle_roll", 0.1, 0.01),  # 11
    ("left_shoulder_pitch", 0.0, 1.0),  # 12
    ("left_shoulder_roll", 0.2, 1.0),  # 13
    ("left_elbow_roll", -0.2, 1.0),  # 14
    ("left_gripper_roll", 0.0, 1.0),  # 15
    ("right_shoulder_pitch", 0.0, 1.0),  # 16
    ("right_shoulder_roll", -0.2, 1.0),  # 17
    ("right_elbow_roll", 0.2, 1.0),  # 18
    ("right_gripper_roll", 0.0, 1.0),  # 19
]


def rotate_quat_by_quat(quat_to_rotate: Array, rotating_quat: Array, inverse: bool = False, eps: float = 1e-6) -> Array:
    """Rotates one quaternion by another quaternion through quaternion multiplication.

    This performs the operation: rotating_quat * quat_to_rotate * rotating_quat^(-1) if inverse=False
    or rotating_quat^(-1) * quat_to_rotate * rotating_quat if inverse=True

    Args:
        quat_to_rotate: The quaternion being rotated (w,x,y,z), shape (*, 4)
        rotating_quat: The quaternion performing the rotation (w,x,y,z), shape (*, 4)
        inverse: If True, rotate by the inverse of rotating_quat
        eps: Small epsilon value to avoid division by zero in normalization

    Returns:
        The rotated quaternion (w,x,y,z), shape (*, 4)
    """
    # Normalize both quaternions
    quat_to_rotate = quat_to_rotate / (jnp.linalg.norm(quat_to_rotate, axis=-1, keepdims=True) + eps)
    rotating_quat = rotating_quat / (jnp.linalg.norm(rotating_quat, axis=-1, keepdims=True) + eps)

    # If inverse requested, conjugate the rotating quaternion (negate x,y,z components)
    if inverse:
        rotating_quat = rotating_quat.at[..., 1:].multiply(-1)

    # Extract components of both quaternions
    w1, x1, y1, z1 = jnp.split(rotating_quat, 4, axis=-1)  # rotating quaternion
    w2, x2, y2, z2 = jnp.split(quat_to_rotate, 4, axis=-1)  # quaternion being rotated

    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = jnp.concatenate([w, x, y, z], axis=-1)

    # Normalize result
    return result / (jnp.linalg.norm(result, axis=-1, keepdims=True) + eps)


@attrs.define(frozen=True)
class UnifiedCommand(ksim.Command):
    """Unifiying all commands into one to allow for covariance control."""

    vx_range: tuple[float, float] = attrs.field()
    vy_range: tuple[float, float] = attrs.field()
    wz_range: tuple[float, float] = attrs.field()
    bh_range: tuple[float, float] = attrs.field()
    bh_standing_range: tuple[float, float] = attrs.field()
    rx_range: tuple[float, float] = attrs.field()
    ry_range: tuple[float, float] = attrs.field()
    ctrl_dt: float = attrs.field()
    switch_prob: float = attrs.field()

    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rng_a, rng_b, rng_c, rng_d, rng_e, rng_f, rng_g, rng_h = jax.random.split(rng, 8)

        # cmd  = [vx, vy, wz, bh, rx, ry]
        vx = jax.random.uniform(rng_b, (1,), minval=self.vx_range[0], maxval=self.vx_range[1])
        vy = jax.random.uniform(rng_c, (1,), minval=self.vy_range[0], maxval=self.vy_range[1])
        wz = jax.random.uniform(rng_d, (1,), minval=self.wz_range[0], maxval=self.wz_range[1])
        bh = jax.random.uniform(rng_e, (1,), minval=self.bh_range[0], maxval=self.bh_range[1])
        bhs = jax.random.uniform(rng_f, (1,), minval=self.bh_standing_range[0], maxval=self.bh_standing_range[1])
        rx = jax.random.uniform(rng_g, (1,), minval=self.rx_range[0], maxval=self.rx_range[1])
        ry = jax.random.uniform(rng_h, (1,), minval=self.ry_range[0], maxval=self.ry_range[1])

        _ = jnp.zeros_like(vx)

        # Create each mode's command vector
        forward_cmd = jnp.concatenate([vx, _, _, bh, _, _])
        sideways_cmd = jnp.concatenate([_, vy, _, bh, _, _])
        rotate_cmd = jnp.concatenate([_, _, wz, bh, _, _])
        # omni_cmd = jnp.concatenate([vx, vy, wz, bh, _, _])
        stand_bend_cmd = jnp.concatenate([_, _, _, bhs, rx, ry])
        stand_cmd = jnp.concatenate([_, _, _, _, _, _])

        # randomly select a mode
        mode = jax.random.randint(rng_a, (), minval=0, maxval=5)  # 0 1 2 3 4s 5s -- 2/6 standing
        cmd = jax.lax.switch(
            mode,
            [
                lambda: forward_cmd,
                lambda: sideways_cmd,
                lambda: rotate_cmd,
                # lambda: omni_cmd,
                lambda: stand_bend_cmd,
                lambda: stand_cmd,
            ],
        )

        # get initial heading
        init_euler = xax.quat_to_euler(physics_data.xquat[1])
        init_heading = init_euler[2] + self.ctrl_dt * cmd[2]  # add 1 step of yaw vel cmd to initial heading.
        cmd = jnp.concatenate([cmd[:3], jnp.array([init_heading]), cmd[3:]])
        assert cmd.shape == (7,)

        return cmd

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        def update_heading(prev_command: Array) -> Array:
            """Update the heading by integrating the angular velocity."""
            wz_cmd, heading = prev_command[2], prev_command[3]
            heading = heading + wz_cmd * self.ctrl_dt
            prev_command = prev_command.at[3].set(heading)
            return prev_command

        continued_command = update_heading(prev_command)

        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_command = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_command, continued_command)


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the linear velocity."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="unified_command")
    norm: xax.NormType = attrs.field(default="l2")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # get base quat, only yaw.
        # careful to only rotate in z, disregard rx and ry, bad conflict with roll and pitch.
        base_euler = xax.quat_to_euler(trajectory.xquat[:, 1, :])
        base_euler = base_euler.at[:, :2].set(0.0)
        base_z_quat = xax.euler_to_quat(base_euler)

        # robot frame vel from global frame vel
        global_vel = trajectory.qvel[:, :3]
        robot_vel = xax.rotate_vector_by_quat(global_vel, base_z_quat, inverse=True)[:, :2]

        # robot frame vel cmd
        robot_vel_cmd = trajectory.command[self.command_name][:, :2]

        # now compute error. special trick: different kernels for standing and walking.
        zero_cmd_mask = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < 1e-3
        x_vel_error = jnp.abs(robot_vel[:, 0] - robot_vel_cmd[:, 0])
        xy_vel_error = jnp.linalg.norm(robot_vel - robot_vel_cmd, axis=-1)

        # to shift center of mass between feet, we need to allow sidewayse movement for x vel walking and angvel rotating
        # For x command, use x_vel_error
        # For y command, use xy_vel_error
        # For wz command, use no error
        # For zero command, use xy_vel_error
        x_cmd_mask = jnp.abs(robot_vel_cmd[:, 0]) > 1e-3
        y_cmd_mask = jnp.abs(robot_vel_cmd[:, 1]) > 1e-3
        vel_error = jnp.where(
            x_cmd_mask, x_vel_error, jnp.where(y_cmd_mask, xy_vel_error, jnp.where(zero_cmd_mask, xy_vel_error, 0.0))
        )

        error = jnp.where(zero_cmd_mask, vel_error, jnp.square(vel_error))
        return jnp.exp(-error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the heading using quaternion-based error computation."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="unified_command")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        base_yaw = xax.quat_to_euler(trajectory.xquat[:, 1, :])[:, 2]
        base_yaw_cmd = trajectory.command[self.command_name][:, 3]

        base_yaw_quat = xax.euler_to_quat(
            jnp.stack([jnp.zeros_like(base_yaw_cmd), jnp.zeros_like(base_yaw_cmd), base_yaw], axis=-1)
        )
        base_yaw_target_quat = xax.euler_to_quat(
            jnp.stack([jnp.zeros_like(base_yaw_cmd), jnp.zeros_like(base_yaw_cmd), base_yaw_cmd], axis=-1)
        )

        # Compute quaternion error
        quat_error = 1 - jnp.sum(base_yaw_target_quat * base_yaw_quat, axis=-1) ** 2
        return jnp.exp(-quat_error / self.error_scale)


@attrs.define(frozen=True)
class XYOrientationReward(ksim.Reward):
    """Reward for tracking the xy base orientation using quaternion-based error computation."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="unified_command")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        euler_orientation = xax.quat_to_euler(trajectory.xquat[:, 1, :])
        euler_orientation = euler_orientation.at[:, 2].set(0.0)  # ignore yaw
        base_xy_quat = xax.euler_to_quat(euler_orientation)

        commanded_euler = jnp.stack(
            [
                trajectory.command[self.command_name][:, 5],
                trajectory.command[self.command_name][:, 6],
                jnp.zeros_like(trajectory.command[self.command_name][:, 6]),
            ],
            axis=-1,
        )
        base_xy_quat_cmd = xax.euler_to_quat(commanded_euler)

        quat_error = 1 - jnp.sum(base_xy_quat_cmd * base_xy_quat, axis=-1) ** 2
        return jnp.exp(-quat_error / self.error_scale)


@attrs.define(frozen=True)
class FeetPositionObservation(ksim.Observation):
    base_idx: int
    foot_left_idx: int
    foot_right_idx: int

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        base_body_name: str,
        foot_left_body_name: str,
        foot_right_body_name: str,
    ) -> Self:
        base = ksim.get_body_data_idx_from_name(physics_model, base_body_name)
        fl = ksim.get_body_data_idx_from_name(physics_model, foot_left_body_name)
        fr = ksim.get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(base_idx=base, foot_left_idx=fl, foot_right_idx=fr)

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        # get global positions
        base_pos = state.physics_state.data.xpos[self.base_idx]
        left_foot_pos = state.physics_state.data.xpos[self.foot_left_idx]
        right_foot_pos = state.physics_state.data.xpos[self.foot_right_idx]

        base_yaw = xax.quat_to_euler(state.physics_state.data.xquat[self.base_idx, :])[2]
        base_yaw_quat = xax.euler_to_quat(
            jnp.stack([jnp.zeros_like(base_yaw), jnp.zeros_like(base_yaw), base_yaw], axis=-1)
        )

        # transform feet pos to base frame
        relative_left_foot_pos = left_foot_pos - base_pos
        relative_right_foot_pos = right_foot_pos - base_pos
        fl_ndarray = xax.rotate_vector_by_quat(relative_left_foot_pos, base_yaw_quat, inverse=True)
        fr_ndarray = xax.rotate_vector_by_quat(relative_right_foot_pos, base_yaw_quat, inverse=True)

        return jnp.concatenate([fl_ndarray, fr_ndarray], axis=-1)


@attrs.define(frozen=True)
class BaseHeightReward(ksim.Reward):
    """Reward for keeping the base height at the commanded height."""

    error_scale: float = attrs.field(default=0.25)
    standard_height: float = attrs.field(default=0.27)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        current_height = trajectory.xpos[:, 1, 2]  # 1st body, because world is 0. 2nd element is z.
        commanded_height = trajectory.command["unified_command"][:, 4] + self.standard_height

        height_error = jnp.abs(current_height - commanded_height)
        # is_zero_cmd = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < 1e-3
        # height_error = jnp.where(is_zero_cmd, height_error, height_error**2)  # smooth kernel for walking.
        return jnp.exp(-height_error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class FeetAirtimeReward(ksim.StatefulReward):
    """Encourages reasonable step frequency by rewarding long swing phases and penalizing quick stepping."""

    scale: float = 1.0
    ctrl_dt: float = 0.02
    touchdown_penalty: float = 0.4
    scale_by_curriculum: bool = False

    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        # initial left and right airtime
        return jnp.array([0.0, 0.0])

    def _airtime_sequence(self, initial_airtime: Array, contact_bool: Array, done: Array) -> tuple[Array, Array]:
        """Returns an array with the airtime (in seconds) for each timestep."""

        def _body(time_since_liftoff: Array, is_contact: Array) -> tuple[Array, Array]:
            new_time = jnp.where(is_contact, 0.0, time_since_liftoff + self.ctrl_dt)
            return new_time, new_time

        # or with done to reset the airtime counter when the episode is done
        contact_or_done = jnp.logical_or(contact_bool, done)
        carry, airtime = jax.lax.scan(_body, initial_airtime, contact_or_done)
        return carry, airtime

    def get_reward_stateful(self, traj: ksim.Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        left_contact = jnp.where(traj.obs["sensor_observation_left_foot_touch"] > 0.1, True, False)[:, 0]
        right_contact = jnp.where(traj.obs["sensor_observation_right_foot_touch"] > 0.1, True, False)[:, 0]

        # airtime counters
        left_carry, left_air = self._airtime_sequence(reward_carry[0], left_contact, traj.done)
        right_carry, right_air = self._airtime_sequence(reward_carry[1], right_contact, traj.done)

        reward_carry = jnp.array([left_carry, right_carry])

        # touchdown boolean (0→1 transition)
        def touchdown(c: Array) -> Array:
            prev = jnp.concatenate([jnp.array([False]), c[:-1]])
            return jnp.logical_and(c, jnp.logical_not(prev))

        td_l = touchdown(left_contact)
        td_r = touchdown(right_contact)

        left_air_shifted = jnp.roll(left_air, 1)
        right_air_shifted = jnp.roll(right_air, 1)

        left_feet_airtime_reward = (left_air_shifted - self.touchdown_penalty) * td_l.astype(jnp.float32)
        right_feet_airtime_reward = (right_air_shifted - self.touchdown_penalty) * td_r.astype(jnp.float32)

        reward = left_feet_airtime_reward + right_feet_airtime_reward

        # standing mask
        is_zero_cmd = jnp.linalg.norm(traj.command["unified_command"][:, :3], axis=-1) < 1e-3
        reward = jnp.where(is_zero_cmd, 0.0, reward)

        return reward, reward_carry


@attrs.define(frozen=True, kw_only=True)
class JointPositionPenalty(ksim.JointDeviationPenalty):
    @classmethod
    def create_from_names(
        cls,
        names: list[str],
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
        error_scale: float = 0.1,
    ) -> Self:
        zeros = {k: v for k, v, _ in JOINT_BIASES}
        weights = {k: v for k, _, v in JOINT_BIASES}
        joint_targets = [zeros[name] for name in names]
        joint_weights = [weights[name] for name in names]

        return cls.create(
            physics_model=physics_model,
            joint_names=tuple(names),
            joint_targets=tuple(joint_targets),
            joint_weights=tuple(joint_weights),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class ArmPositionReward(JointPositionPenalty):
    error_scale: float = attrs.field(default=0.1)

    @classmethod
    def create_reward(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = 0.05,
        error_scale: float = 0.1,
        scale_by_curriculum: bool = False,
    ) -> Self:
        reward = cls.create_from_names(
            names=[
                "right_shoulder_pitch",
                "right_shoulder_roll",
                "right_elbow_roll",
                "right_gripper_roll",
                "left_shoulder_pitch",
                "left_shoulder_roll",
                "left_elbow_roll",
                "left_gripper_roll",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
            error_scale=error_scale,
        )
        return reward

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        error = super().get_reward(trajectory)
        reward = jnp.exp(-error / self.error_scale)
        return reward


@attrs.define(frozen=True, kw_only=True)
class FeetOrientationReward(ksim.Reward):
    """Encourage both feet to stay level."""

    left_idx: int
    right_idx: int
    target_rp: tuple[float, float] = (0.0, 0.0)
    error_scale: float = 0.25
    scale: float = 1.0

    @classmethod
    def create(
        cls,
        physics_model: ksim.PhysicsModel,
        *,
        left_name: str = "Left_Foot",
        right_name: str = "Right_Foot",
        target_rp: tuple[float, float] = (0.0, 0.0),
        error_scale: float = 0.25,
        scale: float = 0.1,
    ) -> Self:
        """Resolve body indices and build the reward instance."""
        left_id = ksim.get_body_data_idx_from_name(physics_model, left_name)
        right_id = ksim.get_body_data_idx_from_name(physics_model, right_name)
        return cls(
            left_idx=left_id,
            right_idx=right_id,
            target_rp=target_rp,
            error_scale=error_scale,
            scale=scale,
        )

    def get_reward(self, traj: ksim.Trajectory) -> jnp.ndarray:
        # if walking, minimize roll and pitch error
        left_rp = xax.quat_to_euler(traj.xquat[:, self.left_idx, :])[:, :2]
        right_rp = xax.quat_to_euler(traj.xquat[:, self.right_idx, :])[:, :2]

        left_rp_quat = xax.euler_to_quat(jnp.concatenate([left_rp, jnp.zeros_like(left_rp[:, :1])], axis=-1))
        right_rp_quat = xax.euler_to_quat(jnp.concatenate([right_rp, jnp.zeros_like(right_rp[:, :1])], axis=-1))

        tgt = xax.euler_to_quat(jnp.array([0, 0, 0]))
        left_rp_error = 1 - jnp.sum(left_rp_quat * tgt, axis=-1) ** 2
        right_rp_error = 1 - jnp.sum(right_rp_quat * tgt, axis=-1) ** 2
        rp_error = left_rp_error + right_rp_error

        # if standing, minimize roll, pitch, AND yaw error
        left_quat = traj.xquat[:, self.left_idx, :]
        right_quat = traj.xquat[:, self.right_idx, :]

        heading = xax.quat_to_euler(traj.xquat[:, 1, :])[:, 2]
        tgt = xax.euler_to_quat(jnp.stack([jnp.zeros_like(heading), jnp.zeros_like(heading), heading], axis=-1))
        left_yaw_error = 1 - jnp.sum(tgt * left_quat, axis=-1) ** 2
        right_yaw_error = 1 - jnp.sum(tgt * right_quat, axis=-1) ** 2
        rpy_error = left_yaw_error + right_yaw_error

        is_zero_cmd = jnp.linalg.norm(traj.command["unified_command"][:, :3], axis=-1) < 1e-3
        total_error = jnp.where(is_zero_cmd, rpy_error, rp_error)

        return jnp.exp(-total_error / self.error_scale)


@attrs.define(frozen=True)
class StandingFeetPositionReward(ksim.Reward):
    """Reward for keeping the feet next to each other when standing still."""

    error_scale: float = attrs.field(default=0.25)
    stance_width: float = attrs.field(default=0.3)
    base_idx: int = attrs.field(default=1)
    foot_left_idx: int = attrs.field(default=0)
    foot_right_idx: int = attrs.field(default=0)

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        base_body_name: str,
        foot_left_body_name: str,
        foot_right_body_name: str,
        scale: float,
        error_scale: float,
        stance_width: float,
    ) -> Self:
        base = ksim.get_body_data_idx_from_name(physics_model, base_body_name)
        fl = ksim.get_body_data_idx_from_name(physics_model, foot_left_body_name)
        fr = ksim.get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(
            base_idx=base,
            foot_left_idx=fl,
            foot_right_idx=fr,
            scale=scale,
            error_scale=error_scale,
            stance_width=stance_width,
        )

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # get global positions
        global_l_foot_pos = trajectory.xpos[:, self.foot_left_idx]
        global_r_foot_pos = trajectory.xpos[:, self.foot_right_idx]
        base_pos = trajectory.xpos[:, self.base_idx]
        base_quat = trajectory.xquat[:, self.base_idx, :]

        # compute feet pos in base frame
        l_foot_pos = xax.rotate_vector_by_quat((global_l_foot_pos - base_pos), base_quat, inverse=True)
        r_foot_pos = xax.rotate_vector_by_quat((global_r_foot_pos - base_pos), base_quat, inverse=True)

        # calculate stance errors
        stance_x_error = jnp.abs(l_foot_pos[:, 0] - r_foot_pos[:, 0])
        stance_y_error = jnp.abs(jnp.abs(l_foot_pos[:, 1] - r_foot_pos[:, 1]) - self.stance_width)
        stance_error = stance_x_error + stance_y_error

        # only apply reward for standing
        zero_cmd_mask = jnp.linalg.norm(trajectory.command["unified_command"][:, :3], axis=-1) < 1e-3
        error = jnp.where(zero_cmd_mask, stance_error, 0.0)
        reward = jnp.exp(-error / self.error_scale)
        return reward


@attrs.define(frozen=True, kw_only=True)
class SingleFootContactReward(ksim.StatefulReward):
    """Reward having one and only one foot in contact with the ground, while walking.

    Allows for small grace period when both feet are in contact for less jumpy gaits.
    """

    scale: float = 1.0
    ctrl_dt: float = 0.02
    grace_period: float = 0.2  # seconds

    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        return jnp.array([0.0])

    def get_reward_stateful(self, traj: ksim.Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        left_contact = jnp.where(traj.obs["sensor_observation_left_foot_touch"] > 0.1, True, False)[:, 0]
        right_contact = jnp.where(traj.obs["sensor_observation_right_foot_touch"] > 0.1, True, False)[:, 0]
        single = jnp.logical_xor(left_contact, right_contact)

        def _body(time_since_single_contact: Array, is_single_contact: Array) -> tuple[Array, Array]:
            new_time = jnp.where(is_single_contact, 0.0, time_since_single_contact + self.ctrl_dt)
            return new_time, new_time

        carry, time_since_single_contact = jax.lax.scan(_body, reward_carry, single)
        single_contact_grace = time_since_single_contact < max(self.ctrl_dt, self.grace_period)
        is_zero_cmd = jnp.linalg.norm(traj.command["unified_command"][:, :3], axis=-1) < 1e-3
        reward = jnp.where(is_zero_cmd, 1.0, single_contact_grace[:, 0])
        return reward, carry

@attrs.define(frozen=True, kw_only=True)
class ContactForcePenalty(ksim.Reward):
    """Penalises vertical forces above threshold."""

    scale: float = -1.0
    max_contact_force: float = 350.0
    sensor_names: tuple[str, ...]

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        forces = jnp.stack([traj.obs[n] for n in self.sensor_names], axis=-1)
        cost = jnp.clip(jnp.abs(forces[:, 2, :]) - self.max_contact_force, 0)
        return jnp.sum(cost, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class ImuOrientationObservation(ksim.StatefulObservation):
    """Observes the IMU orientation, back spun in yaw heading, as commanded.

    This provides an approximation of reading the IMU orientation from
    the IMU on the physical robot, backspun by commanded heading. The `framequat_name` should be the name of
    the framequat sensor attached to the IMU.

    Example: if yaw cmd = 3.14, and IMU reading is [0, 0, 0, 1], then back spun IMU heading obs is [1, 0, 0, 0]

    The policy learns to keep the IMU heading obs around [1, 0, 0, 0].
    """

    framequat_idx_range: tuple[int, int | None] = attrs.field()
    lag_range: tuple[float, float] = attrs.field(
        default=(0.01, 0.1),
        validator=attrs.validators.deep_iterable(
            attrs.validators.and_(
                attrs.validators.ge(0.0),
                attrs.validators.lt(1.0),
            ),
        ),
    )
    bias_euler: tuple[float, float, float] = attrs.field(
        default=(0.0, 0.0, 0.0),
        validator=attrs.validators.deep_iterable(
            attrs.validators.and_(
                attrs.validators.ge(0.0),
                attrs.validators.le(math.pi),
            ),
        ),
    )

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        noise: float = 0.0,
        framequat_name: str,
        lag_range: tuple[float, float] = (0.01, 0.1),
        bias_euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Self:
        """Create an IMU orientation observation from a physics model.

        Args:
            physics_model: MuJoCo physics model
            framequat_name: The name of the framequat sensor
            lag_range: The range of EMA factors to use, to approximate the
                variation in the amount of smoothing of the Kalman filter.
            noise: The observation noise
            bias_euler: The bias in euler angles, in roll, pitch, yaw.
        """
        sensor_name_to_idx_range = ksim.get_sensor_data_idxs_by_name(physics_model)
        if framequat_name not in sensor_name_to_idx_range:
            options = "\n".join(sorted(sensor_name_to_idx_range.keys()))
            raise ValueError(f"{framequat_name} not found in model. Available:\n{options}")

        return cls(
            framequat_idx_range=sensor_name_to_idx_range[framequat_name],
            lag_range=lag_range,
            bias_euler=bias_euler,
            noise=noise,
        )

    def initial_carry(self, physics_state: ksim.PhysicsState, rng: PRNGKeyArray) -> tuple[Array, Array, Array]:
        lrng, brng = jax.random.split(rng, 2)
        minval, maxval = self.lag_range
        lag = jax.random.uniform(lrng, (1,), minval=minval, maxval=maxval)

        bias_range = jnp.array(self.bias_euler)
        bias = jax.random.uniform(brng, (3,), minval=-bias_range, maxval=bias_range)
        bias_quat = xax.euler_to_quat(bias)

        return jnp.zeros((4,)), lag, bias_quat

    def observe_stateful(
        self,
        state: ksim.ObservationInput,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[Array, tuple[Array, Array]]:
        x, lag, bias = state.obs_carry

        framequat_start, framequat_end = self.framequat_idx_range
        framequat_data = state.physics_state.data.sensordata[framequat_start:framequat_end].ravel()

        # apply bias noise
        framequat_data = rotate_quat_by_quat(framequat_data, bias)

        # get heading cmd
        heading_yaw_cmd = state.commands["unified_command"][3]

        # spin back
        heading_yaw_cmd_quat = xax.euler_to_quat(jnp.array([0.0, 0.0, heading_yaw_cmd]))
        backspun_framequat = rotate_quat_by_quat(framequat_data, heading_yaw_cmd_quat, inverse=True)
        # ensure positive quat hemisphere
        backspun_framequat = jnp.where(backspun_framequat[..., 0] < 0, -backspun_framequat, backspun_framequat)

        # Get current Kalman filter state
        x = x * lag + backspun_framequat * (1 - lag)

        return x, (x, lag, bias)


@attrs.define(frozen=True)
class BaseHeightObservation(ksim.Observation):
    """Single-scalar z of body-1 (the robot base)."""

    def observe(self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        # body 0 is world; body 1 is the floating base
        return state.physics_state.data.xpos[1, 2:]


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 2,  # mean and std
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Split into means and stds
        mean_n = out_n[..., : self.num_outputs]
        std_n = out_n[..., self.num_outputs :]

        # Softplus and clip to ensure positive standard deviations
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means
        mean_n = mean_n + jnp.array([v for _, v, _ in JOINT_BIASES])

        # Create diagonal gaussian distribution
        dist_n = distrax.MultivariateNormalDiag(loc=mean_n, scale_diag=std_n)

        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_inputs = NUM_CRITIC_INPUTS
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        hidden_size: int,
        depth: int,
    ) -> None:
        self.actor = Actor(
            key,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=1.0,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.critic = Critic(
            key,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class ZbotWalkingTaskConfig(ksim.PPOConfig):
    """Config for the Z-Bot walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=2.0,
        help="Maximum gradient norm for clipping.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )
    mirror_loss_scale: float = xax.field(
        value=0.01,
        help="Scale for the mirror loss",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=0,
        help="The body id to track with the render camera.",
    )
    render_distance: float = xax.field(
        value=0.8,
        help="The distance to the render camera.",
    )


class ZbotWalkingTask(ksim.PPOTask[ZbotWalkingTaskConfig]):
    delta_max_j: jnp.ndarray | None = None  # set later in get_actuators

    def get_optimizer(self) -> optax.GradientTransformation:
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            (
                optax.adam(self.config.learning_rate)
                if self.config.adam_weight_decay == 0.0
                else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
            ),
        )

        return optimizer

    def mirror_joints(self, j: Array) -> Array:
        """Mirror the joint positions/velocities from left to right and vice versa."""
        assert j.shape[-1] == NUM_JOINTS, f"Joints must be {NUM_JOINTS}-dimensional"
        j_m = jnp.zeros_like(j)

        # Mirror legs (first 12 joints)
        # Right leg (0-5) to left leg (6-11)
        j_m = j_m.at[..., 0:6].set(j[..., 6:12])
        # Left leg (6-11) to right leg (0-5)
        j_m = j_m.at[..., 6:12].set(j[..., 0:6])

        # Mirror arms (next 8 joints)
        # Right arm (12-15) to left arm (16-19)
        j_m = j_m.at[..., 12:16].set(j[..., 16:20])
        # Left arm (16-19) to right arm (12-15)
        j_m = j_m.at[..., 16:20].set(j[..., 12:16])

        # Negate roll and yaw angles while preserving pitch
        # For legs: yaw=0,6; roll=1,7; pitch=2,8; knee=3,9; ankle_pitch=4,10; ankle_roll=5,11
        j_m = j_m.at[..., 0].multiply(-1)  # right hip yaw
        j_m = j_m.at[..., 1].multiply(-1)  # right hip roll
        j_m = j_m.at[..., 5].multiply(-1)  # right ankle roll
        j_m = j_m.at[..., 6].multiply(-1)  # left hip yaw
        j_m = j_m.at[..., 7].multiply(-1)  # left hip roll
        j_m = j_m.at[..., 11].multiply(-1)  # left ankle roll

        # For arms: pitch=12,16; roll=13,17; elbow=14,18; gripper=15,19
        j_m = j_m.at[..., 13].multiply(-1)  # right shoulder roll
        j_m = j_m.at[..., 14].multiply(-1)  # right elbow roll
        j_m = j_m.at[..., 17].multiply(-1)  # left shoulder roll
        j_m = j_m.at[..., 18].multiply(-1)  # left elbow roll

        return j_m

    def mirror_obs(self, obs: xax.FrozenDict[str, Array]) -> xax.FrozenDict[str, Array]:
        """Mirror the observations used by the actor."""
        joint_pos_n_m = self.mirror_joints(obs["joint_position_observation"])
        joint_vel_n_m = self.mirror_joints(obs["joint_velocity_observation"])
        imu_quat_4_m = jnp.concatenate(
            [
                obs["imu_orientation_observation"][..., :1],  # w
                -obs["imu_orientation_observation"][..., 1:2],  # x
                obs["imu_orientation_observation"][..., 2:3],  # y
                -obs["imu_orientation_observation"][..., 3:],  # z
            ],
            axis=-1,
        )

        obs_m = {
            "joint_position_observation": joint_pos_n_m,
            "joint_velocity_observation": joint_vel_n_m,
            "imu_orientation_observation": imu_quat_4_m,
        }
        return obs_m

    def mirror_cmd(self, cmd: xax.FrozenDict[str, Array]) -> xax.FrozenDict[str, Array]:
        """Mirror the commands."""
        cmd_u = cmd["unified_command"]
        cmd_u_m = jnp.concatenate(
            [
                cmd_u[..., :1],  # vx
                -cmd_u[..., 1:2],  # vy
                -cmd_u[..., 2:3],  # wz
                -cmd_u[..., 3:4],  # heading
                cmd_u[..., 4:5],  # base height
                -cmd_u[..., 5:6],  # rx
                cmd_u[..., 6:7],  # ry
            ],
            axis=-1,
        )
        cmd = {"unified_command": cmd_u_m}
        return cmd

    def unmirror_action(self, action: Array) -> Array:
        """Unmirror the action by applying the same mirroring operation."""
        return self.mirror_joints(action)

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("zbot", name="robot"))
        model = mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")
        names_to_idxs = ksim.get_geom_data_idx_by_name(model)
        model.geom_priority[names_to_idxs["floor"]] = 2.0
        return model

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("zbot"))
        # Ensure we're returning a proper RobotURDFMetadataOutput
        if not isinstance(metadata, Metadata):
            raise ValueError("Metadata is not a Metadata")
        return metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.75, scale_upper=1.25),
            ksim.JointDampingRandomizer(scale_lower=0.5, scale_upper=2.5),
            ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-3), scale_upper=math.radians(3)),
            ksim.FloorFrictionRandomizer.from_geom_name(
                model=physics_model, floor_geom_name="floor", scale_lower=0.3, scale_upper=1.5
            ),
            # 1σ ≈ 1.5°, gives ~99.7% within 4.5°
            # enable yaw randomization with 1σ ≈ 1°
            # 5mm standard deviation
            # ksim.IMUAlignmentRandomizer(
            #     site_name="imu_site", tilt_std_rad=math.radians(5), yaw_std_rad=math.radians(1.0), translate_std_m=0.005
            # ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            # ksim.PushEvent(
            #     x_linvel=0.1,
            #     y_linvel=0.1,
            #     z_linvel=0.05,
            #     x_angvel=0.0,
            #     y_angvel=0.0,
            #     z_angvel=0.0,
            #     vel_range=(0.05, 0.15),
            #     interval_range=(2.0, 4.0),
            # ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v, _ in JOINT_BIASES}, scale=0.1),
            ksim.RandomJointVelocityReset(),
            ksim.RandomHeadingReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        obs_list = [
            ksim.JointPositionObservation(noise=math.radians(2)),
            ksim.JointVelocityObservation(noise=math.radians(10)),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            BaseHeightObservation(),
            ImuOrientationObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.1),
                bias_euler=(0.05, 0.05, 0.0),  # roll, pitch, yaw
                noise=math.radians(1),
            ),
            ksim.ActuatorAccelerationObservation(),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(0),
            ),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_touch", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_touch", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_force", noise=0.0),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_force", noise=0.0),
            FeetPositionObservation.create(
                physics_model=physics_model,
                base_body_name="base",
                foot_left_body_name="Right_Foot",
                foot_right_body_name="Left_Foot",
            ),
        ]

        return obs_list

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            UnifiedCommand(
                vx_range=(-0.3, 0.3),  # m/s
                vy_range=(-0.2, 0.2),  # m/s
                wz_range=(-0.5, 0.5),  # rad/s
                bh_range=(0.0, 0.0),  # m # disabled for now, does not work on this robot. reward conflicts
                bh_standing_range=(-0.2, 0.0),  # m
                rx_range=(-0.3, 0.3),  # rad
                ry_range=(-0.3, 0.3),  # rad
                ctrl_dt=self.config.ctrl_dt,
                switch_prob=self.config.ctrl_dt / 25,  # once per x seconds
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            # cmd
            LinearVelocityTrackingReward(scale=0.3, error_scale=0.05),
            AngularVelocityTrackingReward(scale=0.1, error_scale=0.005),
            XYOrientationReward(scale=0.1, error_scale=0.002),
            # shaping
            SingleFootContactReward(scale=0.3, ctrl_dt=self.config.ctrl_dt, grace_period=0.1),
            FeetAirtimeReward(scale=1.0, ctrl_dt=self.config.ctrl_dt, touchdown_penalty=0.4),
            ArmPositionReward.create_reward(physics_model, scale=0.05, error_scale=0.05),
            BaseHeightReward(scale=0.05, error_scale=0.02, standard_height=0.27),  # only works on scene 'smooth'
            FeetOrientationReward.create(
                physics_model,
                target_rp=(0.0, 0.0),
                error_scale=0.02,
                scale=0.05,
            ),
            StandingFeetPositionReward.create(
                physics_model=physics_model,
                base_body_name="base",
                foot_left_body_name="Right_Foot",
                foot_right_body_name="Left_Foot",
                scale=0.02,
                error_scale=0.01,
                stance_width=0.10
            ),
            # ksim.ActionVelocityPenalty(scale=-2.0, scale_by_curriculum=True),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.05, unhealthy_z_upper=0.5),
            ksim.NotUprightTermination(max_radians=math.radians(60)),
            ksim.EpisodeLengthTermination(max_length_sec=24),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.LinearCurriculum(
            step_size=1,
            step_every_n_epochs=1,
            min_level=1.0,  # disable curriculum
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        return Model(
            key,
            num_inputs=NUM_ACTOR_INPUTS,
            num_outputs=NUM_JOINTS,
            min_std=0.03,
            max_std=1.0,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
        rng: PRNGKeyArray,
    ) -> tuple[distrax.Distribution, Array]:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_quat_4 = observations["imu_orientation_observation"]
        
        cmd = commands["unified_command"]
        zero_cmd = (jnp.linalg.norm(cmd[..., :3], axis=-1) < 1e-3)[..., None]
        lin_vel_cmd = cmd[..., :2]
        ang_vel_cmd = cmd[..., 2:3]
        base_height_cmd = cmd[..., 4:5]
        base_roll_pitch_cmd = cmd[..., 5:7]

        obs_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                imu_quat_4,  # 4
                zero_cmd,  # 1
                lin_vel_cmd,  # 2
                ang_vel_cmd,  # 1
                base_height_cmd,  # 1
                base_roll_pitch_cmd,  # 2
            ],
            axis=-1,
        )

        action, carry = model.forward(obs_n, carry)

        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        joint_pos_n = observations["joint_position_observation"]  # should really be the noise free versions
        joint_vel_n = observations["joint_velocity_observation"]
        imu_quat_4 = observations["imu_orientation_observation"]
        cmd = commands["unified_command"]
        zero_cmd = (jnp.linalg.norm(cmd[..., :3], axis=-1) < 1e-3)[..., None]
        lin_vel_cmd = cmd[..., :2]
        ang_vel_cmd = cmd[..., 2:3]
        base_height_cmd = cmd[..., 4:5]
        base_roll_pitch_cmd = cmd[..., 5:7]

        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        left_touch = observations["sensor_observation_left_foot_touch"]
        right_touch = observations["sensor_observation_right_foot_touch"]
        feet_position_6 = observations["feet_position_observation"]
        base_position_3 = observations["base_position_observation"]
        base_orientation_4 = observations["base_orientation_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        base_lin_vel_3 = observations["base_linear_velocity_observation"]
        base_ang_vel_3 = observations["base_angular_velocity_observation"]
        actuator_force_n = observations["actuator_force_observation"]
        base_height = observations["base_height_observation"]

        obs_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n / 10.0,  # NUM_JOINTS
                imu_quat_4,  # 4
                zero_cmd,  # 1
                lin_vel_cmd,  # 2
                ang_vel_cmd,  # 1
                base_height_cmd,  # 1
                base_roll_pitch_cmd,  # 2
                # privileged observations
                imu_gyro_3,
                left_touch,
                right_touch,
                feet_position_6,
                base_position_3,
                base_orientation_4,
                com_inertia_n,
                com_vel_n,
                base_lin_vel_3,
                base_ang_vel_3,
                actuator_force_n / 100.0,
                base_height,
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        step_keys = jax.random.split(rng, trajectory.action.shape[0])

        def scan_fn(
            actor_critic_carry: tuple[Array, Array, Array],
            scan_inputs: tuple[ksim.Trajectory, PRNGKeyArray],
        ) -> tuple[tuple[Array, Array, Array], ksim.PPOVariables]:
            transition, step_key = scan_inputs
            actor_carry, critic_carry, actor_mirror_carry = actor_critic_carry
            actor_dist, next_actor_carry = self.run_actor(
                model=model.actor,
                observations=transition.obs,
                commands=transition.command,
                carry=actor_carry,
                rng=step_key,
            )
            log_probs = actor_dist.log_prob(transition.action)
            assert isinstance(log_probs, Array)
            value, next_critic_carry = self.run_critic(
                model=model.critic,
                observations=transition.obs,
                commands=transition.command,
                carry=critic_carry,
            )

            # compute mirror loss
            mirrored_actor_dist, next_actor_mirror_carry = self.run_actor(
                model=model.actor,
                observations=self.mirror_obs(transition.obs),
                commands=self.mirror_cmd(transition.command),
                carry=actor_mirror_carry,
                rng=step_key,
            )
            unmirrored_actor_dist = self.unmirror_action(mirrored_actor_dist.mean())
            mse_loss = jnp.mean((actor_dist.mean() - unmirrored_actor_dist) ** 2)
            mirror_loss = jnp.mean(mse_loss) * self.config.mirror_loss_scale

            transition_ppo_variables = ksim.PPOVariables(
                log_probs=jnp.expand_dims(log_probs, axis=0),
                values=value.squeeze(-1),
                entropy=jnp.expand_dims(actor_dist.entropy(), axis=0),
                action_std=actor_dist.stddev(),
                aux_losses={"mirror_loss": mirror_loss},
            )

            next_carry = jax.tree.map(
                lambda x, y: jnp.where(transition.done, x, y),
                (
                    jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
                    jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
                    jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
                ),
                (next_actor_carry, next_critic_carry, next_actor_mirror_carry),
            )

            return next_carry, transition_ppo_variables

        # Add a third carry for the mirror actor
        model_carry = model_carry + (jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),)
        next_model_carry, ppo_variables = jax.lax.scan(scan_fn, model_carry, (trajectory, step_keys))
        return ppo_variables, next_model_carry[:-1]

    def get_initial_model_carry(self, model: Model, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry
        rng, actor_rng = jax.random.split(rng)
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
            rng=actor_rng,
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        return ksim.Action(
            action=action_j,
            carry=(actor_carry, critic_carry_in),
        )


if __name__ == "__main__":
    ZbotWalkingTask.launch(
        ZbotWalkingTaskConfig(
            # Training parameters.
            num_envs=4096,
            batch_size=256,
            learning_rate=5e-4,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=2.0,
            gamma=0.95,
            lam=0.94,
            entropy_coef=0.001,
            mirror_loss_scale=1.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            # sim2real parameters.
            action_latency_range=(0.003, 0.01),
            drop_action_prob=0.05,
            # Checkpointing parameters.
            save_every_n_seconds=5 * 60,
            valid_every_n_steps=100,
            valid_every_n_seconds=None,
            render_full_every_n_seconds=10,
            render_azimuth=145.0,
        ),
    )
