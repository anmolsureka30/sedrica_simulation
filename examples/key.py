import time
import math
import numpy as np
import pyglet
from pyglet.window import key

# Import the repo env
from f110_gym.envs.f110_env import F110Env

# ---------------- USER CONFIG ----------------
MAP_PATH = "/Users/anmolsureka/Documents/Coding/sedrica/f1tenth_gym/gym/f110_gym/envs/maps/vegas"
MAP_EXT = ".png"
TIMESTEP = 0.02
FRICTION = 0.995  # decay per step of setpoint
# ------------------------------------------------


class DiffDriveKeyboardEnv(F110Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # key handling (attached once renderer exists)
        self.keys = key.KeyStateHandler()
        self._keys_attached = False

        # diff-drive setpoints
        self.v = 0.0
        self.omega = 0.0
        self.max_v = 7.0
        self.max_omega = 4.5

        # increments (m/s per second, rad/s per second)
        self._v_step = 3.0
        self._omega_step = 4.0

    def vomega_to_ack(self, v, omega):
        """Map diff-drive (v, omega) to (steer, speed) expected by the base env,
        with limits to avoid unrealistic spins at high speed.
        """
        try:
            wb = float(self.sim.agents[0].params['lf'] + self.sim.agents[0].params['lr'])
        except Exception:
            wb = 0.26

        # if nearly stopped, allow direct control
        if abs(v) < 1e-3:
            return 0.0, float(v)

        # --- NEW FIXES ---
        # Limit yaw rate (omega) realistically
        max_yaw_rate = np.deg2rad(60)  # 60 deg/s
        omega = np.clip(omega, -max_yaw_rate, max_yaw_rate)

        # Compute steering using bicycle model
        steer = math.atan((omega * wb) / v)

        # Limit maximum steering angle (like a real F1Tenth car)
        max_steer = np.deg2rad(30)  # 30° max steering
        steer = np.clip(steer, -max_steer, max_steer)

        # Reduce steering authority as speed increases
        steer *= 1.0 / (1.0 + 0.1 * abs(v))  # smoother at high speed

        return float(steer), float(v)

    def _maybe_attach_keys(self):
        """Attach key handler to the renderer window once."""
        if getattr(self, "renderer", None) is None:
            return
        if not self._keys_attached:
            try:
                self.renderer.push_handlers(self.keys)
                self._keys_attached = True
                print("[info] Key handler attached to renderer.")
            except Exception as e:
                print("[warn] Failed to attach key handler:", e)

    def step_keyboard(self):
        """Read keyboard setpoints, apply friction, convert to ackermann and step."""
        self._maybe_attach_keys()
        dt = getattr(self, "timestep", TIMESTEP)

        if self._keys_attached:
            if self.keys[key.UP]:
                self.v = min(self.max_v, self.v + self._v_step * dt)
            if self.keys[key.DOWN]:
                self.v = max(-self.max_v/2.0, self.v - self._v_step * dt)
            if self.keys[key.LEFT]:
                self.omega = max(-self.max_omega, self.omega - self._omega_step * dt)
            if self.keys[key.RIGHT]:
                self.omega = min(self.max_omega, self.omega + self._omega_step * dt)
            if self.keys[key.SPACE]:
                # immediate brake
                self.v = 0.0
                self.omega = 0.0
           

        # friction on setpoints (user must hold keys)
        self.v *= FRICTION
        self.omega *= FRICTION

        steer, speed = self.vomega_to_ack(self.v, self.omega)
        action = np.array([[steer, speed]], dtype=np.float32)

        obs, reward, done, info = super().step(action)
        return obs, reward, done, info


# ------------------- runner -------------------
def main():
    env = DiffDriveKeyboardEnv(map=MAP_PATH, map_ext=MAP_EXT, num_agents=1, timestep=TIMESTEP)

    # Reset to get initial poses
    try:
        start_pose = np.array([[env.start_xs[0], env.start_ys[0], env.start_thetas[0]]])
        obs, _, _, _ = env.reset(start_pose)
    except Exception:
        obs, _, _, _ = env.reset()

    # Force a first render so the renderer object exists
    env.render()
    env.renderer.push_handlers(env.keys)
    env._keys_attached = True
    print("[info] Key handler attached manually after render")
    # Main interactive loop
    DONE = False
    while not DONE:
        if env.renderer is not None and hasattr(env.renderer, 'window'):
            env.renderer.window.dispatch_events() 
        obs, reward, DONE, info = env.step_keyboard()
        env.render(mode='human')
        pyglet.clock.tick()
        time.sleep(0.01)

    env.close()


if __name__ == "__main__":
    main()