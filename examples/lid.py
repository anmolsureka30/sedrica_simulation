import math
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
from PIL import Image
from f110_gym.envs.f110_env import F110Env

# ---------------- USER CONFIG ----------------
MAP_PATH = "/Users/anmolsureka/Documents/Coding/sedrica/f1tenth_gym/gym/f110_gym/envs/maps/vegas"
MAP_EXT = ".png"
TIMESTEP = 0.02
FRICTION = 0.995
LIDAR_ANGLE_RES = np.deg2rad(1.0)
LIDAR_MAX_DIST = 5.0
LIDAR_MIN_DIST = 0.05
LIDAR_DISPLAY_SCALE = 50  # Scaling LIDAR for visualization
# --------------------------------------------

class Lidar2D:
    def __init__(self, map_path, map_ext, angle_res=LIDAR_ANGLE_RES, max_dist=LIDAR_MAX_DIST):
        # Load map
        self.map_img = Image.open(map_path + map_ext).convert("L")
        self.map_data = np.array(self.map_img)
        self.height, self.width = self.map_data.shape
        self.angle_res = angle_res
        self.max_dist = max_dist
        self.min_dist = LIDAR_MIN_DIST

        # LIDAR window
        self.window = pyglet.window.Window(600, 600, caption="LIDAR View")
        self.num_beams = int(2 * np.pi / self.angle_res)
        self.distances = np.full(self.num_beams, self.max_dist, dtype=np.float32)
        self.car_x, self.car_y, self.car_theta = 0.0, 0.0, 0.0

        # Schedule redraw at 60 FPS
        pyglet.clock.schedule_interval(self.redraw, 1/60.0)
        self.window.push_handlers(self)

    def world_to_map(self, x, y):
        px = int(x * 50)
        py = int(y * 50)
        return px, self.height - py

    def scan(self, car_x, car_y, car_theta):
        """Update distances based on car position and orientation"""
        self.car_x, self.car_y, self.car_theta = car_x, car_y, car_theta
        cx, cy = self.world_to_map(car_x, car_y)
        distances = np.full(self.num_beams, self.max_dist, dtype=np.float32)

        for i in range(self.num_beams):
            angle = car_theta + i * self.angle_res
            for d in np.linspace(self.min_dist, self.max_dist, 500):
                mx = int(cx + d * 50 * math.cos(angle))
                my = int(cy + d * 50 * math.sin(angle))
                if mx < 0 or mx >= self.width or my < 0 or my >= self.height:
                    distances[i] = d
                    break
                if self.map_data[my, mx] < 128:  # obstacle
                    distances[i] = d
                    break

        self.distances = distances
        return distances

    def redraw(self, dt):
        """Draw LIDAR rays instead of points"""
        cx, cy = 300, 300  # center of LIDAR window
        self.window.clear()
        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glBegin(gl.GL_LINES)
        for i, d in enumerate(self.distances):
            angle = self.car_theta + i * self.angle_res
            x_end = cx + d * LIDAR_DISPLAY_SCALE * math.cos(angle)
            y_end = cy + d * LIDAR_DISPLAY_SCALE * math.sin(angle)
            gl.glVertex2f(cx, cy)
            gl.glVertex2f(x_end, y_end)
        gl.glEnd()

    def on_close(self):
        pyglet.app.exit()


class DiffDriveKeyboardEnv(F110Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = key.KeyStateHandler()
        self.v, self.omega = 0.0, 0.0
        self.max_v, self.max_omega = 7.0, 4.5
        self._v_step, self._omega_step = 3.0, 4.0
        self.lidar = Lidar2D(MAP_PATH, MAP_EXT)
        self._keys_attached = False
        self.lidar_label = None

    def vomega_to_ack(self, v, omega):
        try:
            wb = float(self.sim.agents[0].params['lf'] + self.sim.agents[0].params['lr'])
        except:
            wb = 0.26
        if abs(v) < 1e-3:
            return 0.0, float(v)
        max_yaw_rate = np.deg2rad(60)
        omega = np.clip(omega, -max_yaw_rate, max_yaw_rate)
        steer = math.atan((omega * wb) / v)
        max_steer = np.deg2rad(30)
        steer = np.clip(steer, -max_steer, max_steer)
        steer *= 1.0 / (1.0 + 0.1 * abs(v))
        return float(steer), float(v)

    def _maybe_attach_keys(self):
        if getattr(self, "renderer", None) is None:
            return
        if not self._keys_attached:
            try:
                self.renderer.push_handlers(self.keys)
                self._keys_attached = True
            except Exception as e:
                print("[warn] Failed to attach key handler:", e)

    def step_keyboard(self, dt):
        self._maybe_attach_keys()
        if self._keys_attached:
            if self.keys[key.UP]: self.v = min(self.max_v, self.v + self._v_step * dt)
            if self.keys[key.DOWN]: self.v = max(-self.max_v / 2, self.v - self._v_step * dt)
            if self.keys[key.LEFT]: self.omega = max(-self.max_omega, self.omega - self._omega_step * dt)
            if self.keys[key.RIGHT]: self.omega = min(self.max_omega, self.omega + self._omega_step * dt)
            if self.keys[key.SPACE]: self.v, self.omega = 0.0, 0.0

        self.v *= FRICTION
        self.omega *= FRICTION

        steer, speed = self.vomega_to_ack(self.v, self.omega)
        action = np.array([[steer, speed]], dtype=np.float32)
        obs, reward, done, info = super().step(action)

        car_x, car_y, car_theta = obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]

        # Update LIDAR dynamically
        lidar_distances = self.lidar.scan(car_x, car_y, car_theta)
        info['lidar'] = lidar_distances

        # Update LIDAR stats label
        if self.lidar_label is None:
            self.lidar_label = pyglet.text.Label(
                '', x=10, y=10, color=(255, 255, 255, 255), batch=self.renderer.batch
            )
        self.lidar_label.text = f"LIDAR: min={np.min(lidar_distances):.2f} m | max={np.max(lidar_distances):.2f} m | mean={np.mean(lidar_distances):.2f} m"

        return obs, reward, done, info


def run(dt):
    global env, DONE
    if DONE:
        return
    obs, reward, DONE, info = env.step_keyboard(TIMESTEP)
    env.render(mode='human')


if __name__ == "__main__":
    env = DiffDriveKeyboardEnv(map=MAP_PATH, map_ext=MAP_EXT, num_agents=1, timestep=TIMESTEP)
    try:
        start_pose = np.array([[env.start_xs[0], env.start_ys[0], env.start_thetas[0]]])
        obs, _, _, _ = env.reset(start_pose)
    except:
        obs, _, _, _ = env.reset(np.array([[0.0, 0.0, 0.0]]))

    env.render()
    env.renderer.push_handlers(env.keys)
    env._keys_attached = True

    DONE = False
    pyglet.clock.schedule_interval(run, TIMESTEP)
    pyglet.app.run()
    env.close()
