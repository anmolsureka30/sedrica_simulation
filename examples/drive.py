#!/usr/bin/env python3
"""
keyboard_drive_f110.py

Keyboard-controlled differential-drive demo using f110_gym (repo classes + renderer).
- Map: f1tenth_gym/gym/f110_gym/envs/maps/vegas.png
- Car image: /Users/anmolsureka/Documents/Coding/sedrica/racecar.jpg

Controls (hold):
  Up    -> increase forward speed
  Down  -> decrease / reverse
  Left  -> increase positive yaw command (turn left)
  Right -> increase negative yaw command (turn right)
  Space -> immediate brake (v_des -> 0)
  R     -> reset to start pose
  Q/Esc -> quit

Notes:
- This wrapper maps keyboard (v, omega) -> (steer, speed) using bicycle model:
    steer = arctan(omega * wheelbase / v)  (if v small, steer=0)
- Uses Integrator.RK4 by default (settable).
"""

import os
import time
import math
import numpy as np
import gym
import pyglet
from pyglet.window import key
from f110_gym.envs.base_classes import Integrator
from argparse import ArgumentParser

# ----------------------------- Config / CLI -----------------------------
def build_args():
    p = ArgumentParser()
    p.add_argument("--map", default="/Users/anmolsureka/Documents/Coding/sedrica/f1tenth_gym/gym/f110_gym/envs/maps/vegas", help="map path WITHOUT extension (repo path is relative)")
    p.add_argument("--map-ext", default=".png", help="map image extension (e.g. .png)")
    p.add_argument("--car-img", default="/Users/anmolsureka/Documents/Coding/sedrica/racecar.jpg",
                   help="absolute path to custom car image (top-down).")
    p.add_argument("--timestep", type=float, default=0.01, help="physics timestep (s)")
    p.add_argument("--integrator", default="RK4", choices=["RK4", "EULER"], help="Integrator (RK4 recommended)")
    p.add_argument("--wheelbase", type=float, default=0.26, help="wheelbase in meters to compute steer mapping")
    p.add_argument("--mpp", type=float, default=0.02, help="meters per pixel")
    p.add_argument("--accel", type=float, default=1.2, help="m/s^2 acceleration rate for keyboard input")
    p.add_argument("--omega-step", type=float, default=3.5, help="rad/s^2 yaw-rate change per second when pressing keys")
    p.add_argument("--friction", type=float, default=0.6, help="simple linear friction coefficient applied to v_des per second")
    return p.parse_args()

# ----------------------------- Helper: map to ackermann steer -----------------------------
def vomega_to_ackermann(v, omega, wheelbase_m):
    """
    Convert commanded linear speed v (m/s) and yaw rate omega (rad/s)
    to an approximate steering angle (rad) for an Ackermann single-track:
        omega = v * tan(delta) / L  => delta = arctan(omega * L / v)
    If |v| is very small, return 0 steering to avoid numerical issues.
    """
    if abs(v) < 1e-3:
        return 0.0
    steer = math.atan(omega * wheelbase_m / v)
    return float(steer)

# ----------------------------- Main -----------------------------
def main():
    args = build_args()

    # integrator enum mapping
    integrator = Integrator.RK4 if args.integrator == "RK4" else Integrator.EULER

    # instantiate environment (single agent)
    # map argument expects path without extension
    env = gym.make('f110_gym:f110-v0',
                   map=args.map,
                   map_ext=args.map_ext,
                   num_agents=1,
                   timestep=args.timestep,
                   integrator=integrator)

    # We'll reset to default start poses (if available) else call reset() without pose.
    # Many f110_env builds have start_xs/start_ys attributes. Try to use them.
    try:
        start_pose = np.array([[env.start_xs[0], env.start_ys[0], env.start_thetas[0]]])
        obs, _, _, _ = env.reset(start_pose)
    except Exception:
        obs, _, _, _ = env.reset()

    # Ensure renderer is created and accessible
    env.render()  # this initializes env.renderer (EnvRenderer)
    renderer = getattr(env, "renderer", None)
    if renderer is None:
        raise RuntimeError("Renderer not available on env (env.renderer is None)")

    # Pyglet key state handler for continuous key press reading
    key_handler = key.KeyStateHandler()
    renderer.push_handlers(key_handler)

    # Load car sprite as pyglet image & create sprite ONCE; we'll attach it to renderer as attribute
    car_sprite = None
    if os.path.exists(args.car_img):
        # load image
        try:
            img = pyglet.image.load(args.car_img)
            # set anchor to center for rotation about center
            img.anchor_x = img.width // 2
            img.anchor_y = img.height // 2
            # sprite will be created later (needs scaling)
        except Exception as e:
            print("Warning: failed to load car image:", e)
            img = None
    else:
        print("Warning: car image path not found:", args.car_img)
        img = None

    # controller setpoints (what keyboard changes)
    v_des = 0.0       # m/s
    omega_des = 0.0   # rad/s

    # parameters for input dynamics and friction
    accel = args.accel     # m/s^2
    omega_step = args.omega_step  # rad/s^2
    friction = args.friction

    sim_time = 0.0
    done = False

    # attach a render callback to draw the sprite (and optionally follow camera)
    def render_callback(env_renderer):
        nonlocal car_sprite, img

        # env_renderer has cars list with vertices (as in reference)
        try:
            car_vertices = env_renderer.cars[0].vertices  # flat list [x0,y0,x1,y1,...]
            # compute center as average of corners
            xs = car_vertices[::2]
            ys = car_vertices[1::2]
            center_x = sum(xs) / (len(xs))
            center_y = sum(ys) / (len(ys))
            # rotation: env provides agent theta in obs, but easiest is to derive from vertices:
            # find heading vector from first corner to second corner center approx
            # Instead use obs poses if available on env.renderer: env_renderer.poses? fallback to 0
            try:
                theta = env.current_obs['poses_theta'][0]
            except Exception:
                theta = 0.0
        except Exception:
            # fallback centers
            center_x, center_y, theta = 0.0, 0.0, 0.0

        if img is not None:
            if car_sprite is None:
                # scale image so that its width approximates vehicle length on screen
                # compute car length in pixels from env parameters if available
                try:
                    car_length_m = env.params['length']
                    mpp = args.mpp
                except Exception:
                    # fallback to 0.35m length and provided mpp
                    car_length_m = 0.35
                    mpp = args.mpp
                target_px = max(8, int(car_length_m / mpp))
                scale = target_px / max(img.width, img.height)
                car_sprite = pyglet.sprite.Sprite(img)
                car_sprite.scale = scale
                car_sprite.update(x=center_x, y=center_y)
                car_sprite.rotation = -math.degrees(theta)
            else:
                car_sprite.update(x=center_x, y=center_y)
                car_sprite.rotation = -math.degrees(theta)
            # Draw sprite onto top of batch (draw directly)
            car_sprite.draw()
        else:
            # If no sprite, we won't draw — rely on existing polygon rendering
            pass

    env.add_render_callback(render_callback)

    print("Starting keyboard control loop. Controls: ↑ accelerate, ↓ decelerate, ←/→ turn, SPACE brake, R reset, Q quit")

    last_time = time.time()
    while not done:
        # compute dt (wall-clock)
        now = time.time()
        dt = now - last_time if last_time is not None else args.timestep
        last_time = now
        sim_time += dt

        # read continuous keys via handler
        if key_handler[key.UP]:
            v_des = min(v_des + accel * dt, env.params.get('v_max', 6.0))
        if key_handler[key.DOWN]:
            v_des = max(v_des - accel * dt, -1.5)  # allow small reverse
        if key_handler[key.LEFT]:
            omega_des = min(omega_des + omega_step * dt, 6.0)
        elif key_handler[key.RIGHT]:
            omega_des = max(omega_des - omega_step * dt, -6.0)
        else:
            # natural decay for omega_des when not pressing
            omega_des *= 0.90
            if abs(omega_des) < 0.01:
                omega_des = 0.0

        # immediate key events (non-continuous) handled via pyglet event queue - check env.renderer._events?
        # Simpler: poll for q/r/space from handler as well:
        if key_handler[key.SPACE]:
            v_des = 0.0
        if key_handler[key.R]:
            # reset to initial start pose if available
            try:
                start_pose = np.array([[env.start_xs[0], env.start_ys[0], env.start_thetas[0]]])
                obs, _, _, _ = env.reset(start_pose)
            except Exception:
                obs, _, _, _ = env.reset()
            # small pause to let renderer update
            env.render()
            time.sleep(0.05)
            last_time = time.time()
            continue
        if key_handler[key.Q] or key_handler[key.ESCAPE]:
            print("Quit requested.")
            break

        # apply simple friction to v_des (decay toward zero) so user must hold throttle
        # friction is applied as linear decay per second: v_des -= sign(v_des) * friction * dt * |v_des|
        # simpler: exponential decay:
        v_des *= max(0.0, 1.0 - friction * dt)

        # Map (v_des, omega_des) -> ackermann (steer, speed)
        speed_cmd = float(v_des)
        steer_cmd = vomega_to_ackermann(speed_cmd, float(omega_des), float(args.wheelbase))

        action = np.array([[steer_cmd, speed_cmd]], dtype=np.float32)

        # step the environment
        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            print("Env.step exception:", e)
            break

        # render (this also processes pyglet events)
        try:
            env.render(mode='human')
        except Exception as e:
            # fallback to default render call
            try:
                env.render()
            except Exception:
                pass

        # tiny sleep to avoid busy loop if env.render is non-blocking
        # but don't oversleep because env.render('human') typically syncs to real-time
        time.sleep(0.0)

    # cleanup
    try:
        env.close()
    except Exception:
        pass
    print("Exited.")

if __name__ == "__main__":
    main()