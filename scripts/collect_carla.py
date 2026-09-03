"""Record scripted pedestrian-crossing episodes from a running CARLA server.

Run this with the CARLA interpreter, not the pixi one:

    .venv-carla/bin/python scripts/collect_carla.py --episodes 20

Why a second interpreter: the `carla` wheel is built for manylinux_2_31 and the
pixi environment's conda-forge python declares manylinux_2_28, so pip refuses
the wheel there. The system python 3.12 in .venv-carla is on glibc 2.39 and
takes it fine. README, "Two environments", has the long version. The important
consequence is that this file must never import torch, mediapipe or anything
else from the training side. The two halves talk through files on disk only.

What it writes, one pair per episode, into --out-dir:

    ep_0000.json    annotation in the format src/pif/data/carla.py reads
    ep_0000.npz     RGB frames, uint8 (T, H, W, 3), skipped with --no-frames

What it does NOT write: gesture, gaze or action labels. A CARLA walker has no
hand signals and no head-pose annotation, so inventing those columns would be
inventing data. The parser leaves them empty and the README says so plainly.

Scenarios, chosen with --scenario:

    straight    walks from the near kerb straight across the ego's lane
    hesitate    walks to the kerb, stops, waits, then commits and crosses
    occluded    starts hidden behind a parked car and steps out into the road
    no_cross    walks along the far pavement, never enters the ego's lane
    mixed       one of each in turn (the default)

The occluded case matters most: a pedestrian appearing from behind a parked
vehicle is the failure mode this project documents, and it is the one thing
that is hard to find enough of in JAAD.

Determinism: synchronous mode, a fixed 0.05 s step (20 Hz), and every random
seed pinned from --seed. Same seed and same town gives the same episodes.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# A soft import, so that the camera maths further down can be imported and
# unit-tested from the pixi environment, which cannot install carla at all.
# Anything that actually drives the simulator is gated on this being non-None.
try:
    import carla
except ImportError:  # pragma: no cover - depends on which interpreter ran this
    carla = None

_NO_CARLA = (
    "the `carla` package is not importable in this interpreter.\n"
    "Run this script with .venv-carla/bin/python, not the pixi python:\n"
    "    .venv-carla/bin/python scripts/collect_carla.py --episodes 1\n"
    'The pixi environment cannot install carla; see README, "Two environments".'
)

# Matches src/pif/data/carla.py. Bump both together if the layout changes.
FORMAT = "pif-carla-recording"
FORMAT_VERSION = 1

SCENARIOS = ("straight", "hesitate", "occluded", "no_cross")

# 20 Hz. JAAD and PIE are 30 fps, so a lead time in frames does not carry over
# between them; pif.data.carla.lead_frames converts from seconds instead.
FIXED_DELTA = 0.05

# Per-frame cross strings, the same vocabulary JAAD and PIE use.
CROSSING = "crossing"
NOT_CROSSING = "not-crossing"
IRRELEVANT = "irrelevant"

# Semantic tags that really do hide a pedestrian when a ray hits them first.
# Road, sidewalk, terrain and ground are left out on purpose: a ray from a
# 1.6 m camera to a person standing on the same ground plane grazes them
# constantly, and counting those would report everyone as occluded.
_OCCLUDER_TAG_NAMES = (
    "Buildings",
    "Fences",
    "Other",
    "Poles",
    "Vehicles",
    "Walls",
    "Vegetation",
    "TrafficSigns",
    "Static",
    "Dynamic",
    "GuardRail",
    "Bridge",
    "Truck",
    "Bus",
    "Car",
)

# Rays starting inside or beside the ego's own bodywork would otherwise report
# the ego vehicle as an occluder of everything.
_RAY_NEAR_CLIP = 1.2
# A hit this close to the target is the target itself, not something in front.
_RAY_TARGET_MARGIN = 0.35


def pick_blueprint(blueprints, wanted: str, fallback: str):
    """A named vehicle if the build has it, otherwise the first of a family.

    Blueprint names come and go between CARLA releases, and an IndexError three
    minutes into a collection run is a poor way to find that out.
    """
    found = blueprints.filter(wanted)
    if found:
        return found[0]
    return blueprints.filter(fallback)[0]


def occluder_tags() -> set:
    labels = carla.CityObjectLabel
    return {getattr(labels, n) for n in _OCCLUDER_TAG_NAMES if hasattr(labels, n)}


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def intrinsics(width: int, height: int, fov: float) -> np.ndarray:
    """Pinhole camera matrix for a CARLA RGB camera.

    CARLA's `fov` is the horizontal field of view in degrees, and its cameras
    have square pixels and no distortion, so one focal length covers both axes.
    """
    f = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project(points: np.ndarray, world_to_camera: np.ndarray, k: np.ndarray):
    """World points (N, 3) -> pixel coordinates (N, 2) and depth (N,).

    The axis shuffle in the middle is the part that catches people out. Unreal,
    and therefore CARLA, uses x forward, y right, z up. The pinhole convention
    the matrix `k` is written for uses x right, y down, z forward. So the
    camera-space vector (x, y, z) becomes (y, -z, x).
    """
    n = len(points)
    homogeneous = np.concatenate([points, np.ones((n, 1))], axis=1).T
    cam = world_to_camera @ homogeneous
    shuffled = np.stack([cam[1], -cam[2], cam[0]])
    depth = shuffled[2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    pixels = (k @ shuffled)[:2] / safe
    return pixels.T, depth


def actor_box_2d(actor, world_to_camera, k, width, height):
    """Axis-aligned 2D extent of an actor's 3D bounding box, or None.

    None means the actor is behind the camera, off the edge of the frame, or
    projects to fewer than two pixels across, which is not a usable box.
    """
    verts = actor.bounding_box.get_world_vertices(actor.get_transform())
    points = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float64)
    pixels, depth = project(points, world_to_camera, k)
    in_front = depth > 0.1
    if not in_front.any():
        return None
    pixels = pixels[in_front]
    x1 = float(np.clip(pixels[:, 0].min(), 0.0, width))
    x2 = float(np.clip(pixels[:, 0].max(), 0.0, width))
    y1 = float(np.clip(pixels[:, 1].min(), 0.0, height))
    y2 = float(np.clip(pixels[:, 1].max(), 0.0, height))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def sample_points(actor) -> list:
    """Points on an actor to aim visibility rays at: the eight box corners
    pulled 25% towards the centre, plus the centre itself. Pulling the corners
    in stops a ray from clipping the very edge of the mesh and reporting a
    clearly visible person as hidden."""
    verts = actor.bounding_box.get_world_vertices(actor.get_transform())
    # Mean of the eight corners rather than Transform.transform, which mutates
    # its argument in place in some CARLA releases and returns None.
    cx = sum(v.x for v in verts) / len(verts)
    cy = sum(v.y for v in verts) / len(verts)
    cz = sum(v.z for v in verts) / len(verts)
    out = [carla.Location(x=cx, y=cy, z=cz)]
    for v in verts:
        out.append(
            carla.Location(
                x=cx + 0.75 * (v.x - cx),
                y=cy + 0.75 * (v.y - cy),
                z=cz + 0.75 * (v.z - cz),
            )
        )
    return out


def box_overlap_fraction(box, occluders) -> float:
    """Fraction of a 2D box hidden behind nearer boxes.

    This backs up the ray casting. `World.cast_ray` reports the level geometry,
    and depending on the build it may not see spawned vehicles at all - which
    would make the parked car in the "occluded" scenario invisible to the
    measurement, and that scenario is the point of the exercise. Comparing the
    projected boxes catches it either way.

    Two approximations, both of which under-report: the boxes are axis-aligned,
    so a car's box covers a little more than the car does, and overlapping
    occluders are taken one at a time rather than unioned.
    """
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    worst = 0.0
    for ox1, oy1, ox2, oy2 in occluders:
        w = min(x2, ox2) - max(x1, ox1)
        h = min(y2, oy2) - max(y1, oy1)
        if w <= 0.0 or h <= 0.0:
            continue
        worst = max(worst, (w * h) / area)
    return min(worst, 1.0)


def visible_fraction(world, camera_location, actor, tags) -> float:
    """Fraction of sample rays from the camera that reach the actor unblocked.

    This is a geometric measurement against the level, not a pixel count, so
    the partial-occlusion code the parser derives from it is real rather than a
    guess. If the build of CARLA in use has no `cast_ray`, the caller falls
    back to not measuring at all rather than to a made-up number.
    """
    clear = 0
    targets = sample_points(actor)
    for target in targets:
        distance = camera_location.distance(target)
        blocked = False
        for hit in world.cast_ray(camera_location, target):
            if hit.label not in tags:
                continue
            d = camera_location.distance(hit.location)
            if d < _RAY_NEAR_CLIP or d > distance - _RAY_TARGET_MARGIN:
                continue
            blocked = True
            break
        if not blocked:
            clear += 1
    return clear / float(len(targets))


# --------------------------------------------------------------------------
# map queries
# --------------------------------------------------------------------------


def on_driving_lane(carla_map, location) -> bool:
    """True when the location sits on a lane vehicles drive on.

    project_to_road=False is the whole point: with it left on, CARLA snaps any
    location to the nearest lane and this would answer True for a person
    standing on the pavement.
    """
    wp = carla_map.get_waypoint(
        location, project_to_road=False, lane_type=carla.LaneType.Driving
    )
    return wp is not None


def count_driving_lanes(waypoint) -> int:
    n = 1
    for step in ("get_left_lane", "get_right_lane"):
        wp = waypoint
        for _ in range(6):
            wp = getattr(wp, step)()
            if wp is None or wp.lane_type != carla.LaneType.Driving:
                break
            n += 1
    return n


def straight_spawn_point(carla_map, spawn_points, run_metres: float, rng):
    """A spawn point whose lane runs roughly straight for `run_metres`.

    The ego holds a fixed heading for the whole episode, so it needs a straight
    stretch; a junction or a bend would put it on the pavement halfway through.
    """
    candidates = list(spawn_points)
    rng.shuffle(candidates)
    for point in candidates:
        wp = carla_map.get_waypoint(point.location)
        if wp is None or wp.is_junction:
            continue
        start_yaw = wp.transform.rotation.yaw
        ok = True
        for d in range(5, int(run_metres) + 1, 5):
            nxt = wp.next(float(d))
            if not nxt or nxt[0].is_junction:
                ok = False
                break
            drift = abs((nxt[0].transform.rotation.yaw - start_yaw + 180) % 360 - 180)
            if drift > 8.0:
                ok = False
                break
        if ok:
            return point
    return None


def near_crosswalk(crosswalk_points, location, radius: float = 6.0) -> bool:
    return any(location.distance(p) < radius for p in crosswalk_points)


def traffic_light_state(vehicle) -> str:
    # get_traffic_light_state() answers Green when no light governs the vehicle
    # at all, which would be a claim the scene does not support. Check that a
    # light is actually attached first.
    if vehicle.get_traffic_light() is None:
        return "n/a"
    state = vehicle.get_traffic_light_state()
    return {
        carla.TrafficLightState.Red: "red",
        carla.TrafficLightState.Yellow: "yellow",
        carla.TrafficLightState.Green: "green",
    }.get(state, "n/a")


def ego_action(speed_mps: float) -> str:
    """JAAD's categorical ego label, for readability in the JSON. The context
    encoder does not use it here: the recording also carries the real speed."""
    if speed_mps < 0.3:
        return "stopped"
    if speed_mps < 6.0:
        return "moving_slow"
    return "moving_fast"


# --------------------------------------------------------------------------
# scenario setup
# --------------------------------------------------------------------------


@dataclass
class WalkerPlan:
    """One scripted walker: where it starts, where it is going, and when it
    hesitates."""

    ped_id: str
    scenario: str
    start: object
    destination: object
    speed: float
    ego_relevant: bool
    pause_from: int = -1     # frame the walker stops at the kerb
    pause_until: int = -1    # frame it starts moving again
    actor: object = None
    controller: object = None
    frames: list = field(default_factory=list)
    bbox: list = field(default_factory=list)
    visible: list = field(default_factory=list)
    cross: list = field(default_factory=list)


def plan_walkers(scenario: str, episode: int, anchor, lane_width: float, rng):
    """Where the walkers go, expressed relative to a waypoint ahead of the ego.

    Everything is built off `anchor`, a waypoint on the ego's own lane some
    distance in front of it, rather than off map coordinates. That way the same
    scenario code works in any town without a table of hand-picked locations.
    """
    tf = anchor.transform
    right = tf.get_right_vector()
    forward = tf.get_forward_vector()
    half = lane_width / 2.0
    kerb = half + 2.0

    def offset(lateral: float, along: float = 0.0, up: float = 1.0):
        return carla.Location(
            x=tf.location.x + right.x * lateral + forward.x * along,
            y=tf.location.y + right.y * lateral + forward.y * along,
            z=tf.location.z + up,
        )

    near = offset(kerb)
    far = offset(-kerb)
    speed = round(rng.uniform(1.1, 1.6), 3)
    pid = f"ep_{episode:04d}_w0"

    if scenario == "straight":
        return [
            WalkerPlan(pid, scenario, near, far, speed, ego_relevant=True)
        ]

    if scenario == "hesitate":
        # Stops for one to two seconds with its toes at the kerb, then commits.
        # This is the case a purely kinematic predictor gets wrong: velocity
        # goes to zero and the intent does not.
        pause_from = rng.randint(40, 70)
        return [
            WalkerPlan(
                pid,
                scenario,
                near,
                far,
                speed,
                ego_relevant=True,
                pause_from=pause_from,
                pause_until=pause_from + rng.randint(20, 40),
            )
        ]

    if scenario == "occluded":
        # Starts further out behind where the parked car will be, so the first
        # part of the walk is genuinely hidden from the camera.
        return [
            WalkerPlan(
                pid, scenario, offset(kerb + 1.5, along=-1.0), far, speed,
                ego_relevant=True,
            )
        ]

    if scenario == "no_cross":
        # Walks along the far pavement. A negative that still moves, which is
        # what the model has to learn to separate from a real approach.
        return [
            WalkerPlan(
                pid, scenario, far, offset(-kerb, along=18.0), speed,
                ego_relevant=False,
            )
        ]

    raise ValueError(f"unknown scenario {scenario!r}")


def parked_vehicle_transform(anchor, lane_width: float):
    """A car parked at the near kerb, between the camera and the walker."""
    tf = anchor.transform
    right = tf.get_right_vector()
    forward = tf.get_forward_vector()
    lateral = lane_width / 2.0 + 1.0
    back = 5.0
    return carla.Transform(
        carla.Location(
            x=tf.location.x + right.x * lateral - forward.x * back,
            y=tf.location.y + right.y * lateral - forward.y * back,
            z=tf.location.z + 0.3,
        ),
        tf.rotation,
    )


# --------------------------------------------------------------------------
# the episode
# --------------------------------------------------------------------------


class Stage:
    """Holds every actor spawned for one episode so teardown cannot miss one.

    Used as a context manager. Leaving the server in synchronous mode with a
    fixed step, or with orphan walkers wandering about, breaks the next run and
    is not obvious when it happens, so restoring both is done in a finally.
    """

    def __init__(self, world, original_settings):
        self.world = world
        self.original_settings = original_settings
        self.sensors: list = []
        self.controllers: list = []
        self.actors: list = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        for sensor in self.sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        for controller in self.controllers:
            try:
                controller.stop()
                controller.destroy()
            except RuntimeError:
                pass
        for actor in self.actors:
            try:
                actor.destroy()
            except RuntimeError:
                pass
        self.world.apply_settings(self.original_settings)
        return False


def run_episode(world, args, episode: int, scenario: str):
    carla_map = world.get_map()
    blueprints = world.get_blueprint_library()
    rng = random.Random(args.seed + episode)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = bool(args.headless)
    world.apply_settings(settings)
    # Pins the walker navigation's own random stream, which is separate from
    # python's and would otherwise make episodes irreproducible.
    world.set_pedestrians_seed(args.seed + episode)

    total_frames = int(round(args.episode_seconds / FIXED_DELTA))
    save_frames = args.frames and not args.headless
    tags = occluder_tags()

    with Stage(world, original_settings) as stage:
        spawn_points = carla_map.get_spawn_points()
        if not spawn_points:
            print(f"  episode {episode}: town has no spawn points, skipping")
            return None
        start = straight_spawn_point(
            carla_map, spawn_points, args.walker_distance + 25.0, rng
        )
        if start is None:
            print(f"  episode {episode}: no straight stretch found, skipping")
            return None

        vehicle_bp = pick_blueprint(blueprints, "vehicle.tesla.model3", "vehicle.*")
        ego = world.try_spawn_actor(vehicle_bp, start)
        if ego is None:
            print(f"  episode {episode}: could not spawn the ego vehicle, skipping")
            return None
        stage.actors.append(ego)

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))
        camera_bp.set_attribute("fov", str(args.fov))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=1.6))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego)
        stage.sensors.append(camera)

        # The camera actor is spawned even with rendering off, because its
        # transform is what the 3D-to-2D projection is measured against. It
        # just does not deliver images in that mode, so nothing listens.
        images: queue.Queue = queue.Queue()
        if not args.headless:
            camera.listen(images.put)
        k = intrinsics(args.width, args.height, args.fov)

        ego_wp = carla_map.get_waypoint(ego.get_location())
        ahead = ego_wp.next(args.walker_distance)
        if not ahead:
            print(f"  episode {episode}: lane ends too soon, skipping")
            return None
        anchor = ahead[0]
        lane_width = anchor.lane_width
        num_lanes = count_driving_lanes(anchor)
        crosswalks = carla_map.get_crosswalks()
        crosswalk_here = near_crosswalk(crosswalks, anchor.transform.location)

        traffic_entries: list[dict] = []
        parked = None
        if scenario == "occluded":
            parked_bp = pick_blueprint(blueprints, "vehicle.audi.a2", "vehicle.*")
            parked = world.try_spawn_actor(
                parked_bp, parked_vehicle_transform(anchor, lane_width)
            )
            if parked is not None:
                stage.actors.append(parked)
                parked.set_simulate_physics(False)
                traffic_entries.append(
                    {
                        "obj_id": f"ep_{episode:04d}_parked",
                        "obj_class": "vehicle",
                        "obj_type": parked.type_id,
                        "frames": [],
                        "bbox": [],
                        "state": [],
                    }
                )

        plans = plan_walkers(scenario, episode, anchor, lane_width, rng)
        walker_bps = blueprints.filter("walker.pedestrian.*")
        controller_bp = blueprints.find("controller.ai.walker")
        for plan in plans:
            bp = walker_bps[rng.randrange(len(walker_bps))]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            actor = None
            for lift in (0.0, 0.5, 1.0, 1.5):
                spot = carla.Transform(
                    carla.Location(
                        x=plan.start.x, y=plan.start.y, z=plan.start.z + lift
                    )
                )
                actor = world.try_spawn_actor(bp, spot)
                if actor is not None:
                    break
            if actor is None:
                print(f"  episode {episode}: could not spawn walker, skipping")
                return None
            plan.actor = actor
            stage.actors.append(actor)

        world.tick()

        for plan in plans:
            controller = world.spawn_actor(
                controller_bp, carla.Transform(), attach_to=plan.actor
            )
            plan.controller = controller
            stage.controllers.append(controller)

        world.tick()

        for plan in plans:
            plan.controller.start()
            plan.controller.go_to_location(plan.destination)
            plan.controller.set_max_speed(plan.speed)

        # Let physics settle and the walkers actually start moving before the
        # first recorded frame, otherwise every episode opens with a person
        # standing still for half a second.
        for _ in range(args.warmup_frames):
            world.tick()
            if not args.headless:
                images.get(timeout=args.timeout)

        # Fetched once: nothing spawns or despawns during an episode, and this
        # is an RPC to the server.
        vehicles = [v for v in world.get_actors().filter("vehicle.*") if v.id != ego.id]

        frames: list[np.ndarray] = []
        ego_rows: dict[str, dict] = {}
        scene_rows: dict[str, dict] = {}

        for frame in range(total_frames):
            velocity = ego.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            throttle = 0.45 if speed < args.ego_speed else 0.0
            ego.apply_control(
                carla.VehicleControl(throttle=throttle, steer=0.0, brake=0.0)
            )

            for plan in plans:
                if plan.pause_from < 0:
                    continue
                if frame == plan.pause_from:
                    plan.controller.set_max_speed(0.0)
                elif frame == plan.pause_until:
                    plan.controller.set_max_speed(plan.speed)

            world.tick()
            if not args.headless:
                image = images.get(timeout=args.timeout)
                if save_frames:
                    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
                    # CARLA hands over BGRA; drop alpha and flip to RGB.
                    frames.append(
                        raw.reshape(args.height, args.width, 4)[:, :, :3][
                            :, :, ::-1
                        ].copy()
                    )

            world_to_camera = np.array(camera.get_transform().get_inverse_matrix())
            camera_location = camera.get_transform().location

            ego_rows[str(frame)] = {
                # Named after PIE's OBD column on purpose: pif.features.context
                # already reads that key and will use this real speed instead
                # of falling back to JAAD's categorical proxy table.
                "OBD_speed": round(speed * 3.6, 3),
                "speed_mps": round(speed, 3),
                "action": ego_action(speed),
            }
            scene_rows[str(frame)] = {
                "ped_crossing": int(crosswalk_here),
                "ped_sign": 0,
                "stop_sign": 0,
                "traffic_light": traffic_light_state(ego),
            }

            # Vehicle boxes, with how far away they are, so a walker behind one
            # can be told from a walker in front of one.
            vehicle_boxes = []
            for other in vehicles:
                other_box = actor_box_2d(
                    other, world_to_camera, k, args.width, args.height
                )
                if other_box is None:
                    continue
                vehicle_boxes.append(
                    (camera_location.distance(other.get_location()), other_box)
                )
                if parked is not None and traffic_entries and other.id == parked.id:
                    traffic_entries[0]["frames"].append(frame)
                    traffic_entries[0]["bbox"].append(other_box)

            for plan in plans:
                box = actor_box_2d(
                    plan.actor, world_to_camera, k, args.width, args.height
                )
                if box is None:
                    continue  # off frame entirely: no annotation for this frame
                location = plan.actor.get_location()
                if args.measure_occlusion:
                    distance = camera_location.distance(location)
                    nearer = [b for d, b in vehicle_boxes if d < distance]
                    visible = min(
                        visible_fraction(world, camera_location, plan.actor, tags),
                        1.0 - box_overlap_fraction(box, nearer),
                    )
                else:
                    visible = 1.0
                if on_driving_lane(carla_map, location):
                    state = CROSSING if plan.ego_relevant else IRRELEVANT
                else:
                    state = NOT_CROSSING
                plan.frames.append(frame)
                plan.bbox.append(box)
                plan.visible.append(round(visible, 3))
                plan.cross.append(state)

        record = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "episode_id": f"ep_{episode:04d}",
            "dataset": "carla",
            "town": carla_map.name,
            "scenario": scenario,
            "seed": args.seed + episode,
            "fps": round(1.0 / FIXED_DELTA, 3),
            "fixed_delta_seconds": FIXED_DELTA,
            "width": args.width,
            "height": args.height,
            "num_frames": total_frames,
            "road_type": "street",
            # Recorded so the projection above can be reproduced or checked
            # later without re-running the simulator.
            "camera": {
                "fov": args.fov,
                "fx": float(k[0, 0]),
                "fy": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
                "location_in_ego": [1.5, 0.0, 1.6],
                "rotation_in_ego": [0.0, 0.0, 0.0],
            },
            # False means visibility was not measured and every walker is
            # recorded as fully visible. Do not read occlusion off a recording
            # with this set to false.
            "occlusion_measured": bool(args.measure_occlusion),
            "frames_file": f"ep_{episode:04d}.npz" if save_frames else None,
            "ego": ego_rows,
            "scene": scene_rows,
            "walkers": [
                {
                    "ped_id": plan.ped_id,
                    "scenario": plan.scenario,
                    "ego_relevant": plan.ego_relevant,
                    "frames": plan.frames,
                    "bbox": plan.bbox,
                    "visible_fraction": plan.visible,
                    "cross": plan.cross,
                    "attributes": {
                        "num_lanes": num_lanes,
                        "signalized": "n/a",
                        "walk_speed_mps": plan.speed,
                    },
                }
                for plan in plans
                if plan.frames
            ],
            "traffic": [t for t in traffic_entries if t["frames"]],
        }
        return record, (np.stack(frames) if frames else None)


def collect(args) -> int:
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()
    if args.town and args.town not in world.get_map().name:
        print(f"loading {args.town}")
        world = client.load_world(args.town)

    if args.measure_occlusion and not hasattr(world, "cast_ray"):
        print(
            "this CARLA build has no World.cast_ray, so visibility cannot be "
            "measured. Recording with occlusion_measured=false: every walker "
            "will be marked fully visible and the occlusion column carries no "
            "information."
        )
        args.measure_occlusion = False
    if args.headless and args.frames:
        print("--headless turns rendering off, so no frames will be saved")

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for episode in range(args.episodes):
        scenario = (
            SCENARIOS[episode % len(SCENARIOS)]
            if args.scenario == "mixed"
            else args.scenario
        )
        print(f"episode {episode}: {scenario}")
        result = run_episode(world, args, episode, scenario)
        if result is None:
            continue
        record, frames = result
        if not record["walkers"]:
            print(f"  episode {episode}: no walker was ever in frame, discarding")
            continue

        json_path = out_dir / f"{record['episode_id']}.json"
        json_path.write_text(json.dumps(record, indent=1) + "\n")
        if frames is not None:
            np.savez_compressed(out_dir / f"{record['episode_id']}.npz", frames=frames)
        written += 1

        walker = record["walkers"][0]
        crossed = CROSSING in walker["cross"]
        print(
            f"  wrote {json_path.name}: {len(walker['frames'])} annotated frames, "
            f"crossed={crossed}"
        )

    print(f"\n{written}/{args.episodes} episodes written to {out_dir}")
    if written:
        print(
            "check them with the pixi python, which is the one that can import "
            "pif:\n"
            f"    .pixi/envs/default/bin/python -c \"from pif.data.carla import "
            f"CARLARecordings; print(CARLARecordings('{out_dir}').video_ids())\""
        )
    return 0 if written else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--out-dir", default="data/carla")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scenario", default="mixed", choices=("mixed",) + SCENARIOS
    )
    p.add_argument("--town", default="Town10HD_Opt")
    p.add_argument(
        "--headless",
        action="store_true",
        help="turn the server's rendering off for this run. Much faster, but "
        "there is then nothing to capture, so no frames are saved and only "
        "the annotations come out.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument(
        "--timeout", type=float, default=20.0, help="seconds to wait on the server"
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--episode-seconds", type=float, default=12.0)
    p.add_argument(
        "--ego-speed", type=float, default=5.0, help="target ego speed, m/s"
    )
    p.add_argument(
        "--walker-distance",
        type=float,
        default=28.0,
        help="metres ahead of the ego that the crossing is staged",
    )
    p.add_argument("--warmup-frames", type=int, default=20)
    frames = p.add_mutually_exclusive_group()
    frames.add_argument("--frames", dest="frames", action="store_true", default=True)
    frames.add_argument(
        "--no-frames",
        dest="frames",
        action="store_false",
        help="annotations only. Much smaller on disk, and enough for everything "
        "except pose extraction.",
    )
    occ = p.add_mutually_exclusive_group()
    occ.add_argument(
        "--measure-occlusion",
        dest="measure_occlusion",
        action="store_true",
        default=True,
    )
    occ.add_argument(
        "--no-measure-occlusion",
        dest="measure_occlusion",
        action="store_false",
        help="skip the visibility ray casts. Faster, but then every walker is "
        "recorded as fully visible and the occlusion column means nothing.",
    )
    args = p.parse_args()
    if carla is None:
        print(_NO_CARLA, file=sys.stderr)
        return 2
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
