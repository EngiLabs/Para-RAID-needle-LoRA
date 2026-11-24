from __future__ import annotations

from typing import Any


# RF(phi, alpha, beta), LF(phi, alpha, beta), RB(phi, alpha, beta), LB(phi, alpha, beta)
POSE_PRESETS: dict[str, list[float]] = {
    "neutral":      [0.0,  -8.98, 68.76,  0.0,  -8.98, 68.76,  0.0,  -8.98, 68.76,  0.0,  -8.98, 68.76],
    "crouch":       [0.0,   6.25, 63.61,  0.0,   6.25, 63.61,  0.0,   6.25, 63.61,  0.0,   6.25, 63.61],
    "tall":         [0.0, -28.28, 54.86,  0.0, -28.28, 54.86,  0.0, -28.28, 54.86,  0.0, -28.28, 54.86],
    "tilt_forward": [0.0,   3.93, 64.21,  0.0,   3.93, 64.21,  0.0, -20.21, 60.88,  0.0, -20.21, 60.88],
    "tilt_back":    [0.0, -20.21, 60.88,  0.0, -20.21, 60.88,  0.0,   3.93, 64.21,  0.0,   3.93, 64.21],
}

SECONDS_PER_90_DEG = 2.0

SPIDOR_TOOLS: list[dict[str, Any]] = [
    {
        "name": "move_robot",
        "description": "Move the quadruped robot in a direction, optionally by distance in meters or duration in seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "left", "right"],
                    "description": "Movement direction: forward, backward, left, right",
                },
                "distance_m": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 5.0,
                    "description": "Distance to travel in meters",
                },
                "duration_s": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 30.0,
                    "description": "Duration of movement in seconds",
                },
                "speed": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 255,
                    "description": "Speed from 1 (slowest) to 255 (fastest)",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "turn_robot",
        "description": "Rotate the robot in place left or right by an angle in degrees.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "Rotation direction: left or right",
                },
                "angle_deg": {
                    "type": "number",
                    "minimum": 5.0,
                    "maximum": 360.0,
                    "description": "Turn angle in degrees",
                },
                "speed": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 255,
                    "description": "Turn speed from 1 to 255",
                },
            },
            "required": ["direction", "angle_deg"],
        },
    },
    {
        "name": "stop_robot",
        "description": "Immediately stop all robot movement.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "set_gait",
        "description": "Change robot walking gait pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "gait": {
                    "type": "string",
                    "enum": ["walk", "trot", "creep"],
                    "description": "Gait type: walk (normal), trot (fast), creep (slow and stable)",
                },
            },
            "required": ["gait"],
        },
    },
    {
        "name": "power_pose",
        "description": "Switch servo power or default stance: home, relax, wake.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["home", "relax", "wake"],
                    "description": "home: stand in default pose; relax: release servos; wake: power on servos",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "set_pose_preset",
        "description": "Set body pose using a named 12-angle servo preset.",
        "parameters": {
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": list(POSE_PRESETS.keys()),
                    "description": "Pose preset: neutral, crouch, tall, tilt_forward, tilt_back",
                },
            },
            "required": ["preset"],
        },
    },
    {
        "name": "set_led",
        "description": "Turn the robot LED light on or off.",
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "true to turn on, false to turn off",
                },
            },
            "required": ["enabled"],
        },
    },
    {
        "name": "capture_photo",
        "description": "Take a photo using the robot camera.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_status",
        "description": "Check Para-RAID server status, robot WS connection, gait, and stream state.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_telemetry",
        "description": "Request fresh sensor telemetry (IR, IMU orientation/accel/gyro, position) from the robot.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "navigate_to",
        "description": "Navigate autonomously to (x, y) coordinates in meters.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Target X coordinate in meters"},
                "y": {"type": "number", "description": "Target Y coordinate in meters"},
            },
            "required": ["x", "y"],
        },
    },
]


def tool_call_to_raid_command(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Needle tool call -> payload for /api/command"""
    if name == "move_robot":
        direction = str(args.get("direction", "forward"))
        cmd: dict[str, Any] = {"action": direction, "direction": direction}
        if args.get("distance_m") is not None:
            cmd["distance_m"] = float(args["distance_m"])
        if args.get("duration_s") is not None:
            cmd["duration_s"] = float(args["duration_s"])
        if args.get("speed") is not None:
            cmd["speed"] = int(args["speed"])
        return cmd

    if name == "turn_robot":
        direction = str(args.get("direction", "left"))
        angle = float(args.get("angle_deg", 90.0))
        speed = int(args.get("speed", 200))
        duration_s = round((angle / 90.0) * SECONDS_PER_90_DEG * (200.0 / max(50, speed)), 2)
        return {
            "action": direction,
            "direction": direction,
            "duration_s": duration_s,
            "speed": speed,
        }

    if name == "stop_robot":
        return {"action": "stop"}

    if name == "set_gait":
        return {"action": "set_gait", "gait": str(args.get("gait", "walk"))}

    if name == "power_pose":
        return {"action": str(args.get("action", "home"))}

    if name == "set_pose_preset":
        preset = str(args.get("preset", "neutral"))
        return {"action": "set_pose", "angles": list(POSE_PRESETS.get(preset, POSE_PRESETS["neutral"]))}

    if name == "set_led":
        return {"action": "set_led", "state": 1 if args.get("enabled", True) else 0}

    if name == "capture_photo":
        return {"action": "photo"}

    if name == "get_status":
        return {"action": "status"}

    if name == "get_telemetry":
        return {"action": "telemetry"}

    if name == "navigate_to":
        return {"action": "goto", "x": float(args["x"]), "y": float(args["y"])}

    raise ValueError(f"Unknown tool name: {name}")
