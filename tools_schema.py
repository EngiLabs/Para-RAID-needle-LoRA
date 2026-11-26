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
        "name": "move",
        "description": "Start moving the quadruped robot continuously in a direction.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "left", "right"],
                    "description": "Movement direction: forward, backward, left, right",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "move_distance",
        "description": "Move the quadruped robot in a direction by a specific distance in meters.",
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
            },
            "required": ["direction", "distance_m"],
        },
    },
    {
        "name": "move_duration",
        "description": "Move the quadruped robot in a direction for a specific duration in seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "left", "right"],
                    "description": "Movement direction: forward, backward, left, right",
                },
                "duration_s": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 30.0,
                    "description": "Duration of movement in seconds",
                },
            },
            "required": ["direction", "duration_s"],
        },
    },
    {
        "name": "set_speed",
        "description": "Set the persistent movement and rotation speed of the robot (1 slowest to 255 fastest).",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 255,
                    "description": "Speed from 1 (slowest) to 255 (fastest)",
                },
            },
            "required": ["speed"],
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
    if name == "move":
        direction = str(args.get("direction", "forward"))
        return {"action": direction, "direction": direction}

    if name == "move_distance":
        direction = str(args.get("direction", "forward"))
        return {
            "action": direction,
            "direction": direction,
            "distance_m": float(args["distance_m"]),
        }

    if name == "move_duration":
        direction = str(args.get("direction", "forward"))
        return {
            "action": direction,
            "direction": direction,
            "duration_s": float(args["duration_s"]),
        }

    if name == "set_speed":
        return {"action": "set_speed", "speed": int(args["speed"])}

    if name == "turn_robot":
        direction = str(args.get("direction", "left"))
        angle = float(args.get("angle_deg", 90.0))
        duration_s = round((angle / 90.0) * SECONDS_PER_90_DEG, 2)
        return {
            "action": direction,
            "direction": direction,
            "duration_s": duration_s,
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
