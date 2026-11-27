from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from tools_schema import SPIDOR_TOOLS


OUT_DIR = Path("data")
MAX_SAMPLES = 4500
VAL_RATIO = 0.15
SEED = 42

DIR_PHRASES: dict[str, list[str]] = {
    "forward": ["вперёд", "вперед", "прямо"],
    "backward": ["назад", "обратно", "задним ходом"],
    "left": ["влево", "налево"],
    "right": ["вправо", "направо"],
}

MOVE_VERBS = ["пройди", "иди", "шагай", "шагни", "двигайся", "продвигайся", "ступай", "ползи", "едь"]

SPEED_MODIFIERS: list[tuple[list[str], int]] = [
    (["медленно", "помедленнее", "потихоньку", "аккуратно", "осторожно", "не спеша"], 100),
    (["со средней скоростью", "в обычном темпе", "на средней скорости"], 180),
    (["быстро", "побыстрее", "максимально быстро", "на полной скорости", "резко", "газ в пол"], 255),
]

DISTANCE_TEMPLATES: list[tuple[str, float]] = [
    ("шаг", 0.3),
    ("один шаг", 0.3),
    ("два шага", 0.6),
    ("2 шага", 0.6),
    ("пару шагов", 0.6),
    ("три шага", 0.9),
    ("3 шага", 0.9),
    ("четыре шага", 1.2),
    ("5 шагов", 1.5),
    ("полметра", 0.5),
    ("пол метра", 0.5),
    ("1 метр", 1.0),
    ("метр", 1.0),
    ("один метр", 1.0),
    ("полтора метра", 1.5),
    ("2 метра", 2.0),
    ("два метра", 2.0),
    ("3 метра", 3.0),
    ("три метра", 3.0),
    ("3м", 3.0),
    ("4 метра", 4.0),
    ("20 см", 0.2),
    ("30 сантиметров", 0.3),
    ("50 см", 0.5),
    ("50 сантиметров", 0.5),
    ("80 см", 0.8),
    ("120 см", 1.2),
    ("150 см", 1.5),
    ("250 сантиметров", 2.5),
    ("350 см", 3.5),
    ("200 мм", 0.2),
    ("500 мм", 0.5),
    ("0.5 метра", 0.5),
    ("1.2 метра", 1.2),
    ("2.5 метра", 2.5),
]

DURATION_TEMPLATES: list[tuple[str, float]] = [
    ("1 секунду", 1.0),
    ("секунду", 1.0),
    ("пару секунд", 2.0),
    ("2 секунды", 2.0),
    ("3 секунды", 3.0),
    ("3с", 3.0),
    ("5 секунд", 5.0),
    ("пять секунд", 5.0),
    ("10 секунд", 10.0),
    ("500 мс", 0.5),
    ("1500 мс", 1.5),
]

TURN_ANGLES: list[tuple[str, float]] = [
    ("30 градусов", 30.0),
    ("45 градусов", 45.0),
    ("45°", 45.0),
    ("60 градусов", 60.0),
    ("90 градусов", 90.0),
    ("90°", 90.0),
    ("девяносто градусов", 90.0),
    ("прямой угол", 90.0),
    ("120 градусов", 120.0),
    ("180 градусов", 180.0),
    ("180°", 180.0),
    ("270 градусов", 270.0),
]

SIMPLE_COMMANDS: list[tuple[list[str], str, dict[str, Any]]] = [
    (
        ["стоп", "остановись", "замри", "хватит", "тормози", "стой", "прекрати движение", "аварийная остановка", "стопэ", "не двигайся", "остановись на месте", "отбой", "прекрати"],
        "stop_robot",
        {},
    ),
    (
        ["включи обычную походку", "походка walk", "переключи на обычный шаг", "иди обычным шагом", "режим walk", "поставь походку шаг"],
        "set_gait",
        {"gait": "walk"},
    ),
    (
        ["включи рысь", "походка trot", "переключи на рысь", "беги рысью", "режим trot", "быстрая походка рысь"],
        "set_gait",
        {"gait": "trot"},
    ),
    (
        ["включи крадущуюся походку", "походка creep", "крадись", "ползи аккуратно в режиме creep", "режим creep", "осторожная походка"],
        "set_gait",
        {"gait": "creep"},
    ),
    (
        ["встань в начальное положение", "прими исходную позу", "домой", "home", "верни ноги в начальное положение", "встань ровно", "исходная позиция", "иди домой", "вернись домой"],
        "power_pose",
        {"action": "home"},
    ),
    (
        ["расслабь сервы", "отключи моторы", "отключи сервы", "выключи питание сервоприводов", "сними нагрузку с моторов", "расслабь ноги", "relax", "убери усилие с сервоприводов", "отдохни", "приляг", "ложись отдыхать", "отпусти сервы"],
        "power_pose",
        {"action": "relax"},
    ),
    (
        ["включи сервы", "подай питание на сервы", "включи сервоприводы", "проснись", "проснить", "подъём", "вставай", "пробудись", "wake", "активируй сервоприводы", "включи моторы и встань", "поднимись и включи моторы"],
        "power_pose",
        {"action": "wake"},
    ),
    (
        ["нейтральная поза", "обычная стойка", "выровняй корпус", "поза neutral", "стандартная поза"],
        "set_pose_preset",
        {"preset": "neutral"},
    ),
    (
        ["пригнись", "присядь", "опусти корпус", "ниже к полу", "поза crouch", "присядь пониже"],
        "set_pose_preset",
        {"preset": "crouch"},
    ),
    (
        ["встань повыше", "подними корпус", "вытянись вверх", "выше стойка", "поза tall"],
        "set_pose_preset",
        {"preset": "tall"},
    ),
    (
        ["наклонись вперёд", "наклон вперед", "посмотри вниз корпусом", "поза tilt_forward", "опусти нос", "наклони корпус вперёд", "опусти переднюю часть", "наклони голову вниз"],
        "set_pose_preset",
        {"preset": "tilt_forward"},
    ),
    (
        ["наклонись назад", "отклонись назад", "задери нос", "поза tilt_back", "приподними нос", "наклони корпус назад", "подними переднюю часть"],
        "set_pose_preset",
        {"preset": "tilt_back"},
    ),
    (
        ["включи свет", "включи светодиод", "зажги подсветку", "вруби led", "подсветку включи", "включи фонарик", "зажги лампу", "включи освещение", "подсвети"],
        "set_led",
        {"enabled": True},
    ),
    (
        ["выключи свет", "погаси светодиод", "отключи подсветку", "выключи подсветку", "погаси подсветку", "убери свет", "выруби led", "выруби подсветку", "выключи led", "погаси свет", "потуши светодиод", "свет выключи", "выключи фонарик", "потуши лампу", "выключи освещение"],
        "set_led",
        {"enabled": False},
    ),
    (
        ["сделай фото", "сфотографируй", "сфоткай", "фотку", "сними кадр", "покажи что видишь", "сделай снимок с камеры", "щёлкни камерой", "сделай снимок экрана", "снимок с камеры", "зафиксируй картинку"],
        "capture_photo",
        {},
    ),
    (
        ["статус сервера", "проверь связь", "робот подключен", "жив ли сервер", "состояние сервера", "какой статус подключения", "проверь коннект", "статус системы", "проверь подключение"],
        "get_status",
        {},
    ),
    (
        ["телеметрия", "дай телеметрию", "запроси телеметрию", "покажи датчики", "проверь датчики", "какие показания датчиков", "что на датчиках", "показания imu", "показания гироскопа", "данные акселерометра", "данные с ик датчиков", "какая ориентация и координаты", "покажи ориентацию и координаты", "сними телеметрию с робота"],
        "get_telemetry",
        {},
    ),
]

NEUTRAL_PREFIXES = ["", "пожалуйста ", "робот ", "эй спидор ", "спидор ", "можешь ", "давай ", "срочно "]
POLITE_PREFIXES = NEUTRAL_PREFIXES
POLITE_SUFFIXES = ["", " пожалуйста", "!", " скорее", " прямо сейчас", " для меня", " сейчас"]


def build_samples(rng: random.Random) -> list[dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}

    def add(query: str, answers: list[dict[str, Any]]) -> None:
        q = " ".join(query.strip().split())
        if q and q not in samples:
            samples[q] = {"query": q, "tools": SPIDOR_TOOLS, "reasoning": "-", "answers": answers}

    for direction, dir_words in DIR_PHRASES.items():
        for dw in dir_words:
            for verb in MOVE_VERBS:
                for pref in NEUTRAL_PREFIXES[:5]:
                    add(f"{pref}{verb} {dw}", [{"name": "move", "arguments": {"direction": direction}}])
                for dist_text, dist_val in DISTANCE_TEMPLATES:
                    for order in (f"{verb} {dw} {dist_text}", f"{verb} {dist_text} {dw}", f"{dw} на {dist_text}", f"{dist_text} {dw}"):
                        add(order, [{"name": "move_distance", "arguments": {"direction": direction, "distance_m": dist_val}}])
                for dur_text, dur_val in DURATION_TEMPLATES:
                    for order in (f"{verb} {dw} {dur_text}", f"{verb} {dw} на {dur_text}", f"{dw} {dur_text}"):
                        add(order, [{"name": "move_duration", "arguments": {"direction": direction, "duration_s": dur_val}}])
                for mod_words, spd in SPEED_MODIFIERS:
                    for mw in mod_words:
                        add(f"{verb} {dw} {mw}", [{"name": "set_speed", "arguments": {"speed": spd}}, {"name": "move", "arguments": {"direction": direction}}])
                        add(f"{mw} {verb} {dw}", [{"name": "set_speed", "arguments": {"speed": spd}}, {"name": "move", "arguments": {"direction": direction}}])
                        dist_text, dist_val = rng.choice(DISTANCE_TEMPLATES)
                        add(f"{verb} {dw} {dist_text} {mw}", [{"name": "set_speed", "arguments": {"speed": spd}}, {"name": "move_distance", "arguments": {"direction": direction, "distance_m": dist_val}}])
                        dur_text, dur_val = rng.choice(DURATION_TEMPLATES)
                        add(f"{verb} {dw} {dur_text} {mw}", [{"name": "set_speed", "arguments": {"speed": spd}}, {"name": "move_duration", "arguments": {"direction": direction, "duration_s": dur_val}}])

    speed_phrases: list[tuple[list[str], int]] = [
        (["сбавь скорость", "помедленнее", "сделай медленнее", "сделай помедленнее", "притормози", "сбавь ход", "потише", "на минимальной скорости", "уменьши скорость", "снизь скорость", "понизь скорость", "медленная скорость", "тише едь"], 100),
        (["обычная скорость", "нормальная скорость", "средняя скорость", "сделай обычную скорость", "сделай нормальную скорость", "средний ход", "верни обычную скорость", "верни скорость", "стандартная скорость", "скорость по умолчанию"], 180),
        (["ускорься", "побыстрее", "прибавь скорость", "увеличь скорость", "максимальная скорость", "сделай максимальную скорость", "быстрая скорость", "полная скорость", "газ в пол", "полный газ", "полный ход", "тапка в пол", "на всю катушку", "максимальный ход", "выжми максимум"], 255),
    ]
    for phrases, spd in speed_phrases:
        for ph in phrases:
            for pref in NEUTRAL_PREFIXES:
                add(f"{pref}{ph}", [{"name": "set_speed", "arguments": {"speed": spd}}])
            for suf in POLITE_SUFFIXES[1:5]:
                add(f"{ph}{suf}", [{"name": "set_speed", "arguments": {"speed": spd}}])

    for spd in (50, 80, 100, 120, 150, 180, 200, 220, 255):
        for tmpl in ("поставь скорость {s}", "установи скорость {s}", "скорость {s}", "задай скорость {s}", "переключи скорость на {s}"):
            q = tmpl.format(s=spd)
            add(q, [{"name": "set_speed", "arguments": {"speed": spd}}])
            for pref in NEUTRAL_PREFIXES[1:5]:
                add(f"{pref}{q}", [{"name": "set_speed", "arguments": {"speed": spd}}])
        d = rng.choice(list(DIR_PHRASES.keys()))
        dw = rng.choice(DIR_PHRASES[d])
        verb = rng.choice(MOVE_VERBS)
        add(f"поставь скорость {spd} и {verb} {dw}", [{"name": "set_speed", "arguments": {"speed": spd}}, {"name": "move", "arguments": {"direction": d}}])

    for direction in ("left", "right"):
        for dw in DIR_PHRASES[direction]:
            for tv in ("поверни", "повернись", "развернись", "крутнись", "поворачивай"):
                add(f"{tv} {dw}", [{"name": "turn_robot", "arguments": {"direction": direction, "angle_deg": 90.0}}])
                for pref in NEUTRAL_PREFIXES[1:5]:
                    add(f"{pref}{tv} {dw}", [{"name": "turn_robot", "arguments": {"direction": direction, "angle_deg": 90.0}}])
                for ang_text, ang_val in TURN_ANGLES:
                    add(f"{tv} {dw} на {ang_text}", [{"name": "turn_robot", "arguments": {"direction": direction, "angle_deg": ang_val}}])
                    add(f"{tv} на {ang_text} {dw}", [{"name": "turn_robot", "arguments": {"direction": direction, "angle_deg": ang_val}}])

    for q, d, ang in (
        ("развернись", "right", 180.0),
        ("разворот", "right", 180.0),
        ("кругом", "right", 180.0),
        ("повернись назад", "right", 180.0),
        ("развернись назад", "right", 180.0),
        ("повернись на 360", "right", 360.0),
        ("крутнись на 360", "right", 360.0),
        ("полный оборот", "right", 360.0),
        ("вокруг себя", "right", 360.0),
    ):
        add(q, [{"name": "turn_robot", "arguments": {"direction": d, "angle_deg": ang}}])
        for pref in NEUTRAL_PREFIXES[1:5]:
            add(f"{pref}{q}", [{"name": "turn_robot", "arguments": {"direction": d, "angle_deg": ang}}])

    for phrases, tool_name, args in SIMPLE_COMMANDS:
        for ph in phrases:
            add(ph, [{"name": tool_name, "arguments": args}])
            for pref in POLITE_PREFIXES[1:]:
                add(f"{pref}{ph}", [{"name": tool_name, "arguments": args}])
            for suf in POLITE_SUFFIXES[1:5]:
                add(f"{ph}{suf}", [{"name": tool_name, "arguments": args}])

    coords = [(0.0, 0.0), (1.0, 2.0), (1.5, -2.0), (-1.0, 3.5), (2.5, 4.0), (3.0, 1.0), (-0.5, -1.5), (4.0, 0.0), (0.5, 0.5), (-2.0, 1.0), (2.0, -3.0), (5.0, 2.5)]
    nav_templates = [
        "иди в точку {x} {y}", "иди к точке {x}, {y}", "двигайся к координатам {x}, {y}",
        "навигация к ({x}, {y})", "дойди до x={x} y={y}", "перейди на точку {x} и {y}",
        "отправляйся в координаты {x} {y}", "проложи маршрут до {x} {y}",
    ]
    for x, y in coords:
        for tmpl in nav_templates:
            add(tmpl.format(x=x, y=y), [{"name": "navigate_to", "arguments": {"x": x, "y": y}}])

    photo_telem_phrases = [
        "сделай фото и проверь телеметрию",
        "сфоткай и покажи телеметрию",
        "сделай снимок и сними телеметрию",
        "сделай фото и запроси телеметрию",
        "сфотографируй и покажи датчики",
        "фотку и телеметрию",
    ]
    for pt in photo_telem_phrases:
        add(pt, [{"name": "capture_photo", "arguments": {}}, {"name": "get_telemetry", "arguments": {}}])
        for pref in NEUTRAL_PREFIXES[1:6]:
            add(f"{pref}{pt}", [{"name": "capture_photo", "arguments": {}}, {"name": "get_telemetry", "arguments": {}}])

    for _ in range(900):
        d1 = rng.choice(["forward", "backward"])
        dw1 = rng.choice(DIR_PHRASES[d1])
        dist_text, dist_val = rng.choice(DISTANCE_TEMPLATES[:12])
        d2 = rng.choice(["left", "right"])
        dw2 = rng.choice(DIR_PHRASES[d2])
        ang_text, ang_val = rng.choice(TURN_ANGLES)
        kind = rng.randint(0, 4)
        if kind == 0:
            add(f"пройди {dw1} {dist_text} и сделай фото", [{"name": "move_distance", "arguments": {"direction": d1, "distance_m": dist_val}}, {"name": "capture_photo", "arguments": {}}])
        elif kind == 1:
            add(f"повернись {dw2} на {ang_text}, затем пройди {dw1} {dist_text}", [{"name": "turn_robot", "arguments": {"direction": d2, "angle_deg": ang_val}}, {"name": "move_distance", "arguments": {"direction": d1, "distance_m": dist_val}}])
        elif kind == 2:
            gait = rng.choice(["walk", "trot", "creep"])
            gait_ru = {"walk": "обычную походку", "trot": "рысь", "creep": "крадущуюся походку"}[gait]
            dur_text, dur_val = rng.choice(DURATION_TEMPLATES[:8])
            add(f"включи {gait_ru} и иди {dw1} {dur_text}", [{"name": "set_gait", "arguments": {"gait": gait}}, {"name": "move_duration", "arguments": {"direction": d1, "duration_s": dur_val}}])
        elif kind == 3:
            add(f"включи свет, пройди {dw1} {dist_text} и сфоткай", [{"name": "set_led", "arguments": {"enabled": True}}, {"name": "move_distance", "arguments": {"direction": d1, "distance_m": dist_val}}, {"name": "capture_photo", "arguments": {}}])
        else:
            add(f"повернись {dw2} на {ang_text} и иди {dw1}", [{"name": "turn_robot", "arguments": {"direction": d2, "angle_deg": ang_val}}, {"name": "move", "arguments": {"direction": d1}}])

    return list(samples.values())


def _bucket_key(sample: dict[str, Any]) -> str:
    ans = sample["answers"]
    if not ans:
        return "offtopic"
    if len(ans) > 1:
        return "chain:" + "+".join(c["name"] for c in ans)
    c = ans[0]
    args = c.get("arguments") or {}
    if c["name"] == "set_speed":
        spd = int(args["speed"])
        if spd in (100, 180, 255) and not any(ch.isdigit() for ch in sample["query"]):
            return f"set_speed.{spd}"
        return "set_speed.num"
    for arg_key in ("direction", "gait", "preset", "action", "enabled"):
        if arg_key in args:
            return f"{c['name']}.{arg_key}={args[arg_key]}"
    return c["name"]


def select_balanced(samples: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        buckets.setdefault(_bucket_key(s), []).append(s)

    for items in buckets.values():
        rng.shuffle(items)

    priority_caps = {
        "chain:set_speed+move": 100,
        "chain:set_speed+move_distance": 115,
        "chain:set_speed+move_duration": 115,
        "chain:move_distance+capture_photo": 95,
        "chain:turn_robot+move_distance": 95,
        "chain:turn_robot+move": 95,
        "chain:set_gait+move_duration": 95,
        "chain:set_led+move_distance+capture_photo": 105,
        "chain:capture_photo+get_telemetry": 45,
        "navigate_to": 120,
        "set_speed.100": 75,
        "set_speed.180": 75,
        "set_speed.255": 75,
        "set_speed.num": 110,
    }
    for d in DIR_PHRASES:
        priority_caps[f"move.direction={d}"] = 75
        priority_caps[f"move_distance.direction={d}"] = 90
        priority_caps[f"move_duration.direction={d}"] = 90

    selected: dict[str, list[dict[str, Any]]] = {}
    for k, cap in priority_caps.items():
        if k in buckets:
            selected[k] = buckets[k][:min(len(buckets[k]), cap)]

    reserved = sum(len(v) for v in selected.values())
    other_keys = [k for k in buckets if k not in priority_caps]
    base_quota = max(50, (MAX_SAMPLES - reserved) // max(1, len(other_keys)))
    for k in other_keys:
        selected[k] = buckets[k][:min(len(buckets[k]), base_quota)]

    all_rows = [item for items in selected.values() for item in items]

    sim_raw_path = OUT_DIR / "sim_raw.json"
    if sim_raw_path.exists():
        seen_q = {r["query"] for r in all_rows}
        for item in json.loads(sim_raw_path.read_text(encoding="utf-8")).get("commands", []):
            q = " ".join(item["query"].strip().split())
            if q and q not in seen_q:
                seen_q.add(q)
                all_rows.append({"query": q, "tools": SPIDOR_TOOLS, "reasoning": "-", "answers": item["answers"]})

    rng.shuffle(all_rows)
    return all_rows


if __name__ == "__main__":
    rng = random.Random(SEED)
    cmd_rows = select_balanced(build_samples(rng), rng)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "cmds.jsonl"
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in cmd_rows) + "\n",
        encoding="utf-8",
    )
    print(f"cmds={len(cmd_rows)} -> {out_path}")
