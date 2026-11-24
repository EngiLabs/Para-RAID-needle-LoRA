from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from tools_schema import SPIDOR_TOOLS


OUT_DIR = Path("data")
MAX_SAMPLES = 1400
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
    (["со средней скоростью", "обычно", "нормально"], 180),
    (["быстро", "побыстрее", "максимально быстро", "на полной скорости", "резко"], 255),
]

DISTANCE_TEMPLATES: list[tuple[str, float]] = [
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
    ("150 см", 1.5),
    ("200 мм", 0.2),
    ("500 мм", 0.5),
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
    ("90 градусов", 90.0),
    ("90°", 90.0),
    ("девяносто градусов", 90.0),
    ("прямой угол", 90.0),
    ("180 градусов", 180.0),
    ("180°", 180.0),
]

SIMPLE_COMMANDS: list[tuple[list[str], str, dict[str, Any]]] = [
    (
        ["стоп", "остановись", "замри", "хватит", "тормози", "стой", "прекрати движение", "аварийная остановка", "стопэ", "не двигайся"],
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
        ["встань в начальное положение", "прими исходную позу", "домой", "home", "верни ноги в начальное положение", "встань ровно", "исходная позиция"],
        "power_pose",
        {"action": "home"},
    ),
    (
        ["расслабь сервы", "отключи моторы", "расслабь ноги", "relax", "убери усилие с сервоприводов", "отдохни"],
        "power_pose",
        {"action": "relax"},
    ),
    (
        ["включи сервы", "проснить", "подъём", "wake", "активируй сервоприводы", "включи моторы и встань"],
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
        ["наклонись вперёд", "наклон вперед", "посмотри вниз корпусом", "поза tilt_forward"],
        "set_pose_preset",
        {"preset": "tilt_forward"},
    ),
    (
        ["наклонись назад", "отклонись назад", "задери нос", "поза tilt_back"],
        "set_pose_preset",
        {"preset": "tilt_back"},
    ),
    (
        ["включи свет", "включи светодиод", "зажги подсветку", "вруби led", "подсветку включи"],
        "set_led",
        {"enabled": True},
    ),
    (
        ["выключи свет", "погаси светодиод", "отключи подсветку", "выруби led", "свет выключи"],
        "set_led",
        {"enabled": False},
    ),
    (
        ["сделай фото", "сфотографируй", "сфоткай", "фотку", "сними кадр", "покажи что видишь", "сделай снимок с камеры", "щёлкни камерой"],
        "capture_photo",
        {},
    ),
    (
        ["статус сервера", "проверь связь", "робот подключен", "жив ли сервер", "состояние сервера", "какой статус подключения", "проверь коннект"],
        "get_status",
        {},
    ),
    (
        ["телеметрия", "запроси телеметрию", "покажи датчики", "что на датчиках", "показания imu", "данные с ик датчиков", "какая ориентация и координаты", "сними телеметрию с робота"],
        "get_telemetry",
        {},
    ),
]

OFF_TOPIC_QUERIES: list[str] = [
    "привет", "здравствуй", "как дела", "как настроение", "что умеешь", "кто тебя создал",
    "расскажи анекдот", "расскажи шутку", "какая сегодня погода", "будет ли дождь",
    "сколько сейчас времени", "какое сегодня число", "что такое обратная кинематика",
    "объясни теорему кастилиано", "напиши стихотворение про пп жук", "покажи заряд батареи",
    "какой заряд аккумулятора", "сколько процентов заряда осталось", "какой вольтаж аккумулятора",
    "сколько вольт на батарее", "пора заряжаться, сколько вольт", "проверь уровень батареи",
    "принеси кофе на кухню", "свари чай", "подпрыгни на два метра", "взлети вверх",
    "включи музыку", "поставь будильник на 8 утра", "позвони маме", "отправь сообщение другу",
    "открой браузер", "переведи это на английский", "реши квадратное уравнение",
    "посоветуй фильм на вечер", "спокойной ночи", "спасибо",
]

POLITE_PREFIXES = ["", "пожалуйста ", "робот ", "эй спидор ", "спидор ", "можешь ", "давай ", "срочно ", "быстро "]
POLITE_SUFFIXES = ["", " пожалуйста", "!", " скорее", " прямо сейчас", " для меня", " сейчас"]


def build_samples(rng: random.Random) -> list[dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}

    def add(query: str, answers: list[dict[str, Any]]) -> None:
        q = " ".join(query.strip().split())
        if q and q not in samples:
            samples[q] = {"query": q, "tools": SPIDOR_TOOLS, "answers": answers}

    for direction, dir_words in DIR_PHRASES.items():
        for dw in dir_words:
            for verb in MOVE_VERBS:
                add(f"{verb} {dw}", [{"name": "move_robot", "arguments": {"direction": direction}}])
                for dist_text, dist_val in DISTANCE_TEMPLATES:
                    for order in (f"{verb} {dw} {dist_text}", f"{verb} {dist_text} {dw}", f"{dw} на {dist_text}"):
                        add(order, [{"name": "move_robot", "arguments": {"direction": direction, "distance_m": dist_val}}])
                for dur_text, dur_val in DURATION_TEMPLATES:
                    for order in (f"{verb} {dw} {dur_text}", f"{verb} {dw} на {dur_text}", f"{dw} {dur_text}"):
                        add(order, [{"name": "move_robot", "arguments": {"direction": direction, "duration_s": dur_val}}])
                for mod_words, spd in SPEED_MODIFIERS:
                    for mw in mod_words:
                        add(f"{verb} {dw} {mw}", [{"name": "move_robot", "arguments": {"direction": direction, "speed": spd}}])
                        dist_text, dist_val = rng.choice(DISTANCE_TEMPLATES)
                        add(f"{verb} {dw} {dist_text} {mw}", [{"name": "move_robot", "arguments": {"direction": direction, "distance_m": dist_val, "speed": spd}}])

    for direction in ("left", "right"):
        for dw in DIR_PHRASES[direction]:
            for tv in ("поверни", "повернись", "развернись", "крутнись", "поворачивай"):
                for ang_text, ang_val in TURN_ANGLES:
                    add(f"{tv} {dw} на {ang_text}", [{"name": "turn_robot", "arguments": {"direction": direction, "angle_deg": ang_val}}])
                    add(f"{tv} на {ang_text} {dw}", [{"name": "turn_robot", "arguments": {"direction": direction, "angle_deg": ang_val}}])

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

    for _ in range(280):
        d1 = rng.choice(["forward", "backward"])
        dw1 = rng.choice(DIR_PHRASES[d1])
        dist_text, dist_val = rng.choice(DISTANCE_TEMPLATES[:12])
        d2 = rng.choice(["left", "right"])
        dw2 = rng.choice(DIR_PHRASES[d2])
        ang_text, ang_val = rng.choice(TURN_ANGLES)
        kind = rng.randint(0, 4)
        if kind == 0:
            add(f"пройди {dw1} {dist_text} и сделай фото", [{"name": "move_robot", "arguments": {"direction": d1, "distance_m": dist_val}}, {"name": "capture_photo", "arguments": {}}])
        elif kind == 1:
            add(f"повернись {dw2} на {ang_text}, затем пройди {dw1} {dist_text}", [{"name": "turn_robot", "arguments": {"direction": d2, "angle_deg": ang_val}}, {"name": "move_robot", "arguments": {"direction": d1, "distance_m": dist_val}}])
        elif kind == 2:
            gait = rng.choice(["walk", "trot", "creep"])
            gait_ru = {"walk": "обычную походку", "trot": "рысь", "creep": "крадущуюся походку"}[gait]
            dur_text, dur_val = rng.choice(DURATION_TEMPLATES[:8])
            add(f"включи {gait_ru} и иди {dw1} {dur_text}", [{"name": "set_gait", "arguments": {"gait": gait}}, {"name": "move_robot", "arguments": {"direction": d1, "duration_s": dur_val}}])
        elif kind == 3:
            add(f"включи свет, пройди {dw1} {dist_text} и сфоткай", [{"name": "set_led", "arguments": {"enabled": True}}, {"name": "move_robot", "arguments": {"direction": d1, "distance_m": dist_val}}, {"name": "capture_photo", "arguments": {}}])
        else:
            add("сделай фото и проверь телеметрию", [{"name": "capture_photo", "arguments": {}}, {"name": "get_telemetry", "arguments": {}}])

    for q in OFF_TOPIC_QUERIES:
        add(q, [])
        for pref in ("робот, ", "спидор, ", "пожалуйста, ", "скажи, ", "эй, "):
            add(f"{pref}{q}", [])

    return list(samples.values())


def split_balanced(samples: list[dict[str, Any]], rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        ans = s["answers"]
        key = "offtopic" if not ans else ("chain" if len(ans) > 1 else ans[0]["name"])
        buckets.setdefault(key, []).append(s)

    for items in buckets.values():
        rng.shuffle(items)

    priority_caps = {"move_robot": 320, "chain": 200, "offtopic": 160, "turn_robot": 120}
    selected: dict[str, list[dict[str, Any]]] = {}
    remaining = MAX_SAMPLES

    base_quota = max(20, MAX_SAMPLES // max(1, len(buckets)))
    for k in buckets:
        if k not in priority_caps:
            take = min(len(buckets[k]), max(base_quota, 70))
            selected[k] = buckets[k][:take]
            remaining -= take

    cap_sum = sum(priority_caps[k] for k in priority_caps if k in buckets) or 1
    for k, cap in priority_caps.items():
        if k in buckets:
            selected[k] = buckets[k][:min(len(buckets[k]), max(20, round(remaining * (cap / cap_sum))))]

    train, val = [], []
    for items in selected.values():
        n_val = max(1, int(round(len(items) * VAL_RATIO))) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


rng = random.Random(SEED)
train_rows, val_rows = split_balanced(build_samples(rng), rng)

OUT_DIR.mkdir(parents=True, exist_ok=True)
for name, rows in (("train.jsonl", train_rows), ("val.jsonl", val_rows)):
    (OUT_DIR / name).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
print(f"train={len(train_rows)}, val={len(val_rows)} -> {OUT_DIR}")
