from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


BASE_MODEL = "cointegrated/rubert-tiny2"
CLF_PATH = Path("models/offtopic_clf.joblib")

POLITE_COMMA_RE = re.compile(
    r",\s*(?:пожалуйста|плиз|пж|скорее|прямо\s+сейчас|сейчас|для\s+меня|робот|спидор)\b",
    re.IGNORECASE,
)

OUT_OF_DOMAIN_RE = re.compile(
    r"(?:\?\s*не\b|\bне,\s*забей\b|\bзабей\b|\bотмена\b|\bшучу\b|\bхотя\s+нет\b|\bпургу\b|"
    r"\b(?:инструкци\w*|игнорир\w+|забудь\b|промпт\w*|сортировк\w*|пузырьк\w*|питон\w*|python|"
    r"джава\w*|код\b|программ\w+|алгоритм\w*|массив\w*|граф\w*|интеграл\w*|уравнени\w*|корень\b|"
    r"квадратн\w*|синус\w*|косинус\w*|переведи\w*|расскажи\w*|объясни\w*|сочини\w*|напиши\w*|"
    r"стих\w*|сказк\w*|анекдот\w*|рецепт\w*|тьюринг\w*|эссе\b|"
    r"прыг\w*|подпрыг\w*|перепрыг\w*|сальто|бэкфлип\w*|флип\w*|взлет\w*|лети\b|летай\w*|плавай\w*|"
    r"хвост\w*|крыль\w*|ухо\b|ухом\b|уши\b|руки\b|ладош\w*|голово\w*|танцуй\w*|станцуй\w*|"
    r"спляш\w*|спой\b|песн\w*|шпагат\w*|кувырк\w*|"
    r"компьютер\w*|комп\b|ноутбук\w*|пк\b|телевизор\w*|телик\w*|телефон\w*|смартфон\w*|"
    r"браузер\w*|сайт\w*|впн\b|блютуз\w*|роутер\w*|вайфай\w*|музык\w*|музон\w*|музло\w*|трек\b|"
    r"радио\b|подкаст\w*|стрим\w*|канал\w*|звук\w*|громкост\w*|микрофон\w*|будильник\w*|таймер\w*|"
    r"минут\w*|час(?:а|ов|ам|ах|ы)?\b|окно\b|форточк\w*|балкон\w*|дверь\b|холодильник\w*|"
    r"чайник\w*|плит\w*|кофе\b|пив\w*|борщ\w*|пицц\w*|шаурм\w*|дошик\w*|погод\w*|"
    r"алиса\b|маруся\b|сири\b|гугл\b|эвакуатор\w*|закатай\w*)\b)",
    re.IGNORECASE,
)

DOMAIN_RESCUE_RE = re.compile(
    r"\b(?:рысь\w*|trot|creep|walk|фар[уы]\b|лампочк\w*|фонар\w*|светодиод\w*|подсветк\w*|"
    r"селфи|скриншот\w*|сфотк\w*|сфотографир\w*|телеметри\w*|гироскоп\w*|акселерометр\w*|"
    r"серв(?:ы|у|ам|ами|опривод\w*)|не\s+гони\b|притормози\w*|"
    r"присядь\w*|пригнись|останови\w*|стоп\b|стопэ\b|замри|крадись)\b",
    re.IGNORECASE,
)

GAIT_STEP_RE = re.compile(r"\b(?:обычн\w+|осторожн\w+|походк\w+)\s+шаг\w*\b", re.IGNORECASE)

COMPLEX_PATTERNS = re.compile(
    r"(\d|"
    r"\b(?:метр\w*|полметр\w*|полтора|полторы|сантиметр\w*|см|мм|"
    r"шаг(?:а|ов|ам|ами)?|"
    r"секунд\w*|полсекунд\w*|мс|"
    r"градус\w*|прямой\s+угол|девяност\w*|"
    r"координат\w*|точк\w*|маршрут\w*|навигаци\w*|"
    r"один|одну|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять|пар[ауы]|"
    r"затем|потом|после\s+чего)\b|"
    r"\sи\s|,)",
    re.IGNORECASE,
)

SPEED_MOD_RULES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\b(?:со\s+средней\s+скоростью|на\s+средней\s+скорости|в\s+обычном\s+темпе)\b", re.I), 180),
    (re.compile(r"\b(?:максимально\s+быстро|на\s+полной\s+скорости|газ\s+в\s+пол|побыстрее|быстро|резко)\b", re.I), 255),
    (re.compile(r"\b(?:не\s+спеша|помедленнее|медленно|потихоньку|аккуратно|осторожно)\b", re.I), 100),
]

MOVE_DIR_RE = re.compile(
    r"\b(?:вперёд|вперед|прямо(?!\s+сейчас)|назад|обратно|задним\s+ходом|влево|налево|вправо|направо)\b",
    re.I,
)
GAIT_CONTEXT_RE = re.compile(r"\b(?:походк\w*|режим\w*|рысь\w*|walk|trot|creep)\b", re.I)
NAV_QUERY_RE = re.compile(r"(?:\b(?:координат\w*|точк[уеи]|маршрут\w*|навигаци\w*)\b|\bx\s*=)", re.I)
TURN_VERB_RE = re.compile(r"\b(?:поверн\w*|разверн\w*|крутн\w*|поворач\w*|доверн\w*|разворот|оборот|кругом)\b", re.I)

CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,\s*затем\b|,\s*потом\b|,\s*после\s+чего\b|,\s*дальше\b|,\s*далее\b|,\s*следом\b|,\s*а\s+потом\b|,\s*а\s+затем\b|"
    r"\bзатем\b|\bпотом\b|\bпосле\s+чего\b|\bдальше\b|\bдалее\b|\bследом\b|\bа\s+потом\b|\bа\s+затем\b|"
    r",\s*и\b|\s+и\s+|,|;)\s*",
    re.I,
)

CLAUSE_ACTION_RE = re.compile(
    r"(?:\b(?:пройди|иди|шагай|шагни|двигайся|продвигайся|ступай|ползи|едь|беги|сдай|отступай|сдвигайся|принимай|"
    r"подвигайся|отойди|откатись|сместись|подайся|прошагай|проползи|поиди|подъедь|езжай|доберись|дойди|перейди|отправляйся|проложи|кинь\s+кости|"
    r"впер[её]?д\w*|впере[её]д\w*|прямо(?!\s+сейчас)|назад\w*|обратн\w*|задним\s+ходом|влев\w*|налев\w*|вправ\w*|направ\w*|"
    r"поверни\w*|разверни\w*|крутни\w*|поворачивай|доверни\w*|разворот|кругом|оборот|"
    r"стоп\w*|останови\w*|замри|хватит|тормози|стой|прекрати|отбой|отмени|"
    r"скорост\w*|ускорься|побыстрее|помедленнее|притормози|сбавь|прибавь|увеличь|уменьши|снизь|понизь|выкрути|ограничь|установи|поставь|задай|переключи\w*|"
    r"включи\w*|выключи\w*|отключи\w*|вруби|выруби|зажги|погаси|потуши|убери|подсвети|подай|сними|расслабь|отпусти|активируй|"
    r"встань|прими|верни\w*|отдохни|приляг|ложись|проснись|проснить|подъём|вставай|пробудись|просыпайся|поднимайся|подними\w*|"
    r"пригнись|присядь|опусти|выровняй|вытянись|наклони\w*|отклонись|задери|приподними|прижмись|откинь|"
    r"сделай|сфотографируй|сфоткай|фотку|селфи|скриншот|снимок|покажи|щёлкни|зафиксируй|зафоткай|"
    r"статус|проверь|жив\s+ли|состояние|"
    r"телеметри\w*|датчик\w*|показания|гироскоп\w*|акселерометр\w*|ориентаци\w*|"
    r"полметра|полтора\s+метра|прямой\s+угол)\b|"
    r"\d+\s*(?:см|мм|м\b|метр\w*|сантиметр\w*|шаг\w*|секунд\w*|с\b|мс\b|градус\w*|°))",
    re.I,
)

SINGLE_SYNONYM_PAIRS_RE = re.compile(
    r"\b(?:ориентаци\w*\s+и\s+координат\w*|дальномер\s+и\s+imu|позиция\s+и\s+углы|вебсокет\s+и\s+сервер|"
    r"сервер,\s*на\s+связи|то,\s*что\s+перед\s+тобой|"
    r"включи\s+моторы\s+и\s+встань|поднимись\s+и\s+включи\s+моторы|просыпайся,\s*включай\s+моторы|"
    r"поднимайся\s+и\s+активируй\s+сервы|ложись\s+и\s+расслабь\s+лапы|выруби\s+сервоприводы,\s*отдохни|"
    r"всё,\s*хватит,\s*стой|отмени\s+движение,\s*замри)\b",
    re.I,
)

WORD_NUMS: dict[str, float] = {
    "полтора": 1.5,
    "полторы": 1.5,
    "один": 1.0,
    "одну": 1.0,
    "одна": 1.0,
    "два": 2.0,
    "две": 2.0,
    "пару": 2.0,
    "пара": 2.0,
    "три": 3.0,
    "четыре": 4.0,
    "пять": 5.0,
    "шесть": 6.0,
    "семь": 7.0,
    "восемь": 8.0,
    "девять": 9.0,
    "десять": 10.0,
}

DIR_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:впер[её]?д\w*|впере[её]д\w*|прямо(?!\s+сейчас))\b", re.I), "forward"),
    (re.compile(r"\b(?:назад\w*|обратн\w*|задним\s+ходом)\b", re.I), "backward"),
    (re.compile(r"\b(?:влев\w*|налев\w*|левее|в\s+левую\s+сторону)\b", re.I), "left"),
    (re.compile(r"\b(?:вправ\w*|направ\w*|правее|в\s+правую\s+сторону)\b", re.I), "right"),
]


def strip_polite_commas(query: str) -> str:
    return POLITE_COMMA_RE.sub(lambda m: " " + m.group(0).lstrip(", "), query).strip()


def extract_speed_modifier(clause: str) -> tuple[int | None, str]:
    if GAIT_CONTEXT_RE.search(clause) or not MOVE_DIR_RE.search(clause):
        return None, clause
    for pat, spd in SPEED_MOD_RULES:
        m = pat.search(clause)
        if m:
            rest = (clause[: m.start()] + " " + clause[m.end() :]).strip(" ,")
            rest = " ".join(rest.split())
            if rest and MOVE_DIR_RE.search(rest):
                return spd, rest
    return None, clause


def split_chain_clauses(query: str) -> list[str]:
    q = strip_polite_commas(query)
    if SINGLE_SYNONYM_PAIRS_RE.search(q):
        return [q]
    if NAV_QUERY_RE.search(q):
        q_prot = re.sub(r"([+-]?\d+(?:\.\d+)?)\s*(?:,|и)\s*([+-]?\d+(?:\.\d+)?)", r"\1 __COORD_SEP__ \2", q)
        parts = [p.replace("__COORD_SEP__", ",").strip() for p in CLAUSE_SPLIT_RE.split(q_prot) if p.strip()]
    else:
        parts = [p.strip() for p in CLAUSE_SPLIT_RE.split(q) if p.strip()]
    if len(parts) <= 1:
        return [q]
    if all(CLAUSE_ACTION_RE.search(p) for p in parts):
        return parts
    return [q]


def parse_direction(text: str) -> str | None:
    for pat, d in DIR_MAP:
        if pat.search(text):
            return d
    return None


def parse_distance_m(text: str) -> float | None:
    t = text.lower()
    if GAIT_STEP_RE.search(t):
        return None
    if re.search(r"\bпол\s*метра\b|\bполметра\b", t):
        return 0.5
    if re.search(r"\bполтора\s+метра\b", t):
        return 1.5
    m = re.search(r"(?<![-\w])(\d+(?:\.\d+)?)\s*(?:сантиметр\w*|см)\b", t)
    if m:
        return round(float(m.group(1)) / 100.0, 4)
    m = re.search(r"(?<![-\w])(\d+(?:\.\d+)?)\s*(?:миллиметр\w*|мм)\b", t)
    if m:
        return round(float(m.group(1)) / 1000.0, 4)
    m = re.search(r"(?<![-\w])(\d+(?:\.\d+)?)\s*(?:метр\w*|м)\b", t)
    if m:
        return round(float(m.group(1)), 4)
    m = re.search(r"(?<![-\w])(\d+(?:\.\d+)?)\s*шаг(?:а|ов|ам|ами)?\b", t)
    if m:
        return round(float(m.group(1)) * 0.3, 4)
    for w, val in WORD_NUMS.items():
        if re.search(rf"\b{w}\s+(?:сантиметр\w*|см)\b", t):
            return round(val / 100.0, 4)
        if re.search(rf"\b{w}\s+метр\w*\b", t):
            return round(val, 4)
        if re.search(rf"\b{w}\s+шаг(?:а|ов|ам|ами)?\b", t):
            return round(val * 0.3, 4)
    if re.search(r"\bметр\b", t):
        return 1.0
    if re.search(r"\bшаг\b", t):
        return 0.3
    return None


def parse_duration_s(text: str) -> float | None:
    t = text.lower()
    if re.search(r"\bпол\s*секунд\w*\b|\bполсекунд\w*\b", t):
        return 0.5
    m = re.search(r"(?<![-\w])(\d+(?:\.\d+)?)\s*(?:миллисекунд\w*|мс)\b", t)
    if m:
        return round(float(m.group(1)) / 1000.0, 4)
    m = re.search(r"(?<![-\w])(\d+(?:\.\d+)?)\s*(?:секунд\w*|с)\b", t)
    if m:
        return round(float(m.group(1)), 4)
    for w, val in WORD_NUMS.items():
        if re.search(rf"\b{w}\s+секунд\w*\b", t):
            return round(val, 4)
    if re.search(r"\bсекунду\b", t):
        return 1.0
    return None


def parse_angle_deg(text: str) -> float | None:
    t = text.lower()
    if re.search(r"\bпрямой\s+угол\b|\bдевяност\w*\s+градус\w*\b", t):
        return 90.0
    m = re.search(r"(?<![-\w])(\d+(?:\.\d+)?)\s*(?:градус\w*|°)", t)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(?:на|угол)\s+(\d+(?:\.\d+)?)\b", t)
    if m and TURN_VERB_RE.search(t):
        return float(m.group(1))
    return None


def verify_and_fix_calls(clause: str, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    t = clause.lower()
    m_spd = re.search(r"\bскорост\w*\s+(?:до\s+|на\s+)?(\d+)\b", t)
    if m_spd and not MOVE_DIR_RE.search(t):
        return [{"name": "set_speed", "arguments": {"speed": max(1, min(255, int(m_spd.group(1))))}}]

    if NAV_QUERY_RE.search(t):
        nums = [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?", t)]
        if len(nums) == 2:
            return [{"name": "navigate_to", "arguments": {"x": nums[0], "y": nums[1]}}]

    dist = parse_distance_m(t)
    dur = parse_duration_s(t)
    ang = parse_angle_deg(t)
    d = parse_direction(t)

    if TURN_VERB_RE.search(t) and dist is None and dur is None:
        if d in ("left", "right"):
            return [{"name": "turn_robot", "arguments": {"direction": d, "angle_deg": ang if ang is not None else 90.0}}]
        if re.search(r"\b(?:полный\s+(?:оборот|разворот)|вокруг\s+себя|360)\b", t):
            return [{"name": "turn_robot", "arguments": {"direction": "right", "angle_deg": 360.0}}]
        return [{"name": "turn_robot", "arguments": {"direction": "right", "angle_deg": ang if ang is not None else 180.0}}]

    if d is not None and dist is not None and not TURN_VERB_RE.search(t):
        return [{"name": "move_distance", "arguments": {"direction": d, "distance_m": dist}}]

    if d is not None and dur is not None and not TURN_VERB_RE.search(t):
        return [{"name": "move_duration", "arguments": {"direction": d, "duration_s": dur}}]

    if re.search(r"\b(?:сфотк\w*|сфотографир\w*|фотк[уи]\b|селфи|скриншот\w*|снимок|сними\s+(?:кадр|то\b)|щёлкни\s+(?:камерой|фотку)|зафиксируй\s+картинку|зафотк\w*|покажи\s+(?:что\s+видишь|картинку))\b", t):
        return [{"name": "capture_photo", "arguments": {}}]

    if re.search(r"\b(?:стоп\w*|останови\w*|замри|хватит|тормози|стой(?!к)|прекрати\w*|отбой)\b", t):
        return [{"name": "stop_robot", "arguments": {}}]

    if re.search(r"\b(?:свет\w*|светодиод\w*|подсветк\w*|led\b|фонар\w*|ламп\w*|фар[уы]\b|освещени\w*)\b", t):
        if re.search(r"\b(?:выключи\w*|погаси\w*|потуши\w*|отключи\w*|убери\w*|выруби\w*)\b", t):
            return [{"name": "set_led", "arguments": {"enabled": False}}]
        if re.search(r"\b(?:включи\w*|зажги\w*|вруби\w*|подсвети\w*)\b", t):
            return [{"name": "set_led", "arguments": {"enabled": True}}]

    if not MOVE_DIR_RE.search(t) and re.search(r"\b(?:не\s+гони|притормози\w*|сбавь\s+(?:скорость|ход|темп)|помедленнее|потише|тише\s+едь)\b", t):
        return [{"name": "set_speed", "arguments": {"speed": 100}}]

    if re.search(r"\b(?:стартов\w+\s+стойк\w*|исходн\w+\s+(?:положени\w*|поз\w*|позици\w*)|начальн\w+\s+положени\w*|домой|\bhome\b|встань\s+ровно)\b", t):
        return [{"name": "power_pose", "arguments": {"action": "home"}}]

    if re.search(r"\b(?:приляг\w*|отдохн\w*|расслабь\w*|отпусти\s+серв\w*|\brelax\b|ложись)\b", t):
        return [{"name": "power_pose", "arguments": {"action": "relax"}}]

    if re.search(r"\b(?:задери\s+(?:нос|передн\w+)|отклонись\s+назад|наклони\w*\s+(?:корпус\s+)?назад|приподними\s+нос|подними\s+передн\w+|откинь\s+корпус\s+назад|tilt_back)\b", t):
        return [{"name": "set_pose_preset", "arguments": {"preset": "tilt_back"}}]

    if re.search(r"\b(?:нос\s+пониже|опусти\s+(?:нос|передн\w+)|наклони\w*\s+(?:корпус\s+|голову\s+|мордой\s+)?(?:вперёд|вперед|вниз)|посмотри\s+вниз|tilt_forward)\b", t):
        return [{"name": "set_pose_preset", "arguments": {"preset": "tilt_forward"}}]

    return calls


def is_complex_query(query: str) -> bool:
    q = strip_polite_commas(query)
    if SINGLE_SYNONYM_PAIRS_RE.search(q):
        return False
    if extract_speed_modifier(q)[0] is not None:
        return True
    q_check = GAIT_STEP_RE.sub("", q)
    return bool(COMPLEX_PATTERNS.search(q_check))


def atomic_label_for_row(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
    ans = row.get("answers") or []
    if len(ans) != 1 or is_complex_query(row["query"]):
        return "COMPLEX", None
    call = ans[0]
    name = call["name"]
    args = call.get("arguments") or {}
    if name in {"move_distance", "move_duration", "navigate_to"}:
        return "COMPLEX", None
    if name == "turn_robot" and float(args.get("angle_deg", 90.0)) not in (90.0, 180.0, 360.0):
        return "COMPLEX", None
    if name == "set_speed" and int(args.get("speed", 180)) not in (100, 180, 255):
        return "COMPLEX", None
    label = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
    return label, ans


class OfftopicFilter:
    def __init__(
        self,
        model_name: str = BASE_MODEL,
        clf_path: Path | str | None = CLF_PATH,
        threshold: float = 0.35,
        atomic_threshold: float = 0.25,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.threshold = threshold
        self.atomic_threshold = atomic_threshold

        loaded = joblib.load(clf_path) if clf_path and Path(clf_path).exists() else None
        if isinstance(loaded, dict):
            self.binary_clf = loaded.get("binary_clf")
            self.intent_clf = loaded.get("intent_clf")
            self.intent_map: dict[str, list[dict[str, Any]]] = loaded.get("intent_map") or {}
        else:
            self.binary_clf = loaded
            self.intent_clf = None
            self.intent_map = {}

    def embed(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        clean = [str(t).encode("utf-8", "ignore").decode("utf-8").strip() or "-" for t in texts]
        vecs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(clean), batch_size):
                enc = self.tokenizer(
                    clean[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                out = self.model(**enc)
                cls = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1)
                vecs.append(cls.cpu().numpy())
        return np.vstack(vecs) if vecs else np.empty((0, 312), dtype=np.float32)

    def is_command_vec(self, vec: np.ndarray, query: str | None = None) -> tuple[bool, float]:
        if self.binary_clf is None:
            raise RuntimeError("Classifier not loaded; run train_filter.py first.")
        if query is not None and OUT_OF_DOMAIN_RE.search(query):
            return False, 0.0
        p_cmd = float(self.binary_clf.predict_proba(vec)[0, 1])
        if query is not None and p_cmd < self.threshold and DOMAIN_RESCUE_RE.search(query):
            return True, max(p_cmd, self.threshold)
        return p_cmd >= self.threshold, p_cmd

    def prob_command(self, query: str) -> float:
        _, p_cmd = self.is_command_vec(self.embed([query]), query=query)
        return p_cmd

    def is_command(self, query: str) -> bool:
        ok, _ = self.is_command_vec(self.embed([query]), query=query)
        return ok

    def predict_atomic_vec(
        self,
        query: str,
        vec: np.ndarray,
        min_prob: float | None = None,
    ) -> list[dict[str, Any]] | None:
        if self.intent_clf is None or is_complex_query(query):
            return None
        probs = self.intent_clf.predict_proba(vec)[0]
        idx = int(np.argmax(probs))
        label = str(self.intent_clf.classes_[idx])
        prob = float(probs[idx])
        th = self.atomic_threshold if min_prob is None else min_prob
        if label != "COMPLEX" and prob >= th and label in self.intent_map:
            return [dict(c) for c in self.intent_map[label]]
        return None

    def _route_single_clause(self, clause: str, vec: np.ndarray, agent: Any) -> tuple[list[dict[str, Any]], str]:
        spd, rest_clause = extract_speed_modifier(clause)
        if spd is not None:
            speed_call = {"name": "set_speed", "arguments": {"speed": spd}}
            rest_vec = self.embed([rest_clause])
            sub_calls, sub_src = self._route_single_clause(rest_clause, rest_vec, agent)
            return [speed_call] + sub_calls, sub_src

        atomic = self.predict_atomic_vec(clause, vec)
        if atomic is not None:
            return verify_and_fix_calls(clause, atomic), "fast_atomic"

        agent.reset()
        resp = agent.complete(clause)
        native = resp.get("function_calls") or []
        if native:
            return verify_and_fix_calls(clause, native), "function_calls"
        suppressed = resp.get("suppressed_calls") or []
        if suppressed:
            return verify_and_fix_calls(clause, suppressed), "suppressed_calls"

        fallback = self.predict_atomic_vec(clause, vec, min_prob=0.15)
        if fallback is not None:
            return fallback, "fast_atomic"
        fixed = verify_and_fix_calls(clause, [])
        if fixed:
            return fixed, "suppressed_calls"
        return [], "empty"

    def route(self, query: str, agent: Any) -> tuple[list[dict[str, Any]], str]:
        if OUT_OF_DOMAIN_RE.search(query):
            return [], "filtered"

        clauses = split_chain_clauses(query)
        if len(clauses) > 1:
            vecs = self.embed(clauses)
            all_calls: list[dict[str, Any]] = []
            last_src = "fast_atomic"
            for cl, v in zip(clauses, vecs):
                v_row = v[None, :]
                ok, _ = self.is_command_vec(v_row, query=cl)
                if not ok:
                    return [], "filtered"
                cl_calls, cl_src = self._route_single_clause(cl, v_row, agent)
                if cl_src in ("function_calls", "suppressed_calls"):
                    last_src = cl_src
                for c in cl_calls:
                    if not all_calls or all_calls[-1] != c:
                        all_calls.append(c)
            return all_calls, (last_src if all_calls else "empty")

        clean_q = clauses[0]
        vec = self.embed([clean_q])
        ok, _ = self.is_command_vec(vec, query=clean_q)
        if not ok:
            return [], "filtered"
        return self._route_single_clause(clean_q, vec, agent)
