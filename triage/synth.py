"""
Synthetic alert generator. FOR PIPELINE TESTING ONLY.

Numbers produced on this data say NOTHING about real-world performance or safety.
Templates are far simpler than real RCC text. Use it to verify that splitting,
training, gating, slicing, serving and logging all work end to end, then replace it
with real, redacted, outcome-linked data.

    python -m triage.synth --n 20000 --out data
"""
from __future__ import annotations

import argparse
import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .split import grouped_temporal_split

PLACES = ["Cape Point", "the north headland", "Port Elmo", "the river mouth", "Mount Karu", "the outer reef"]
SOURCES = ["owner_call", "emergency_contact", "rcc_log", "radio_relay", "asr_transcript"]

TEXT = {
    "en": {
        "DISTRESS": [
            "Owner reports vessel taking water, {n} persons on board, requesting assistance.",
            "Skipper says engine failed and the boat is drifting toward rocks near {place}.",
            "Emergency contact confirms aircraft is overdue and has not arrived at destination.",
            "Caller reports fire on board, crew preparing to abandon vessel.",
            "Persons are not safe. {n} people in the water near {place}.",
            "This is not a false alarm, we need help now.",
            "Hiker with injured leg cannot walk, {n} persons waiting for rescue.",
        ],
        "FALSE_ACCIDENTAL": [
            "Owner states beacon was accidentally activated while stowing gear, all persons safe.",
            "No one is in danger, the beacon was knocked by a crew member.",
            "There was no distress, activation was a mistake and the vessel is safe.",
        ],
        "FALSE_TEST_MAINT": [
            "Beacon was being tested during scheduled maintenance, no distress.",
            "Technician confirms self test went wrong during servicing, nobody at risk.",
            "Drill exercise in progress, beacon activated on purpose.",
        ],
        "FALSE_EQUIP": [
            "Owner reports battery fault and water ingress in the housing, everyone is safe ashore.",
            "Beacon malfunctioned in its bracket, vessel is safe in harbour.",
            "Unit transmitted after a mount failure, no emergency on board.",
        ],
        "FALSE_DISPOSAL_STORAGE": [
            "Beacon found in storage shed, vessel was sold last year.",
            "Old beacon was thrown away in household waste and activated at the dump.",
            "Beacon transported in a cargo container, no persons involved.",
        ],
        "AMBIG": [
            "Owner could not be reached by phone.",
            "Left voicemail, awaiting callback.",
            "Number is disconnected, trying emergency contact.",
            "Weather conditions are poor in the area.",
            "Registration details checked, vessel last seen two days ago.",
        ],
    },
    "es": {
        "DISTRESS": [
            "El propietario informa que el barco hace agua, {n} personas a bordo, solicita ayuda.",
            "Las personas no estan a salvo, hay incendio a bordo cerca de {place}.",
            "Contacto de emergencia confirma que la aeronave no ha llegado a su destino.",
        ],
        "FALSE_ACCIDENTAL": ["Nadie esta en peligro, la baliza se activo por accidente y todos estan a salvo."],
        "FALSE_TEST_MAINT": ["Prueba de mantenimiento programada, sin emergencia."],
        "FALSE_EQUIP": ["Fallo de la bateria, el barco esta a salvo en puerto."],
        "FALSE_DISPOSAL_STORAGE": ["La baliza estaba guardada en un almacen, el barco fue vendido."],
        "AMBIG": ["No se pudo localizar al propietario.", "Se dejo un mensaje, esperando respuesta."],
    },
    "fr": {
        "DISTRESS": [
            "Le proprietaire signale que le bateau prend l'eau, {n} personnes a bord, demande de l'aide.",
            "Les personnes ne sont pas en securite, incendie a bord pres de {place}.",
            "Le contact d'urgence confirme que l'avion n'est pas arrive a destination.",
        ],
        "FALSE_ACCIDENTAL": ["Personne n'est en danger, la balise a ete activee par erreur, tout le monde est sain et sauf."],
        "FALSE_TEST_MAINT": ["Test de maintenance prevu, aucune urgence."],
        "FALSE_EQUIP": ["Panne de batterie, le navire est en securite au port."],
        "FALSE_DISPOSAL_STORAGE": ["Balise retrouvee dans un hangar, le bateau a ete vendu."],
        "AMBIG": ["Le proprietaire est injoignable.", "Message laisse, en attente de rappel."],
    },
}

FINE_PRIORS = {
    "DISTRESS_CONFIRMED": 0.05, "DISTRESS_PROBABLE": 0.03, "UNRESOLVED": 0.03,
    "FALSE_ACCIDENTAL": 0.35, "FALSE_TEST_MAINT": 0.20, "FALSE_EQUIP": 0.14,
    "FALSE_DISPOSAL_STORAGE": 0.20,
}


def _asr_noise(s: str, rng: random.Random) -> str:
    words = [w for w in s.split() if rng.random() > 0.05]
    words = [w[:-1] + rng.choice("aeiou") if (len(w) > 3 and rng.random() < 0.04) else w for w in words]
    return " ".join(words)


def _pick(lang, key, rng):
    pool = TEXT[lang][key]
    return rng.choice(pool).format(n=rng.randint(1, 8), place=rng.choice(PLACES))


def _artifacts(fine, lang, rng):
    key = "DISTRESS" if fine.startswith("DISTRESS") else fine
    if fine == "UNRESOLVED":
        p_amb = 1.0
    elif fine == "DISTRESS_PROBABLE":
        p_amb = 0.5
    else:
        p_amb = 0.3
    k = rng.randint(1, 4)
    arts = []
    for _ in range(k):
        use_amb = rng.random() < p_amb
        arts.append((rng.choice(SOURCES), _pick(lang, "AMBIG" if use_amb else key, rng)))
    return [(s, _asr_noise(t, rng) if s == "asr_transcript" else t) for s, t in arts]


def generate(n: int, seed: int = 7, label_noise: float = 0.0) -> pd.DataFrame:
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    fines, probs = list(FINE_PRIORS), list(FINE_PRIORS.values())
    n_beacons = max(10, n // 4)
    zipf_w = 1.0 / np.arange(1, n_beacons + 1) ** 0.8
    zipf_w /= zipf_w.sum()
    t0 = datetime(2023, 1, 1)
    rows = []
    for i in range(n):
        fine = rng.choices(fines, probs)[0]
        lang = rng.choices(["en", "es", "fr"], [0.7, 0.15, 0.15])[0]
        distress = fine.startswith("DISTRESS") or fine == "UNRESOLVED"
        bidx = int(nrng.choice(n_beacons, p=zipf_w)) if not distress else rng.randrange(n_beacons)
        beacon = hashlib.sha256(f"b{bidx}".encode()).hexdigest()[:16]
        btype = rng.choices(["EPIRB", "ELT", "PLB"], [0.4, 0.3, 0.3])[0]
        env = {"EPIRB": "sea", "ELT": "air", "PLB": rng.choice(["sea", "land"])}[btype]
        motion = rng.choices(["moving", "stationary", "unknown"], [0.5, 0.3, 0.2] if distress else [0.15, 0.7, 0.15])[0]
        reg = rng.random() < (0.9 if distress else 0.75)
        arts = _artifacts(fine, lang, rng)
        meta = f"[META] {btype} {env} {motion} registered={reg}"
        text = meta + " " + " ".join(f"[{s.upper()}] {t}" for s, t in arts)
        label = int(distress)
        if label_noise and rng.random() < label_noise:
            label = 1 - label
        rows.append({
            "alert_id": f"A{i:07d}", "beacon_id": beacon,
            "timestamp": (t0 + timedelta(seconds=rng.randrange(730 * 86400))).isoformat(),
            "administration": rng.choice(["ADM_A", "ADM_B", "ADM_C"]), "language": lang,
            "beacon_type": btype, "env_class": env,
            "source_mix": "typed" if all(s != "asr_transcript" for s, _ in arts) else "asr",
            "text": text, "label": label, "fine_label": fine,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--label-noise", type=float, default=0.0)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    df = generate(a.n, a.seed, a.label_noise)
    tr, va, te = grouped_temporal_split(df)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, d in zip(["train", "val", "test"], [tr, va, te]):
        d.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: n={len(d)} positives={int(d.label.sum())} ({d.label.mean():.1%})")
    print("SYNTHETIC DATA: for pipeline testing only. Results are not evidence of safety.")


if __name__ == "__main__":
    main()
