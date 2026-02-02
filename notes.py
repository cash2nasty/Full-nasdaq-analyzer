import json
from pathlib import Path
from datetime import date
import pandas as pd


NOTES_FILE = Path("nasdaq_notes.json")
CLOSE_HISTORY_FILE = Path("nasdaq_close_history.csv")


def load_notes() -> dict:
    if NOTES_FILE.exists():
        with open(NOTES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_notes(notes: dict):
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, indent=2)


def add_note_for_date(note_date: date, text: str):
    notes = load_notes()
    key = note_date.isoformat()
    notes.setdefault(key, [])
    notes[key].append(text)
    save_notes(notes)


def get_notes_for_date(note_date: date):
    notes = load_notes()
    return notes.get(note_date.isoformat(), [])


def update_close_history(df: pd.DataFrame):
    latest_date = df.index[-1]
    latest_close = df.loc[latest_date, "Close"]
    row = pd.DataFrame([{"Date": latest_date.isoformat(), "Close": latest_close}])

    if CLOSE_HISTORY_FILE.exists():
        existing = pd.read_csv(CLOSE_HISTORY_FILE)
        if latest_date.isoformat() in existing["Date"].values:
            return
        updated = pd.concat([existing, row], ignore_index=True)
    else:
        updated = row

    updated.to_csv(CLOSE_HISTORY_FILE, index=False)