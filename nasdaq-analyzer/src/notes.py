# This file manages user notes, allowing users to add, retrieve, and update notes for specific dates.

import pandas as pd
import os

notes_file = "notes.csv"

def load_notes():
    if os.path.exists(notes_file):
        return pd.read_csv(notes_file, parse_dates=['date'])
    return pd.DataFrame(columns=['date', 'note'])

def save_notes(notes_df):
    notes_df.to_csv(notes_file, index=False)

def get_notes_for_date(selected_date):
    notes_df = load_notes()
    return notes_df[notes_df['date'] == selected_date]['note'].tolist()

def add_note_for_date(selected_date, note):
    notes_df = load_notes()
    new_note = pd.DataFrame({'date': [selected_date], 'note': [note]})
    notes_df = pd.concat([notes_df, new_note], ignore_index=True)
    save_notes(notes_df)

def update_note_for_date(selected_date, old_note, new_note):
    notes_df = load_notes()
    notes_df.loc[(notes_df['date'] == selected_date) & (notes_df['note'] == old_note), 'note'] = new_note
    save_notes(notes_df)