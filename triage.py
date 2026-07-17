"""Train a simple classifier and categorize IT support ticket text."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).parent / "data" / "sample_tickets.csv"


def load_tickets(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return [row["ticket_text"] for row in rows], [row["category"] for row in rows]


def train_model() -> Pipeline:
    ticket_texts, categories = load_tickets(DATA_PATH)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )
    return model.fit(ticket_texts, categories)


def print_results(model: Pipeline, tickets: list[str]) -> None:
    predictions = model.predict(tickets)
    probabilities = model.predict_proba(tickets)
    rows = []
    for ticket, category, scores in zip(tickets, predictions, probabilities):
        rows.append((ticket, category, f"{max(scores):.1%}"))

    widths = [max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(("Ticket", "Category", "Confidence"))]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(border)
    print("|" + "|".join(f" {header:<{widths[i]}} " for i, header in enumerate(("Ticket", "Category", "Confidence"))) + "|")
    print(border)
    for row in rows:
        print("|" + "|".join(f" {value:<{widths[i]}} " for i, value in enumerate(row)) + "|")
    print(border)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickets", nargs="*", help="Ticket descriptions to classify")
    args = parser.parse_args()
    tickets = args.tickets or [
        "My virtual desktop is frozen after connecting",
        "VPN disconnects every few minutes",
        "Outlook will not send my email",
        "My password is rejected at sign in",
        "The office printer is out of paper",
    ]
    print_results(train_model(), tickets)


if __name__ == "__main__":
    main()
