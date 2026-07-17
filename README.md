# IT Ticket Triage Classifier

A small machine-learning project that categorizes IT support tickets as **VDI**, **Network**, **Outlook**, **Login**, or **Other**. It uses a scikit-learn pipeline with TF-IDF text features and a logistic regression classifier.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Classify one or more tickets from the command line:

```powershell
python triage.py "My virtual desktop is frozen" "VPN disconnects every few minutes"
```

Run the built-in examples when no ticket text is supplied:

```powershell
python triage.py
```

## Example output

```text
+--------------------------------------+----------+------------+
| Ticket                               | Category | Confidence |
+--------------------------------------+----------+------------+
| My virtual desktop is frozen         | VDI      | 42.6%      |
| VPN disconnects every few minutes    | Network  | 40.1%      |
+--------------------------------------+----------+------------+
```

The included dataset is intentionally small and synthetic, so confidence scores and accuracy are illustrative rather than production-ready. Add labeled tickets to `data/sample_tickets.csv` to improve the model.

## Project structure

```text
it-ticket-triage/
|-- data/
|   `-- sample_tickets.csv
|-- README.md
|-- requirements.txt
`-- triage.py
```

## Privacy

The sample tickets are fictional. Remove names, email addresses, device identifiers, and other sensitive information before using real support data.
