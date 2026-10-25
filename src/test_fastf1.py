
import fastf1

YEAR = 2026
ROUND = 18

print("Loading 2026 United States GP qualifying...")

session = fastf1.get_session(YEAR, ROUND, "Q")
session.load()

results = session.results

print("\nSession loaded successfully.")
print("Drivers:", len(results))

columns = [
    column
    for column in [
        "Abbreviation",
        "FullName",
        "TeamName",
        "Position",
        "Q1",
        "Q2",
        "Q3",
    ]
    if column in results.columns
]

print(results[columns].to_string(index=False))
