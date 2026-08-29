"""Прив'язка ще не реалізованих сценаріїв (@wip).

Сенс: сценарії з features/ мають бути ВИДИМИМИ й ЧЕРВОНИМИ, а не лежати
текстом, про який усі забули. Тому вони прив'язані до pytest, але типово
деселектовані через `-m 'not wip'` у pyproject.toml.

    pytest            — CI: тільки реалізоване, має бути зелено
    pytest -m wip     — backlog: показує, які кроки ще не написані

У міру реалізації тег @wip знімається з конкретного функціоналу, і його
сценарії переїжджають у звичайний прогін.
"""

from __future__ import annotations

from pytest_bdd import scenarios

# semantic.feature і multilingual.feature переїхали в test_search_steps.py —
# вони реалізовані на M1.
scenarios("faces.feature")
scenarios("video.feature")
scenarios("performance.feature")
