"""Крокові визначення для features/offline.feature — доказ вимоги п.9."""

from __future__ import annotations

import os
import shutil
import socket

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from vsearch.backends.netguard import NetworkAccessDenied, no_network
from vsearch.backends.registry import ModelNotFetched, ModelRegistry

scenarios("offline.feature")


@given("заблоковані зовнішні мережеві зʼєднання", target_fixture="netguard")
def _blocked_network(context):
    guard = no_network()
    guard.__enter__()
    yield guard
    guard.__exit__(None, None, None)


@when(parsers.parse('компонент намагається зʼєднатися з "{host}"'))
def _try_connect(context, netguard, host):
    try:
        socket.getaddrinfo(host, 443)
        context["denied"] = False
    except NetworkAccessDenied as exc:
        context["denied"] = True
        context["error"] = str(exc)
    except OSError:
        # DNS не розвʼязався з інших причин — запобіжник тут ні до чого.
        context["denied"] = False


@then("зʼєднання має бути відхилене")
def _connection_denied(context):
    assert context["denied"], "запобіжник пропустив зовнішнє зʼєднання"
    assert "мережі немає" in context["error"]


@then("зʼєднання не має бути відхилене запобіжником")
def _connection_allowed(context):
    assert not context["denied"], (
        "запобіжник заблокував loopback — Qdrant у контурі не підніметься"
    )


@when("я читаю маніфест моделей")
def _read_manifest(context):
    context["registry"] = ModelRegistry()


@then(parsers.parse("маніфест має містити щонайменше {count:d} записів"))
def _manifest_size(context, count):
    assert len(context["registry"].names) >= count


@then("кожен запис має мати вказану ліцензію")
def _each_has_license(context):
    missing = [e.name for e in context["registry"] if not e.license]
    assert not missing, f"без ліцензії: {missing}"


@then("кожен запис має мати локальну теку")
def _each_has_local_dir(context):
    missing = [e.name for e in context["registry"] if not e.local_dir]
    assert not missing, f"без local_dir: {missing}"


@when("я імпортую пакет vsearch")
def _import_package(context):
    import vsearch  # noqa: F401 — імпорт має побічний ефект

    context["imported"] = True


@then(parsers.parse('змінна оточення "{name}" має дорівнювати "{value}"'))
def _env_equals(context, name, value):
    assert os.environ.get(name) == value, (
        f"{name}={os.environ.get(name)!r}, очікувалося {value!r}"
    )


@when(parsers.parse('я прошу в реєстру модель за іменем "{name}"'))
def _ask_by_repo_id(context, name):
    registry = ModelRegistry()
    with pytest.raises(KeyError) as excinfo:
        registry.entry(name)
    context["error"] = str(excinfo.value)


@then("реєстр має відмовити і перелічити відомі імена")
def _registry_refuses(context):
    assert "не описана" in context["error"]
    assert "florence2_base" in context["error"], "підказка має перелічувати відомі імена"


@when(parsers.parse('я прошу локальний шлях до незавантаженої моделі "{name}"'))
def _ask_missing_weights(context, name, tmp_path):
    # Реєстр навмисно спрямований на порожню теку з копією маніфесту: інакше
    # на машині, де ваги вже є, сценарій просто пропускався б — і перестав би
    # захищати саме той випадок, заради якого написаний.
    empty = tmp_path / "models"
    empty.mkdir()
    shutil.copy(ModelRegistry().models_dir / "manifest.lock", empty / "manifest.lock")

    registry = ModelRegistry(empty)
    assert not registry.is_fetched(name)
    with pytest.raises(ModelNotFetched) as excinfo:
        registry.local_path(name)
    context["error"] = str(excinfo.value)


@then("помилка має пояснити, що ваги приїжджають на етапі збірки образу")
def _explains_build_phase(context):
    assert "етапі збірки образу" in context["error"]
    assert "fetch_models.py" in context["error"], "підказка має містити конкретну команду"
