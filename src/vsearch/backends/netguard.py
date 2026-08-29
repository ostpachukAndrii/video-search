"""Запобіжник мережі: доказ вимоги п.9 замість обіцянки.

Обгортає socket так, що будь-яка спроба вийти за межі loopback кидає виняток.
Loopback свідомо лишається дозволеним: Qdrant у контурі працює на localhost, і
вимога звучить «немає доступу до інтернету», а не «немає сокетів взагалі».

Використання:
    with no_network():
        run_full_pipeline()   # впаде, якщо хоч щось спробує в мережу
"""

from __future__ import annotations

import ipaddress
import socket
from contextlib import contextmanager
from typing import Any, Iterator

_LOOPBACK_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost", ""}


class NetworkAccessDenied(RuntimeError):
    """Код спробував вийти в мережу там, де мережі не буде."""


def _is_loopback(host: Any) -> bool:
    if not isinstance(host, str):
        return False
    if host.lower() in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _describe(address: Any) -> str:
    if isinstance(address, tuple) and address:
        return f"{address[0]}:{address[1] if len(address) > 1 else '?'}"
    return repr(address)


@contextmanager
def no_network(*, allow_loopback: bool = True) -> Iterator[None]:
    """Заборонити зовнішні зʼєднання на час блоку."""
    real_socket_cls = socket.socket
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def _check(address: Any, *, via: str) -> None:
        host = address[0] if isinstance(address, tuple) and address else address
        if allow_loopback and _is_loopback(host):
            return
        raise NetworkAccessDenied(
            f"Спроба мережевого зʼєднання через {via} до {_describe(address)}.\n"
            f"У середовищі виконання мережі немає (п.9). Ваги й дані мають приїхати "
            f"на етапі збірки образу за models/manifest.lock."
        )

    class GuardedSocket(real_socket_cls):  # type: ignore[misc, valid-type]
        def connect(self, address: Any) -> None:  # noqa: D102
            _check(address, via="socket.connect")
            super().connect(address)

        def connect_ex(self, address: Any) -> int:  # noqa: D102
            _check(address, via="socket.connect_ex")
            return super().connect_ex(address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any):
        _check(address, via="socket.create_connection")
        return real_create_connection(address, *args, **kwargs)

    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any):
        # DNS-резолв зовнішнього імені сам по собі вже означає похід у мережу.
        if not (allow_loopback and _is_loopback(host)):
            _check((host, port), via="socket.getaddrinfo")
        return real_getaddrinfo(host, port, *args, **kwargs)

    socket.socket = GuardedSocket  # type: ignore[assignment, misc]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = real_socket_cls  # type: ignore[assignment, misc]
        socket.create_connection = real_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = real_getaddrinfo  # type: ignore[assignment]
