"""Каталог активів і провенанс (п.7).

Векторна БД зберігає «що схоже». Каталог зберігає «звідки це взялося»: хеш
джерела, шлях, час індексації та версії всіх моделей, якими артефакт
побудований. Для розслідування результат без цього непридатний — його
неможливо ані перевірити, ані відтворити.

Окремо тут живе підпис індексу. Вектори, побудовані іншим ембедером або з
іншою роздільністю NaFlex, лежать в іншому просторі; мовчазне змішування дало б
не помилку, а просто погані результати — найгірший різновид поломки.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

DEFAULT_CATALOG = Path("catalog.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id    TEXT PRIMARY KEY,          -- sha256 вмісту, а не шлях
    path        TEXT NOT NULL,
    media_type  TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    width       INTEGER,
    height      INTEGER,
    duration_ms INTEGER,
    indexed_at  INTEGER NOT NULL,
    profile     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assets_path ON assets(path);

CREATE TABLE IF NOT EXISTS frames (
    frame_id   TEXT PRIMARY KEY,
    asset_id   TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    ts_ms      INTEGER,
    shot_id    TEXT,
    width      INTEGER,
    height     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_frames_asset ON frames(asset_id);

-- Реєстр прототипів: формулювання, поріг і чи він калібрований.
-- Зберігається, бо саме поріг визначає межу рішення фасета, а отже —
-- відтворюваність результату в матеріалах справи.
-- Кеш sha256 за (шлях, розмір, час зміни). Без нього кожен прогін перечитує
-- КОЖЕН файл цілком заради відповіді «цей актив уже проіндексований», тобто
-- читає весь корпус, щоб нічого не зробити (ADR-015).
--
-- Ключ навмисно НЕ лише шлях: файл за тим самим шляхом міг змінитися, і тоді
-- старий хеш означав би не той актив. Розмір і mtime це ловлять.
CREATE TABLE IF NOT EXISTS file_hashes (
    path   TEXT PRIMARY KEY,
    size   INTEGER NOT NULL,
    mtime  REAL NOT NULL,
    sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prototypes (
    name        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,             -- category | attribute
    positive    TEXT NOT NULL,             -- JSON-масив формулювань
    negative    TEXT NOT NULL,
    threshold   REAL,
    calibrated  INTEGER NOT NULL DEFAULT 0,
    example_count INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    created_at  INTEGER NOT NULL
);

-- Версії моделей на кожен артефакт: без них результат неможливо відтворити
-- через півроку, коли ваги вже оновляться.
CREATE TABLE IF NOT EXISTS model_versions (
    scope      TEXT NOT NULL,             -- embed | detect | ocr | asr | face
    model_name TEXT NOT NULL,
    repo_id    TEXT NOT NULL,
    revision   TEXT NOT NULL,
    recorded_at INTEGER NOT NULL,
    PRIMARY KEY (scope, model_name, revision)
);
"""


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    path: str
    media_type: str
    size_bytes: int
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None


class SignatureMismatch(RuntimeError):
    """Індекс побудований іншими параметрами, ніж ті, якими його намагаються читати."""


class Catalog:
    """SQLite-каталог. Відкривається на кожну операцію, не тримає зʼєднання."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or DEFAULT_CATALOG)
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Одне зʼєднання на весь каталог, у режимі WAL.

        Раніше кожен виклик відкривав НОВЕ зʼєднання й закривав його після
        одного рядка: на знімок виходило три повні цикли «відкрити — записати —
        закрити», кожен із fsync журналу. WAL прибирає найдорожчу частину, а
        спільне зʼєднання — решту.

        `check_same_thread=False` потрібне, бо інжест уже читає файли й рахує
        ембединги в потоках (torch звільняє GIL). Записи в каталог короткі й
        серіалізуються самим SQLite.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            # WAL переживає перезапуск і лишається властивістю файлу, тож
            # виставляється один раз.
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        return self._conn

    @contextmanager
    def _session(self):
        """Спільне зʼєднання без закриття.

        Раніше тут стояло `closing(self._connect())`, тобто кожен виклик
        закривав зʼєднання після одного рядка. З одним зʼєднанням на прогін
        закривати його на кожному записі означало б скасувати весь виграш.
        """
        yield self._connect()

    def cached_hash(self, path, size: int, mtime: float) -> str | None:
        """sha256 з кешу, якщо файл не змінився.

        Порівнюються розмір і час зміни: збіг обох означає, що вміст той
        самий із практичною певністю, а розбіжність хоч одного змушує
        перечитати файл. Хибний збіг тут коштував би переплутаного активу,
        тому дешевшої перевірки (лише шлях) не досить.
        """
        with self._session() as conn:
            row = conn.execute(
                "SELECT size, mtime, sha256 FROM file_hashes WHERE path=?", (str(path),)
            ).fetchone()
        if not row or int(row["size"]) != int(size):
            return None
        # mtime у float; порівнюємо з допуском, бо файлові системи округлюють
        # його по-різному (APFS — до наносекунд, деякі мережеві — до секунди).
        if abs(float(row["mtime"]) - float(mtime)) > 1e-6:
            return None
        return str(row["sha256"])

    def remember_hash(self, path, size: int, mtime: float, digest: str) -> None:
        with self._session() as conn:
            conn.execute(
                "INSERT INTO file_hashes(path, size, mtime, sha256) VALUES(?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET size=excluded.size, "
                "mtime=excluded.mtime, sha256=excluded.sha256",
                (str(path), int(size), float(mtime), digest),
            )
            conn.commit()

    def _init_schema(self) -> None:
        with self._session() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    # ── підпис індексу ──────────────────────────────────────────────────────

    def signature(self) -> dict[str, Any] | None:
        with self._session() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='index_signature'").fetchone()
        return json.loads(row["value"]) if row else None

    def set_signature(self, signature: dict[str, Any]) -> None:
        with self._session() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('index_signature', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (json.dumps(signature, sort_keys=True),),
            )
            conn.commit()

    def corpus_threshold(self, collection: str, name: str) -> tuple[float | None, int] | None:
        """Збережена корпусна межа фасета й розмір корпусу, на якому її взято.

        Потрібна для ІНКРЕМЕНТАЛЬНОГО перерахунку, і це питання коректності, а
        не швидкості. Межа виводиться з РОЗПОДІЛУ корпусу; якщо після додавання
        одного знімка вивести її наново з самої лише нової партії, вийде межа
        іншої величини, і той самий фасет означатиме на нових точках не те, що
        на старих. Тому межа береться корпусна й зберігається.
        """
        key = f"corpus_threshold:{collection}:{name}"
        with self._session() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        data = json.loads(row["value"])
        # Межа може бути None — і це ЗНАННЯ, а не його відсутність: означає
        # «на цьому корпусі розриву немає» (як у `person`, який є майже скрізь).
        # Без запамʼятовування саме такі категорії щоразу запускали повний
        # обхід корпусу, тобто інкремент не працював саме там, де дорого.
        value = data["threshold"]
        return (None if value is None else float(value)), int(data.get("points", 0))

    def save_corpus_threshold(
        self, collection: str, name: str, threshold: float | None, points: int
    ) -> None:
        key = f"corpus_threshold:{collection}:{name}"
        with self._session() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps({"threshold": threshold, "points": points})),
            )
            conn.commit()

    def assert_compatible(self, signature: dict[str, Any]) -> None:
        """Звірити параметри індексу. Порожній індекс приймає будь-який підпис."""
        current = self.signature()
        if current is None:
            self.set_signature(signature)
            return
        if current == signature:
            return
        differing = {
            key: (current.get(key), signature.get(key))
            for key in set(current) | set(signature)
            if current.get(key) != signature.get(key)
        }
        raise SignatureMismatch(
            "Індекс побудований іншими параметрами — вектори лежать в іншому просторі.\n"
            + "\n".join(f"  {k}: в індексі {was!r}, зараз {now!r}" for k, (was, now) in differing.items())
            + "\nПотрібна переіндексація: vsearch index <шлях> --recreate"
        )

    # ── активи й кадри ──────────────────────────────────────────────────────

    def clear_assets(self) -> int:
        """Очистити активи й кадри, лишивши підпис і реєстр прототипів.

        Викликається при перебудові індексу. Без цього каталог пережив би
        видалення колекцій Qdrant і надалі стверджував, що актив уже
        проіндексований, — тож повторна індексація мовчки пропускала б файли,
        яких у векторному індексі вже немає. Розходження каталогу з індексом
        не дає помилки, лише порожні місця у видачі.
        """
        with self._session() as conn:
            removed = conn.execute("SELECT COUNT(*) c FROM assets").fetchone()["c"]
            conn.execute("DELETE FROM frames")
            conn.execute("DELETE FROM assets")
            conn.commit()
        return removed

    def add_asset(self, record: AssetRecord, profile: str) -> None:
        with self._session() as conn:
            conn.execute(
                "INSERT INTO assets(asset_id, path, media_type, size_bytes, width, height,"
                " duration_ms, indexed_at, profile) VALUES(?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(asset_id) DO UPDATE SET path=excluded.path,"
                " indexed_at=excluded.indexed_at, profile=excluded.profile",
                (
                    record.asset_id, record.path, record.media_type, record.size_bytes,
                    record.width, record.height, record.duration_ms, int(time.time()), profile,
                ),
            )
            conn.commit()

    def add_frame(
        self,
        frame_id: str,
        asset_id: str,
        *,
        ts_ms: int | None = None,
        shot_id: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        with self._session() as conn:
            conn.execute(
                "INSERT INTO frames(frame_id, asset_id, ts_ms, shot_id, width, height)"
                " VALUES(?,?,?,?,?,?) ON CONFLICT(frame_id) DO NOTHING",
                (frame_id, asset_id, ts_ms, shot_id, width, height),
            )
            conn.commit()

    def has_asset(self, asset_id: str) -> bool:
        with self._session() as conn:
            row = conn.execute("SELECT 1 FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        return row is not None

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        return dict(row) if row else None

    def assets(self) -> Iterator[dict[str, Any]]:
        with self._session() as conn:
            for row in conn.execute("SELECT * FROM assets ORDER BY indexed_at"):
                yield dict(row)

    def counts(self) -> dict[str, int]:
        with self._session() as conn:
            assets = conn.execute("SELECT COUNT(*) c FROM assets").fetchone()["c"]
            frames = conn.execute("SELECT COUNT(*) c FROM frames").fetchone()["c"]
        return {"assets": assets, "frames": frames}

    # ── провенанс моделей ───────────────────────────────────────────────────

    def record_model(self, scope: str, model_name: str, repo_id: str, revision: str) -> None:
        with self._session() as conn:
            conn.execute(
                "INSERT INTO model_versions(scope, model_name, repo_id, revision, recorded_at)"
                " VALUES(?,?,?,?,?) ON CONFLICT DO NOTHING",
                (scope, model_name, repo_id, revision, int(time.time())),
            )
            conn.commit()

    # ── реєстр прототипів ───────────────────────────────────────────────────

    def save_prototype(self, prototype) -> None:
        with self._session() as conn:
            conn.execute(
                "INSERT INTO prototypes(name, kind, positive, negative, threshold,"
                " calibrated, example_count, description, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(name) DO UPDATE SET positive=excluded.positive,"
                " negative=excluded.negative, threshold=excluded.threshold,"
                " calibrated=excluded.calibrated, example_count=excluded.example_count,"
                " description=excluded.description",
                (
                    prototype.name, prototype.kind,
                    json.dumps(list(prototype.positive), ensure_ascii=False),
                    json.dumps(list(prototype.negative), ensure_ascii=False),
                    prototype.threshold, int(prototype.calibrated),
                    prototype.example_count, prototype.description, int(time.time()),
                ),
            )
            conn.commit()

    def prototypes(self) -> list[dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute("SELECT * FROM prototypes ORDER BY kind, name").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["positive"] = json.loads(item["positive"])
            item["negative"] = json.loads(item["negative"])
            item["calibrated"] = bool(item["calibrated"])
            result.append(item)
        return result

    def model_versions(self) -> list[dict[str, Any]]:
        with self._session() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM model_versions ORDER BY scope")]
