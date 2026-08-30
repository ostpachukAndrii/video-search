"""Gradio-інтерфейс для перевірки прототипу.

Призначення — дати змогу побачити те, що інакше видно лише в числах: чи
справді знайдено ТОЙ обʼєкт, чи спрацювало заперечення, чи не деградував
пошук мовчки. Тому кожен результат показується з рамкою, а розбір запиту —
окремою панеллю поруч із видачею.

Моделі вантажаться ліниво: інтерфейс має підніматися за секунду, а не за
хвилину очікування 4.5 ГБ ваг.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from vsearch.api import render
from vsearch.config import PROFILES, get_profile
from vsearch.search.retrieve import MIN_PROBABILITY

logger = logging.getLogger(__name__)

PROFILE_NAMES = list(PROFILES)
SCOPES = ["auto", "regions", "frames", "both"]

_searcher_instance = None


def _searcher():
    """Єдиний пошуковець на профіль, під який зібрано індекс.

    Профіль більше не обирається при пошуку: індекс один, його роздільність
    зафіксована при індексації, а перемикач змінював лише глибину відбору,
    обіцяючи зміну якості (див. `do_search`).
    """
    global _searcher_instance
    if _searcher_instance is None:
        from vsearch.search.retrieve import Searcher

        _searcher_instance = Searcher()
    return _searcher_instance


# ── пошук ───────────────────────────────────────────────────────────────────


def do_search(query: str, limit: int, use_parser: bool, category: str,
              show_weak: bool = False, best_only: bool = False,
              refine: bool = False, similar: list[str] | None = None,
              unlike: list[str] | None = None):
    """Генератор (галерея, таблиця, статус, розбір, повідомлення, стан).

    Генератор, а не функція, заради уточнення обумовленою детекцією: воно
    коштує ~1.6 с на кадр. Перша видача йде одразу, уточнена заміщає її по
    кадру.

    Профіль сюди більше НЕ передається. Індекс зібраний під один профіль, і
    його роздільність зафіксована при індексації; перемикач у пошуку лише
    змінював глибину відбору, але обіцяв зміну якості. Гірше: вибір `fast`
    мовчки вимикав лексичний шар (`use_ocr=False`), і запити на номери й
    прізвища переставали знаходитись. Профіль лишився там, де він справді
    вирішує, — у вкладці індексації.
    """
    # Текстові поля приходять як None, поки їх не торкалися: Gradio не
    # підставляє порожній рядок, якщо не заданий `value`. Один `.strip()` на
    # None падав просто у видачу.
    query = (query or "").strip()
    category = (category or "").strip()
    if not query:
        yield ([], [], "_Введіть запит._", "", "", [], "")
        return

    started = time.perf_counter()

    # Розбір показується ОКРЕМИМ кроком: пошук чекає на LLM, і порожній екран
    # на цей час виглядає як зависання.
    parsed_preview = ""
    if use_parser:
        try:
            from vsearch.search.parse import get_parser

            # get_parser(), а НЕ QueryParser(): інтерфейс створював другий
            # екземпляр на кожен пошук, тобто вантажив GGUF удруге й обходив
            # кеш префікса. Розбір коштував 10 с там, де мав коштувати 1.8.
            parser = get_parser()
            if parser.is_available() and not parser.should_bypass(query):
                yield ([], [], "⏳ розбираю запит…", "", "", [], "")
                parsed_preview = render.describe_parse(parser.parse(query))
        except Exception:  # noqa: BLE001 — розбір не мусить валити пошук
            logger.debug("попередній розбір не вдався", exc_info=True)
    yield ([], [], "⏳ шукаю по індексу…", parsed_preview, "", [], "")

    try:
        response = _searcher().search(
            query,
            limit=int(limit),
            parse=use_parser,
            category=(category or None),
            min_probability=0.0 if show_weak else MIN_PROBABILITY,
            similar_to=list(similar or []),
            unlike=list(unlike or []),
        )
    except Exception as exc:  # noqa: BLE001 — помилку показуємо, а не ховаємо
        logger.exception("пошук не вдався")
        yield ([], [], f"❌ {type(exc).__name__}: {exc}", parsed_preview, "", [], "")
        return

    def package(resp, extra: str = ""):
        shots, rows = [], []
        for rank, res in enumerate(resp.results, start=1):
            picture = render.annotate(res, best_only=best_only)
            if picture is None:
                # Не пропускаємо: результат, який неможливо намалювати, все
                # одно є знахідкою. Мовчазний пропуск зсував нумерацію.
                picture = render.placeholder(res)
            shots.append((picture, render.caption(res, rank)))
            rows.append(render.table_row(res, rank))
        status = _status_line(resp, started, extra)
        notes = "\n\n".join(
            f"> {n}" for n in _notices(resp) if n
        )
        return (shots, rows, status, render.describe_parse(resp.parsed), notes,
                [r.frame_id for r in resp.results], _debug_dump(resp, rows, started))

    yield package(response, "уточнюю детекцією…" if refine and response.results else "")
    if not refine or not response.results:
        return

    # Уточнення заміщає видачу покадрово, тож видно, як воно складається.
    note = ""
    done = 0
    total = len(response.results)
    try:
        for updated in _searcher().refine_by_detection(response):
            done += 1
            # Лічильник рахує від РЕАЛЬНОЇ довжини видачі. Раніше тут стояла
            # константа 10, і на запиті з двома сутностями глибина ділилася
            # навпіл — рядок «уточнено» не зʼявлявся ніколи.
            note = (f"уточнено {done} з ~{total}…" if done < total
                    else f"уточнено {done}: рамки точні під слово запиту, порядок не змінено")
            yield package(updated, note)
        # `note` ініціалізовано ДО циклу. Якщо файли верхівки відсутні на
        # диску, уточнення не робить жодного кроку — і звертання до
        # неоголошеної змінної падало з NameError просто в інтерфейс.
        yield package(response, note or "уточнення не дало жодного кадру")
    except Exception as exc:  # noqa: BLE001 — уточнення не мусить валити видачу
        logger.exception("уточнення не вдалося")
        yield package(response, f"уточнення не вдалося: {exc}")


def _debug_dump(resp, rows, started: float) -> str:
    """Видача у вигляді, придатному для копіювання в звіт чи повідомлення.

    Потрібне саме для налагодження: половина дефектів цього проєкту знайшлася
    тим, що хтось подивився на конкретну видачу й побачив у ній дивне число.
    Переказувати таку видачу словами довго й неточно — простіше скопіювати
    цілком, разом із розбором запиту, який її й пояснює.

    Формат навмисно текстовий, а не JSON: його читають очима в переписці, а
    не парсять.
    """
    parsed = resp.parsed
    lines = [
        f"запит:     {resp.query!r}",
        f"query_en:  {(parsed.query_en if parsed else None)!r}",
    ]
    if parsed and (parsed.must or parsed.must_not):
        lines.append(
            "must:      " + repr([
                (e.object.value, [(a.name.value, a.value) for a in e.attributes],
                 *( [f"×{e.count}"] if e.count > 1 else []))
                for e in parsed.must
            ])
        )
        if parsed.must_not:
            lines.append(
                "must_not:  " + repr([
                    (e.object.value, [(a.name.value, a.value) for a in e.attributes])
                    for e in parsed.must_not
                ])
            )
    lines += [
        f"режим:     {'жорсткий' if resp.is_strict else 'мʼякий'}"
        f" · збігів {resp.total_matches or len(resp.results)}"
        f" · показано {len(resp.results)}"
        f" · {(time.perf_counter() - started):.1f} с",
    ]
    if resp.notice:
        lines.append(f"примітка:  {resp.notice}")

    # Таблиця з вирівняними колонками — той самий вигляд, що й на екрані.
    header = render.TABLE_COLUMNS
    table = [header] + [[c or "—" for c in r] for r in rows]
    widths = [max(len(str(row[i])) for row in table) for i in range(len(header))]
    lines.append("")
    for row in table:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())

    # Рамки: для дефектів показу вони і є предметом розмови.
    lines.append("")
    for rank, res in enumerate(resp.results, start=1):
        named = [g for g in res.regions if getattr(g, "entity", "")]
        if not named:
            continue
        for g in named:
            box = [round(v, 3) for v in g.bbox]
            score = f"{g.entity_score:.2f}" if g.entity_score is not None else "—"
            lines.append(f"#{rank} {g.entity!r} bbox={box} площа "
                         f"{g.bbox[2] * g.bbox[3]:.1%} фасет {score}")
    return "\n".join(lines)


def _status_line(resp, started: float, extra: str = "") -> str:
    """Один рядок сталого формату — щоб очима порівнювати запити між собою."""
    mode = "🟢 жорсткий" if resp.is_strict else "🟡 мʼякий"
    # Скільки кадрів узагалі підійшло під умови — у СТАТУС, а не в
    # повідомлення. Раніше це число ховалося в реченні «Умовам відповідає
    # кадрів: 372, показано 5», тобто в тому самому місці, де й попередження
    # про справжні проблеми.
    total = f" з {resp.total_matches}" if resp.total_matches else ""
    parts = [
        f"**{len(resp.results)}**{total} знахідок",
        f"{(time.perf_counter() - started):.1f} с",
        mode,
    ]
    if extra:
        parts.append(f"⏳ {extra}")
    return " · ".join(parts)


def _notices(resp) -> list[str]:
    """Повідомлення, які варті окремого рядка, — і лише правдиві."""
    out = []
    if resp.notice:
        # «Умовам відповідає кадрів: N, показано M» переїхало в статус — тут
        # воно лише дублювало б число поруч зі справжніми попередженнями.
        notice = re.sub(
            r"Умовам відповідає кадрів: \d+, показано \d+\.\s*"
            r"Збільште ліміт, щоб побачити решту\.\s*", "", resp.notice
        ).strip()
        if notice:
            out.append(notice)
    if not resp.results:
        out.append("Нічого не знайдено. Індекс порожній? Вкладка «Індексація».")
    # Рядка про «показано лише наявні» більше немає: відсутні файли
    # ПОКАЗУЮТЬСЯ плиткою-заглушкою, тож твердження було неправдою.
    return out


def do_parse_only(query: str):
    """Лише розбір, без пошуку — швидка перевірка розуміння запиту."""
    if not query.strip():
        return "_Введіть запит._"
    from vsearch.search.parse import QueryParser

    parser = QueryParser()
    if not parser.is_available():
        return "❌ Немає ваг парсера: `python scripts/fetch_models.py --only query_parser_gguf`"
    if parser.should_bypass(query.strip()):
        return "_Розбір обійдено: короткий запит без маркерів заперечення._"
    started = time.perf_counter()
    parsed = parser.parse(query.strip())
    return render.describe_parse(parsed) + f"\n\n_Розбір за {(time.perf_counter()-started):.1f} с._"


# ── індексація ──────────────────────────────────────────────────────────────


def do_index(folder: str, profile: str, recreate: bool):
    from vsearch.index.catalog import SignatureMismatch
    from vsearch.ingest.images import index_paths

    path = Path(folder.strip()).expanduser()
    if not path.exists():
        return f"❌ Немає такого шляху: `{path}`"
    try:
        stats = index_paths(path, profile=get_profile(profile), recreate=recreate)
    except SignatureMismatch as exc:
        return f"❌ **Несумісний індекс**\n\n```\n{exc}\n```"
    except Exception as exc:  # noqa: BLE001
        logger.exception("індексація не вдалася")
        return f"❌ `{type(exc).__name__}: {exc}`"

    report = [f"✅ {stats.summary()}"]
    if stats.failed:
        report.append(f"\n**Не вдалося прочитати ({len(stats.failed)}):**")
        report += [f"- `{p}`: {e}" for p, e in stats.failed[:10]]
    return "\n".join(report)


# ── категорії ───────────────────────────────────────────────────────────────


def list_categories():
    from vsearch.index.catalog import Catalog

    rows = Catalog().prototypes()
    if not rows:
        return "_Реєстр порожній — спершу проіндексуйте щось._"
    lines = ["| Назва | Тип | Межа | Стан |", "|---|---|---|---|"]
    for row in rows:
        threshold = f"{row['threshold']:.4f}" if row["threshold"] is not None else "типова"
        state = "калібрований" if row["calibrated"] else "⚠️ некалібрований"
        lines.append(f"| `{row['name']}` | {row['kind']} | {threshold} | {state} |")
    lines.append(
        "\n> Некалібрований фасет придатний для ранжування, "
        "але як жорсткий фільтр ненадійний."
    )
    return "\n".join(lines)


def add_category(name: str, text: str, profile: str):
    from vsearch.index import schema
    from vsearch.index.catalog import Catalog
    from vsearch.index.store import VectorStore
    from vsearch.represent.categories import CategoryEngine
    from vsearch.represent.embed import Siglip2Embedder

    name, text = name.strip(), text.strip()
    if not name or not text:
        return "❌ Потрібні і назва, і опис."
    try:
        store = VectorStore()
        engine = CategoryEngine.with_defaults(
            Siglip2Embedder(profile=get_profile(profile)), store
        )
        prototype = engine.add_user_category(name, text)
        Catalog().save_prototype(prototype)
        reports = [
            engine.apply_to_collection(c, name).summary()
            for c in (schema.FRAMES, schema.REGIONS)
            if store.count(c)
        ]
    except Exception as exc:  # noqa: BLE001
        logger.exception("додавання категорії не вдалося")
        return f"❌ `{type(exc).__name__}: {exc}`"
    return "✅ " + "\n\n✅ ".join(reports) + f"\n\nШукати: поле «категорія» = `{name}`"


# ── стан ────────────────────────────────────────────────────────────────────


def system_status():
    from vsearch import goldenset
    from vsearch.backends import device
    from vsearch.backends.registry import get_registry
    from vsearch.index.catalog import Catalog
    from vsearch.index.store import VectorStore
    from vsearch.licensing import audit

    registry = get_registry()
    fetched = set(registry.fetched_names())
    store = VectorStore()
    alive = store.is_alive()

    lines = [
        "### Обчислення",
        f"- пристрої: `{', '.join(device.available_devices())}` → обрано "
        f"`{device.resolve().device}`",
        "",
        "### Індекс",
        f"- Qdrant `{store.url}`: {'🟢 доступний' if alive else '🔴 НЕДОСТУПНИЙ'}",
    ]
    if alive:
        lines.append(f"- кадрів: **{store.count('frames')}** · регіонів: **{store.count('regions')}**")
        if Path("catalog.db").exists():
            lines.append(f"- каталог: `{Catalog().counts()}`")
    else:
        lines.append("- `docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant`")

    lines += ["", "### Моделі"]
    for entry in registry:
        mark = "🟢" if entry.name in fetched else ("⚪" if entry.optional else "🔴")
        lines.append(f"- {mark} `{entry.name}` — {entry.license}")

    report = audit(registry)
    lines += [
        "",
        f"### Ліцензії\nблокувальних: **{len(report.blocking)}** · "
        f"непідтверджених: **{len(report.unverified)}**",
        "",
        "### Золоті набори",
    ]
    for name in goldenset.available():
        gs = goldenset.load(name)
        missing = len(gs.missing_files())
        state = "ціла" if not gs.validate() else "⚠️ ПОШКОДЖЕНА"
        extra = f" · немає медіа: {missing}" if missing else ""
        lines.append(f"- `{name}`: {len(gs)} активів, {len(gs.queries)} запитів, розмітка {state}{extra}")
    return "\n".join(lines)


# ── інтерфейс ───────────────────────────────────────────────────────────────

EXAMPLES = [
    "чоловік в окулярах",
    "дівчина в чорному з дитиною",
    "чоловік без окулярів і без головного убору",
    "полуниця",
    "PXCLD-1624",
    "a wheat brooch",
]

#: Стисла шапка. Довгий вступ на кожній вкладці читали один раз, а місце він
#: займав завжди.
INTRO = "### vsearch — семантичний пошук по фото та відео · офлайн, вільні ліцензії"

HELP = """
**Рамка на фото** показує, ЩО саме знайдено: 🔴 пропозиція детектора ·
🔵 плитка 288 px · 🟢 кадр цілком · 🟠 детекція під слово запиту.
Товста рамка з великим відсотком — ділянка, за якою кадр потрапив у видачу.

**Чому чисел два.** «текст» — наскільки кадр схожий на фразу запиту.
«сутності» — наскільки підтверджено те, що запит НАЗВАВ. Кадр може слабко
збігатися з фразою, але впевнено містити названий обʼєкт, і навпаки. Порядок
зливає обидва за рангами, тому жодне окреме число не спадає рівно згори вниз.

**Колонка «ділянок»** — скільки ще підтверджень у тому самому кадрі; для
відео це кадри однієї сцени, згорнутої в одну позицію.

**Пусто в колонці** означає «значення немає», а не «нуль».
"""

#: Власний CSS. Головне тут — `tabular-nums`: без нього цифри різної ширини
#: не вирівнюються в колонку, і таблиця перестає читатися очима, заради чого
#: вона й робилася.
CSS = """
.vs-table table { font-variant-numeric: tabular-nums; font-size: 12px; }
.vs-table td { padding: 4px 8px !important; }
.vs-status { font-size: 13px; opacity: 0.85; }
.vs-status strong { font-size: 15px; }
footer { display: none !important; }
.gradio-container { max-width: 100% !important; }
"""


def build():
    """Зібрати Gradio-інтерфейс."""
    import gradio as gr

    with gr.Blocks(title="vsearch") as app:
        gr.Markdown(INTRO)

        with gr.Tab("Пошук"):
            with gr.Row():
                query = gr.Textbox(
                    label=None, show_label=False, placeholder="Запит будь-якою мовою",
                    scale=6, autofocus=True, container=False,
                )
                run = gr.Button("Шукати", variant="primary", scale=1)

            # Часте — на видноті, решта в акордеоні. Раніше девʼять контролів
            # стояли в одному ряду однакової ваги, і серед них губилися ті
            # чотири, які справді перемикають при налагодженні.
            with gr.Row():
                limit = gr.Slider(1, 40, value=12, step=1, label="Результатів", scale=2)
                show_weak = gr.Checkbox(value=False, label="Слабкі збіги", scale=1)
                best_only = gr.Checkbox(value=False, label="Одна рамка", scale=1)
                refine = gr.Checkbox(value=False, label="Уточнити детекцією", scale=1)

            with gr.Accordion("Ще", open=False):
                with gr.Row():
                    use_parser = gr.Checkbox(
                        value=True, label="LLM-розбір",
                        info="потрібен для заперечень; ~1.8 с",
                    )
                    category = gr.Textbox(
                        label="Категорія", placeholder="weapon", value="",
                        info="фасетний фільтр; діє лише на щільному проході",
                    )
                gr.Markdown(
                    "_«Схоже на» задається кліком по результату, а не номером._"
                )
                with gr.Row():
                    similar_view = gr.Markdown("Зразків: —")
                    clear_fb = gr.Button("Очистити зразки", size="sm")

            status = gr.Markdown(elem_classes="vs-status")
            notices = gr.Markdown()

            with gr.Row():
                gallery = gr.Gallery(
                    label=None, show_label=False, columns=6, height=340,
                    object_fit="contain", scale=1, allow_preview=True,
                )
            table = gr.Dataframe(
                headers=render.TABLE_COLUMNS, datatype=["str"] * len(render.TABLE_COLUMNS),
                interactive=False, wrap=True, elem_classes="vs-table",
                label=None, show_label=False,
            )

            with gr.Accordion("Розбір запиту", open=False):
                parse_view = gr.Markdown()
            with gr.Accordion("Скопіювати для дебагу", open=False):
                debug_dump = gr.Code(
                    label=None, show_label=False, language=None, lines=14,
                    interactive=False,
                )
            with gr.Accordion("Як читати результати", open=False):
                gr.Markdown(HELP)

            gr.Examples(EXAMPLES, inputs=[query], label="Приклади")

            last_frames = gr.State([])
            similar = gr.State([])
            unlike = gr.State([])

            inputs = [query, limit, use_parser, category, show_weak, best_only,
                      refine, similar, unlike]
            outputs = [gallery, table, status, parse_view, notices, last_frames,
                       debug_dump]
            run.click(do_search, inputs, outputs)
            query.submit(do_search, inputs, outputs)

            def _pick(frames, chosen, evt: gr.SelectData):
                """Клік по результату додає його у зразки «схоже на».

                Замінює два текстові поля, куди треба було вручну вписувати
                номери, вичитані з підпису. Номер до того ж означав РІЗНІ
                кадри на різних запитах, бо стан перезаписувався видачею.
                """
                if not frames or evt.index is None or evt.index >= len(frames):
                    return chosen, f"Зразків: {len(chosen)}"
                frame = frames[evt.index]
                picked = [f for f in chosen if f != frame]
                if len(picked) == len(chosen):
                    picked.append(frame)
                return picked, (f"Зразків: {len(picked)}" if picked else "Зразків: —")

            gallery.select(_pick, [last_frames, similar], [similar, similar_view])
            clear_fb.click(lambda: ([], "Зразків: —"), None, [similar, similar_view])

        with gr.Tab("Розбір"):
            with gr.Row():
                parse_query = gr.Textbox(
                    label=None, show_label=False, scale=4, container=False,
                    placeholder="Перевірити, як парсер розуміє запит",
                )
                parse_btn = gr.Button("Розібрати", variant="primary", scale=1)
            parse_out = gr.Markdown()
            parse_btn.click(do_parse_only, parse_query, parse_out)
            parse_query.submit(do_parse_only, parse_query, parse_out)

        with gr.Tab("Індексація"):
            gr.Markdown(
                "Профіль визначає роздільність, плиткування й пропозиції детектора. "
                "**Це єдине місце, де він щось вирішує** — вектори будуються саме тут."
            )
            with gr.Row():
                folder = gr.Textbox(label="Тека", value="data/personal", scale=3)
                index_profile = gr.Dropdown(
                    PROFILE_NAMES, value="balanced", label="Профіль"
                )
            recreate = gr.Checkbox(
                label="Перебудувати з нуля",
                info="обовʼязково при зміні профілю: вектори несумісні",
            )
            index_btn = gr.Button("Проіндексувати", variant="primary")
            index_out = gr.Markdown()
            index_btn.click(do_index, [folder, index_profile, recreate], index_out)

        with gr.Tab("Категорії"):
            gr.Markdown(
                "Категорія — текстовий прототип, а не окрема модель: застосовується "
                "до **готового** індексу за секунди."
            )
            with gr.Row():
                cat_name = gr.Textbox(label="Назва", placeholder="removed_plates")
                cat_text = gr.Textbox(
                    label="Опис словами",
                    placeholder="a car with removed license plates", scale=2,
                )
            add_btn = gr.Button("Додати і застосувати", variant="primary")
            add_out = gr.Markdown()
            add_btn.click(
                lambda n, t: add_category(n, t, "balanced"),
                [cat_name, cat_text], add_out,
            )
            refresh = gr.Button("Оновити реєстр", size="sm")
            cat_list = gr.Markdown(value=list_categories())
            refresh.click(list_categories, None, cat_list)

        with gr.Tab("Стан"):
            status_btn = gr.Button("Оновити", variant="primary", size="sm")
            status_out = gr.Markdown(value=system_status())
            status_btn.click(system_status, None, status_out)

    return app


def serve(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    import gradio as gr

    # У Gradio 6 тема задається при запуску, а не в конструкторі Blocks.
    # У Gradio 6 і тема, і CSS задаються при запуску, а не в конструкторі.
    build().launch(
        server_name=host, server_port=port, share=share,
        inbrowser=False, theme=gr.themes.Soft(), css=CSS,
    )
