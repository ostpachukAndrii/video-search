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
import time
from pathlib import Path

from vsearch.api import render
from vsearch.config import PROFILES, get_profile
from vsearch.search.retrieve import MIN_PROBABILITY

logger = logging.getLogger(__name__)

PROFILE_NAMES = list(PROFILES)
SCOPES = ["auto", "regions", "frames", "both"]

_searchers: dict[str, object] = {}


def _searcher(profile_name: str):
    """Пошуковець на профіль, створюється один раз і кешується."""
    from vsearch.search.retrieve import Searcher

    if profile_name not in _searchers:
        _searchers[profile_name] = Searcher(profile=get_profile(profile_name))
    return _searchers[profile_name]


# ── пошук ───────────────────────────────────────────────────────────────────


def do_search(query: str, profile: str, scope: str, limit: int, use_parser: bool,
              category: str, show_weak: bool = False, best_only: bool = False,
              refine: bool = False, similar_ranks: str = "", unlike_ranks: str = "",
              last_frames: list[str] | None = None):
    """Генератор (галерея, розбір, звіт, провенанс).

    Генератор, а не функція, заради уточнення обумовленою детекцією: воно
    коштує ~1.6 с на кадр, тобто близько 16 с на десятку. Чекати на них із
    порожнім екраном неприйнятно, а відмовитися від уточнення означає лишити
    дрібні обʼєкти незнайденими. Тому перша видача йде одразу, а уточнена
    заміщає її по кадру.
    """
    if not query.strip():
        yield [], "", "_Введіть запит._", "", []
        return

    started = time.perf_counter()

    # Розбір показується ОКРЕМИМ кроком, а не разом із результатами. Пошук
    # разом із розбором триває близько дванадцяти секунд, і весь цей час
    # екран був порожній — при тому, що розбір готовий уже на другій. Тепер
    # видно, як система зрозуміла запит, поки вона ще шукає; а на розмитому
    # запиті це ще й дає змогу не чекати решти.
    parsed_preview = ""
    if use_parser:
        yield [], "", "⏳ **Розбираю запит…**", "", []
        try:
            from vsearch.search.parse import QueryParser

            parser = QueryParser()
            if parser.is_available():
                parsed_preview = render.describe_parse(parser.parse(query.strip()))
        except Exception:  # noqa: BLE001 — попередній показ не мусить нічого ламати
            logger.debug("попередній розбір не вдався", exc_info=True)
    yield [], parsed_preview, "⏳ **Шукаю по індексу…**", "", []

    # Слідчий позначає кадри НОМЕРАМИ з видачі, які щойно бачив, — не
    # ідентифікаторами. Переклад номерів у `frame_id` робиться тут, бо тільки
    # інтерфейс знає, який порядок був на екрані.
    seen = list(last_frames or [])

    def _ranks(text: str) -> list[str]:
        picked: list[str] = []
        for chunk in str(text or "").replace(";", ",").split(","):
            chunk = chunk.strip().lstrip("#")
            if chunk.isdigit() and 1 <= int(chunk) <= len(seen):
                picked.append(seen[int(chunk) - 1])
        return picked

    try:
        response = _searcher(profile).search(
            query.strip(),
            similar_to=_ranks(similar_ranks),
            unlike=_ranks(unlike_ranks),
            # Глибина уточнення дорівнює показу. Тягнути глибше мало сенс,
            # поки уточнення ПЕРЕСТАВЛЯЛО верхівку; відколи воно лише уточнює
            # рамку, зайві кадри довелося б потім обрізати — і користувач
            # бачив, як видача, на яку він щойно дивився, коротшає.
            limit=int(limit),
            scope=scope,
            parse=use_parser,
            category=(category.strip() or None),
            # Нуль означає «показати все»: слідчий сам вирішує, чи дивитися
            # на те, у чому модель не впевнена.
            min_probability=0.0 if show_weak else MIN_PROBABILITY,
        )
    except Exception as exc:  # noqa: BLE001 — помилку показуємо, а не ховаємо
        logger.exception("пошук не вдався")
        yield [], "", f"❌ **Помилка:** `{type(exc).__name__}: {exc}`", "", []
        return

    lines = [
        f"**Знайдено:** {len(response.results)} · "
        f"**{response.latency_ms:.0f} мс** (загалом {(time.perf_counter()-started)*1000:.0f} мс)",
        f"**Режим:** {'🟢 жорсткий' if response.is_strict else '🟡 мʼякий (fallback)'} · "
        f"**scope:** `{scope}` · **профіль:** `{profile}`",
    ]
    if response.results:
        lines.append(f"**Що спрацювало:** {render.sources_summary(response.results)}")
        confident = sum(1 for r in response.results if r.probability >= 0.5)
        weak = sum(1 for r in response.results if r.probability < 0.05)
        lines.append(
            f"**Впевненість:** висока (>50%) — {confident} · "
            f"низька (<5%) — {weak} з {len(response.results)}"
        )
        # Легенда стоїть тут, а не в бічній панелі провенансу: там вона
        # формально була, але поруч із версіями моделей її ніхто не читав, і
        # питання «котра з синіх рамок саме та» лишалося без відповіді.
        lines.append("")
        named = any(
            getattr(region, "entity", "")
            for result in response.results for region in result.regions
        )
        if any(r.entity_confidence is not None for r in response.results):
            lines.append(render.LEGEND_TWO_SCORES)
        if best_only:
            lines.append(render.LEGEND_BEST_ONLY)
        elif named:
            lines.append(render.LEGEND_ENTITIES)
        else:
            lines.append(render.LEGEND)
    if response.notice:
        lines.append(f"\n> ⚠️ {response.notice}")
    if not response.results:
        lines.append("\n> Нічого не знайдено. Індекс порожній? Вкладка «Індексація».")
    missing = sum(1 for r in response.results if not Path(r.path).exists())
    if missing:
        lines.append(f"\n> ⚠️ {missing} файлів немає на диску — показано лише наявні.")

    provenance = ""
    if response.results:
        p = response.results[0].provenance
        provenance = (
            f"**Модель:** `{p.get('embed_model')}` @ `{str(p.get('embed_revision'))[:12]}`  \n"
            f"**Роздільність:** {p.get('max_num_patches')} патчів · "
            f"**Пристрій:** `{p.get('device')}`"
        )

    def package(resp, extra: str = ""):
        shots = []
        for rank, res in enumerate(resp.results, start=1):
            picture = render.annotate(res, best_only=best_only)
            if picture is None:
                # Не пропускаємо: результат, який неможливо намалювати, все
                # одно є знахідкою. Мовчазний пропуск зсував нумерацію, і
                # верхній кадр виглядав як неіснуючий.
                picture = render.placeholder(res)
            shots.append((picture, render.caption(res, rank)))
        body = list(lines)
        if extra:
            body.insert(2, extra)
        return (shots, render.describe_parse(resp.parsed), "\n".join(body),
                provenance, [r.frame_id for r in resp.results])

    yield package(
        response,
        "⏳ **Уточнюю обумовленою детекцією…**" if refine and response.results else "",
    )
    if not refine or not response.results:
        return

    # Уточнення заміщає видачу покадрово. Кожен крок — окремий yield, тож
    # порядок на екрані видно, як він складається, а не через 16 секунд.
    done = 0
    try:
        for updated in _searcher(profile).refine_by_detection(response):
            done += 1
            remaining = min(len(response.results), 10) - done
            note = (
                f"⏳ **Уточнено {done}, лишилось {remaining}…**" if remaining > 0
                else f"✅ **Уточнено обумовленою детекцією: {done} кадрів.** "
                     f"Рамки тепер точні під слово запиту. Порядок НЕ змінено — "
                     f"його визначило злиття свідчень при пошуку."
            )
            yield package(updated, note)
        yield package(response, note)
    except Exception as exc:  # noqa: BLE001 — уточнення не мусить валити видачу
        logger.exception("уточнення не вдалося")
        yield package(response, f"⚠️ **Уточнення не вдалося:** `{exc}`")


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
    ["чоловік в окулярах", "balanced", "auto", 8, True, "", False],
    ["жінка", "balanced", "auto", 8, True, "", False],
    ["білий велосипед", "balanced", "auto", 8, False, "", True],
    ["אדם ללא משקפיים", "balanced", "auto", 8, True, "", False],
    ["червона сумка всередині чорного авто", "balanced", "auto", 8, True, "", False],
]

INTRO = """
# vsearch — семантичний пошук по фото та відео

Прототип для розслідувань. Працює офлайн, усі моделі вільні для комерційного
використання.

**Як перевіряти:** рамка на зображенні показує, що саме знайдено — 🔴 обʼєкт
детектора, 🔵 плитка, 🟢 кадр цілком. Панель «Розбір запиту» показує, як
система зрозуміла заперечення. Значок режиму каже, чи довелося знімати
жорсткі обмеження.
"""


def build():
    """Зібрати Gradio-інтерфейс."""
    import gradio as gr

    with gr.Blocks(title="vsearch") as app:
        gr.Markdown(INTRO)

        with gr.Tab("🔍 Пошук"):
            with gr.Row():
                query = gr.Textbox(
                    label="Запит будь-якою мовою",
                    placeholder="чоловік без окулярів біля червоної машини",
                    scale=4, autofocus=True,
                )
                run = gr.Button("Шукати", variant="primary", scale=1)
            with gr.Row():
                profile = gr.Dropdown(PROFILE_NAMES, value="balanced", label="Профіль")
                scope = gr.Dropdown(
                    SCOPES, value="auto", label="Де шукати",
                    info="auto: регіони ранжують, кадри доповнюють",
                )
                limit = gr.Slider(1, 30, value=8, step=1, label="Результатів")
                use_parser = gr.Checkbox(
                    value=True, label="LLM-розбір",
                    info="потрібен для заперечень; додає ~1.6 с",
                )
                category = gr.Textbox(label="Категорія", placeholder="напр. weapon", scale=1)
                show_weak = gr.Checkbox(
                    value=False, label="Слабкі збіги",
                    info=f"показати те, у чому модель невпевнена (<{MIN_PROBABILITY:.0%})",
                )
                best_only = gr.Checkbox(
                    value=False, label="Лише найкраща ділянка",
                    info="лишити одну рамку — ту, за якою кадр ранжується",
                )
                similar_ranks = gr.Textbox(
                    label="Схоже на #", placeholder="напр. 1, 3", scale=1,
                    info=(
                        "номери з ПОПЕРЕДНЬОЇ видачі: запит зсувається до цих "
                        "кадрів (Rocchio). Найчастіший слідчий сценарій — "
                        "«знайди ще таких самих», який текстом не формулюється"
                    ),
                )
                unlike_ranks = gr.Textbox(
                    label="Не схоже на #", placeholder="напр. 2", scale=1,
                    info="відкинуті кадри важать утричі менше за зразки",
                )
                refine = gr.Checkbox(
                    value=False, label="Уточнити детекцією ⚠",
                    info=(
                        "точна рамка під слово запиту, ~1.6 с на кадр. "
                        "Порядок видачі НЕ змінює — виміряно, що будь-яка "
                        "участь детекції в порядку його псує (Recall@5 0.79 → "
                        "0.51). Вмикайте, щоб побачити, ДЕ саме обʼєкт"
                    ),
                )

            report = gr.Markdown()
            with gr.Row():
                gallery = gr.Gallery(
                    label="Результати", columns=4, height=460,
                    object_fit="contain", show_label=True, scale=3,
                )
                with gr.Column(scale=2):
                    parse_view = gr.Markdown(label="Розбір запиту")
                    provenance = gr.Markdown()

            gr.Examples(
                EXAMPLES,
                inputs=[query, profile, scope, limit, use_parser, category, show_weak],
                label="Приклади — від простого до складного",
            )
            # Порядок попередньої видачі живе у стані сторінки: слідчий
            # позначає номери, які бачить, а не ідентифікатори кадрів.
            last_frames = gr.State([])
            inputs = [
                query, profile, scope, limit, use_parser, category, show_weak,
                best_only, refine, similar_ranks, unlike_ranks, last_frames,
            ]
            outputs = [gallery, parse_view, report, provenance, last_frames]
            run.click(do_search, inputs, outputs)
            query.submit(do_search, inputs, outputs)

        with gr.Tab("🧠 Розбір запиту"):
            gr.Markdown(
                "Перевірка того, як парсер розуміє запит, без самого пошуку.\n\n"
                "Спробуйте заперечення різними мовами, подвійне заперечення "
                "(«без окулярів і без головного убору») та просторові відношення "
                "(«сумка всередині авто»)."
            )
            with gr.Row():
                parse_query = gr.Textbox(label="Запит", scale=4)
                parse_btn = gr.Button("Розібрати", variant="primary", scale=1)
            parse_out = gr.Markdown()
            parse_btn.click(do_parse_only, parse_query, parse_out)
            parse_query.submit(do_parse_only, parse_query, parse_out)

        with gr.Tab("📥 Індексація"):
            gr.Markdown(
                "Індексація теки із зображеннями. Профіль впливає на роздільність, "
                "плиткування й пропозиції детектора — і на те, скільки це триватиме."
            )
            with gr.Row():
                folder = gr.Textbox(
                    label="Тека із зображеннями",
                    value="tests/golden/smoke/media", scale=3,
                )
                index_profile = gr.Dropdown(PROFILE_NAMES, value="balanced", label="Профіль")
            recreate = gr.Checkbox(
                label="Перебудувати з нуля",
                info="обовʼязково при зміні профілю: вектори інших параметрів несумісні",
            )
            index_btn = gr.Button("Проіндексувати", variant="primary")
            index_out = gr.Markdown()
            index_btn.click(do_index, [folder, index_profile, recreate], index_out)

        with gr.Tab("🏷 Категорії"):
            gr.Markdown(
                "Категорія — це текстовий прототип, а не окрема модель. Тому нова "
                "категорія застосовується до **готового** індексу за секунди, без "
                "повторного читання зображень."
            )
            with gr.Row():
                cat_name = gr.Textbox(label="Коротка назва", placeholder="removed_plates")
                cat_text = gr.Textbox(
                    label="Опис словами",
                    placeholder="a car with removed license plates", scale=2,
                )
                cat_profile = gr.Dropdown(PROFILE_NAMES, value="balanced", label="Профіль")
            add_btn = gr.Button("Додати і застосувати", variant="primary")
            add_out = gr.Markdown()
            add_btn.click(add_category, [cat_name, cat_text, cat_profile], add_out)

            gr.Markdown("### Реєстр")
            refresh = gr.Button("Оновити")
            cat_list = gr.Markdown(value=list_categories())
            refresh.click(list_categories, None, cat_list)

        with gr.Tab("⚙️ Стан системи"):
            status_btn = gr.Button("Оновити", variant="primary")
            status_out = gr.Markdown(value=system_status())
            status_btn.click(system_status, None, status_out)

    return app


def serve(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    import gradio as gr

    # У Gradio 6 тема задається при запуску, а не в конструкторі Blocks.
    build().launch(
        server_name=host, server_port=port, share=share,
        inbrowser=False, theme=gr.themes.Soft(),
    )
