"""Командний рядок vsearch.

На M0 доступні лише діагностичні команди — вони перевіряють контур, у якому
працюватиме пошук. `index` та `search` зʼявляться з M1, і поки що чесно
повідомляють про це, а не імітують роботу.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Стан середовища: пристрій, профілі, ваги, ліцензії."""
    from vsearch import goldenset
    from vsearch.backends import device
    from vsearch.backends.registry import ManifestError, get_registry
    from vsearch.index.catalog import Catalog
    from vsearch.config import PROFILES, get_profile
    from vsearch.licensing import audit

    print("Пристрої")
    spec = device.resolve()
    print(f"  доступні:  {', '.join(device.available_devices())}")
    print(f"  обраний:   {spec.device} (dtype {spec.dtype})")

    print("\nПрофілі")
    active = get_profile()
    for name, profile in PROFILES.items():
        mark = "→" if name == active.name else " "
        print(
            f"  {mark} {name:9} ембединг={profile.embed_model:16} "
            f"плитка={profile.tiling.tile_size or 'вимкнено':>7} px "
            f"(≤{profile.tiling.max_tiles}) rerank={profile.rerank_top_k or 'вимкнено'}"
        )

    print("\nМоделі")
    try:
        registry = get_registry()
    except ManifestError as exc:
        print(f"  ПОМИЛКА: {exc}")
        return 1
    fetched = set(registry.fetched_names())
    for entry in registry:
        state = "завантажено" if entry.name in fetched else "немає"
        suffix = " (опційна)" if entry.optional else ""
        print(f"  [{'+' if entry.name in fetched else '-'}] {entry.name:20} {state}{suffix}")

    report = audit(registry)
    print(
        f"\nЛіцензії: блокувальних={len(report.blocking)}, "
        f"непідтверджених={len(report.unverified)}, "
        f"незакріплених ревізій={len(report.unpinned_revisions)}"
    )

    print("\nІндекс")
    from vsearch.index.store import VectorStore

    store = VectorStore()
    if store.is_alive():
        counts = Catalog().counts() if Path("catalog.db").exists() else {}
        print(f"  Qdrant {store.url}: доступний")
        print(f"  колекції: {', '.join(store.collections()) or '—'}")
        print(f"  кадрів у frames: {store.count('frames')}")
        if counts:
            print(f"  каталог: активів={counts['assets']}, кадрів={counts['frames']}")
    else:
        print(f"  Qdrant {store.url}: НЕДОСТУПНИЙ")
        print("    docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant")

    print("\nЗолоті набори")
    for name in goldenset.available() or ["— немає"]:
        if name.startswith("—"):
            print(f"  {name}")
            continue
        gs = goldenset.load(name)
        missing = len(gs.missing_files())
        print(
            f"  {name:18} активів={len(gs):3} запитів={len(gs.queries):3} "
            f"розмітка={'ціла' if not gs.validate() else 'ПОШКОДЖЕНА'}"
            + (f" (медіафайлів немає: {missing})" if missing else "")
        )
    return 0


def _cmd_licenses(args: argparse.Namespace) -> int:
    from vsearch.backends.registry import get_registry
    from vsearch.licensing import audit

    report = audit(get_registry())
    for finding in report.findings:
        print(f"{finding.model:20} {finding.license:14} {finding.verdict.value}")
    return 0 if report.is_clean(strict=args.strict) else 1


def _cmd_golden(args: argparse.Namespace) -> int:
    from vsearch import goldenset

    failures = 0
    for name in ([args.name] if args.name else goldenset.available()):
        gs = goldenset.load(name)
        problems = gs.validate()
        failures += bool(problems)
        print(f"[{'!' if problems else '+'}] {name}: активів={len(gs)}, запитів={len(gs.queries)}")
        for problem in problems:
            print(f"      {problem}")
        if args.show_missing and (missing := gs.missing_files()):
            print(f"      медіафайлів немає ({len(missing)}): {', '.join(missing[:5])}…")
    return 1 if failures else 0


def _cmd_index(args: argparse.Namespace) -> int:
    """Проіндексувати теку зображень."""
    from vsearch.config import get_profile
    from vsearch.index.catalog import SignatureMismatch
    from vsearch.ingest.images import index_paths

    profile = get_profile(args.profile)
    print(f"Профіль {profile.name}: {profile.embed_model}, "
          f"max_num_patches={profile.max_num_patches}, dim={profile.embed_dim}")
    try:
        stats = index_paths(args.path, profile=profile, recreate=args.recreate)
    except SignatureMismatch as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print(stats.summary())
    for path, error in stats.failed[:10]:
        print(f"  ! {path}: {error}", file=sys.stderr)
    if stats.facets_error:
        print(f"\nФасети не пораховано: {stats.facets_error}", file=sys.stderr)
        return 1
    return 1 if stats.failed and not stats.indexed else 0


def _cmd_search(args: argparse.Namespace) -> int:
    """Пошук за текстовим запитом."""
    from vsearch.config import get_profile
    from vsearch.search.retrieve import Searcher

    from vsearch.search.retrieve import REFINE_TOP_K

    refine = getattr(args, "refine", False)
    searcher = Searcher(profile=get_profile(args.profile))
    # З уточненням тягнемо ГЛИБШЕ, ніж показуємо: воно вміє лише переставити
    # вже підняте. На «дівчина в синій спідниці» потрібний кадр стояв 8-м, і
    # при `--limit 6` уточнювати було б уже нічого.
    response = searcher.search(
        args.query,
        limit=max(args.limit, REFINE_TOP_K) if refine else args.limit,
        category=args.category, scope=args.scope, parse=not args.no_parse,
    )
    if refine and response.results:
        # У терміналі стрімити нема куди, тож просто дочекатися останнього
        # кроку. Прогрес друкуємо, щоб мовчазна пауза на 16 секунд не
        # виглядала зависанням.
        import sys as _sys

        for step, response in enumerate(searcher.refine_by_detection(response), 1):
            print(f"\rуточнення: {step} кадрів…", end="", file=_sys.stderr, flush=True)
        print("\r" + " " * 32 + "\r", end="", file=_sys.stderr)
        response.results = response.results[: args.limit]
    if response.notice:
        print(f"⚠ {response.notice}\n")
    if not response.results:
        print("Нічого не знайдено. Індекс порожній? vsearch doctor")
        return 0

    print(f"{args.query!r} — {len(response)} результатів за {response.latency_ms:.0f}мс "
          f"(профіль {response.profile})\n")
    for rank, result in enumerate(response.results, start=1):
        print(f"{rank:2}. {Path(result.path).name}")
        print(f"    {result.explain()}")
        if args.verbose:
            print(f"    джерело: {result.path}")
            print(f"    актив:   {result.asset_id[:16]}…")
            print(f"    модель:  {result.provenance['embed_model']}"
                  f"@{result.provenance['embed_revision'][:12]}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Підняти веб-інтерфейс."""
    from vsearch.api.ui import serve

    print(f"Інтерфейс: http://{args.host}:{args.port}")
    serve(host=args.host, port=args.port, share=args.share)
    return 0


def _cmd_parse(args: argparse.Namespace) -> int:
    """Показати, як парсер зрозумів запит. Діагностика для складних випадків."""
    from vsearch.search.parse import QueryParser

    parser = QueryParser()
    if not parser.is_available():
        print("Немає ваг парсера: python scripts/fetch_models.py --only query_parser_gguf",
              file=sys.stderr)
        return 1
    if parser.should_bypass(args.query):
        print(f"{args.query!r} — розбір обійдено (короткий запит без заперечення)")
        return 0

    parsed = parser.parse(args.query)
    print(f"мова оригіналу: {parsed.language}")
    print(f"опис для ембедера: {parsed.query_en!r}")
    print(f"композитний: {parsed.is_compositional}   заперечення: {parsed.has_negation}")
    for slot, entities in (("must", parsed.must), ("must_not", parsed.must_not)):
        print(f"\n{slot}:")
        for index, entity in enumerate(entities):
            attrs = ", ".join(f"{a.name.value}={a.value}" for a in entity.attributes) or "—"
            print(f"  [{index}] {entity.object.value}: {attrs}")
        if not entities:
            print("  —")
    if parsed.relations:
        print("\nвідношення:")
        for relation in parsed.relations:
            kind = "геометрія" if relation.is_geometric else "потребує VLM"
            print(f"  {relation.subject} —{relation.predicate.value}→ {relation.target}  ({kind})")
    return 0


def _cmd_categories(args: argparse.Namespace) -> int:
    """Перелік прототипів із індексу разом зі станом калібрування."""
    from vsearch.index.catalog import Catalog

    rows = Catalog().prototypes()
    if not rows:
        print("Реєстр порожній — спершу проіндексуйте щось: vsearch index <шлях>")
        return 0

    width = max(len(r["name"]) for r in rows)
    print(f"{'НАЗВА'.ljust(width)}  {'ТИП':10} {'МЕЖА':>8}  СТАН")
    print("-" * (width + 46))
    for row in rows:
        threshold = f"{row['threshold']:.4f}" if row["threshold"] is not None else "типова"
        state = "калібрований" if row["calibrated"] else "некалібрований"
        if row["example_count"]:
            state += f", прикладів {row['example_count']}"
        print(f"{row['name'].ljust(width)}  {row['kind']:10} {threshold:>8}  {state}")
        if args.verbose:
            print(f"{' ' * width}  позитив: {', '.join(row['positive'])}")
            if row["negative"]:
                print(f"{' ' * width}  негатив: {', '.join(row['negative'][:3])}…")
    print(f"\nВсього: {len(rows)}; некаліброваних: {sum(1 for r in rows if not r['calibrated'])}")
    print("Некалібрований фасет придатний для ранжування, але як жорсткий фільтр ненадійний.")
    return 0


def _cmd_category_add(args: argparse.Namespace) -> int:
    """Додати категорію текстом і застосувати до наявного індексу."""
    from vsearch.config import get_profile
    from vsearch.index import schema
    from vsearch.index.catalog import Catalog
    from vsearch.index.store import VectorStore
    from vsearch.represent.categories import CategoryEngine
    from vsearch.represent.embed import Siglip2Embedder

    profile = get_profile(args.profile)
    store = VectorStore()
    catalog = Catalog()
    engine = CategoryEngine.with_defaults(Siglip2Embedder(profile=profile), store)

    examples = []
    if args.examples:
        from vsearch.ingest.images import discover, load_image

        examples = [load_image(p) for p in discover(Path(args.examples))]
        print(f"прикладів для уточнення: {len(examples)}")

    prototype = engine.add_user_category(
        args.name, args.text, examples=examples, description=args.description
    )
    catalog.save_prototype(prototype)

    for collection in (schema.FRAMES, schema.REGIONS):
        if store.count(collection):
            print(engine.apply_to_collection(collection, args.name).summary())
    print(f"\nШукати в категорії: vsearch search \"...\" --category {args.name}")
    return 0


def _cmd_golden_init(args: argparse.Namespace) -> int:
    """Створити каркас золотого набору з теки реальних фото.

    Розмітку доводиться робити руками — саме вона й визначає всі метрики, і
    саме її неможливо згенерувати. Ця команда бере на себе решту: копіює
    медіа, проставляє ідентифікатори й готує файли, у які лишається вписати
    мітки та запити.
    """
    import json
    import shutil

    from vsearch import goldenset
    from vsearch.ingest.images import discover

    target = Path(goldenset.DEFAULT_GOLDEN_DIR) / args.name
    media = target / "media"
    if target.exists() and not args.force:
        print(f"Набір {args.name} вже існує: {target}. --force щоб перезаписати.", file=sys.stderr)
        return 1
    media.mkdir(parents=True, exist_ok=True)

    assets = []
    for order, source in enumerate(discover(Path(args.source)), start=1):
        asset_id = f"{args.prefix}_{order:04d}"
        name = f"{asset_id}{source.suffix.lower()}"
        (shutil.copy2 if args.copy else _link)(source, media / name)
        assets.append({
            "asset_id": asset_id,
            "path": f"media/{name}",
            "media_type": "image",
            "source": str(source),
            "labels": {},
            "objects": [],
            "notes": "",
        })

    if not assets:
        print(f"У {args.source} не знайдено зображень.", file=sys.stderr)
        return 1

    (target / "assets.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in assets) + "\n", encoding="utf-8"
    )
    if not (target / "queries.jsonl").exists():
        (target / "queries.jsonl").write_text(
            "// Один запит на рядок. relevant — що має знайтися,\n"
            "// forbidden — що НЕ має (саме воно робить заперечення вимірюваним),\n"
            "// gains — градуйована релевантність для nDCG.\n"
            '// {"query_id": "q1", "text": "чоловік без окулярів", "lang": "uk",\n'
            '//  "relevant": ["' + assets[0]["asset_id"] + '"], "forbidden": [],\n'
            '//  "expected_parse": {"must": [], "must_not": []}}\n',
            encoding="utf-8",
        )
    (target / ".gitignore").write_text("media/\n", encoding="utf-8")

    print(f"Створено каркас набору {args.name}: {len(assets)} активів")
    print(f"  {target}")
    print("\nЗалишилося зробити руками — і тільки це визначає всі метрики:")
    print(f"  1. проставити labels у {target / 'assets.jsonl'}")
    print(f"  2. описати запити в {target / 'queries.jsonl'}")
    print(f"  3. перевірити: vsearch golden {args.name}")
    print(f"  4. порівняти конфігурації: python scripts/compare_configs.py --golden {args.name}")
    return 0


def _link(source: Path, target: Path) -> None:
    """Симлінк замість копії: реальні матеріали можуть важити гігабайти."""
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def _not_yet(milestone: str):
    def handler(args: argparse.Namespace) -> int:
        print(
            f"Команда ще не реалізована — заплановано на {milestone}.\n"
            f"Сценарії вже написані: pytest -m wip",
            file=sys.stderr,
        )
        return 2

    return handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vsearch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="стан середовища: пристрій, ваги, ліцензії, набори")
    doctor.set_defaults(func=_cmd_doctor)

    licenses = sub.add_parser("licenses", help="інвентар ліцензій (п.6)")
    licenses.add_argument("--strict", action="store_true", help="режим релізу")
    licenses.set_defaults(func=_cmd_licenses)

    golden = sub.add_parser("golden", help="перевірити цілісність золотого набору")
    golden.add_argument("name", nargs="?", help="імʼя набору (типово — усі)")
    golden.add_argument("--show-missing", action="store_true", help="перелічити відсутні медіа")
    golden.set_defaults(func=_cmd_golden)

    notes = sub.add_parser(
        "notes", help="коментарі, залишені в інтерфейсі до конкретних кадрів"
    )
    notes.set_defaults(func=cmd_notes)

    categories = sub.add_parser("categories", help="перелік категорій та атрибутів (п.4)")
    categories.add_argument("-v", "--verbose", action="store_true", help="показати формулювання")
    categories.set_defaults(func=_cmd_categories)

    add = sub.add_parser("category-add", help="додати свою категорію текстом")
    add.add_argument("name", help="коротке імʼя, напр. removed_plates")
    add.add_argument("text", help="опис словами, напр. \"a car with removed license plates\"")
    add.add_argument("--examples", default=None, help="тека з 3–5 прикладами (необовʼязково)")
    add.add_argument("--description", default="", help="пояснення для реєстру")
    add.add_argument("--profile", default=None)
    add.set_defaults(func=_cmd_category_add)

    init = sub.add_parser("golden-init", help="каркас золотого набору з теки реальних фото")
    init.add_argument("name", help="імʼя набору, напр. real_photos")
    init.add_argument("source", help="тека із зображеннями")
    init.add_argument("--prefix", default="rp", help="префікс ідентифікаторів активів")
    init.add_argument("--copy", action="store_true", help="копіювати замість симлінків")
    init.add_argument("--force", action="store_true", help="перезаписати наявний набір")
    init.set_defaults(func=_cmd_golden_init)

    index = sub.add_parser("index", help="проіндексувати теку зображень")
    index.add_argument("path", type=Path)
    index.add_argument("--profile", default=None, help="fast | balanced | quality")
    index.add_argument("--recreate", action="store_true", help="перебудувати індекс з нуля")
    index.set_defaults(func=_cmd_index)

    search = sub.add_parser("search", help="пошук за текстовим запитом")
    search.add_argument("query")
    search.add_argument("--profile", default=None)
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--category", default=None, help="обмежити пошук категорією")
    search.add_argument(
        "--scope", default="auto", choices=("auto", "regions", "frames", "both"),
        help="де шукати: auto — регіони ранжують, кадри доповнюють повноту",
    )
    search.add_argument("-v", "--verbose", action="store_true", help="показати провенанс")
    search.add_argument("--no-parse", action="store_true",
                        help="без LLM-розбору: лише щільний пошук, ~1.6с швидше")
    search.add_argument("--refine", action="store_true",
                        help="уточнити верхівку обумовленою детекцією (~1.6с на кадр)")

    parse_cmd = sub.add_parser("parse", help="показати розбір запиту без пошуку")
    parse_cmd.add_argument("query")
    parse_cmd.set_defaults(func=_cmd_parse)
    search.set_defaults(func=_cmd_search)

    serve = sub.add_parser("serve", help="підняти веб-інтерфейс для перевірки")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--share", action="store_true", help="публічне посилання Gradio")
    serve.set_defaults(func=_cmd_serve)

    return parser


def cmd_notes(args) -> int:
    """Показати коментарі, залишені людиною в інтерфейсі.

    Це зародок розмітки: майже кожен дефект тут знайдено тим, що хтось
    подивився на видачу й описав словами, що з нею не так. Команда дає
    прочитати ці описи разом, щоб перетворити їх на мітки золотого набору.
    """
    from vsearch.index.catalog import Catalog

    catalog = Catalog()
    rows = catalog.notes()
    if not rows:
        print("Коментарів немає. Додайте їх у колонці «коментар» в інтерфейсі.")
        return 0
    found = catalog.notes_with_state()
    stale = [r for r in found if r["state"] != "актуальний"]
    print(f"коментарів: {len(found)}"
          + (f" · потребують уваги: {len(stale)}" if stale else "") + "\n")
    marks = {"актуальний": "  ", "вміст змінився": "⚠ ", "файлу немає": "✗ ",
             "поза індексом": "· "}
    for row in found:
        name = Path(row["path"]).name or row["asset_id"][:12]
        mark = marks.get(row["state"], "  ")
        print(f"{mark}{name}" + ("" if row["state"] == "актуальний"
                                 else f"   [{row['state']}]"))
        print(f"    {row['note']}")
    if stale:
        print("\n⚠ «вміст змінився» означає, що за цим шляхом тепер ІНШЕ "
              "зображення.\n  Коментар його не стосується — він прикріплений до "
              "вмісту, а не до файлу.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
