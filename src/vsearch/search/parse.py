"""Розбір запиту: будь-яка мова → StructuredQuery.

Модель обмежена GBNF-граматикою, побудованою з Pydantic-схеми. Це не спосіб
«зменшити кількість помилок парсингу», а спосіб зробити їх неможливими:
граматика діє на рівні семплювання токенів, тож модель фізично не може
згенерувати ані невалідний JSON, ані неіснуючий атрибут. Перелік атрибутів
у схемі закритий, і це та сама гарантія — вигадати `attr_wearing_hat` замість
`attr_headwear` не вийде.

Парсер робить три речі одним викликом:
  * нормалізує запит в англійську (WebLI має 90% англійських підписів, тож
    канонічна EN-фраза дає кращий вектор, ніж оригінал);
  * витягує звʼязані ознаки в `must` / `must_not`;
  * витягує просторові відношення між обʼєктами.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from vsearch.backends.registry import ModelNotFetched, get_registry
from vsearch.search.query_model import EMPTY, StructuredQuery

if TYPE_CHECKING:
    from vsearch.backends.registry import ModelRegistry

logger = logging.getLogger(__name__)

MODEL_NAME = "query_parser_gguf"

SYSTEM_PROMPT = """You convert image-search queries into structured JSON.

THE MOST IMPORTANT RULE: include ONLY objects that the query actually names.
Never add an object that is not mentioned. If the query names one object,
output exactly one object. Inventing a car, a person or anything else that the
user did not ask about makes the search return wrong frames.

SECOND RULE, equally important: an adjective is NOT a separate object.
"a woman's handbag", "жіноча сумочка", "men's watch" describe ONE object —
a bag, a watch — with a style qualifier. Do not emit a person for them. Emit a
person only when the query says a person is actually present in the frame
("a woman with a handbag", "жінка з сумкою"). Splitting an adjective into a
second object makes the search demand two things where the user asked for one,
and the frame count collapses.

Other rules:
- query_en: a short, literal English description of what must be VISIBLE.
  Drop negations from it — they belong in must_not, not in the description.
  Write NATURAL English. Substances and materials take no article: write
  "champagne", "water", "blood", "smoke" — never "a champagne". An
  ungrammatical phrase scores far worse than the bare word: "a champagne"
  matched a real bottle at 9%, while "champagne" matched it at 53%.
  A countable thing KEEPS its article — "a tank", not bare "tank". Measured:
  the same crop scored 63.9% for "tank" and 0.05% for "a tank".
- language: ISO code of the original query (uk, en, he, de, pl, ru, ...).
- must: objects that must be present, with attributes bound to THAT object.
- count: how many of that object the frame must contain. Default 1. Set it
  when the query says a number or a plural: "дві дівчини" → count 2,
  "three cars" → count 3, "two girls in blue skirts" → one entity with
  count 2, NOT two entities. Plain plurals without a number stay at 1 —
  "люди" means "people are present", not a specific quantity.
- must_not: objects whose presence disqualifies the frame. EVERY negated
  attribute gets its OWN entry: two negations mean two entries in must_not,
  three mean three. Merging them into one entry would exclude only frames
  where all of them hold together, letting the rest through.
- relations: spatial links between objects, by their index in must.
- How a person LOOKS goes on the PERSON, as attribute "wearing", in plain
  English. This covers clothing AND the visible state of the body:
  "дівчина в купальнику" → person{gender:female, wearing:"a swimsuit"};
  "дівчина в білому купальнику" → person{gender:female, wearing:"a white swimsuit"};
  "чоловік у камуфляжі" → person{gender:male, wearing:"camouflage"};
  "a man with a bare torso" → person{gender:male, wearing:"a bare torso"};
  "shirtless man" → person{gender:male, wearing:"a bare torso"}, query_en
  "a shirtless man";
  "topless man" → person{gender:male, wearing:"a bare torso"}, query_en
  "a topless man";
  "barefoot girl" → person{gender:female, wearing:"bare feet"}, query_en
  "a barefoot girl".
  These adjectives ARE the description: they must survive in BOTH query_en and
  the attribute, and must NEVER become must_not — "topless" says what IS there,
  not what is absent.
  "жінка з татуюванням" → person{gender:female, wearing:"a tattoo"}.
  Never emit it as a separate object — a swimsuit lying on a chair is not a
  girl in a swimsuit, and splitting them makes the search accept exactly that.
- KEEP EVERY MODIFIER of a garment: colour, pattern, material, condition.
  "a white swimsuit" stays "a white swimsuit", never "a swimsuit"; the same
  for "a red dress", "a torn jacket", "a leather bag". Dropping the modifier
  looks harmless because everything downstream still works — it just answers
  a WIDER question than was asked, and the frame the user meant then ranks
  among every other swimsuit in the corpus.
  This applies to query_en as well: both must carry the modifier.
- "bare", "shirtless", "topless", "barefoot" describe how the person LOOKS.
  They are NOT negations: never put them in must_not, and never drop them
  from query_en. Dropping one turned "a man with a bare torso" into plain
  "a man", and the search then returned every man in the corpus.
- The "object" class becomes a HARD FILTER, so use a specific class only when
  the query names something unmistakably of that class. When in doubt, use
  "other" — it narrows nothing and lets similarity decide.
  A construction crane, an excavator, a tractor attachment, a wind turbine are
  NOT "vehicle": "vehicle" means road, water or air transport that carries
  people or goods (car, bus, motorcycle, boat, plane). Machinery and structures
  → "other".
  This is not pedantry: guessing "vehicle" for a tower crane filtered the query
  down to 40 frames and threw away the only correct photo, which similarity
  alone ranked first.
- Use only the allowed enum values for attribute NAMES. The value of
  "wearing" is free text; every other value comes from the closed list.
- Anything that is not a person, vehicle, bag, weapon, document, phone,
  animal or building is "other".
- A child, a boy, a girl, a baby, a teenager IS a person — emit
  object "person" with age_band=child, never "other". Emitting "other" throws
  the age away, and the search then cannot tell "a woman with a child" from
  "a woman with anything at all".
- A BODY PART is NOT a person: a leg, a foot, a hand, an arm, a face, hair,
  a tattoo, a wound. Emit "other" and nothing else. Adding a person for them
  is the same invention the first rule forbids, and it costs real results: on
  "a damaged leg" an invented person filtered out the close-up photograph of
  the injury — the only frame that answered the query.

Examples below cover different shapes of query on purpose: one object and
several, with and without negation, person and non-person.

Query: "шампанське"
{"query_en":"champagne","language":"uk","must":[{"object":"other","attributes":[]}],"must_not":[],"relations":[],"categories":[]}

Query: "ніж"
{"query_en":"a knife","language":"uk","must":[{"object":"weapon","attributes":[]}],"must_not":[],"relations":[],"categories":[]}

Query: "чоловік з банером"
{"query_en":"a man with a banner","language":"uk","must":[{"object":"person","attributes":[{"name":"gender","value":"male"}]},{"object":"other","attributes":[]}],"must_not":[],"relations":[],"categories":[]}

Query: "червона машина"
{"query_en":"a red car","language":"uk","must":[{"object":"vehicle","attributes":[{"name":"color","value":"red"}]}],"must_not":[],"relations":[],"categories":[]}

Query: "чоловік без окулярів"
{"query_en":"a man","language":"uk","must":[{"object":"person","attributes":[{"name":"gender","value":"male"}]}],"must_not":[{"object":"person","attributes":[{"name":"glasses","value":"true"}]}],"relations":[],"categories":[]}

Query: "a woman without a beard and without a mask near a dog"
{"query_en":"a woman near a dog","language":"en","must":[{"object":"person","attributes":[{"name":"gender","value":"female"}]},{"object":"animal","attributes":[]}],"must_not":[{"object":"person","attributes":[{"name":"beard","value":"true"}]},{"object":"person","attributes":[{"name":"mask","value":"true"}]}],"relations":[{"subject":0,"predicate":"near","target":1}],"categories":[]}

Query: "дві дівчини в синіх спідницях"
{"query_en":"two girls in blue skirts","language":"uk","must":[{"object":"person","attributes":[{"name":"gender","value":"female"},{"name":"wearing","value":"a blue skirt"}],"count":2}],"must_not":[],"relations":[],"categories":[]}

Query: "a man with a bare torso"
{"query_en":"a man with a bare torso","language":"en","must":[{"object":"person","attributes":[{"name":"gender","value":"male"},{"name":"wearing","value":"a bare torso"}]}],"must_not":[],"relations":[],"categories":[]}

Query: "дівчина в купальнику"
{"query_en":"a girl in a swimsuit","language":"uk","must":[{"object":"person","attributes":[{"name":"gender","value":"female"},{"name":"wearing","value":"a swimsuit"}]}],"must_not":[],"relations":[],"categories":[]}

Query: "a girl in a white swimsuit"
{"query_en":"a girl in a white swimsuit","language":"en","must":[{"object":"person","attributes":[{"name":"gender","value":"female"},{"name":"wearing","value":"a white swimsuit"}]}],"must_not":[],"relations":[],"categories":[]}

Query: "жіноча сумочка"
{"query_en":"a handbag","language":"uk","must":[{"object":"bag","attributes":[]}],"must_not":[],"relations":[],"categories":[]}

Query: "чорна жіноча сумочка"
{"query_en":"a black handbag","language":"uk","must":[{"object":"bag","attributes":[{"name":"color","value":"black"}]}],"must_not":[],"relations":[],"categories":[]}

Query: "пошкоджена нога"
{"query_en":"a damaged leg","language":"uk","must":[{"object":"other","attributes":[]}],"must_not":[],"relations":[],"categories":[]}

Query: "жінка з дитиною"
{"query_en":"a woman with a child","language":"uk","must":[{"object":"person","attributes":[{"name":"gender","value":"female"}]},{"object":"person","attributes":[{"name":"age_band","value":"child"}]}],"must_not":[],"relations":[],"categories":[]}

Query: "жінка з сумкою"
{"query_en":"a woman with a bag","language":"uk","must":[{"object":"person","attributes":[{"name":"gender","value":"female"}]},{"object":"bag","attributes":[]}],"must_not":[],"relations":[],"categories":[]}

Query: "אדם ללא משקפיים"
{"query_en":"a person","language":"he","must":[{"object":"person","attributes":[]}],"must_not":[{"object":"person","attributes":[{"name":"glasses","value":"true"}]}],"relations":[],"categories":[]}

Query: "red bag inside a black car"
{"query_en":"a red bag inside a black car","language":"en","must":[{"object":"bag","attributes":[{"name":"color","value":"red"}]},{"object":"vehicle","attributes":[{"name":"color","value":"black"}]}],"must_not":[],"relations":[{"subject":0,"predicate":"inside","target":1}],"categories":[]}

Query: "щось підозріле біля паркану"
{"query_en":"something suspicious near a fence","language":"uk","must":[],"must_not":[],"relations":[],"categories":[]}
"""

#: Запити, коротші за це, парсер обходить.
#:
#: Спочатку тут стояло 2: на односкладовому «ніж» модель вигадувала дві
#: сутності person і перекладала слово як "needle". Але після виправлення
#: промпту (пряма заборона вигадувати плюс збалансовані приклади) односкладові
#: розбираються правильно 8 із 8 — «чоловік» дає gender=male, «дитина» дає
#: age_band=child, «собака» дає animal.
#:
#: І обхід став ШКОДИТИ: запит «чоловік» саме той, де стать вирішує, а він
#: пропускався, тож «чоловік» і «жінка» повертали частково те саме. Тому
#: обхід лишається тільки для порожнього запиту.
BYPASS_MAX_WORDS = 0

#: Стеля генерації. Виміряно на реальних запитах: коректний розбір займає
#: 55–88 токенів, тож 256 дає майже триразовий запас.
#:
#: Це не економія памʼяті, а ціна ПОМИЛКИ. Обрізання виявляється лише тоді,
#: коли модель уперлася в стелю, тож стеля і є вартістю невдачі. З лімітом 768
#: розмитий запит коштував 18.8 с, перш ніж впасти; з 256 — утричі менше.
#: Піднято з 256, коли в схему додалася кількість (`Entity.count`): поле
#: зʼявляється в КОЖНІЙ сутності, тож на двосутнісному запиті JSON виріс і
#: почав обриватися. Ціна помилки та сама — стеля це вартість невдачі, — але
#: обрив на правильному запиті гірший за зайві токени на розмитому.
MAX_PARSE_TOKENS = 384

#: Слова, наявність яких означає, що розбір потрібен навіть у короткому запиті.
NEGATION_MARKERS = (
    "без", "не ", "немає", "without", "no ", "not ", "ohne", "bez", "ללא", "בלי",
)


class ParserUnavailable(RuntimeError):
    """Ваг парсера немає або llama-cpp не встановлено."""


class TruncatedGeneration(RuntimeError):
    """Генерацію обірвано на межі токенів — JSON неповний попри граматику."""


class QueryParser:
    """Обгортка над Qwen3-GGUF із граматичним обмеженням виводу."""

    def __init__(
        self,
        registry: "ModelRegistry | None" = None,
        *,
        context_size: int = 4096,
        gpu_layers: int = -1,
        seed: int = 0,
    ) -> None:
        self.registry = registry or get_registry()
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.seed = seed
        self._llm = None
        self._grammar = None

    # ── завантаження ────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            return bool(self._gguf_path())
        except (ModelNotFetched, KeyError, ImportError):
            return False

    def _gguf_path(self) -> str:
        base = self.registry.local_path(MODEL_NAME)
        candidates = sorted(base.rglob("*.gguf"))
        if not candidates:
            raise ModelNotFetched(f"у {base} немає жодного файлу .gguf")
        return str(candidates[0])

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama, LlamaGrammar
        except ImportError as exc:  # pragma: no cover — залежить від оточення
            raise ParserUnavailable(f"llama-cpp-python не встановлено: {exc}") from exc

        path = self._gguf_path()
        logger.info("завантаження парсера %s", path)
        self._llm = Llama(
            model_path=path,
            n_ctx=self.context_size,
            n_gpu_layers=self.gpu_layers,
            seed=self.seed,
            verbose=False,
        )
        # Граматика будується зі схеми, а не пишеться руками: коли схема
        # зміниться, обмеження зміниться разом із нею й розʼїхатися не зможе.
        # Кеш префікса. Системний промпт це ~3000 токенів із 4096 контексту, і
        # він НЕ ЗМІНЮЄТЬСЯ між запитами — змінюється лише останній рядок із
        # самим запитом. Без кешу llama.cpp прораховує ці 3000 токенів заново
        # на КОЖНОМУ розборі, і саме це, а не генерація, займало більшість часу
        # (виміряно: 10.3 с у середньому).
        #
        # `LlamaRAMCache` шукає найдовший збіг префікса, тож наш випадок для
        # нього ідеальний: спільна частина максимальна, розбіжність — у хвості.
        try:
            from llama_cpp import LlamaRAMCache

            self._llm.set_cache(LlamaRAMCache(capacity_bytes=2 << 30))
        except ImportError:  # старіші версії llama-cpp-python
            logger.debug("LlamaRAMCache недоступний — розбір буде повільніший")

        self._grammar = LlamaGrammar.from_json_schema(
            json.dumps(StructuredQuery.model_json_schema()), verbose=False
        )

    # ── розбір ──────────────────────────────────────────────────────────────

    @staticmethod
    def should_bypass(query: str) -> bool:
        """Чи вартий запит розбору взагалі.

        З одного іменника витягувати нічого: заперечень немає, відношень
        немає, звʼязувати нічого. Єдиним наслідком розбору лишається ризик
        галюцинації — виміряно на «ніж», де модель вигадала дві сутності
        person і переклала слово як "needle".
        """
        lowered = f" {query.lower().strip()} "
        if any(marker in lowered for marker in NEGATION_MARKERS):
            return False
        return len(query.split()) <= BYPASS_MAX_WORDS

    def parse(self, query: str, *, max_tokens: int = MAX_PARSE_TOKENS) -> StructuredQuery:
        """Запит → StructuredQuery. Валідність JSON гарантована граматикою."""
        if self.should_bypass(query):
            return EMPTY.model_copy(update={"query_en": query})
        self._ensure_loaded()
        response = self._llm.create_chat_completion(  # type: ignore[union-attr]
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query!r}"},
            ],
            grammar=self._grammar,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        choice = response["choices"][0]
        # Граматика гарантує ФОРМУ, але не те, що генерація вміститься в
        # бюджет токенів. На розмитому запиті модель розговорилася й обірвалася
        # посеред рядка — JSON лишився невалідним попри граматику. Тож
        # обрізання перевіряємо явно, а не сподіваємося, що його не буде.
        if choice.get("finish_reason") == "length":
            raise TruncatedGeneration(
                f"розбір обірвано на межі {max_tokens} токенів — запит надто розмитий"
            )
        parsed = StructuredQuery.model_validate_json(choice["message"]["content"])
        return _repair(parsed, fallback_text=query)

    def parse_or_empty(self, query: str) -> StructuredQuery:
        """Розбір із мʼякою деградацією.

        Якщо парсер недоступний, пошук має лишитися щільним, а не впасти.
        Відсутність ваг на машині розробника — не привід ламати пошук.
        """
        try:
            return self.parse(query)
        except Exception as exc:  # noqa: BLE001 — деградація важливіша за причину
            logger.warning("розбір запиту не вдався (%s), працюємо щільним пошуком", exc)
            return EMPTY.model_copy(update={"query_en": query})


def _repair(parsed: StructuredQuery, *, fallback_text: str) -> StructuredQuery:
    """Прибрати те, що граматика пропустила, а сенс — ні."""
    if not parsed.query_en.strip():
        parsed = parsed.model_copy(update={"query_en": fallback_text})
    # Відношення, що вказують за межі переліку обʼєктів, безпечніше відкинути,
    # ніж пробувати вгадати намір.
    valid = [
        relation
        for relation in parsed.relations
        if relation.subject < len(parsed.must) and relation.target < len(parsed.must)
    ]
    if len(valid) != len(parsed.relations):
        logger.debug("відкинуто %d відношень із хибними індексами",
                     len(parsed.relations) - len(valid))
        parsed = parsed.model_copy(update={"relations": valid})
    return parsed


_default: QueryParser | None = None


def get_parser() -> QueryParser:
    global _default
    if _default is None:
        _default = QueryParser()
    return _default
