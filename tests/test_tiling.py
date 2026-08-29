"""Тести плиткування — чиста геометрія, без завантаження моделей."""

from __future__ import annotations

import pytest

from vsearch.represent.tiling import Region, deduplicate, tile_pixels


class TestRegion:
    def test_площа(self):
        assert Region(0.1, 0.1, 0.5, 0.4, "tile").area_ratio == pytest.approx(0.2)

    def test_рамка_поза_кадром_це_помилка(self):
        with pytest.raises(ValueError, match="поза кадром"):
            Region(1.5, 0.1, 0.2, 0.2, "tile")

    def test_порожня_рамка_це_помилка(self):
        with pytest.raises(ValueError, match="порожня"):
            Region(0.1, 0.1, 0.0, 0.2, "tile")

    def test_переведення_в_пікселі(self):
        assert Region(0.25, 0.5, 0.5, 0.5, "tile").to_pixels(800, 400) == (200, 200, 600, 400)

    def test_пікселі_не_вилазять_за_кадр(self):
        left, top, right, bottom = Region(0.9, 0.9, 0.1, 0.1, "tile").to_pixels(100, 100)
        assert right <= 100 and bottom <= 100

    def test_iou_однакових_рамок(self):
        r = Region(0.1, 0.1, 0.3, 0.3, "tile")
        assert r.iou(r) == pytest.approx(1.0)

    def test_iou_рамок_що_не_перетинаються(self):
        a = Region(0.0, 0.0, 0.2, 0.2, "tile")
        b = Region(0.5, 0.5, 0.2, 0.2, "tile")
        assert a.iou(b) == 0.0


def _covered(width: int, height: int, tiles: list[Region]) -> bool:
    """Чи покривають плитки кадр цілком, без смуг між ними.

    Перевіряється покроково по сітці точок: дірка шириною менше за крок сітки
    все одно провалила б хоч одну точку, бо крок вибрано дрібнішим за будь-яке
    можливе перекриття.
    """
    for gx in range(0, width, 17):
        for gy in range(0, height, 17):
            x, y = gx / width, gy / height
            if not any(
                t.x <= x < t.x + t.w and t.y <= y < t.y + t.h for t in tiles
            ):
                return False
    return True


class TestTilePixels:
    def test_знімок_менший_за_плитку_не_ділиться(self):
        """Цілий кадр індексується окремо, тож плитка на весь кадр — дублікат."""
        assert tile_pixels(300, 200, tile_size=384) == []

    def test_роздільність_плитки_НЕ_залежить_від_розміру_знімка(self):
        """Властивість, заради якої часткова сітка й замінена (ADR-010).

        Сітка 2×2 дає на 1280×960 плитку 768 px, а на 4032×3024 — 2419 px:
        частка кадру та сама, а стискання під модель уп'ятеро сильніше. Саме
        тому дрібні обʼєкти зникали переважно на великих файлах.
        """
        for width, height in [(640, 480), (1280, 960), (2000, 1500)]:
            first = tile_pixels(width, height, tile_size=288, max_tiles=256)[0]
            left, top, right, bottom = first.to_pixels(width, height)
            assert (right - left, bottom - top) == (288, 288)

    def test_плитки_покривають_кадр_повністю(self):
        for width, height in [(640, 480), (960, 1280), (1600, 900)]:
            tiles = tile_pixels(width, height, tile_size=288)
            assert _covered(width, height, tiles), f"дірка в покритті {width}×{height}"

    def test_ліміт_плиток_не_ламає_покриття(self):
        """Найтонше місце запобіжника.

        Обмежити кількість плиток можна двома способами: збільшити крок або
        збільшити саму плитку. Перший дешевший і лишає між плитками смуги —
        обʼєкт у такій смузі зникає остаточно, і жоден поріг цього не видно.
        """
        width, height = 4032, 3024
        tiles = tile_pixels(width, height, tile_size=288, max_tiles=64)
        assert len(tiles) <= 64
        assert _covered(width, height, tiles)

    def test_ліміт_збільшує_плитку_а_не_крок(self):
        width, height = 4032, 3024
        tiles = tile_pixels(width, height, tile_size=288, max_tiles=64)
        left, top, right, bottom = tiles[0].to_pixels(width, height)
        assert right - left > 288, "плитка мала вирости, інакше з'явилися б дірки"

    def test_обʼєкт_на_межі_плиток_цілком_у_якійсь_плитці(self):
        """Заради цього й потрібне перекриття.

        Без нього обʼєкт рівно на стику розрізається навпіл і не знаходиться
        в жодній плитці — саме той випадок, заради якого плиткування робиться.
        """
        width, height = 1000, 1000
        tiles = tile_pixels(width, height, tile_size=384, overlap=0.2)
        step = int(384 * 0.8)
        # Обʼєкт 40 px по центру першого стику.
        obj = Region((step - 20) / width, (step - 20) / height, 40 / width, 40 / height, "object")
        assert any(
            t.x <= obj.x and t.y <= obj.y
            and t.x + t.w >= obj.x + obj.w and t.y + t.h >= obj.y + obj.h
            for t in tiles
        ), "обʼєкт на стику не вміщується цілком у жодну плитку"

    def test_дрібніша_плитка_дає_більший_приріст_частки(self):
        """Монотонність, без якої профіль quality не мав би сенсу.

        Приріст дорівнює 1/площа_плитки: обʼєкт, що займав 1% кадру, у плитці
        займає в стільки разів більше. Виміряний обрив між 288 і 352 px саме
        про це — на 288 сумки дали 54.9%, на 352 лише 2.1%.
        """
        width, height = 1280, 960
        coarse = tile_pixels(width, height, tile_size=384, max_tiles=256)[0]
        fine = tile_pixels(width, height, tile_size=224, max_tiles=256)[0]
        assert 1.0 / fine.area_ratio > 1.0 / coarse.area_ratio

    def test_некоректний_розмір_плитки(self):
        with pytest.raises(ValueError):
            tile_pixels(800, 600, tile_size=0)

    def test_некоректне_перекриття(self):
        with pytest.raises(ValueError):
            tile_pixels(800, 600, tile_size=288, overlap=1.0)


class TestРозрізанняНаСтику:
    """Межа перекриття, яку варто тримати на видноті.

    Обʼєкт, ширший за перекриття, може лягти рівно на стик і не вміститися
    цілком у жодну плитку. Перекриття 0.2 від плитки 288 px це 58 px, тобто
    гарантія поширюється на обʼєкти приблизно до 58 px, а не на будь-які.

    Це свідома межа, а не недогляд: обʼєкти, заради яких плиткування й
    робиться, дрібніші за неї, а більші знаходяться й на цілому кадрі. Але
    залежність між двома числами існує, і тест фіксує її явно — бо зниження
    перекриття заради швидкості зламало б її беззвучно.
    """

    #: Плитка 288 px із перекриттям 0.2 йде з кроком 230, тож смуга спільних
    #: пікселів між сусідніми плитками — рівно 58.
    TILE, OVERLAP, SEAM = 288, 0.2, 58

    def test_обʼєкт_ширший_за_перекриття_може_не_вміститися(self):
        width = height = 1000
        tiles = tile_pixels(
            width, height, tile_size=self.TILE, overlap=self.OVERLAP, max_tiles=256
        )
        step = self.TILE - self.SEAM
        wide = self.SEAM + 20
        # Обʼєкт починається трохи раніше смуги перекриття й закінчується
        # трохи пізніше: для лівої плитки він завеликий, у праву не влазить
        # початком. Саме та позиція, де гарантія вичерпується.
        obj = Region((step - 5) / width, 0.4, wide / width, wide / height, "object")

        assert not any(obj.containment(t) > 0.999 for t in tiles), (
            "тест застарів: перекриття тепер покриває й ширші обʼєкти, "
            "отже гарантію можна посилити"
        )

    def test_обʼєкт_вужчий_за_перекриття_вміщується_завжди(self):
        width = height = 1000
        tiles = tile_pixels(
            width, height, tile_size=self.TILE, overlap=self.OVERLAP, max_tiles=256
        )
        narrow = self.SEAM - 2

        for offset in range(0, width - narrow, 7):
            obj = Region(offset / width, 0.4, narrow / width, narrow / height, "object")
            assert any(obj.containment(t) > 0.999 for t in tiles), (
                f"обʼєкт {narrow} px на позиції {offset} не вміщується цілком "
                f"у жодну плитку, хоча вужчий за перекриття"
            )


class TestDeduplicate:
    def test_дублікати_прибираються(self):
        a = Region(0.1, 0.1, 0.3, 0.3, "object", score=0.9)
        almost = Region(0.105, 0.105, 0.3, 0.3, "object", score=0.5)
        assert len(deduplicate([a, almost])) == 1

    def test_лишається_рамка_з_вищою_оцінкою(self):
        a = Region(0.1, 0.1, 0.3, 0.3, "object", label="краща", score=0.9)
        b = Region(0.105, 0.105, 0.3, 0.3, "object", label="гірша", score=0.5)
        assert deduplicate([a, b])[0].label == "краща"

    def test_різні_рамки_лишаються(self):
        a = Region(0.0, 0.0, 0.2, 0.2, "object")
        b = Region(0.6, 0.6, 0.2, 0.2, "object")
        assert len(deduplicate([a, b])) == 2
