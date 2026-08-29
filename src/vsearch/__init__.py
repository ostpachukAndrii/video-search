"""vsearch — семантичний пошук по відео та фото для розслідувань.

Офлайн-режим вмикається на імпорті пакета, а не в точці входу: інакше будь-який
скрипт чи тест, що імпортує vsearch напряму, обійшов би цей запобіжник.
"""

from vsearch.backends.registry import enforce_offline_env

enforce_offline_env()

__version__ = "0.1.0"
