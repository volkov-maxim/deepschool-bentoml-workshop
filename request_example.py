import bentoml
import httpx
from PIL import Image as Image_pil
import numpy as np
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

SERVICE_URL = "http://localhost:3000"
# При первом запуске сервис скачивает веса энкодера, поэтому ждём с запасом
READY_TIMEOUT = 300

# Контрольный пример: исходное изображение 822x602 (RGB),
# пайплайн делает Resize(512, 512) и возвращает один канал вероятностей
CONTROL_IMAGE = Path("data/inimages/T02_Box051_2017-10-31T11-51-17-647_19.png")
EXPECTED_MASK_SHAPE = (1, 512, 512)
THRESHOLD = 0.5


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image_pil.open(str(path)).convert("RGB"), dtype=np.uint8)


def wait_ready(url: str, timeout: float) -> None:
    """/readyz: модель загружена, сервис готов принимать запросы."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            if httpx.get(f"{url}/readyz", timeout=5).status_code == 200:
                logging.info("/readyz: OK")
                return
        except httpx.TransportError:
            pass  # сервис ещё не поднял порт
        if time.monotonic() > deadline:
            raise TimeoutError(f"Сервис {url} не стал готов за {timeout} с")
        time.sleep(2)


def check_live(url: str) -> None:
    """/livez: процесс сервиса запущен и отвечает."""
    resp = httpx.get(f"{url}/livez", timeout=5)
    assert resp.status_code == 200, f"/livez вернул {resp.status_code}"
    logging.info("/livez: OK")


def check_mask(mask: np.ndarray) -> None:
    assert isinstance(mask, np.ndarray), f"ожидали np.ndarray, получили {type(mask)}"
    assert mask.shape == EXPECTED_MASK_SHAPE, (
        f"форма маски {mask.shape}, ожидали {EXPECTED_MASK_SHAPE}"
    )
    assert np.issubdtype(mask.dtype, np.floating), f"тип маски {mask.dtype}"
    assert 0.0 <= mask.min() and mask.max() <= 1.0, (
        f"значения вне [0, 1]: min={mask.min()}, max={mask.max()}"
    )


if __name__ == "__main__":
    try:
        wait_ready(SERVICE_URL, READY_TIMEOUT)
        check_live(SERVICE_URL)

        input_fg = load_image(CONTROL_IMAGE)
        logging.debug(f"Start processing: {CONTROL_IMAGE.name}, {input_fg.shape}")

        with bentoml.SyncHTTPClient(SERVICE_URL) as client:
            logging.info(f"Model version: {client.get_model_version()}")

            result = client.run_processing(input_fg)

        check_mask(result)
        logging.info(
            f"Ready: shape={result.shape}, dtype={result.dtype}, "
            f"min={result.min():.3f}, max={result.max():.3f}"
        )
    except (AssertionError, TimeoutError, httpx.HTTPError) as e:
        logging.error(f"Проверка не пройдена: {e}")
        sys.exit(1)

    out_path = f"{CONTROL_IMAGE.stem}_bento_res.png"
    Image_pil.fromarray(((result[0] > THRESHOLD) * 255).astype(np.uint8)).save(out_path)
    logging.info(f"Маска сохранена: {out_path}")
