# crypti_anasys

Проєкт учениці: аналіз новин на льоту і перевірка, чи новини допомагають передбачати рух BTC, ETH і золота.

## Поточне завдання

Локальний файнтюн моделі на DGX Spark — повний план у [FINETUNE_PLAN.md](FINETUNE_PLAN.md).
Скрипти файнтюну й черга запусків описані нижче; результати й звіт — у `experiments/2026-11-23_news-direction/`.

## Файли

- `news_bd1_annotated.csv`, `news_bd2_annotated.csv` — новини з розміткою (роздільник `;`, UTF-8 з BOM).
- `crypto_price_bd.csv` — ціни BTC/ETH/USDT/золота з кроком 30 хв (роздільник `;`).
- `annotate_news.py` — розмітка новин через Claude API (потрібен ключ у `.env`, для файнтюну не потрібен).
- `analyze_news_prices.py` — зіставлення новин з цінами і кореляції.
- `METHODS.md` — детальний опис методів розмітки, аналізу, навчання й оцінки.
- `add_price_features.py` — додає в `*_annotated.csv` стан ринку перед новиною: `inertia_*` (чи тримає курс напрям за 24 год) і `volatility_*`, `volatility_*_7d` (наскільки сильно рухається курс, % за день). Після `annotate_news.py collect` запускати знову.
- `finetune_openai.py` — спроба файнтюну через OpenAI (не вдалася: платформу згортають); локальному файнтюну не потрібен.
- `db_news.py`, `cryp_rate_db.py`, `desc.py` — оригінальні скрипти збору даних.
- Локальний файнтюн (FINETUNE_PLAN.md):
  - `ft_common.py` — спільні константи (межі поділу, пороги), метрики й графіки;
  - `ft_data.py` — датасет `experiments/data/` (у git не йде, перебудовується скриптом);
  - `ft_baselines.py` — базові правила;
  - `ft_train.py` — навчання (LoRA або повне) і оцінка однієї моделі;
  - `ft_zeroshot.py` — велика модель без навчання (vLLM);
  - `ft_report.py` — зведена таблиця і графіки всіх запусків;
  - `ft_predict.py` — використання навченої моделі на звичайному комп'ютері без GPU, інструкція — `INFERENCE_CPU.md`.
- `experiments/` — Dockerfile, запуск у контейнері на DGX Spark (`spark_run.sh`), черга запусків (`run_all.sh`, `queue.txt`).

## Правила репозиторію

- **Усі коміти — від імені учениці:** `honorablement <143546481+Vacpize@users.noreply.github.com>`.
  Перед першим комітом на новому комп'ютері налаштуй це локально для репо:
  ```bash
  git config user.name "honorablement"
  git config user.email "143546481+Vacpize@users.noreply.github.com"
  ```
- **Ніколи не додавай `Co-Authored-By: Claude`** чи будь-які інші підписи ШІ в коміти та PR.
- `.env` з ключами в git не потрапляє; ваги моделей і чекпойнти теж не комітити.
- Python запускати системним `python3` (або в NVIDIA-контейнері на DGX Spark), без `.venv` у репозиторії.
