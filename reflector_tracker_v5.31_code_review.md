# Code review: reflector_tracker_v5.31(1).py

## 1. Область проверки

Проверен единый Python-файл `reflector_tracker_v5.31(1).py` (8665 строк).

Проверки:
- синтаксический разбор AST и `py_compile`;
- импорт модуля;
- структура классов и методов;
- ориентировочная цикломатическая сложность;
- обработка ошибок;
- конфигурация/JSON round-trip;
- RTSP/FFmpeg lifecycle;
- Tkinter main-loop и фоновые потоки;
- алгоритм детектора/трекера;
- повторяющиеся блоки;
- синтетическая дифференциальная проверка исходного и оптимизированного вариантов.

Ограничение: реальная Hikvision/RTSP-камера и натурные изображения отражателей в этой проверке не подключались. Поэтому замечания, требующие физического видеопотока, помечены как «требует натурной проверки».

## 2. Итоговая оценка

Код содержит сильную прикладную логику и значительный объём защитных механизмов, но в текущем виде это высокосвязный монолит. Главный риск — не синтаксис, а регрессии: изменения в одном большом методе легко влияют на сопровождение, калибровку, визуализацию и сохранённое состояние.

Ключевые метрики исходника:
- `ReflectorApp`: ~4950 строк;
- `ReflectorTracker`: ~1175 строк;
- `ReflectorDetector`: ~690 строк;
- `_create_widgets`: 625 строк;
- `detect_in_region`: 612 строк, ориентировочная сложность ~75;
- `_update_track`: 551 строк, ориентировочная сложность ~84;
- `_analyze_ir_flash_frames`: 543 строки, сложность ~54;
- 23 обработчика `except Exception`;
- один `except Exception: pass`.

## 3. Замечания

| ID | Приоритет | Замечание | Риск | Статус в optimized |
|---|---|---|---|---|
| CR-01 | P1 | **Потеря части ИК-модели при JSON round-trip.** `to_dict()` сохраняет `ir_confirmed_models` целиком, но `load()` выбрасывает `tracking_template`, `tracking_template_size`, `tracking_offset_valid`, `signature_center_x/y`. | После перезапуска сопровождение работает не с той же моделью, которая была создана после Day↔Night-калибровки. | **Исправлено.** Поля валидируются и восстанавливаются. |
| CR-02 | P1 | **Покадровый бесконечный reset при некорректной Pn.** `reset_regions()` исключает `w<3/h<3`, а `update_settings()` сравнивает сигнатуру со всем исходным списком. | Повторная инициализация треков на каждом кадре, потеря истории/скорости/сглаживания. | **Исправлено.** Сигнатуры нормализованы одинаково. |
| CR-03 | P1 | **RTSP reader при EOF/аварии не сбрасывает `is_connected`.** | Приложение считает мёртвый поток подключённым и может бесконечно показывать `FPS: --`. | **Исправлено.** Состояние сбрасывается в `finally`, сохраняется причина. |
| CR-04 | P1 | **Гонка состояния RTSP при старте.** Reader thread запускается раньше присваивания `is_connected=True`; мгновенный exit может оставить неправильное состояние. | Ложное состояние подключения. | **Исправлено.** Флаг устанавливается до старта thread. |
| CR-05 | P1 | **Подключение RTSP блокирует Tk main thread.** `_connect_rtsp()` синхронно вызывает `ffprobe` с timeout до 10 с. | Окно «зависает» при плохой сети/URL. | Не менялось: требует более крупного async-refactor цепочки запуска. |
| CR-06 | P1 | **Некорректное число в UI silently ignored.** `_update_calibration_from_ui()` ловит `TclError/ValueError` и просто `return`; запуск/сохранение могли продолжиться со старыми параметрами. | Пользователь видит одно значение, алгоритм использует другое. | **Исправлено для критичных действий.** Метод возвращает success/failure; запуск, калибровка, выбор областей, пресеты и сохранение валидируются. |
| CR-07 | P1 | **Противоречие семантики нулевой координаты.** Комментарий/Help говорят «первый реальный LOCK», но `reset_regions()` при сохранённом IR anchor сразу задаёт `initial_position=center`. | dX/dY/dR могут иметь другое начало отсчёта, чем ожидает пользователь. | Не менялось: нужно выбрать требуемую физическую семантику. |
| CR-08 | P1 | **Сопровождение существенно зависит от FPS.** `velocity` — px/frame, `max_jump` — px/frame, `lost_hold_frames` — frames, damping/smoothing — per-frame. | При 10 FPS и 30 FPS один и тот же физический ход обрабатывается по-разному. Это особенно важно при просадках CPU/RTSP. | Не менялось: нужен dt-aware tracker и натурная проверка. |
| CR-09 | P1 | **Нет PTS/capture timestamp исходного RTSP кадра.** CSV пишет локальное время обработки/записи окна. | Для измерительной системы невозможно точно отделить время захвата камерой от сетевой/декодирующей задержки. | Не менялось. |
| CR-10 | P1 | **Загрузка полного JSON не валидирует большинство диапазонов и межполей.** | Повреждённый/ручной JSON может создать отрицательные/неадекватные параметры или `max_area < min_area`. | Частично защищено downstream; полной schema-validation пока нет. |
| CR-11 | P1 | `make_regions_exclusive()` в крайнем случае может вернуть меньше элементов, чем reference regions; несколько callers затем используют позиционный индекс. | При некорректных/off-frame областях возможна потеря соответствия Pn→region. | Не менялось; требуется изменить контракт на `List[Optional[Rect]]` либо гарантировать cardinality. |
| CR-12 | P1 | **Фоновая разность использует `absdiff`.** Затем `max(gray, absdiff)`: сильное *потемнение* относительно фона превращается в яркий сигнал. | Тень/закрытие светлого участка потенциально может выглядеть как bright candidate. | Не менялось: алгоритмическая правка требует проверки на реальных кадрах. |
| CR-13 | P1/P2 Security | Для HTTPS ISAPI полностью отключены hostname/certificate verification. | Возможен MITM в недоверенной сети; передаются реквизиты управления камерой. | Не менялось, чтобы не сломать камеры с self-signed cert. Нужен явный `allow_insecure_tls`. |
| CR-14 | P2 | `RTSPCamera.connect()` считает подключение успешным сразу после запуска FFmpeg, не дожидаясь первого валидного кадра. | «RTSP подключен» может появиться для процесса, который через мгновение завершится. | Частично улучшено lifecycle; first-frame handshake не добавлялся. |
| CR-15 | P2 | FFmpeg RTSP запускается с `stderr=DEVNULL`. | Теряется диагностическая причина падения декодера/RTSP после старта. | Ошибка старта через ffprobe теперь показывается; runtime stderr по-прежнему не читается. |
| CR-16 | P2 | Нет watchdog/reconnect RTSP. | Краткий сетевой разрыв требует ручного переподключения. | Не менялось. |
| CR-17 | P2 | WindowRecorder после timeout делал только `terminate()` без повторного wait/kill. | Возможен зависший/оставшийся FFmpeg-процесс. | **Исправлено.** terminate→wait→kill→wait. |
| CR-18 | P2 | `logging.basicConfig()` выполнялся при импорте модуля. | Модуль меняет глобальную logging-конфигурацию внешнего приложения/тестов. | **Исправлено.** Настройка перенесена в `main()`. |
| CR-19 | P2 | Main-loop всегда делал `after(30)` **после** обработки. | Реальный период = processing_time + 30 ms; 20 ms обработки дают ~20 FPS вместо ~33 FPS. | **Исправлено.** Delay компенсирует время обработки. |
| CR-20 | P2 | FPS рассчитывался через `time.time()`. | Перевод системных часов/NTP может дать выброс. | **Исправлено.** Для FPS используется `perf_counter()`. |
| CR-21 | P2 | Лишняя копия полного кадра: `current_frame = frame.copy()`, после чего tracker/visualization создаёт следующие копии. | Лишняя memory bandwidth на каждом кадре. | **Исправлено.** Сохраняется ссылка на уже независимый доставленный frame. |
| CR-22 | P2 | `normalized_reflector_patch()` и `locate_reflector_by_template()` могут конвертировать полный BGR frame в grayscale для каждого трека. | При нескольких Pn — повторная дорогостоящая `cvtColor` одного кадра. | Не менялось: следующий безопасный optimization — один shared gray frame per cycle. |
| CR-23 | P2 | Очень крупные методы `_update_track`, `detect_in_region`, `_analyze_ir_flash_frames`. | Трудно тестировать, сложно локализовать регрессии, много неявных invariants. | Не дробил на модули; в едином файле стоит хотя бы делить на private helpers. |
| CR-24 | P2 | `ReflectorApp` смешивает GUI, storage, RTSP, ISAPI, recording, IR analysis и orchestration. | Высокая связанность и сложность сопровождения. | Не менялось радикально для сохранения одного запускаемого файла. |
| CR-25 | P2 | IR candidates/models передаются как `Dict` с десятками строковых ключей. | Опечатка в ключе обнаруживается только runtime; IDE/type checker почти не помогает. | Не менялось; рекомендованы `TypedDict`/dataclass. |
| CR-26 | P2 | Схема настроек повторяется в `PRESET_FIELDS`, `REGION_DETECTION_FIELDS`, metadata, defaults, `to_dict`, `load`, UI sync. | Schema drift: поле можно добавить в одно место и забыть в другом. CR-01 уже является примером такого drift. | Частично уменьшено; нужна единая декларативная schema. |
| CR-27 | P2 | Три IR-настройки фактически инварианты (`search_scale=1`, strict=true, global=false), но присутствуют в JSON, UI variables и нескольких sync blocks. | Лишний код и риск расхождения. | **Централизовано** через `_enforce_ir_region_invariants()`. |
| CR-28 | P2 | Два почти идентичных блока merge/deduplicate IR candidates. | Исправление одного блока легко забыть перенести во второй. | **Исправлено.** Общий `merge_ir_candidates()`, поведение regression-tested. |
| CR-29 | P2 | Три копии вычисления общего bounding ROI. | Дублирование. | **Исправлено.** `bounding_rect()`. |
| CR-30 | P2 | `_diamond_halo_score()` в `ReflectorApp` был однострочным proxy к global function. | Лишний слой без семантики. | **Удалено.** |
| CR-31 | P2 | 23 `except Exception`; heatmap имел `except Exception: pass`. | Скрытие programmer errors, затруднение диагностики. | Silent heatmap exception заменён на конкретные типы + debug log; остальные требуют адресного разбора. |
| CR-32 | P2 | `_update_display()` ловит любой Exception каждый кадр и продолжает цикл. | Постоянный bug может генерировать stack trace каждые ~30 ms и забить log/CPU. | Не менялось; нужен rate limiter + stop/escalation после N одинаковых ошибок. |
| CR-33 | P2 | `save_calibration()` не обрабатывал I/O/serialization error. | Permission/disk error превращается в Tk callback traceback. | **Исправлено.** |
| CR-34 | P2 | JSON сохраняется непосредственно в target file без temp+atomic replace. | Crash/отключение питания во время записи может повредить единственный файл. | Не менялось; рекомендуется atomic write. |
| CR-35 | P2 | Импорт bool preset: любая неизвестная строка (`"tru"`) silently становится False. | Тихое искажение импортированной настройки. | Не менялось; нужен strict parser. |
| CR-36 | P2 | User-Agent был `ReflectorTracker/5.16`, UI — v5.31. | Версионный drift и плохая диагностика. | **Исправлено.** `APP_VERSION`. |
| CR-37 | P2 | Для интервалов/таймаутов в нескольких местах используется `time.time()`. | Wall clock не подходит для elapsed durations. | FPS исправлен; остальные интервалы желательно перевести на monotonic/perf_counter. |
| CR-38 | P2 | CSV использует naive `datetime.now()` без UTC offset. | В ночь перехода DST локальные timestamps могут быть неоднозначны. | Не менялось; рекомендован `datetime.now().astimezone().isoformat()`. |
| CR-39 | P3 | CSV state file делает `flush()` на каждом записываемом кадре. | Лишний I/O; на медленном диске может влиять на UI. | Не менялось; 5 FPS делает риск умеренным. |
| CR-40 | P3 | `ImageGrab.grab()` всего окна 5 раз/с + resize + RGB→bytes. | Существенная нагрузка CPU/RAM; зависит от OS/window compositor. | Не менялось: это функциональная запись интерфейса. |
| CR-41 | P3 | RTSP декодируется в raw BGR24 через pipe. | Высокий memory bandwidth на 2K/4K stream. | Не менялось, поскольку tracking требует исходного resolution. |
| CR-42 | P3 | В коде остаётся строка `local+global`, хотя global fallback фактически запрещён. | Vestigial branch/терминология. | Сохранено для совместимости diagnostics. |
| CR-43 | P3 | `mser_delta` сохраняется, но фактически legacy-only. | Шум schema. | Сохранено осознанно ради старых JSON. |
| CR-44 | P3 | Help говорит, что HOLD/LOST «привязан к исходной области», тогда как verified IR tracking двигает локальный search center за последней подтверждённой позицией. | Документация не полностью соответствует фактическому алгоритму. | Не менялось; нужно уточнить формулировку после выбора ожидаемой семантики. |
| CR-45 | P3 | Нет выделенного regression/unit test suite рядом с приложением. | Изменения в трекере приходится проверять вручную; высокая вероятность возврата старых ошибок. | Для review выполнены отдельные synthetic regression tests, но в файл приложения tests не встраивались. |
| CR-46 | P3 | Нет явного файла зависимостей/зафиксированных версий Python/OpenCV/Pillow/Numpy. | Разные версии OpenCV/NumPy могут давать отличия в computer vision pipeline. | Не менялось. |
| CR-47 | P3 | Часть строк/методов не имеет точных return types; используются `Dict` без параметров. | Слабее статическая проверка. | Не менялось полностью. |
| CR-48 | P3 | Некоторые широкие exception handlers являются допустимой UI boundary-защитой, но не логируют stack trace. | Ошибки конфигурации/IO сложнее расследовать по log. | Требует адресного разделения expected/user errors и programmer errors. |

## 4. Наиболее важное алгоритмическое замечание

Сопровождение сейчас **frame-based**, а не **time-based**.

Примеры:
- `velocity = measurement - old_position` → px/frame;
- `max_jump` → px/frame;
- `lost_hold_frames` → количество кадров;
- decay velocity/confidence → коэффициент на кадр;
- smoothing alpha → коэффициент на кадр;
- расширение ROI → шаг на пропущенный кадр.

Из-за этого при падении FPS поведение фильтра меняется, хотя физическое движение отражателя то же самое. Для измерительной системы корректнее перейти к `dt`:
- velocity в px/s;
- max speed/acceleration в px/s и px/s²;
- HOLD в секундах;
- `alpha(dt)=1-exp(-dt/tau)`;
- expansion rate в px/s.

Это изменение не внесено автоматически, потому что оно меняет динамику трекера и должно проверяться на записанном реальном видеопотоке.

## 5. Отдельное замечание по фону

В `_signal()`:

```python
difference = cv2.absdiff(gray, background_crop.astype(np.uint8))
gray = np.maximum(gray, difference).astype(np.uint8)
```

`absdiff` одинаково усиливает и посветление, и потемнение. Если светлый фон был 240, а текущий пиксель стал 20, `difference=220`, поэтому он становится сильным положительным signal. Если задача — ловить именно появление яркого отражателя, надо проверить вариант положительной разности:

```python
difference = cv2.subtract(gray, background_crop.astype(np.uint8))
```

или разделить bright-change и dark-change на разные каналы признаков. Без реальных кадров эта правка не внесена.

## 6. Что изменено в reflector_tracker_v5.31_optimized.py

1. Исправлен round-trip `ir_confirmed_models`.
2. Исправлен бесконечный reset при invalid region.
3. Исправлен RTSP lifecycle и race `is_connected`.
4. Добавлено сохранение текста ошибки RTSP startup.
5. Усилен shutdown FFmpeg recorder/RTSP.
6. Убран global logging side-effect при import.
7. Добавлены `APP_VERSION`, `IR_MODEL_VERSION` и constants invariants.
8. Удалён proxy `_diamond_halo_score`.
9. Объединены два IR candidate merge blocks.
10. Объединены повторные вычисления bounding ROI.
11. Централизована синхронизация display layer fields.
12. Централизованы fixed IR invariants.
13. Убран silent `except Exception: pass` для heatmap.
14. Убрана лишняя полнокадровая копия `current_frame`.
15. FPS переведён на monotonic high-resolution clock.
16. Tk update cadence компенсирует processing time.
17. Добавлена валидация UI перед критичными действиями.
18. Добавлена обработка ошибки сохранения JSON.
19. Исправлен User-Agent v5.16→v5.31.
20. При runtime loss RTSP состояние камеры теперь становится disconnected.

## 7. Regression checks optimized

Пройдено:
- `py_compile`;
- import;
- equality default `CalibrationData.to_dict()` исходник vs optimized;
- JSON round-trip tracking template/model;
- 500 randomized `clip_rect`;
- 500 randomized `rect_distance`;
- randomized `diamond_halo_score` / `stable_component_center`;
- 200 randomized comparisons old-vs-refactored IR candidate merge;
- synthetic multi-frame detector/tracker differential: позиции, missed state и rendered frame совпадают с исходником;
- invalid-region test: повторный `update_settings()` не создаёт новый track.

## 8. Рекомендуемый следующий этап

Не переписывать сейчас весь tracker. Следующий инженерно оправданный шаг — создать regression harness на ваших реальных записанных проблемных сценах:
- благоприятный фон;
- яркое солнце/засветка;
- неподвижные отражатели;
- искусственное перемещение на известные 1/2/5/10 px;
- временная потеря;
- Day↔Night;
- RTSP dropout.

На каждый frame сохранять expected P1/P2 и автоматически считать:
- center error, px;
- jitter RMS на неподвижном объекте;
- false jumps;
- reacquisition time;
- lost rate;
- ID switches.

После этого можно безопасно менять frame-based фильтр на dt-aware и отдельно тестировать background `absdiff`.
