"""Idempotent Google Sheets logging for completed Kaggle experiments.

The module deliberately uses the Sheets REST API directly. Authentication is
the only Google-specific dependency, while the request transport remains easy
to replace in unit tests.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote


SPREADSHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
DEFAULT_SECRET_NAME = "GOOGLE_SERVICE_ACCOUNT_JSON"
DEFAULT_CREDENTIAL_DATASET_SLUG = "ecom-matching-google-sheets-credentials"
DEFAULT_CREDENTIAL_FILENAME = "google-service-account.json"
EXPERIMENTS_SHEET = "experiments"
CATEGORY_METRICS_SHEET = "category_metrics"

EXPERIMENT_HEADERS = (
    "run_id",
    "started_at_utc",
    "completed_at_utc",
    "synced_at_utc",
    "status",
    "experiment",
    "model",
    "dataset_ref",
    "kaggle_kernel_ref",
    "code_bundle_sha256",
    "training_wall_seconds",
    "training_seconds",
    "validation_seconds",
    "total_pipeline_seconds",
    "training_examples",
    "original_training_examples",
    "validation_examples",
    "validation_positive_examples",
    "validation_positive_rate",
    "macro_average_precision",
    "overall_average_precision",
    "examples_per_second",
    "padding_efficiency",
    "peak_vram_gib",
    "mean_score_order_gap",
    "epochs",
    "batch_size",
    "eval_batch_size",
    "gradient_accumulation",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "max_length",
    "attention_implementation",
    "sampling",
    "train_subset",
    "loss_weighting",
    "lexical_hard_negative_strength",
    "symmetric_validation",
    "label_smoothing",
    "seed",
    "config_json",
    "report_json",
)

CATEGORY_HEADERS = (
    "run_id",
    "completed_at_utc",
    "experiment",
    "model",
    "category",
    "average_precision",
)

TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
RequestCallable = Callable[..., Any]
SleepCallable = Callable[[float], None]


class SheetsLoggerError(RuntimeError):
    """Raised when an experiment cannot be safely synchronized."""


class SheetsApiError(SheetsLoggerError):
    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def column_letter(index: int) -> str:
    """Return the A1 column label for a one-based column index."""
    if index < 1:
        raise ValueError("Column index must be positive")
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _json_cell(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _cell(value: Any) -> Any:
    value = _json_safe(value)
    return "" if value is None else value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _peak_vram(report: Mapping[str, Any]) -> float | str:
    values = report.get("peak_vram_gib_by_rank")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ""
    finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return max(finite) if finite else ""


def build_experiment_row(
    completion: Mapping[str, Any],
    *,
    synced_at_utc: str | None = None,
) -> list[Any]:
    """Flatten a completion report while preserving the full JSON payload."""
    run_id = str(completion.get("run_id", "")).strip()
    if not run_id:
        raise SheetsLoggerError("Experiment completion is missing a non-empty run_id")
    report = _mapping(completion.get("training_report"))
    args = _mapping(report.get("args"))
    model = completion.get("model") or args.get("model") or ""
    values = {
        "run_id": run_id,
        "started_at_utc": completion.get("started_at_utc"),
        "completed_at_utc": completion.get("completed_at_utc"),
        "synced_at_utc": synced_at_utc or utc_now(),
        "status": completion.get("status"),
        "experiment": completion.get("experiment"),
        "model": model,
        "dataset_ref": completion.get("dataset_ref"),
        "kaggle_kernel_ref": completion.get("kaggle_kernel_ref"),
        "code_bundle_sha256": completion.get("code_bundle_sha256"),
        "training_wall_seconds": completion.get("training_wall_seconds"),
        "training_seconds": report.get("training_seconds"),
        "validation_seconds": report.get("validation_seconds"),
        "total_pipeline_seconds": report.get("total_pipeline_seconds"),
        "training_examples": report.get("training_examples"),
        "original_training_examples": report.get("original_training_examples"),
        "validation_examples": report.get("validation_examples"),
        "validation_positive_examples": report.get("validation_positive_examples"),
        "validation_positive_rate": report.get("validation_positive_rate"),
        "macro_average_precision": report.get("macro_average_precision"),
        "overall_average_precision": report.get("overall_average_precision"),
        "examples_per_second": report.get("examples_per_second"),
        "padding_efficiency": report.get("padding_efficiency"),
        "peak_vram_gib": _peak_vram(report),
        "mean_score_order_gap": report.get("mean_score_order_gap"),
        "epochs": args.get("epochs"),
        "batch_size": args.get("batch_size"),
        "eval_batch_size": args.get("eval_batch_size"),
        "gradient_accumulation": args.get("gradient_accumulation"),
        "learning_rate": args.get("learning_rate"),
        "weight_decay": args.get("weight_decay"),
        "warmup_ratio": args.get("warmup_ratio"),
        "max_length": args.get("max_length"),
        "attention_implementation": args.get("attention_implementation"),
        "sampling": args.get("sampling"),
        "train_subset": args.get("train_subset"),
        "loss_weighting": args.get("loss_weighting"),
        "lexical_hard_negative_strength": args.get("lexical_hard_negative_strength"),
        "symmetric_validation": args.get("symmetric_validation"),
        "label_smoothing": args.get("label_smoothing"),
        "seed": args.get("seed"),
        "config_json": _json_cell(args),
        "report_json": _json_cell(report),
    }
    return [_cell(values[header]) for header in EXPERIMENT_HEADERS]


def build_category_rows(completion: Mapping[str, Any]) -> list[list[Any]]:
    run_id = str(completion.get("run_id", "")).strip()
    if not run_id:
        raise SheetsLoggerError("Experiment completion is missing a non-empty run_id")
    report = _mapping(completion.get("training_report"))
    args = _mapping(report.get("args"))
    metrics = _mapping(report.get("per_category_average_precision"))
    prefix = [
        run_id,
        _cell(completion.get("completed_at_utc")),
        _cell(completion.get("experiment")),
        _cell(completion.get("model") or args.get("model")),
    ]
    return [
        prefix + [str(category), _cell(score)]
        for category, score in sorted(metrics.items(), key=lambda item: str(item[0]))
    ]


def load_service_account_info(service_account_json: str) -> dict[str, Any]:
    try:
        info = json.loads(service_account_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise SheetsLoggerError(
            f"{DEFAULT_SECRET_NAME} must contain a valid service-account JSON object"
        ) from error
    if not isinstance(info, dict):
        raise SheetsLoggerError(
            f"{DEFAULT_SECRET_NAME} must contain a JSON object"
        )
    required = {"type", "client_email", "private_key", "token_uri"}
    if info.get("type") != "service_account" or required - set(info):
        raise SheetsLoggerError(
            f"{DEFAULT_SECRET_NAME} is not a complete Google service-account key"
        )
    return info


def service_account_token(service_account_json: str) -> str:
    info = load_service_account_info(service_account_json)
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.service_account import Credentials
    except ImportError as error:
        raise SheetsLoggerError(
            "google-auth is required for Google Sheets synchronization"
        ) from error
    credentials = Credentials.from_service_account_info(
        info,
        scopes=[SPREADSHEETS_SCOPE],
    )
    credentials.refresh(Request())
    if not credentials.token:
        raise SheetsLoggerError("Google authentication returned an empty access token")
    return str(credentials.token)


def kaggle_secret(name: str = DEFAULT_SECRET_NAME) -> str:
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as error:
        raise SheetsLoggerError("kaggle_secrets is available only inside Kaggle") from error
    value = UserSecretsClient().get_secret(name)
    if not value:
        raise SheetsLoggerError(f"Kaggle Secret {name!r} is empty or unavailable")
    return value


def kaggle_service_account_json(
    *,
    secret_name: str = DEFAULT_SECRET_NAME,
    input_root: Path = Path("/kaggle/input"),
    dataset_slug: str = DEFAULT_CREDENTIAL_DATASET_SLUG,
    credential_filename: str = DEFAULT_CREDENTIAL_FILENAME,
) -> str:
    """Load a Kaggle Secret first, then the attached private Dataset file.

    The fixed Dataset path avoids scanning arbitrary notebook inputs for JSON
    credentials. Every returned value is validated before it leaves this
    function, and credential material is never included in an error message.
    """
    try:
        secret_value = kaggle_secret(secret_name)
        load_service_account_info(secret_value)
        return secret_value
    except Exception:
        pass

    credential_path = input_root / dataset_slug / credential_filename
    try:
        dataset_value = credential_path.read_text(encoding="utf-8")
        load_service_account_info(dataset_value)
    except Exception as error:
        raise SheetsLoggerError(
            "Google service-account credentials are unavailable: configure "
            f"Kaggle Secret {secret_name!r} or attach private Dataset "
            f"{dataset_slug!r} containing {credential_filename!r}"
        ) from error
    return dataset_value


def safe_error_message(error: BaseException) -> str:
    """Return a compact error string with common credential material redacted."""
    message = str(error)
    message = re.sub(
        r"-----BEGIN PRIVATE KEY-----.*?-----END PRIVATE KEY-----",
        "[private key redacted]",
        message,
        flags=re.DOTALL,
    )
    message = re.sub(r"Bearer\s+[A-Za-z0-9._~+/-]+", "Bearer [redacted]", message)
    return message[:1000]


def _a1_sheet(title: str) -> str:
    return "'" + title.replace("'", "''") + "'"


def _response_error(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping) and error.get("message"):
            return str(error["message"])
        if payload.get("message"):
            return str(payload["message"])
    text = getattr(response, "text", "")
    return str(text).strip()[:500] or "empty response"


@dataclass
class SheetsRestClient:
    spreadsheet_id: str
    access_token: str
    request: RequestCallable
    sleep: SleepCallable = time.sleep
    timeout: float = 30.0
    max_attempts: int = 4

    def _call(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
        retry: bool,
    ) -> dict[str, Any]:
        url = "https://sheets.googleapis.com/v4/" + path
        attempts = self.max_attempts if retry else 1
        last_error: SheetsApiError | None = None
        for attempt in range(attempts):
            try:
                response = self.request(
                    method,
                    url,
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json; charset=utf-8",
                    },
                    params=dict(params or {}),
                    json=dict(body) if body is not None else None,
                    timeout=self.timeout,
                )
            except Exception as error:
                last_error = SheetsApiError(
                    f"Google Sheets network request failed: {type(error).__name__}",
                    transient=True,
                )
            else:
                status = int(getattr(response, "status_code", 0))
                if 200 <= status < 300:
                    if status == 204 or not getattr(response, "content", b""):
                        return {}
                    payload = response.json()
                    return payload if isinstance(payload, dict) else {}
                last_error = SheetsApiError(
                    f"Google Sheets API returned HTTP {status}: {_response_error(response)}",
                    transient=status in TRANSIENT_STATUS_CODES,
                )
            assert last_error is not None
            if not last_error.transient or attempt + 1 >= attempts:
                raise last_error
            self.sleep(min(8.0, 0.5 * 2**attempt))
        raise last_error or SheetsApiError("Unknown Google Sheets error", transient=False)

    @property
    def _spreadsheet_path(self) -> str:
        return f"spreadsheets/{quote(self.spreadsheet_id, safe='')}"

    def metadata(self) -> dict[str, Any]:
        return self._call(
            "GET",
            self._spreadsheet_path,
            params={
                "fields": "properties.title,sheets.properties(sheetId,title,gridProperties)"
            },
            retry=True,
        )

    def batch_update_spreadsheet(self, requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return self._call(
            "POST",
            self._spreadsheet_path + ":batchUpdate",
            body={"requests": list(requests)},
            retry=False,
        )

    def get_values(self, range_name: str) -> list[list[Any]]:
        payload = self._call(
            "GET",
            self._spreadsheet_path + "/values/" + quote(range_name, safe=""),
            params={"majorDimension": "ROWS"},
            retry=True,
        )
        values = payload.get("values", [])
        return values if isinstance(values, list) else []

    def update_values(self, range_name: str, rows: Sequence[Sequence[Any]]) -> dict[str, Any]:
        return self._call(
            "PUT",
            self._spreadsheet_path + "/values/" + quote(range_name, safe=""),
            params={"valueInputOption": "RAW"},
            body={"majorDimension": "ROWS", "values": [list(row) for row in rows]},
            retry=True,
        )

    def batch_update_values(self, updates: Sequence[tuple[str, Sequence[Any]]]) -> dict[str, Any]:
        return self._call(
            "POST",
            self._spreadsheet_path + "/values:batchUpdate",
            body={
                "valueInputOption": "RAW",
                "data": [
                    {
                        "range": range_name,
                        "majorDimension": "ROWS",
                        "values": [list(row)],
                    }
                    for range_name, row in updates
                ],
            },
            retry=True,
        )

    def append_values_once(self, range_name: str, rows: Sequence[Sequence[Any]]) -> dict[str, Any]:
        return self._call(
            "POST",
            self._spreadsheet_path + "/values/" + quote(range_name, safe="") + ":append",
            params={
                "valueInputOption": "RAW",
                "insertDataOption": "INSERT_ROWS",
            },
            body={"majorDimension": "ROWS", "values": [list(row) for row in rows]},
            retry=False,
        )


def _sheet_properties(metadata: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for item in metadata.get("sheets", []):
        if not isinstance(item, Mapping):
            continue
        properties = item.get("properties")
        if isinstance(properties, Mapping) and isinstance(properties.get("title"), str):
            result[str(properties["title"])] = properties
    return result


def _format_requests(
    sheet_id: int,
    column_count: int,
    *,
    json_columns: bool,
) -> list[dict[str, Any]]:
    header_format = {
        "backgroundColor": {"red": 0.10, "green": 0.25, "blue": 0.45},
        "textFormat": {
            "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
            "bold": True,
        },
        "horizontalAlignment": "CENTER",
        "verticalAlignment": "MIDDLE",
        "wrapStrategy": "WRAP",
    }
    requests: list[dict[str, Any]] = [
        {
            "updateSheetProperties": {
                "properties": {
                    "sheetId": sheet_id,
                    "gridProperties": {"frozenRowCount": 1, "hideGridlines": True},
                },
                "fields": "gridProperties.frozenRowCount,gridProperties.hideGridlines",
            }
        },
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": column_count,
                },
                "cell": {"userEnteredFormat": header_format},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }
        },
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "ROWS",
                    "startIndex": 0,
                    "endIndex": 1,
                },
                "properties": {"pixelSize": 36},
                "fields": "pixelSize",
            }
        },
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": 0,
                    "endIndex": column_count,
                },
                "properties": {"pixelSize": 135},
                "fields": "pixelSize",
            }
        },
        {
            "setBasicFilter": {
                "filter": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "startColumnIndex": 0,
                        "endColumnIndex": column_count,
                    }
                }
            }
        },
    ]
    if json_columns:
        requests.append(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": column_count - 2,
                        "endIndex": column_count,
                    },
                    "properties": {"pixelSize": 420},
                    "fields": "pixelSize",
                }
            }
        )
    else:
        requests.append(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": 4,
                        "endIndex": 5,
                    },
                    "properties": {"pixelSize": 240},
                    "fields": "pixelSize",
                }
            }
        )
    return requests


def ensure_tables(client: SheetsRestClient) -> dict[str, int]:
    tables = {
        EXPERIMENTS_SHEET: EXPERIMENT_HEADERS,
        CATEGORY_METRICS_SHEET: CATEGORY_HEADERS,
    }
    metadata = client.metadata()
    properties = _sheet_properties(metadata)
    missing = [title for title in tables if title not in properties]
    if missing:
        client.batch_update_spreadsheet(
            [
                {
                    "addSheet": {
                        "properties": {
                            "title": title,
                            "gridProperties": {
                                "rowCount": 1000,
                                "columnCount": max(50, len(tables[title])),
                            },
                        }
                    }
                }
                for title in missing
            ]
        )
        properties = _sheet_properties(client.metadata())

    sheet_ids: dict[str, int] = {}
    formatting: list[dict[str, Any]] = []
    for title, headers in tables.items():
        sheet = properties.get(title)
        if sheet is None:
            raise SheetsLoggerError(f"Google Sheets did not create worksheet {title!r}")
        sheet_id = int(sheet["sheetId"])
        sheet_ids[title] = sheet_id
        grid = _mapping(sheet.get("gridProperties"))
        current_columns = int(grid.get("columnCount", 0) or 0)
        if current_columns < len(headers):
            formatting.append(
                {
                    "updateSheetProperties": {
                        "properties": {
                            "sheetId": sheet_id,
                            "gridProperties": {"columnCount": len(headers)},
                        },
                        "fields": "gridProperties.columnCount",
                    }
                }
            )
        last_column = column_letter(len(headers))
        header_rows = client.get_values(f"{_a1_sheet(title)}!A1:{last_column}1")
        existing = list(header_rows[0]) if header_rows else []
        while existing and existing[-1] == "":
            existing.pop()
        if existing and list(headers[: len(existing)]) != existing:
            raise SheetsLoggerError(
                f"Worksheet {title!r} has an incompatible header; refusing to shift data"
            )
        if existing != list(headers):
            client.update_values(
                f"{_a1_sheet(title)}!A1:{last_column}1",
                [headers],
            )
        formatting.extend(
            _format_requests(
                sheet_id,
                len(headers),
                json_columns=title == EXPERIMENTS_SHEET,
            )
        )
    if formatting:
        client.batch_update_spreadsheet(formatting)
    return sheet_ids


def _padded(row: Sequence[Any], width: int) -> list[Any]:
    return list(row[:width]) + [""] * max(0, width - len(row))


def _append_with_verification(
    client: SheetsRestClient,
    range_name: str,
    rows: Sequence[Sequence[Any]],
    committed: Callable[[], bool],
) -> None:
    last_error: SheetsApiError | None = None
    for attempt in range(client.max_attempts):
        try:
            client.append_values_once(range_name, rows)
            return
        except SheetsApiError as error:
            last_error = error
            if not error.transient:
                raise
            if committed():
                return
            if attempt + 1 < client.max_attempts:
                client.sleep(min(8.0, 0.5 * 2**attempt))
    raise last_error or SheetsLoggerError("Google Sheets append failed")


def _upsert_experiment(client: SheetsRestClient, row: Sequence[Any]) -> str:
    run_id = str(row[0])
    last_column = column_letter(len(EXPERIMENT_HEADERS))
    rows = client.get_values(f"{_a1_sheet(EXPERIMENTS_SHEET)}!A2:{last_column}")
    matches = [index + 2 for index, current in enumerate(rows) if current and str(current[0]) == run_id]
    if len(matches) > 1:
        raise SheetsLoggerError(f"Worksheet {EXPERIMENTS_SHEET!r} contains duplicate run_id {run_id!r}")
    if matches:
        row_number = matches[0]
        client.update_values(
            f"{_a1_sheet(EXPERIMENTS_SHEET)}!A{row_number}:{last_column}{row_number}",
            [row],
        )
        return "updated"

    def committed() -> bool:
        current = client.get_values(f"{_a1_sheet(EXPERIMENTS_SHEET)}!A2:A")
        return any(values and str(values[0]) == run_id for values in current)

    _append_with_verification(
        client,
        f"{_a1_sheet(EXPERIMENTS_SHEET)}!A:{last_column}",
        [row],
        committed,
    )
    return "appended"


def _upsert_categories(
    client: SheetsRestClient,
    sheet_id: int,
    category_rows: Sequence[Sequence[Any]],
    run_id: str,
) -> str:
    last_column = column_letter(len(CATEGORY_HEADERS))
    existing = client.get_values(
        f"{_a1_sheet(CATEGORY_METRICS_SHEET)}!A2:{last_column}"
    )
    positions: dict[str, int] = {}
    run_rows: list[int] = []
    for index, raw_row in enumerate(existing, start=2):
        row = _padded(raw_row, len(CATEGORY_HEADERS))
        if str(row[0]) != run_id:
            continue
        category = str(row[4])
        if category in positions:
            raise SheetsLoggerError(
                f"Worksheet {CATEGORY_METRICS_SHEET!r} contains duplicate key "
                f"({run_id!r}, {category!r})"
            )
        positions[category] = index
        run_rows.append(index)

    incoming = {str(row[4]): list(row) for row in category_rows}
    if set(positions) == set(incoming):
        if incoming:
            client.batch_update_values(
                [
                    (
                        f"{_a1_sheet(CATEGORY_METRICS_SHEET)}!A{positions[category]}:"
                        f"{last_column}{positions[category]}",
                        incoming[category],
                    )
                    for category in sorted(incoming)
                ]
            )
        return "updated" if positions else "empty"

    if run_rows:
        client.batch_update_spreadsheet(
            [
                {
                    "deleteDimension": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "ROWS",
                            "startIndex": row_number - 1,
                            "endIndex": row_number,
                        }
                    }
                }
                for row_number in sorted(run_rows, reverse=True)
            ]
        )
    if not category_rows:
        return "cleared" if run_rows else "empty"

    expected_keys = {(run_id, str(row[4])) for row in category_rows}

    def committed() -> bool:
        current = client.get_values(
            f"{_a1_sheet(CATEGORY_METRICS_SHEET)}!A2:{last_column}"
        )
        observed = {
            (str(row[0]), str(row[4]))
            for row in (_padded(item, len(CATEGORY_HEADERS)) for item in current)
            if str(row[0]) == run_id
        }
        return observed == expected_keys

    _append_with_verification(
        client,
        f"{_a1_sheet(CATEGORY_METRICS_SHEET)}!A:{last_column}",
        category_rows,
        committed,
    )
    return "replaced" if run_rows else "appended"


def sync_experiment(
    *,
    spreadsheet_id: str,
    service_account_json: str,
    completion: Mapping[str, Any],
    request: RequestCallable | None = None,
    sleep: SleepCallable = time.sleep,
) -> dict[str, Any]:
    spreadsheet_id = spreadsheet_id.strip()
    if not spreadsheet_id:
        raise SheetsLoggerError("Google spreadsheet_id must not be empty")
    token = service_account_token(service_account_json)
    if request is None:
        try:
            import requests
        except ImportError as error:
            raise SheetsLoggerError(
                "requests is required for Google Sheets synchronization"
            ) from error
        request = requests.request
    client = SheetsRestClient(
        spreadsheet_id=spreadsheet_id,
        access_token=token,
        request=request,
        sleep=sleep,
    )
    sheet_ids = ensure_tables(client)
    synced_at = utc_now()
    experiment_row = build_experiment_row(completion, synced_at_utc=synced_at)
    category_rows = build_category_rows(completion)
    experiment_action = _upsert_experiment(client, experiment_row)
    category_action = _upsert_categories(
        client,
        sheet_ids[CATEGORY_METRICS_SHEET],
        category_rows,
        str(completion["run_id"]),
    )
    return {
        "run_id": str(completion["run_id"]),
        "synced_at_utc": synced_at,
        "spreadsheet_id": spreadsheet_id,
        "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
        "experiment_action": experiment_action,
        "category_metrics_action": category_action,
        "category_metrics_count": len(category_rows),
    }


def sync_from_kaggle_secrets(
    *,
    spreadsheet_id: str,
    completion: Mapping[str, Any],
    secret_name: str = DEFAULT_SECRET_NAME,
) -> dict[str, Any]:
    return sync_experiment(
        spreadsheet_id=spreadsheet_id,
        service_account_json=kaggle_secret(secret_name),
        completion=completion,
    )


def sync_from_kaggle_credentials(
    *,
    spreadsheet_id: str,
    completion: Mapping[str, Any],
    secret_name: str = DEFAULT_SECRET_NAME,
    input_root: Path = Path("/kaggle/input"),
    dataset_slug: str = DEFAULT_CREDENTIAL_DATASET_SLUG,
    credential_filename: str = DEFAULT_CREDENTIAL_FILENAME,
) -> dict[str, Any]:
    """Sync using a Kaggle Secret or the attached private credential Dataset."""
    return sync_experiment(
        spreadsheet_id=spreadsheet_id,
        service_account_json=kaggle_service_account_json(
            secret_name=secret_name,
            input_root=input_root,
            dataset_slug=dataset_slug,
            credential_filename=credential_filename,
        ),
        completion=completion,
    )
