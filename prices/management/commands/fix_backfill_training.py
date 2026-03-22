"""Extend backfill ForecastData to cover the 22:00-22:00 training window.

Historical backfill Forecasts (name = "YYYY-MM-DD 16:15") have ForecastData covering
only the calendar day (00:00-23:30, 48 rows).  The training filter in ``update.py``
selects rows in ``[ag_start, ag_end)`` where::

    ag_start = normalize(created_at) + 22h  →  YYYY-MM-DD 22:00
    ag_end   = normalize(created_at) + 46h  →  YYYY-MM-DD+1 22:00

Only 4 of the 48 calendar-day rows fall in this window (22:00–23:30), leaving 44
slots missing (YYYY-MM-DD+1 00:00–21:30).

Fix: for each such forecast, copy the missing YYYY-MM-DD+1 00:00–21:30 FD rows
from the adjacent "YYYY-MM-DD+1 16:15" backfill entry.  After the fix every
backfill entry contributes the full 48 training points.

Usage::

    python manage.py fix_backfill_training             # fix all
    python manage.py fix_backfill_training --dry-run   # preview only
"""

from __future__ import annotations

import pytz
from django.core.management.base import BaseCommand

from ...models import ForecastData, Forecasts

GB_TZ = pytz.timezone("Europe/London")


def _name_to_dt(name: str):
    """Parse a Forecast name like '2026-03-08 16:15' → timezone-aware datetime."""
    import pandas as pd  # noqa: PLC0415

    return pd.Timestamp(name).tz_localize("Europe/London")


class Command(BaseCommand):
    help = "Extend backfill ForecastData to cover the full 22:00-22:00 training window"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be done without making DB changes",
        )

    def handle(self, *args, **options):
        import pandas as pd  # noqa: PLC0415

        dry_run: bool = options.get("dry_run", False)

        fixed_count = 0
        skipped_already_ok = 0
        skipped_no_next = 0
        total_rows_added = 0

        backfill_qs = Forecasts.objects.filter(name__endswith=" 16:15").order_by("name")
        total = backfill_qs.count()
        self.stdout.write(f"Checking {total} '* 16:15' backfill forecasts ...")

        for i, fc in enumerate(backfill_qs):
            fc_dt = _name_to_dt(fc.name)
            ag_start = fc_dt.normalize() + pd.Timedelta(hours=22)
            ag_end = fc_dt.normalize() + pd.Timedelta(hours=46)

            # How many FD rows currently fall inside the training window?
            training_count = ForecastData.objects.filter(
                forecast=fc,
                date_time__gte=ag_start.to_pydatetime(),
                date_time__lt=ag_end.to_pydatetime(),
            ).count()

            if training_count >= 48:
                skipped_already_ok += 1
                continue

            # Locate the next-day "16:15" forecast as the data source
            next_date_str = (fc_dt.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            next_name = f"{next_date_str} 16:15"

            try:
                next_fc = Forecasts.objects.get(name=next_name)
            except Forecasts.DoesNotExist:
                if dry_run:
                    self.stdout.write(f"  {fc.name}: no next-day '16:15' forecast, skipping")
                skipped_no_next += 1
                continue

            # The 44 slots we need: YYYY-MM-DD+1 00:00 to 21:30
            next_day_00 = (fc_dt.normalize() + pd.Timedelta(days=1)).to_pydatetime()
            next_day_2130 = (ag_end - pd.Timedelta(minutes=30)).to_pydatetime()

            source_rows = list(
                ForecastData.objects.filter(
                    forecast=next_fc,
                    date_time__gte=next_day_00,
                    date_time__lte=next_day_2130,
                )
            )

            # Determine which date_times are already present to avoid duplicates
            existing_dts = set(
                ForecastData.objects.filter(
                    forecast=fc,
                    date_time__gte=next_day_00,
                    date_time__lte=next_day_2130,
                ).values_list("date_time", flat=True)
            )

            new_objects = [
                ForecastData(
                    forecast=fc,
                    date_time=src.date_time,
                    bm_wind=src.bm_wind,
                    solar=src.solar,
                    emb_wind=src.emb_wind,
                    temp_2m=src.temp_2m,
                    wind_10m=src.wind_10m,
                    rad=src.rad,
                    demand=src.demand,
                    day_ahead=src.day_ahead,
                )
                for src in source_rows
                if src.date_time not in existing_dts
            ]

            added = len(new_objects)
            if added == 0:
                skipped_already_ok += 1
                continue

            if not dry_run:
                ForecastData.objects.bulk_create(new_objects)

            fixed_count += 1
            total_rows_added += added

            # In dry-run, show each entry; otherwise show progress every 100
            if dry_run:
                self.stdout.write(
                    f"  [DRY RUN] {fc.name}: would add {added} FD rows"
                    f" (training_count {training_count} -> {training_count + added})"
                )
            elif fixed_count % 100 == 0:
                self.stdout.write(f"  ... fixed {fixed_count}/{total} ({fc.name})")

        action = "Would fix" if dry_run else "Fixed"
        row_action = "would insert" if dry_run else "inserted"
        self.stdout.write(
            f"\n{action} {fixed_count} forecasts ({row_action} {total_rows_added} FD rows)"
            f"  |  already-ok: {skipped_already_ok}"
            f"  |  no-next-entry: {skipped_no_next}"
        )
