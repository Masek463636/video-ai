"""Validated, content-addressed Gemini cache and bounded fifth-style session."""
from __future__ import annotations
import hashlib
import json
import time
import uuid
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


class FifthSession:
    """One shared client for the whole job; cache hits never consume budget.

    Only validated responses enter the cache. Neither credentials, prompts nor
    provider error bodies are persisted in the metrics report.
    """
    def __init__(self, client, cache_dir, report_path, *, max_calls=6):
        self.client = client
        self.cache = Path(cache_dir)
        self.report_path = Path(report_path)
        self.max_calls = max_calls
        self.failed = False
        self.started = time.monotonic()
        self.stats = dict(logical_calls=0, cache_hits=0, invalid_responses=0,
                          network_attempts=0, rate_limits=0, retry_wait_seconds=0.,
                          input_tokens=0, output_tokens=0, cached_input_tokens=0,
                          provider_seconds=0., events=[])
        client.telemetry = self.stats

    def save(self):
        self.stats['elapsed_seconds'] = round(time.monotonic() - self.started, 3)
        atomic_json(self.report_path, self.stats)

    def remember(self, parts, value, *, version, temperature=.05):
        identity = dict(version=version, models=self.client.models, temperature=temperature, parts=parts)
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        atomic_json(self.cache / (digest + '.json'), dict(response=value, model=self.client.last_model, schema=version))

    def ask(self, stage, parts, validate, *, version, temperature=.05):
        identity = dict(version=version, models=self.client.models,
                        temperature=temperature, parts=parts)
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = self.cache / (digest + '.json')
        try:
            saved = json.loads(path.read_text(encoding='utf-8'))
            value = validate(saved['response'])
            self.stats['cache_hits'] += 1
            self.stats['events'].append(dict(stage=stage, status='cached', key=digest))
            self.save()
            return value
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if self.failed or self.stats['logical_calls'] >= self.max_calls:
            self.save()
            raise RuntimeError('Fifth-style request budget exhausted or provider unavailable; saved progress retained')
        self.stats['logical_calls'] += 1
        before = time.monotonic()
        try:
            raw = self.client._generate_json(parts, temperature=temperature)
            try:
                value = validate(raw)
            except (ValueError, TypeError, KeyError) as exc:
                self.stats['invalid_responses'] += 1
                self.stats['events'].append(dict(stage=stage, status='invalid_response', key=digest))
                raise ValueError(str(exc)) from None
            atomic_json(path, dict(response=raw, model=self.client.last_model, schema=version))
            self.stats['events'].append(dict(stage=stage, status='generated', key=digest))
            return value
        except ValueError:
            raise
        except Exception:
            # No cascading retries through every subsequent scene after an outage.
            self.failed = True
            self.stats['events'].append(dict(stage=stage, status='provider_unavailable', key=digest))
            raise RuntimeError('Gemini unavailable; fifth-style progress retained, further requests stopped') from None
        finally:
            self.stats['provider_seconds'] += round(time.monotonic() - before, 3)
            self.save()
