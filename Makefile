# AVFORGE -- cross-dataset audio-visual deepfake detection
#
# `make help` lists everything. `make smoke` is the one to run first: it
# exercises the entire chain on synthetic fixtures in a few minutes, with no
# dataset access at all.

# Use the project venv when it exists; fall back to the Python on PATH, which
# is what CI uses (it installs into the runner's interpreter, not a .venv).
PY       := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
PIP      := $(if $(wildcard .venv/bin/pip),.venv/bin/pip,pip)
BIN      := $(if $(wildcard .venv/bin/python),.venv/bin/,)
# Modules held to strict typing (see [tool.mypy] in pyproject.toml). Checking
# the whole package strictly would fail on modules never annotated to that
# standard; the gate is on the code every workstream depends on.
TYPED    := ddetect/contracts.py ddetect/metrics.py ddetect/calibrate.py \
            ddetect/ood.py ddetect/losses.py ddetect/report.py ddetect/stability.py ddetect/utils api/schemas.py \
            api/security.py agents/core
RUN      ?= runs/smoke_test/seed0
MANIFEST ?= data/manifests/fixture.parquet
CACHE    ?= processed
PORT     ?= 8000

.DEFAULT_GOAL := help
.PHONY: help venv install install-all fixtures manifest preprocess smoke train eval \
        matrix serve web web-dev test test-fast lint fmt typecheck audit scope \
        agents agent-audit eval-final export drift metrics worker e2e \
        preflight datasets get-dfdc bootstrap \
        clean clean-runs docker-build docker-up docker-down

# ---------------------------------------------------------------- setup
help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv:  ## create the virtualenv
	python3 -m venv .venv && $(PIP) install -q --upgrade pip setuptools wheel

install: venv  ## install the package with dev + face extras
	$(PIP) install -e ".[dev,face]"

install-all: venv  ## install everything, including serving, tracking and export
	$(PIP) install -e ".[dev,face,audio,serve,track,export]"

# ---------------------------------------------------------------- data
preflight:  ## is this machine ready for a real run?
	$(PY) scripts/preflight.py --cache-root $(CACHE) $(ARGS)

datasets:  ## list the datasets and how to obtain each
	$(PY) scripts/download_datasets.py --list

get-dfdc:  ## download the DFDC sample from Kaggle (the unblocked path)
	$(PY) scripts/download_datasets.py dfdc --out $(or $(DATA),~/data) --sample

bootstrap: preflight datasets  ## check the machine, then show what data is needed
	@echo "\nNext: `make get-dfdc` for the instant path, and submit the three"
	@echo "access forms (FF++, Celeb-DF v2, FakeAVCeleb) -- approval takes days."

fixtures:  ## generate the synthetic test clips
	$(PY) scripts/make_fixtures.py

manifest: fixtures  ## build the fixture manifest
	$(PY) -m ddetect.data.build_manifest --dataset fixture \
	  --root tests/fixtures/videos --out $(MANIFEST) --audit

preprocess:  ## populate the preprocessing cache for $(MANIFEST)
	$(PY) -m ddetect.data.run_preprocess --manifest $(MANIFEST) \
	  --cache-root $(CACHE) --n-frames 16 --permissive-detector --workers 4

# ---------------------------------------------------------------- the smoke test
smoke: manifest preprocess  ## full chain on fixtures: data -> train -> eval -> detector
	$(PY) -m ddetect.train --exp smoke_test --model smoke --epochs 2 \
	  --batch-size 2 --n-frames 8 --image-size 224 --audio-seconds 2.0 \
	  --max-sync-windows 4 --num-workers 0 --freeze-epochs 1 --sbi \
	  --manifest $(MANIFEST) --cache-root $(CACHE)
	$(PY) -m ddetect.evaluate --run runs/smoke_test/seed0
	@echo "\n  smoke test passed: the whole chain runs end to end."

# ---------------------------------------------------------------- train / eval
train:  ## train (override EXP, MODEL, EPOCHS, MANIFEST)
	$(PY) -m ddetect.train --exp $(or $(EXP),baseline_a) --model $(or $(MODEL),baseline) \
	  --epochs $(or $(EPOCHS),12) --manifest $(MANIFEST) --cache-root $(CACHE)

eval:  ## evaluate a run on its val split
	$(PY) -m ddetect.evaluate --run $(RUN) --bootstrap 2000

eval-final:  ## THE sanctioned target-test evaluation -- refuses a dirty tree
	@$(PY) -c "from agents.core.firewall import eval_final_allowed; \
	  ok,msg=eval_final_allowed(); print(('OK: ' if ok else 'REFUSED: ')+msg); \
	  raise SystemExit(0 if ok else 1)"
	$(PY) -m ddetect.evaluate --run $(RUN) --bootstrap 2000 \
	  --group-by forgery_method compression has_audio

matrix:  ## run the experiment grid
	$(PY) -m experiments.run_matrix --dry-run

export:  ## TorchScript + ONNX + INT8 export, parity-checked against eager
	$(PY) -m ddetect.export --run $(RUN)

drift:  ## current input/score drift report from the running API
	@curl -s localhost:$(PORT)/v1/drift | $(PY) -m json.tool

metrics:  ## scrape the running API's Prometheus endpoint
	@curl -s localhost:$(PORT)/metrics | grep -E "^avforge_" | head -40

worker:  ## run the out-of-process analysis worker (needs REDIS_URL)
	DDETECT_RUN_DIR=$(RUN) $(BIN)arq api.worker.WorkerSettings

e2e:  ## Playwright end-to-end against the real stack
	cd web && DDETECT_RUN_DIR=$(RUN) npm run e2e

# ---------------------------------------------------------------- serving
serve:  ## run the API (set RUN to pick a checkpoint)
	DDETECT_RUN_DIR=$(RUN) $(PY) -m uvicorn api.main:app --host 127.0.0.1 --port $(PORT) --reload

web:  ## build the frontend into web/dist (served by the API)
	cd web && npm install && npm run build

web-dev:  ## frontend dev server with API proxy
	cd web && npm run dev

# ---------------------------------------------------------------- quality
test:  ## full test suite
	$(PY) -m pytest -q

test-fast:  ## skip the slow integration tests
	$(PY) -m pytest -q -m "not slow" --ignore=tests/test_api.py --ignore=tests/test_detector_parity.py

lint:  ## ruff
	$(BIN)ruff check ddetect api agents experiments tests scripts
	$(BIN)ruff format --check ddetect api agents experiments tests scripts

fmt:  ## autoformat
	$(BIN)ruff check --fix ddetect api agents experiments tests scripts
	$(BIN)ruff format ddetect api agents experiments tests scripts

typecheck:  ## mypy on the strictly-typed modules
	$(BIN)mypy $(TYPED)

scope:  ## Fail if a generative dependency appears
	$(PY) scripts/scope_audit.py

agents:  ## run the agent eval suites (a1 citations + a7 grounding are blocking)
	$(PY) -m agents.evals.runner

agent-audit:  ## run the research agents over the current state
	-$(PY) -m agents.a2_data --manifest $(MANIFEST)
	-$(PY) -m agents.a1_literature
	-$(PY) -m agents.a3_experiment
	-$(PY) -m agents.a5_paper
	-$(PY) -m agents.a6_defence

audit:  ## F5 leakage sweep over $(MANIFEST)
	$(PY) -m ddetect.data.run_preprocess --manifest $(MANIFEST) \
	  --cache-root $(CACHE) --phash-audit --limit 0 || true
	$(PY) -m pytest -q tests/test_leakage.py

# ---------------------------------------------------------------- containers
docker-build:
	docker compose -f docker/compose.yml build

docker-up:
	docker compose -f docker/compose.yml up -d

docker-down:
	docker compose -f docker/compose.yml down -v

# ---------------------------------------------------------------- cleaning
clean:  ## remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache web/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-runs:  ## DELETE every training run (irreversible)
	@printf "This deletes runs/ and results/. Type yes to confirm: " && read ans && [ "$$ans" = "yes" ]
	rm -rf runs results
