# Makefile

py-files := $(shell find . -name '*.py' -not -path "*/run_*/*" -not -path "*/build/*")

install:
	@pip install --upgrade --upgrade-strategy eager -r requirements.txt
.PHONY: install

install-dev:
	@pip install ruff mypy
.PHONY: install-dev

format:
	@ruff format $(py-files)
	@ruff check --fix $(py-files)
.PHONY: format

static-checks:
	@mkdir -p .mypy_cache
	@ruff check $(py-files)
	@mypy --install-types --non-interactive $(py-files)
.PHONY: lint

notebook:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
.PHONY: notebook

build-docker:
	docker build -t zbot-policy-walking .
.PHONY: build-docker

train:
	docker run -d --gpus all \
		-v $(CURDIR):/app \
		$(if $(AWS_ACCESS_KEY_ID),-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID)) \
		$(if $(AWS_SECRET_ACCESS_KEY),-e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)) \
		$(if $(S3_BUCKET),-e S3_BUCKET=$(S3_BUCKET)) \
		--cap-add SYS_ADMIN \
		--device /dev/fuse \
		--security-opt apparmor:unconfined \
		--privileged \
		zbot-policy-walking \
		python -m train $(ARGS)
.PHONY: train
