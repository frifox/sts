# STS.app: `make` builds it fresh and installs it in /Applications.
#
#   make           build + install (the same as `make deploy`)
#   make build     build mac/build/STS.app only
#   make install   install the last build in /Applications (STS must be quit)
#   make test      run the tests (in .venv-mlx)

APP        := STS.app
BUILT      := mac/build/$(APP)
INSTALLED  := /Applications/$(APP)
LSREGISTER := /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister

.PHONY: deploy build install test check-not-running

deploy: build install

build: check-not-running
	zsh mac/build.sh

# The build rewrites mac/build/STS.app and the install replaces
# /Applications/STS.app: a copy of STS running from either would lose its
# files mid-run (its Python lives inside the app).
check-not-running:
	@if pgrep -f "$(APP)/Contents/" >/dev/null; then \
		echo "STS is running - quit it first (STS > Quit STS), then run make again."; \
		exit 1; \
	fi

install: check-not-running
	@test -d "$(BUILT)" || { echo "No build yet: run make build"; exit 1; }
	rm -rf "$(INSTALLED)"
	ditto "$(BUILT)" "$(INSTALLED)"
	$(LSREGISTER) -f "$(INSTALLED)"
	codesign --verify --deep "$(INSTALLED)"
	@echo "Installed $(INSTALLED) ($$(du -sh "$(INSTALLED)" | cut -f1))"

test:
	.venv-mlx/bin/python -m pytest -q tests
