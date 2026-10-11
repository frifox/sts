# Translate Live.app: `make` builds it fresh and installs it in /Applications.
#
#   make           build + install (the same as `make deploy`)
#   make build     build mac/build/Translate Live.app only
#   make install   install the last build in /Applications (the app must be quit)
#   make test      run the tests (in .venv-mlx)

APP        := Translate Live.app
# The app's name before the rename (STS): removed on install.
OLD_APP    := /Applications/STS.app
BUILT      := mac/build/$(APP)
INSTALLED  := /Applications/$(APP)
LSREGISTER := /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister

.PHONY: deploy build install test check-not-running

deploy: build install

build: check-not-running
	zsh mac/build.sh

# The build rewrites the app in mac/build and the install replaces the one
# in /Applications: a copy running from either would lose its
# files mid-run (its Python lives inside the app).
check-not-running:
	@if pgrep -f "$(APP)/Contents/|STS.app/Contents/" >/dev/null; then \
		echo "Translate Live is running - quit it first (Translate Live > Quit), then run make again."; \
		exit 1; \
	fi

install: check-not-running
	@test -d "$(BUILT)" || { echo "No build yet: run make build"; exit 1; }
	rm -rf "$(INSTALLED)" "$(OLD_APP)"
	ditto "$(BUILT)" "$(INSTALLED)"
	$(LSREGISTER) -f "$(INSTALLED)"
	codesign --verify --deep "$(INSTALLED)"
	@echo "Installed $(INSTALLED) ($$(du -sh "$(INSTALLED)" | cut -f1))"

test:
	.venv-mlx/bin/python -m pytest -q tests
