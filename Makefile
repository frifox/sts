# Live Translate.app: `make` builds it fresh and installs it in /Applications.
#
#   make           build + install (the same as `make deploy`)
#   make build     build mac/build/Live Translate.app only
#   make install   install the last build in /Applications (the app must be quit)
#   make test      run the tests (in .venv-mlx)

APP        := Live Translate.app
# The app's earlier names (STS, then Translate Live): removed on install.
OLD_APPS   := "/Applications/STS.app" "/Applications/Translate Live.app"
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
	@if pgrep -f "$(APP)/Contents/|STS.app/Contents/|Translate Live.app/Contents/" >/dev/null; then \
		echo "Live Translate is running - quit it first (Live Translate > Quit), then run make again."; \
		exit 1; \
	fi

install: check-not-running
	@test -d "$(BUILT)" || { echo "No build yet: run make build"; exit 1; }
	rm -rf "$(INSTALLED)" $(OLD_APPS)
	ditto "$(BUILT)" "$(INSTALLED)"
	$(LSREGISTER) -f "$(INSTALLED)"
	codesign --verify --deep "$(INSTALLED)"
	@echo "Installed $(INSTALLED) ($$(du -sh "$(INSTALLED)" | cut -f1))"

test:
	.venv-mlx/bin/python -m pytest -q tests
