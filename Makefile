dist: init-submodule
	cd frontend && pnpm i && pnpm build && cd .. && cp -r frontend/dist ./
init-submodule:
	git submodule update --init --recursive

clean:
	rm -rf dist frontend/dist

all: init-submodule dist
