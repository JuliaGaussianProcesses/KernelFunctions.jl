JULIA=$(shell which julia)

.PHONY: help format test

help:
	@echo "The following make targets are available:"
	@echo "	format			auto-format code"
	@echo "	test			run all tests"

format:
	@if [ "$(JULIA)" = "" ]; then echo 'Julia not found; run `make JULIA=/path/to/julia format`'; exit 1; fi
	$(JULIA) -e 'using Pkg; Pkg.activate(; temp=true); Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".")'

test:
	@if [ "$(JULIA)" = "" ]; then echo 'Julia not found; run `make JULIA=/path/to/julia test`'; exit 1; fi
	$(JULIA) --project=. -e 'using Pkg; Pkg.test()'
