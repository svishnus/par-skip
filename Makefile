# Metric skip lists on ParlayLib (header-only). Targets:
#   make test        build + run all tests
#   make bench       build benchmarks
#   make SEQ=1 ...   compile with PARLAY_SEQUENTIAL (single-threaded, easier debugging)
#   make DEBUG=1 ... -O0 -g with ASan/UBSan
ifeq ($(origin CXX),default)   # make predefines CXX=c++; only override that
  CXX := clang++
endif
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Wpedantic
CXXFLAGS += -Iinclude -isystem external/parlaylib/include
LDFLAGS  ?=

ifdef DEBUG
  CXXFLAGS += -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer
  LDFLAGS  += -fsanitize=address,undefined
else
  CXXFLAGS += -O3 -march=native -DNDEBUG
endif
ifdef SEQ
  CXXFLAGS += -DPARLAY_SEQUENTIAL
endif

BUILD   := build
HEADERS := $(wildcard include/mskip/*.hpp)
TESTS   := $(patsubst tests/%.cpp,$(BUILD)/tests/%,$(wildcard tests/*.cpp))
BENCHES := $(patsubst bench/%.cpp,$(BUILD)/bench/%,$(wildcard bench/*.cpp))

# clangd/IntelliSense: compile_commands.json is regenerated whenever a source is
# added or the flags change. Headers get their own entries (-x c++) so clangd
# parses them with the right flags instead of guessing from a nearby .cpp.
COMPDB_SRCS := $(wildcard tests/*.cpp bench/*.cpp)
COMPDB      := compile_commands.json

.PHONY: all test bench clean compile_commands
all: $(COMPDB) $(TESTS) $(BENCHES)

compile_commands: $(COMPDB)

$(COMPDB): Makefile $(COMPDB_SRCS) $(HEADERS)
	@{ \
	  echo '['; sep=''; \
	  for f in $(COMPDB_SRCS); do \
	    printf '%b  {"directory": "%s", "file": "%s", "command": "%s"}' "$$sep" "$(CURDIR)" "$(CURDIR)/$$f" "$(CXX) $(CXXFLAGS) -c $$f"; sep=',\n'; \
	  done; \
	  for f in $(HEADERS); do \
	    printf '%b  {"directory": "%s", "file": "%s", "command": "%s"}' "$$sep" "$(CURDIR)" "$(CURDIR)/$$f" "$(CXX) $(CXXFLAGS) -x c++ -c $$f"; sep=',\n'; \
	  done; \
	  echo; echo ']'; \
	} > $@
	@echo "wrote $@ ($(words $(COMPDB_SRCS) $(HEADERS)) entries)"

$(BUILD)/tests/%: tests/%.cpp $(HEADERS) | $(BUILD)/tests
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD)/bench/%: bench/%.cpp $(HEADERS) | $(BUILD)/bench
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD)/tests $(BUILD)/bench:
	mkdir -p $@

test: $(COMPDB) $(TESTS)
	@status=0; for t in $(TESTS); do echo "== $$t"; $$t || status=1; done; exit $$status

bench: $(COMPDB) $(BENCHES)

clean:
	rm -rf $(BUILD) $(COMPDB)
