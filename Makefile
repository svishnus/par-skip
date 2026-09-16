# Metric skip lists on ParlayLib (header-only). Targets:
#   make test        build + run all tests
#   make test-all    tests with the default scheduler, one worker, SEQ=1 and DEBUG=1
#   make bench       build benchmarks
#   make SEQ=1 ...   compile with PARLAY_SEQUENTIAL (single-threaded, easier debugging)
#   make DEBUG=1 ... -O0 -g with ASan/UBSan
#   make TSAN=1 ...  -O1 -g with ThreadSanitizer (ParlayLib's own reports suppressed)
# CXXFLAGS given on the command line replace the defaults; the flags below are
# always appended (override).
ifeq ($(origin CXX),default)   # make predefines CXX=c++; only override that
  CXX := clang++
endif
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Wpedantic
override CXXFLAGS += -Iinclude -isystem external/parlaylib/include
LDFLAGS  ?=

ifdef DEBUG
  override CXXFLAGS += -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer
  override LDFLAGS  += -fsanitize=address,undefined
else ifdef TSAN
  override CXXFLAGS += -O1 -g -fsanitize=thread -fno-omit-frame-pointer
  override LDFLAGS  += -fsanitize=thread
  export TSAN_OPTIONS ?= suppressions=$(CURDIR)/tests/tsan.supp:abort_on_error=0
else
  override CXXFLAGS += -O3 -march=native -DNDEBUG
endif
ifdef SEQ
  override CXXFLAGS += -DPARLAY_SEQUENTIAL
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

.PHONY: all test test-all bench clean compile_commands
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

# Same tests under every scheduler configuration that must agree bit for bit.
test-all:
	$(MAKE) test
	@status=0; for t in $(TESTS); do echo "== PARLAY_NUM_THREADS=1 $$t"; PARLAY_NUM_THREADS=1 $$t || status=1; done; exit $$status
	$(MAKE) SEQ=1 BUILD=$(BUILD)/seq test
	$(MAKE) DEBUG=1 BUILD=$(BUILD)/debug test

bench: $(COMPDB) $(BENCHES)

clean:
	rm -rf $(BUILD) $(COMPDB)
