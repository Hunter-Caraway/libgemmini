ifndef RISCV
$(error RISCV is unset)
else
$(info Running with RISCV=$(RISCV))
endif

PREFIX ?= $RISCV/

# changed: added. N:M pattern of the model, must match the chip (e.g. make NM_N=4 NM_M=8)
NM_N ?= 2
NM_M ?= 4

# changed: sparse-loop-fix, NM_DENSE_BUILD=1 models a dense chip (nm_sparsity = None): functs 27/28 abort (N11, N14). Unset or 0: n:m chip
NM_DENSE_BUILD ?=
ifneq ($(filter-out 0,$(NM_DENSE_BUILD)),)
NM_DENSE_FLAG = -DNM_DENSE_BUILD
endif

default: libgemmini.so

libgemmini.so: gemmini.cc gemmini.h gemmini_params.h
	g++ -L $(RISCV)/lib -Wl,-rpath,$(RISCV)/lib -shared -o $@ -std=c++17 -I $(RISCV)/include -fPIC -O3 -DNM_N=$(NM_N) -DNM_M=$(NM_M) $(NM_DENSE_FLAG) $<

.PHONY: install
install: libgemmini.so
	cp libgemmini.so $(RISCV)/lib

clean:
	rm -rf *.o *.so
