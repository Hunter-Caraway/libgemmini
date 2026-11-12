ifndef RISCV
$(error RISCV is unset)
else
$(info Running with RISCV=$(RISCV))
endif

PREFIX ?= $RISCV/

# changed: added. N:M pattern of the model, must match the chip (e.g. make NM_N=4 NM_M=8)
NM_N ?= 2
NM_M ?= 4

default: libgemmini.so

libgemmini.so: gemmini.cc gemmini.h gemmini_params.h
	g++ -L $(RISCV)/lib -Wl,-rpath,$(RISCV)/lib -shared -o $@ -std=c++17 -I $(RISCV)/include -fPIC -O3 -DNM_N=$(NM_N) -DNM_M=$(NM_M) $<

.PHONY: install
install: libgemmini.so
	cp libgemmini.so $(RISCV)/lib

clean:
	rm -rf *.o *.so
