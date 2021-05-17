include ./Makefile.rule

# SOURCE_FILES := $(foreach d, $(DIRS), $(wildcard $(d)*.c) )

#LIBS += libblasfeo.a
LIBS += $(CURRENT_DIR)/lib/libblasfeo.a

# add different link library for different EXTERNAL_BLAS implementation
#include ../Makefile.external_blas
LIBS += $(LIBS_EXTERNAL_BLAS)

ifeq ($(COMPLEMENT_WITH_NETLIB_BLAS), 1)
LIBS += -lgfortran
endif

LIBS += -lm

## getting started
OBJS = getting_started.o

### all individual tests ###

getting_started \
: %: %.o
	$(CC) -o $@.out $^ $(LIBS)
	./$@.out

CFLAGS += -I./include/

build: common $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LIBS) -o getting_started.out

common:
	cp ./lib/libblasfeo.a .

run:
	./example.out

adb_push:
	adb push example.out /data/local/tmp/getting_started.out

adb_run:
	adb shell /data/local/tmp/getting_started.out

clean:
	rm -rf ./*.o
	rm -rf ./*.out

deep_clean: clean