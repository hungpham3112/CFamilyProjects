# ca plante en 64 avec 4.2.1

ARCHI=64

ARCHITECTURE  = -m$(ARCHI)

CPPFLAGS= $(ARCHITECTURE) -g -I../include
CPP=g++


OBJ= \
cadna_add_d.o cadna_computedzero_d.o cadna_convert_d.o cadna_digitnumber_d.o cadna_div_d.o \
cadna_eq_d.o cadna_ge_d.o cadna_gt_d.o cadna_intr_d.o cadna_le_d.o cadna_lt_d.o \
cadna_mul_d.o cadna_ne_d.o cadna_str_d.o cadna_sub_d.o cadna_to_d.o \
cadna_add_s.o cadna_computedzero_s.o cadna_convert_s.o cadna_digitnumber_s.o cadna_div_s.o \
cadna_eq_s.o cadna_ge_s.o cadna_gt_s.o cadna_intr_s.o cadna_le_s.o cadna_lt_s.o \
cadna_mul_s.o cadna_ne_s.o  cadna_str_s.o cadna_sub_s.o cadna_to_s.o \
 cadna_random.o \
cadna_math_gnu.o \
cadna_type.o cadna_unstab.o cadna_rounding_$(ARCHI).o



#cadna_math.o \

all : libcadnaC.a 

cadna_rounding_$(ARCHI).o : cadna_rounding_$(ARCHI).s
	gcc $(ARCHITECTURE)  -c  -o $@ $<


libcadnaC.a : $(OBJ) 
	ar cr $@ $+

install : all
	cp cadna.h ../include
	cp libcadnaC.a ../lib	


clean : 
	rm -f *.a *.o