.SUFFIXES: _cad _cad.o

CXX=g++ 
# ca plante en 64 avec 4.2.1
ARCHI=-m64

CFLAGS= $(ARCHI) -g -O0 
CXXFLAGS=-I../include $(CFLAGS)

LDFLAGS=-lm 
#CADNALIB=../lib/libcadnaC.a
# CADNALDFLAGS= ../C/*.o
CADNALDFLAGS= -L../lib -lcadnaC
EXAMPLES= ex1_cad ex2_cad  ex3_cad ex4_cad \
	  ex6_cad  ex7_cad 

all : $(EXAMPLES)

################################
ex1_cad : ex1_cad.o $(CADNALIB)
	$(CXX) -v $(ARCHI)  $< -o $@ $(CADNALDFLAGS)
################################
ex2_cad : ex2_cad.o $(CADNALIB)
	$(CXX) $(ARCHI) $< -o $@ $(CADNALDFLAGS)
################################

ex3_cad : ex3_cad.o $(CADNALIB)
	$(CXX) $(ARCHI) $< -o $@ $(CADNALDFLAGS)
################################
ex4_cad : ex4_cad.o $(CADNALIB)
	$(CXX) $(ARCHI) $< -o $@ $(CADNALDFLAGS)
################################
ex5_cad : ex5_cad.o $(CADNALIB)
	$(CXX) $(ARCHI) $< -o $@ $(CADNALDFLAGS)

ex5_cad_opt : ex5_cad_opt.o
	$(CXX) $(ARCHI) $< -o $@ $(CADNALDFLAGS)
################################
ex6_cad : ex6_cad.o $(CADNALIB)
	$(CXX) $(ARCHI) $< -o $@ $(CADNALDFLAGS)
################################
ex7_cad : ex7_cad.o $(CADNALIB)
	$(CXX) $(ARCHI) $< -o $@ $(CADNALDFLAGS)
################################

clean :
	rm -fr *.o $(EXAMPLES)

