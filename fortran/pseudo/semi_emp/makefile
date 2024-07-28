# COMPHY directory path:
COMPHY = ~/               # = /home/cesare/lectures/Computational_Quantum_Mechanics_VO/Thijssen
FC 	= 	gfortran
FOPTS	=	-O -c
LDOPTS	=	-O
LIBS	=	$(COMPHY)/lib/geneig.a  -llapack -lblas

pseudo: pseudo.o 
	$(FC) $(LDOPTS) -o pseudo pseudo.o $(LIBS)

pseudo.o: pseudo.f pseudo.glb
	$(FC) $(FOPTS) pseudo.f
clean:
	rm -f pseudo *.o
