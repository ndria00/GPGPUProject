# definisce la macro CPPC
ifndef CPPC
	CPPC=nvcc
endif

ifdef TEST
	HDR=../data/test_header.txt
	DEM=../data/test_dem.txt
	SRC=../data/test_source.txt
	OUT=./test_output_test
	STEPS=2
else
	HDR=../data/tessina_header.txt
	DEM=../data/tessina_dem.txt
	SRC=../data/tessina_source.txt
	OUT=./tessina_output
	STEPS=4000
endif

# definisce le macro contenenti i nomei degli eseguibili
# e il numero di thread omp per la versione parallela
NT = 2 # numero di threads OpenMP
EXEC = sciddicaTStraightforward
BLOCK_X = 32
BLOCK_Y = 32
# definisce il target di default, utile in
# caso di invocazione di make senza parametri
default:all

# compila le versioni seriale e OpenMP
all:
	$(CPPC) sciddicaT.cu -o $(EXEC) -O3

# esegue la simulazione 
run:
	./$(EXEC) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(BLOCK_X) $(BLOCK_Y) &&  md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

# elimina l'eseguibile, file oggetto e file di output
clean:
	rm -f $(EXEC)  *.o *output*

# elimina file oggetto e di output
wipe:
	rm -f *.o *output*
