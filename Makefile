CC = gcc
C_FLAGS = -lm -lopenblas -pthread

all: Project_0

Project_0: Project_0.o quick_select.o
	$(CC) Project_0.o quick_select.o -o Project_0 $(C_FLAGS) 

Project_0.o: Project_0.c
	$(CC) -c Project_0.c

quick_select.o: ./Quick_Select/quick_select.c ./Quick_Select/quick_select.h
	$(CC) -c ./Quick_Select/quick_select.c

clean:
	rm *.o Project_0
