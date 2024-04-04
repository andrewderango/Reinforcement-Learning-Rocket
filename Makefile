# Compiler
CC = g++

# Compiler flags
CFLAGS = -std=c++11 #-Wall -Wextra

# Target executable
TARGET = genetic_programming_01

# Source files
SOURCES = $(filter-out test.cpp test2.cpp test3.cpp, $(wildcard *.cpp))

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)
	-rm test1.o
	-rm test2.o
	-rm test3.o