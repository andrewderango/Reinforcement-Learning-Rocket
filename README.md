# Rocket Landing with Reinforcement Learning

This project implements a genetic programming approach to land a rocket at a specified point in a simulated virtual physics environment by controlling its rocket thrusts. It utilizes reinforcement learning techniques to evolve a population of expression trees that represent control policies for the rocket. 

## Demo

A video demo is available [here]([fakelink.com](https://youtu.be/1u2UUE-3-uU)) that trains the rocket and shows the final simulated environment after 100 generations.

## Features
**Genetic Programming**: The program uses genetic programming techniques to evolve an expression tree that determines the rocket's thrust based on its position and velocity.

**Simulation Environment**: A virtual physics environment simulates the rocket's movement and landing. The simulation considers parameters such as mass, force, time intervals, and maximum positions and velocities.

**Random Expression Generation**: The program generates random expression trees with a specified maximum depth to form the initial population for genetic evolution.

**Fitness Evaluation**: Fitness of each expression tree is evaluated based on its performance in landing the rocket. The fitness is measured in terms of the mean reward and mean steps per episode over multiple episodes.

**Genetic Algorithm**: The genetic algorithm involves selection, crossover, and mutation operations to evolve the population of expression trees across multiple generations.

**Visualization**: Visual simulations of the rocket landing are provided to better understand the performance of evolved expression trees.

## Running the Program

To run the program, compile and execute using the following terminal commands:
```
make
./land_rocket
```

## Dependencies
**C++ Compiler**: The project requires a C++ compiler with support for C++11 standards.<br>
**GNU Make**: The provided Makefile uses GNU Make for compilation.

## Repository Structure
``main.cpp``: Contains the main program code implementing genetic programming for rocket landing.<br>
``rocketCentering.h``: Header file defining the rocket landing simulation environment.<br>
``Makefile``: Makefile for compiling the project.<br>
``README.md``: Documentation for the project.
