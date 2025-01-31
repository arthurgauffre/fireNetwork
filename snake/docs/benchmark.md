# Benchmark

## Table of Contents

- [Benchmark](#benchmark)
  - [Table of Contents](#table-of-contents)
  - [Main Documentation](#main-documentation)
  - [Benchmark 1](#benchmark-1)
    - [Changes](#changes)
    - [Stats](#stats)

## Main Documentation

- [Main Documentation](../README.md)

## Benchmark 1

### Changes

Original snake with 11 neurons:

- 3 : danger around the snake (1 ex radius)
- 4 : possible direction (UP/DOWN/LEFT/RIGHT)
- 4 : food location boolean (if the food is at the UP/DOWN/LEFT/RIGHT of the snake)

Rewards:

- **Positive :**
  - +1 for each apple eat

- **Negative :**
  - -10 if the snake die or he doesn't eat an apple during a moment

### Stats

<img src="images/Stats1.png" alt="image5" style="width:400px;"/>
