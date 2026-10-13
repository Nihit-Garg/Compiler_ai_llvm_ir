// Like b8, with two operands that only become constants after peephole.
int g(int a, int b) {
    return (a * 0 + 4) * (b * 0 + 5);
}
