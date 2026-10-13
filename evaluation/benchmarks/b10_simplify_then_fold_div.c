// Like b8, with a division whose operands only become constants after peephole.
int k(int a, int b) {
    return (b * 0 + 6) / (a * 0 + 2) - (a + 0);
}
