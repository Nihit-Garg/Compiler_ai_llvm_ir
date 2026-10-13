// Algebraic identities: peephole must run before DCE to clean up.
int scale(int a) {
    return a * 1 + 0;
}
