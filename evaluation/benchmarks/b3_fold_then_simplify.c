// Folding creates identities (a * 1, a * 0) that peephole then removes.
int f(int a) {
    return a * (3 - 2) + a * (5 - 5);
}
