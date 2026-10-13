// Identities inside a loop body.
int sum(int n) {
    int s = 0;
    int i = 0;
    while (i < n) {
        s = s + i * 1;
        i = i + 1 + 0;
    }
    return s;
}
