// Pure constant arithmetic: constant folding then DCE removes everything.
int main() {
    return (2 + 3) * 4 - 6 / 2;
}
