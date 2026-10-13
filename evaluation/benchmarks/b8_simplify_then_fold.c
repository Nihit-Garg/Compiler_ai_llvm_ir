// Needs peephole BEFORE constant folding: a * 0 + 2 simplifies to 2, and only
// then can 2 * 3 be folded. b3_fold_then_simplify needs the opposite order,
// so no single fixed order is best for both.
int f(int a) {
    return (a * 0 + 2) * 3;
}
