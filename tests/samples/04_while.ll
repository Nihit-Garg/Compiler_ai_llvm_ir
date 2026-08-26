define i32 @count_to(i32 %n) {
entry:
  %n.addr = alloca i32
  %i = alloca i32
  store i32 %n, i32* %n.addr
  store i32 0, i32* %i
  br label %while.cond

while.cond:
  %0 = load i32, i32* %i
  %1 = load i32, i32* %n.addr
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %while.body, label %while.end

while.body:
  %2 = load i32, i32* %i
  %add = add i32 %2, 1
  store i32 %add, i32* %i
  br label %while.cond

while.end:
  %3 = load i32, i32* %i
  ret i32 %3
}
