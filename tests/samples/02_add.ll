define i32 @add(i32 %a, i32 %b) {
entry:
  %a.addr = alloca i32
  %b.addr = alloca i32
  %c = alloca i32
  store i32 %a, i32* %a.addr
  store i32 %b, i32* %b.addr
  %0 = load i32, i32* %a.addr
  %1 = load i32, i32* %b.addr
  %add = add i32 %0, %1
  store i32 %add, i32* %c
  %2 = load i32, i32* %c
  ret i32 %2
}
