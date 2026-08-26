define i32 @double_it(i32 %a) {
entry:
  %a.addr = alloca i32
  store i32 %a, i32* %a.addr
  %0 = load i32, i32* %a.addr
  %mul = mul i32 %0, 2
  ret i32 %mul
}

define i32 @main() {
entry:
  %res = alloca i32
  %call = call i32 @double_it(i32 5)
  store i32 %call, i32* %res
  %0 = load i32, i32* %res
  ret i32 %0
}
