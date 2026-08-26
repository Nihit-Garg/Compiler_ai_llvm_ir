define i32 @max(i32 %x, i32 %y) {
entry:
  %retval = alloca i32
  %x.addr = alloca i32
  %y.addr = alloca i32
  store i32 %x, i32* %x.addr
  store i32 %y, i32* %y.addr
  %0 = load i32, i32* %x.addr
  %1 = load i32, i32* %y.addr
  %cmp = icmp sgt i32 %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %2 = load i32, i32* %x.addr
  store i32 %2, i32* %retval
  br label %return

if.else:
  %3 = load i32, i32* %y.addr
  store i32 %3, i32* %retval
  br label %return

return:
  %4 = load i32, i32* %retval
  ret i32 %4
}
