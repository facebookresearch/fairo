### lua-struct
Implementation of binary packing/unpacking in pure lua

### LuaRocks
```luarocks install lua-struct```

### what is it for?
You can use it to pack and unpack binary data in pure lua. The idea is very similar to PHP unpack and pack functions.

### byte order
You can use < or > at the beginning of the format string to specify the byte order. Default is little endian (<), but you can change it to big endian (>) as well. It is possible to dynamically change the byte order within the format string, so in general you can save types in different byte orders.

### available types
```
"b" a signed char.
"B" an unsigned char.
"h" a signed short (2 bytes).
"H" an unsigned short (2 bytes).
"i" a signed int (4 bytes).
"I" an unsigned int (4 bytes).
"l" a signed long (8 bytes).
"L" an unsigned long (8 bytes).
"f" a float (4 bytes).
"d" a double (8 bytes).
"s" a zero-terminated string.
"cn" a sequence of exactly n chars corresponding to a single Lua string.
```
### how to use it?
```lua
require "struct"

local packed = struct.pack('<LIhBsbfd', 123456789123456789, 123456789, -3200, 255, 'Test message', -1, 1.56789, 1.56789)
local L, I, h, B, s, b, f, d = struct.unpack('<LIhBsbfd', packed)
print(L, I, h, B, s, b, f, d)

1.2345678912346e+017    123456789    -3200    255    Test message    -1    1.5678899288177    1.56789
```
