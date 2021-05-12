#!/bin/sh

../examples/test14 -v "3.2 -47.11 0" > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test86.out; then
	exit 0
else
	exit 1
fi
