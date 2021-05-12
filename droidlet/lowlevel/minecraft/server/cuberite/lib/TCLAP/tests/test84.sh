#!/bin/sh

# docbookoutput. The when this test fails due to e.g. formatting
# changes the results needs to be manually reviewed and the test81.out
# updated
../examples/test25 -h x > tmp.out 2>&1

if cmp -s tmp.out $srcdir/test84.out; then
	exit 0
else
	exit 1
fi

