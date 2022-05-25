var test = require('tape');
var timestamp = require('../');

test('stamps', function (t) {
    t.plan(1);
    t.equal(
        timestamp('Thu Mar 14 19:16:19 2013 -0700'),
        '2013-03-14T19:16:19-07:00'
    );
});

test('non-local timezone', function (t) {
    t.plan(1);
    t.equal(
        timestamp('Thu Mar 14 19:16:19 2013 +0400'),
        '2013-03-14T19:16:19+04:00'
    );
});
