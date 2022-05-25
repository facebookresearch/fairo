var test = require('tap').test;
var OrderedEmitter = require('../');

test('clear prototype', function (t) {
    var a = new OrderedEmitter;
    var b = new OrderedEmitter;
    t.notEqual(a._events, b._events);
    t.end();
});
