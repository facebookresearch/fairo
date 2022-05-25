var test = require('tap').test;
var OrderedEmitter = require('../');

test('verify order', function (t) {
    var em = new OrderedEmitter({ span : true });
    
    var events = [
        [ 'baz', { order : 2 } ],
        [ 'bar', { order : 4 } ],
        [ 'bar', { order : 1 } ],
        [ 'foo', { order : 0 } ],
        [ 'baz', { order : 5 } ],
        [ 'foo', { order : 3 } ],
        [ 'foo', { order : 6 } ],
    ];
    
    var iv = setInterval(function () {
        var ev = events.shift();
        if (!ev) {
            clearInterval(iv);
            em.emit('end');
        }
        else {
            em.emit(ev[0], ev[1]);
        }
    }, 5);
    
    var order = [];
    
    em.on('foo', function (obj) { order.push([ 'foo', obj ]) });
    em.on('bar', function (obj) { order.push([ 'bar', obj ]) });
    em.on('baz', function (obj) { order.push([ 'baz', obj ]) });
    
    em.on('end', function () {
        t.deepEqual(order, [
            [ 'foo', { order : 0 } ],
            [ 'bar', { order : 1 } ],
            [ 'baz', { order : 2 } ],
            [ 'foo', { order : 3 } ],
            [ 'bar', { order : 4 } ],
            [ 'baz', { order : 5 } ],
            [ 'foo', { order : 6 } ],
        ]);
        t.end();
    });
});
