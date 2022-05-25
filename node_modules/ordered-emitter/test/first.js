var test = require('tap').test;
var OrderedEmitter = require('../');

test('first object ready', function (t) {
    var em = new OrderedEmitter;
    
    var events = [
        { order : 0 },
        { order : 2 },
        { order : 3 },
        { order : 1 },
    ];
    
    var iv = setInterval(function () {
        var ev = events.shift();
        if (!ev) {
            clearInterval(iv);
            em.emit('end');
        }
        else {
            em.emit('data', ev);
        }
    }, 5);
    
    var order = [];
    em.on('data', function (obj) {
        order.push(obj);
    });
    
    em.on('end', function () {
        t.deepEqual(order, [
            { order : 0 },
            { order : 1 },
            { order : 2 },
            { order : 3 },
        ]);
        t.end();
    });
});
