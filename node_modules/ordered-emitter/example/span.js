var OrderedEmitter = require('../');
var em = new OrderedEmitter({ span : true });

em.on('beep', function (obj) {
    console.log('beep ' + obj.x);
});

em.on('boop', function (obj) {
    console.log('boop ' + obj.x);
});

em.emit('beep', { order : 1, x : 'b' });
em.emit('beep', { order : 3, x : 'd' });
em.emit('boop', { order : 0, x : 'a' });
em.emit('beep', { order : 2, x : 'c' });
em.emit('boop', { order : 5, x : 'f' });
em.emit('boop', { order : 4, x : 'e' });
