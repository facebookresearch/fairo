var OrderedEmitter = require('../');
var em = new OrderedEmitter;

em.on('beep', function (obj) {
    console.dir(obj);
});

var objects = [
    { order : 1 },
    { order : 2 },
    { order : 4 },
    { order : 0 },
    { order : 3 },
];

var iv = setInterval(function () {
    var obj = objects.shift();
    if (!obj) clearInterval(iv)
    else em.emit('beep', obj)
}, 500);
