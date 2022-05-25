ordered-emitter
===============

Buffer events that may arrive out of order so that they are emitted in order.

Just emit event objects with an `"order"` key starting at 0.

example
=======

emit.js
-------

``` js
var OrderedEmitter = require('ordered-emitter');
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
```

output:

```
{ order: 0 }
{ order: 1 }
{ order: 2 }
{ order: 3 }
{ order: 4 }
```

span.js
-------

``` js
var OrderedEmitter = require('ordered-emitter');
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
```

output:

```
boop a
beep b
beep c
beep d
boop e
boop f
```

methods
=======

var OrderedEmitter = require('ordered-emitter');

var em = new OrderedEmitter(opts={})
------------------------------------

`OrderedEmitter` acts just like an EventEmitter, except that any event that
emits objects as its first argument with numeric `order` keys will be buffered
so that the events will be emitted in order.

By default, order keys are isolated by event names so the order keys from
different event names won't influence each other. However, you can have order
keys work across multiple event names by setting `opts.span` to `true`.

em.reset(eventName)
-------------------

Reset the counter for an ordered emitter back to 0.

If `eventName` is `undefined`, reset all the counters to 0.

install
=======

With [npm](http://npmjs.org) do:

```
npm install ordered-emitter
```

license
=======

MIT/X11
