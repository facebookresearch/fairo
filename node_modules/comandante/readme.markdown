# commandante

Spawn commands returning a duplex stream. Emit stderr when the command exits
with a non-zero code.

[![build status](https://secure.travis-ci.org/substack/comandante.png)](http://travis-ci.org/substack/comandante)

## example

``` js
var run = require('commandante');
run('git', [ 'log' ]).pipe(process.stdout);
```

in a git directory we get:

```
$ node example/log.js | head -n3
commit ae5045cce4980a87b7151cfe91bc5889951aae39
Author: James Halliday <mail@substack.net>
Date:   Tue Oct 2 09:08:18 2012 -0700
```

in a non-git directory we get:

```
events.js:66
        throw arguments[1]; // Unhandled 'error' event
                       ^
Error: non-zero exit code 128: fatal: Not a git repository (or any of the parent directories): .git

    at ChildProcess.<anonymous> (/home/substack/projects/comandante/index.js:19:27)
    at ChildProcess.EventEmitter.emit (events.js:91:17)
    at maybeClose (child_process.js:634:16)
    at Socket.ChildProcess.spawn.stdin (child_process.js:805:11)
    at Socket.EventEmitter.emit (events.js:88:17)
    at Socket._destroy.destroyed (net.js:358:10)
    at process.startup.processNextTick.process._tickCallback (node.js:244:9)
```

# methods

``` js
var commandante = require('commandante')
```

## commandante(cmd, args, opts={})

Spawn a new process like `require('child_process')`.spawn()`, except the return
value is a duplex stream combining `stdout` and `stdin`.

If the process exits with a non-zero status, emit an `'error'` event with the
stderr data and the code in an informative message.

If `opts.showCommand` is not `false`, show the actual command in the informative
error message. If you are running a command with passwords in the command
arguments make sure to set `showCommand` to `false`.

# install

With [npm](https://npmjs.org) do:

```
npm install comandante
```

# license

MIT
