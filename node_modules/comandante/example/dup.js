var run = require('commandante');

var node = run('node');
process.stdin.pipe(node).pipe(process.stdout);
