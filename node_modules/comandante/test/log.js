var test = require('tap').test;
var run = require('../');
var through = require('through');

test('stdout capture', function (t) {
    t.plan(1);
    
    var data = '';
    var ws = through(
        function (buf) { data += buf },
        function () {
            var lines = data.split('\n');
            t.ok(
                lines[lines.length - 6],
                'commit 95e4802118459f2eec942cba789bd451702e3aa4'
            );
        }
    );
    run('git', [ 'log' ]).pipe(ws);
});

test('stderr capture', function (t) {
    t.plan(2);
    
    var ps = run('git', [ 'log' ], { cwd : '/tmp' });
    ps.on('error', function (err) {
        t.ok(/non-zero exit code/.test(err));
        t.ok(/Not a git repository/.test(err));
    });
});
