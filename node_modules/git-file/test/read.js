var test = require('tap').test;
var git = require('../');
var through = require('through');

test('read an old example', function (t) {
    t.plan(1);
    var commit = '1b37f60da787df42edb863e5ea726e4dd7ce0457';
    var file = 'example/x.js';
    
    var data = '';
    var tr = through(
        function (buf) { data += buf },
        function () {
            t.equal(data, [
                "var gitFile = require('../');",
                "var joinStream = require('join-stream');",
                "",
                "gitFile.list(process.argv[2], process.argv[3])",
                "    .pipe(joinStream('\\n'))",
                "    .pipe(process.stdout)",
                ";",
                ""
            ].join('\n'));
        }
    );
    
    git.read(commit, file).pipe(tr);
});
